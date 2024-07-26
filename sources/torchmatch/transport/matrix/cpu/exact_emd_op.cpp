// Torch adapter for the network-simplex EMD CPU implementation.
//
// Wraps EMD_wrapper.cpp with torch::Tensor I/O and at::parallel_for over
// the batch dim. Single-instance perf inherits the serial solver; the
// parallel_for parallelises across batch only.

#include <ATen/Functions.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <cstdint>
#include <limits>
#include <optional>

#include "exact/EMD.h"

namespace {

constexpr int kEmdOptimal = 1; // ProblemType::OPTIMAL
constexpr uint64_t kMaxIter = 100'000;

at::Tensor exact_emd_kernel_cpu(const at::Tensor &cost,
                                const std::optional<at::Tensor> &mask,
                                const std::optional<at::Tensor> &a,
                                const std::optional<at::Tensor> &b) {
    TORCH_CHECK(cost.dim() == 3,
                "torch.ops.transport.exact_emd: cost.ndim must be 3, got ", cost.dim());
    TORCH_CHECK(cost.device().is_cpu(),
                "torch.ops.transport.exact_emd: cost must be CPU, got ", cost.device());
    TORCH_CHECK(cost.dtype() == at::kFloat || cost.dtype() == at::kDouble,
                "torch.ops.transport.exact_emd: cost dtype must be float32 or float64");

    const int64_t B = cost.size(0);
    const int64_t N = cost.size(1);
    const int64_t M = cost.size(2);

    at::Tensor cost_eff = cost.contiguous();
    if (mask.has_value()) {
        cost_eff =
            cost_eff.masked_fill(~(*mask), std::numeric_limits<double>::infinity());
    }

    at::Tensor a_eff = a.has_value() ? a->to(at::kDouble).contiguous()
                                     : at::full({B, N}, 1.0 / static_cast<double>(N),
                                                cost_eff.options().dtype(at::kDouble));
    at::Tensor b_eff = b.has_value() ? b->to(at::kDouble).contiguous()
                                     : at::full({B, M}, 1.0 / static_cast<double>(M),
                                                cost_eff.options().dtype(at::kDouble));

    at::Tensor plan = at::zeros({B, N, M}, cost.options());
    at::Tensor cost_double = cost_eff.to(at::kDouble).contiguous();

    at::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bi = begin; bi < end; ++bi) {
            at::Tensor cost_b = cost_double[bi].contiguous();
            at::Tensor a_b = a_eff[bi].contiguous();
            at::Tensor b_b = b_eff[bi].contiguous();
            at::Tensor plan_double = at::zeros({N, M}, cost_double.options());
            at::Tensor alpha = at::zeros({N}, cost_double.options());
            at::Tensor beta = at::zeros({M}, cost_double.options());

            double total_cost = 0.0;
            int status = EMD_wrap(
                static_cast<int>(N), static_cast<int>(M), a_b.data_ptr<double>(),
                b_b.data_ptr<double>(), cost_b.data_ptr<double>(),
                plan_double.data_ptr<double>(), alpha.data_ptr<double>(),
                beta.data_ptr<double>(), &total_cost, kMaxIter, nullptr, nullptr);
            TORCH_CHECK(
                status == kEmdOptimal,
                "torch.ops.transport.exact_emd: network-simplex solver returned "
                "non-OPTIMAL status ",
                status, " on batch index ", bi);

            plan[bi].copy_(plan_double.to(cost.dtype()));
        }
    });

    return plan;
}

} // namespace

TORCH_LIBRARY_IMPL(transport, CPU, m) { m.impl("exact_emd", &exact_emd_kernel_cpu); }
