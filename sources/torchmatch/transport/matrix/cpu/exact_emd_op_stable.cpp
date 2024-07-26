// Stable-ABI variant of exact_emd_op.cpp.
// Differences from the ATen version:
//   - torch::stable::* APIs instead of at::*
//   - masked_fill applied via a manual loop (not in stable API surface)
//   - Tensor slicing via torch::stable::select instead of operator[]
//   - STD_TORCH_CHECK instead of TORCH_CHECK

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cstdint>
#include <limits>
#include <optional>

#include "exact/EMD.h"

namespace ts = torch::stable;
using ScT = torch::headeronly::ScalarType;
using Tensor = ts::Tensor;

namespace {

constexpr int kEmdOptimal = 1;
constexpr uint64_t kMaxIter = 100'000;

Tensor exact_emd_kernel_cpu(const Tensor &cost,
                             const std::optional<Tensor> &mask,
                             const std::optional<Tensor> &a,
                             const std::optional<Tensor> &b) {
    STD_TORCH_CHECK(cost.dim() == 3,
                "torch.ops.transport.exact_emd: cost.ndim must be 3, got ", cost.dim());
    STD_TORCH_CHECK(cost.is_cpu(),
                "torch.ops.transport.exact_emd: cost must be CPU");
    STD_TORCH_CHECK(cost.scalar_type() == ScT::Float || cost.scalar_type() == ScT::Double,
                "torch.ops.transport.exact_emd: cost dtype must be float32 or float64");

    const int64_t B = cost.size(0);
    const int64_t N = cost.size(1);
    const int64_t M = cost.size(2);

    // Make cost contiguous; clone before applying mask so we don't alias input.
    Tensor cost_eff;
    if (mask.has_value()) {
        STD_TORCH_CHECK(mask->is_cpu(),
                    "torch.ops.transport.exact_emd: mask must be a CPU tensor");
        STD_TORCH_CHECK(mask->scalar_type() == ScT::Bool,
                    "torch.ops.transport.exact_emd: mask must be a bool tensor");
        STD_TORCH_CHECK(mask->dim() == 3 &&
                        mask->size(0) == B && mask->size(1) == N && mask->size(2) == M,
                    "torch.ops.transport.exact_emd: mask shape must match cost shape (B,N,M)");
        cost_eff = ts::clone(ts::contiguous(cost));
        Tensor mask_cont = ts::contiguous(*mask);
        const bool *mask_ptr = mask_cont.const_data_ptr<bool>();
        const int64_t total = B * N * M;
        if (cost_eff.scalar_type() == ScT::Double) {
            double *cp = cost_eff.mutable_data_ptr<double>();
            for (int64_t i = 0; i < total; ++i)
                if (!mask_ptr[i])
                    cp[i] = std::numeric_limits<double>::infinity();
        } else {
            float *cp = cost_eff.mutable_data_ptr<float>();
            for (int64_t i = 0; i < total; ++i)
                if (!mask_ptr[i])
                    cp[i] = std::numeric_limits<float>::infinity();
        }
    } else {
        cost_eff = ts::contiguous(cost);
    }

    if (a.has_value()) {
        STD_TORCH_CHECK(a->is_cpu(),
                    "torch.ops.transport.exact_emd: a must be a CPU tensor");
        STD_TORCH_CHECK(a->dim() == 2 && a->size(0) == B && a->size(1) == N,
                    "torch.ops.transport.exact_emd: a shape must be (B, N)");
    }
    Tensor a_eff = a.has_value()
        ? ts::contiguous(ts::to(*a, ScT::Double))
        : ts::full({B, N}, 1.0 / static_cast<double>(N), ScT::Double);

    if (b.has_value()) {
        STD_TORCH_CHECK(b->is_cpu(),
                    "torch.ops.transport.exact_emd: b must be a CPU tensor");
        STD_TORCH_CHECK(b->dim() == 2 && b->size(0) == B && b->size(1) == M,
                    "torch.ops.transport.exact_emd: b shape must be (B, M)");
    }
    Tensor b_eff = b.has_value()
        ? ts::contiguous(ts::to(*b, ScT::Double))
        : ts::full({B, M}, 1.0 / static_cast<double>(M), ScT::Double);

    Tensor plan = ts::full({B, N, M}, 0.0, cost.scalar_type());
    Tensor cost_double = ts::contiguous(ts::to(cost_eff, ScT::Double));

    ts::parallel_for(0, B, 1, [&](int64_t begin, int64_t end) {
        for (int64_t bi = begin; bi < end; ++bi) {
            Tensor cost_b     = ts::contiguous(ts::select(cost_double, 0, bi));
            Tensor a_b        = ts::contiguous(ts::select(a_eff, 0, bi));
            Tensor b_b        = ts::contiguous(ts::select(b_eff, 0, bi));
            Tensor plan_d     = ts::full({N, M}, 0.0, ScT::Double);
            Tensor alpha      = ts::full({N}, 0.0, ScT::Double);
            Tensor beta       = ts::full({M}, 0.0, ScT::Double);

            double total_cost = 0.0;
            int status = EMD_wrap(
                static_cast<int>(N), static_cast<int>(M),
                a_b.mutable_data_ptr<double>(),
                b_b.mutable_data_ptr<double>(),
                cost_b.mutable_data_ptr<double>(),
                plan_d.mutable_data_ptr<double>(),
                alpha.mutable_data_ptr<double>(),
                beta.mutable_data_ptr<double>(),
                &total_cost, kMaxIter, nullptr, nullptr);
            STD_TORCH_CHECK(
                status == kEmdOptimal,
                "torch.ops.transport.exact_emd: network-simplex solver returned "
                "non-OPTIMAL status ", status, " on batch index ", bi);

            Tensor plan_bi = ts::select(plan, 0, bi);
            ts::copy_(plan_bi, ts::to(plan_d, cost.scalar_type()));
        }
    });

    return plan;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(transport, CPU, m) {
    m.impl("exact_emd", TORCH_BOX(&exact_emd_kernel_cpu));
}
