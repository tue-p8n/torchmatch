// CUDA bindings for the assignment::* ops.
//
// The primed-zeros Hungarian sub-family ships three solvers, each in
// its own translation unit (they share __managed__ / __constant__
// symbol names and cannot coexist in one TU):
//
//   munkres (munkres.cu): Munkres' classical augmenting-path
//       implementation with primed/starred zeros; single-path
//       augmentation per outer iteration; column-major float. Faster
//       on sparse or tied instances.
//   hybrid  (hybrid.cu):  experimental adaptive variant. Undocumented.
//   lawler  (lawler.cu):  Lawler's tree-augmentation implementation;
//       parallel BFS finds every vertex-disjoint augmenting path per
//       outer iteration; row-major double. Faster on dense instances.
//
// jonker_dense_batch (CUDA backend): shared-memory tiled
// Jonker-Volgenant kernel from jonker_tiled.cuh, one CUDA block per
// problem. Constraints: 3D input (B, K, K) with K <= MAX_TILE; the
// kernel solves B problems in parallel.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <torch/library.h>

#include <cstdint>

#include "cuda_common.cuh"
#include "jonker_tiled.cuh"

namespace match_munkres {
at::Tensor solve(const at::Tensor &cost, cudaStream_t stream);
}
namespace match_hybrid {
at::Tensor solve(const at::Tensor &cost, cudaStream_t stream);
}
namespace match_lawler {
at::Tensor solve(const at::Tensor &cost, cudaStream_t stream);
}

namespace {

using SolverFn = at::Tensor (*)(const at::Tensor &, cudaStream_t);

at::Tensor match_dispatch(const at::Tensor &cost, SolverFn solver) {
    TORCH_CHECK(cost.is_cuda(), "cost must be a CUDA tensor");
    TORCH_CHECK(cost.dim() == 2, "cost must be 2-dimensional, got ", cost.dim());
    TORCH_CHECK(cost.is_contiguous(), "cost must be contiguous");

    const at::cuda::CUDAGuard device_guard(cost.device());

    if (cost.size(0) == 0 || cost.size(1) == 0) {
        return at::full({cost.size(0)}, -1, cost.options().dtype(at::kLong));
    }

    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(cost.device().index()).stream();
    return solver(cost, stream);
}

at::Tensor munkres_cuda(const at::Tensor &cost) {
    return match_dispatch(cost, &match_munkres::solve);
}

at::Tensor hybrid_cuda(const at::Tensor &cost) {
    return match_dispatch(cost, &match_hybrid::solve);
}

at::Tensor lawler_cuda(const at::Tensor &cost) {
    return match_dispatch(cost, &match_lawler::solve);
}

at::Tensor jonker_dense_batch_cuda(const at::Tensor &costs) {
    TORCH_CHECK(costs.is_cuda(), "costs must be a CUDA tensor");
    TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, K, K), got ", costs.dim());
    TORCH_CHECK(costs.size(1) == costs.size(2), "costs must be square (B, K, K)");
    TORCH_CHECK(costs.is_contiguous(), "costs must be contiguous");

    const at::cuda::CUDAGuard device_guard(costs.device());
    cudaStream_t stream =
        at::cuda::getCurrentCUDAStream(costs.device().index()).stream();

    const auto B = costs.size(0);
    const auto K = costs.size(1);

    if (B == 0 || K == 0) {
        return at::full({B, K}, -1, costs.options().dtype(at::kLong));
    }

    auto cost_f = costs.to(at::kFloat);
    // One sentinel across the whole batch, matching the CPU
    // jonker_dense_batch policy in cpu/ops.cpp.
    const float sentinel = ::assign_lap::compute_inf_sentinel<float>(cost_f);
    cost_f = ::assign_lap::rewrite_inf_to_sentinel(cost_f, sentinel).contiguous();

    int tile = match_batch::select_tile(static_cast<int>(K));
    TORCH_CHECK(tile > 0, "Batched JV (CUDA) supports K <= ", match_batch::MAX_TILE,
                ", got K=", K);

    auto result = at::empty({B, K}, costs.options().dtype(at::kLong));

    match_batch::launch(tile, static_cast<int>(B), static_cast<int>(K),
                        cost_f.data_ptr<float>(), result.data_ptr<int64_t>(), stream);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "batched JV kernel launch failed: ", cudaGetErrorString(err));
    return result;
}

} // namespace

TORCH_LIBRARY_IMPL(assignment, CUDA, m) {
    m.impl("munkres", &munkres_cuda);
    m.impl("hybrid", &hybrid_cuda);
    m.impl("lawler", &lawler_cuda);
    m.impl("jonker_dense_batch", &jonker_dense_batch_cuda);
}
