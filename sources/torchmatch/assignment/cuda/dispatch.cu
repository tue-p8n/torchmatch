// CUDA bindings for the assignment::* ops. Stable-ABI: every tensor
// touched here is a torch::stable::Tensor, manipulated exclusively via
// torch::stable::ops (torch/csrc/stable/ops.h) and the AOTI C-shim
// (torch/csrc/inductor/aoti_torch/c/shim.h) for the two things
// torch::stable doesn't cover (current-stream lookup, device index).
// No at::Tensor / ATen headers are included in this file, and the tensor
// pipeline in munkres.cu / lawler.cu / hybrid.cu is likewise
// torch::stable-only, down to the workspace cache those three share
// (workspace.cuh), whose device buffers are stable tensors allocated
// through the AOTI C shim. The extension is therefore free of ATen/c10
// headers throughout and builds with -DTORCH_TARGET_VERSION.
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

// aoti_torch_get_current_cuda_stream() is declared in the AOTI C-shim
// only when USE_CUDA is defined (torch/csrc/inductor/aoti_torch/c/shim.h,
// under `#ifdef USE_CUDA`). That macro is an internal PyTorch build flag
// that nvcc does not define for downstream extension compiles, so this
// TU — which is unconditionally CUDA (compiled by nvcc, includes
// cuda_runtime.h) — defines it itself before pulling in the shim.
// This relies on an internal torch build-macro convention rather than a
// documented public contract, so if a future torch upgrade changes how
// `USE_CUDA` gates that declaration (or removes the guard entirely),
// this define may need to be revisited or dropped.
#ifndef USE_CUDA
#define USE_CUDA 1
#endif
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

#include "cuda_common.cuh"
#include "jonker_tiled.cuh"

using Tensor = torch::stable::Tensor;
namespace ts = torch::stable;
using ScT = torch::headeronly::ScalarType;

namespace match_munkres {
Tensor solve(const Tensor &cost, cudaStream_t stream);
}
namespace match_hybrid {
Tensor solve(const Tensor &cost, cudaStream_t stream);
}
namespace match_lawler {
Tensor solve(const Tensor &cost, cudaStream_t stream);
}

namespace {

// Look up the CUDA stream PyTorch currently has active for `device`, via
// the AOTI C-shim (no c10::cuda / at::cuda dependency).
cudaStream_t current_stream_for(int32_t device) {
    void *stream_ptr = nullptr;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(device, &stream_ptr));
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

using SolverFn = Tensor (*)(const Tensor &, cudaStream_t);

// munkres/hybrid/lawler each do a host-side cudaStreamSynchronize inside
// their outer-iteration loop, and jonker_dense_batch's sentinel
// computation (compute_inf_sentinel in cuda_common.cuh) does its own
// host sync to read back scan results (see the `cudagraph_unsafe`
// discussion in ops.cpp: the stable-ABI op-registration path has no way
// to attach that tag in this torch version, so nothing upstream stops
// any of these ops from being captured into a CUDA graph). A host sync
// inside stream capture is undefined behavior for the CUDA driver — at
// best a capture-time error, at worst a hang — so guard explicitly at
// each op's entry point rather than relying on the (currently
// unavailable) op tag. `op_name` is folded into the error message so it
// names whichever op actually triggered the guard.
void check_not_capturing(cudaStream_t stream, const char *op_name) {
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    CUDA_RUNTIME(cudaStreamIsCapturing(stream, &capture_status));
    STD_TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone, op_name,
                    " performs a host-side CUDA synchronization and cannot "
                    "run inside CUDA graph capture (e.g. "
                    "torch.compile(mode=\"reduce-overhead\")); this op "
                    "cannot be safely captured — call it outside the "
                    "compiled region.");
}

Tensor match_dispatch(const Tensor &cost, SolverFn solver, const char *op_name) {
    STD_TORCH_CHECK(cost.is_cuda(), "cost must be a CUDA tensor");
    STD_TORCH_CHECK(cost.dim() == 2, "cost must be 2-dimensional");
    STD_TORCH_CHECK(cost.is_contiguous(), "cost must be contiguous");

    const int32_t dev = cost.get_device_index();
    const assign_lap::SimpleCudaDeviceGuard device_guard(dev);

    if (cost.size(0) == 0 || cost.size(1) == 0) {
        return ts::full({cost.size(0)}, -1.0, ScT::Long, std::nullopt, cost.device());
    }

    cudaStream_t stream = current_stream_for(dev);
    check_not_capturing(stream, op_name);
    return solver(cost, stream);
}

Tensor munkres_cuda(const Tensor &cost) {
    return match_dispatch(cost, &match_munkres::solve, "munkres");
}

Tensor hybrid_cuda(const Tensor &cost) {
    return match_dispatch(cost, &match_hybrid::solve, "hybrid");
}

Tensor lawler_cuda(const Tensor &cost) {
    return match_dispatch(cost, &match_lawler::solve, "lawler");
}

Tensor jonker_dense_batch_cuda(const Tensor &costs) {
    STD_TORCH_CHECK(costs.is_cuda(), "costs must be a CUDA tensor");
    STD_TORCH_CHECK(costs.dim() == 3, "costs must be 3D (B, K, K), got ", costs.dim());
    STD_TORCH_CHECK(costs.size(1) == costs.size(2), "costs must be square (B, K, K)");
    STD_TORCH_CHECK(costs.is_contiguous(), "costs must be contiguous");

    const int32_t dev = costs.get_device_index();
    const assign_lap::SimpleCudaDeviceGuard device_guard(dev);

    const auto B = costs.size(0);
    const auto K = costs.size(1);

    if (B == 0 || K == 0) {
        return ts::full({B, K}, -1.0, ScT::Long, std::nullopt, costs.device());
    }

    cudaStream_t stream = current_stream_for(dev);
    check_not_capturing(stream, "jonker_dense_batch");

    auto cost_f = ts::to(costs, ScT::Float);
    // One sentinel across the whole batch, matching the CPU
    // jonker_dense_batch policy in cpu/ops_stable.cpp.
    const float sentinel = assign_lap::compute_inf_sentinel<float>(cost_f, stream);
    cost_f =
        ts::contiguous(assign_lap::rewrite_inf_to_sentinel(cost_f, sentinel, stream));

    int tile = match_batch::select_tile(static_cast<int>(K));
    STD_TORCH_CHECK(tile > 0, "Batched JV (CUDA) supports K <= ", match_batch::MAX_TILE,
                    ", got K=", K);

    Tensor result = ts::new_empty(costs, {B, K}, ScT::Long);

    match_batch::launch(tile, static_cast<int>(B), static_cast<int>(K),
                        cost_f.const_data_ptr<float>(),
                        result.mutable_data_ptr<int64_t>(), stream);

    cudaError_t err = cudaGetLastError();
    STD_TORCH_CHECK(err == cudaSuccess,
                    "batched JV kernel launch failed: ", cudaGetErrorString(err));
    return result;
}

} // namespace

STABLE_TORCH_LIBRARY_IMPL(assignment, CUDA, m) {
    m.impl("munkres", TORCH_BOX(&munkres_cuda));
    m.impl("hybrid", TORCH_BOX(&hybrid_cuda));
    m.impl("lawler", TORCH_BOX(&lawler_cuda));
    m.impl("jonker_dense_batch", TORCH_BOX(&jonker_dense_batch_cuda));
}
