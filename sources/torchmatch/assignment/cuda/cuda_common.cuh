#pragma once

// Shared utilities for the CUDA Hungarian solvers (munkres.cu,
// hybrid.cu, lawler.cu) and the tiled batched JV kernel (dispatch.cu).
// Each backend originally redefined the same boilerplate (gpuAssert,
// CUDA_RUNTIME, LAUNCH, near_zero, cost_inf, vertex-state constants,
// tensor input prep). Lifting them here keeps the three solvers
// symmetric and makes per-iteration discipline (no host readbacks,
// single managed page) a property of the shared layer rather than a
// per-file convention.
//
// Stable ABI: everything below that touches tensors operates on
// torch::stable::Tensor via torch::stable::ops (torch/csrc/stable/ops.h)
// and small CUDA kernels over raw data_ptr()s — never at::Tensor. The
// AOTI C-shim / torch::stable expose no isnan/isfinite/masked_select/
// where equivalents, so compute_inf_sentinel and rewrite_inf_to_sentinel
// scan and rewrite the cost buffer with dedicated kernels instead,
// mirroring the raw-pointer-loop idiom cpu/ops_stable.cpp's
// compute_sentinel() already uses for the CPU backend (just
// parallelized over the device buffer rather than looped on the host).

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/shim_utils.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace assign_lap {

inline void gpu_assert_throw(cudaError_t code, const char *file, int line) {
    if (code == cudaSuccess)
        return;
    char buf[256];
    std::snprintf(buf, sizeof(buf), "CUDA error: %s at %s:%d", cudaGetErrorString(code),
                  file, line);
    throw std::runtime_error(buf);
}

// RAII holder for a cudaMallocAsync'd device buffer: guarantees the
// cudaFreeAsync happens even if a CUDA_RUNTIME(...) check between
// allocation and the intended free throws (e.g. a launch or memcpy
// failure), instead of leaking the allocation on that error path.
template <typename T>
class AsyncDeviceBuffer {
  public:
    AsyncDeviceBuffer(size_t count, cudaStream_t stream) : stream_(stream) {
        gpu_assert_throw(cudaMallocAsync(&ptr_, count * sizeof(T), stream_), __FILE__,
                         __LINE__);
    }
    AsyncDeviceBuffer(const AsyncDeviceBuffer &) = delete;
    AsyncDeviceBuffer &operator=(const AsyncDeviceBuffer &) = delete;
    ~AsyncDeviceBuffer() {
        if (ptr_ != nullptr)
            (void)cudaFreeAsync(ptr_, stream_);
    }

    T *get() const { return ptr_; }

  private:
    T *ptr_ = nullptr;
    cudaStream_t stream_;
};

} // namespace assign_lap

// Throw on any non-success cudaError_t. Used by both runtime API calls
// and the LAUNCH macro to surface kernel-launch errors at the call site.
#define CUDA_RUNTIME(expr) ::assign_lap::gpu_assert_throw((expr), __FILE__, __LINE__)

// Launch a kernel on a stream and immediately check for launch errors.
// Errors during execution surface at the next synchronizing call; this
// only catches launch-time failures (bad config, missing device code).
#define LAUNCH(kernel, grid, block, stream, ...)                                       \
    do {                                                                               \
        (kernel)<<<dim3(grid), dim3(block), 0, (stream)>>>(__VA_ARGS__);               \
        CUDA_RUNTIME(cudaGetLastError());                                              \
    } while (0)

namespace assign_lap {

// Default tolerance for near-zero slack tests in the primed-zeros
// loop. `1e-6` matches the historical Date-Nagi / HLAP code; lawler.cu
// uses a tighter `1e-8` because its double-precision slack accumulates
// fewer rounding errors and a false-positive zero would cost an
// iteration.
inline constexpr double EPS_DEFAULT = 1e-6;
inline constexpr double EPS_TIGHT = 1e-8;

template <typename T>
__host__ __device__ constexpr T cost_inf() {
    // Large finite sentinel rather than std::numeric_limits<T>::infinity()
    // so reductions over masked-out cells still return a finite T that
    // compares against other masked cells in a defined way.
    return T(1e30);
}

template <typename T>
__device__ inline bool near_zero(T val, T eps = T(EPS_DEFAULT)) {
    return val > -eps && val < eps;
}

// Vertex states for the tree-augmentation kernels (hybrid.cu, lawler.cu).
// Shared because the same encoding flows through both implementations
// and renumbering one in isolation would silently break the other.
enum VertexState : int {
    DORMANT = 0,
    ACTIVE = 1,
    VISITED = 2,
    REVERSE = 3,
    AUGMENT = 4,
    MODIFIED = 5,
};

// RAII CUDA device guard built on the plain CUDA runtime API (no ATen /
// c10::cuda::CUDAGuard). Mirrors what the pre-stable-ABI code got from
// at::cuda::CUDAGuard: kernels launched against a tensor living on a
// non-default device must run with that device current, since a CUDA
// stream is only valid to use while its owning device is current.
class SimpleCudaDeviceGuard {
  public:
    explicit SimpleCudaDeviceGuard(int device) {
        CUDA_RUNTIME(cudaGetDevice(&prev_device_));
        if (prev_device_ != device)
            CUDA_RUNTIME(cudaSetDevice(device));
    }
    SimpleCudaDeviceGuard(const SimpleCudaDeviceGuard &) = delete;
    SimpleCudaDeviceGuard &operator=(const SimpleCudaDeviceGuard &) = delete;
    ~SimpleCudaDeviceGuard() {
        // Best-effort restore; a destructor must not throw.
        (void)cudaSetDevice(prev_device_);
    }

  private:
    int prev_device_ = 0;
};

// ---------------------------------------------------------------------
// Stable-ABI cost-matrix preprocessing (no ATen tensor ops).
// ---------------------------------------------------------------------

// CAS-loop atomicMax over float/double, correct for any sign (unlike the
// integer-bit-reinterpretation trick, which only works for non-negative
// values). Standard CUDA idiom for atomic float/double max.
__device__ __forceinline__ void atomic_max_val(float *addr, float val) {
    int *addr_i = reinterpret_cast<int *>(addr);
    int old = *addr_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_i, assumed,
                        __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void atomic_max_val(double *addr, double val) {
    unsigned long long *addr_i = reinterpret_cast<unsigned long long *>(addr);
    unsigned long long old = *addr_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_i, assumed,
                        __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

template <typename scalar_t>
struct SentinelStats {
    int has_nan;
    int has_neg_inf;
    int any_finite;
    scalar_t max_finite;
};

template <typename scalar_t>
__global__ void k_scan_sentinel_stats(const scalar_t *__restrict__ data, int64_t n,
                                      SentinelStats<scalar_t> *stats) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        const scalar_t v = data[i];
        if (isnan(v)) {
            stats->has_nan = 1;
        } else if (isinf(v)) {
            if (v < scalar_t(0))
                stats->has_neg_inf = 1;
            // +inf is a forbidden-edge marker, not an error; it is
            // excluded from the max-finite reduction below.
        } else {
            stats->any_finite = 1;
            atomic_max_val(&stats->max_finite, v);
        }
    }
}

// Inspect a CUDA cost tensor (2-D or 3-D; K is derived from the trailing
// two dims and shared across the batch, matching the CPU policy) and
// return the +inf sentinel (max_finite + 1) * (K + 1). Rejects NaN
// (caller bug: use +inf for forbidden pairs) and -inf (unboundedly
// preferable edge has no defensible meaning and breaks the solver's
// dual-variable invariants) — same policy as compute_sentinel in
// cpu/ops_stable.cpp, computed here via a device-side scan kernel
// instead of a host loop.
template <typename scalar_t>
inline scalar_t compute_inf_sentinel(const torch::stable::Tensor &cost,
                                     cudaStream_t stream) {
    const int64_t n = cost.numel();
    const int64_t rows = cost.size(cost.dim() - 2);
    const int64_t cols = cost.size(cost.dim() - 1);
    const int64_t K = std::max(rows, cols);

    SentinelStats<scalar_t> h{};
    h.max_finite = -std::numeric_limits<scalar_t>::infinity();

    // Holds the allocation via RAII so a throwing CUDA_RUNTIME(...) check
    // below (launch failure, memcpy failure) still frees the buffer
    // instead of leaking it.
    AsyncDeviceBuffer<SentinelStats<scalar_t>> d_stats(1, stream);
    CUDA_RUNTIME(
        cudaMemcpyAsync(d_stats.get(), &h, sizeof(h), cudaMemcpyHostToDevice, stream));

    if (n > 0) {
        constexpr int threads = 256;
        const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
        k_scan_sentinel_stats<scalar_t><<<blocks, threads, 0, stream>>>(
            static_cast<const scalar_t *>(cost.const_data_ptr()), n, d_stats.get());
        CUDA_RUNTIME(cudaGetLastError());
    }

    CUDA_RUNTIME(
        cudaMemcpyAsync(&h, d_stats.get(), sizeof(h), cudaMemcpyDeviceToHost, stream));
    CUDA_RUNTIME(cudaStreamSynchronize(stream));

    STD_TORCH_CHECK(!h.has_nan,
                    "cost matrix contains NaN; use +inf for forbidden pairs, "
                    "NaN signals an upstream bug");
    STD_TORCH_CHECK(!h.has_neg_inf,
                    "cost matrix contains -inf; -inf has no forbidden-edge meaning");

    if (!h.any_finite)
        return static_cast<scalar_t>(1);
    return (h.max_finite + static_cast<scalar_t>(1)) *
           (static_cast<scalar_t>(K) + static_cast<scalar_t>(1));
}

template <typename scalar_t>
__global__ void k_rewrite_pos_inf(scalar_t *data, int64_t n, scalar_t sentinel) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (int64_t i = idx; i < n; i += stride) {
        if (isinf(data[i]))
            data[i] = sentinel;
    }
}

// Rewrite +inf cells to `sentinel` on a fresh clone of `cost` (the input
// is never mutated in place). NaN and -inf are already rejected upstream
// by compute_inf_sentinel; only +inf cells survive into this call.
template <typename scalar_t>
inline torch::stable::Tensor rewrite_inf_to_sentinel(const torch::stable::Tensor &cost,
                                                     scalar_t sentinel,
                                                     cudaStream_t stream) {
    torch::stable::Tensor out = torch::stable::clone(cost);
    const int64_t n = out.numel();
    if (n > 0) {
        constexpr int threads = 256;
        const int blocks = (int)std::min<int64_t>((n + threads - 1) / threads, 4096);
        k_rewrite_pos_inf<scalar_t><<<blocks, threads, 0, stream>>>(
            out.mutable_data_ptr<scalar_t>(), n, sentinel);
        CUDA_RUNTIME(cudaGetLastError());
    }
    return out;
}

// Build the row->col output from a host-resident column-assignment array.
// Cells assigned to padded columns (col >= original cols) normalize to
// -1, the standard "unmatched row" marker. Fills a CPU stable::Tensor on
// the host (matching the CPU-fill-then-transfer shape of the pre-stable
// version) and moves it to `device` via torch::stable::to.
inline torch::stable::Tensor repack_row_to_col(const int *h_col_assignment,
                                               int64_t rows, int64_t cols,
                                               torch::stable::Device device) {
    torch::stable::Tensor row_to_col =
        torch::stable::empty({rows}, torch::headeronly::ScalarType::Long);
    int64_t *out = row_to_col.mutable_data_ptr<int64_t>();
    for (int64_t r = 0; r < rows; r++) {
        int c = h_col_assignment[r];
        out[r] = (c >= 0 && c < cols) ? static_cast<int64_t>(c) : -1;
    }
    return torch::stable::to(row_to_col, device);
}

} // namespace assign_lap
