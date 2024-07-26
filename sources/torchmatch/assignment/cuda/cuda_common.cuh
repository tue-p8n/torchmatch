#pragma once

// Shared utilities for the CUDA Hungarian solvers (munkres.cu,
// hybrid.cu, lawler.cu). Each backend originally redefined the same
// boilerplate (gpuAssert, CUDA_RUNTIME, LAUNCH, near_zero, cost_inf,
// vertex-state constants, PyTorch input prep). Lifting them here
// keeps the three solvers symmetric and makes per-iteration discipline
// (no host readbacks, single managed page) a property of the shared
// layer rather than a per-file convention.

#include <ATen/ATen.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
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

// Inspect a CUDA cost tensor and return the +inf sentinel
// (max_finite + 1) * (K + 1) used to rewrite forbidden edges before
// the solve. Rejects NaN (caller bug: use +inf for forbidden edges)
// and -inf (unboundedly preferable edge has no defensible meaning and
// breaks the solver's dual-variable invariants). Works for both 2-D
// (N, M) and 3-D (B, N, M) inputs: K is derived from the trailing two
// dims and shared across the batch, matching the CPU policy.
template <typename scalar_t>
inline scalar_t compute_inf_sentinel(const at::Tensor &cost_typed) {
    const bool has_nan = at::isnan(cost_typed).any().template item<bool>();
    TORCH_CHECK(!has_nan, "cost matrix contains NaN; use +inf for forbidden pairs, "
                          "NaN signals an upstream bug");
    const auto neg_inf = -std::numeric_limits<scalar_t>::infinity();
    const bool has_neg_inf = (cost_typed == neg_inf).any().template item<bool>();
    TORCH_CHECK(!has_neg_inf,
                "cost matrix contains -inf; -inf has no forbidden-edge meaning");
    const auto finite_mask = at::isfinite(cost_typed);
    const auto rows = cost_typed.size(-2);
    const auto cols = cost_typed.size(-1);
    const auto K = std::max(rows, cols);
    if (!finite_mask.any().template item<bool>()) {
        return static_cast<scalar_t>(1);
    }
    const auto max_finite =
        cost_typed.masked_select(finite_mask).max().template item<scalar_t>();
    return (max_finite + static_cast<scalar_t>(1)) *
           (static_cast<scalar_t>(K) + static_cast<scalar_t>(1));
}

// Rewrite non-finite (+inf) cells to the supplied sentinel. NaN and
// -inf are already rejected upstream by compute_inf_sentinel; only
// +inf cells survive into this call.
inline at::Tensor rewrite_inf_to_sentinel(const at::Tensor &cost_typed,
                                          double sentinel) {
    return at::where(at::isfinite(cost_typed), cost_typed, sentinel);
}

// Apply the +inf rewrite and pad the cost to square (K, K) where K =
// max(rows, cols). Padded cells take the sentinel so any match into
// them is provably suboptimal and gets dropped on output repacking.
inline at::Tensor pad_and_rewrite(const at::Tensor &cost_finite_or_inf,
                                  double sentinel) {
    auto finite_mask = at::isfinite(cost_finite_or_inf);
    auto cost = at::where(finite_mask, cost_finite_or_inf, sentinel);
    const auto rows = cost.size(0);
    const auto cols = cost.size(1);
    if (rows == cols)
        return cost.contiguous();
    const auto K = std::max(rows, cols);
    auto padded = at::full({K, K}, sentinel, cost.options());
    padded.narrow(0, 0, rows).narrow(1, 0, cols).copy_(cost);
    return padded;
}

// Build the row→col output from a host-resident column-assignment array.
// Cells assigned to padded columns (col >= original cols) normalize to
// -1, the standard "unmatched row" marker.
inline at::Tensor repack_row_to_col(const int *h_col_assignment, int64_t rows,
                                    int64_t cols, at::Device device) {
    auto row_to_col = at::empty({rows}, at::TensorOptions().dtype(at::kLong));
    int64_t *out = row_to_col.data_ptr<int64_t>();
    for (int64_t r = 0; r < rows; r++) {
        int c = h_col_assignment[r];
        out[r] = (c >= 0 && c < cols) ? static_cast<int64_t>(c) : -1;
    }
    return row_to_col.to(device);
}

} // namespace assign_lap
