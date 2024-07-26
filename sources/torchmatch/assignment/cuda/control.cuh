// Shared device-side control plumbing for the torch-assign LAP backends.
//
// Each backend defines its own *ControlBlock* struct (per-backend
// layout) in its translation unit. This header provides:
//   - LapErrorCode enum     uniform error reporting across backends
//   - alloc_cb / free_cb    helpers for a single managed page
//   - prefetch_cb           one-time hint to keep the page on device
//   - set_error             device helper; atomic-CAS first-error wins
//
// Intent: collapse the per-iteration host readbacks of multiple
// `__managed__` flags into a single hot managed page so managed
// dereferences after a kernel sync are page-fault-free and cudaMemcpy
// round-trips drop out of the inner loop.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace assign_lap {

enum LapErrorCode : int {
    ERR_OK = 0,
    ERR_NONPOS_MIN = 1, // dual-update minimum was non-positive
    ERR_MAX_ITERS = 2,  // saturating iteration counter exceeded
};

// Allocate a managed page sized for type T. Caller must invoke free_cb.
//
// Intentionally NOT calling cudaMemAdvise(SetPreferredLocation):
// pinning the page to the device makes every host read fault,
// migrate, and flap against the next kernel write. The default policy
// (no preferred location, on-demand migration) keeps the page wherever
// it was last touched, which matches the access pattern of "kernel
// writes, host reads, kernel writes, host reads".
template <typename T>
inline T *alloc_cb(int /*device*/) {
    T *p = nullptr;
    cudaError_t err = cudaMallocManaged(&p, sizeof(T));
    if (err != cudaSuccess)
        return nullptr;
    cudaMemset(p, 0, sizeof(T));
    return p;
}

template <typename T>
inline void free_cb(T *p) {
    if (p)
        cudaFree(p);
}

template <typename T>
inline void prefetch_cb(T *p, int device, cudaStream_t stream) {
    if (p)
        cudaMemPrefetchAsync(p, sizeof(T), device, stream);
}

// Device helper: set error code if not already set; first error wins.
template <typename CB>
__device__ inline void set_error(CB *cb, LapErrorCode code) {
    atomicCAS(&cb->error_code, static_cast<int>(ERR_OK), static_cast<int>(code));
}

} // namespace assign_lap
