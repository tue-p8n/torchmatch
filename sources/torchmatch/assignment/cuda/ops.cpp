// Op schema for the torchmatch CUDA extension.
//
// Registers each solver as its own op under torch.ops.assignment. The
// CUDA implementations bind in dispatch.cu via STABLE_TORCH_LIBRARY_IMPL.
//
// Uses STABLE_TORCH_LIBRARY_FRAGMENT so the sibling CPU extension can
// contribute additional ops (and an additional CPU implementation of
// `jonker_dense_batch`) to the same `assignment` namespace from its own
// translation unit.
//
// The three primed-zeros Hungarian solvers (Munkres classical, Lawler
// tree, experimental Munkres hybrid) each have inner loops that do a
// host-side `cudaStreamSynchronize` to read managed-memory flags. Stream
// capture (CUDA graphs, `torch.compile(mode="reduce-overhead")`) cannot
// proceed across a sync. Before this stable-ABI port, that was handled
// by tagging these three ops `at::Tag::cudagraph_unsafe`, which made
// the dispatcher route around them with a graph break instead of
// crashing at capture time.
//
// Resolved: the stable-ABI `StableLibrary::def()`
// (torch/csrc/stable/library.h, torch 2.11) takes only a schema string —
// `aoti_torch_library_def()` has no tag parameter anywhere in the AOTI
// C-shim (torch/csrc/inductor/aoti_torch/c/shim.h) as of this torch
// version, so there is no stable-ABI-safe way to attach
// `cudagraph_unsafe` here. Keeping this one `m.def()` call on the
// regular (non-stable) `TORCH_LIBRARY_FRAGMENT`/`at::Tag` would mean op
// *registration* still depends on `c10::Library`/`FunctionSchema`
// layout, defeating the point of the port. The human-ratified decision
// is to drop the tag and register uniformly via
// `STABLE_TORCH_LIBRARY_FRAGMENT` (this file's current state), and rely
// instead on an explicit runtime guard: `check_not_capturing()` in
// `dispatch.cu`, called from `match_dispatch()` (the shared entry point
// for `munkres`/`hybrid`/`lawler`) and from `jonker_dense_batch_cuda`,
// raises a clear `STD_TORCH_CHECK` error naming the offending op instead
// of silently crashing on the host sync mid-capture. The tradeoff this
// accepts: CUDA-graph capture of these ops now raises immediately
// instead of gracefully graph-breaking around them, so
// `torch.compile(mode="reduce-overhead")` requires calling them outside
// the compiled/captured region.
//
// `jonker_dense_batch`'s CUDA backend dispatches to a
// single-block-per-problem shared-memory JV kernel (jonker_tiled.cuh)
// with no global syncs inside the kernel itself, but its host-side
// wrapper (`compute_inf_sentinel` in cuda_common.cuh) does its own
// `cudaMallocAsync`/`cudaMemcpyAsync`/`cudaStreamSynchronize`/
// `cudaFreeAsync` round trip to read back NaN/-inf/max-finite stats —
// equally illegal during stream capture — so it carries the same
// `check_not_capturing()` guard as the other three ops.

#include <torch/csrc/stable/library.h>

// `jonker_dense_batch` is registered (m.def) in the CPU extension's
// ops_stable.cpp; it is a shared op with CPU as the default backend. The
// CUDA extension only adds a CUDA impl in dispatch.cu.

STABLE_TORCH_LIBRARY_FRAGMENT(assignment, m) {
    m.def("munkres(Tensor cost) -> Tensor");
    m.def("hybrid(Tensor cost) -> Tensor");
    m.def("lawler(Tensor cost) -> Tensor");
}
