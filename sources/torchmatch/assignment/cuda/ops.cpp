// Op schema for the torchmatch CUDA extension.
//
// Registers each solver as its own op under torch.ops.assignment. The
// CUDA implementations bind in dispatch.cu via TORCH_LIBRARY_IMPL.
//
// Uses TORCH_LIBRARY_FRAGMENT so the sibling CPU extension can
// contribute additional ops (and an additional CPU implementation of
// `jonker_dense_batch`) to the same `assignment` namespace from its own
// translation unit.
//
// The three primed-zeros Hungarian solvers (Munkres classical, Lawler
// tree, experimental Munkres hybrid) all carry the `cudagraph_unsafe`
// tag because their inner loops do a host-side `cudaStreamSynchronize`
// to read managed-memory flags. Stream capture (CUDA graphs,
// `torch.compile(mode="reduce-overhead")`) cannot proceed across a
// sync, so the dispatcher routes around them with a graph break
// instead of crashing at capture time. The CUDA backend of
// `jonker_dense_batch` is also a Hungarian solver (Jonker-Volgenant
// sub-family) but is fully capturable; see below.
//
// `jonker_dense_batch` deliberately carries no tag: its CUDA backend
// dispatches to a single-block-per-problem shared-memory JV kernel
// (jonker_tiled.cuh) with no global syncs and no managed-memory
// readbacks. The host shim only calls `cudaGetLastError()`, which is
// host-only, so the kernel is fully capturable.

#include <ATen/core/enum_tag.h>
#include <torch/library.h>

// `jonker_dense_batch` is registered (m.def) in the CPU extension's
// ops.cpp; it is a shared op with CPU as the default backend. The CUDA
// extension only adds a CUDA impl in dispatch.cu.

TORCH_LIBRARY_FRAGMENT(assignment, m) {
    m.def("munkres(Tensor cost) -> Tensor", {at::Tag::cudagraph_unsafe});
    m.def("hybrid(Tensor cost) -> Tensor", {at::Tag::cudagraph_unsafe});
    m.def("lawler(Tensor cost) -> Tensor", {at::Tag::cudagraph_unsafe});
}
