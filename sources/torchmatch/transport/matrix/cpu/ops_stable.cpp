// Stable-ABI variant of ops.cpp — schema registration only.
// Uses STABLE_TORCH_LIBRARY_FRAGMENT instead of TORCH_LIBRARY_FRAGMENT.

#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(transport, m) {
    m.def("exact_emd(Tensor cost, Tensor? mask=None, Tensor? a=None, "
          "Tensor? b=None) -> Tensor");
}
