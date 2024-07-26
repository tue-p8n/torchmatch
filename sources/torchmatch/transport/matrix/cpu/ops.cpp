// Schemas for transport.matrix CPU ops.
//
// Python-side custom_op-registered ops (log_sinkhorn, sinkhorn_divergence,
// unbalanced_sinkhorn) declare their own schemas via the decorator.
// exact_emd has a C++ impl, so its schema is declared here and the impl
// is registered in exact_emd_op.cpp.

#include <torch/library.h>

TORCH_LIBRARY_FRAGMENT(transport, m) {
    m.def("exact_emd(Tensor cost, Tensor? mask=None, Tensor? a=None, "
          "Tensor? b=None) -> Tensor");
}
