"""Direct handles for torch.ops.transport.* matrix-face ops."""

from __future__ import annotations

import torch

exact_emd = torch.ops.transport.exact_emd
log_sinkhorn = torch.ops.transport.log_sinkhorn
sinkhorn_divergence = torch.ops.transport.sinkhorn_divergence
unbalanced_sinkhorn = torch.ops.transport.unbalanced_sinkhorn

__all__ = [
    "exact_emd",
    "log_sinkhorn",
    "sinkhorn_divergence",
    "unbalanced_sinkhorn",
]
