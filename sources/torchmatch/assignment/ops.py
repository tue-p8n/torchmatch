"""
Direct access to ``torch.ops.assignment.*`` ops.

Each attribute is the corresponding op handle, so callers that want to
pin a specific backend (e.g. for benchmarking or to sidestep the
dispatcher's AUTO branch) write::

    from torchmatch.assignment.ops import jonker_dense

    result = jonker_dense(cost)
"""

from __future__ import annotations

import torch

_OPS = torch.ops.assignment

jonker_scalar = _OPS.jonker_scalar
jonker_dense = _OPS.jonker_dense
jonker_compact = _OPS.jonker_compact

jonker_dense_batch = _OPS.jonker_dense_batch
jonker_compact_batch = _OPS.jonker_compact_batch
jonker_dense_batch_unpacked = _OPS.jonker_dense_batch_unpacked
jonker_compact_batch_unpacked = _OPS.jonker_compact_batch_unpacked

greedy = _OPS.greedy

# CUDA-only ops; bind when present, omit when the CUDA extension is
# absent (CPU-only build). hasattr() on torch.ops.<ns> returns True
# only after the op is registered, which happens during load_cuda().
if hasattr(_OPS, "munkres"):
    munkres = _OPS.munkres
if hasattr(_OPS, "hybrid"):
    hybrid = _OPS.hybrid
if hasattr(_OPS, "lawler"):
    lawler = _OPS.lawler

# CUDA-only entries are exported only when the CUDA extension is loaded;
# `from torchmatch.assignment.ops import *` then follows the same rule.
__all__ = [
    "greedy",
    "jonker_compact",
    "jonker_compact_batch",
    "jonker_compact_batch_unpacked",
    "jonker_dense",
    "jonker_dense_batch",
    "jonker_dense_batch_unpacked",
    "jonker_scalar",
]
if "munkres" in globals():
    __all__ += ["munkres"]
if "hybrid" in globals():
    __all__ += ["hybrid"]
if "lawler" in globals():
    __all__ += ["lawler"]
