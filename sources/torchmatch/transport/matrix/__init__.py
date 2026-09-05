"""
Cost-matrix optimal-transport solvers.

Takes a (B, N, M) cost matrix and returns a transport plan (or scalar
divergence). The dispatcher pattern mirrors :mod:`torchmatch.assignment`.

See ``docs/superpowers/specs/2026-05-21-transport-ops-design.md``.
"""

from __future__ import annotations

import torch

from torchmatch.transport.matrix._cpu import load_cpu
from torchmatch.transport.matrix._cuda import load_cuda

# Populate torch.ops.transport.* before ops.py and _solve.py import-time
# bind to it. Mirrors the load order in torchmatch.assignment.
load_cpu()
if torch.cuda.is_available():
    load_cuda()

from torchmatch.transport.matrix import ops  # noqa: E402
from torchmatch.transport.matrix._solve import (  # noqa: E402
    Backend,
    marginal_error,
    solve,
)

__all__ = ["Backend", "marginal_error", "ops", "solve"]
