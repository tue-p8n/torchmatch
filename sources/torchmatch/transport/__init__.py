"""
Continuous optimal-transport (OT) solvers.

Two sub-packages:

- :mod:`torchmatch.transport.matrix`: cost-matrix in, plan / divergence
  out. The primary dispatcher; mirrors :mod:`torchmatch.assignment`.
- :mod:`torchmatch.transport.samples`: point-clouds in, scalar loss
  out. Backed by Triton kernels (CUDA-only).

See ``docs/superpowers/specs/2026-05-21-transport-ops-design.md``.
"""

from __future__ import annotations

import torch

from torchmatch.transport.matrix._cpu import load_cpu
from torchmatch.transport.matrix._cuda import load_cuda

load_cpu()
if torch.cuda.is_available():
    load_cuda()

from torchmatch.transport import matrix, samples  # noqa: E402

__all__ = ["load_cpu", "load_cuda", "matrix", "samples"]
