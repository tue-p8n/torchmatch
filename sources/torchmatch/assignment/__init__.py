"""
Integer linear assignment problem (LAP) solvers.

Public surface:

- :func:`solve` -- the dispatcher; resolves AUTO at call time so the
  picked op traces cleanly under ``torch.compile``.
- :class:`Backend` -- backend choices accepted by :func:`solve`.
- :mod:`ops` -- direct access to ``torch.ops.assignment.*`` op handles
  for callers that want to pin a backend.
"""

from __future__ import annotations

import torch

from torchmatch.assignment._cpu import load_cpu
from torchmatch.assignment._cuda import load_cuda

# Load native extensions first so torch.ops.assignment.* is populated
# before _solve.py and ops.py import-time bind to it.
load_cpu()
if torch.cuda.is_available():
    load_cuda()

# Triggers the @torch.library.custom_op registration so the op exists at
# torch.ops.assignment.greedy before ops.py imports run.
import torchmatch.assignment._greedy  # noqa: E402, F401
from torchmatch.assignment import ops  # noqa: E402
from torchmatch.assignment._auction import auction_assignment  # noqa: E402
from torchmatch.assignment._cost import assignment_cost  # noqa: E402
from torchmatch.assignment._solve import Backend, resolve_backend, solve  # noqa: E402

__all__ = [
    "Backend",
    "assignment_cost",
    "auction_assignment",
    "ops",
    "resolve_backend",
    "solve",
]
