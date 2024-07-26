"""Backward-compatible re-exports for apply kernels.

The actual implementations live in:
- apply_raw.py: raw-form kernels (mat5, deprecated vec/mat wrappers)
- apply_shifted.py: shifted-form kernels (vec/mat with shifted potentials)
"""

from __future__ import annotations

from torchmatch.transport.samples.kernels.apply_raw import (  # noqa: F401
    apply_plan_vec_sqeuclid,
    apply_plan_mat_sqeuclid,
    mat5_sqeuclid,
)
from torchmatch.transport.samples.kernels.apply_shifted import (  # noqa: F401
    apply_plan_mat_shifted,
    apply_plan_vec_shifted,
)
