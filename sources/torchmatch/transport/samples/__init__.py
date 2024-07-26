"""
Samples-face optimal transport.

Point clouds in, scalar OT cost / divergence out. Requires CUDA +
Triton. See ``docs/superpowers/specs/2026-05-21-transport-ops-design.md``.
"""

from __future__ import annotations

from torchmatch.transport.samples._loss import loss

__all__ = ["loss"]
