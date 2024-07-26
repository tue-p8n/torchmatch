"""End-to-end integration tests for the transport sub-package."""

from __future__ import annotations

import pytest
import torch
import torchmatch
from _triton_guard import triton_kernel_compiles


def test_both_faces_present():
    assert hasattr(torchmatch.transport, "matrix")
    assert hasattr(torchmatch.transport, "samples")


def test_matrix_face_backends_runnable():
    cost = torch.rand(2, 3, 3, dtype=torch.float64)
    from torchmatch.transport.matrix import Backend, solve

    for backend in (
        Backend.LOG_SINKHORN,
        Backend.SINKHORN_DIVERGENCE,
        Backend.UNBALANCED_SINKHORN,
        Backend.EXACT_EMD,
    ):
        out = solve(cost, backend=backend, reg=0.1, n_iter=20)
        assert torch.isfinite(out).all()


@pytest.mark.skipif(
    not triton_kernel_compiles(),
    reason="CUDA + functional Triton launcher required for samples face",
)
def test_samples_face_runnable():
    x = torch.randn(64, 8, device="cuda")
    y = torch.randn(64, 8, device="cuda")
    from torchmatch.transport.samples import loss

    out = loss(x, y, blur=0.1)
    assert torch.isfinite(out)


def test_no_orphaned_top_level_solve():
    # The relocate in Stage 0 must hold: no top-level transport.solve.
    assert not hasattr(torchmatch.transport, "solve")
    assert not hasattr(torchmatch.transport, "Backend")
