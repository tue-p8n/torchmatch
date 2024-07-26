"""User-facing tests for transport.samples.loss(...)."""

from __future__ import annotations

import pytest
import torch
from _triton_guard import triton_kernel_compiles
from torchmatch.transport.samples import loss

cuda_required = pytest.mark.skipif(
    not triton_kernel_compiles(),
    reason="CUDA + functional Triton launcher required for samples face",
)


@cuda_required
def test_loss_returns_scalar():
    x = torch.randn(64, 8, device="cuda")
    y = torch.randn(64, 8, device="cuda")
    out = loss(x, y, blur=0.1)
    assert out.shape == ()
    assert torch.isfinite(out)


@cuda_required
def test_loss_debias_self_self_near_zero():
    torch.manual_seed(0)
    x = torch.randn(64, 8, device="cuda")
    out = loss(x, x, blur=0.1, debias=True)
    assert out.item() == pytest.approx(0.0, abs=1e-2)


@cuda_required
def test_loss_unbalanced_reach_param():
    x = torch.randn(64, 8, device="cuda")
    y = torch.randn(64, 8, device="cuda")
    out_balanced = loss(x, y, blur=0.1, reach=None)
    out_unbalanced = loss(x, y, blur=0.1, reach=1.0)
    assert not torch.allclose(out_balanced, out_unbalanced, atol=1e-3)


@cuda_required
def test_loss_grad_flows():
    x = torch.randn(64, 8, device="cuda", requires_grad=True)
    y = torch.randn(64, 8, device="cuda")
    out = loss(x, y, blur=0.1)
    out.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@cuda_required
def test_loss_batched_returns_shape():
    x = torch.rand(3, 16, 4, device="cuda")
    y = torch.rand(3, 12, 4, device="cuda")
    out = loss(x, y, blur=0.1)
    assert out.shape == (3,)
    assert out.isfinite().all()


def test_loss_batched_rejects_mismatched_ndim():
    x = torch.randn(3, 8, 4)
    y = torch.randn(8, 4)
    with pytest.raises(ValueError, match="both be 3-D"):
        loss(x, y, blur=0.1)


def test_loss_rejects_unsupported_p():
    # CPU-side check; the p validation runs before any CUDA call.
    x = torch.randn(64, 8)
    y = torch.randn(64, 8)
    with pytest.raises(ValueError, match=r"p=2 is the only supported"):
        loss(x, y, blur=0.1, p=1)


def test_loss_rejects_cpu():
    x = torch.randn(64, 8)
    y = torch.randn(64, 8)
    with pytest.raises(RuntimeError, match="requires CUDA"):
        loss(x, y, blur=0.1)


def test_loss_rejects_non_positive_blur():
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    with pytest.raises(ValueError, match="blur must be positive"):
        loss(x, y, blur=0.0)


def test_loss_rejects_out_of_range_scaling():
    x = torch.randn(8, 4)
    y = torch.randn(8, 4)
    with pytest.raises(ValueError, match=r"scaling must be in \(0, 1\)"):
        loss(x, y, blur=0.1, scaling=1.0)
    with pytest.raises(ValueError, match=r"scaling must be in \(0, 1\)"):
        loss(x, y, blur=0.1, scaling=0.0)
