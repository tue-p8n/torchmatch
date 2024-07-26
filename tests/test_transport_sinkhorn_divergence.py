"""Correctness tests for transport.matrix sinkhorn_divergence."""

from __future__ import annotations

import torch
from torchmatch.transport.matrix import Backend, solve


def test_divergence_returns_scalar():
    cost = torch.rand(3, 4, dtype=torch.float64)
    out = solve(cost, backend=Backend.SINKHORN_DIVERGENCE, reg=0.1, n_iter=200)
    assert out.shape == ()
    assert torch.isfinite(out)


def test_divergence_returns_scalar_per_batch():
    cost = torch.rand(4, 3, 3, dtype=torch.float64)
    out = solve(cost, backend=Backend.SINKHORN_DIVERGENCE, reg=0.5, n_iter=200)
    assert out.shape == (4,)


def test_divergence_symmetric():
    # S_eps(a, b) ~ S_eps(b, a). Different costs encoding different problems.
    torch.manual_seed(0)
    cost_ab = torch.rand(3, 3, dtype=torch.float64)
    cost_ba = cost_ab.T.contiguous()
    s_ab = solve(
        cost_ab,
        backend=Backend.SINKHORN_DIVERGENCE,
        reg=0.1,
        n_iter=300,
    )
    s_ba = solve(
        cost_ba,
        backend=Backend.SINKHORN_DIVERGENCE,
        reg=0.1,
        n_iter=300,
    )
    assert abs(s_ab.item() - s_ba.item()) < 1e-4


def test_divergence_non_negative_on_well_posed():
    # Positivity of S_eps(a, b) = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b) holds
    # when all three OT problems share the same cost geometry (Feydy 2019).
    # Derive cost_aa and cost_bb from the same squared-Euclidean point clouds.
    # Higher reg (0.5) and enough iterations for the dual-potential bound to
    # tighten sufficiently at finite iteration count.
    torch.manual_seed(1)
    x = torch.rand(4, 3, dtype=torch.float64)
    y = torch.rand(4, 3, dtype=torch.float64)
    cost_xy = torch.cdist(x, y).pow(2)
    cost_xx = torch.cdist(x, x).pow(2)
    cost_yy = torch.cdist(y, y).pow(2)
    out = solve(
        cost_xy,
        backend=Backend.SINKHORN_DIVERGENCE,
        reg=0.5,
        n_iter=500,
        cost_aa=cost_xx,
        cost_bb=cost_yy,
    )
    assert out.item() >= -1e-5
