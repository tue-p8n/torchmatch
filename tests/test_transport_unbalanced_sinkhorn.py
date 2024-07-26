"""Correctness tests for transport.matrix unbalanced_sinkhorn."""

from __future__ import annotations

import math

import torch
from torchmatch.transport.matrix import Backend, solve


def test_unbalanced_returns_log_plan_shape():
    cost = torch.rand(2, 3, 4, dtype=torch.float64)
    out = solve(
        cost,
        backend=Backend.UNBALANCED_SINKHORN,
        reg=0.1,
        n_iter=100,
        rho=1.0,
    )
    assert out.shape == (2, 3, 4)


def test_unbalanced_total_mass_below_balanced():
    # With finite rho, the relaxed problem transports strictly less mass
    # than the balanced version (some mass is "destroyed" via KL penalty).
    torch.manual_seed(0)
    cost = torch.rand(3, 3, dtype=torch.float64) * 5.0
    balanced = solve(
        cost,
        backend=Backend.LOG_SINKHORN,
        reg=0.1,
        n_iter=300,
    ).exp()
    unbalanced = solve(
        cost,
        backend=Backend.UNBALANCED_SINKHORN,
        reg=0.1,
        n_iter=300,
        rho=0.1,
    ).exp()
    assert unbalanced.sum().item() < balanced.sum().item()


def test_unbalanced_large_rho_approaches_balanced():
    # Very large rho recovers the balanced Sinkhorn problem.
    torch.manual_seed(0)
    cost = torch.rand(3, 3, dtype=torch.float64)
    balanced = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=300)
    unbalanced = solve(
        cost,
        backend=Backend.UNBALANCED_SINKHORN,
        reg=0.1,
        n_iter=300,
        rho=1e6,
    )
    assert torch.allclose(balanced, unbalanced, atol=1e-3)


def test_unbalanced_1x1_matches_closed_form():
    # Uniform-reference unbalanced KL-KL OT in 1x1:
    #   F(pi) = c*pi + eps*KL(pi|1) + rho*KL(pi|a) + rho*KL(pi|b)
    # First-order condition gives the closed form
    #   log pi* = (rho * (log a + log b) - c) / (eps + 2 * rho)
    # This anchor catches the round-4 bug: the buggy convex-combination
    # iteration converged to the balanced fixed point, which would give
    # log pi = log a - LSE(log_kernel + log v), not the formula above.
    a0, b0, c00 = 0.7, 0.9, 0.3
    rho, eps = 0.5, 0.1
    expected = (rho * (math.log(a0) + math.log(b0)) - c00) / (eps + 2.0 * rho)

    cost = torch.tensor([[[c00]]], dtype=torch.float64)
    a = torch.tensor([[a0]], dtype=torch.float64)
    b = torch.tensor([[b0]], dtype=torch.float64)
    log_plan = solve(
        cost,
        backend=Backend.UNBALANCED_SINKHORN,
        reg=eps,
        n_iter=2000,
        rho=rho,
        a=a,
        b=b,
    )
    assert torch.allclose(
        log_plan,
        torch.tensor([[[expected]]], dtype=torch.float64),
        atol=1e-10,
    )


def test_unbalanced_rejects_non_positive_rho():
    import pytest

    cost = torch.rand(2, 3, 3, dtype=torch.float64)
    with pytest.raises(ValueError, match="rho must be positive"):
        solve(cost, backend=Backend.UNBALANCED_SINKHORN, reg=0.1, rho=0.0)
    with pytest.raises(ValueError, match="rho must be positive"):
        solve(cost, backend=Backend.UNBALANCED_SINKHORN, reg=0.1, rho=-1.0)


def test_unbalanced_unpack_potentials_reconstruct_log_plan():
    # With f = -eps * log u and g = -eps * log v, log_plan_ij must satisfy
    #   log_plan_ij = (-f_i - cost_ij - g_j) / eps
    # i.e. cost_ij = -f_i - g_j - eps * log_plan_ij.
    # The unpacked path computes f, g from the converged log_plan via
    # f = rho * (logsumexp(log_plan, dim=-1) - log a). This round-trip
    # confirms the rho factor (and sign) without depending on a reference
    # implementation.
    torch.manual_seed(0)
    cost = torch.rand(3, 4, dtype=torch.float64) * 2.0
    eps, rho = 0.1, 0.5
    log_plan, f, g = solve(
        cost,
        backend=Backend.UNBALANCED_SINKHORN,
        reg=eps,
        n_iter=2000,
        rho=rho,
        unpack=True,
    )
    reconstructed_log_plan = (-f[:, None] - cost - g[None, :]) / eps
    assert torch.allclose(log_plan, reconstructed_log_plan, atol=1e-6)
