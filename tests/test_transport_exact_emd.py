"""Correctness tests for transport.matrix exact_emd."""

from __future__ import annotations

import pytest
import torch
from torchmatch.transport.matrix import Backend, solve


def test_exact_emd_returns_plan_shape():
    cost = torch.rand(2, 3, 4, dtype=torch.float64)
    plan = solve(cost, backend=Backend.EXACT_EMD)
    assert plan.shape == (2, 3, 4)


def test_exact_emd_plan_satisfies_marginals():
    torch.manual_seed(0)
    cost = torch.rand(4, 4, dtype=torch.float64)
    plan = solve(cost, backend=Backend.EXACT_EMD)
    a_uniform = torch.full((4,), 0.25, dtype=torch.float64)
    b_uniform = torch.full((4,), 0.25, dtype=torch.float64)
    assert torch.allclose(plan.sum(dim=1), a_uniform, atol=1e-6)
    assert torch.allclose(plan.sum(dim=0), b_uniform, atol=1e-6)


def test_exact_emd_low_eps_log_sinkhorn_approaches_exact():
    # Convergence sanity: small eps log_sinkhorn ~ exact_emd on small problem.
    torch.manual_seed(0)
    cost = torch.rand(4, 4, dtype=torch.float64)
    exact = solve(cost, backend=Backend.EXACT_EMD)
    log_plan = solve(cost, backend=Backend.LOG_SINKHORN, reg=1e-3, n_iter=500)
    sinkhorn_plan = log_plan.exp()
    assert torch.allclose(sinkhorn_plan, exact, atol=1e-2)


def test_exact_emd_rejects_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the negative test")
    cost = torch.rand(2, 3, 3, dtype=torch.float64, device="cuda")
    with pytest.raises(ValueError, match="EXACT_EMD is CPU-only"):
        solve(cost, backend=Backend.EXACT_EMD)


def test_exact_emd_rejects_inf_cost():
    cost = torch.rand(2, 3, 3, dtype=torch.float64)
    cost[0, 1, 2] = float("inf")
    with pytest.raises(ValueError, match=r"does not support \+inf"):
        solve(cost, backend=Backend.EXACT_EMD)


def test_exact_emd_rejects_mask():
    # mask is fused into cost as +inf before dispatch; EXACT_EMD then
    # rejects the +inf entries with the same clear error.
    cost = torch.rand(2, 3, 3, dtype=torch.float64)
    mask = torch.ones(2, 3, 3, dtype=torch.bool)
    mask[0, 1, 2] = False
    with pytest.raises(ValueError, match=r"does not support \+inf"):
        solve(cost, backend=Backend.EXACT_EMD, mask=mask)
