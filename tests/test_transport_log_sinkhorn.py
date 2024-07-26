"""Correctness tests for torchmatch.transport.matrix log_sinkhorn (2D core)."""

from __future__ import annotations

import torch
from torchmatch.transport.matrix._log_sinkhorn import (
    _sinkhorn_log_plan_2d,
    _sinkhorn_log_plan_3d,
)


def test_uniform_cost_yields_uniform_plan():
    # Uniform cost -> uniform plan; row sums to a, col sums to b.
    n, m = 4, 4
    cost = torch.ones(n, m, dtype=torch.float64)
    a = torch.full((n,), 1.0 / n, dtype=torch.float64)
    b = torch.full((m,), 1.0 / m, dtype=torch.float64)
    log_plan = _sinkhorn_log_plan_2d(cost, eps=0.1, n_iter=100, a=a, b=b)
    plan = log_plan.exp()
    assert torch.allclose(plan.sum(dim=1), a, atol=1e-6)
    assert torch.allclose(plan.sum(dim=0), b, atol=1e-6)


def test_forbidden_row_yields_minus_inf_log_plan():
    # A row of all +inf cost must yield a row of all -inf log_plan;
    # the orthogonal axis must not be poisoned with NaN.
    cost = torch.tensor(
        [
            [float("inf"), float("inf"), float("inf")],
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
        ],
        dtype=torch.float64,
    )
    a = torch.full((3,), 1.0 / 3, dtype=torch.float64)
    b = torch.full((3,), 1.0 / 3, dtype=torch.float64)
    log_plan = _sinkhorn_log_plan_2d(cost, eps=0.1, n_iter=100, a=a, b=b)
    assert torch.isneginf(log_plan[0]).all()
    assert torch.isfinite(log_plan[1:]).all()


def test_logsumexp_is_finite_when_some_entries_finite():
    # Mixed forbidden / finite costs: as long as every row has at least one
    # finite entry, the output must be finite where the input was.
    cost = torch.tensor(
        [
            [1.0, float("inf"), 3.0],
            [float("inf"), 2.0, float("inf")],
            [4.0, 5.0, 6.0],
        ],
        dtype=torch.float64,
    )
    a = torch.full((3,), 1.0 / 3, dtype=torch.float64)
    b = torch.full((3,), 1.0 / 3, dtype=torch.float64)
    log_plan = _sinkhorn_log_plan_2d(cost, eps=0.1, n_iter=100, a=a, b=b)
    # finite-cost entries -> finite log_plan; +inf-cost entries -> -inf log_plan.
    finite_mask = torch.isfinite(cost)
    assert torch.isfinite(log_plan[finite_mask]).all()
    assert torch.isneginf(log_plan[~finite_mask]).all()


def test_3d_batch_decouples():
    # Two independent 3x3 problems; per-batch plans must each satisfy
    # marginal constraints independently.
    cost = torch.stack(
        [
            torch.tensor(
                [[1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [2.0, 3.0, 1.0]], dtype=torch.float64
            ),
            torch.tensor(
                [[3.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 1.0]], dtype=torch.float64
            ),
        ]
    )
    bsz, n, m = cost.shape
    a = torch.full((bsz, n), 1.0 / n, dtype=torch.float64)
    b = torch.full((bsz, m), 1.0 / m, dtype=torch.float64)
    log_plan = _sinkhorn_log_plan_3d(cost, eps=0.1, n_iter=100, a=a, b=b)
    plan = log_plan.exp()
    assert plan.shape == (2, 3, 3)
    for i in range(2):
        assert torch.allclose(plan[i].sum(dim=1), a[i], atol=1e-6)
        assert torch.allclose(plan[i].sum(dim=0), b[i], atol=1e-6)


def test_3d_empty_batch_returns_empty():
    # (0, N, M) input must produce (0, N, M) output without crashing.
    cost = torch.empty(0, 3, 4, dtype=torch.float64)
    a = torch.empty(0, 3, dtype=torch.float64)
    b = torch.empty(0, 4, dtype=torch.float64)
    log_plan = _sinkhorn_log_plan_3d(cost, eps=0.1, n_iter=100, a=a, b=b)
    assert log_plan.shape == (0, 3, 4)


def test_eps_scaling_converges_to_same_plan_as_fixed_eps():
    # On a benign problem, eps-scaling and fixed-eps with enough iterations
    # should converge to (approximately) the same plan.
    torch.manual_seed(0)
    cost = torch.rand(5, 5, dtype=torch.float64) * 2.0
    a = torch.full((5,), 0.2, dtype=torch.float64)
    b = torch.full((5,), 0.2, dtype=torch.float64)

    from torchmatch.transport.matrix._log_sinkhorn import log_sinkhorn_plan

    plan_fixed = (
        log_sinkhorn_plan(
            cost.unsqueeze(0),
            eps=0.05,
            n_iter=500,
            a=a.unsqueeze(0),
            b=b.unsqueeze(0),
            scaling=None,
        )
        .squeeze(0)
        .exp()
    )
    plan_scaled = (
        log_sinkhorn_plan(
            cost.unsqueeze(0),
            eps=0.05,
            n_iter=200,
            a=a.unsqueeze(0),
            b=b.unsqueeze(0),
            scaling=0.5,
        )
        .squeeze(0)
        .exp()
    )

    assert torch.allclose(plan_fixed, plan_scaled, atol=1e-4)
