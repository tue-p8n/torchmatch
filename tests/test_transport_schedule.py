"""Tests for transport.matrix epsilon-scaling schedule."""

from __future__ import annotations

import pytest
import torch
from torchmatch.transport.matrix._schedule import build_eps_schedule


def test_fixed_eps_when_scaling_is_none():
    schedule = build_eps_schedule(
        cost=torch.rand(3, 4),
        reg=0.1,
        n_iter=50,
        scaling=None,
    )
    assert isinstance(schedule, torch.Tensor)
    assert len(schedule) == 50
    assert all(float(eps) == pytest.approx(0.1) for eps in schedule)


def test_geometric_decay_when_scaling_set():
    # max(cost) is 1.0; initial_eps = max(0.5 * 1.0, 0.1) = 0.5
    # schedule[k] = 0.5 * 0.5**k until it reaches 0.1, then holds at 0.1.
    cost = torch.full((3, 4), 1.0)
    schedule = build_eps_schedule(
        cost=cost,
        reg=0.1,
        n_iter=50,
        scaling=0.5,
    )
    # First entry should be initial_eps = 0.5
    assert float(schedule[0]) == pytest.approx(0.5)
    # Geometric decay, monotone non-increasing across the full schedule.
    for i in range(1, len(schedule)):
        assert schedule[i] <= schedule[i - 1]
    # Last entry must be reg (max(..., reg) clamp guarantees this, modulo
    # float32 round-trip through the clamp's Python-float bound).
    assert float(schedule[-1]) == pytest.approx(0.1)


def test_schedule_capped_by_n_iter():
    # Very small scaling factor: schedule would naturally go on forever;
    # n_iter must cap it.
    schedule = build_eps_schedule(
        cost=torch.full((3, 4), 1.0),
        reg=1e-8,
        n_iter=10,
        scaling=0.99,
    )
    assert len(schedule) <= 10


def test_schedule_ignores_inf_when_computing_initial():
    # +inf entries in cost (forbidden edges) must not blow up the initial epsilon.
    cost = torch.tensor([[1.0, float("inf")], [2.0, 3.0]])
    schedule = build_eps_schedule(
        cost=cost,
        reg=0.1,
        n_iter=20,
        scaling=0.5,
    )
    # initial_eps = max(0.5 * 3.0, 0.1) = 1.5
    assert float(schedule[0]) == pytest.approx(1.5)


def test_schedule_rejects_scaling_at_or_above_1():
    with pytest.raises(ValueError, match=r"scaling must be in \(0, 1\)"):
        build_eps_schedule(
            cost=torch.zeros(3, 4),
            reg=0.1,
            n_iter=20,
            scaling=1.0,
        )


def test_schedule_rejects_non_positive_reg():
    with pytest.raises(ValueError, match="reg must be positive"):
        build_eps_schedule(
            cost=torch.zeros(3, 4),
            reg=0.0,
            n_iter=20,
            scaling=None,
        )


def test_schedule_handles_an_all_non_finite_cost():
    # No finite entry at all: the max collapses to -inf, which the clamp
    # must fold to reg rather than propagating.
    cost = torch.full((2, 2), float("inf"))
    schedule = build_eps_schedule(cost=cost, reg=0.1, n_iter=5, scaling=0.5)
    assert torch.equal(schedule, torch.full((5,), 0.1))


def test_schedule_handles_an_empty_cost():
    cost = torch.empty(0, 3, 4)
    schedule = build_eps_schedule(cost=cost, reg=0.1, n_iter=5, scaling=0.5)
    assert torch.equal(schedule, torch.full((5,), 0.1))


def test_schedule_is_detached_from_cost():
    # eps is a hyperparameter of the iteration, not something a caller
    # differentiates through; a replayed backward must not leak a gradient
    # into cost via the schedule.
    cost = torch.full((3, 4), 1.0, requires_grad=True)
    schedule = build_eps_schedule(cost=cost, reg=0.1, n_iter=5, scaling=0.5)
    assert not schedule.requires_grad


def test_schedule_matches_cost_device_and_dtype():
    cost = torch.rand(3, 4, dtype=torch.float64)
    schedule = build_eps_schedule(cost=cost, reg=0.1, n_iter=5, scaling=0.5)
    assert schedule.dtype == cost.dtype
    assert schedule.device == cost.device
