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
    assert len(schedule) == 50
    assert all(eps == 0.1 for eps in schedule)


def test_geometric_decay_when_scaling_set():
    # max(cost) is 1.0; initial_eps = max(0.5 * 1.0, 0.1) = 0.5
    # schedule[k] = 0.5 * 0.5**k until we reach 0.1, then append 0.1 and stop.
    cost = torch.full((3, 4), 1.0)
    schedule = build_eps_schedule(
        cost=cost,
        reg=0.1,
        n_iter=50,
        scaling=0.5,
    )
    # First entry should be initial_eps = 0.5
    assert schedule[0] == pytest.approx(0.5)
    # Geometric decay, monotone non-increasing across the full schedule.
    for i in range(1, len(schedule)):
        assert schedule[i] <= schedule[i - 1]
    # Last entry must be exactly reg (max(..., reg) clamp guarantees this).
    assert schedule[-1] == 0.1


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
    assert schedule[0] == pytest.approx(1.5)


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
