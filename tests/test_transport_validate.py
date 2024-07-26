"""Tests for transport.matrix input validation and mask/marginal coercion."""

from __future__ import annotations

import pytest
import torch
from torchmatch.transport.matrix._validate import (
    coerce_marginals,
    fuse_mask_into_cost,
    validate_cost,
)


def test_validate_cost_accepts_2d():
    validate_cost(torch.zeros(3, 4))


def test_validate_cost_accepts_3d():
    validate_cost(torch.zeros(2, 3, 4))


def test_validate_cost_rejects_1d():
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        validate_cost(torch.zeros(5))


def test_validate_cost_rejects_4d():
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        validate_cost(torch.zeros(2, 3, 4, 5))


def test_validate_cost_rejects_nan():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="cost contains NaN"):
        validate_cost(bad)


def test_validate_cost_rejects_neg_inf():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("-inf")
    with pytest.raises(RuntimeError, match="cost contains -inf"):
        validate_cost(bad)


def test_validate_cost_accepts_pos_inf():
    ok = torch.zeros(3, 3)
    ok[0, 0] = float("inf")
    validate_cost(ok)  # must not raise


def test_validate_cost_rejects_int_dtype():
    with pytest.raises(ValueError, match="float32 or float64"):
        validate_cost(torch.zeros(3, 3, dtype=torch.int64))


def test_coerce_marginals_defaults_uniform_2d():
    cost = torch.zeros(3, 4)
    a, b = coerce_marginals(cost, None, None)
    assert a.shape == (1, 3)
    assert b.shape == (1, 4)
    assert torch.allclose(a, torch.full_like(a, 1.0 / 3))
    assert torch.allclose(b, torch.full_like(b, 1.0 / 4))


def test_coerce_marginals_defaults_uniform_3d():
    cost = torch.zeros(2, 3, 4)
    a, b = coerce_marginals(cost, None, None)
    assert a.shape == (2, 3)
    assert b.shape == (2, 4)


def test_coerce_marginals_accepts_user_2d():
    cost = torch.zeros(3, 4)
    a_in = torch.tensor([0.1, 0.4, 0.5])
    b_in = torch.tensor([0.25, 0.25, 0.25, 0.25])
    a, b = coerce_marginals(cost, a_in, b_in)
    assert torch.equal(a, a_in.unsqueeze(0))
    assert torch.equal(b, b_in.unsqueeze(0))


def test_coerce_marginals_rejects_negative():
    cost = torch.zeros(3, 4)
    a_bad = torch.tensor([-0.1, 0.4, 0.7])
    with pytest.raises(ValueError, match="non-negative"):
        coerce_marginals(cost, a_bad, None)


def test_fuse_mask_replaces_forbidden_with_inf():
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, True]])
    out = fuse_mask_into_cost(cost, mask)
    assert out[0, 0] == 1.0
    assert out[0, 1] == float("inf")
    assert out[1, 0] == 3.0
    assert out[1, 1] == 4.0


def test_fuse_mask_none_returns_unchanged():
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert fuse_mask_into_cost(cost, None) is cost


def test_fuse_mask_broadcasts():
    # 3D cost with 2D mask broadcasts.
    cost = torch.ones(2, 3, 3)
    mask = torch.tensor([[True, False, True], [True, True, True], [True, True, True]])
    out = fuse_mask_into_cost(cost, mask)
    assert out.shape == (2, 3, 3)
    assert torch.isinf(out[:, 0, 1]).all()
