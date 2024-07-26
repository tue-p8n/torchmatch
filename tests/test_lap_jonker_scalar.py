"""Correctness tests for ``assignment::jonker_scalar``.

The tests exercise the full surface: square, tall, wide, and empty
inputs, NaN and -inf rejection, +inf-blocked edges.
"""

from __future__ import annotations

import pytest
import torch


def _solve_cost(c: torch.Tensor) -> float:
    row_to_col = torch.ops.assignment.jonker_scalar(c)
    rows = torch.arange(c.size(0))
    matched = row_to_col >= 0
    return float(c[rows[matched], row_to_col[matched]].sum())


@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("seed", range(10))
def test_square_optimal(ref_cost, n, seed):
    torch.manual_seed(seed)
    cost = torch.rand(n, n, dtype=torch.float64)
    got = _solve_cost(cost)
    ref = ref_cost(cost)
    assert abs(got - ref) < 1e-9, f"n={n} seed={seed}: {got} vs {ref}"


@pytest.mark.parametrize(("nr", "nc"), [(4, 8), (8, 4), (1, 16), (16, 1), (5, 7)])
def test_rectangular_optimal(ref_cost, nr, nc):
    torch.manual_seed(0)
    cost = torch.rand(nr, nc, dtype=torch.float64)
    got = _solve_cost(cost)
    ref = ref_cost(cost)
    assert abs(got - ref) < 1e-9

    row_to_col = torch.ops.assignment.jonker_scalar(cost)
    assert row_to_col.shape == (nr,)
    # Exactly min(nr, nc) rows should be matched.
    assert int((row_to_col >= 0).sum()) == min(nr, nc)


def test_empty_inputs():
    assert torch.ops.assignment.jonker_scalar(torch.empty(0, 0)).numel() == 0
    out = torch.ops.assignment.jonker_scalar(torch.empty(3, 0))
    assert out.tolist() == [-1, -1, -1]


def test_float32():
    torch.manual_seed(0)
    c32 = torch.rand(64, 64, dtype=torch.float32)
    row_to_col = torch.ops.assignment.jonker_scalar(c32)
    assert row_to_col.dtype == torch.long
    # Cost should be near-optimal under f32 → f64 lifting.
    assert int((row_to_col >= 0).sum()) == 64


def test_plus_inf_is_infeasible_edge():
    # Both diagonals blocked; the only feasible assignment is the identity.
    cost = torch.tensor([[1.0, float("inf")], [float("inf"), 1.0]])
    row_to_col = torch.ops.assignment.jonker_scalar(cost)
    assert row_to_col.tolist() == [0, 1]


def test_nan_rejected():
    cost = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])
    with pytest.raises(RuntimeError, match="NaN"):
        torch.ops.assignment.jonker_scalar(cost)


def test_neg_inf_rejected():
    cost = torch.tensor([[1.0, float("-inf")], [2.0, 3.0]])
    with pytest.raises(RuntimeError, match="-inf"):
        torch.ops.assignment.jonker_scalar(cost)
