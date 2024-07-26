"""Correctness tests for torchmatch.assignment.ops.greedy."""

from __future__ import annotations

import pytest
import torch
from torchmatch.assignment import Backend, solve
from torchmatch.assignment.ops import greedy


def test_greedy_2x2_unambiguous():
    # Min is (0,0)=1.0; assign 0->0, then only (1,1)=4.0 is left.
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    result = greedy(cost)
    assert torch.equal(result, torch.tensor([0, 1], dtype=torch.int64))


def test_greedy_3x3_unambiguous():
    # Min is (1,0)=0.1; assign 1->0. Next min in remaining
    # rows/cols is (0,2)=0.2; assign 0->2. Then 2->1.
    cost = torch.tensor(
        [
            [9.0, 5.0, 0.2],
            [0.1, 9.0, 9.0],
            [9.0, 0.3, 9.0],
        ]
    )
    result = greedy(cost)
    assert torch.equal(result, torch.tensor([2, 0, 1], dtype=torch.int64))


def test_greedy_rectangular_more_cols_than_rows():
    cost = torch.tensor([[1.0, 0.5, 9.0, 9.0], [9.0, 9.0, 2.0, 0.3]])
    result = greedy(cost)
    # (1,3)=0.3 wins first, then (0,1)=0.5.
    assert torch.equal(result, torch.tensor([1, 3], dtype=torch.int64))


def test_greedy_rectangular_more_rows_than_cols():
    cost = torch.tensor(
        [
            [1.0, 9.0],
            [9.0, 2.0],
            [3.0, 4.0],
        ]
    )
    result = greedy(cost)
    # First (0,0)=1.0, then (1,1)=2.0; row 2 has no column left.
    assert torch.equal(result, torch.tensor([0, 1, -1], dtype=torch.int64))


def test_greedy_returns_valid_partial_permutation():
    torch.manual_seed(0)
    cost = torch.rand(10, 10)
    result = greedy(cost)
    matched = result[result >= 0]
    # No column reused.
    assert len(set(matched.tolist())) == len(matched)
    # Every row matched (square case).
    assert (result >= 0).all()


def test_greedy_heuristic_at_least_as_bad_as_jv():
    torch.manual_seed(0)
    cost = torch.rand(20, 20)
    g = greedy(cost)
    o = solve(cost, backend=Backend.JONKER)
    g_total = cost[torch.arange(20), g].sum()
    o_total = cost[torch.arange(20), o].sum()
    assert g_total >= o_total  # greedy is never better than optimal


@pytest.mark.parametrize("shape", [(5, 5), (5, 7), (3, 5, 5), (3, 5, 7)])
def test_greedy_faketensor(shape):
    cost = torch.empty(*shape, dtype=torch.float32, device="meta")
    torch.library.opcheck(
        torch.ops.assignment.greedy,
        (cost,),
        test_utils=("test_schema", "test_faketensor"),
    )


def test_greedy_dispatches_via_backend_enum():
    torch.manual_seed(0)
    cost = torch.rand(8, 8)
    result_direct = greedy(cost)
    result_dispatch = solve(cost, backend=Backend.GREEDY)
    assert torch.equal(result_direct, result_dispatch)


def test_greedy_rejects_nan():
    cost = torch.tensor([[float("nan"), 1.0], [1.0, 1.0]])
    with pytest.raises(RuntimeError, match="NaN"):
        solve(cost, backend=Backend.GREEDY)


def test_greedy_op_rejects_nan_direct():
    cost = torch.tensor([[float("nan"), 1.0], [1.0, 1.0]])
    with pytest.raises(RuntimeError, match="NaN"):
        greedy(cost)


def test_greedy_op_rejects_neg_inf_direct():
    cost = torch.tensor([[float("-inf"), 1.0], [1.0, 1.0]])
    with pytest.raises(RuntimeError, match="-inf"):
        greedy(cost)


@pytest.mark.parametrize("bad_ndim_shape", [(4,), (2, 3, 3, 3)])
def test_greedy_rejects_invalid_ndim(bad_ndim_shape):
    cost = torch.rand(*bad_ndim_shape)
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        greedy(cost)


def test_greedy_3d_batch():
    # Two independent 2x2 problems; verify they decouple.
    cost = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[9.0, 0.1], [0.2, 9.0]],
        ]
    )
    result = greedy(cost)
    expected = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    assert torch.equal(result, expected)


def test_greedy_3d_dispatches_via_backend_enum():
    torch.manual_seed(0)
    cost = torch.rand(4, 5, 5)
    result_direct = greedy(cost)
    result_dispatch = solve(cost, backend=Backend.GREEDY)
    assert torch.equal(result_direct, result_dispatch)


@pytest.mark.parametrize("shape", [(0, 0), (3, 0), (0, 4)])
def test_greedy_empty_shape_returns_all_minus_one(shape):
    cost = torch.empty(*shape, dtype=torch.float32)
    result = greedy(cost)
    assert result.shape == (shape[0],)
    assert result.dtype == torch.int64
    assert (result == -1).all()


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4), (2, 3, 0)])
def test_greedy_3d_empty_shape_returns_all_minus_one(shape):
    cost = torch.empty(*shape, dtype=torch.float32)
    result = greedy(cost)
    assert result.shape == (shape[0], shape[1])
    assert result.dtype == torch.int64
    assert (result == -1).all()


def test_greedy_all_inf_row_returns_minus_one():
    # Row 0 is fully forbidden; rows 1 and 2 take their min cell each
    # in sequence, then the loop must break on the +inf-only residual
    # without misassigning row 0 to a masked column.
    cost = torch.tensor(
        [
            [float("inf"), float("inf"), float("inf")],
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
        ]
    )
    result = greedy(cost)
    assert result[0].item() == -1
    matched = result[result >= 0]
    assert len(set(matched.tolist())) == len(matched)
    assert result[1].item() == 0
    assert result[2].item() == 1
