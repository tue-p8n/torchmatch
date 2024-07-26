"""Uniform `+inf` mask + NaN / -inf rejection across every matching op."""

from __future__ import annotations

import pytest
import torch
import torchmatch  # noqa: F401  (eager extension load)

_CUDA = torch.cuda.is_available()

# (op_name, requires_cuda, ndim_in, square_only)
CPU_2D_OPS = [
    ("jonker_scalar", False, 2, False),
    ("jonker_dense", False, 2, False),
    ("jonker_compact", False, 2, True),
]
CPU_3D_OPS = [
    ("jonker_dense_batch", False, 3, True),
    ("jonker_compact_batch", False, 3, True),
]
CPU_3D_UNPACKED_OPS = [
    ("jonker_dense_batch_unpacked", False, 3, True),
    ("jonker_compact_batch_unpacked", False, 3, True),
]
CUDA_2D_OPS = [
    ("munkres", True, 2, False),
    ("lawler", True, 2, False),
    ("hybrid", True, 2, False),
]
CUDA_3D_OPS = [
    ("jonker_dense_batch", True, 3, True),
]

ALL_2D_OPS = CPU_2D_OPS + CUDA_2D_OPS
ALL_3D_PACKED_OPS = CPU_3D_OPS + CUDA_3D_OPS


def _op(name: str):
    return getattr(torch.ops.assignment, name)


def _to_device(c: torch.Tensor, requires_cuda: bool) -> torch.Tensor:
    return c.cuda() if requires_cuda else c


@pytest.mark.parametrize(("op_name", "requires_cuda", "_ndim", "_sq"), ALL_2D_OPS)
def test_2d_plus_inf_forbids_edge(op_name, requires_cuda, _ndim, _sq):
    if requires_cuda and not _CUDA:
        pytest.skip("CUDA required")
    cost = torch.tensor([[1.0, float("inf")], [float("inf"), 1.0]])
    cost = _to_device(cost, requires_cuda)
    row_to_col = _op(op_name)(cost)
    assert row_to_col.cpu().tolist() == [0, 1], op_name


@pytest.mark.parametrize(
    ("op_name", "requires_cuda", "_ndim", "_sq"), ALL_3D_PACKED_OPS
)
def test_3d_plus_inf_forbids_edge(op_name, requires_cuda, _ndim, _sq):
    if requires_cuda and not _CUDA:
        pytest.skip("CUDA required")
    cost = torch.tensor(
        [
            [[1.0, float("inf")], [float("inf"), 1.0]],
            [[2.0, float("inf")], [float("inf"), 3.0]],
        ]
    )
    cost = _to_device(cost, requires_cuda)
    out = _op(op_name)(cost)
    assert out.cpu().tolist() == [[0, 1], [0, 1]], op_name


@pytest.mark.parametrize(("op_name", "_rc", "_ndim", "_sq"), CPU_3D_UNPACKED_OPS)
def test_3d_unpacked_plus_inf_forbids_edge(op_name, _rc, _ndim, _sq):
    cost = torch.tensor(
        [
            [[1.0, float("inf")], [float("inf"), 1.0]],
            [[2.0, float("inf")], [float("inf"), 3.0]],
        ]
    )
    _matches, _ur, _uc, n_matched = _op(op_name)(cost)
    assert n_matched.tolist() == [2, 2], op_name


@pytest.mark.parametrize(("op_name", "requires_cuda", "_ndim", "_sq"), ALL_2D_OPS)
def test_2d_nan_rejected_at_op(op_name, requires_cuda, _ndim, _sq):
    if requires_cuda and not _CUDA:
        pytest.skip("CUDA required")
    cost = torch.tensor([[1.0, float("nan")], [2.0, 3.0]])
    cost = _to_device(cost, requires_cuda)
    with pytest.raises(RuntimeError, match="NaN"):
        _op(op_name)(cost)


@pytest.mark.parametrize(("op_name", "requires_cuda", "_ndim", "_sq"), ALL_2D_OPS)
def test_2d_neg_inf_rejected_at_op(op_name, requires_cuda, _ndim, _sq):
    if requires_cuda and not _CUDA:
        pytest.skip("CUDA required")
    cost = torch.tensor([[1.0, float("-inf")], [2.0, 3.0]])
    cost = _to_device(cost, requires_cuda)
    with pytest.raises(RuntimeError, match="-inf"):
        _op(op_name)(cost)


@pytest.mark.parametrize(
    ("op_name", "requires_cuda", "_ndim", "_sq"), ALL_3D_PACKED_OPS
)
def test_3d_nan_rejected_at_op(op_name, requires_cuda, _ndim, _sq):
    if requires_cuda and not _CUDA:
        pytest.skip("CUDA required")
    cost = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, float("nan")], [3.0, 4.0]]])
    cost = _to_device(cost, requires_cuda)
    with pytest.raises(RuntimeError, match="NaN"):
        _op(op_name)(cost)


@pytest.mark.parametrize(
    ("op_name", "requires_cuda", "_ndim", "_sq"), ALL_3D_PACKED_OPS
)
def test_3d_neg_inf_rejected_at_op(op_name, requires_cuda, _ndim, _sq):
    if requires_cuda and not _CUDA:
        pytest.skip("CUDA required")
    cost = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, float("-inf")], [3.0, 4.0]]])
    cost = _to_device(cost, requires_cuda)
    with pytest.raises(RuntimeError, match="-inf"):
        _op(op_name)(cost)


@pytest.mark.parametrize(("op_name", "_rc", "_ndim", "_sq"), CPU_3D_UNPACKED_OPS)
def test_3d_unpacked_nan_rejected_at_op(op_name, _rc, _ndim, _sq):
    cost = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, float("nan")], [3.0, 4.0]]])
    with pytest.raises(RuntimeError, match="NaN"):
        _op(op_name)(cost)


@pytest.mark.parametrize(("op_name", "_rc", "_ndim", "_sq"), CPU_3D_UNPACKED_OPS)
def test_3d_unpacked_neg_inf_rejected_at_op(op_name, _rc, _ndim, _sq):
    cost = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[1.0, float("-inf")], [3.0, 4.0]]])
    with pytest.raises(RuntimeError, match="-inf"):
        _op(op_name)(cost)
