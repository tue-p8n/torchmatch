"""Smoke tests for the transport scaffolding namespace."""

from __future__ import annotations

import pytest
import torch
import torchmatch
from torchmatch.transport.matrix import Backend, solve


def test_transport_subpackage_imports():
    assert hasattr(torchmatch, "transport")
    assert hasattr(torchmatch.transport.matrix, "solve")
    assert hasattr(torchmatch.transport.matrix, "Backend")
    assert hasattr(torchmatch.transport.matrix, "ops")


def test_torch_ops_transport_exists():
    # The namespace is registered even when empty (empty fragment).
    assert hasattr(torch.ops, "transport")


def test_transport_ops_module_exposes_implemented_ops():
    from torchmatch.transport.matrix import ops

    assert ops.__all__ == [
        "exact_emd",
        "log_sinkhorn",
        "sinkhorn_divergence",
        "unbalanced_sinkhorn",
    ]


def test_solve_accepts_string_backend():
    cost = torch.rand(2, 3, 3)
    solve(cost, backend="log_sinkhorn")


def test_solve_rejects_unknown_backend_with_prefixed_message():
    cost = torch.rand(2, 3, 3)
    with pytest.raises(
        ValueError, match=r"torchmatch\.transport\.matrix\.solve: unknown backend"
    ):
        solve(cost, backend="not_a_backend")


def test_solve_rejects_nan_input():
    cost = torch.rand(2, 3, 3)
    cost[0, 0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        solve(cost, backend=Backend.LOG_SINKHORN)


def test_solve_rejects_neg_inf_input():
    cost = torch.rand(2, 3, 3)
    cost[1, 1, 2] = float("-inf")
    with pytest.raises(RuntimeError, match="-inf"):
        solve(cost, backend=Backend.LOG_SINKHORN)


def test_solve_rejects_invalid_ndim():
    cost = torch.rand(4)
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        solve(cost, backend=Backend.LOG_SINKHORN)


def test_solve_rejects_invalid_dtype():
    cost = torch.randint(0, 10, (3, 3), dtype=torch.int64)
    with pytest.raises(ValueError, match="float32 or float64"):
        solve(cost, backend=Backend.LOG_SINKHORN)
