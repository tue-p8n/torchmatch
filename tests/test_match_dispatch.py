"""Tests for the torchmatch.assignment.solve dispatcher."""

from __future__ import annotations

import types as _types

import pytest
import torch
import torchmatch
from torchmatch.assignment import Backend, solve

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_backend_enum_members():
    assert Backend.AUTO == "auto"
    assert Backend.JONKER == "jonker"
    assert Backend.MUNKRES == "munkres"
    assert Backend.LAWLER == "lawler"


def test_backend_string_coercion_unknown_raises():
    with pytest.raises(
        ValueError, match=r"torchmatch\.assignment\.solve: unknown backend"
    ):
        solve(torch.zeros(2, 2), backend="not_a_backend")


def test_solve_callable():
    assert callable(solve)
    # The package re-exports the assignment sub-package.
    assert torchmatch.assignment.solve is solve


@pytest.fixture
def record_ops(monkeypatch):
    """Patch torchmatch.assignment._solve._OPS with stubs that record their calls."""
    from torchmatch.assignment import _solve as solve_mod

    calls: list[tuple[str, tuple]] = []

    def _stub(name):
        def _fn(*args):
            calls.append((name, args))
            cost = args[0]
            if cost.ndim == 2:
                return torch.zeros(cost.size(0), dtype=torch.long, device=cost.device)
            return torch.zeros(
                cost.size(0), cost.size(1), dtype=torch.long, device=cost.device
            )

        return _fn

    fake = _types.SimpleNamespace(
        **{
            name: _stub(name)
            for name in (
                "jonker_scalar",
                "jonker_dense",
                "jonker_compact",
                "jonker_dense_batch",
                "jonker_compact_batch",
                "jonker_dense_batch_unpacked",
                "jonker_compact_batch_unpacked",
                "munkres",
                "lawler",
            )
        }
    )
    monkeypatch.setattr(solve_mod, "_OPS", fake)
    return calls


def test_validate_rejects_ndim():
    with pytest.raises(ValueError, match=r"cost\.ndim must be 2 or 3"):
        solve(torch.zeros(4), backend="jonker")
    with pytest.raises(ValueError, match=r"cost\.ndim must be 2 or 3"):
        solve(torch.zeros(2, 2, 2, 2), backend="jonker")


def test_validate_rejects_dtype():
    with pytest.raises(ValueError, match=r"cost\.dtype must be float32 or float64"):
        solve(torch.zeros(2, 2, dtype=torch.int32), backend="jonker")


def test_validate_rejects_nan():
    bad = torch.zeros(2, 2)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="cost contains NaN"):
        solve(bad, backend="jonker")


def test_validate_rejects_neg_inf():
    bad = torch.zeros(2, 2)
    bad[0, 0] = float("-inf")
    with pytest.raises(RuntimeError, match="cost contains -inf"):
        solve(bad, backend="jonker")


def test_validate_accepts_pos_inf():
    cost = torch.zeros(2, 2)
    cost[0, 0] = float("inf")
    # Validation must pass; dispatch downstream returns a tensor.
    out = solve(cost, backend="jonker")
    assert out.shape == (2,)


def test_route_cpu_jonker_2d_square_uses_compact(record_ops):
    solve(torch.zeros(16, 16), backend="jonker")
    assert record_ops[-1][0] == "jonker_compact"


def test_route_cpu_jonker_2d_rectangular_uses_dense(record_ops):
    solve(torch.zeros(8, 16), backend="jonker")
    assert record_ops[-1][0] == "jonker_dense"


def test_route_cpu_jonker_2d_tiny_uses_scalar(record_ops):
    solve(torch.zeros(3, 3), backend="jonker")
    assert record_ops[-1][0] == "jonker_scalar"


def test_route_cpu_jonker_3d_square_uses_compact_batch(record_ops):
    solve(torch.zeros(4, 8, 8), backend="jonker")
    assert record_ops[-1][0] == "jonker_compact_batch"


def test_route_cpu_jonker_3d_rectangular_uses_dense_batch(record_ops):
    solve(torch.zeros(4, 8, 12), backend="jonker")
    assert record_ops[-1][0] == "jonker_dense_batch"


@pytest.mark.parametrize("n", [8, 32, 64])
def test_cpu_jonker_2d_optimal(ref_cost, n):
    torch.manual_seed(n)
    cost = torch.rand(n, n, dtype=torch.float64)
    row_to_col = solve(cost, backend="jonker")
    matched = row_to_col >= 0
    rows = torch.arange(n)
    got = float(cost[rows[matched], row_to_col[matched]].sum())
    ref = ref_cost(cost)
    assert abs(got - ref) < 1e-9


def test_route_munkres_cpu_errors():
    with pytest.raises(ValueError, match="munkres is CUDA-only"):
        solve(torch.zeros(4, 4), backend="munkres")


def test_route_lawler_cpu_errors():
    with pytest.raises(ValueError, match="lawler is CUDA-only"):
        solve(torch.zeros(4, 4), backend="lawler")


@cuda_only
def test_route_munkres_2d_calls_munkres(record_ops):
    solve(torch.zeros(8, 8, device="cuda"), backend="munkres")
    assert record_ops[-1][0] == "munkres"


@cuda_only
def test_route_lawler_2d_calls_lawler(record_ops):
    solve(torch.zeros(8, 8, device="cuda"), backend="lawler")
    assert record_ops[-1][0] == "lawler"


@cuda_only
def test_route_munkres_3d_loops(record_ops):
    solve(torch.zeros(3, 4, 4, device="cuda"), backend="munkres")
    names = [c[0] for c in record_ops]
    assert names.count("munkres") == 3


@cuda_only
@pytest.mark.parametrize("n", [8, 32])
def test_cuda_munkres_2d_optimal(ref_cost, n):
    torch.manual_seed(n)
    cost = torch.rand(n, n, dtype=torch.float32, device="cuda")
    row_to_col = solve(cost, backend="munkres")
    matched = row_to_col >= 0
    rows = torch.arange(n, device="cuda")
    got = float(cost[rows[matched], row_to_col[matched]].sum().cpu())
    ref = ref_cost(cost.cpu().double())
    assert abs(got - ref) < 1e-4


@cuda_only
def test_route_jonker_2d_cuda_errors():
    with pytest.raises(ValueError, match="no CUDA single-instance JV"):
        solve(torch.zeros(8, 8, device="cuda"), backend="jonker")


@cuda_only
def test_route_jonker_3d_cuda_rectangular_errors():
    with pytest.raises(ValueError, match="rectangular"):
        solve(torch.zeros(2, 8, 12, device="cuda"), backend="jonker")


@cuda_only
def test_route_jonker_3d_cuda_k_too_large_errors():
    with pytest.raises(ValueError, match="K <= 64"):
        solve(torch.zeros(2, 80, 80, device="cuda"), backend="jonker")


@cuda_only
def test_route_jonker_3d_cuda_square_uses_dense_batch(record_ops):
    solve(torch.zeros(2, 8, 8, device="cuda"), backend="jonker")
    assert record_ops[-1][0] == "jonker_dense_batch"


@cuda_only
@pytest.mark.parametrize("k", [4, 16, 64])
def test_cuda_jonker_3d_optimal(ref_cost, k):
    torch.manual_seed(k)
    costs = torch.rand(3, k, k, dtype=torch.float32, device="cuda")
    out = solve(costs, backend="jonker")
    for b in range(3):
        cb = costs[b]
        ob = out[b]
        matched = ob >= 0
        rows = torch.arange(k, device="cuda")
        got = float(cb[rows[matched], ob[matched]].sum().cpu())
        ref = ref_cost(cb.cpu().double())
        assert abs(got - ref) < 1e-3


def test_auto_cpu_2d_uses_jonker(record_ops):
    solve(torch.zeros(16, 16))
    assert record_ops[-1][0] in ("jonker_dense", "jonker_compact", "jonker_scalar")


def test_auto_cpu_3d_square_uses_compact_batch(record_ops):
    solve(torch.zeros(2, 8, 8))
    assert record_ops[-1][0] == "jonker_compact_batch"


def test_auto_cpu_3d_rect_uses_dense_batch(record_ops):
    solve(torch.zeros(2, 8, 12))
    assert record_ops[-1][0] == "jonker_dense_batch"


@cuda_only
def test_auto_cuda_2d_small_uses_munkres(record_ops):
    solve(torch.zeros(8, 8, device="cuda"))
    assert record_ops[-1][0] == "munkres"


@cuda_only
def test_auto_cuda_2d_large_uses_lawler(record_ops):
    solve(torch.zeros(128, 128, device="cuda"))
    assert record_ops[-1][0] == "lawler"


@cuda_only
def test_auto_cuda_3d_square_small_uses_jonker_dense_batch(record_ops):
    solve(torch.zeros(2, 16, 16, device="cuda"))
    assert record_ops[-1][0] == "jonker_dense_batch"


@cuda_only
def test_auto_cuda_3d_rect_falls_back_to_loop(record_ops):
    solve(torch.zeros(3, 8, 12, device="cuda"))
    names = [c[0] for c in record_ops]
    # AUTO picks munkres (N=8 < threshold); three problems -> three calls.
    assert names.count("munkres") == 3


def test_unpack_2d_tight():
    cost = torch.tensor([[1.0, 2.0, 3.0], [4.0, 0.5, 6.0]], dtype=torch.float64)
    out = solve(cost, backend="jonker", unpack=True)
    matches, ur, uc = out
    # Two rows, three cols: K = min(N, M) = 2 matched, 0 unmatched rows,
    # 1 unmatched col.
    assert matches.shape == (2, 2)
    assert ur.shape == (0,)
    assert uc.shape == (1,)


def test_unpack_3d_native_cpu_jonker_matches_packed():
    torch.manual_seed(0)
    costs = torch.rand(3, 6, 6, dtype=torch.float64)
    packed = solve(costs, backend="jonker")
    matches, _ur, _uc, n_matched = solve(costs, backend="jonker", unpack=True)
    for b in range(3):
        k = int(n_matched[b])
        rows = matches[b, :k, 0]
        cols = matches[b, :k, 1]
        assert torch.equal(packed[b, rows], cols)


def test_unpack_3d_derived_matches_native():
    torch.manual_seed(1)
    costs = torch.rand(2, 5, 7, dtype=torch.float64)
    native = solve(costs, backend="jonker", unpack=True)
    from torchmatch.assignment import _solve as solve_mod

    solve_mod._ALLOW_NATIVE_UNPACK = False
    try:
        derived = solve(costs, backend="jonker", unpack=True)
    finally:
        solve_mod._ALLOW_NATIVE_UNPACK = True
    assert torch.equal(native[3], derived[3])  # n_matched
    for b in range(costs.size(0)):
        k = int(native[3][b])
        # Compare sorted (matched_row, matched_col) pairs.
        nat_pairs = native[0][b, :k].sort(dim=0).values
        der_pairs = derived[0][b, :k].sort(dim=0).values
        assert torch.equal(nat_pairs, der_pairs)


def test_auto_determinism(record_ops):
    cost = torch.zeros(8, 8)
    solve(cost)
    first = record_ops[-1][0]
    solve(cost)
    second = record_ops[-1][0]
    assert first == second


@pytest.mark.parametrize("shape", [(2, 8, 8), (2, 8, 12)])
def test_cpu_jonker_3d_optimal(ref_cost, shape):
    torch.manual_seed(0)
    costs = torch.rand(*shape, dtype=torch.float64)
    out = solve(costs, backend="jonker")
    assert out.shape == (shape[0], shape[1])
    for b in range(shape[0]):
        cb = costs[b]
        ob = out[b]
        matched = ob >= 0
        rows = torch.arange(shape[1])
        got = float(cb[rows[matched], ob[matched]].sum())
        ref = ref_cost(cb)
        assert abs(got - ref) < 1e-9
