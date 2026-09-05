"""
Unified dispatcher over the LAP ops registered under ``torch.ops.assignment``.

``torchmatch.assignment.solve`` is the recommended entrypoint. It
validates the input, picks a backend, and returns a tensor whose rank
is ``cost.ndim - 1`` (or a tuple if ``unpack=True``). Individual ops
remain exported at ``torchmatch.assignment.ops.<op_name>`` for callers
who want explicit control or are benchmarking.

See ``docs/superpowers/specs/2026-05-17-match-dispatcher-design.md`` for
the full routing matrix and design rationale.
"""

from __future__ import annotations

import types
from enum import StrEnum
from typing import Literal, overload

import torch

from torchmatch.assignment._validate import check_finite

__all__ = ["Backend", "resolve_backend", "solve"]


class Backend(StrEnum):
    AUTO = "auto"
    JONKER = "jonker"
    MUNKRES = "munkres"
    LAWLER = "lawler"
    GREEDY = "greedy"


# Indirection so tests can substitute recording stubs without poking
# torch.ops.assignment, which is harder to monkeypatch cleanly.
_OPS = types.SimpleNamespace(
    jonker_scalar=torch.ops.assignment.jonker_scalar,
    jonker_dense=torch.ops.assignment.jonker_dense,
    jonker_compact=torch.ops.assignment.jonker_compact,
    jonker_dense_batch=torch.ops.assignment.jonker_dense_batch,
    jonker_compact_batch=torch.ops.assignment.jonker_compact_batch,
    jonker_dense_batch_unpacked=torch.ops.assignment.jonker_dense_batch_unpacked,
    jonker_compact_batch_unpacked=torch.ops.assignment.jonker_compact_batch_unpacked,
    munkres=getattr(torch.ops.assignment, "munkres", None),
    lawler=getattr(torch.ops.assignment, "lawler", None),
    greedy=torch.ops.assignment.greedy,
)


def _coerce_backend(value: Backend | str) -> Backend:
    try:
        return Backend(value)
    except ValueError as err:
        msg = (
            f"torchmatch.assignment.solve: unknown backend {value!r}; "
            f"valid values are {[b.value for b in Backend]}"
        )
        raise ValueError(msg) from err


# Thresholds: derived from tests/benchmark_single.py and
# tests/benchmark_batched.py. Revisit if dispatch tuning changes.
_JONKER_SCALAR_MAX_CELLS = 64  # below this, scalar wins on overhead
_CUDA_2D_LAWLER_N = 32  # at or above this, lawler outperforms munkres
_CUDA_BATCH_MAX_TILE = 64  # mirrors match_batch::MAX_TILE


def _pick_jonker_cpu_2d(n: int, m: int) -> str:
    if n * m <= _JONKER_SCALAR_MAX_CELLS:
        return "jonker_scalar"
    if n == m:
        return "jonker_compact"
    return "jonker_dense"


def _pick_jonker_cpu_3d(n: int, m: int) -> str:
    return "jonker_compact_batch" if n == m else "jonker_dense_batch"


def _pick_auto_cuda_2d(n: int) -> str:
    return "munkres" if n < _CUDA_2D_LAWLER_N else "lawler"


def _dispatch_auto(
    cost: torch.Tensor, *, unpack: bool
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    is_cuda = cost.device.type == "cuda"
    if cost.ndim == 2:
        if is_cuda:
            packed = _dispatch_primed(cost, _pick_auto_cuda_2d(cost.size(0)))
            return _unpack_packed(packed, cost.size(1)) if unpack else packed
        return _dispatch_jonker(cost, unpack=unpack)
    b, n, m = cost.shape
    if not is_cuda:
        return _dispatch_jonker(cost, unpack=unpack)
    if n == m and n <= _CUDA_BATCH_MAX_TILE:
        packed = _OPS.jonker_dense_batch(cost)
        return _unpack_packed(packed, m) if unpack else packed
    # CUDA 3D fallback: loop the 2D AUTO choice over the batch dim.
    op_name = _pick_auto_cuda_2d(n)
    op = getattr(_OPS, op_name)
    packed = torch.stack([op(cost[i]) for i in range(b)], dim=0)
    return _unpack_packed(packed, m) if unpack else packed


def _dispatch_primed(cost: torch.Tensor, op_name: str) -> torch.Tensor:
    if cost.device.type != "cuda":
        msg = f"torchmatch.assignment.solve: {op_name} is CUDA-only"
        raise ValueError(msg)
    op = getattr(_OPS, op_name)
    if op is None:
        msg = (
            f"torchmatch.assignment.solve: {op_name} is unavailable "
            "(CUDA extension not loaded)"
        )
        raise ValueError(msg)
    if cost.ndim == 2:
        return op(cost)
    return torch.stack([op(cost[b]) for b in range(cost.size(0))], dim=0)


_ALLOW_NATIVE_UNPACK = True


def _unpack_packed_3d(
    packed: torch.Tensor, m: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorised compaction of a packed ``(B, N)`` row→col tensor.

    Returns (matches, unmatched_rows, unmatched_cols, n_matched) with the
    same shape contract as ``jonker_*_batch_unpacked``: matched entries
    sit at ``[0, n_matched[b])`` in each output, padding past that is
    unspecified.
    """
    b, n = packed.shape
    device = packed.device

    matched_mask = packed >= 0
    n_matched = matched_mask.sum(dim=-1)

    # Stable argsort on ~matched_mask: False (matched) sorts ahead of
    # True (unmatched), so matched rows compact to the front.
    match_order = torch.argsort((~matched_mask).long(), dim=-1, stable=True)
    row_idx = torch.arange(n, device=device).expand(b, n)
    sorted_rows = torch.gather(row_idx, 1, match_order)
    sorted_cols = torch.gather(packed, 1, match_order)
    matches = torch.stack([sorted_rows, sorted_cols], dim=-1)  # (B, N, 2)

    # Unmatched rows: argsort matched_mask itself; False (unmatched)
    # sorts ahead. Compaction front-loads the unmatched row indices.
    unmatched_order = torch.argsort(matched_mask.long(), dim=-1, stable=True)
    unmatched_rows = torch.gather(row_idx, 1, unmatched_order)

    # Unmatched cols: mark matched col indices in a (B, m) bool, invert,
    # then apply the same front-loading compaction.
    col_mask = torch.zeros(b, m, dtype=torch.bool, device=device)
    col_idx_for_scatter = packed.clamp(min=0)
    col_mask.scatter_(1, col_idx_for_scatter, matched_mask)
    unmatched_col_mask = ~col_mask
    col_order = torch.argsort((~unmatched_col_mask).long(), dim=-1, stable=True)
    col_range = torch.arange(m, device=device).expand(b, m)
    unmatched_cols = torch.gather(col_range, 1, col_order)

    return matches, unmatched_rows, unmatched_cols, n_matched


def _unpack_packed(packed: torch.Tensor, m: int) -> tuple[torch.Tensor, ...]:
    """Derive the unpacked tuple from a packed row→col tensor."""
    if packed.ndim == 1:
        matches3, ur3, uc3, n_matched = _unpack_packed_3d(packed.unsqueeze(0), m)
        k = int(n_matched.item())
        n = packed.size(0)
        return matches3[0, :k], ur3[0, : n - k], uc3[0, : m - k]
    return _unpack_packed_3d(packed, m)


def _dispatch_jonker(
    cost: torch.Tensor, *, unpack: bool
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    if cost.device.type == "cuda":
        if cost.ndim == 2:
            msg = (
                "torchmatch.assignment.solve: no CUDA single-instance JV op; "
                "transfer to CPU or pick Backend.MUNKRES / Backend.LAWLER"
            )
            raise ValueError(msg)
        _b, n, m = cost.shape
        if n != m:
            msg = (
                "torchmatch.assignment.solve: no CUDA batched JV op for rectangular "
                f"inputs (got shape {tuple(cost.shape)}); transfer to CPU"
            )
            raise ValueError(msg)
        if n > _CUDA_BATCH_MAX_TILE:
            msg = (
                "torchmatch.assignment.solve: CUDA batched JV requires K <= 64, "
                f"got K={n}; transfer to CPU"
            )
            raise ValueError(msg)
        packed = _OPS.jonker_dense_batch(cost)
        return _unpack_packed(packed, m) if unpack else packed
    if cost.ndim == 2:
        op = getattr(_OPS, _pick_jonker_cpu_2d(cost.size(0), cost.size(1)))
        packed = op(cost)
        return _unpack_packed(packed, cost.size(1)) if unpack else packed
    op_name = _pick_jonker_cpu_3d(cost.size(1), cost.size(2))
    if unpack and _ALLOW_NATIVE_UNPACK:
        op = getattr(_OPS, f"{op_name}_unpacked")
        return op(cost)
    op = getattr(_OPS, op_name)
    packed = op(cost)
    return _unpack_packed(packed, cost.size(2)) if unpack else packed


def _validate(cost: torch.Tensor) -> None:
    if cost.ndim not in (2, 3):
        msg = f"torchmatch.assignment.solve: cost.ndim must be 2 or 3, got {cost.ndim}"
        raise ValueError(msg)
    if cost.dtype not in (torch.float32, torch.float64):
        msg = (
            "torchmatch.assignment.solve: cost.dtype must be float32 or float64, "
            f"got {cost.dtype}"
        )
        raise ValueError(msg)
    if cost.device.type not in ("cpu", "cuda"):
        msg = (
            "torchmatch.assignment.solve: cost.device must be cpu or cuda, "
            f"got {cost.device}"
        )
        raise ValueError(msg)
    # RuntimeError (not ValueError) mirrors what the C++ ops raise via
    # TORCH_CHECK on the same invariants, so a caller catching by type sees
    # one consistent surface. check_finite stands aside under tracing (see
    # its docstring); solve() is a plain function, so tracing it directly
    # (torch.compile, make_fx, a fake tensor reaching it outside either)
    # would otherwise hit this branch on a value that cannot be read.
    bad = check_finite(cost)
    if bad is not None:
        msg = f"torchmatch.assignment.solve: cost contains {bad}"
        raise RuntimeError(msg)


@overload
def solve(
    cost: torch.Tensor,
    *,
    backend: Backend | str = ...,
    unpack: Literal[False] = False,
) -> torch.Tensor: ...


@overload
def solve(
    cost: torch.Tensor,
    *,
    backend: Backend | str = ...,
    unpack: Literal[True],
) -> tuple[torch.Tensor, ...]: ...


def solve(
    cost: torch.Tensor,
    *,
    backend: Backend | str = Backend.AUTO,
    unpack: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    backend = _coerce_backend(backend)
    _validate(cost)
    if backend == Backend.AUTO:
        return _dispatch_auto(cost, unpack=unpack)
    if backend == Backend.JONKER:
        return _dispatch_jonker(cost, unpack=unpack)
    if backend in (Backend.MUNKRES, Backend.LAWLER):
        packed = _dispatch_primed(cost, backend.value)
        if unpack:
            return _unpack_packed(packed, cost.size(-1))
        return packed
    if backend == Backend.GREEDY:
        packed = _OPS.greedy(cost)
        if unpack:
            return _unpack_packed(packed, cost.size(-1))
        return packed
    msg = f"torchmatch.assignment.solve: backend {backend!r} not dispatched"
    raise RuntimeError(msg)  # unreachable; enum is exhaustive


def resolve_backend(
    cost: torch.Tensor,
    *,
    backend: Backend | str = Backend.AUTO,
) -> str:
    """
    Return the op name that ``solve()`` would dispatch to.

    Useful for debugging AUTO routing and for asserting which backend fires
    in a test or benchmark.

    Parameters
    ----------
    cost
        Cost matrix (N, M) or (B, N, M) on the target device.
    backend
        Backend hint. Non-``AUTO`` values are echoed back as their string value
        (e.g. ``"munkres"``).

    Returns
    -------
    op_name
        Exact ``torch.ops.assignment.<name>`` key — e.g. ``"jonker_compact"``,
        ``"munkres"``, ``"lawler"``, ``"greedy"``.

    """
    backend = _coerce_backend(backend)
    if backend != Backend.AUTO:
        return backend.value
    is_cuda = cost.device.type == "cuda"
    if cost.ndim == 2:
        if is_cuda:
            return _pick_auto_cuda_2d(cost.size(0))
        return _pick_jonker_cpu_2d(cost.size(0), cost.size(1))
    _b, n, m = cost.shape
    if not is_cuda:
        return _pick_jonker_cpu_3d(n, m)
    if n == m and n <= _CUDA_BATCH_MAX_TILE:
        return "jonker_dense_batch"
    return _pick_auto_cuda_2d(n)
