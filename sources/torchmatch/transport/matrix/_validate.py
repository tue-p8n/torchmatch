"""
Input validation and shape coercion for ``transport.matrix.solve``.

Three helpers:

- :func:`validate_cost` rejects NaN / -inf, off-grid ndim / dtype / device.
- :func:`coerce_marginals` lifts 1-D defaults to ``(1, N)`` / ``(1, M)``
  (so the batched solver always sees ``(B, N)`` / ``(B, M)``); uniform
  marginals when ``None``; non-negativity checked.
- :func:`fuse_mask_into_cost` runs ``cost.masked_fill(~mask, inf)`` when
  a mask is provided, so the downstream solver sees a uniform cost.
"""

from __future__ import annotations

import torch


def validate_cost(cost: torch.Tensor) -> None:
    """
    Validate ``cost`` dtype, device, ndim, and reject NaN / ``-inf``.

    Raises
    ------
    ValueError
        If ``cost`` is not float32/float64, not on CPU/CUDA, or not 2-D/3-D.
    RuntimeError
        If ``cost`` contains NaN or ``-inf`` (uses ``.any()`` which forces a
        host sync and breaks ``torch.compile`` graphs; call the ops directly
        via ``torch.ops.transport.<op>`` to bypass this check).

    """
    if cost.ndim not in (2, 3):
        msg = (
            "torchmatch.transport.matrix.solve: cost.ndim must be 2 or 3, "
            f"got {cost.ndim}"
        )
        raise ValueError(msg)
    if cost.dtype not in (torch.float32, torch.float64):
        msg = (
            "torchmatch.transport.matrix.solve: cost.dtype must be float32 "
            f"or float64, got {cost.dtype}"
        )
        raise ValueError(msg)
    if cost.device.type not in ("cpu", "cuda"):
        msg = (
            "torchmatch.transport.matrix.solve: cost.device must be cpu or "
            f"cuda, got {cost.device}"
        )
        raise ValueError(msg)
    # Mirrors torchmatch.assignment._solve._validate: RuntimeError, not
    # ValueError, matches the TORCH_CHECK surface the C++ ops raise on the
    # same invariants. The .any() result forces a host materialization and
    # breaks the graph under torch.compile; this is the price of fail-fast
    # at the dispatcher entry — far cheaper than a kernel crash inside an
    # LSE loop. Direct-op callers that need to avoid the sync can call
    # torch.ops.transport.<op> instead (the ops do their own validation).
    if torch.isnan(cost).any():
        msg = "torchmatch.transport.matrix.solve: cost contains NaN"
        raise RuntimeError(msg)
    if (cost == float("-inf")).any():
        msg = "torchmatch.transport.matrix.solve: cost contains -inf"
        raise RuntimeError(msg)


def _coerce_one_marginal(
    cost: torch.Tensor,
    x: torch.Tensor | None,
    *,
    name: str,
    dim_label: str,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Lift one side (a or b) to ``shape`` (= ``(batch, expected_len)``)."""
    batch, expected_len = shape
    device = cost.device
    dtype = cost.dtype
    if x is None:
        # An axis of length zero has no uniform-marginal value to fill in; the
        # tensor is empty so the fill value is unused, but 1/0 would raise.
        fill = 0.0 if expected_len == 0 else 1.0 / expected_len
        return torch.full(
            (batch, expected_len),
            fill,
            device=device,
            dtype=dtype,
        )
    if x.ndim == 1:
        if x.shape != (expected_len,):
            msg = (
                f"torchmatch.transport.matrix.solve: {name} must have shape "
                f"({dim_label},) = ({expected_len},) for 2D cost, "
                f"got {tuple(x.shape)}"
            )
            raise ValueError(msg)
        out = x.to(device=device, dtype=dtype).unsqueeze(0)
        if cost.ndim == 3:
            out = out.expand(batch, expected_len).contiguous()
    else:
        if x.shape != (batch, expected_len):
            msg = (
                f"torchmatch.transport.matrix.solve: {name} must have shape "
                f"(B, {dim_label}) = ({batch}, {expected_len}) for 3D cost, "
                f"got {tuple(x.shape)}"
            )
            raise ValueError(msg)
        out = x.to(device=device, dtype=dtype)
    if (out < 0).any():
        msg = f"torchmatch.transport.matrix.solve: {name} must be non-negative"
        raise ValueError(msg)
    return out


def coerce_marginals(
    cost: torch.Tensor,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lift a, b to (B, N) / (B, M) with uniform defaults.

    Always returns 3-D-style marginals so the solver dispatches a single
    shape. The dispatcher squeezes the leading B back for 2-D input on
    the return path.
    """
    if cost.ndim == 2:
        n, m = cost.shape
        batch = 1
    else:
        batch, n, m = cost.shape

    a_out = _coerce_one_marginal(
        cost,
        a,
        name="a",
        dim_label="N",
        shape=(batch, n),
    )
    b_out = _coerce_one_marginal(
        cost,
        b,
        name="b",
        dim_label="M",
        shape=(batch, m),
    )
    return a_out, b_out


def fuse_mask_into_cost(
    cost: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    """Replace ``~mask`` entries of cost with ``+inf``; pass-through if None."""
    if mask is None:
        return cost
    return cost.masked_fill(~mask, float("inf"))
