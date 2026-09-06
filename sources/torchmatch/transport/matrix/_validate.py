"""
Input validation and shape coercion for ``transport.matrix.solve``.

Three helpers:

- :func:`validate_cost` rejects NaN / -inf, off-grid ndim / dtype / device.
- :func:`coerce_marginals` lifts 1-D defaults to ``(1, N)`` / ``(1, M)``
  (so the batched solver always sees ``(B, N)`` / ``(B, M)``); uniform
  marginals when ``None``; non-negativity checked.
- :func:`fuse_mask_into_cost` runs ``cost.masked_fill(~mask, inf)`` when
  a mask is provided, so the downstream solver sees a uniform cost.

Structural checks (ndim, dtype, device) read metadata and are free.
The value checks are not: each reduces a tensor to one number that
Python then branches on, which materializes a device value on the host.
:func:`skip_value_checks` says when to omit them.
"""

from __future__ import annotations

import math

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def skip_value_checks(tensor: torch.Tensor) -> bool:
    """
    Report whether ``tensor``'s value checks must be skipped.

    Branching on a reduced tensor is not merely slow under tracing, it is
    either impossible or wrong. Under Dynamo the branch is a graph break;
    under ``torch.jit.trace`` the taken branch is burned into the trace as
    if it held for every future input; under ``make_fx`` and on a fake
    tensor there is no value to read, so the branch raises outright. Each
    case wants the same thing, which is for the check not to run.

    The conditions overlap only partly, so all are needed.
    ``torch.compiler.is_compiling()`` covers Dynamo and export but is False
    under ``torch.jit.trace`` and ``make_fx``; a proxy mode is active under
    ``make_fx`` in every tracing mode; ``is_fake`` catches a fake tensor
    reached outside any of those, including one wrapped by a functional or
    other traceable wrapper subclass, which a bare ``isinstance`` misses.
    """
    return (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or get_proxy_mode() is not None
        or is_fake(tensor)
    )


def validate_cost(cost: torch.Tensor, *, check_values: bool = True) -> None:
    """
    Validate ``cost`` dtype, device, ndim, and reject NaN / ``-inf``.

    Parameters
    ----------
    cost
        The cost matrix to check.
    check_values
        Whether to run the NaN / ``-inf`` rejection. The structural checks
        always run: they read metadata, cost nothing, and turning them off
        would only move a clear error to a confusing one deeper in. Pass
        False on a hot path whose costs are known finite by construction.
        The check is skipped regardless when :func:`skip_value_checks`
        holds, since under tracing it cannot be answered.

    Raises
    ------
    ValueError
        If ``cost`` is not float32/float64, not on CPU/CUDA, or not 2-D/3-D.
    RuntimeError
        If ``cost`` contains NaN or ``-inf``.

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
    if not check_values or skip_value_checks(cost) or cost.numel() == 0:
        return
    # Mirrors torchmatch.assignment._solve._validate: RuntimeError, not
    # ValueError, matches the TORCH_CHECK surface the C++ ops raise on the
    # same invariants. Fail-fast at the entry is far cheaper than a kernel
    # crash inside an LSE loop, so this stays the default. One reduction and
    # one host read answer both questions: min propagates NaN, and -inf is
    # the minimum whenever it is present. +inf never wins a min, so a
    # forbidden edge passes.
    lowest = cost.min().item()
    if math.isnan(lowest):
        msg = "torchmatch.transport.matrix.solve: cost contains NaN"
        raise RuntimeError(msg)
    if lowest == -math.inf:
        msg = "torchmatch.transport.matrix.solve: cost contains -inf"
        raise RuntimeError(msg)


def _coerce_one_marginal(  # noqa: PLR0913
    cost: torch.Tensor,
    x: torch.Tensor | None,
    *,
    name: str,
    dim_label: str,
    shape: tuple[int, int],
    check_values: bool = True,
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
    if (
        check_values
        and not skip_value_checks(out)
        and out.numel() != 0
        and out.min().item() < 0
    ):
        msg = f"torchmatch.transport.matrix.solve: {name} must be non-negative"
        raise ValueError(msg)
    return out


def coerce_marginals(
    cost: torch.Tensor,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
    *,
    check_values: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lift a, b to (B, N) / (B, M) with uniform defaults.

    Always returns 3-D-style marginals so the solver dispatches a single
    shape. The dispatcher squeezes the leading B back for 2-D input on
    the return path.

    ``check_values`` gates the non-negativity check, on the same terms as
    :func:`validate_cost`: it is the one part of this that has to read a
    value back to the host.
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
        check_values=check_values,
    )
    b_out = _coerce_one_marginal(
        cost,
        b,
        name="b",
        dim_label="M",
        shape=(batch, m),
        check_values=check_values,
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
