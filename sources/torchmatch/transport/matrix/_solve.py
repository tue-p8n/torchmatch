"""
Cost-matrix optimal-transport dispatcher.

``torchmatch.transport.matrix.solve`` validates the input, fuses any
mask into the cost matrix, picks a backend by the ``Backend`` enum
(or string alias), and returns a transport plan (or scalar divergence).

See ``docs/superpowers/specs/2026-05-21-transport-ops-design.md``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, overload

import torch

from torchmatch.transport.matrix._exact_emd import exact_emd
from torchmatch.transport.matrix._log_sinkhorn import log_sinkhorn_plan
from torchmatch.transport.matrix._sinkhorn_divergence import sinkhorn_divergence
from torchmatch.transport.matrix._unbalanced_sinkhorn import (
    unbalanced_sinkhorn_plan,
)
from torchmatch.transport.matrix._validate import (
    coerce_marginals,
    fuse_mask_into_cost,
    validate_cost,
)

# The Sinkhorn-family backends call the Python iteration functions
# directly rather than torch.ops.transport.<op>. The ops now carry an
# autograd formula of their own (see _autograd), so the original reason
# for this split -- that the custom_op boundary was opaque to autograd --
# no longer holds, and routing solve() through the ops would additionally
# give a compiled caller one opaque node instead of an unrolled loop.
# That is deliberately left as a separate change: it would move every
# existing gradient consumer of solve() onto a new path in the same commit
# that introduces the path. EXACT_EMD has no autograd either way and goes
# through its op.

__all__ = ["Backend", "marginal_error", "solve"]


class Backend(StrEnum):
    AUTO = "auto"
    LOG_SINKHORN = "log_sinkhorn"
    SINKHORN_DIVERGENCE = "sinkhorn_divergence"
    UNBALANCED_SINKHORN = "unbalanced_sinkhorn"
    EXACT_EMD = "exact_emd"


def _coerce_backend(value: Backend | str) -> Backend:
    try:
        return Backend(value)
    except ValueError as err:
        msg = (
            f"torchmatch.transport.matrix.solve: unknown backend {value!r}; "
            f"valid values are {[b.value for b in Backend]}"
        )
        raise ValueError(msg) from err


@overload
def solve(
    cost: torch.Tensor,
    *,
    backend: Backend | str = ...,
    reg: float = ...,
    n_iter: int = ...,
    mask: torch.Tensor | None = ...,
    a: torch.Tensor | None = ...,
    b: torch.Tensor | None = ...,
    scaling: float | None = ...,
    rho: float = ...,
    cost_aa: torch.Tensor | None = ...,
    cost_bb: torch.Tensor | None = ...,
    unpack: Literal[False] = False,
    validate: bool = ...,
) -> torch.Tensor: ...


@overload
def solve(
    cost: torch.Tensor,
    *,
    backend: Backend | str = ...,
    reg: float = ...,
    n_iter: int = ...,
    mask: torch.Tensor | None = ...,
    a: torch.Tensor | None = ...,
    b: torch.Tensor | None = ...,
    scaling: float | None = ...,
    rho: float = ...,
    cost_aa: torch.Tensor | None = ...,
    cost_bb: torch.Tensor | None = ...,
    unpack: Literal[True],
    validate: bool = ...,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]: ...


def solve(  # noqa: PLR0913, PLR0911, PLR0912, C901
    cost: torch.Tensor,
    *,
    backend: Backend | str = Backend.AUTO,
    reg: float = 0.1,
    n_iter: int = 100,
    mask: torch.Tensor | None = None,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    scaling: float | None = None,
    rho: float = 1.0,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
    unpack: bool = False,
    validate: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Solve a cost-matrix optimal-transport problem.

    ``validate`` gates the input checks that have to read a value back to
    the host: the NaN / ``-inf`` rejection on ``cost`` and the
    non-negativity check on the marginals. The structural checks on ndim,
    dtype, device and shape always run, because they read metadata and cost
    nothing. Pass ``validate=False`` on a hot path whose inputs are finite
    by construction; the checks are skipped automatically under tracing
    regardless, where they cannot be answered at all.
    """
    backend = _coerce_backend(backend)
    validate_cost(cost, check_finite=validate)

    squeeze_batch = cost.ndim == 2
    if squeeze_batch:
        cost = cost.unsqueeze(0)

    a_, b_ = coerce_marginals(cost, a, b, check_finite=validate)
    mask_to_fuse = mask.unsqueeze(0) if (squeeze_batch and mask is not None) else mask
    cost = fuse_mask_into_cost(cost, mask_to_fuse)

    if backend in (Backend.AUTO, Backend.LOG_SINKHORN):
        if unpack:
            log_plan, f, g = log_sinkhorn_plan(
                cost,
                eps=reg,
                n_iter=n_iter,
                a=a_,
                b=b_,
                scaling=scaling,
                return_potentials=True,
            )
            if squeeze_batch:
                return log_plan.squeeze(0), f.squeeze(0), g.squeeze(0)
            return log_plan, f, g
        log_plan = log_sinkhorn_plan(
            cost,
            eps=reg,
            n_iter=n_iter,
            a=a_,
            b=b_,
            scaling=scaling,
        )
        if squeeze_batch:
            return log_plan.squeeze(0)
        return log_plan

    if backend == Backend.SINKHORN_DIVERGENCE:
        # cost_aa/cost_bb: 2-D inputs are expanded to (B, N, N)/(B, M, M) inside
        # sinkhorn_divergence(). squeeze_batch only applies to `cost` itself;
        # self-cost matrices are always provided in full by the caller.
        result = sinkhorn_divergence(
            cost,
            eps=reg,
            n_iter=n_iter,
            a=a_,
            b=b_,
            scaling=scaling,
            cost_aa=cost_aa,
            cost_bb=cost_bb,
        )
        if squeeze_batch:
            return result.squeeze(0)
        return result
    if backend == Backend.UNBALANCED_SINKHORN:
        log_plan = unbalanced_sinkhorn_plan(
            cost,
            eps=reg,
            n_iter=n_iter,
            rho=rho,
            a=a_,
            b=b_,
            scaling=scaling,
        )
        if unpack:
            # At the Chizat 2018 fixed point with log u = damping*(log a - L)
            # and damping = rho/(rho+eps), the relation
            # logsumexp(log_plan, dim=2) - log a = -(eps/rho) * log u holds,
            # so f = -eps * log u = rho * (logsumexp - log a). Two
            # NaN-grad cases must be silenced for autograd safety:
            # (1) zero-marginal entries (a_i == 0): log(0) = -inf.
            # (2) all-+inf cost rows: the whole log_plan row is -inf, and
            #     logsumexp of an all-(-inf) vector has NaN backward.
            # torch.where alone is not enough because its backward
            # propagates gradient through both branches; substitute with a
            # finite stand-in before the log / logsumexp call.
            mask_a = a_ > 0
            mask_b = b_ > 0
            neg_inf_row = (log_plan == float("-inf")).all(dim=2)
            neg_inf_col = (log_plan == float("-inf")).all(dim=1)
            zero_f = neg_inf_row | ~mask_a
            zero_g = neg_inf_col | ~mask_b
            log_plan_a_safe = torch.where(
                zero_f[..., None],
                torch.zeros_like(log_plan),
                log_plan,
            )
            log_plan_b_safe = torch.where(
                zero_g[:, None, :],
                torch.zeros_like(log_plan),
                log_plan,
            )
            a_safe = torch.where(mask_a, a_, torch.ones_like(a_))
            b_safe = torch.where(mask_b, b_, torch.ones_like(b_))
            f_raw = rho * (torch.logsumexp(log_plan_a_safe, dim=2) - a_safe.log())
            g_raw = rho * (torch.logsumexp(log_plan_b_safe, dim=1) - b_safe.log())
            f = torch.where(zero_f, torch.zeros_like(a_), f_raw)
            g = torch.where(zero_g, torch.zeros_like(b_), g_raw)
            if squeeze_batch:
                return log_plan.squeeze(0), f.squeeze(0), g.squeeze(0)
            return log_plan, f, g
        if squeeze_batch:
            return log_plan.squeeze(0)
        return log_plan
    if backend == Backend.EXACT_EMD:
        plan = exact_emd(cost, a=a_, b=b_, mask=None)
        if unpack:
            if squeeze_batch:
                return plan.squeeze(0), None, None
            return plan, None, None
        if squeeze_batch:
            return plan.squeeze(0)
        return plan

    msg = f"torchmatch.transport.matrix.solve: backend {backend!r} not dispatched"
    raise RuntimeError(msg)


def marginal_error(
    log_plan: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> tuple[float, float]:
    """
    Compute the max marginal error of a log-domain transport plan.

    Useful for checking Sinkhorn convergence quality after ``solve()``.
    Works with any backend that returns a log-plan (``LOG_SINKHORN``,
    ``UNBALANCED_SINKHORN``, ``EXACT_EMD``).

    Parameters
    ----------
    log_plan
        Log-domain transport plan (B, N, M) or (N, M), as returned by
        ``solve()``.
    a
        Source marginals (B, N) or (N,). Must match the ``a`` passed to
        ``solve()``, or uniform weights if ``a=None`` was used.
    b
        Target marginals (B, M) or (M,).

    Returns
    -------
    row_err, col_err
        Max absolute deviation of plan row-sums and col-sums from ``a``
        and ``b``, respectively.

    """
    plan = log_plan.exp()
    row_err = (plan.sum(dim=-1) - a).abs().max().item()
    col_err = (plan.sum(dim=-2) - b).abs().max().item()
    return float(row_err), float(col_err)
