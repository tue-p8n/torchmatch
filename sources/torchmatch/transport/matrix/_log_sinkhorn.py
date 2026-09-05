"""
Log-domain Sinkhorn iterations on a 2-D or 3-D cost matrix.

The 2-D core (:func:`_sinkhorn_log_plan_2d`) implements log-domain
Sinkhorn (Schmitzer 2019, SIAM J. Sci. Comput. 41(3); see
references.bib). It runs in log-space throughout, so ``+inf`` entries
in the cost (forbidden edges) become ``-inf`` in ``log K``, which
:func:`torch.logsumexp` correctly treats as contributing zero mass.

NaN-row handling: a row whose every entry is ``+inf`` would give a
``-inf`` log-sum-exp whose backward (a softmax of an all-``-inf``
vector) is NaN. ``torch.where(lse.isfinite(), ...)`` masks the forward
update but the autograd graph still flows the NaN through both
branches. We pre-substitute fully-``+inf`` rows / cols of the log
kernel with a finite stand-in (``-80``) before the LSE so backward
sees a finite softmax, then restore ``-inf`` on the LSE output via a
detached mask. The final log plan keeps ``-inf`` on forbidden cells.
"""

from __future__ import annotations

import torch

from torchmatch.transport.matrix._autograd import register_replay_autograd
from torchmatch.transport.matrix._schedule import build_eps_schedule
from torchmatch.transport.matrix._validate import (
    fuse_mask_into_cost,
    validate_cost,
)

# log-kernel substitute for fully-+inf rows/cols. exp(-80) ~ 1.8e-35 is
# negligible mass in both float32 and float64 (above the denormal floor)
# and the value is masked out before it reaches the returned log plan.
_LOG_KERNEL_SAFE_FILL: float = -80.0


def _inf_row_col_masks(cost: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Detect rows / columns of ``cost`` that are entirely +inf.

    Operates on the last two dims so both (N, M) and (B, N, M) inputs
    work via dim=-1 / dim=-2.
    """
    is_inf = cost == float("inf")
    return is_inf.all(dim=-1), is_inf.all(dim=-2)


def _sinkhorn_log_plan_2d(
    cost: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return the log-domain Sinkhorn plan for a single 2-D cost matrix."""
    rows, cols = cost.shape
    device = cost.device
    dtype = cost.dtype

    log_mu = a.log()
    log_nu = b.log()

    inf_row, inf_col = _inf_row_col_masks(cost)
    log_kernel = -cost / eps
    log_kernel_row_safe = torch.where(
        inf_row[:, None],
        _LOG_KERNEL_SAFE_FILL,
        log_kernel,
    )
    log_kernel_col_safe = torch.where(
        inf_col[None, :],
        _LOG_KERNEL_SAFE_FILL,
        log_kernel,
    )

    log_u = torch.zeros(rows, device=device, dtype=dtype)
    log_v = torch.zeros(cols, device=device, dtype=dtype)

    for _ in range(n_iter):
        lse_col = torch.logsumexp(log_kernel_row_safe + log_v[None, :], dim=1)
        lse_col = torch.where(inf_row, float("-inf"), lse_col)
        log_u = torch.where(lse_col.isfinite(), log_mu - lse_col, log_u)
        lse_row = torch.logsumexp(log_kernel_col_safe + log_u[:, None], dim=0)
        lse_row = torch.where(inf_col, float("-inf"), lse_row)
        log_v = torch.where(lse_row.isfinite(), log_nu - lse_row, log_v)

    return log_u[:, None] + log_kernel + log_v[None, :]


def _sinkhorn_log_plan_3d(
    cost: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """Return the log-domain Sinkhorn plan for a (B, N, M) batch."""
    b_size, rows, cols = cost.shape
    if b_size == 0:
        return cost.new_empty(b_size, rows, cols)

    device = cost.device
    dtype = cost.dtype

    log_mu = a.log()
    log_nu = b.log()

    inf_row, inf_col = _inf_row_col_masks(cost)
    log_kernel = -cost / eps
    log_kernel_row_safe = torch.where(
        inf_row[:, :, None],
        _LOG_KERNEL_SAFE_FILL,
        log_kernel,
    )
    log_kernel_col_safe = torch.where(
        inf_col[:, None, :],
        _LOG_KERNEL_SAFE_FILL,
        log_kernel,
    )

    log_u = torch.zeros(b_size, rows, device=device, dtype=dtype)
    log_v = torch.zeros(b_size, cols, device=device, dtype=dtype)

    for _ in range(n_iter):
        lse_col = torch.logsumexp(log_kernel_row_safe + log_v[:, None, :], dim=2)
        lse_col = torch.where(inf_row, float("-inf"), lse_col)
        log_u = torch.where(lse_col.isfinite(), log_mu - lse_col, log_u)
        lse_row = torch.logsumexp(log_kernel_col_safe + log_u[:, :, None], dim=1)
        lse_row = torch.where(inf_col, float("-inf"), lse_row)
        log_v = torch.where(lse_row.isfinite(), log_nu - lse_row, log_v)

    return log_u[:, :, None] + log_kernel + log_v[:, None, :]


def log_sinkhorn_plan(  # noqa: PLR0913
    cost: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float | None = None,
    return_potentials: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the log-domain Sinkhorn plan, optionally with epsilon-scaling.

    ``cost`` is expected to be (B, N, M); ``a`` is (B, N); ``b`` is
    (B, M). Validation lives in :mod:`_validate`; this function is the
    pure math.

    When ``scaling`` is ``None``, runs ``n_iter`` fixed-epsilon iterations
    at ``eps``. When ``scaling`` is in (0, 1), uses the schedule from
    :func:`._schedule.build_eps_schedule`.

    When ``return_potentials`` is False (default), returns ``log_plan``
    of shape ``(B, N, M)``. When True, returns
    ``(log_plan, f, g)`` where ``f = -eps_final * log_u`` and
    ``g = -eps_final * log_v`` are the dual potentials. The dispatcher
    uses these for ``unpack=True`` and the Sinkhorn-divergence helper
    consumes them directly to compute ``OT_eps = <a, f> + <b, g>``
    (avoids materializing ``<C, P>``).
    """
    schedule = build_eps_schedule(
        cost=cost,
        reg=eps,
        n_iter=n_iter,
        scaling=scaling,
    )

    b_size, rows, cols = cost.shape
    if b_size == 0:
        empty_plan = cost.new_empty(b_size, rows, cols)
        if return_potentials:
            return (
                empty_plan,
                cost.new_empty(b_size, rows),
                cost.new_empty(b_size, cols),
            )
        return empty_plan

    device = cost.device
    dtype = cost.dtype

    log_mu = a.log()
    log_nu = b.log()

    inf_row, inf_col = _inf_row_col_masks(cost)

    log_u = torch.zeros(b_size, rows, device=device, dtype=dtype)
    log_v = torch.zeros(b_size, cols, device=device, dtype=dtype)

    for step_eps in schedule:
        log_kernel = -cost / step_eps
        log_kernel_row_safe = torch.where(
            inf_row[:, :, None],
            _LOG_KERNEL_SAFE_FILL,
            log_kernel,
        )
        log_kernel_col_safe = torch.where(
            inf_col[:, None, :],
            _LOG_KERNEL_SAFE_FILL,
            log_kernel,
        )
        lse_col = torch.logsumexp(log_kernel_row_safe + log_v[:, None, :], dim=2)
        lse_col = torch.where(inf_row, float("-inf"), lse_col)
        log_u = torch.where(lse_col.isfinite(), log_mu - lse_col, log_u)
        lse_row = torch.logsumexp(log_kernel_col_safe + log_u[:, :, None], dim=1)
        lse_row = torch.where(inf_col, float("-inf"), lse_row)
        log_v = torch.where(lse_row.isfinite(), log_nu - lse_row, log_v)

    final_eps = schedule[-1]
    log_kernel = -cost / final_eps
    log_plan = log_u[:, :, None] + log_kernel + log_v[:, None, :]

    if return_potentials:
        f = -final_eps * log_u
        g = -final_eps * log_v
        return log_plan, f, g
    return log_plan


@torch.library.custom_op("transport::log_sinkhorn", mutates_args=())
def _log_sinkhorn_op(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    """``torch.ops.transport.log_sinkhorn`` entry point."""
    validate_cost(cost)
    if cost.ndim != 3:
        msg = (
            "torch.ops.transport.log_sinkhorn: cost.ndim must be 3, got "
            f"{cost.ndim}. Call torchmatch.transport.matrix.solve(...) for "
            "2-D ergonomic dispatch."
        )
        raise ValueError(msg)
    cost = fuse_mask_into_cost(cost, mask)
    return log_sinkhorn_plan(
        cost,
        eps=eps,
        n_iter=n_iter,
        a=a,
        b=b,
        scaling=scaling,
    )


@_log_sinkhorn_op.register_fake
def _log_sinkhorn_fake(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    del eps, n_iter, a, b, mask, scaling
    torch._check(cost.dim() == 3, lambda: "cost must be 3D (B, N, M)")
    return cost.new_empty(cost.size(0), cost.size(1), cost.size(2))


register_replay_autograd("log_sinkhorn", log_sinkhorn_plan)
