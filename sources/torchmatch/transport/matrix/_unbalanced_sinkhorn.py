"""
Unbalanced Sinkhorn with KL marginal penalty.

Chizat et al. 2018 / Sejourne et al. 2019: replace strict marginal
constraints with KL divergence penalties weighted by ``rho``. The
iteration updates the dual potentials with a damping factor
``damping = 1 / (1 + eps / rho)`` so the new value is a damped average
of the old and the freshly-computed LSE.

As ``rho`` grows the damping approaches 1 and the iteration recovers
the balanced Sinkhorn. As ``rho`` shrinks the damping approaches 0 and
the iteration stalls (no update is applied).
"""

from __future__ import annotations

import torch

from torchmatch.transport.matrix._autograd import register_replay_autograd
from torchmatch.transport.matrix._log_sinkhorn import (
    _LOG_KERNEL_SAFE_FILL,
    _inf_row_col_masks,
)
from torchmatch.transport.matrix._schedule import build_eps_schedule
from torchmatch.transport.matrix._validate import (
    fuse_mask_into_cost,
    validate_cost,
)


def unbalanced_sinkhorn_plan(  # noqa: PLR0913
    cost: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    rho: float,
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float | None = None,
) -> torch.Tensor:
    """Log-domain unbalanced Sinkhorn plan with KL marginal penalty."""
    if rho <= 0:
        msg = (
            "torchmatch.transport.matrix.solve: rho must be positive for "
            f"UNBALANCED_SINKHORN, got {rho}"
        )
        raise ValueError(msg)

    schedule = build_eps_schedule(
        cost=cost,
        reg=eps,
        n_iter=n_iter,
        scaling=scaling,
    )

    b_size, rows, cols = cost.shape
    if b_size == 0:
        return cost.new_empty(b_size, rows, cols)

    device = cost.device
    dtype = cost.dtype

    log_mu = a.log()
    log_nu = b.log()

    inf_row, inf_col = _inf_row_col_masks(cost)

    log_u = torch.zeros(b_size, rows, device=device, dtype=dtype)
    log_v = torch.zeros(b_size, cols, device=device, dtype=dtype)

    for step_eps in schedule:
        damping = 1.0 / (1.0 + step_eps / rho)
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
        # Chizat 2018 Algorithm 2 (Generalized Scaling): u <- (a/Kv)^damping,
        # i.e. log u <- damping * (log a - LSE). A convex combination with
        # the previous log_u would converge to the balanced fixed point
        # log u = log a - LSE, which defeats the unbalanced relaxation.
        log_u = torch.where(
            lse_col.isfinite(),
            damping * (log_mu - lse_col),
            log_u,
        )
        lse_row = torch.logsumexp(log_kernel_col_safe + log_u[:, :, None], dim=1)
        lse_row = torch.where(inf_col, float("-inf"), lse_row)
        log_v = torch.where(
            lse_row.isfinite(),
            damping * (log_nu - lse_row),
            log_v,
        )

    log_kernel = -cost / schedule[-1]
    return log_u[:, :, None] + log_kernel + log_v[:, None, :]


@torch.library.custom_op("transport::unbalanced_sinkhorn", mutates_args=())
def _unbalanced_sinkhorn_op(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    rho: float,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    """``torch.ops.transport.unbalanced_sinkhorn`` entry point."""
    validate_cost(cost)
    if cost.ndim != 3:
        msg = (
            "torch.ops.transport.unbalanced_sinkhorn: cost.ndim must be 3, "
            f"got {cost.ndim}"
        )
        raise ValueError(msg)
    if rho <= 0:
        msg = "torch.ops.transport.unbalanced_sinkhorn: rho must be positive"
        raise ValueError(msg)
    cost = fuse_mask_into_cost(cost, mask)
    return unbalanced_sinkhorn_plan(
        cost,
        eps=eps,
        n_iter=n_iter,
        rho=rho,
        a=a,
        b=b,
        scaling=scaling,
    )


@_unbalanced_sinkhorn_op.register_fake
def _unbalanced_sinkhorn_fake(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    rho: float,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> torch.Tensor:
    del eps, n_iter, rho, a, b, mask, scaling
    torch._check(cost.dim() == 3, lambda: "cost must be 3D (B, N, M)")
    return cost.new_empty(cost.size(0), cost.size(1), cost.size(2))


register_replay_autograd("unbalanced_sinkhorn", unbalanced_sinkhorn_plan)
