"""
Sinkhorn divergence.

Computes ``S_eps(a, b) = OT_eps(a, b) - 0.5 * OT_eps(a, a) - 0.5 * OT_eps(b, b)``.

Each ``OT_eps`` is evaluated via the dual-potential lower bound
``<a, f> + <b, g>`` from the converged Sinkhorn iteration.

The sign convention in this module is ``f = -eps * log_u`` and
``g = -eps * log_v`` (Sinkhorn scaling variables), so the raw dual
inner product equals **negative** OT cost.  ``sinkhorn_divergence``
compensates by negating the combination, returning the correct
non-negative ``S_eps``.

When ``cost_aa`` / ``cost_bb`` are omitted, zero-cost matrices are
used for the self-self problems.  The result is then ``OT_eps(a, b)``
rather than the textbook divergence — non-negative, but not debiased.
Pass the true geometric self-cost matrices, or use
``transport.samples.loss(x, y, debias=True)`` for the point-cloud API.
"""

from __future__ import annotations

import torch

from torchmatch.transport.matrix._log_sinkhorn import log_sinkhorn_plan
from torchmatch.transport.matrix._validate import (
    fuse_mask_into_cost,
    validate_cost,
)


def _ot_cost_from_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """``OT_eps = <a, f> + <b, g>``, batched over leading dim."""
    return (a * f).sum(dim=-1) + (b * g).sum(dim=-1)


def _expand_self_cost(
    c_self: torch.Tensor | None,
    ref: torch.Tensor,
    b: int,
    k: int,
) -> torch.Tensor:
    """Return ``c_self`` broadcast to ``(b, k, k)``, or zeros if ``None``."""
    if c_self is None:
        return torch.zeros(b, k, k, device=ref.device, dtype=ref.dtype)
    c_self = c_self.to(device=ref.device, dtype=ref.dtype)
    if c_self.ndim == 2:
        c_self = c_self.unsqueeze(0)
    return c_self.expand(b, -1, -1)


def sinkhorn_divergence(  # noqa: PLR0913
    cost: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the Sinkhorn divergence ``S_eps(a, b)``.

    ``cost`` is (B, N, M); ``a`` is (B, N); ``b`` is (B, M).
    Returns a (B,) scalar tensor.

    When ``cost_aa`` and ``cost_bb`` are provided (shape ``(N, N)`` or
    ``(B, N, N)`` / ``(M, M)`` or ``(B, M, M)``), the self-self OT
    sub-problems use the supplied cost matrices, giving the textbook
    non-negative Sinkhorn divergence (Feydy 2019). When omitted, zeros
    are used (entropic-bias-only approximation, can be negative).
    """
    batch = cost.size(0)
    n = a.size(-1)
    m = b.size(-1)

    _, f_ab, g_ab = log_sinkhorn_plan(
        cost,
        eps=eps,
        n_iter=n_iter,
        a=a,
        b=b,
        scaling=scaling,
        return_potentials=True,
    )
    ot_ab = _ot_cost_from_potentials(f_ab, g_ab, a, b)

    c_aa = _expand_self_cost(cost_aa, cost, batch, n)
    _, f_aa, g_aa = log_sinkhorn_plan(
        c_aa,
        eps=eps,
        n_iter=n_iter,
        a=a,
        b=a,
        scaling=scaling,
        return_potentials=True,
    )
    ot_aa = _ot_cost_from_potentials(f_aa, g_aa, a, a)

    c_bb = _expand_self_cost(cost_bb, cost, batch, m)
    _, f_bb, g_bb = log_sinkhorn_plan(
        c_bb,
        eps=eps,
        n_iter=n_iter,
        a=b,
        b=b,
        scaling=scaling,
        return_potentials=True,
    )
    ot_bb = _ot_cost_from_potentials(f_bb, g_bb, b, b)

    # _ot_cost_from_potentials returns -(true OT cost) because f = -eps*log_u
    # (sign convention from the Sinkhorn scaling variables). Negating here
    # recovers S_eps(a,b) = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b).
    return -(ot_ab - 0.5 * ot_aa - 0.5 * ot_bb)


@torch.library.custom_op("transport::sinkhorn_divergence", mutates_args=())
def _sinkhorn_divergence_op(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor:
    """``torch.ops.transport.sinkhorn_divergence`` entry point."""
    validate_cost(cost)
    if cost.ndim != 3:
        msg = (
            "torch.ops.transport.sinkhorn_divergence: cost.ndim must be 3, "
            f"got {cost.ndim}"
        )
        raise ValueError(msg)
    cost = fuse_mask_into_cost(cost, mask)
    return sinkhorn_divergence(
        cost,
        eps=eps,
        n_iter=n_iter,
        a=a,
        b=b,
        scaling=scaling,
        cost_aa=cost_aa,
        cost_bb=cost_bb,
    )


@_sinkhorn_divergence_op.register_fake
def _sinkhorn_divergence_fake(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor:
    del eps, n_iter, a, b, mask, scaling, cost_aa, cost_bb
    torch._check(cost.dim() == 3, lambda: "cost must be 3D (B, N, M)")
    return cost.new_empty(cost.size(0))
