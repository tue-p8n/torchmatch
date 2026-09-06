"""
Epsilon-scaling schedule for log-domain Sinkhorn.

When ``scaling is None`` this emits a flat schedule of length ``n_iter``
at the target ``reg``.

When ``scaling`` is set this uses the Schmitzer 2019 epsilon-scaling
geometric decay: ``initial = max(0.5 * max(finite cost), reg)``; each
step multiplies by ``scaling`` until the value drops to ``reg``, then
the remaining slots are filled with ``reg`` so the schedule has
exactly ``n_iter`` entries.
The trailing fixed-``reg`` tail lets the solver use the full iteration
budget at the target ``reg`` after the warmup decay finishes.
"""

from __future__ import annotations

import math

import torch


def build_eps_schedule(
    *,
    cost: torch.Tensor,
    reg: float,
    n_iter: int,
    scaling: float | None,
) -> torch.Tensor:
    """
    Build the epsilon schedule for log-domain Sinkhorn.

    Parameters
    ----------
    cost
        Cost matrix (B, N, M) or (N, M). Used only to compute the initial
        epsilon when ``scaling`` is set; ``+inf`` entries are ignored.
    reg
        Target (final) regularization epsilon. Must be positive.
    n_iter
        Total number of schedule entries (= number of Sinkhorn iterations).
    scaling
        Geometric decay factor in ``(0, 1)``. ``None`` returns a flat
        schedule of ``n_iter`` copies of ``reg``.

    Returns
    -------
    schedule
        1-D tensor of ``n_iter`` entries, same device/dtype as ``cost``,
        starting at ``initial`` and decaying toward ``reg``; the tail is
        clamped at ``reg``. Detached: eps is a hyperparameter of the
        iteration, not a value a caller differentiates through, and this
        keeps a replayed backward from leaking one.

    """
    if reg <= 0:
        msg = f"torchmatch.transport.matrix.solve: reg must be positive, got {reg}"
        raise ValueError(msg)
    if n_iter < 1:
        msg = f"torchmatch.transport.matrix.solve: n_iter must be >= 1, got {n_iter}"
        raise ValueError(msg)

    if scaling is None:
        return torch.full((n_iter,), reg, device=cost.device, dtype=cost.dtype)

    if not (0.0 < scaling < 1.0):
        msg = (
            "torchmatch.transport.matrix.solve: scaling must be in (0, 1), "
            f"got {scaling}"
        )
        raise ValueError(msg)

    # cost.numel() == 0 is shape metadata (safe to branch on under tracing),
    # not a value read; amax() below would raise on a genuinely empty
    # reduction, so the empty case takes reg directly.
    if cost.numel() == 0:
        initial = torch.tensor(reg, device=cost.device, dtype=cost.dtype)
    else:
        # A tensor expression throughout, so this stays traceable under
        # torch.compile / make_fx: no host read, no Python branch on a
        # tensor value. Non-finite entries (+inf forbidden edges, NaN) are
        # mapped to -inf so they cannot win the max; when no finite entry
        # exists, -inf * 0.5 is still -inf and the clamp below floors it to
        # reg, folding that case into the same expression as the decay's
        # own floor.
        finite_max = torch.where(torch.isfinite(cost), cost, -math.inf).amax()
        initial = torch.clamp(0.5 * finite_max, min=reg)

    # initial * scaling**k decays geometrically, then the reg floor clamps
    # it for good once a term drops below reg: for m > k with
    # scaling in (0, 1), initial * scaling**m < initial * scaling**k <= reg,
    # so the clamp keeps picking reg. This closed form is exactly the
    # recurrence current = reg if current <= reg else max(current * scaling, reg)
    # unrolled, without the sequential Python loop or its per-step branch.
    exponents = torch.arange(n_iter, device=cost.device, dtype=cost.dtype)
    schedule = torch.clamp(initial * scaling**exponents, min=reg)
    return schedule.detach()
