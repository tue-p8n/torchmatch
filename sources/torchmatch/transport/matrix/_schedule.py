"""
Epsilon-scaling schedule for log-domain Sinkhorn.

When ``scaling is None`` we emit a flat schedule of length ``n_iter``
at the target ``reg``.

When ``scaling`` is set we use the Schmitzer 2019 epsilon-scaling
geometric decay: ``initial = max(0.5 * max(finite cost), reg)``; each
step multiplies by ``scaling`` until the value drops to ``reg``, then
the remaining slots are filled with ``reg`` so the schedule has
exactly ``n_iter`` entries.
The trailing fixed-``reg`` tail lets the solver use the full iteration
budget at the target ``reg`` after the warmup decay finishes.
"""

from __future__ import annotations

import torch


def build_eps_schedule(
    *,
    cost: torch.Tensor,
    reg: float,
    n_iter: int,
    scaling: float | None,
) -> list[float]:
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
        List of ``n_iter`` floats, starting at ``initial`` and decaying
        toward ``reg``; the tail is clamped at ``reg``.

    """
    if reg <= 0:
        msg = f"torchmatch.transport.matrix.solve: reg must be positive, got {reg}"
        raise ValueError(msg)
    if n_iter < 1:
        msg = f"torchmatch.transport.matrix.solve: n_iter must be >= 1, got {n_iter}"
        raise ValueError(msg)

    if scaling is None:
        return [reg] * n_iter

    if not (0.0 < scaling < 1.0):
        msg = (
            "torchmatch.transport.matrix.solve: scaling must be in (0, 1), "
            f"got {scaling}"
        )
        raise ValueError(msg)

    finite = cost[torch.isfinite(cost)]
    initial = reg if finite.numel() == 0 else max(0.5 * finite.max().item(), reg)

    schedule: list[float] = []
    current = initial
    for _ in range(n_iter):
        schedule.append(current)
        current = reg if current <= reg else max(current * scaling, reg)
    return schedule
