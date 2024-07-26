"""Utility for computing total assignment cost from a solution."""

from __future__ import annotations

import torch


def assignment_cost(
    cost: torch.Tensor,
    matches: torch.Tensor,
    *,
    reduction: str = "sum",
) -> torch.Tensor:
    """
    Compute the total cost of a LAP assignment.

    Parameters
    ----------
    cost
        Cost matrix (N, M) or (B, N, M). float32 or float64.
    matches
        Row→col assignment (N,) or (B, N). int64. Unmatched rows have ``-1``.
    reduction
        How to aggregate per-row costs: ``"sum"`` (default) sums all matched
        rows; ``"mean"`` divides by the number of matched rows; ``"none"``
        returns per-row costs with unmatched rows set to 0.

    Returns
    -------
    total
        Scalar for 2-D input or (B,) for 3-D input, unless
        ``reduction="none"``, in which case the shape is (N,) or (B, N).

    """
    if cost.ndim == 2:
        n = cost.size(0)
        valid = matches >= 0
        matched = matches.clamp(min=0)
        row_idx = torch.arange(n, device=cost.device)
        costs = cost[row_idx, matched]
        costs = torch.where(valid, costs, costs.new_zeros(()))
        if reduction == "none":
            return costs
        if reduction == "mean":
            n_matched = valid.sum().clamp(min=1)
            return costs.sum() / n_matched.to(costs.dtype)
        return costs.sum()

    b, n = cost.shape[:2]
    valid = matches >= 0
    matched = matches.clamp(min=0)
    batch_idx = torch.arange(b, device=cost.device)[:, None].expand(b, n)
    row_idx = torch.arange(n, device=cost.device)[None, :].expand(b, n)
    costs = cost[batch_idx, row_idx, matched]
    costs = torch.where(valid, costs, costs.new_zeros(()))
    if reduction == "none":
        return costs
    if reduction == "mean":
        n_matched = valid.sum(dim=-1).clamp(min=1)
        return costs.sum(dim=-1) / n_matched.to(costs.dtype)
    return costs.sum(dim=-1)
