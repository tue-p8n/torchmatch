r"""Bertsekas-style auction algorithm for linear assignment."""

from __future__ import annotations

import torch

from torchmatch.assignment._validate import check_finite

__all__ = ["auction_assignment"]


@torch.no_grad()
def auction_assignment(  # noqa: PLR0915
    cost_matrix: torch.Tensor,
    bid_size: float,
    max_iters: int = 100_000,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Solve a linear assignment problem using Bertsekas' auction algorithm.

    Converts the cost matrix to a profit matrix, then runs synchronous
    bidding until every row (or every column, for rectangular problems) is
    assigned. Non-finite entries mark forbidden pairs and are penalised
    internally so the solver never selects them.

    Parameters
    ----------
    cost_matrix
        ``(N, M)`` cost matrix. ``+inf`` entries mark forbidden pairs.
        NaN and ``-inf`` are rejected.
    bid_size
        Auction bid step size. The internal ``epsilon`` is derived as
        ``min(bid_size / min(N, M), 1e-3)``.
    max_iters
        Maximum number of bidding iterations. Raises ``RuntimeError`` if
        the algorithm has not converged within this budget.

    Returns
    -------
    matches : torch.Tensor
        ``(K, 2)`` long tensor of matched ``(row, col)`` indices.
    unmatched_rows : torch.Tensor
        ``(N - K,)`` long tensor of unmatched row indices.
    unmatched_cols : torch.Tensor
        ``(M - K,)`` long tensor of unmatched column indices.

    """
    if cost_matrix.ndim != 2:
        msg = f"cost_matrix must be a 2D tensor, got {cost_matrix.ndim}D"
        raise ValueError(msg)
    if bid_size <= 0:
        msg = f"bid_size must be strictly positive, got {bid_size}"
        raise ValueError(msg)
    # auction_assignment is a plain function with no fake-tensor-routed
    # forward, so tracing it directly (torch.compile, make_fx, a fake
    # tensor reaching it outside either) would otherwise hit this branch
    # on a value that cannot be read; check_finite stands aside then.
    bad = check_finite(cost_matrix)
    if bad is not None:
        msg = f"cost_matrix contains {bad}"
        raise ValueError(msg)

    # Cast non-floating-point inputs and low-precision floats to float32.
    # 1e5 overflows float16 (max ~65504) and eps underflows in bfloat16.
    # Outputs are integer indices so the upcast is lossless.
    if not cost_matrix.is_floating_point() or cost_matrix.dtype in (
        torch.float16,
        torch.bfloat16,
    ):
        cost_matrix = cost_matrix.to(torch.float32)

    device = cost_matrix.device
    dtype = cost_matrix.dtype
    rows, cols = cost_matrix.shape
    valid_mask = torch.isfinite(cost_matrix)

    if rows == 0 or cols == 0 or not valid_mask.any():
        return (
            torch.empty((0, 2), dtype=torch.long, device=device),
            torch.arange(rows, dtype=torch.long, device=device),
            torch.arange(cols, dtype=torch.long, device=device),
        )

    max_valid_cost = cost_matrix[valid_mask].max()
    penalty = max_valid_cost.abs() * 0.1 + 1e5
    safe_cost = torch.where(valid_mask, cost_matrix, max_valid_cost + penalty)

    profit = safe_cost * -1
    profit = profit - profit.min()
    # clamp avoids a host-device sync (if profit_max > 0 would pull to CPU).
    # If all costs are identical profit_max == 0; clamping to tiny keeps the
    # division well-defined and profit stays all-zero, which is correct.
    profit = profit / profit.max().clamp(min=torch.finfo(dtype).tiny)

    eps = min(bid_size / min(rows, cols), 1e-3)

    price = torch.zeros(cols, dtype=dtype, device=device)
    ass = torch.full((rows,), -1, device=device, dtype=torch.long)
    # 1D per-column accumulators replace the rows x cols bids tensor.
    # Memory: O(cols) instead of O(rows x cols); fill ops touch cols not rows x cols.
    col_bid = torch.full((cols,), float("-inf"), dtype=dtype, device=device)
    col_winner = torch.full((cols,), -1, dtype=torch.long, device=device)

    # Restructured to call nonzero() once per iteration (one CUDA sync) and
    # reuse the result for both the exit check and the bidding step, avoiding
    # the second sync that (ass < 0).sum() in the while condition would add.
    iters = 0
    limit = max(0, rows - cols)
    while True:
        unassigned = (ass == -1).nonzero(as_tuple=True)[0]
        if unassigned.numel() <= limit:
            break
        if iters >= max_iters:
            msg = f"Auction algorithm did not converge within {max_iters} iterations."
            raise RuntimeError(msg)
        iters += 1

        # value is always finite: safe_cost replaced all non-finite entries
        # with a finite penalty, price only accumulates finite increments.
        value = profit[unassigned] - price
        if cols < 2:
            # topk(2) requires at least 2 columns; pad with a sentinel strictly
            # below all real values. Subtracting a flat 1.0 loses precision when
            # value.min() has large magnitude in float32; subtract the full range
            # (clamped to ≥ 1.0) so the gap is always representable.
            v_range = (value.max() - value.min()).clamp(min=1.0)
            pad = value.new_full((value.shape[0], 2 - cols), value.min() - v_range)
            value_for_topk = torch.cat([value, pad], dim=1)
        else:
            value_for_topk = value
        top_value, top_idx = value_for_topk.topk(2, dim=1)

        first_idx = top_idx[:, 0]
        first_value, second_value = top_value[:, 0], top_value[:, 1]
        bid_increments = first_value - second_value + eps

        # Scatter-reduce: find the max bid and its row per column in one pass.
        # col_bid[j] = max bid placed on column j this round.
        # col_winner[j] = row index of the highest bidder for column j.
        col_bid.fill_(float("-inf"))
        col_bid.scatter_reduce_(
            0, first_idx, bid_increments, reduce="amax", include_self=True
        )

        # Compute the per-column bid mask once; reuse for have_bidder lookup
        # and prev_holders scan to avoid a second col_bid > -inf evaluation.
        # have_bidder is always non-empty here: unassigned is non-empty and
        # every unassigned row places a strictly positive bid (eps > 0).
        bid_col_mask = col_bid > float("-inf")
        have_bidder = bid_col_mask.nonzero(as_tuple=True)[0]

        col_winner.fill_(-1)
        is_max_bidder = bid_increments == col_bid[first_idx]
        col_winner.scatter_(0, first_idx[is_max_bidder], unassigned[is_max_bidder])

        high_bids = col_bid[have_bidder]
        high_bidders = col_winner[have_bidder]
        price[have_bidder] += high_bids

        # For unassigned rows ass == -1; clamp to 0 so the index is valid, then
        # mask those rows out with (ass >= 0). No extra tensor allocations.
        prev_holders = (bid_col_mask[ass.clamp(min=0)] & (ass >= 0)).nonzero(
            as_tuple=True
        )[0]

        ass[prev_holders] = -1
        ass[high_bidders] = have_bidder

    idx = torch.arange(rows, device=device)
    valid_assignments = ass >= 0
    matched_rows = idx[valid_assignments]
    matched_cols = ass[valid_assignments]

    actually_valid = valid_mask[matched_rows, matched_cols]
    final_rows = matched_rows[actually_valid]
    final_cols = matched_cols[actually_valid]

    matches = torch.stack([final_rows, final_cols], dim=1)

    all_cols = torch.arange(cols, device=device)
    unmatched_rows_mask = torch.ones(rows, dtype=torch.bool, device=device)
    unmatched_rows_mask[final_rows] = False
    unmatched_rows = idx[unmatched_rows_mask]
    unmatched_cols_mask = torch.ones(cols, dtype=torch.bool, device=device)
    unmatched_cols_mask[final_cols] = False
    unmatched_cols = all_cols[unmatched_cols_mask]

    return matches, unmatched_rows, unmatched_cols
