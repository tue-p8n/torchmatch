"""Single-problem Jonker-Volgenant solvers return the optimal assignment."""

from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize("op_name", ["jonker_dense", "jonker_compact"])
@pytest.mark.parametrize("n", [8, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("seed", range(10))
def test_lap_jv_optimal(ref_cost, op_name, n, seed):
    torch.manual_seed(seed)
    cost = torch.rand(n, n, dtype=torch.float64)

    op = getattr(torch.ops.assignment, op_name)
    row_to_col = op(cost)

    rows = torch.arange(n)
    matched = row_to_col >= 0
    got = float(cost[rows[matched], row_to_col[matched]].sum())

    ref = ref_cost(cost)
    assert abs(got - ref) < 1e-9, f"{op_name} n={n} seed={seed}: {got} vs {ref}"
