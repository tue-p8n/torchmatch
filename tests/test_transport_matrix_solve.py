"""End-to-end tests for transport.matrix.solve(...) on LOG_SINKHORN."""

from __future__ import annotations

import pytest
import torch
from torchmatch.transport.matrix import Backend, solve


def test_solve_2d_returns_2d_log_plan():
    cost = torch.rand(3, 4, dtype=torch.float64)
    out = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=50)
    assert out.shape == (3, 4)
    plan = out.exp()
    assert torch.allclose(
        plan.sum(dim=1),
        torch.full((3,), 1.0 / 3, dtype=torch.float64),
        atol=1e-5,
    )
    assert torch.allclose(
        plan.sum(dim=0),
        torch.full((4,), 1.0 / 4, dtype=torch.float64),
        atol=1e-5,
    )


def test_solve_3d_returns_3d_log_plan():
    cost = torch.rand(2, 3, 4, dtype=torch.float64)
    out = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=50)
    assert out.shape == (2, 3, 4)


def test_solve_unpack_returns_plan_f_g_triple_2d():
    cost = torch.rand(3, 4, dtype=torch.float64)
    plan, f, g = solve(cost, backend=Backend.LOG_SINKHORN, unpack=True)
    assert plan.shape == (3, 4)
    assert f.shape == (3,)
    assert g.shape == (4,)


def test_solve_unpack_returns_plan_f_g_triple_3d():
    cost = torch.rand(2, 3, 4, dtype=torch.float64)
    plan, f, g = solve(cost, backend=Backend.LOG_SINKHORN, unpack=True)
    assert plan.shape == (2, 3, 4)
    assert f.shape == (2, 3)
    assert g.shape == (2, 4)


def test_solve_auto_backend_is_log_sinkhorn():
    cost = torch.rand(3, 4, dtype=torch.float64)
    out_auto = solve(cost, backend=Backend.AUTO, reg=0.1, n_iter=50)
    out_log = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=50)
    assert torch.equal(out_auto, out_log)


def test_solve_accepts_mask():
    cost = torch.rand(3, 4, dtype=torch.float64)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[0, 1] = False
    out = solve(
        cost,
        backend=Backend.LOG_SINKHORN,
        mask=mask,
        reg=0.1,
        n_iter=50,
    )
    assert torch.isneginf(out[0, 1])


def test_solve_accepts_custom_marginals_2d():
    cost = torch.rand(3, 4, dtype=torch.float64)
    a = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float64)
    b = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64)
    out = solve(
        cost,
        backend=Backend.LOG_SINKHORN,
        a=a,
        b=b,
        reg=0.1,
        n_iter=200,
    )
    plan = out.exp()
    assert torch.allclose(plan.sum(dim=1), a, atol=1e-4)
    assert torch.allclose(plan.sum(dim=0), b, atol=1e-4)


def test_solve_eps_scaling_runs():
    cost = torch.rand(3, 4, dtype=torch.float64)
    out = solve(
        cost,
        backend=Backend.LOG_SINKHORN,
        reg=0.05,
        n_iter=50,
        scaling=0.5,
    )
    assert out.shape == (3, 4)


def test_solve_rejects_invalid_ndim():
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        solve(torch.zeros(5), backend=Backend.LOG_SINKHORN)


def test_solve_rejects_nan():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="cost contains NaN"):
        solve(bad, backend=Backend.LOG_SINKHORN)


def test_solve_rejects_neg_inf():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("-inf")
    with pytest.raises(RuntimeError, match="cost contains -inf"):
        solve(bad, backend=Backend.LOG_SINKHORN)


@pytest.mark.parametrize("shape", [(0, 3, 4), (2, 0, 4), (2, 3, 0), (0, 0)])
def test_solve_handles_empty_shapes(shape):
    cost = torch.empty(*shape, dtype=torch.float64)
    out = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=10)
    assert out.shape == shape


def test_solve_rejects_unknown_backend():
    cost = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="unknown backend"):
        solve(cost, backend="not_a_backend")


def test_solve_rejects_non_positive_reg():
    cost = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="reg must be positive"):
        solve(cost, backend=Backend.LOG_SINKHORN, reg=0.0, scaling=0.5)


def test_auto_resolves_to_log_sinkhorn_regardless_of_shape():
    # 2D and 3D, CPU; result identical to explicit LOG_SINKHORN.
    for shape in [(3, 4), (2, 3, 4)]:
        torch.manual_seed(0)
        cost = torch.rand(*shape, dtype=torch.float64)
        out_auto = solve(cost, backend=Backend.AUTO, reg=0.1, n_iter=50)
        out_log = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=50)
        assert torch.equal(out_auto, out_log)
