"""Autograd on the cost-matrix ops themselves, not via the Python path.

Every existing gradient test calls ``solve()``, which reaches the Python
iteration functions directly and so exercises no op registration at all.
Those tests pass whether or not ``torch.ops.transport.*`` is differentiable,
which is exactly how the ops came to report ``requires_grad=True`` while
raising "no autograd formula was registered" at backward. These call the ops.
"""

from __future__ import annotations

import pytest
import torch
from torchmatch.transport.matrix import Backend, solve


def _uniform(n: int, m: int) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.full((1, n), 1.0 / n, dtype=torch.float64)
    b = torch.full((1, m), 1.0 / m, dtype=torch.float64)
    return a, b


def test_log_sinkhorn_op_backward_does_not_raise():
    # The defect: the op built a graph node with no formula behind it, so
    # the failure surfaced only once someone called backward.
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    plan = torch.ops.transport.log_sinkhorn(cost, 0.1, 20, a, b)
    (plan.exp() * cost).sum().backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()


def test_log_sinkhorn_op_gradcheck_cost():
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)

    def fn(c):
        return torch.ops.transport.log_sinkhorn(c, 0.5, 20, a, b).exp().sum()

    assert torch.autograd.gradcheck(fn, cost, eps=1e-6, atol=1e-4)


def test_log_sinkhorn_op_gradcheck_marginals():
    # a and b take gradients too; the formula must not silently drop them.
    cost = torch.rand(1, 3, 4, dtype=torch.float64)
    a = torch.full((1, 3), 1.0 / 3, dtype=torch.float64, requires_grad=True)
    b = torch.full((1, 4), 1.0 / 4, dtype=torch.float64, requires_grad=True)

    def fn(a_in, b_in):
        return torch.ops.transport.log_sinkhorn(cost, 0.5, 20, a_in, b_in).exp().sum()

    assert torch.autograd.gradcheck(fn, (a, b), eps=1e-6, atol=1e-4)


def test_unbalanced_sinkhorn_op_gradcheck():
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)

    def fn(c):
        return (
            torch.ops.transport.unbalanced_sinkhorn(c, 0.5, 20, 0.5, a, b).exp().sum()
        )

    assert torch.autograd.gradcheck(fn, cost, eps=1e-6, atol=1e-4)


def test_sinkhorn_divergence_op_gradcheck():
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)

    def fn(c):
        return torch.ops.transport.sinkhorn_divergence(c, 0.5, 20, a, b).sum()

    assert torch.autograd.gradcheck(fn, cost, eps=1e-6, atol=1e-4)


@pytest.mark.parametrize("reg", [0.05, 0.5])
@pytest.mark.parametrize("n_iter", [5, 50])
def test_op_gradient_equals_the_python_path(reg: float, n_iter: int):
    # The claim the replay formula rests on: it differentiates the iteration
    # as written, so it must agree with solve() at any iteration count,
    # including counts far short of convergence where a fixed-point formula
    # would not.
    torch.manual_seed(0)
    base = torch.rand(2, 4, 5, dtype=torch.float64)
    a = torch.full((2, 4), 1.0 / 4, dtype=torch.float64)
    b = torch.full((2, 5), 1.0 / 5, dtype=torch.float64)

    via_solve = base.clone().requires_grad_()
    solve(
        via_solve, backend=Backend.LOG_SINKHORN, reg=reg, n_iter=n_iter, a=a, b=b
    ).exp().mul(via_solve).sum().backward()

    via_op = base.clone().requires_grad_()
    torch.ops.transport.log_sinkhorn(via_op, reg, n_iter, a, b).exp().mul(
        via_op
    ).sum().backward()

    assert torch.allclose(via_solve.grad, via_op.grad, atol=1e-10)


def test_op_gradient_survives_an_all_inf_row():
    # A fully forbidden row makes the log kernel all -inf, whose softmax
    # backward is NaN. The solver substitutes a finite stand-in; replaying it
    # must reproduce the same substitution, not a naive re-derivation.
    base = torch.rand(1, 3, 3, dtype=torch.float64)
    finite = torch.ones(1, 3, 3, dtype=torch.bool)
    finite[0, 1, :] = False
    cost = torch.where(finite, base, torch.full_like(base, float("inf")))
    cost = cost.requires_grad_()
    a, b = _uniform(3, 3)

    plan = torch.ops.transport.log_sinkhorn(cost, 0.1, 50, a, b)
    plan.where(finite, torch.zeros_like(plan)).exp().sum().backward()

    assert cost.grad is not None
    assert torch.isfinite(cost.grad[finite]).all()


def test_op_gradient_with_a_mask():
    # The mask is fused into the cost inside the op, so the replay has to
    # fuse it again or it differentiates a different problem.
    a, b = _uniform(3, 3)
    base = torch.rand(1, 3, 3, dtype=torch.float64)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    mask[0, 0, 2] = False

    masked = base.clone().requires_grad_()
    torch.ops.transport.log_sinkhorn(masked, 0.5, 20, a, b, mask).exp().sum().backward()

    fused = base.masked_fill(~mask, float("inf")).requires_grad_()
    torch.ops.transport.log_sinkhorn(fused, 0.5, 20, a, b).exp().sum().backward()

    assert torch.allclose(masked.grad[mask], fused.grad[mask], atol=1e-10)


def test_op_backward_skips_marginals_that_need_no_gradient():
    # Only the requested inputs come back, and the rest are None rather than
    # zeros, so an unused marginal costs no backward work.
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    plan = torch.ops.transport.log_sinkhorn(cost, 0.5, 20, a, b)
    plan.exp().sum().backward()
    assert cost.grad is not None
    assert a.grad is None
    assert b.grad is None
