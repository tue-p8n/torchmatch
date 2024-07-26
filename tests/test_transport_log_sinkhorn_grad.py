"""Gradient correctness tests for transport.matrix.solve LOG_SINKHORN."""

from __future__ import annotations

import torch
from torchmatch.transport.matrix import Backend, solve


def test_log_sinkhorn_gradcheck_cost():
    # Small problem so gradcheck is tractable.
    torch.manual_seed(0)
    cost = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
    a = torch.full((3,), 1.0 / 3, dtype=torch.float64)
    b = torch.full((3,), 1.0 / 3, dtype=torch.float64)

    def fn(c):
        return (
            solve(
                c,
                backend=Backend.LOG_SINKHORN,
                reg=0.5,
                n_iter=20,
                a=a,
                b=b,
            )
            .exp()
            .sum()
        )

    assert torch.autograd.gradcheck(fn, cost, eps=1e-6, atol=1e-4)


def test_log_sinkhorn_gradient_flows():
    # End-to-end smoke: gradient w.r.t. cost is non-trivial. Reduce by the OT
    # cost <P, C> rather than sum(P): the latter is the total mass (=1 at
    # convergence) and so has zero gradient w.r.t. C up to round-off.
    cost = torch.rand(2, 3, 3, dtype=torch.float64, requires_grad=True)
    out = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=50)
    (out.exp() * cost).sum().backward()
    assert cost.grad is not None
    assert not torch.isnan(cost.grad).any()
    assert (cost.grad.abs() > 1e-12).any()


def test_log_sinkhorn_gradient_safe_through_all_inf_row():
    # A row of cost = +inf (forbidden) is a legitimate input. Naive
    # logsumexp of an all-(-inf) kernel row has NaN backward (softmax of
    # -inf); the implementation substitutes a finite stand-in before LSE
    # and restores -inf on the forward via a detached mask. Verify no
    # NaN gradient propagates to the finite entries of cost.
    base = torch.rand(3, 3, dtype=torch.float64)
    finite_mask = torch.ones(3, 3, dtype=torch.bool)
    finite_mask[1, :] = False  # entire row 1 forbidden
    cost = torch.where(finite_mask, base, torch.full_like(base, float("inf")))
    cost = cost.requires_grad_()
    out = solve(cost, backend=Backend.LOG_SINKHORN, reg=0.1, n_iter=50)
    # Only the finite cells of log_plan are differentiable; reduce on them.
    out.where(finite_mask, torch.zeros_like(out)).exp().sum().backward()
    assert cost.grad is not None
    finite_grad = cost.grad[finite_mask]
    assert torch.isfinite(finite_grad).all(), (
        f"NaN gradient leaked into finite cells: {finite_grad}"
    )


def test_unbalanced_gradient_safe_through_all_inf_row():
    base = torch.rand(3, 3, dtype=torch.float64)
    finite_mask = torch.ones(3, 3, dtype=torch.bool)
    finite_mask[1, :] = False
    cost = torch.where(finite_mask, base, torch.full_like(base, float("inf")))
    cost = cost.requires_grad_()
    out = solve(
        cost,
        backend=Backend.UNBALANCED_SINKHORN,
        reg=0.1,
        n_iter=50,
        rho=0.5,
    )
    out.where(finite_mask, torch.zeros_like(out)).exp().sum().backward()
    assert cost.grad is not None
    finite_grad = cost.grad[finite_mask]
    assert torch.isfinite(finite_grad).all()
