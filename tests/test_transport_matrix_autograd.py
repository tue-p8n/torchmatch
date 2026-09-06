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
import torchmatch.transport.matrix._sinkhorn_divergence
from torchmatch.transport.matrix import Backend, solve

_SINKHORN_OPS = ("log_sinkhorn", "unbalanced_sinkhorn", "sinkhorn_divergence")


def _uniform(n: int, m: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.full((batch, n), 1.0 / n, dtype=torch.float64)
    b = torch.full((batch, m), 1.0 / m, dtype=torch.float64)
    return a, b


def _reduce(op: str, cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    """Call ``op`` at eps 0.5 for 20 iterations and reduce to a scalar."""
    handle = getattr(torch.ops.transport, op)
    if op == "unbalanced_sinkhorn":
        return handle(cost, 0.5, 20, 0.5, a, b).exp().sum()
    if op == "sinkhorn_divergence":
        return handle(cost, 0.5, 20, a, b).sum()
    return handle(cost, 0.5, 20, a, b).exp().sum()


def test_log_sinkhorn_op_backward_does_not_raise():
    # The defect: the op built a graph node with no formula behind it, so
    # the failure surfaced only once someone called backward.
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    plan = torch.ops.transport.log_sinkhorn(cost, 0.1, 20, a, b)
    (plan.exp() * cost).sum().backward()
    assert cost.grad is not None
    assert torch.isfinite(cost.grad).all()


@pytest.mark.parametrize("op", _SINKHORN_OPS)
def test_op_gradcheck_cost(op: str):
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda c: _reduce(op, c, a, b), cost, eps=1e-6, atol=1e-4
    )


def test_log_sinkhorn_op_gradcheck_marginals():
    # a and b take gradients too; the formula must not silently drop them.
    cost = torch.rand(1, 3, 4, dtype=torch.float64)
    a = torch.full((1, 3), 1.0 / 3, dtype=torch.float64, requires_grad=True)
    b = torch.full((1, 4), 1.0 / 4, dtype=torch.float64, requires_grad=True)

    def fn(a_in, b_in):
        return torch.ops.transport.log_sinkhorn(cost, 0.5, 20, a_in, b_in).exp().sum()

    assert torch.autograd.gradcheck(fn, (a, b), eps=1e-6, atol=1e-4)


def test_divergence_op_differentiates_the_self_costs():
    # In the debiased setting cost_aa = c(x, x) depends on the same
    # predictions as cost, so its gradient is what debiasing contributes;
    # the op must hand it back exactly as the Python path does.
    torch.manual_seed(0)
    a, b = _uniform(3, 3)
    base = [torch.rand(1, 3, 3, dtype=torch.float64) for _ in range(3)]

    via_solve = [t.clone().requires_grad_() for t in base]
    solve(
        via_solve[0],
        backend=Backend.SINKHORN_DIVERGENCE,
        reg=0.5,
        n_iter=20,
        a=a,
        b=b,
        cost_aa=via_solve[1],
        cost_bb=via_solve[2],
    ).sum().backward()

    via_op = [t.clone().requires_grad_() for t in base]
    torch.ops.transport.sinkhorn_divergence(
        via_op[0], 0.5, 20, a, b, None, None, via_op[1], via_op[2]
    ).sum().backward()

    for left, right in zip(via_solve, via_op, strict=True):
        assert right.grad is not None
        assert torch.allclose(left.grad, right.grad, atol=1e-10)
    assert via_op[1].grad.abs().sum() > 0


@pytest.mark.parametrize("reg", [0.05, 0.5])
@pytest.mark.parametrize("n_iter", [5, 50])
def test_op_gradient_equals_the_python_path(reg: float, n_iter: int):
    # The claim the replay formula rests on: it differentiates the iteration
    # as written, so it must agree with solve() at any iteration count,
    # including counts far short of convergence where a fixed-point formula
    # would not.
    torch.manual_seed(0)
    base = torch.rand(2, 4, 5, dtype=torch.float64)
    a, b = _uniform(4, 5, batch=2)

    via_solve = base.clone().requires_grad_()
    solve(
        via_solve, backend=Backend.LOG_SINKHORN, reg=reg, n_iter=n_iter, a=a, b=b
    ).exp().mul(via_solve).sum().backward()

    via_op = base.clone().requires_grad_()
    torch.ops.transport.log_sinkhorn(via_op, reg, n_iter, a, b).exp().mul(
        via_op
    ).sum().backward()

    assert torch.allclose(via_solve.grad, via_op.grad, atol=1e-10)


def test_op_second_order_gradient_matches_the_python_path():
    # A gradient taken with create_graph must stay connected to the input,
    # which the replay only achieves by differentiating through the caller's
    # tensor rather than a detached copy. Gradient penalties and
    # Hessian-vector products rest on this.
    torch.manual_seed(0)
    a, b = _uniform(3, 3)
    base = torch.rand(1, 3, 3, dtype=torch.float64)
    direction = torch.rand(1, 3, 3, dtype=torch.float64)

    def hvp(fn):
        x = base.clone().requires_grad_()
        (g,) = torch.autograd.grad(fn(x), x, create_graph=True)
        (h,) = torch.autograd.grad((g * direction).sum(), x)
        return h

    via_solve = hvp(
        lambda x: (
            solve(x, backend=Backend.LOG_SINKHORN, reg=0.5, n_iter=10, a=a, b=b)
            .exp()
            .mul(x)
            .sum()
        )
    )
    via_op = hvp(
        lambda x: torch.ops.transport.log_sinkhorn(x, 0.5, 10, a, b).exp().mul(x).sum()
    )
    assert torch.allclose(via_solve, via_op, atol=1e-10)
    assert torch.autograd.gradgradcheck(
        lambda x: torch.ops.transport.log_sinkhorn(x, 0.5, 10, a, b).exp().sum(),
        base.clone().requires_grad_(),
        eps=1e-6,
        atol=1e-4,
    )


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


def test_op_rejects_an_in_place_edit_of_the_mask_after_forward():
    # Every tensor the replay reads goes through save_for_backward, so an
    # edit between forward and backward is caught by the version check
    # instead of silently changing the problem being differentiated.
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    out = torch.ops.transport.log_sinkhorn(cost, 0.5, 10, a, b, mask)
    mask[0, 0, :2] = False
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.exp().sum().backward()


@pytest.mark.parametrize("op", _SINKHORN_OPS)
def test_op_backward_handles_an_empty_batch(op: str):
    # The solvers return early on a (0, N, M) batch without reading any
    # input, so the replay output carries no graph; backward must hand back
    # None rather than fail on an output with no grad_fn.
    cost = torch.empty(0, 3, 4, dtype=torch.float64, requires_grad=True)
    a, b = _uniform(3, 4, batch=0)
    other = torch.ones(2, dtype=torch.float64, requires_grad=True)
    out = _reduce(op, cost, a, b)
    assert out.requires_grad
    (out + (other * 2).sum()).backward()
    assert torch.equal(other.grad, torch.full((2,), 2.0, dtype=torch.float64))


@pytest.mark.parametrize(
    ("wanted", "expected_calls"),
    [
        (("cost",), 1),
        (("a",), 2),
        (("cost_bb",), 1),
        (("cost", "b"), 2),
        (("cost", "a", "b", "cost_aa", "cost_bb"), 3),
    ],
)
def test_divergence_op_backward_replays_only_needed_terms(
    monkeypatch, wanted: tuple[str, ...], expected_calls: int
):
    # ab, aa and bb are independent solves tied together only through the
    # marginals; a wanted input pulls in only the sub-solves it actually
    # feeds, so an unwanted term costs nothing in the backward replay.
    calls = 0
    real = torchmatch.transport.matrix._sinkhorn_divergence.log_sinkhorn_plan

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(
        torchmatch.transport.matrix._sinkhorn_divergence, "log_sinkhorn_plan", spy
    )

    a, b = _uniform(3, 3)
    tensors = {
        "cost": torch.rand(1, 3, 3, dtype=torch.float64),
        "a": a.clone(),
        "b": b.clone(),
        "cost_aa": torch.rand(1, 3, 3, dtype=torch.float64),
        "cost_bb": torch.rand(1, 3, 3, dtype=torch.float64),
    }
    for name in wanted:
        tensors[name].requires_grad_()

    out = torch.ops.transport.sinkhorn_divergence(
        tensors["cost"],
        0.5,
        10,
        tensors["a"],
        tensors["b"],
        None,
        None,
        tensors["cost_aa"],
        tensors["cost_bb"],
    ).sum()
    calls = 0  # the forward always solves all three terms; only backward is at issue
    out.backward()

    assert calls == expected_calls
    for name in wanted:
        assert tensors[name].grad is not None


def test_divergence_op_backward_handles_a_dtype_mismatched_self_cost():
    # _expand_self_cost casts cost_aa to cost's dtype before the solve; that
    # cast is differentiable, so the gradient must come back in cost_aa's
    # own (narrower) dtype, matching the solve() path exactly.
    torch.manual_seed(0)
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64)
    cost_bb = torch.rand(1, 3, 3, dtype=torch.float64)

    cost_aa_via_solve = torch.rand(1, 3, 3, dtype=torch.float32, requires_grad=True)
    solve(
        cost,
        backend=Backend.SINKHORN_DIVERGENCE,
        reg=0.5,
        n_iter=10,
        a=a,
        b=b,
        cost_aa=cost_aa_via_solve,
        cost_bb=cost_bb,
    ).sum().backward()

    cost_aa_via_op = cost_aa_via_solve.detach().clone().requires_grad_()
    torch.ops.transport.sinkhorn_divergence(
        cost, 0.5, 10, a, b, None, None, cost_aa_via_op, cost_bb
    ).sum().backward()

    assert cost_aa_via_op.grad.dtype == torch.float32
    assert torch.allclose(
        cost_aa_via_solve.grad, cost_aa_via_op.grad, atol=1e-6, rtol=1e-4
    )


def test_divergence_op_backward_handles_a_2d_self_cost_with_batch_greater_than_one():
    # _self_ot expands a 2-D self-cost to (B, N, N) via ref.size(0); the
    # gradient autograd hands back for that expand must sum over the batch
    # dim to match the (N, N) leaf, exactly as it does on the solve() path.
    torch.manual_seed(0)
    a, b = _uniform(3, 3, batch=2)
    cost = torch.rand(2, 3, 3, dtype=torch.float64)

    cost_aa_via_solve = torch.rand(3, 3, dtype=torch.float64, requires_grad=True)
    solve(
        cost,
        backend=Backend.SINKHORN_DIVERGENCE,
        reg=0.5,
        n_iter=10,
        a=a,
        b=b,
        cost_aa=cost_aa_via_solve,
    ).sum().backward()

    cost_aa_via_op = cost_aa_via_solve.detach().clone().requires_grad_()
    torch.ops.transport.sinkhorn_divergence(
        cost, 0.5, 10, a, b, None, None, cost_aa_via_op
    ).sum().backward()

    assert cost_aa_via_op.grad.shape == (3, 3)
    assert torch.allclose(cost_aa_via_solve.grad, cost_aa_via_op.grad, atol=1e-10)


def test_divergence_op_second_order_gradient_of_a_single_term_matches_solve():
    # The reduced replay's single-add sum(terms) for a one-term case (only
    # cost_bb wanted here, pulling in only the bb sub-solve) must still keep
    # create_graph's connection back to the input, the same guarantee
    # test_op_second_order_gradient_matches_the_python_path pins for
    # log_sinkhorn.
    torch.manual_seed(0)
    a, b = _uniform(3, 3)
    cost = torch.rand(1, 3, 3, dtype=torch.float64)
    base = torch.rand(1, 3, 3, dtype=torch.float64)
    direction = torch.rand(1, 3, 3, dtype=torch.float64)

    def hvp(fn):
        x = base.clone().requires_grad_()
        (g,) = torch.autograd.grad(fn(x), x, create_graph=True)
        (h,) = torch.autograd.grad((g * direction).sum(), x)
        return h

    via_solve = hvp(
        lambda x: solve(
            cost,
            backend=Backend.SINKHORN_DIVERGENCE,
            reg=0.5,
            n_iter=10,
            a=a,
            b=b,
            cost_bb=x,
        ).sum()
    )
    via_op = hvp(
        lambda x: torch.ops.transport.sinkhorn_divergence(
            cost, 0.5, 10, a, b, None, None, None, x
        ).sum()
    )
    assert torch.allclose(via_solve, via_op, atol=1e-10)
    assert torch.autograd.gradgradcheck(
        lambda x: torch.ops.transport.sinkhorn_divergence(
            cost, 0.5, 10, a, b, None, None, None, x
        ).sum(),
        base.clone().requires_grad_(),
        eps=1e-6,
        atol=1e-4,
    )


def test_op_backward_attaches_only_inputs_that_need_a_gradient(monkeypatch):
    # Only the requested inputs are differentiated through the replay, so an
    # unused marginal costs no backward work. Observed through the list the
    # replay hands to torch.autograd.grad, since a leaf without requires_grad
    # has grad None under any implementation.
    attached: list[int] = []
    real_grad = torch.autograd.grad

    def spy(outputs, inputs, *args, **kwargs):
        attached.append(len(inputs))
        return real_grad(outputs, inputs, *args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", spy)
    a, b = _uniform(3, 3)

    cost = torch.rand(1, 3, 3, dtype=torch.float64, requires_grad=True)
    torch.ops.transport.log_sinkhorn(cost, 0.5, 20, a, b).exp().sum().backward()
    assert attached == [1]

    a.requires_grad_()
    cost = cost.detach().requires_grad_()
    torch.ops.transport.log_sinkhorn(cost, 0.5, 20, a, b).exp().sum().backward()
    assert attached == [1, 2]
    assert a.grad is not None
    assert b.grad is None


@pytest.mark.parametrize("scaling", [None, 0.5])
def test_op_backward_compiles_with_and_without_eps_scaling(scaling: float | None):
    # build_eps_schedule is a tensor expression with no host read, so a
    # scaled call is traceable by AOTAutograd exactly like scaling=None: a
    # compiled caller gets the same gradient as eager either way.
    torch.manual_seed(0)
    a, b = _uniform(3, 3)
    base = torch.rand(1, 3, 3, dtype=torch.float64)

    def loss(x):
        return (
            torch.ops.transport.log_sinkhorn(x, 0.5, 5, a, b, None, scaling)
            .exp()
            .mul(x)
            .sum()
        )

    eager = base.clone().requires_grad_()
    loss(eager).backward()

    compiled = base.clone().requires_grad_()
    torch.compile(loss, backend="aot_eager", fullgraph=True)(compiled).backward()

    assert torch.allclose(eager.grad, compiled.grad, atol=1e-10)
