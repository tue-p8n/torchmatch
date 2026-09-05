"""
Autograd registration for the cost-matrix Sinkhorn ops.

``torch.ops.transport.log_sinkhorn`` and its two siblings are
``torch.library.custom_op`` registrations, which are opaque to autograd
until a formula is registered. Without one they still build a graph node
whose output reports ``requires_grad=True``, and the failure only surfaces
at backward, as "no autograd formula was registered". That is why
:mod:`._solve` calls the Python iteration functions directly.

The formula here is the derivative of the iteration as written, obtained
by replaying it. ``setup_context`` saves the inputs; ``backward`` re-runs
the same solver under ``enable_grad`` on detached copies and takes the
vector-Jacobian product through the replay. This is gradient checkpointing
of the Sinkhorn loop: identical by construction to differentiating the
unrolled iteration, which is the semantics ``solve()`` already has and its
gradcheck tests already pin, while storing one cost matrix instead of the
intermediates of every iteration.

The alternative is implicit differentiation of the Sinkhorn fixed point,
which the samples face uses. It is not interchangeable here. It returns
the derivative of the converged solution, whereas these ops return the
result of exactly ``n_iter`` steps and are routinely called at iteration
counts far short of convergence. Differentiating something the forward did
not compute would make gradcheck fail at small ``n_iter``, correctly.

Replaying costs one extra forward per backward. The saved memory is the
whole iteration history, which at the iteration counts these solvers run
at is the larger quantity.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from torchmatch.transport.matrix._log_sinkhorn import log_sinkhorn_plan
from torchmatch.transport.matrix._sinkhorn_divergence import sinkhorn_divergence
from torchmatch.transport.matrix._unbalanced_sinkhorn import (
    unbalanced_sinkhorn_plan,
)
from torchmatch.transport.matrix._validate import fuse_mask_into_cost

__all__ = ["register_matrix_autograd"]

# Positions of (cost, a, b) in each op's argument tuple. They are the only
# floating-point inputs; eps, n_iter, rho, scaling are Python scalars and
# mask is boolean, none of which take a gradient.
_DIFFERENTIABLE: dict[str, tuple[int, int, int]] = {
    "log_sinkhorn": (0, 3, 4),
    "unbalanced_sinkhorn": (0, 4, 5),
    "sinkhorn_divergence": (0, 3, 4),
}


def _replay_vjp(
    replay: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    saved: Sequence[torch.Tensor],
    needs_grad: Sequence[bool],
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor | None, ...]:
    """
    Re-run ``replay`` on detached copies and take its vector-Jacobian product.

    ``saved`` is ``(cost, a, b)`` as the forward received them and
    ``needs_grad`` is the matching slice of ``ctx.needs_input_grad``. Only
    the inputs that need one are attached, so an unused marginal costs no
    backward work. Returns gradients positionally for ``(cost, a, b)``,
    with ``None`` wherever one was not requested.
    """
    with torch.enable_grad():
        attached = [
            tensor.detach().requires_grad_(requires_grad=want)
            for tensor, want in zip(saved, needs_grad, strict=True)
        ]
        output = replay(*attached)

    wanted = [tensor for tensor, want in zip(attached, needs_grad, strict=True) if want]
    if not wanted:
        return (None, None, None)

    # allow_unused: a marginal is genuinely absent from the graph for an
    # empty batch, where the solver returns early without reading it.
    grads = torch.autograd.grad(
        output,
        wanted,
        grad_output,
        allow_unused=True,
    )
    supplied = iter(grads)
    return tuple(next(supplied) if want else None for want in needs_grad)


def _expand(
    grads: tuple[torch.Tensor | None, ...],
    positions: tuple[int, int, int],
    arity: int,
) -> tuple[torch.Tensor | None, ...]:
    """Scatter the (cost, a, b) gradients into a full-arity result tuple."""
    out: list[torch.Tensor | None] = [None] * arity
    for grad, position in zip(grads, positions, strict=True):
        out[position] = grad
    return tuple(out)


def _register_log_sinkhorn() -> None:
    positions = _DIFFERENTIABLE["log_sinkhorn"]

    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        del output
        cost, eps, n_iter, a, b, mask, scaling = inputs
        ctx.save_for_backward(cost, a, b)
        ctx.mask, ctx.eps, ctx.n_iter, ctx.scaling = mask, eps, n_iter, scaling

    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        def replay(
            cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            return log_sinkhorn_plan(
                fuse_mask_into_cost(cost, ctx.mask),
                eps=ctx.eps,
                n_iter=ctx.n_iter,
                a=a,
                b=b,
                scaling=ctx.scaling,
            )

        grads = _replay_vjp(
            replay,
            ctx.saved_tensors,
            [ctx.needs_input_grad[i] for i in positions],
            grad_output,
        )
        return _expand(grads, positions, arity=7)

    torch.library.register_autograd(
        "transport::log_sinkhorn",
        backward,
        setup_context=setup_context,
    )


def _register_unbalanced_sinkhorn() -> None:
    positions = _DIFFERENTIABLE["unbalanced_sinkhorn"]

    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        del output
        cost, eps, n_iter, rho, a, b, mask, scaling = inputs
        ctx.save_for_backward(cost, a, b)
        ctx.mask, ctx.eps, ctx.n_iter = mask, eps, n_iter
        ctx.rho, ctx.scaling = rho, scaling

    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        def replay(
            cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            return unbalanced_sinkhorn_plan(
                fuse_mask_into_cost(cost, ctx.mask),
                eps=ctx.eps,
                n_iter=ctx.n_iter,
                rho=ctx.rho,
                a=a,
                b=b,
                scaling=ctx.scaling,
            )

        grads = _replay_vjp(
            replay,
            ctx.saved_tensors,
            [ctx.needs_input_grad[i] for i in positions],
            grad_output,
        )
        return _expand(grads, positions, arity=8)

    torch.library.register_autograd(
        "transport::unbalanced_sinkhorn",
        backward,
        setup_context=setup_context,
    )


def _register_sinkhorn_divergence() -> None:
    positions = _DIFFERENTIABLE["sinkhorn_divergence"]

    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        del output
        cost, eps, n_iter, a, b, mask, scaling, cost_aa, cost_bb = inputs
        ctx.save_for_backward(cost, a, b)
        ctx.mask, ctx.eps, ctx.n_iter, ctx.scaling = mask, eps, n_iter, scaling
        # The self-cost matrices are inputs to the debiasing terms but carry
        # no gradient of their own: they are fixed geometry, not predictions.
        ctx.cost_aa, ctx.cost_bb = cost_aa, cost_bb

    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        def replay(
            cost: torch.Tensor, a: torch.Tensor, b: torch.Tensor
        ) -> torch.Tensor:
            return sinkhorn_divergence(
                fuse_mask_into_cost(cost, ctx.mask),
                eps=ctx.eps,
                n_iter=ctx.n_iter,
                a=a,
                b=b,
                scaling=ctx.scaling,
                cost_aa=ctx.cost_aa,
                cost_bb=ctx.cost_bb,
            )

        grads = _replay_vjp(
            replay,
            ctx.saved_tensors,
            [ctx.needs_input_grad[i] for i in positions],
            grad_output,
        )
        return _expand(grads, positions, arity=9)

    torch.library.register_autograd(
        "transport::sinkhorn_divergence",
        backward,
        setup_context=setup_context,
    )


def register_matrix_autograd() -> None:
    """
    Attach the replay formula to all three Sinkhorn matrix ops.

    Called once at package import. ``exact_emd`` is excluded: it is a
    network simplex whose output is piecewise constant in the cost, so
    replaying it yields no gradient rather than a wrong one.
    """
    _register_log_sinkhorn()
    _register_unbalanced_sinkhorn()
    _register_sinkhorn_divergence()
