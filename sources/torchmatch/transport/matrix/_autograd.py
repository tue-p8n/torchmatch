"""
Autograd registration for the cost-matrix Sinkhorn ops.

``torch.ops.transport.log_sinkhorn`` and its two siblings are
``torch.library.custom_op`` registrations, which are opaque to autograd
until a formula is registered. Without one they still build a graph node
whose output reports ``requires_grad=True``, and the failure only surfaces
at backward, as "no autograd formula was registered".

The formula here is the derivative of the iteration as written, obtained
by replaying it. ``setup_context`` saves the tensor inputs; ``backward``
re-runs the same solver under ``enable_grad`` and takes the
vector-Jacobian product through the replay. This is gradient checkpointing
of the Sinkhorn loop: identical by construction to differentiating the
unrolled iteration, which is the semantics ``solve()`` already has and its
gradcheck tests already pin, while storing the inputs instead of the
intermediates of every iteration. Inputs that need a gradient enter the
replay as the caller's own tensors rather than detached copies, so a
gradient taken with ``create_graph`` stays connected to them and
second-order use (gradient penalties, Hessian-vector products) works.

The alternative is implicit differentiation of the Sinkhorn fixed point,
which the samples face uses. It is not interchangeable here. It returns
the derivative of the converged solution, whereas these ops return the
result of exactly ``n_iter`` steps and are routinely called at iteration
counts far short of convergence. Differentiating something the forward did
not compute would make gradcheck fail at small ``n_iter``, correctly.

Replaying costs one extra forward per backward. The saved memory is the
whole iteration history, which at the iteration counts these solvers run
at is the larger quantity.

Argument names, positions and which inputs are tensors all come from the
op schema, so one registration serves every op, and each op module
attaches its own formula right after defining the op. Every tensor input
is differentiable, including the self-cost matrices of the divergence: in
the debiased setting they are functions of the same predictions as the
cost, and their gradient is exactly what debiasing contributes.

The eps schedule the replayed solver rebuilds (:func:`_schedule.build_eps_schedule`)
is itself a tensor expression with no host read, so a scaled call is
compile-traceable exactly like ``scaling=None``.

``sinkhorn_divergence`` does not use this generic replay: its own module
registers a formula that replays only the ab / aa / bb sub-solve each
wanted input actually touches, since the three are independent OT
problems tied together only by the marginals. See its docstring.

``exact_emd`` gets no formula: its output is piecewise constant in the
cost, so replaying it yields no gradient rather than a wrong one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from torchmatch.transport.matrix._validate import fuse_mask_into_cost

__all__ = ["register_replay_autograd", "schema_layout"]


def _is_tensor_arg(arg_type: Any) -> bool:
    if isinstance(arg_type, torch.OptionalType):
        arg_type = arg_type.getElementType()
    return isinstance(arg_type, torch.TensorType)


def schema_layout(op_name: str) -> tuple[list[str], list[int]]:
    """
    Return ``(argument names, tensor-argument positions)`` for an op's schema.

    Shared by :func:`register_replay_autograd` and any op module that
    registers its own specialized formula (e.g. ``sinkhorn_divergence``),
    so argument names and positions come from one place instead of a
    hand-maintained table per registration.
    """
    schema = getattr(torch.ops.transport, op_name).default._schema
    names = [arg.name for arg in schema.arguments]
    tensor_positions = [
        i for i, arg in enumerate(schema.arguments) if _is_tensor_arg(arg.type)
    ]
    return names, tensor_positions


def register_replay_autograd(op_name: str, solver: Callable[..., torch.Tensor]) -> None:
    """
    Attach the replay formula to ``torch.ops.transport.<op_name>``.

    ``solver`` is the op's Python iteration function. The replay calls it
    with the op's own arguments rebuilt by name from the schema, after
    fusing ``mask`` into ``cost`` the way the op body does.
    """
    names, tensor_positions = schema_layout(op_name)

    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        del output
        # save_for_backward accepts None, so the optional tensors go through
        # it too and get the in-place-modification check like the rest.
        ctx.save_for_backward(*(inputs[i] for i in tensor_positions))
        ctx.scalars = {
            names[i]: value
            for i, value in enumerate(inputs)
            if i not in tensor_positions
        }

    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        # needs_input_grad follows the argument list as the dispatcher passed
        # it, which drops trailing defaults, so it can be shorter than the
        # schema; anything past its end was a default and takes no gradient.
        needs = ctx.needs_input_grad
        grads: list[torch.Tensor | None] = [None] * len(names)
        tensors: dict[str, torch.Tensor | None] = {}
        wanted: list[tuple[int, torch.Tensor]] = []
        for i, saved in zip(tensor_positions, ctx.saved_tensors, strict=True):
            if saved is None:
                pass
            elif i < len(needs) and needs[i] and saved.requires_grad:
                # The caller's tensor, not a detached copy: a gradient taken
                # with create_graph then stays connected to it.
                wanted.append((i, saved))
            else:
                saved = saved.detach()
            tensors[names[i]] = saved
        if not wanted:
            return tuple(grads)

        with torch.enable_grad():
            cost = fuse_mask_into_cost(tensors.pop("cost"), tensors.pop("mask", None))
            output = solver(cost, **tensors, **ctx.scalars)

        # An empty batch returns early without touching any input, so the
        # replay output carries no graph and every gradient is None.
        if not output.requires_grad:
            return tuple(grads)
        # allow_unused covers a wanted input the solver never reads, which
        # autograd would otherwise report as an error rather than a zero.
        found = torch.autograd.grad(
            output,
            [tensor for _, tensor in wanted],
            grad_output,
            allow_unused=True,
            create_graph=torch.is_grad_enabled(),
        )
        for (i, _), grad in zip(wanted, found, strict=True):
            grads[i] = grad
        return tuple(grads)

    torch.library.register_autograd(
        f"transport::{op_name}", backward, setup_context=setup_context
    )
