"""
Sinkhorn divergence.

Computes ``S_eps(a, b) = OT_eps(a, b) - 0.5 * OT_eps(a, a) - 0.5 * OT_eps(b, b)``.

Each ``OT_eps`` is evaluated via the dual-potential lower bound
``<a, f> + <b, g>`` from the converged Sinkhorn iteration.

The sign convention in this module is ``f = -eps * log_u`` and
``g = -eps * log_v`` (Sinkhorn scaling variables), so the raw dual
inner product equals **negative** OT cost.  ``sinkhorn_divergence``
compensates by negating the combination, returning the correct
non-negative ``S_eps``.

When ``cost_aa`` / ``cost_bb`` are omitted, zero-cost matrices are
used for the self-self problems.  The result is then ``OT_eps(a, b)``
rather than the textbook divergence: non-negative, but not debiased.
Pass the true geometric self-cost matrices, or use
``transport.samples.loss(x, y, debias=True)`` for the point-cloud API.

Autograd: the three OT sub-problems above (``ab``, ``aa``, ``bb``) are
independent Sinkhorn solves whose only shared inputs are the marginals
``a`` (``ab`` and ``aa``) and ``b`` (``ab`` and ``bb``); ``cost`` feeds only
``ab``, ``cost_aa`` only ``aa``, ``cost_bb`` only ``bb``. The backward
registered below uses that separability: it replays only the sub-solves a
wanted input actually touches, rather than the generic
:func:`_autograd.register_replay_autograd` formula the other two ops use,
which would replay all three regardless of which inputs need a gradient.
Since the un-replayed terms never depend on the wanted inputs, dropping
them from the graph does not change the gradient with respect to those
inputs: it is an exact reduction, not an approximation.
"""

from __future__ import annotations

from typing import Any

import torch

from torchmatch.transport.matrix._autograd import schema_layout
from torchmatch.transport.matrix._log_sinkhorn import log_sinkhorn_plan
from torchmatch.transport.matrix._validate import (
    fuse_mask_into_cost,
    validate_cost,
)

# Which named tensor inputs each of the three sub-solves depends on. Drives
# both which sub-solves the backward below must replay for a given set of
# wanted inputs, and (implicitly, via their union) which never touches an
# unwanted input at all.
_TERM_INPUTS: dict[str, frozenset[str]] = {
    "ab": frozenset({"cost", "mask", "a", "b"}),
    "aa": frozenset({"cost_aa", "a"}),
    "bb": frozenset({"cost_bb", "b"}),
}


def _ot_cost_from_potentials(
    f: torch.Tensor,
    g: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
) -> torch.Tensor:
    """``OT_eps = <a, f> + <b, g>``, batched over leading dim."""
    return (a * f).sum(dim=-1) + (b * g).sum(dim=-1)


def _expand_self_cost(
    c_self: torch.Tensor | None,
    ref: torch.Tensor,
    b: int,
    k: int,
) -> torch.Tensor:
    """Return ``c_self`` broadcast to ``(b, k, k)``, or zeros if ``None``."""
    if c_self is None:
        return torch.zeros(b, k, k, device=ref.device, dtype=ref.dtype)
    c_self = c_self.to(device=ref.device, dtype=ref.dtype)
    if c_self.ndim == 2:
        c_self = c_self.unsqueeze(0)
    return c_self.expand(b, -1, -1)


def _ab_ot(  # noqa: PLR0913
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    scaling: float | None,
) -> torch.Tensor:
    """``OT_eps(a, b)`` via the dual potentials of the ab solve."""
    _, f, g = log_sinkhorn_plan(
        cost, eps=eps, n_iter=n_iter, a=a, b=b, scaling=scaling, return_potentials=True
    )
    return _ot_cost_from_potentials(f, g, a, b)


def _self_ot(  # noqa: PLR0913
    cost_self: torch.Tensor | None,
    marginal: torch.Tensor,
    ref: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    scaling: float | None,
) -> torch.Tensor:
    """``OT_eps(marginal, marginal)`` for the aa or bb self-solve."""
    c_self = _expand_self_cost(cost_self, ref, ref.size(0), marginal.size(-1))
    _, f, g = log_sinkhorn_plan(
        c_self,
        eps=eps,
        n_iter=n_iter,
        a=marginal,
        b=marginal,
        scaling=scaling,
        return_potentials=True,
    )
    return _ot_cost_from_potentials(f, g, marginal, marginal)


def sinkhorn_divergence(  # noqa: PLR0913
    cost: torch.Tensor,
    *,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the Sinkhorn divergence ``S_eps(a, b)``.

    ``cost`` is (B, N, M); ``a`` is (B, N); ``b`` is (B, M).
    Returns a (B,) scalar tensor.

    When ``cost_aa`` and ``cost_bb`` are provided (shape ``(N, N)`` or
    ``(B, N, N)`` / ``(M, M)`` or ``(B, M, M)``), the self-self OT
    sub-problems use the supplied cost matrices, giving the textbook
    non-negative Sinkhorn divergence (Feydy 2019). When omitted, zeros
    are used (entropic-bias-only approximation, can be negative).
    """
    ot_ab = _ab_ot(cost, a, b, eps=eps, n_iter=n_iter, scaling=scaling)
    ot_aa = _self_ot(cost_aa, a, cost, eps=eps, n_iter=n_iter, scaling=scaling)
    ot_bb = _self_ot(cost_bb, b, cost, eps=eps, n_iter=n_iter, scaling=scaling)

    # _ot_cost_from_potentials returns -(true OT cost) because f = -eps*log_u
    # (sign convention from the Sinkhorn scaling variables). Negating here
    # recovers S_eps(a,b) = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b).
    return -(ot_ab - 0.5 * ot_aa - 0.5 * ot_bb)


@torch.library.custom_op("transport::sinkhorn_divergence", mutates_args=())
def _sinkhorn_divergence_op(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor:
    """``torch.ops.transport.sinkhorn_divergence`` entry point."""
    validate_cost(cost)
    if cost.ndim != 3:
        msg = (
            "torch.ops.transport.sinkhorn_divergence: cost.ndim must be 3, "
            f"got {cost.ndim}"
        )
        raise ValueError(msg)
    cost = fuse_mask_into_cost(cost, mask)
    return sinkhorn_divergence(
        cost,
        eps=eps,
        n_iter=n_iter,
        a=a,
        b=b,
        scaling=scaling,
        cost_aa=cost_aa,
        cost_bb=cost_bb,
    )


@_sinkhorn_divergence_op.register_fake
def _sinkhorn_divergence_fake(  # noqa: PLR0913
    cost: torch.Tensor,
    eps: float,
    n_iter: int,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
    scaling: float | None = None,
    cost_aa: torch.Tensor | None = None,
    cost_bb: torch.Tensor | None = None,
) -> torch.Tensor:
    del eps, n_iter, a, b, mask, scaling, cost_aa, cost_bb
    torch._check(cost.dim() == 3, lambda: "cost must be 3D (B, N, M)")
    return cost.new_empty(cost.size(0))


def _collect_wanted(
    names: list[str],
    tensor_positions: list[int],
    needs: tuple,
    saved_tensors: tuple,
) -> tuple[dict[str, torch.Tensor | None], dict[str, tuple[int, torch.Tensor]]]:
    """
    Split saved tensors into a by-name lookup and the subset that is wanted.

    A wanted entry keeps the caller's own (non-detached) tensor, so a
    gradient taken with ``create_graph`` stays connected to it; everything
    else is detached. Mirrors :func:`_autograd.register_replay_autograd`'s
    same split.
    """
    tensors: dict[str, torch.Tensor | None] = {}
    wanted: dict[str, tuple[int, torch.Tensor]] = {}
    for i, saved in zip(tensor_positions, saved_tensors, strict=True):
        name = names[i]
        if saved is None:
            tensors[name] = None
        elif i < len(needs) and needs[i] and saved.requires_grad:
            wanted[name] = (i, saved)
            tensors[name] = saved
        else:
            tensors[name] = saved.detach()
    return tensors, wanted


def _replay_needed_terms(
    tensors: dict[str, torch.Tensor | None],
    needed_terms: set[str],
    scalars: dict[str, Any],
) -> torch.Tensor:
    """Recompute the signed sum of only the sub-solves ``needed_terms`` names."""
    terms: list[torch.Tensor] = []
    if "ab" in needed_terms:
        fused = fuse_mask_into_cost(tensors["cost"], tensors.get("mask"))
        terms.append(-_ab_ot(fused, tensors["a"], tensors["b"], **scalars))
    if "aa" in needed_terms:
        aa = _self_ot(tensors.get("cost_aa"), tensors["a"], tensors["cost"], **scalars)
        terms.append(0.5 * aa)
    if "bb" in needed_terms:
        bb = _self_ot(tensors.get("cost_bb"), tensors["b"], tensors["cost"], **scalars)
        terms.append(0.5 * bb)
    return sum(terms)


def _register_divergence_autograd() -> None:
    names, tensor_positions = schema_layout("sinkhorn_divergence")

    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        del output
        ctx.save_for_backward(*(inputs[i] for i in tensor_positions))
        ctx.scalars = {
            names[i]: value
            for i, value in enumerate(inputs)
            if i not in tensor_positions
        }

    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple:
        grads: list[torch.Tensor | None] = [None] * len(names)
        tensors, wanted = _collect_wanted(
            names, tensor_positions, ctx.needs_input_grad, ctx.saved_tensors
        )
        if not wanted:
            return tuple(grads)

        needed_terms = {
            term for term, names_ in _TERM_INPUTS.items() if names_ & wanted.keys()
        }
        with torch.enable_grad():
            reduced = _replay_needed_terms(tensors, needed_terms, ctx.scalars)

        # An empty batch returns early without touching any input, so the
        # replay output carries no graph and every gradient is None.
        if not reduced.requires_grad:
            return tuple(grads)
        # allow_unused covers a wanted input the replayed terms never read
        # (e.g. cost_aa unwanted and unset while a is wanted), which
        # autograd would otherwise report as an error rather than a zero.
        wanted_tensors = [tensor for _, tensor in wanted.values()]
        found = torch.autograd.grad(
            reduced,
            wanted_tensors,
            grad_output,
            allow_unused=True,
            create_graph=torch.is_grad_enabled(),
        )
        for (i, _), grad in zip(wanted.values(), found, strict=True):
            grads[i] = grad
        return tuple(grads)

    torch.library.register_autograd(
        "transport::sinkhorn_divergence", backward, setup_context=setup_context
    )


_register_divergence_autograd()
