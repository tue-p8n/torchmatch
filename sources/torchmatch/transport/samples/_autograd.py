"""
Autograd registration for the samples-face Sinkhorn forward.

Wraps the samples-face Sinkhorn forward in a torch.library.custom_op
with register_autograd so it composes with torch.compile / torch.export
out of the box.

The forward call returns (loss, f_potential, g_potential); the
backward uses the saved potentials to recover gradients via the
analytic-gradient path from samples._implicit_grad.
"""

from __future__ import annotations

import torch


# rho / threshold use 0.0 as the "unset" sentinel because torch.library
# op signatures cannot carry Optional[float] without a custom converter.
@torch.library.custom_op(
    "transport::_sinkhorn_samples_fwd",
    mutates_args=(),
)
def _sinkhorn_samples_fwd(  # noqa: PLR0913
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    blur: float,
    n_iter: int,
    use_symmetric: bool,
    rho_x: float,
    rho_y: float,
    half_cost: bool,
    threshold: float,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Lazy import so that loading this module does not trigger Triton
    # autotune setup or CUDA-only side effects at package import time.
    from torchmatch.transport.samples._solvers import (
        sinkhorn_alternating,
        sinkhorn_symmetric,
    )

    cost_scale = 0.5 if half_cost else 1.0
    rho_x_arg = None if rho_x == 0.0 else rho_x
    rho_y_arg = None if rho_y == 0.0 else rho_y
    threshold_arg = None if threshold == 0.0 else threshold

    if use_symmetric:
        f, g = sinkhorn_symmetric(
            x,
            y,
            a,
            b,
            blur=blur,
            scaling=scaling,
            cost_scale=cost_scale,
            rho_x=rho_x_arg,
            rho_y=rho_y_arg,
            threshold=threshold_arg,
        )
    else:
        eps = blur * blur
        f, g = sinkhorn_alternating(
            x,
            y,
            a,
            b,
            eps=eps,
            n_iters=n_iter,
            cost_scale=cost_scale,
            rho_x=rho_x_arg,
            rho_y=rho_y_arg,
            threshold=threshold_arg,
        )

    # Each side contributes (rho + eps/2) * sum(a * (1 - exp(-f/rho))) when
    # its rho is set (KL-relaxed; Chizat 2018) and <a, f> otherwise (strict
    # marginal). The two sides combine independently so the semi-unbalanced
    # case (one rho set) gets the correct asymmetric reading.
    eps_final = blur * blur
    if rho_x_arg is not None:
        loss_x = (rho_x_arg + 0.5 * eps_final) * (
            a * (1 - (-f / rho_x_arg).exp())
        ).sum()
    else:
        loss_x = (a * f).sum()
    if rho_y_arg is not None:
        loss_y = (rho_y_arg + 0.5 * eps_final) * (
            b * (1 - (-g / rho_y_arg).exp())
        ).sum()
    else:
        loss_y = (b * g).sum()
    return loss_x + loss_y, f, g


@_sinkhorn_samples_fwd.register_fake
def _fake(  # noqa: PLR0913
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    blur: float,
    n_iter: int,
    use_symmetric: bool,
    rho_x: float,
    rho_y: float,
    half_cost: bool,
    threshold: float,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del a, b, blur, n_iter, use_symmetric, rho_x, rho_y
    del half_cost, threshold, scaling
    return (
        x.new_empty(()),
        x.new_empty(x.size(0)),
        y.new_empty(y.size(0)),
    )


def _setup_context(ctx, inputs, output):
    x, y, a, b = inputs[:4]
    _loss, f, g = output
    ctx.save_for_backward(x, y, a, b, f, g)
    # blur and half_cost reconstruct the eps and cost_scale used in the
    # forward; the analytic-gradient kernel needs both to retile the
    # squared-Euclidean cost matrix on-the-fly.
    ctx.blur = inputs[4]
    ctx.cost_scale = 0.5 if inputs[9] else 1.0


def _backward(ctx, grad_loss, grad_f, grad_g):
    # f and g are intermediate potentials; only the scalar loss is
    # differentiable. grad_loss is None when the user backs propagated
    # through f or g directly, which would feed a None grad_scale to
    # the Triton kernel — fail loudly instead.
    if grad_loss is None:
        msg = (
            "torchmatch.transport.samples: backward is only defined for the "
            "scalar loss output; f and g are intermediate potentials"
        )
        raise RuntimeError(msg)

    from torchmatch.transport.samples.kernels.grad_sqeuclid import (
        sinkhorn_online_grad_sqeuclid,
    )

    del grad_f, grad_g

    x, y, a, b, f, g = ctx.saved_tensors
    eps = float(ctx.blur) * float(ctx.blur)
    cost_scale = float(ctx.cost_scale)

    # Analytic-gradient backward: treat the converged potentials as
    # constants (Danskin's envelope), recompute the transport plan
    # tile-by-tile, and form dL/dx, dL/dy in one pass. The parity test
    # is calibrated against this kernel, not against the IFT path.
    grad_x, grad_y = sinkhorn_online_grad_sqeuclid(
        x,
        y,
        a,
        b,
        f,
        g,
        eps=eps,
        cost_scale=cost_scale,
        grad_scale=grad_loss,
        compute_grad_x=True,
        compute_grad_y=True,
        allow_tf32=False,
    )

    return (
        grad_x,
        grad_y,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


torch.library.register_autograd(
    "transport::_sinkhorn_samples_fwd",
    _backward,
    setup_context=_setup_context,
)
