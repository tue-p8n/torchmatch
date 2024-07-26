"""
User-facing entry point for the samples-face Sinkhorn.

Multi-mode dispatcher: balanced / debiased / unbalanced / semi-unbalanced
all flow through this single Python function. The torch.compile-friendly
underlying ops are registered in :mod:`_autograd`.

Squared-Euclidean cost only. Requires CUDA + Triton.
"""

from __future__ import annotations

import torch

# Importing _autograd registers the underlying custom_op + autograd.
from torchmatch.transport.samples import _autograd  # noqa: F401


def _fwd(
    x: torch.Tensor,
    y: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    blur: float,
    rho_x: float,
    rho_y: float,
    half_cost: bool,
    threshold: float,
    scaling: float,
) -> torch.Tensor:
    # use_symmetric=True picks the symmetric solver variant. n_iter is
    # unused in that path (the solver runs an eps schedule); see loss(...)
    # which rejects non-default n_iter at the entry point.
    loss_value, _f, _g = torch.ops.transport._sinkhorn_samples_fwd(
        x,
        y,
        a,
        b,
        blur,
        0,
        True,
        rho_x,
        rho_y,
        half_cost,
        threshold,
        scaling,
    )
    return loss_value


def loss(  # noqa: PLR0913
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    blur: float = 0.05,
    debias: bool = False,
    reach: float | None = None,
    reach_x: float | None = None,
    reach_y: float | None = None,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    scaling: float = 0.5,
    n_iter: int | None = None,
    threshold: float | None = None,
    half_cost: bool = False,
    p: int = 2,
) -> torch.Tensor:
    """Scalar OT cost / divergence over point clouds (CUDA only).

    Parameters
    ----------
    x
        Source point cloud (n, d), float32, CUDA.
    y
        Target point cloud (m, d), float32, CUDA.
    blur
        Bandwidth parameter; the entropy regularization is ``eps = blur**2``.
    debias
        If True, returns the Sinkhorn divergence
        ``S_eps(x,y) - 0.5*S_eps(x,x) - 0.5*S_eps(y,y)`` which vanishes
        when ``x == y`` (unbiased). Requires three Sinkhorn solves.
    reach
        Unbalanced OT: KL marginal penalty ``rho = reach**2`` applied to
        both source and target. ``None`` = balanced OT.
    reach_x
        Semi-unbalanced: KL penalty for the source marginal only.
    reach_y
        Semi-unbalanced: KL penalty for the target marginal only.
    a
        Source weights (n,). Uniform if None.
    b
        Target weights (m,). Uniform if None.
    scaling
        Geometric decay factor for the epsilon schedule, in ``(0, 1)``.
        Smaller values converge faster but may be less numerically stable.
    n_iter
        Not supported; always raises. Control iterations via ``scaling``.
    threshold
        Early-stopping threshold on potential change. Incompatible with
        ``torch.compile`` (forces a host sync each check).
    half_cost
        If True, uses ``cost = 0.5 * ||x - y||²`` instead of ``||x - y||²``.
    p
        Cost exponent. Only ``p=2`` is supported.

    Returns
    -------
    loss
        Scalar OT cost (shape ``()`` for 2-D input; ``(B,)`` for 3-D batch).

    Notes
    -----
    Gradients w.r.t. both ``x`` and ``y`` are computed via the analytic
    online backward registered for ``transport::_sinkhorn_samples_fwd``.
    The ``NotImplementedError`` for ``grad_y`` in ``_c_transform`` is an
    internal building-block restriction and does not affect this entry point.

    Batched input ``x (B, N, D)`` and ``y (B, M, D)`` is supported: the
    function loops over the batch dimension and stacks the results. A native
    batched kernel path is planned for a future release.

    """
    if x.ndim == 3:
        if y.ndim != 3:
            msg = (
                "torchmatch.transport.samples.loss: x and y must both be "
                "3-D for batched input"
            )
            raise ValueError(msg)
        batch = x.size(0)
        if a is None:
            a_list: list[torch.Tensor | None] = [None] * batch
        elif a.ndim == 1:
            a_list = [a] * batch
        elif a.ndim == 2:
            if a.size(0) != batch:
                msg = (
                    f"torchmatch.transport.samples.loss: a must have first "
                    f"dimension {batch} (batch size), got {a.size(0)}"
                )
                raise ValueError(msg)
            a_list = [a[i] for i in range(batch)]
        else:
            msg = (
                f"torchmatch.transport.samples.loss: a must be 1-D or 2-D "
                f"for batched input, got {a.ndim}-D"
            )
            raise ValueError(msg)
        if b is None:
            b_list: list[torch.Tensor | None] = [None] * batch
        elif b.ndim == 1:
            b_list = [b] * batch
        elif b.ndim == 2:
            if b.size(0) != batch:
                msg = (
                    f"torchmatch.transport.samples.loss: b must have first "
                    f"dimension {batch} (batch size), got {b.size(0)}"
                )
                raise ValueError(msg)
            b_list = [b[i] for i in range(batch)]
        else:
            msg = (
                f"torchmatch.transport.samples.loss: b must be 1-D or 2-D "
                f"for batched input, got {b.ndim}-D"
            )
            raise ValueError(msg)
        return torch.stack(
            [
                loss(
                    x[i],
                    y[i],
                    blur=blur,
                    debias=debias,
                    reach=reach,
                    reach_x=reach_x,
                    reach_y=reach_y,
                    a=a_list[i],
                    b=b_list[i],
                    scaling=scaling,
                    threshold=threshold,
                    half_cost=half_cost,
                    p=p,
                )
                for i in range(batch)
            ]
        )

    if p != 2:
        msg = (
            "torchmatch.transport.samples.loss: p=2 is the only supported "
            f"cost exponent (got p={p})"
        )
        raise ValueError(msg)

    if blur <= 0:
        msg = (
            "torchmatch.transport.samples.loss: blur must be positive "
            f"(eps = blur**2; got blur={blur})"
        )
        raise ValueError(msg)

    if not (0.0 < scaling < 1.0):
        msg = (
            "torchmatch.transport.samples.loss: scaling must be in (0, 1), "
            f"got {scaling}"
        )
        raise ValueError(msg)

    # The symmetric solver runs its own eps schedule; a user-supplied
    # n_iter has no effect. Fail loudly rather than silently ignore.
    if n_iter is not None:
        msg = (
            "torchmatch.transport.samples.loss: n_iter is not configurable "
            "for the symmetric solver (which runs an eps schedule). Tune via "
            "scaling instead."
        )
        raise ValueError(msg)

    if not (x.is_cuda and y.is_cuda):
        msg = (
            "torchmatch.transport.samples.loss: requires CUDA tensors "
            "(the Triton kernels are CUDA-only)"
        )
        raise RuntimeError(msg)

    n = x.size(0)
    m = y.size(0)

    if a is None:
        a = torch.full((n,), 1.0 / n, device=x.device, dtype=x.dtype)
    if b is None:
        b = torch.full((m,), 1.0 / m, device=y.device, dtype=y.dtype)

    if reach is not None and (reach_x is not None or reach_y is not None):
        msg = (
            "torchmatch.transport.samples.loss: specify either reach, or "
            "reach_x/reach_y, not both"
        )
        raise ValueError(msg)
    if reach is not None:
        rho_x = reach**2
        rho_y = reach**2
    else:
        rho_x = (reach_x**2) if reach_x is not None else 0.0
        rho_y = (reach_y**2) if reach_y is not None else 0.0

    threshold_arg = threshold if threshold is not None else 0.0

    fwd_kwargs = {
        "blur": blur,
        "rho_x": rho_x,
        "rho_y": rho_y,
        "half_cost": half_cost,
        "threshold": threshold_arg,
        "scaling": scaling,
    }

    loss_xy = _fwd(x, y, a, b, **fwd_kwargs)
    if not debias:
        return loss_xy
    loss_xx = _fwd(x, x, a, a, **fwd_kwargs)
    loss_yy = _fwd(y, y, b, b, **fwd_kwargs)
    return loss_xy - 0.5 * loss_xx - 0.5 * loss_yy
