"""
Python wrapper around the network-simplex EMD CPU op.

The C++ implementation registers an impl for CPU only. The samples
face does not cover exact EMD (the samples face is entropic-only).

The dense network simplex has no special-case for +inf costs; it
will happily pivot on an infinite edge and return wrong assignments.
We reject non-finite costs at the entry point; callers that need
forbidden edges with EXACT_EMD must drop or sparsify those rows first.
"""

from __future__ import annotations

import torch

from torchmatch.transport.matrix._validate import validate_cost


def exact_emd(
    cost: torch.Tensor,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the exact Earth Mover's Distance transport plan (CPU only).

    Wraps ``torch.ops.transport.exact_emd``, which uses the network-simplex
    algorithm. ``+inf`` entries are **not** supported: the network simplex
    will pivot on infinite-cost arcs and return wrong results. Use
    ``LOG_SINKHORN`` for problems with forbidden edges.

    Parameters
    ----------
    cost
        Cost matrix (B, N, M) or (N, M), float32 or float64, CPU only.
    a
        Source marginal weights (B, N) or (N,).
    b
        Target marginal weights (B, M) or (M,).
    mask
        Boolean mask (B, N, M) or (N, M). False entries are set to ``+inf``
        before the solver, which raises an error (see above).

    Returns
    -------
    plan
        Transport plan of the same shape and dtype as ``cost``.

    Raises
    ------
    ValueError
        If ``cost`` is not on CPU, contains ``+inf``, or the C++ extension
        is unavailable.

    """
    validate_cost(cost)
    if cost.device.type != "cpu":
        msg = (
            "torchmatch.transport.matrix.solve: EXACT_EMD is CPU-only "
            f"(got device {cost.device})"
        )
        raise ValueError(msg)
    if torch.isinf(cost).any():
        msg = (
            "torchmatch.transport.matrix.solve: EXACT_EMD does not support "
            "+inf (forbidden-edge) costs. The dense network simplex does "
            "not special-case +inf and would return wrong pivots. Use "
            "LOG_SINKHORN, or drop the forbidden rows before calling."
        )
        raise ValueError(msg)
    op = getattr(torch.ops.transport, "exact_emd", None)
    if op is None:
        msg = (
            "torchmatch.transport.matrix.solve: EXACT_EMD requires the "
            "transport CPU extension. Build without TORCHMATCH_SKIP_TRANSPORT, "
            "or rebuild on a host with a C++ toolchain for JIT fallback."
        )
        raise RuntimeError(msg)
    return op(cost, mask, a, b)


def _register_fake() -> None:
    """Register the FakeTensor kernel; called only when the schema exists."""

    @torch.library.register_fake("transport::exact_emd")
    def _exact_emd_fake(
        cost: torch.Tensor,
        mask: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        b: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mask, a, b
        torch._check(cost.dim() == 3, lambda: "cost must be 3D (B, N, M)")
        return cost.new_empty(cost.size(0), cost.size(1), cost.size(2))
