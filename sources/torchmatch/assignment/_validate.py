"""
Tracing-safe NaN / ``-inf`` rejection shared within the assignment package.

Branching on a reduced tensor is not merely slow under tracing, it is
either impossible or wrong: under Dynamo the branch is a graph break;
under ``torch.jit.trace`` the taken branch is burned into the trace as if
it held for every future input; under ``make_fx`` and on a fake tensor
there is no value to read, so the branch raises outright.
:func:`skip_value_checks` says when to omit it, and :func:`check_finite`
does the check itself.

This mirrors ``torchmatch.transport.matrix._validate``, duplicated rather
than imported across the package boundary: each sub-package stays
importable on its own, the way ``_loader.py`` is duplicated for the same
reason (see CLAUDE.md).

Callers of :func:`check_finite` are the ergonomic entry points
(:func:`torchmatch.assignment.solve`, ``torch.ops.assignment.greedy``,
``auction_assignment``): plain Python functions or bodies of ops without a
fake-tensor-routed forward, so tracing into them directly is possible
and, for ``solve`` and ``auction_assignment`` in particular, expected.
"""

from __future__ import annotations

import math

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def skip_value_checks(tensor: torch.Tensor) -> bool:
    """Report whether ``tensor``'s value checks must be skipped under tracing."""
    return (
        torch.compiler.is_compiling()
        or torch.jit.is_tracing()
        or get_proxy_mode() is not None
        or is_fake(tensor)
    )


def check_finite(cost: torch.Tensor) -> str | None:
    """
    Return ``"NaN"`` or ``"-inf"`` if ``cost`` contains one, else ``None``.

    Skipped (returns ``None`` without reading a value) when
    :func:`skip_value_checks` holds, or on an empty tensor. One reduction
    and one host read answers both questions: min propagates NaN, and
    -inf is the minimum whenever it is present. +inf never wins a min, so
    a forbidden edge passes.
    """
    if skip_value_checks(cost) or cost.numel() == 0:
        return None
    lowest = cost.min().item()
    if math.isnan(lowest):
        return "NaN"
    if lowest == -math.inf:
        return "-inf"
    return None
