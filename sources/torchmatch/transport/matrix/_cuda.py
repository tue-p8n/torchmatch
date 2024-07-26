"""Loader for the transport CUDA extension."""

from __future__ import annotations

import functools

import torch

from torchmatch.transport._loader import find_prebuilt

__all__ = ["load_cuda"]


def _register_fakes() -> None:
    # See _cpu.py; the same custom_op decorator covers CUDA via the meta
    # surface (torch.logsumexp dispatches to both backends transparently).
    from torchmatch.transport.matrix import (  # noqa: F401, PLC0415
        _log_sinkhorn,
        _sinkhorn_divergence,
        _unbalanced_sinkhorn,
    )


@functools.cache
def load_cuda() -> None:
    if (path := find_prebuilt("_transport_cuda_impl")) is not None:
        torch.ops.load_library(str(path))
    _register_fakes()
