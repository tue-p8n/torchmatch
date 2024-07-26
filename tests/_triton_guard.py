"""
Skip predicate for CUDA + Triton-dependent samples-face tests.

In the Nix dev shell, ``torch.cuda.is_available()`` returns True but
Triton's gcc launcher receives ``-L<file>`` instead of ``-L<dir>`` for
``libcuda.so.1``, so the first kernel compile blows up at link time.
Trigger a tiny no-op kernel and skip when the linker fails. The first
launch is cached, so the cost is one-time per session.
"""

from __future__ import annotations

import functools

import torch


@functools.lru_cache(maxsize=1)
def triton_kernel_compiles() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        import triton
        import triton.language as tl

        @triton.jit
        def _noop(out_ptr):
            tl.store(out_ptr, 0.0)

        out = torch.zeros(1, device="cuda")
        _noop[(1,)](out)
        torch.cuda.synchronize()
    except Exception:
        return False
    return True
