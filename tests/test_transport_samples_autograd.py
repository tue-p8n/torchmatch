"""torch.compile composition test for transport.samples.loss."""

from __future__ import annotations

import pytest
import torch
from _triton_guard import triton_kernel_compiles

cuda_required = pytest.mark.skipif(
    not triton_kernel_compiles(),
    reason="CUDA + functional Triton launcher required",
)


@cuda_required
def test_torchmatch_loss_composes_with_compile():
    from torchmatch.transport.samples import loss

    torch.manual_seed(1)
    x = torch.randn(64, 8, device="cuda")
    y = torch.randn(64, 8, device="cuda")

    @torch.compile(fullgraph=True)
    def fn(x, y):
        return loss(x, y, blur=0.1)

    out = fn(x, y)
    assert out.shape == ()
