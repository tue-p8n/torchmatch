"""Regression test for the CUDA-graph-capture guard in dispatch.cu.

``munkres``/``hybrid``/``lawler`` cannot carry the ``cudagraph_unsafe`` op
tag under the stable-ABI registration path used in
``sources/torchmatch/assignment/cuda/ops.cpp``: torch's stable-ABI
``StableLibrary::def()`` (and the AOTI C-shim it wraps) has no way to
attach op tags in this torch version. Without the tag, nothing upstream
would stop these ops from being captured into a CUDA graph
(``torch.cuda.graph``, ``torch.compile(mode="reduce-overhead")``) — and
each of them does a host-side ``cudaStreamSynchronize`` inside its outer
iteration loop, which is undefined behavior for the CUDA driver during
stream capture. ``jonker_dense_batch``'s CUDA backend has no such loop,
but its +inf-sentinel computation (``compute_inf_sentinel`` in
``cuda_common.cuh``) does its own host sync via
``cudaMallocAsync``/``cudaMemcpyAsync``/``cudaStreamSynchronize``/
``cudaFreeAsync``, which is equally illegal during stream capture.

``check_not_capturing`` in ``dispatch.cu`` guards each of the four ops
directly with ``cudaStreamIsCapturing()`` and raises before ever reaching
the unsafe host sync. This test exercises that guard for all four ops.

Runs in a subprocess with a hard wall-clock timeout rather than calling
``torch.cuda.graph(...)`` in-process: a regression in the guard means the
op proceeds into a host sync mid-capture, which is exactly the kind of
CUDA-driver misbehavior that can hang rather than cleanly error, and a
test that can hang the runner/CI is worse than no test at all. Under
``subprocess.run(..., timeout=...)``, a hang kills the child process
instead of the test session.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest
import torch

pytestmark = pytest.mark.timeout(90)

_SCRIPT_TEMPLATE = textwrap.dedent(
    """
    import sys

    import torch
    import torchmatch.assignment  # noqa: F401  (eager extension load)

    {setup}
    static_cost = cost.clone()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            torch.ops.assignment.{op}(static_cost)
    except RuntimeError as exc:
        message = str(exc)
        if "CUDA graph capture" in message and {op!r} in message:
            print("GUARD_RAISED_OK")
        else:
            print(f"GUARD_RAISED_WRONG_MESSAGE: {{message}}")
        sys.exit(0)
    else:
        print("GUARD_DID_NOT_RAISE")
        sys.exit(1)
    """
)

_CASES = {
    "munkres": 'cost = torch.rand(8, 8, device="cuda", dtype=torch.float32)',
    "hybrid": 'cost = torch.rand(8, 8, device="cuda", dtype=torch.float32)',
    "lawler": 'cost = torch.rand(8, 8, device="cuda", dtype=torch.float32)',
    "jonker_dense_batch": (
        'cost = torch.rand(4, 8, 8, device="cuda", dtype=torch.float32)'
    ),
}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("op", sorted(_CASES))
def test_op_rejects_cuda_graph_capture(op: str) -> None:
    script = _SCRIPT_TEMPLATE.format(setup=_CASES[op], op=op)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "GUARD_RAISED_OK" in result.stdout, combined
    assert result.returncode == 0, combined
