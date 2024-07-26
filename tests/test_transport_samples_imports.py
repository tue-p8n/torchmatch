"""Smoke test for the transport.samples module imports."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "torchmatch.transport.samples",
        "torchmatch.transport.samples.kernels",
        "torchmatch.transport.samples.kernels._common",
        "torchmatch.transport.samples.kernels._triton_helpers",
        "torchmatch.transport.samples.kernels.streaming_sqeuclid",
        "torchmatch.transport.samples.kernels.grad_sqeuclid",
        "torchmatch.transport.samples.kernels.apply_shifted",
    ],
)
def test_import(module):
    importlib.import_module(module)


@pytest.mark.parametrize(
    "module",
    [
        "torchmatch.transport.samples._solvers",
        "torchmatch.transport.samples._c_transform",
        "torchmatch.transport.samples._cg",
        "torchmatch.transport.samples._hvp",
        "torchmatch.transport.samples._implicit_grad",
    ],
)
def test_solver_module_imports(module):
    importlib.import_module(module)
