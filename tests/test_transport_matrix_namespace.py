"""Sanity tests for the relocated transport.matrix namespace."""

from __future__ import annotations

import torchmatch


def test_matrix_subpackage_imports():
    assert hasattr(torchmatch.transport, "matrix")
    assert hasattr(torchmatch.transport.matrix, "solve")
    assert hasattr(torchmatch.transport.matrix, "Backend")
    assert hasattr(torchmatch.transport.matrix, "ops")


def test_top_level_solve_is_gone():
    assert not hasattr(torchmatch.transport, "solve")
    assert not hasattr(torchmatch.transport, "Backend")
