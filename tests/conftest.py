"""Shared pytest fixtures.

``import torchmatch`` loads the available extensions eagerly at package
init time, so no explicit ``load_*`` call is needed here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import scipy.optimize as sp
import torch
import torchmatch  # noqa: F401  (eager extension load happens at import)


@pytest.fixture
def ref_cost() -> Callable[[torch.Tensor], float]:
    """Return the optimal LAP cost of ``c`` via scipy as a float."""

    def _impl(c: torch.Tensor) -> float:
        r, col = sp.linear_sum_assignment(c.numpy())
        return float(c.numpy()[r, col].sum())

    return _impl
