"""
Tracing-safety tests for the assignment package's NaN / -inf rejection.

Mirrors ``tests/test_transport_validate.py``'s tracing-guard coverage: the
same class of defect (a reduced tensor branched on in a plain Python
function) exists at ``solve()``'s entry and in ``auction_assignment``, both
of which run their own checks and cannot rely on an op-level FakeTensor
kernel to route around them the way ``greedy`` (a ``custom_op``) can.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torchmatch.assignment import auction_assignment, solve
from torchmatch.assignment._validate import check_finite, skip_value_checks


def test_check_finite_runs_in_ordinary_eager_use():
    assert skip_value_checks(torch.zeros(3, 3)) is False


def test_check_finite_detects_nan():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("nan")
    assert check_finite(cost) == "NaN"


def test_check_finite_detects_neg_inf():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("-inf")
    assert check_finite(cost) == "-inf"


def test_check_finite_accepts_pos_inf():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("inf")
    assert check_finite(cost) is None


def test_check_finite_skips_on_an_empty_tensor():
    assert check_finite(torch.empty(0, 3)) is None


def test_solve_rejects_nan():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        solve(cost)


def test_solve_rejects_neg_inf():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("-inf")
    with pytest.raises(RuntimeError, match="-inf"):
        solve(cost)


def test_auction_assignment_rejects_nan():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        auction_assignment(cost, bid_size=0.1)


def test_auction_assignment_rejects_neg_inf():
    cost = torch.zeros(3, 3)
    cost[0, 0] = float("-inf")
    with pytest.raises(ValueError, match="-inf"):
        auction_assignment(cost, bid_size=0.1)


def test_check_finite_is_skipped_on_a_fake_tensor():
    # Not an optimization: reducing a fake tensor to a Python bool raises,
    # so before the guard this path could not be traced at all.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.zeros(3, 3))
        assert check_finite(fake) is None


def test_solve_value_check_is_skipped_on_a_fake_tensor():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.zeros(3, 3))
        assert solve(fake).shape == (3,)


def test_solve_compiles_fullgraph_under_dynamo():
    # Before the guard, branching on the reduced tensor was unsupported
    # data-dependent control flow under fullgraph=True; the guard removes
    # the branch from the traced frame entirely. The underlying op still
    # validates its own input (its own TORCH_CHECK, unrelated to this
    # guard), so a genuinely bad cost is still rejected, just later, at
    # actual kernel execution instead of at Python-level tracing.
    good = torch.rand(3, 3, dtype=torch.float64)

    @torch.compile(backend="eager", fullgraph=True, dynamic=False)
    def run(c):
        return solve(c)

    assert run(good).shape == (3,)

    bad = torch.rand(3, 3, dtype=torch.float64)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NaN"):
        run(bad)


def test_solve_value_check_is_skipped_under_jit_trace():
    # A traced branch is burned in as if it held for every future input,
    # which is worse than slow: it is silently wrong.
    seen = {}

    def probe(x):
        seen["skipped"] = skip_value_checks(x)
        return x.sum()

    with warnings.catch_warnings():
        # torch.jit.trace warns that it is deprecated; the guard still has
        # to hold for as long as the entry point exists.
        warnings.simplefilter("ignore", DeprecationWarning)
        torch.jit.trace(probe, torch.zeros(3, 3), check_trace=False)
    assert seen["skipped"] is True


def test_solve_value_check_is_skipped_under_make_fx():
    from torch.fx.experimental.proxy_tensor import make_fx

    def run(c):
        return solve(c)

    sample = torch.rand(3, 3, dtype=torch.float64)
    make_fx(torch.func.functionalize(run), tracing_mode="fake")(sample)
    make_fx(run, tracing_mode="real")(sample)


def test_auction_assignment_value_check_no_longer_fails_first_on_a_fake_tensor():
    # auction_assignment's own bidding loop is inherently data-dependent
    # (a while loop whose trip count depends on the data) and is not made
    # traceable by this fix; that is a separate, out-of-scope concern. What
    # this fix removes is the NaN / -inf check itself being the first thing
    # to fail on a fake tensor.
    from torch._subclasses.fake_tensor import (
        DataDependentOutputException,
        FakeTensorMode,
    )

    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.rand(3, 3, dtype=torch.float64))
        with pytest.raises(DataDependentOutputException) as err:
            auction_assignment(fake, bid_size=0.1)
    assert "contains NaN" not in str(err.value)
    assert "contains -inf" not in str(err.value)
