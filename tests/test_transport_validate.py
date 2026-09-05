"""Tests for transport.matrix input validation and mask/marginal coercion."""

from __future__ import annotations

import warnings

import pytest
import torch
from torchmatch.transport.matrix._validate import (
    coerce_marginals,
    fuse_mask_into_cost,
    validate_cost,
)


def test_validate_cost_accepts_2d():
    validate_cost(torch.zeros(3, 4))


def test_validate_cost_accepts_3d():
    validate_cost(torch.zeros(2, 3, 4))


def test_validate_cost_rejects_1d():
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        validate_cost(torch.zeros(5))


def test_validate_cost_rejects_4d():
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        validate_cost(torch.zeros(2, 3, 4, 5))


def test_validate_cost_rejects_nan():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="cost contains NaN"):
        validate_cost(bad)


def test_validate_cost_rejects_neg_inf():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("-inf")
    with pytest.raises(RuntimeError, match="cost contains -inf"):
        validate_cost(bad)


def test_validate_cost_accepts_pos_inf():
    ok = torch.zeros(3, 3)
    ok[0, 0] = float("inf")
    validate_cost(ok)  # must not raise


def test_validate_cost_rejects_int_dtype():
    with pytest.raises(ValueError, match="float32 or float64"):
        validate_cost(torch.zeros(3, 3, dtype=torch.int64))


def test_coerce_marginals_defaults_uniform_2d():
    cost = torch.zeros(3, 4)
    a, b = coerce_marginals(cost, None, None)
    assert a.shape == (1, 3)
    assert b.shape == (1, 4)
    assert torch.allclose(a, torch.full_like(a, 1.0 / 3))
    assert torch.allclose(b, torch.full_like(b, 1.0 / 4))


def test_coerce_marginals_defaults_uniform_3d():
    cost = torch.zeros(2, 3, 4)
    a, b = coerce_marginals(cost, None, None)
    assert a.shape == (2, 3)
    assert b.shape == (2, 4)


def test_coerce_marginals_accepts_user_2d():
    cost = torch.zeros(3, 4)
    a_in = torch.tensor([0.1, 0.4, 0.5])
    b_in = torch.tensor([0.25, 0.25, 0.25, 0.25])
    a, b = coerce_marginals(cost, a_in, b_in)
    assert torch.equal(a, a_in.unsqueeze(0))
    assert torch.equal(b, b_in.unsqueeze(0))


def test_coerce_marginals_rejects_negative():
    cost = torch.zeros(3, 4)
    a_bad = torch.tensor([-0.1, 0.4, 0.7])
    with pytest.raises(ValueError, match="non-negative"):
        coerce_marginals(cost, a_bad, None)


def test_fuse_mask_replaces_forbidden_with_inf():
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.tensor([[True, False], [True, True]])
    out = fuse_mask_into_cost(cost, mask)
    assert out[0, 0] == 1.0
    assert out[0, 1] == float("inf")
    assert out[1, 0] == 3.0
    assert out[1, 1] == 4.0


def test_fuse_mask_none_returns_unchanged():
    cost = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert fuse_mask_into_cost(cost, None) is cost


def test_fuse_mask_broadcasts():
    # 3D cost with 2D mask broadcasts.
    cost = torch.ones(2, 3, 3)
    mask = torch.tensor([[True, False, True], [True, True, True], [True, True, True]])
    out = fuse_mask_into_cost(cost, mask)
    assert out.shape == (2, 3, 3)
    assert torch.isinf(out[:, 0, 1]).all()


# ---------------------------------------------------------------------------
# Opting out of the value checks, and the cases where they cannot run at all.
# ---------------------------------------------------------------------------


def test_validate_cost_skips_nan_check_when_asked():
    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")
    validate_cost(bad, check_values=False)  # must not raise


def test_validate_cost_still_checks_structure_when_value_checks_are_off():
    # check_values=False buys out of the checks that read a value back to
    # the host, not out of the free ones. Turning those off would only turn
    # a clear error into a confusing one deeper in.
    with pytest.raises(ValueError, match="ndim must be 2 or 3"):
        validate_cost(torch.zeros(5), check_values=False)
    with pytest.raises(ValueError, match="float32 or float64"):
        validate_cost(torch.zeros(3, 3, dtype=torch.int64), check_values=False)


def test_solve_accepts_a_non_finite_cost_when_validation_is_off():
    from torchmatch.transport.matrix import solve

    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")
    out = solve(bad, reg=0.5, n_iter=3, validate=False)
    assert out.shape == (3, 3)


def test_solve_still_rejects_a_non_finite_cost_by_default():
    from torchmatch.transport.matrix import solve

    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="cost contains NaN"):
        solve(bad, reg=0.5, n_iter=3)


def test_negative_marginals_are_rejected_by_default_and_skipped_when_off():
    cost = torch.zeros(3, 3)
    bad_a = torch.tensor([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="a must be non-negative"):
        coerce_marginals(cost, bad_a, None)
    coerce_marginals(cost, bad_a, None, check_values=False)  # must not raise


def test_value_checks_are_skipped_on_a_fake_tensor():
    # Not an optimization: reducing a fake tensor to a Python bool raises,
    # so before the guard this path could not be traced at all.
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.zeros(3, 3))
        validate_cost(fake)  # must not raise
        coerce_marginals(fake, None, None)


def test_value_checks_are_skipped_under_dynamo():
    # Under torch.compile the branch is a graph break rather than an error,
    # so the observable is that a NaN cost stops being rejected: the guard
    # is what removes the break.
    from torchmatch.transport.matrix import solve

    bad = torch.zeros(3, 3)
    bad[0, 0] = float("nan")

    @torch.compile(backend="eager", dynamic=False)
    def run(c):
        return solve(c, reg=0.5, n_iter=3)

    assert run(bad).shape == (3, 3)


def test_value_checks_are_skipped_under_jit_trace():
    # A traced branch is burned in as if it held for every future input,
    # which is worse than slow: it is silently wrong.
    from torchmatch.transport.matrix._validate import skip_value_checks

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


def test_value_checks_are_skipped_under_make_fx():
    # make_fx wraps the fake tensors AOTAutograd hands it in a functional
    # wrapper, which a bare isinstance(FakeTensor) does not see through;
    # real mode traces on ordinary tensors under a proxy mode instead.
    from torch.fx.experimental.proxy_tensor import make_fx
    from torchmatch.transport.matrix import solve

    def run(c):
        return solve(c, reg=0.5, n_iter=3)

    sample = torch.rand(3, 3, dtype=torch.float64)
    make_fx(torch.func.functionalize(run), tracing_mode="fake")(sample)
    make_fx(run, tracing_mode="real")(sample)


def test_exact_emd_honors_validate_off():
    # The flag has to reach every backend; with it off, a NaN cost gets
    # past the entry check and fails inside the simplex instead.
    from torchmatch.transport.matrix import Backend, solve

    bad = torch.zeros(3, 3, dtype=torch.float64)
    bad[0, 0] = float("nan")
    with pytest.raises(RuntimeError) as err:
        solve(bad, backend=Backend.EXACT_EMD, validate=False)
    assert "contains NaN" not in str(err.value)


def test_exact_emd_value_checks_are_skipped_on_a_fake_tensor():
    # The +inf rejection is a host read too and must stand aside under
    # tracing like the others.
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torchmatch.transport.matrix import Backend, solve

    with FakeTensorMode() as mode:
        fake = mode.from_tensor(torch.zeros(3, 3, dtype=torch.float64))
        assert solve(fake, backend=Backend.EXACT_EMD).shape == (3, 3)


def test_value_checks_run_in_ordinary_eager_use():
    from torchmatch.transport.matrix._validate import skip_value_checks

    assert skip_value_checks(torch.zeros(3, 3)) is False
