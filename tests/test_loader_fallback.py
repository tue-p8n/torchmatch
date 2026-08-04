"""
Prebuilt-or-JIT loader fallback tests.

A prebuilt `.so` can be *found* by `find_prebuilt` yet fail to load (corrupt
file, wrong ABI, wrong torch version). `load_extension_module` must catch
that failure and fall back to `jit_build()` instead of letting the exception
propagate and crash the import.
"""

from __future__ import annotations

import pytest

from torchmatch.assignment import _loader as assignment_loader
from torchmatch.transport import _loader as transport_loader


@pytest.fixture
def bogus_so(tmp_path):
    """A file with a `.so` name that is not a valid shared library."""
    path = tmp_path / "_bogus_impl.so"
    path.write_text("not an ELF shared object")
    return path


@pytest.mark.parametrize("loader", [assignment_loader, transport_loader])
def test_load_extension_module_falls_back_to_jit_on_bad_prebuilt(
    monkeypatch, bogus_so, loader
):
    """A found-but-broken prebuilt .so must not crash the import."""
    monkeypatch.setattr(loader, "find_prebuilt", lambda stem: bogus_so)
    monkeypatch.setattr(loader, "force_jit", lambda: False)

    jit_calls = []
    fake_calls = []

    loader.load_extension_module(
        "_bogus_impl",
        jit_build=lambda: jit_calls.append(True),
        register_fakes=lambda: fake_calls.append(True),
    )

    assert jit_calls == [True], "jit_build must be called when prebuilt load fails"
    assert fake_calls == [True], "register_fakes must still run after fallback"


@pytest.mark.parametrize("loader", [assignment_loader, transport_loader])
def test_load_extension_module_uses_jit_when_no_prebuilt_found(monkeypatch, loader):
    """No prebuilt found -> jit_build runs directly (no exception involved)."""
    monkeypatch.setattr(loader, "find_prebuilt", lambda stem: None)
    monkeypatch.setattr(loader, "force_jit", lambda: False)

    jit_calls = []
    fake_calls = []

    loader.load_extension_module(
        "_bogus_impl",
        jit_build=lambda: jit_calls.append(True),
        register_fakes=lambda: fake_calls.append(True),
    )

    assert jit_calls == [True]
    assert fake_calls == [True]


@pytest.mark.parametrize("loader", [assignment_loader, transport_loader])
def test_force_jit_skips_prebuilt_path_entirely(monkeypatch, loader):
    """TORCHMATCH_FORCE_JIT=1 must not even attempt find_prebuilt/load_library."""
    monkeypatch.setattr(loader, "force_jit", lambda: True)

    def _boom(stem):
        raise AssertionError("find_prebuilt must not be called when force_jit()")

    monkeypatch.setattr(loader, "find_prebuilt", _boom)

    jit_calls = []
    fake_calls = []

    loader.load_extension_module(
        "_bogus_impl",
        jit_build=lambda: jit_calls.append(True),
        register_fakes=lambda: fake_calls.append(True),
    )

    assert jit_calls == [True]
    assert fake_calls == [True]
