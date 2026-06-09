# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for ``Force_Minimizer.__init__`` construction and bns initialization."""

import os.path

import pytest

from spectre import SPECTRE, Force_Minimizer

curr_path = os.path.dirname(__file__) + "/"


def test_force_minimizer_accepts_toml_path(tmp_path, monkeypatch):
    """A TOML path builds a fresh SPECTRE as the field_obj."""
    # Run inside tmp_path so the snapshot-dir side-effects don't pollute cwd.
    monkeypatch.chdir(tmp_path)

    fm = Force_Minimizer(curr_path + "qa_7vol_test.toml")
    assert isinstance(fm.field_obj, SPECTRE)


def test_force_minimizer_accepts_prebuilt_spectre(tmp_path, monkeypatch):
    """A prebuilt SPECTRE instance is reused as field_obj, not reconstructed."""
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)
    fm = Force_Minimizer(sp)
    # Must reuse the same instance, not construct a fresh one.
    assert fm.field_obj is sp


def test_force_minimizer_rejects_garbage_type():
    """A non-SPECTRE, non-path argument raises TypeError."""
    with pytest.raises(TypeError, match="SPECTRE instance or a str/PathLike"):
        Force_Minimizer(42)


def test_force_minimizer_rejects_non_toml_string():
    """A non-.toml path string raises RuntimeError."""
    with pytest.raises(RuntimeError, match=".toml"):
        Force_Minimizer("/tmp/not_a_toml.txt")


def test_initialize_bns_from_current_default_true_runs_update_bns_in_freeboundary(
    tmp_path, monkeypatch
):
    """Default True + lfreebound=1 → Force_Minimizer calls update_bns once."""
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "test_freeb_w7x.toml", verbose=False)

    # Spy on update_bns: replace with a counter so we don't actually run the
    # (slow) Newton loop in this test.
    call_count = {"n": 0}
    monkeypatch.setattr(
        type(sp),
        "update_bns",
        lambda self, *a, **kw: call_count.__setitem__("n", call_count["n"] + 1),
    )

    _ = Force_Minimizer(sp)
    assert call_count["n"] == 1, (
        "Default initialize_bns_from_current=True + lfreebound=1 should "
        "trigger exactly one update_bns call"
    )


def test_initialize_bns_from_current_skipped_for_fixed_boundary(tmp_path, monkeypatch):
    """Default True but lfreebound=0 → Force_Minimizer must NOT call update_bns."""
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)
    assert sp.input_list_mod.lfreebound == 0, "Test precondition: fixed boundary"

    call_count = {"n": 0}
    monkeypatch.setattr(
        type(sp),
        "update_bns",
        lambda self, *a, **kw: call_count.__setitem__("n", call_count["n"] + 1),
    )

    _ = Force_Minimizer(sp)
    assert call_count["n"] == 0, (
        "update_bns must not be called for fixed-boundary cases regardless "
        "of the initialize_bns_from_current flag"
    )


def test_force_minimizer_does_not_call_field_obj_run_at_init(tmp_path, monkeypatch):
    """Force_Minimizer.__init__ must not run xspech (no field_obj.run).

    The init path used to call ``self.field_obj.run(save_output=True, ...)``
    to (a) create an init-time .h5 file (which nothing downstream reads) and
    (b) force an initial Bns refresh. (b) is now handled explicitly by
    ``update_bns()``, and xspech's `initialize_internal_arrays` reset of
    dtflux/iBns from TOML would actively undo what update_bns just did. This
    test guards against the regression: Force_Minimizer must rely on the
    state SPECTRE was constructed with, never re-run xspech at init.
    """
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)

    call_count = {"n": 0}
    original_run = type(sp).run

    def _spy_run(self, *args, **kwargs):
        call_count["n"] += 1
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(type(sp), "run", _spy_run)

    _ = Force_Minimizer(sp)
    assert call_count["n"] == 0, (
        "Force_Minimizer.__init__ should not call SPECTRE.run() — that would "
        "re-trigger xspech and reset dtflux/iBns from the stale TOML state"
    )


def test_force_minimizer_bnserr_stays_low_after_init_in_freeboundary(tmp_path, monkeypatch):
    """Check bnserr stays at/below gbntol through the end of free-boundary init.

    Proves that update_bns's effect on iBns survives through ``self.f_init``
    (the first ``obj_func`` evaluation).
    """
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "test_freeb_w7x.toml", verbose=False)
    gbntol = float(sp.input_list_mod.gbntol)

    _ = Force_Minimizer(sp)

    bnserr_after_init = float(sp.allglobal_mod.bnserr)
    assert bnserr_after_init <= gbntol, (
        f"bnserr={bnserr_after_init:.3e} > gbntol={gbntol:.3e} after init. "
        f"update_bns set bnserr<=gbntol but something downstream "
        f"(self.f_init? a stray xspech run?) drifted iBns"
    )


def test_initialize_bns_from_current_override_false_suppresses_in_freeboundary(
    tmp_path, monkeypatch
):
    """Override False → Force_Minimizer must NOT call update_bns even in free-boundary."""
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "test_freeb_w7x.toml", verbose=False)
    sp.min_params.initialize_bns_from_current = False

    call_count = {"n": 0}
    monkeypatch.setattr(
        type(sp),
        "update_bns",
        lambda self, *a, **kw: call_count.__setitem__("n", call_count["n"] + 1),
    )

    _ = Force_Minimizer(sp)
    assert call_count["n"] == 0, (
        "Explicit override to False must suppress update_bns even when lfreebound=1"
    )
