# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for ``Force_Minimizer.__init__`` construction and bns initialization."""

import os
import os.path

import numpy as np
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


def test_iter_arrays_sized_for_resumed_run(tmp_path, monkeypatch):
    """Resumed run sizes the iter_* arrays to the global iteration budget.

    On resume, the callback indexes the iter_* diagnostic lists by the *global*
    iteration counter, which reaches ``iteration_start + max_niter``. Sizing the
    lists to only ``max_niter + 1`` (the old behaviour) overflows them whenever
    ``iteration_start > 0``, crashing the run after useful work with an
    IndexError. This guards the array length against that regression.
    """
    monkeypatch.chdir(tmp_path)

    sp = SPECTRE(curr_path + "test_freeb_w7x.toml", verbose=False)
    # Skip the slow update_bns Newton loop — irrelevant to array sizing.
    sp.min_params.initialize_bns_from_current = False

    # Fabricate xs_fixb snapshots so _set_init_iter takes the resume path.
    # The last snapshot is loaded into field_obj.xin, so save xin to keep shape.
    n_snapshots = 3
    os.mkdir("xs_fixb")
    for i in range(n_snapshots):
        np.save(f"xs_fixb/x_000_{i:04}.npy", sp.xin)

    sp.min_params.resume = True
    sp.min_params.max_niter = 5  # must be > n_snapshots, else _set_init_iter exits

    fm = Force_Minimizer(sp)

    # Resume path actually taken: iteration_start = num_infaces - 1 > 0.
    assert fm.iteration_start == n_snapshots - 1
    expected_len = fm.iteration_start + fm.min_params.max_niter + 1
    for name in (
        "iter_iter",
        "iter_nfev",
        "iter_cost",
        "iter_cost_red",
        "iter_step_norm",
        "iter_opt",
        "iter_fsa",
        "iter_freal",
    ):
        arr = getattr(fm, name)
        assert len(arr) == expected_len, (
            f"{name} sized {len(arr)}, expected {expected_len} "
            f"(iteration_start={fm.iteration_start}, max_niter={fm.min_params.max_niter})"
        )

    # The highest global index the callback writes must be in range.
    max_global_it = fm.iteration_start + fm.min_params.max_niter
    fm.iter_iter[max_global_it] = max_global_it  # must not raise IndexError
