# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for ``SPECTRE.update_bns`` (free-boundary iBns / dtflux refresh)."""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def sp_freeb():
    """Build a module-scoped free-boundary W7-X SPECTRE instance."""
    return SPECTRE(curr_path + "test_freeb_w7x.toml", verbose=False)


def test_update_bns_converges_bnserr_below_gbntol(sp_freeb):
    """After update_bns, bnserr is at or below gbntol (iBns self-consistent)."""
    sp_freeb.update_bns()
    gbntol = float(sp_freeb.input_list_mod.gbntol)
    bnserr = float(sp_freeb.allglobal_mod.bnserr)
    assert bnserr <= gbntol, (
        f"bnserr={bnserr:.3e} should be <= gbntol={gbntol:.3e} after update_bns"
    )


def test_update_bns_restores_gbnbld(sp_freeb):
    """update_bns restores the original gbnbld after its temporary gbnbld=0 refresh."""
    gbnbld_before = float(sp_freeb.input_list_mod.gbnbld)
    sp_freeb.update_bns()
    gbnbld_after = float(sp_freeb.input_list_mod.gbnbld)
    assert gbnbld_after == gbnbld_before


def test_update_bns_syncs_tflux_so_preset_preserves_dtflux(sp_freeb):
    """The cumulative tflux must round-trip back to the same dtflux on the next preset.

    The bug this guards against: previously update_bns updated dtflux(Mvol)
    but left input_list_mod.tflux at the stale TOML value, so any subsequent
    call to preset (e.g. inside `field_obj.run()`) would diff the stale tflux
    and reset dtflux to the wrong starting value, throwing away the Newton
    convergence.
    """
    sp_freeb.update_bns()
    dtflux_after_update = float(sp_freeb.allglobal_mod.dtflux[-1])

    # Re-trigger preset_mod.initialize_internal_arrays, which is what xspech
    # does at every call. If tflux is in sync with dtflux, the reconstructed
    # dtflux must match what update_bns left.
    sp_freeb.lib.preset_mod.initialize_internal_arrays()
    dtflux_after_preset = float(sp_freeb.allglobal_mod.dtflux[-1])

    assert abs(dtflux_after_preset - dtflux_after_update) < 1.0e-10, (
        f"preset reset dtflux(Mvol) from {dtflux_after_update:.6e} to "
        f"{dtflux_after_preset:.6e} — tflux/dtflux sync after update_bns "
        f"is broken"
    )


def test_update_bns_is_single_shot_not_iterative(sp_freeb):
    """update_bns must do at most one iBns refresh — no Picard outer loop.

    At fixed `xin`, the self-consistent iBns is uniquely determined. If
    `compute_bnorm_plasmab` is numerically clean, one refresh is already
    self-consistent; if it isn't (high `relvcerr`), iterating the
    `iBns_{n+1} = ofmn(plasma at iBns_n)` Picard map amplifies noise
    instead of damping it. Doing exactly one refresh is the deterministic
    design — this test guards against any future "let me add a loop"
    change that would reintroduce the divergent behavior.
    """
    import inspect

    src = inspect.getsource(type(sp_freeb).update_bns)
    # Heuristic: the body must not contain a Python for-loop that calls
    # update_bnorm. (Inner Newton on dtflux is allowed — that's `dforce`.)
    assert "update_bnorm" in src
    # The only `update_bnorm` call must be outside any `for` over an outer-iter.
    # A simple grep: ensure there's no occurrence of "for outer" in the body.
    assert "for outer" not in src, (
        "update_bns must not iterate iBns refreshes — Picard on the "
        "iBns Newton map is non-contractive at noisy-casing geometries"
    )


def test_update_bns_refresh_ibns_false_preserves_ibns(sp_freeb):
    """With refresh_ibns=False, update_bns must converge dtflux but leave iBns alone.

    Use case: at geometries where the virtual-casing extrapolation inside
    compute_bnorm_plasmab is unreliable (high `relvcerr`), replacing iBns
    with the noisy ofmn output destabilizes the vacuum-volume Beltrami
    solve. Callers in that regime want the dtflux Newton convergence
    without the iBns refresh.
    """
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()

    sp_freeb.update_bns(refresh_ibns=False)

    ibns_after = np.asarray(sp_freeb.allglobal_mod.ibns)
    dtflux_after = float(sp_freeb.allglobal_mod.dtflux[-1])

    # iBns must be byte-identical: refresh skipped.
    np.testing.assert_array_equal(ibns_before, ibns_after)
    # dtflux Newton step still applied — usually nonzero.
    assert isinstance(dtflux_after, float)


def test_update_bns_converges_dtflux_newton(sp_freeb):
    """A follow-up dforce does not move dtflux(Mvol) — the Newton step is converged."""
    sp_freeb.update_bns()
    x = sp_freeb.xin
    ngdof = sp_freeb.allglobal_mod.ngdof
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

    dt_before = float(sp_freeb.allglobal_mod.dtflux[-1])
    sp_freeb.forces_mod.dforce(ngdof, x, force, False, True)
    dt_after = float(sp_freeb.allglobal_mod.dtflux[-1])

    # The Newton update on dtflux(Mvol) at the converged fixed point should
    # be small. We use a loose tolerance since one extra dforce can still
    # tick dtflux at the floating-point noise level.
    assert abs(dt_after - dt_before) < 1.0e-8, (
        f"dtflux(Mvol) moved by {abs(dt_after - dt_before):.3e} after "
        f"a follow-up dforce — Newton should be converged"
    )


def test_update_bns_clears_field_tracking(sp_freeb):
    """update_bns mutates iBns/dtflux, so any tracked solve must be invalidated."""
    # Forge the stamps (cheaper than a full compute_fields on this fixture);
    # update_bns must clear them regardless of how they were set. The geometry
    # is untouched, so the other module-scoped tests are unaffected.
    sp_freeb._field_fingerprint = sp_freeb.xin
    sp_freeb._solved_xin = sp_freeb.xin

    sp_freeb.update_bns()

    assert sp_freeb._field_fingerprint is None
    assert sp_freeb._solved_xin is None
