# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for ``SPECTRE.update_bns`` (free-boundary iBns refresh and dtflux convergence)."""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, remove_file
from spectre.utils import pad_dead_slot

curr_path = os.path.dirname(__file__) + "/"


def _make(toml_name):
    """Build a free-boundary SPECTRE instance with its Beltrami solution settled."""
    path = f"{curr_path}{toml_name}"
    test = SPECTRE(path, verbose=False)
    force_real(test.xin, test)
    return test, path


@pytest.fixture(scope="module")
def sp_freeb():
    test, path = _make("test_freeb_w7x.toml")
    yield test
    remove_file(path + ".end")


@pytest.fixture(scope="module")
def sp_vc():
    test, path = _make("test_freeb_w7x_vc.toml")
    yield test
    remove_file(path + ".end")


def _dforce_at_current_x(test):
    """One raw dforce at the current geometry, returning dtflux of the vacuum volume."""
    ngdof = test.allglobal_mod.ngdof
    x = pad_dead_slot(test.xin)
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")
    test.forces_mod.dforce(ngdof, x, force, False, True)
    return float(test.allglobal_mod.dtflux[-1])


def _fixed_point_residual(test):
    """``mean|iBns - ofmn|`` from the current solution, without moving ``iBns``.

    The independent reference for what ``update_bns`` must report: the same average the
    Fortran forms, taken with the non-mutating virtual-casing diagnostic.
    """
    mn = int(test.allglobal_mod.mn)
    efm = np.zeros(mn, dtype=np.float64, order="F")
    ofm = np.zeros(mn, dtype=np.float64, order="F")
    test.wrapper_funcs_mod.get_bnorm_plasmab(mn, efm, ofm)

    ibns = np.asarray(test.allglobal_mod.ibns)[:mn]
    residual = float(np.sum(np.abs(ibns[1:] - ofm[1:])) / (mn - 1))
    if not bool(test.input_list_mod.enforce_stell_sym):
        ibnc = np.asarray(test.allglobal_mod.ibnc)[:mn]
        residual += float(np.sum(np.abs(ibnc - efm)) / mn)
    return residual


def test_rejects_unknown_mode(sp_freeb):
    with pytest.raises(ValueError, match="mode must be"):
        sp_freeb.update_bns(mode="bogus")


@pytest.mark.parametrize("mode", ["refresh", "initialize", "flux"])
def test_geometry_and_gbnbld_are_preserved(sp_freeb, mode):
    """Only the boundary flux quantities may move: geometry and the blend factor persist."""
    xin_before = sp_freeb.xin.copy()
    gbnbld_before = float(sp_freeb.input_list_mod.gbnbld)

    sp_freeb.update_bns(mode=mode)

    np.testing.assert_array_equal(sp_freeb.xin, xin_before)
    assert float(sp_freeb.input_list_mod.gbnbld) == gbnbld_before


def test_initialize_replaces_ibns_and_grades_the_one_it_replaced(sp_freeb):
    """A full replacement leaves iBns equal to the plasma extrapolation.

    Its reported bnserr therefore has to grade the *incoming* iBns: taken after the
    replacement it would compare iBns with the ofmn it was just set to and be identically
    zero, which says nothing about anything.
    """
    # "initialize" settles the Beltrami solution before extrapolating, so settle it here
    # too and the reference is taken from the same field.
    _dforce_at_current_x(sp_freeb)
    expected = _fixed_point_residual(sp_freeb)
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()

    bnserr = sp_freeb.update_bns(mode="initialize")

    assert not np.array_equal(np.asarray(sp_freeb.allglobal_mod.ibns), ibns_before)
    assert bnserr == pytest.approx(expected, rel=1e-6)
    assert bnserr > 0.0


def test_refresh_moves_ibns_and_reports_bnserr(sp_freeb):
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    expected = _fixed_point_residual(sp_freeb)

    bnserr = sp_freeb.update_bns(mode="refresh")

    assert bnserr == pytest.approx(expected, rel=1e-10)
    # The damped update is a no-op only in the degenerate gbnbld=1 case.
    if float(sp_freeb.input_list_mod.gbnbld) < 1.0:
        assert not np.array_equal(np.asarray(sp_freeb.allglobal_mod.ibns), ibns_before)


@pytest.mark.parametrize("gbnbld", [0.0, 0.5, 1.0])
def test_bnserr_does_not_depend_on_the_blend_factor(sp_freeb, gbnbld):
    """The residual grades the iBns that was solved at, so damping cannot scale it.

    Measured after the blend it would come out as gbnbld times this, vanishing entirely at
    gbnbld=0 -- exactly where the iteration is least damped and most in need of a check.
    """
    expected = _fixed_point_residual(sp_freeb)
    gbnbld_saved = float(sp_freeb.input_list_mod.gbnbld)
    ibns_saved = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    try:
        sp_freeb.input_list_mod.gbnbld = gbnbld
        bnserr = sp_freeb.update_bns(mode="refresh")
    finally:
        sp_freeb.input_list_mod.gbnbld = gbnbld_saved
        sp_freeb.allglobal_mod.ibns[:] = ibns_saved

    assert bnserr == pytest.approx(expected, rel=1e-10)


def test_flux_mode_leaves_ibns_untouched(sp_freeb):
    """The escape hatch for noisy virtual casing: converge dtflux, do not touch iBns."""
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    sp_freeb.update_bns(mode="flux")
    np.testing.assert_array_equal(np.asarray(sp_freeb.allglobal_mod.ibns), ibns_before)


def test_dtflux_newton_is_converged(sp_freeb):
    """A follow-up dforce must not move dtflux(Mvol) any further."""
    sp_freeb.update_bns(mode="flux")
    dt_before = float(sp_freeb.allglobal_mod.dtflux[-1])
    dt_after = _dforce_at_current_x(sp_freeb)
    assert abs(dt_after - dt_before) < 1.0e-8


def test_tflux_syncs_so_preset_reproduces_dtflux(sp_freeb):
    """The cumulative tflux must round-trip: a fresh preset rebuilds the same dtflux.

    dforce moves the per-volume dtflux in place but leaves input_list_mod.tflux at the
    TOML guess. Without the write-back, re-running preset would silently revert to the
    unconverged state.
    """
    sp_freeb.update_bns(mode="flux")
    dtflux_after_update = float(sp_freeb.allglobal_mod.dtflux[-1])

    sp_freeb.lib.preset_mod.initialize_internal_arrays()
    dtflux_after_preset = float(sp_freeb.allglobal_mod.dtflux[-1])

    assert abs(dtflux_after_preset - dtflux_after_update) < 1.0e-10


@pytest.mark.parametrize("mode", ["refresh", "initialize", "flux"])
def test_vc_direct_solve_is_a_declared_no_op(sp_vc, mode):
    """With lvcvacuum=1 the vacuum operator carries the plasma field; iBns is never read."""
    ibns_before = np.asarray(sp_vc.allglobal_mod.ibns).copy()
    tflux_before = np.asarray(sp_vc.input_list_mod.tflux).copy()

    result = sp_vc.update_bns(mode=mode)

    assert np.isnan(result)
    np.testing.assert_array_equal(np.asarray(sp_vc.allglobal_mod.ibns), ibns_before)
    np.testing.assert_array_equal(np.asarray(sp_vc.input_list_mod.tflux), tflux_before)
