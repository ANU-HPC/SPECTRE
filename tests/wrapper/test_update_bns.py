# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for ``SPECTRE.update_bns`` (iterated damped Picard free-boundary self-consistency).

The coupled state ``(iBns, dtflux(Mvol))`` at fixed geometry obeys an affine sweep map, so
``update_bns`` iterates a damped Picard sweep. These tests pin the sweep semantics (pre-blend
``bnserr``, in-place blend, flux write-back), the contraction at moderate damping, the
bail-on-growth guard and the declared no-op cases.
"""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, remove_file

curr_path = os.path.dirname(__file__) + "/"


def _make(toml_name):
    """Build a free-boundary SPECTRE instance (field not yet solved)."""
    path = f"{curr_path}{toml_name}"
    test = SPECTRE(path, verbose=False)
    return test, path


@pytest.fixture(scope="module")
def sp_freeb():
    """The Picard (lvcvacuum=0) W7-X fixture."""
    test, path = _make("test_freeb_w7x.toml")
    yield test
    remove_file(path + ".end")


@pytest.fixture(scope="module")
def sp_vc():
    """The direct-solve (lvcvacuum=1) W7-X fixture."""
    test, path = _make("test_freeb_w7x_vc.toml")
    yield test
    remove_file(path + ".end")


def _fixed_point_residual(test):
    """``mean|iBns - ofm|`` from the current solution, without moving ``iBns``.

    The independent reference for what ``update_bns`` must report: the same average the
    blend uses, taken with the non-mutating virtual-casing diagnostic.
    """
    mn = int(test.allglobal_mod.mn)
    efm = np.zeros(mn, dtype=np.float64, order="F")
    ofm = np.zeros(mn, dtype=np.float64, order="F")
    test.allglobal_mod.first_free_bound = False
    test.wrapper_funcs_mod.get_bnorm_plasmab(mn, efm, ofm)

    ibns = np.asarray(test.allglobal_mod.ibns)[:mn]
    residual = float(np.sum(np.abs(ibns[1:] - ofm[1:])) / (mn - 1))
    if not bool(test.input_list_mod.enforce_stell_sym):
        ibnc = np.asarray(test.allglobal_mod.ibnc)[:mn]
        residual += float(np.sum(np.abs(ibnc - efm)) / mn)
    return residual


def test_rejects_unknown_mode(sp_freeb):
    """Only the four documented modes exist."""
    with pytest.raises(ValueError, match="mode must be"):
        sp_freeb.update_bns(mode="bogus")


def test_bnserr_contracts_geometrically_at_half_damping(sp_freeb):
    """Damped Picard sweeps contract the pre-blend residual at damping = 0.5.

    The perturbation puts iBns a finite distance from the fixed point; each sweep must
    then shrink the residual by a roughly constant factor below one (measured spectral
    radius of the damped map is well inside the unit circle in the gbnbld 0.3-0.7 band).
    """
    mn = int(sp_freeb.allglobal_mod.mn)
    sp_freeb.allglobal_mod.ibns[1:mn] *= 0.9

    errs = [sp_freeb.update_bns(mode="refresh", damping=0.5) for _ in range(4)]

    assert all(e > 0.0 for e in errs)
    ratios = [errs[i + 1] / errs[i] for i in range(len(errs) - 1)]
    assert all(r < 0.9 for r in ratios), f"non-contracting sweep ratios {ratios}"


@pytest.mark.parametrize("mode", ["refresh", "converge", "initialize", "flux"])
def test_geometry_and_gbnbld_are_preserved(sp_freeb, mode):
    """Only the boundary flux quantities may move: geometry and the blend knob persist."""
    xin_before = sp_freeb.xin.copy()
    gbnbld_before = float(sp_freeb.input_list_mod.gbnbld)

    sp_freeb.update_bns(mode=mode)

    np.testing.assert_array_equal(sp_freeb.xin, xin_before)
    assert float(sp_freeb.input_list_mod.gbnbld) == gbnbld_before


def test_refresh_moves_ibns_and_reports_preblend_bnserr(sp_freeb):
    """One sweep blends iBns and returns the residual of the iBns it started from."""
    sp_freeb.update_bns(mode="flux")  # settle dtflux so the reference sees the same field
    expected = _fixed_point_residual(sp_freeb)
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()

    bnserr = sp_freeb.update_bns(mode="refresh")

    assert bnserr == pytest.approx(expected, rel=1e-8)
    assert float(sp_freeb.allglobal_mod.bnserr) == bnserr
    if bnserr > 0.0 and float(sp_freeb.input_list_mod.gbnbld) < 1.0:
        assert not np.array_equal(np.asarray(sp_freeb.allglobal_mod.ibns), ibns_before)


@pytest.mark.parametrize("damping", [0.0, 0.5, 1.0])
def test_bnserr_does_not_depend_on_the_blend_factor(sp_freeb, damping):
    """The residual grades the iBns the field was solved at, so damping cannot scale it.

    Measured after the blend it would come out as damping times this, vanishing entirely
    at damping=0 -- exactly where the iteration is least damped and most in need of a
    check.
    """
    sp_freeb.update_bns(mode="flux")
    expected = _fixed_point_residual(sp_freeb)
    ibns_saved = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    try:
        bnserr = sp_freeb.update_bns(mode="refresh", damping=damping)
    finally:
        sp_freeb.allglobal_mod.ibns[:] = ibns_saved

    assert bnserr == pytest.approx(expected, rel=1e-8)


def test_converge_reaches_tolerance_and_reports_last_residual(sp_freeb):
    """The default mode sweeps until the pre-blend residual is below tol."""
    tol = 1e-8
    bnserr = sp_freeb.update_bns(mode="converge", tol=tol, max_sweeps=30)

    assert bnserr < tol
    # A follow-up sweep starts from a self-consistent state, so its residual is small too.
    assert sp_freeb.update_bns(mode="refresh") < 10 * tol


def test_initialize_converges_from_scrambled_ibns(sp_freeb):
    """The bootstrap must recover self-consistency from an untrusted iBns.

    The first sweep is a full replacement (nothing of the incoming iBns may survive), so
    convergence cannot depend on the scrambled starting values.
    """
    mn = int(sp_freeb.allglobal_mod.mn)
    sp_freeb.allglobal_mod.ibns[1:mn] = 0.0  # worst case: no information at all

    tol = 1e-8
    bnserr = sp_freeb.update_bns(mode="initialize", tol=tol, max_sweeps=30)

    assert bnserr < tol
    assert not np.all(np.asarray(sp_freeb.allglobal_mod.ibns)[1:mn] == 0.0)


def test_bail_on_growth_warns_and_stops(sp_freeb):
    """A non-contracting map must warn and bail, not spin to max_sweeps.

    damping=2.0 turns the damped map into an amplifier (eigenvalues 2 - mu), so the
    residual grows every sweep and the guard has to fire.
    """
    mn = int(sp_freeb.allglobal_mod.mn)
    ibns_saved = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    sp_freeb.allglobal_mod.ibns[1:mn] *= 0.99  # a small residual for the amplifier to grow
    try:
        with pytest.warns(UserWarning, match="not contracting"):
            sp_freeb.update_bns(mode="converge", damping=2.0, tol=1e-30, max_sweeps=10)
    finally:
        sp_freeb.allglobal_mod.ibns[:] = ibns_saved


def test_flux_mode_leaves_ibns_untouched(sp_freeb):
    """The escape hatch for noisy virtual casing: converge dtflux, do not touch iBns."""
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    result = sp_freeb.update_bns(mode="flux")
    assert np.isnan(result)
    np.testing.assert_array_equal(np.asarray(sp_freeb.allglobal_mod.ibns), ibns_before)


def test_dtflux_newton_is_converged(sp_freeb):
    """A follow-up dforce must not move dtflux(Mvol) any further."""
    sp_freeb.update_bns(mode="flux")
    ivac = int(sp_freeb.allglobal_mod.mvol) - 1
    dt_before = float(sp_freeb.allglobal_mod.dtflux[ivac])

    sp_freeb.update_bns(mode="flux")

    assert abs(float(sp_freeb.allglobal_mod.dtflux[ivac]) - dt_before) < 1.0e-8


def test_cross_rank_state_is_identical(sp_freeb):
    """The collective sweep must leave bit-identical state on every rank."""
    sp_freeb.update_bns(mode="refresh")

    ibns = np.asarray(sp_freeb.allglobal_mod.ibns).copy()
    gathered = sp_freeb.comm.allgather(ibns)
    for other in gathered[1:]:
        np.testing.assert_array_equal(gathered[0], other)

    bnserrs = sp_freeb.comm.allgather(float(sp_freeb.allglobal_mod.bnserr))
    assert all(b == bnserrs[0] for b in bnserrs)

    tflux = np.asarray(sp_freeb.input_list_mod.tflux).copy()
    for other in sp_freeb.comm.allgather(tflux)[1:]:
        np.testing.assert_array_equal(tflux, other)


def test_refresh_invalidates_the_field_solve_skip(sp_freeb):
    """A refresh moves iBns, so a repeated compute_fields at the same xin must re-solve.

    compute_fields skips bitwise-identical repeat solves; silently keeping that skip
    across an iBns move would hand the driver a field solved against the previous
    boundary data. The refresh must drop the skip key.
    """
    x = sp_freeb.xin.copy()
    calls = []
    orig = sp_freeb.field_mod.solve_field
    sp_freeb.field_mod.solve_field = lambda *a: (calls.append(1), orig(*a))
    try:
        sp_freeb.compute_fields(xin=x)
        sp_freeb.compute_fields(xin=x)
        assert len(calls) == 1, "identical repeat call must skip the solve"

        sp_freeb.update_bns(mode="refresh")
        sp_freeb.compute_fields(xin=x)
        assert len(calls) == 2, "an iBns refresh must invalidate the skip"
    finally:
        sp_freeb.field_mod.solve_field = orig


def test_bootstrap_sweep_alone_integrates_a_solved_field(sp_freeb):
    """``initialize`` with ``max_sweeps=0`` is the bare bootstrap sweep, from a solved field.

    The flux convergence has its own pass cap, so a zero sweep budget cannot starve it of
    the ``dforce`` that solves the field before the virtual casing integrates it. And the
    plasma-side surface-current selector is call-scoped: leaking ``first_free_bound`` would
    trip the fatal in a later ``lvcvacuum=1`` instance of the same process.
    """
    mn = int(sp_freeb.allglobal_mod.mn)
    sp_freeb.allglobal_mod.ibns[1:mn] = 0.0

    sp_freeb.update_bns(mode="initialize", max_sweeps=0)

    assert bool(sp_freeb.allglobal_mod.first_free_bound) is False
    assert np.max(np.abs(np.asarray(sp_freeb.allglobal_mod.ibns)[1:mn])) > 1e-3


def test_non_finite_virtual_casing_raises_before_touching_ibns(sp_freeb, monkeypatch):
    """A NaN from the virtual casing is an error, never blended into iBns.

    Blended in, it would poison iBns for good and defeat every stopping test (``nan < tol``
    and ``nan > prev`` are both False), so the driver would burn its whole budget on it.
    """

    def nan_bnorm(nmn, efm, ofm):
        ofm[:] = np.nan

    monkeypatch.setattr(sp_freeb.wrapper_funcs_mod, "get_bnorm_plasmab", nan_bnorm)
    ibns_before = np.asarray(sp_freeb.allglobal_mod.ibns).copy()

    with pytest.raises(FloatingPointError, match="non-finite"):
        sp_freeb.update_bns(mode="refresh")

    np.testing.assert_array_equal(np.asarray(sp_freeb.allglobal_mod.ibns), ibns_before)
    assert bool(sp_freeb.allglobal_mod.first_free_bound) is False


def test_tflux_syncs_so_preset_reproduces_dtflux(sp_freeb):
    """The cumulative tflux must round-trip: a fresh preset rebuilds the same dtflux.

    dforce moves the per-volume dtflux in place but leaves input_list_mod.tflux at the
    input guess. Without the write-back, re-running preset would silently revert to the
    unconverged state. (This test resets the Fortran internal state; keep it last among
    the sp_freeb tests.)
    """
    sp_freeb.update_bns(mode="flux")
    ivac = int(sp_freeb.allglobal_mod.mvol) - 1
    dtflux_after_update = float(sp_freeb.allglobal_mod.dtflux[ivac])

    sp_freeb.lib.preset_mod.initialize_internal_arrays()
    dtflux_after_preset = float(sp_freeb.allglobal_mod.dtflux[ivac])

    assert abs(dtflux_after_preset - dtflux_after_update) < 1.0e-10


@pytest.mark.parametrize("mode", ["refresh", "converge", "initialize", "flux"])
def test_vc_direct_solve_is_a_declared_no_op(sp_vc, mode):
    """With lvcvacuum=1 the vacuum operator carries the plasma field; iBns is never read."""
    ibns_before = np.asarray(sp_vc.allglobal_mod.ibns).copy()
    tflux_before = np.asarray(sp_vc.input_list_mod.tflux).copy()

    result = sp_vc.update_bns(mode=mode)

    assert np.isnan(result)
    np.testing.assert_array_equal(np.asarray(sp_vc.allglobal_mod.ibns), ibns_before)
    np.testing.assert_array_equal(np.asarray(sp_vc.input_list_mod.tflux), tflux_before)


def test_fixed_boundary_is_an_error():
    """A fixed-boundary case has no free-boundary state to converge."""
    test = SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)
    with pytest.raises(RuntimeError, match="lfreebound"):
        test.update_bns()
