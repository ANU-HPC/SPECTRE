# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Golden test of the force-Jacobian JVP against the dense Jacobian build.

``force_real_jvp`` computes J @ d with one general-direction linearised solve per volume
(Fortran ``solve_field_deriv_dir``); the dense build computes one column per interface
harmonic. Both are exact linearisations of the same discretisation, so they must agree to
rounding for unit directions (single dense columns) and for random directions spanning
every interface, R and Z, cosine and sine, including the constraint (dmupfdx) coupling.
"""

import os.path

import numpy as np
import pytest
import spectre4py_wrap
from helpers import spectre_with_lconstraint

from spectre import (
    force_real,
    force_real_jac,
    force_real_jvp,
    get_xinit_specwrap,
    remove_file,
)
from spectre.force_targets import _process_input_nt_nz

curr_path = os.path.dirname(__file__) + "/"


def test_force_jvp_context_activation_reset_and_status_isolation():
    """The private context owns activation identity and persistent batch diagnostics."""
    context = spectre4py_wrap.force_jvp_context_mod
    context.jvp_context_initialize(3)

    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)
    np.testing.assert_array_equal([context.jvp_context_status(vvol) for vvol in range(1, 4)], 0)

    scope = context.jvp_context_open()
    context.jvp_context_reserve(5, 12)
    context.jvp_context_activate(5, 8, 2)
    assert context.jvp_context_is_active()
    assert context.jvp_context_identity() == (2, 5, 8)

    context.jvp_context_raise_status(2, 4)
    context.jvp_context_raise_status(2, 1)
    assert context.jvp_context_status(2) == 4

    context.jvp_context_reset()
    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)
    assert context.jvp_context_status(2) == 4

    context.jvp_context_activate(5, 12, 3)
    assert context.jvp_context_is_active()
    context.finalize_jvp_scope(scope)
    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)
    context.jvp_context_reset_status()
    np.testing.assert_array_equal([context.jvp_context_status(vvol) for vvol in range(1, 4)], 0)


def test_force_jvp_context_cleanup_after_unsupported_early_return():
    """The scope finalizer deactivates the context when the JVP driver returns early."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 2)
    xinit = np.asarray(get_xinit_specwrap(test))
    dirs = np.zeros((int(test.allglobal_mod.ngdof) + 1, 1), order="F")

    # Multiple MPI ranks are the remaining unsupported configuration; the rank count is
    # only read by the configuration check, which returns before any state is touched.
    test.allglobal_mod.ncpu = 2
    try:
        jvp, status = force_real_jvp(xinit, dirs, test, filter_stellsym=False, return_status=True)
    finally:
        test.allglobal_mod.ncpu = 1

    np.testing.assert_array_equal(jvp, 0.0)
    np.testing.assert_array_equal(status, -1)
    assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
    assert test.lib.force_jvp_context_mod.jvp_context_identity() == (0, 0, 0)


@pytest.mark.parametrize("lcons", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_force_real_jvp_matches_dense_jacobian(fname, lcons):
    test = spectre_with_lconstraint(fname, lcons)
    assert int(test.input_list_mod.lconstraint) == lcons
    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J, jac_status = force_real_jac(
        xinit, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(jac_status, 0)
    J = J.reshape(-1, ngdof + 1)

    # unit directions (single dense columns) and a random direction across all interfaces
    rng = np.random.default_rng(11)
    dirs = np.zeros((ngdof + 1, 3), order="F")
    dirs[1, 0] = 1.0
    dirs[min(5, ngdof), 1] = 1.0
    dirs[1:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
    jvp = jvp.reshape(-1, dirs.shape[1])

    ref = J @ dirs
    for k in range(dirs.shape[1]):
        scale = max(np.max(np.abs(ref[:, k])), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
        assert rel < 1e-10, (
            f"{os.path.basename(fname)} lcons={lcons}: JVP direction {k} deviates from the "
            f"dense Jacobian by rel {rel:.3e}"
        )

    remove_file(fname + ".end")


@pytest.mark.parametrize("lbeltramisolver", [0, 2])
def test_force_real_jvp_lcons2_findiff_conserves_helicity(lbeltramisolver):
    """Finite-difference validation of the helicity-constrained (lconstraint=2) JVP.

    At a perturbed geometry the forward solve re-iterates mu until each volume matches its
    helicity target again, so central differences of ``force_real`` differentiate along the
    constrained manifold. The JVP's Schur-complement response (da = y + dmu*z with
    <dMD.a0, da> = 0) must reproduce them, and every perturbed state must conserve
    helicity. ``lbeltramisolver=2`` repeats the check with the matrix-free (GMRES) backend
    behind both the forward solves and the JVP solves.
    """
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 2)
    test.input_list_mod.lbeltramisolver = lbeltramisolver
    if lbeltramisolver:
        test.input_list_mod.nitergmres = 3000
        test.input_list_mod.ngmresrestart = 60
        test.input_list_mod.epsgmres = 1e-13
        test.input_list_mod.epsgmres_jac = 1e-13
    nt, nz = _process_input_nt_nz(None, None, test)
    nvol = int(test.input_list_mod.nvol)
    k_target = np.array(test.input_list_mod.helicity[:nvol], copy=True)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    np.testing.assert_allclose(test.allglobal_mod.labintegral[:nvol], k_target, rtol=0, atol=1e-9)

    # one harmonic per interface plus a random direction across all interfaces
    rng = np.random.default_rng(3)
    dirs = np.zeros((ngdof + 1, 3), order="F")
    dirs[1, 0] = 1.0
    dirs[min(37, ngdof), 1] = 1.0
    dirs[1:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    jvp = jvp.reshape(-1, dirs.shape[1])

    eps = 1e-8
    mu_response = np.zeros((dirs.shape[1], nvol))
    for k in range(dirs.shape[1]):
        f_pm = []
        mu_pm = []
        for sign in (+1.0, -1.0):
            f = force_real(
                xinit + sign * eps * dirs[:, k], test, nt=nt, nz=nz, filter_stellsym=False
            )
            f_pm.append(np.asarray(f).flatten())
            mu_pm.append(np.array(test.input_list_mod.mu[:nvol], copy=True))
            # every constrained re-solve conserves the helicity targets
            np.testing.assert_allclose(
                test.allglobal_mod.labintegral[:nvol], k_target, rtol=0, atol=1e-9
            )
        fd = (f_pm[0] - f_pm[1]) / (2 * eps)
        mu_response[k] = (mu_pm[0] - mu_pm[1]) / (2 * eps)
        scale = max(np.max(np.abs(fd)), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - fd)) / scale
        assert rel < 5e-6, (
            f"lbeltramisolver={lbeltramisolver}: JVP direction {k} deviates from finite "
            f"differences by rel {rel:.3e}"
        )

    # the constraint is active: mu responds to the geometry perturbations, so the JVP would
    # be wrong without the helicity Schur-complement term
    assert np.max(np.abs(mu_response)) > 1e-3

    remove_file(fname + ".end")


def test_force_real_jvp_lcons2_freeb_conserves_plasma_helicity():
    """Free-boundary lconstraint=2: plasma volumes conserve their helicity while the vacuum
    volume varies its fluxes to match (curtor, curpol) and its helicity stays free.

    The JVP must match the dense Jacobian (both carry the vacuum flux response through
    dmupfdx / dmupf_dir) and central finite differences of ``force_real``. Every perturbed
    re-solve must hold the plasma helicity targets and the current constraints, while the
    vacuum helicity responds freely to the geometry.
    """
    fname = f"{curr_path}test_freeb_w7x.toml"
    test = spectre_with_lconstraint(fname, 2)
    nt, nz = _process_input_nt_nz(None, None, test)
    nvol = int(test.input_list_mod.nvol)
    mvol = int(test.allglobal_mod.mvol)
    k_target = np.array(test.input_list_mod.helicity[:nvol], copy=True)
    curtor = float(test.input_list_mod.curtor)
    curpol = float(test.input_list_mod.curpol)

    def check_constrained_state():
        # plasma helicity at target; vacuum currents at target (mupftol = 1e-16 here)
        np.testing.assert_allclose(
            test.allglobal_mod.labintegral[:nvol], k_target, rtol=0, atol=1e-12
        )
        ditgp = np.array(test.allglobal_mod.ditgpdxtp)
        np.testing.assert_allclose(ditgp[0, 1, mvol - 1], curtor, rtol=0, atol=1e-10)
        np.testing.assert_allclose(ditgp[1, 1, mvol - 1], curpol, rtol=0, atol=1e-8)
        return float(test.allglobal_mod.labintegral[mvol - 1])

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    check_constrained_state()

    J, jac_status = force_real_jac(
        xinit, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(jac_status, 0)
    J = J.reshape(-1, ngdof + 1)

    # an inner-interface harmonic, a plasma-boundary harmonic (exercising the vacuum flux
    # response), and a random direction across both interfaces
    rng = np.random.default_rng(5)
    dirs = np.zeros((ngdof + 1, 3), order="F")
    dirs[1, 0] = 1.0
    dirs[ngdof, 1] = 1.0
    dirs[1:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    jvp = jvp.reshape(-1, dirs.shape[1])

    ref = J @ dirs
    for k in range(dirs.shape[1]):
        scale = max(np.max(np.abs(ref[:, k])), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
        assert rel < 1e-10, f"freeb lcons2: JVP direction {k} deviates from dense by rel {rel:.3e}"

    # finite differences along the plasma-boundary harmonic and the random direction
    eps = 1e-7
    for k in (1, 2):
        f_pm = []
        mu_pm = []
        k_vac_pm = []
        for sign in (+1.0, -1.0):
            f = force_real(
                xinit + sign * eps * dirs[:, k], test, nt=nt, nz=nz, filter_stellsym=False
            )
            f_pm.append(np.asarray(f).flatten())
            mu_pm.append(np.array(test.input_list_mod.mu[:nvol], copy=True))
            k_vac_pm.append(check_constrained_state())
        fd = (f_pm[0] - f_pm[1]) / (2 * eps)
        scale = max(np.max(np.abs(fd)), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - fd)) / scale
        assert rel < 5e-6, (
            f"freeb lcons2: JVP direction {k} deviates from finite differences by rel {rel:.3e}"
        )
        # the plasma helicity constraint is active and the vacuum helicity responds freely
        assert np.max(np.abs(mu_pm[0] - mu_pm[1])) / (2 * eps) > 1e-3
        assert abs(k_vac_pm[0] - k_vac_pm[1]) / (2 * eps) > 1e-3

    remove_file(fname + ".end")


@pytest.mark.parametrize("lcons", [1, 2, 3])
def test_force_real_jvp_with_factor_cache(lcons):
    """With Lbeltrami_factor_cache the JVP reuses the factorizations deposited by the
    preceding force evaluation instead of rebuilding them; the products must be identical
    to rounding, including on a second batch (which takes them back from the put-back)."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, lcons)
    assert int(test.input_list_mod.lconstraint) == lcons
    test.allglobal_mod.lbeltrami_factor_cache = True
    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J = force_real_jac(xinit, test, nt=nt, nz=nz, filter_stellsym=False).reshape(-1, ngdof + 1)

    rng = np.random.default_rng(7)
    dirs = np.zeros((ngdof + 1, 2), order="F")
    dirs[1:, 0] = rng.standard_normal(ngdof)
    dirs[1:, 1] = rng.standard_normal(ngdof)
    ref = J @ dirs

    for label in ("cached", "put-back"):
        jvp = force_real_jvp(xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False)
        assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
        jvp = jvp.reshape(-1, dirs.shape[1])
        for k in range(dirs.shape[1]):
            scale = max(np.max(np.abs(ref[:, k])), 1e-300)
            rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
            assert rel < 1e-10, (
                f"lcons={lcons} ({label}): JVP direction {k} deviates from the dense "
                f"Jacobian by rel {rel:.3e}"
            )

    test.allglobal_mod.lbeltrami_factor_cache = False
    # the cache is process-global; do not leak it to the next test
    test.lib.beltrami_factor_cache_mod.factor_cache_clear()
    remove_file(fname + ".end")
