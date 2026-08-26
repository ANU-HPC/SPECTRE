# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Corrected iterative solves for the direct (lfreebsolver = 1) free-boundary vacuum volume.

The matrix-free Beltrami operator carries the virtual-casing rows as an immutable compact
row-correction payload, so every matvec applies the true operator (dMA - dVC) while the
same-(m,n) preconditioner stays uncorrected. These tests pin that against the dense direct
solve at three levels: the raw matvec, a single GMRES volume solve, and the full forward
field/force/Jacobian evaluation.

The Fortran library is a process singleton, so the dense force and Jacobian are computed
once in the module fixture and cached as arrays; the forced-iterative instance is only
constructed by the LAST test in this file (constructing it re-initializes the global
Fortran state and invalidates the fixture's live handle).
"""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap
from spectre.force_targets import force_real_jac

curr_path = os.path.dirname(__file__) + "/"
FIXTURE = f"{curr_path}test_freeb_w7x_vc.toml"


def _make_iterative_instance():
    test = SPECTRE(FIXTURE, verbose=False)
    test.input_list_mod.lbeltramisolver = 2
    test.input_list_mod.nitergmres = 3000
    test.input_list_mod.ngmresrestart = 60
    test.input_list_mod.epsgmres = 1e-13
    # exercise the iterative path on every volume of this small case, including the
    # coordinate-singularity volume (fine here; the exclusion is a large-case default)
    test.input_list_mod.nadofgmres = 1
    test.input_list_mod.nadofgmres_sing = 1
    return test


@pytest.fixture(scope="module")
def w7x_vc_solved():
    """One solved dense direct-VC instance, with its force and Jacobian cached as arrays."""
    test = SPECTRE(FIXTURE, verbose=False)
    xin = get_xinit_specwrap(test)
    f_dense = np.asarray(force_real(xin, test)).copy()
    ngdof = int(test.allglobal_mod.ngdof)
    jac_dense = np.asarray(force_real_jac(xin, test)).reshape(-1, ngdof).copy()
    yield {"test": test, "xin": xin, "f_dense": f_dense, "jac_dense": jac_dense}


def test_corrected_matvec_matches_dense_reference(w7x_vc_solved):
    """op_apply with the row correction equals the dense (dMA - dVC) action to machine precision."""
    test = w7x_vc_solved["test"]
    mvol = int(test.allglobal_mod.mvol)
    max_reldiff = test.lib.matrix_free_test_support_mod.beltrami_matvec_selftest(mvol, 4, 0.0)
    assert max_reldiff < 1e-12, f"corrected matvec differs from dense: {max_reldiff:.3e}"


def test_corrected_gmres_solve_matches_dense_reference(w7x_vc_solved):
    """A single corrected GMRES vacuum solve reproduces the dense (dMA - dVC) direct solve."""
    test = w7x_vc_solved["test"]
    mvol = int(test.allglobal_mod.mvol)
    its, relerr = test.lib.matrix_free_test_support_mod.beltrami_solve_selftest(
        mvol, 0.0, 60, 3000, 1e-13, 1
    )
    assert relerr < 1e-9, f"corrected GMRES solve differs from dense: {relerr:.3e}"
    assert its > 0


def test_jvp_matches_dense_jacobian_in_direct_vc_mode(w7x_vc_solved):
    """The JVP serves lfreebsolver=1: the directional VC contraction must reproduce J @ dirs.

    Directions include single plasma-boundary harmonics (exercising the vacuum volume's
    virtual-casing derivative term through the directional contraction) and a dense random
    direction across every interface.
    """
    from spectre.force_targets import force_real_jvp

    test = w7x_vc_solved["test"]
    xin = w7x_vc_solved["xin"]
    jac = w7x_vc_solved["jac_dense"]
    ngdof = int(test.allglobal_mod.ngdof)
    lgdof = ngdof // 2

    rng = np.random.default_rng(7)
    dirs = np.zeros((ngdof, 3), order="F")
    dirs[0, 0] = 1.0  # first interior-interface harmonic
    dirs[lgdof, 1] = 1.0  # first plasma-boundary harmonic (moves the VC source surface)
    dirs[:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(xin, dirs, test, return_status=True)
    np.testing.assert_array_equal(status, 0)
    jvp = np.asarray(jvp).reshape(-1, 3)

    # Dense Jacobian assembly is intentionally rank-0-only on the operator seam.
    if test.comm.Get_rank() != 0:
        return

    ref = jac @ dirs
    scale = np.max(np.abs(ref))
    assert scale > 0.0
    assert np.allclose(jvp, ref, atol=1e-11 * scale, rtol=0.0), (
        f"JVP vs J @ dirs max diff {np.max(np.abs(jvp - ref)):.3e} at scale {scale:.3e}"
    )


def test_iterative_direct_vc_matches_dense(w7x_vc_solved):
    """Forced-iterative force AND analytic Jacobian reproduce the dense direct VC.

    KEEP LAST IN THIS FILE: constructing the iterative instance re-initializes the global
    Fortran state, so the module fixture's live handle must not be used afterwards (the
    dense reference values were cached as arrays when the fixture was built).
    """
    f_dense = w7x_vc_solved["f_dense"]
    jac_dense = w7x_vc_solved["jac_dense"]

    test = _make_iterative_instance()
    xin = get_xinit_specwrap(test)
    ngdof = int(test.allglobal_mod.ngdof)
    f_iter = np.asarray(force_real(xin, test)).copy()

    status = np.asarray(test.allglobal_mod.beltrami_gmres_status)
    assert np.all(status == 0), f"GMRES did not converge: status {status}"

    rel = np.linalg.norm(f_iter - f_dense) / max(np.linalg.norm(f_dense), 1e-300)
    assert rel < 1e-8, f"corrected iterative force differs from dense direct VC by rel {rel:.3e}"

    jac_iter = np.asarray(force_real_jac(xin, test)).reshape(-1, ngdof)
    assert int(test.allglobal_mod.force_jac_status.max()) == 0

    rel = np.linalg.norm(jac_iter - jac_dense) / max(np.linalg.norm(jac_dense), 1e-300)
    assert rel < 1e-8, (
        f"corrected iterative force Jacobian differs from dense direct VC by rel {rel:.3e}"
    )
