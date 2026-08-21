# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Finite-difference validation of the analytic force Jacobian under the direct free-boundary
solve (lfreebsolver = 1).

With the Laplace-virtual-casing matrix in the vacuum volume, perturbing the plasma boundary
(interface Nvol) changes the virtual-casing operator itself; the analytic Jacobian carries this
through the perturbed right-hand side term (d dVC / dx) . a0 (Henneberg et al. eq. 4.5,
``vc_matrix_mod::vc_deriv_rhs_add``). These tests check every geometry degree of freedom of the
plasma-boundary interface (where that term acts) and of the interior interface against
finite differences, for the stellarator-symmetric and non-stellarator-symmetric fixtures.
"""

import os.path

import numpy as np
import pytest

from spectre import (
    SPECTRE,
    approx_derivative,
    force_modes_jac_speclike,
    force_real,
    force_real_jac,
    get_fmn_force_mask,
    get_xinit_specwrap,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


def _zbc_columns(test):
    """0-based packed-DOF columns of interface 1 that are Zbc harmonics (packxi ordering)."""
    mn = int(test.allglobal_mod.mn)
    cols = []
    idof = 0
    for ii in range(1, mn + 1):
        for irz in (0, 1):
            for issym in (0, 1):
                if ii == 1 and irz == 1 and issym == 0:
                    continue
                if ii == 1 and irz == 0 and issym == 1:
                    continue
                idof += 1
                if irz == 1 and issym == 1:
                    cols.append(idof - 1)
    return np.array(cols, dtype=int)


def _fd_against_analytic(test, xinit, jac_analytic_full, dofs):
    """3-point FD over the selected columns against the cached analytic Jacobian."""
    jac_fd = approx_derivative(
        lambda x: force_real(x, test).flatten(order="C"),
        xinit,
        method="3-point",
        dofs=dofs,
        abs_step=np.ones(len(dofs)) * 1e-8,
    )
    jac_analytic = jac_analytic_full[:, :, dofs].reshape(-1, len(dofs))
    return jac_fd, jac_analytic


def _check_jac_columns(test, xinit, jac_analytic_full, dofs, lgdof):
    jac_fd, jac_analytic = _fd_against_analytic(test, xinit, jac_analytic_full, dofs)

    maxelem = max(np.max(np.abs(jac_fd)), np.max(np.abs(jac_analytic)))
    assert np.allclose(jac_fd, jac_analytic, atol=5e-6 * maxelem, rtol=5e-6)

    # the plasma-boundary block alone (columns >= lgdof), where (d dVC / dx) . a0 acts
    pb = dofs >= lgdof
    assert np.allclose(jac_fd[:, pb], jac_analytic[:, pb], atol=5e-6 * maxelem, rtol=5e-6)


@pytest.fixture(scope="module")
def nss_solved():
    """One solved non-stellarator-symmetric instance with its analytic Jacobian cached.

    Shared by the full-coverage FD test and the strict-xfail Zbc-defect test (both FD around
    the same xinit against the same analytic Jacobian), saving a duplicate setup, solve, and
    Jacobian build. The Fortran library is a process singleton: the later stellarator-symmetric
    tests construct their own instances, which invalidates this handle — the two nss tests
    below run consecutively and no other test touches the fixture afterwards.
    """
    sp_fname = f"{curr_path}test_freeb_w7x_nss.toml"
    test = SPECTRE(sp_fname, verbose=False)
    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)
    jac_analytic_full = np.asarray(force_real_jac(xinit, test)).copy()
    yield {"test": test, "xinit": xinit, "jac": jac_analytic_full}
    remove_file(sp_fname + ".end")


def test_force_real_jac_lfreebsolver_stellsym():
    sp_fname = f"{curr_path}test_freeb_w7x_vc.toml"
    test = SPECTRE(sp_fname, verbose=False)

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)
    jac_analytic_full = force_real_jac(xinit, test)

    ngdof = int(test.allglobal_mod.ngdof)
    lgdof = ngdof // 2  # two interfaces in this fixture

    # every plasma-boundary column (where the dVC derivative acts) + a sample of interior ones
    inner = np.linspace(0, lgdof - 1, 8, dtype=int)
    dofs = np.concatenate([inner, np.arange(lgdof, ngdof)])

    _check_jac_columns(test, xinit, jac_analytic_full, dofs, lgdof)

    remove_file(sp_fname + ".end")


def test_force_real_jac_lfreebsolver_nss(nss_solved):
    test = nss_solved["test"]
    xinit = nss_solved["xinit"]

    ngdof = int(test.allglobal_mod.ngdof)
    lgdof = ngdof // 2

    inner = np.linspace(0, lgdof - 1, 8, dtype=int)
    dofs = np.concatenate([inner, np.arange(lgdof, ngdof)])
    # Pre-existing defect, independent of lfreebsolver (identical with lfreebsolver = 0): the
    # analytic derivative w.r.t. Zbc harmonics of an interface bordering the coordinate-
    # singular (Zernike) volume is wrong. Interface 2 borders no singular volume and all its
    # columns are kept, so the non-stellsym virtual-casing derivative is fully exercised.
    dofs = np.setdiff1d(dofs, _zbc_columns(test))

    _check_jac_columns(test, xinit, nss_solved["jac"], dofs, lgdof)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Pre-existing non-stellarator-symmetric Zbc Jacobian defect at the interface bordering "
        "the coordinate-singular volume"
    ),
)
def test_known_nss_zbc_jacobian_defect(nss_solved):
    """Keep the known Zbc defect visible until its non-VC geometry derivative is corrected."""
    test = nss_solved["test"]
    xinit = nss_solved["xinit"]

    cols = _zbc_columns(test)
    jac_fd, jac_analytic = _fd_against_analytic(test, xinit, nss_solved["jac"], cols)
    scale = max(np.max(np.abs(jac_fd)), np.max(np.abs(jac_analytic)))
    assert np.allclose(jac_fd, jac_analytic, atol=5e-6 * scale, rtol=5e-6)


def test_fourier_jac_matches_realspace_lfreebsolver():
    """The dfp200 (Fourier/Hessian) path consumes the same perturbed solutions: it must agree
    with the FD-validated real-space Jacobian under lfreebsolver = 1."""
    sp_fname = f"{curr_path}test_freeb_w7x_vc.toml"
    test = SPECTRE(sp_fname, verbose=False)

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    fmn_jac_custom = force_modes_jac_speclike(xinit, test)

    test.wrapper_funcs_mod.alloc_for_hessian(int(test.allglobal_mod.ntz))
    test.field_mod.alloc_ntz_arrays(int(test.allglobal_mod.nt), int(test.allglobal_mod.nz))

    test.dforce(xinit, compute_derivatives=True)

    fmn_jac_spec = test.allglobal_mod.force_ft_trunc_grad[:, :]
    fmn_jac_spec_bsqrd = fmn_jac_spec[get_fmn_force_mask(test)]

    test.wrapper_funcs_mod.dealloc_hessian()

    maxelem = max(np.max(np.abs(fmn_jac_custom)), np.max(np.abs(fmn_jac_spec_bsqrd)))
    assert np.allclose(fmn_jac_custom, fmn_jac_spec_bsqrd.T, atol=1e-11 * maxelem, rtol=1e-11)

    remove_file(sp_fname + ".end")
