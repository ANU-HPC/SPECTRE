# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

# Tests calculation of the force_real (and jacobian) on a reduced/increased mesh resolution

import pytest
import numpy as np
import os.path

from spectre import (
    SPECTRE,
    force_real,
    force_real_jac,
    reshape_fjac,
    approx_derivative,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lcons", [1, 3])
def test_force(fname, lcons):

    add_pressure = True
    pert_axis = True
    atol = 1e-12
    rtol = 1e-8

    test = SPECTRE(fname, verbose=False)
    # test.read(lcons=lcons)
    test.input_list_mod.lcons = lcons

    xin = test.xin

    ndiscrete = test.input_list_mod.ndiscrete
    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor

    nt_orig = mpol * 4 * ndiscrete
    nz_orig = ntor * 4 * ndiscrete

    f_orig = force_real(
        xin, test, add_pressure=add_pressure, pert_axis=pert_axis, nt=nt_orig, nz=nz_orig
    )
    f_orig_shpd = f_orig.reshape(-1, nz_orig, nt_orig, order="C")

    nt_red = mpol * 2 * ndiscrete
    nz_red = ntor * 2 * ndiscrete

    f_red = force_real(
        xin, test, add_pressure=add_pressure, pert_axis=pert_axis, nt=nt_red, nz=nz_red
    )
    f_red_shpd = f_red.reshape(-1, nz_red, nt_red, order="C")

    # Compare with reduced mesh
    assert np.allclose(f_red_shpd, f_orig_shpd[:, ::2, ::2], atol=atol, rtol=rtol)

    nt_red = mpol * 2 * ndiscrete
    nz_red = ntor * 8 * ndiscrete

    f_red = force_real(
        xin, test, add_pressure=add_pressure, pert_axis=pert_axis, nt=nt_red, nz=nz_red
    )
    f_red_shpd = f_red.reshape(-1, nz_red, nt_red, order="C")

    # Compare with reduced nt mesh and increased nz mesh
    assert np.allclose(f_red_shpd[:, ::2, :], f_orig_shpd[:, :, ::2], atol=atol, rtol=rtol)

    remove_file(fname + ".end")


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lcons", [1, 3])
def test_force_jac(fname, lcons):

    pert_axis = True
    atol = 5e-10
    rtol = 1e-8

    test = SPECTRE(fname, verbose=False)
    # test.read(lcons=lcons)
    test.input_list_mod.lcons = lcons

    xin = test.xin

    ndiscrete = test.input_list_mod.ndiscrete
    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor

    nt_orig = mpol * 4 * ndiscrete
    nz_orig = ntor * 4 * ndiscrete

    f_orig = force_real(xin, test, pert_axis=pert_axis, nt=nt_orig, nz=nz_orig)
    fjac_orig = force_real_jac(xin, test, pert_axis=pert_axis, nt=nt_orig, nz=nz_orig).reshape(
        -1, len(xin)
    )
    fjac_orig_shpd = reshape_fjac(fjac_orig, nt=nt_orig, nz=nz_orig)

    nt_red = mpol * 2 * ndiscrete
    nz_red = ntor * 2 * ndiscrete

    f_red = force_real(xin, test, pert_axis=pert_axis, nt=nt_red, nz=nz_red)
    fjac_red = force_real_jac(xin, test, pert_axis=pert_axis, nt=nt_red, nz=nz_red).reshape(
        -1, len(xin)
    )
    fjac_red_shpd = reshape_fjac(fjac_red, nt=nt_red, nz=nz_red)

    # Compare with reduced mesh
    assert np.allclose(fjac_orig_shpd[:, :, ::2, ::2], fjac_red_shpd, atol=atol, rtol=rtol)

    nt_red = mpol * 2 * ndiscrete
    nz_red = ntor * 8 * ndiscrete

    f_red = force_real(xin, test, pert_axis=pert_axis, nt=nt_red, nz=nz_red)
    fjac_red = force_real_jac(xin, test, pert_axis=pert_axis, nt=nt_red, nz=nz_red).reshape(
        -1, len(xin)
    )
    fjac_red_shpd = reshape_fjac(fjac_red, nt=nt_red, nz=nz_red)

    # Compare with reduced nt mesh and increased nz mesh
    assert np.allclose(
        fjac_red_shpd[:, :, ::2, :], fjac_orig_shpd[:, :, :, ::2], atol=atol, rtol=rtol
    )

    remove_file(fname + ".end")


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lcons", [1, 3])
def test_force_jac_findiff(fname, lcons):

    dofs = np.array([1, 5, 14])
    abs_step = np.ones(3) * 1e-8

    test = SPECTRE(fname, verbose=False)
    # test.read(lcons=lcons)
    test.input_list_mod.lcons = lcons

    pert_axis = True

    xinit = test.xin

    # multiplier = 4 * test.input_list_mod.ndiscrete
    nt_force = test.input_list_mod.mpol * 4
    nz_force = test.input_list_mod.ntor * 5

    fr = force_real(xinit, test, nt=nt_force, nz=nz_force, pert_axis=pert_axis)

    freal_jac_analytic_full = force_real_jac(
        xinit, test, nt=nt_force, nz=nz_force, pert_axis=pert_axis
    )

    freal_jac_numeric = approx_derivative(
        lambda x: force_real(x, test, nt=nt_force, nz=nz_force, pert_axis=pert_axis).flatten(
            order="C"
        ),
        xinit,
        method="3-point",
        dofs=dofs + 0,
        abs_step=abs_step,
    )

    freal_jac_analytic = freal_jac_analytic_full[:, :, dofs]

    freal_jac_analytic = freal_jac_analytic.reshape(-1, len(dofs))

    maxelem = max(np.max(np.abs(freal_jac_numeric)), np.max(np.abs(freal_jac_analytic)))
    assert np.allclose(freal_jac_numeric, freal_jac_analytic, atol=5e-6 * maxelem, rtol=5e-6)

    remove_file(fname + ".end")
