# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Finite-difference validation of the pointwise vacuum-field derivative API.

``solve_evaluate_vacuum_field_deriv`` computes and evaluates the derivative of the vacuum vector
potential w.r.t. one raw plasma-boundary harmonic at fixed enclosed fluxes;
(sqrt(g) B) and cylindrical B, or their derivatives at fixed (s, theta, zeta), at arbitrary
points of the vacuum annulus. The finite-difference reference perturbs the same harmonic through
the packed geometry vector (times psifactor, the packing scale) and re-solves the field at
fixed fluxes (lconstraint = -1), evaluating at the same coordinate points.
"""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap

curr_path = os.path.dirname(__file__) + "/"

# scattered evaluation points inside the vacuum annulus (s, theta, zeta)
NPTS = 8
RNG = np.random.default_rng(7)
S_PTS = np.asfortranarray(RNG.uniform(-0.7, 0.7, NPTS))
T_PTS = np.asfortranarray(RNG.uniform(0.0, 2.0 * np.pi, NPTS))
Z_PTS = np.asfortranarray(RNG.uniform(0.0, 2.0 * np.pi / 5.0, NPTS))


def _packed_col(test, ii, irz, issym):
    """0-based packed-DOF index of harmonic (ii, irz, issym) within one interface (packxi order)."""
    mn = int(test.allglobal_mod.mn)
    idof = 0
    for jj in range(1, mn + 1):
        for jrz in (0, 1):
            for jsym in (0, 1):
                if jsym == 1 and bool(test.input_list_mod.enforce_stell_sym):
                    continue
                if jj == 1 and jrz == 1 and jsym == 0:
                    continue
                if jj == 1 and jrz == 0 and jsym == 1:
                    continue
                idof += 1
                if (jj, jrz, jsym) == (ii, irz, issym):
                    return idof - 1
    raise ValueError("harmonic not found")


def _fields_at_points(test):
    gb = np.zeros((3, NPTS), dtype=np.float64, order="F")
    bcyl = np.zeros((3, NPTS), dtype=np.float64, order="F")
    dmap = np.zeros((2, NPTS), dtype=np.float64, order="F")
    test.wrapper_funcs_mod.evaluate_vacuum_field(NPTS, S_PTS, T_PTS, Z_PTS, gb, bcyl, dmap)
    return gb, bcyl, dmap


@pytest.mark.parametrize("dof", [(2, 0, 0), (5, 1, 0), (9, 0, 0)])
def test_pointwise_vacuum_db(tmp_path, dof):
    ii, irz, issym = dof

    # fixed-flux variant of the direct-solve fixture, matching the API's derivative definition
    with open(f"{curr_path}test_freeb_w7x_vc.toml") as fh:
        src = fh.read()
    sp_fname = str(tmp_path / "vc_lcm1.toml")
    with open(sp_fname, "w") as fh:
        fh.write(src.replace("lconstraint = 3", "lconstraint = -1"))

    test = SPECTRE(sp_fname, verbose=False)
    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    ngdof = int(test.allglobal_mod.ngdof)
    lgdof = ngdof // 2
    col = lgdof + _packed_col(test, ii, irz, issym)  # plasma boundary = second interface
    psifac = float(test.allglobal_mod.psifactor[ii - 1, int(test.input_list_mod.nvol) - 1])

    # analytic derivative from the perturbed vacuum solve (slot -1)
    gb_an = np.zeros((3, NPTS), dtype=np.float64, order="F")
    bcyl_an = np.zeros((3, NPTS), dtype=np.float64, order="F")
    dmap_an = np.zeros((2, NPTS), dtype=np.float64, order="F")
    test.wrapper_funcs_mod.solve_evaluate_vacuum_field_deriv(
        NPTS, S_PTS, T_PTS, Z_PTS, ii, irz, issym, gb_an, bcyl_an, dmap_an
    )

    # central finite difference through the packed geometry vector, at the same (s, theta, zeta)
    h = 1e-6
    xp = xinit.copy()
    xp[col] += h
    force_real(xp, test)
    gb_p, bcyl_p, _ = _fields_at_points(test)
    xm = xinit.copy()
    xm[col] -= h
    force_real(xm, test)
    gb_m, bcyl_m, _ = _fields_at_points(test)
    force_real(xinit, test)  # restore

    gb_fd = (gb_p - gb_m) / (2.0 * h) / psifac
    bcyl_fd = (bcyl_p - bcyl_m) / (2.0 * h) / psifac

    gb_scale = np.max(np.abs(gb_fd))
    assert gb_scale > 0.0
    assert np.allclose(gb_an, gb_fd, atol=1e-6 * gb_scale, rtol=1e-6)

    bc_scale = np.max(np.abs(bcyl_fd))
    assert np.allclose(bcyl_an, bcyl_fd, atol=1e-5 * bc_scale, rtol=1e-5)

    # map displacement of the evaluation points: analytic (1-s)/2-weighted basis function
    mi = int(test.allglobal_mod.im[ii - 1])
    ni = int(test.allglobal_mod.in_[ii - 1])
    arg = mi * T_PTS - ni * Z_PTS
    basis = np.cos(arg) if (irz, issym) in ((0, 0), (1, 1)) else np.sin(arg)
    dval = basis * (1.0 - S_PTS) / 2.0
    expected = np.zeros((2, NPTS))
    expected[irz] = dval
    assert np.allclose(dmap_an, expected, atol=1e-12)
