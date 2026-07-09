# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest

from spectre import (
    SPECTRE,
    approx_derivative,
    get_solvegrid_jbar,
    get_solvegrid_jbar_jac,
    remove_file,
)
from spectre.angle_targets import _solvegrid
from spectre.geometry import get_jacobian

curr_path = os.path.dirname(__file__) + "/"

# s_floor well above any sqrt(g) on the solve grid -> the hinge is active everywhere,
# so the FD check validates the core -d(sqrt g)/dX derivative (no kink ambiguity).
S_FLOOR = 1.0e3


def _block_start(iface, mn):
    return 1 + (iface - 1) * (2 * mn - 1)


def _dofs_for_volume(lvol, mn, mvol, im):
    """Representative DOFs in the interface block(s) bounding `lvol` that are actual
    DOFs (neither the axis nor the CB is): the first modes (m=0) plus the first m>0
    R and Z modes -- the latter have interface-dependent psifactor scaling, so they
    catch a psifactor row mix-up that the m=0 modes (constant factor) cannot."""
    ifaces = []
    if lvol >= 1:
        ifaces.append(lvol)
    if lvol + 1 <= mvol - 1:
        ifaces.append(lvol + 1)
    k0 = int(np.argmax(im > 0))  # first m>0 mode index
    dofs = []
    for iface in ifaces:
        s = _block_start(iface, mn)
        # block layout: [R_0, R_1, Z_1, R_2, Z_2, ...] -> R_k at s+2k-1, Z_k at s+2k
        dofs += [s, s + 1, s + 2, s + 2 * k0 - 1, s + 2 * k0]
    return np.array(dofs)


def test_solvegrid_jbar_value():
    """Aggregated residual matches a brute-force quadrature of the hinge over the
    solve grid computed with get_jacobian (independent evaluation path, including
    the regularized map of the axis volume)."""
    sp_fname = f"{curr_path}qa_7vol_test.toml"
    test = SPECTRE(sp_fname, verbose=False)

    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    mvol = test.allglobal_mod.mvol
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    irbc = test.allglobal_mod.irbc  # matches test.xin at construction
    izbs = test.allglobal_mod.izbs

    snodes, sweights, thetas, zetas = _solvegrid(test)
    w_ang = (2 * np.pi / thetas.size) * (2 * np.pi / (nfp * zetas.size))

    res = get_solvegrid_jbar(test.xin, test, s_floor=S_FLOOR)
    assert res.shape == (mvol, zetas.size)

    for lvol in range(mvol):
        sqrtg = get_jacobian(
            geometry,
            irbc[:, lvol],
            irbc[:, lvol + 1],
            izbs[:, lvol],
            izbs[:, lvol + 1],
            im,
            in_,
            snodes[lvol],
            thetas,
            zetas,
            lvol,
        )
        # axis volume: floor scaled by sbar (healthy sqrt g vanishes linearly there)
        sbar = (snodes[lvol] + 1.0) / 2.0
        floor = S_FLOOR * sbar[:, None, None] if lvol == 0 else S_FLOOR
        h = np.maximum(0.0, floor - sqrtg)
        ref = np.sqrt(np.einsum("ijk,i->k", h * h, sweights[lvol] * w_ang))
        assert np.allclose(res[lvol], ref, rtol=1e-10, atol=1e-10)

    # healthy grid, realistic floor: the hinge must be identically zero
    assert np.all(get_solvegrid_jbar(test.xin, test, s_floor=1e-3) == 0.0)

    remove_file(sp_fname + ".end")


@pytest.mark.parametrize("volume", ["axis", "interior", "vacuum"])
def test_solvegrid_jbar_jac(volume):
    sp_fname = f"{curr_path}qa_7vol_test.toml"
    test = SPECTRE(sp_fname, verbose=False)

    mvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    lvol = {"axis": 0, "interior": mvol // 2, "vacuum": mvol - 1}[volume]
    volumes = [lvol]
    dofs = _dofs_for_volume(lvol, mn, mvol, test.allglobal_mod.im)

    xinit = test.xin

    def residual(x):
        return get_solvegrid_jbar(x, test, s_floor=S_FLOOR, volumes=volumes).flatten(order="C")

    jac_analytic = get_solvegrid_jbar_jac(xinit, test, s_floor=S_FLOOR, volumes=volumes)
    jac_analytic = jac_analytic.reshape(-1, test.allglobal_mod.ngdof + 1)[:, dofs]

    # aggregated residuals are O(1e3) with S_FLOOR huge, so a 1e-8 step is
    # roundoff-dominated; 1e-5 sits on the truncation/roundoff plateau (~1e-7 abs)
    jac_numeric = approx_derivative(
        residual,
        xinit,
        method="3-point",
        dofs=dofs + 0,
        abs_step=np.ones(len(dofs)) * 1e-5,
    )

    # the hinge is active everywhere (S_FLOOR huge), so the gradient is genuinely
    # non-trivial -- guards against a vacuously-passing all-zero compare
    assert np.max(np.abs(jac_analytic)) > 0

    maxelem = max(np.max(np.abs(jac_numeric)), np.max(np.abs(jac_analytic)))
    assert np.allclose(jac_numeric, jac_analytic, atol=1e-6 * maxelem, rtol=1e-6)

    remove_file(sp_fname + ".end")
