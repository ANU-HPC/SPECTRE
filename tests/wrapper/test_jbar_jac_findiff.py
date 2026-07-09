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
    get_grid_jbar,
    get_grid_jbar_jac,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"

# small fixed grid so the finite-difference sweep stays cheap
SARR = np.linspace(-1.0, 1.0, 5)[:-1]
THETAS = np.linspace(0, 2 * np.pi, 8)
ZETAS = np.linspace(0, 2 * np.pi / 2, 3)
# s_floor well above any sqrt(g) on the test grid -> the hinge is active everywhere,
# so the FD check validates the core -d(sqrt g)/dX derivative (no kink ambiguity).
S_FLOOR = 1.0e3


def _block_start(iface, mn):
    return 1 + (iface - 1) * (2 * mn - 1)


def _dofs_for_volume(lvol, mn, mvol, im):
    """Representative DOFs in the interface block(s) bounding `lvol` that are actual
    DOFs (the CB at Mvol is not): the first modes (m=0) plus the first m>0 R and Z
    modes -- the latter have interface-dependent psifactor scaling, so they catch
    a psifactor row mix-up that the m=0 modes (constant factor) cannot."""
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


@pytest.mark.parametrize("volume", ["vacuum", "interior"])
def test_grid_jbar_jac(volume):
    sp_fname = f"{curr_path}qa_7vol_test.toml"
    test = SPECTRE(sp_fname, verbose=False)

    mvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    lvol = mvol - 1 if volume == "vacuum" else mvol // 2
    volumes = [lvol]
    dofs = _dofs_for_volume(lvol, mn, mvol, test.allglobal_mod.im)

    xinit = test.xin

    def residual(x):
        return get_grid_jbar(
            x, test, sarr=SARR, thetas=THETAS, zetas=ZETAS, volumes=volumes, s_floor=S_FLOOR
        ).flatten(order="C")

    jac_analytic = get_grid_jbar_jac(
        xinit, test, sarr=SARR, thetas=THETAS, zetas=ZETAS, volumes=volumes, s_floor=S_FLOOR
    )
    jac_analytic = jac_analytic.reshape(-1, test.allglobal_mod.ngdof + 1)[:, dofs]

    jac_numeric = approx_derivative(
        residual,
        xinit,
        method="3-point",
        dofs=dofs + 0,
        abs_step=np.ones(len(dofs)) * 1e-8,
    )

    # the hinge is active everywhere (S_FLOOR huge), so the gradient is genuinely
    # non-trivial -- guards against a vacuously-passing all-zero compare
    assert np.max(np.abs(jac_analytic)) > 0

    maxelem = max(np.max(np.abs(jac_numeric)), np.max(np.abs(jac_analytic)))
    assert np.allclose(jac_numeric, jac_analytic, atol=1e-6 * maxelem, rtol=1e-6)

    remove_file(sp_fname + ".end")
