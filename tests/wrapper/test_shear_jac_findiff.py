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
    get_grid_shear,
    get_grid_shear_jac,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"

# small fixed grid so the finite-difference sweep stays cheap
SARR = np.linspace(-1.0, 1.0, 5)[:-1]
THETAS = np.linspace(0, 2 * np.pi, 8)
ZETAS = np.linspace(0, 2 * np.pi / 2, 3)


def _block_start(iface, mn):
    """Flat index of the first DOF of interface `iface` (1-based) in the
    infaces layout: leading slot + (iface-1) blocks of (2*mn-1)."""
    return 1 + (iface - 1) * (2 * mn - 1)


def _dofs_for_volume(lvol, mn, mvol):
    """Representative DOFs (R[m=0], R[mode1], Z[mode1]) in the interface
    block(s) bounding `lvol` that are actual DOFs (the CB at Mvol is not)."""
    ifaces = []
    if lvol >= 1:
        ifaces.append(lvol)  # inner interface
    if lvol + 1 <= mvol - 1:
        ifaces.append(lvol + 1)  # outer interface (excluded when it is the CB)
    dofs = []
    for iface in ifaces:
        s = _block_start(iface, mn)
        dofs += [s, s + 1, s + 2]
    return np.array(dofs)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("volume", ["vacuum", "interior"])
def test_grid_shear_jac(volume, normalize):
    sp_fname = f"{curr_path}qa_7vol_test.toml"
    test = SPECTRE(sp_fname, verbose=False)

    mvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    lvol = mvol - 1 if volume == "vacuum" else mvol // 2
    volumes = [lvol]
    dofs = _dofs_for_volume(lvol, mn, mvol)

    xinit = test.xin

    def residual(x):
        return get_grid_shear(
            x, test, sarr=SARR, thetas=THETAS, zetas=ZETAS, volumes=volumes, normalize=normalize
        ).flatten(order="C")

    jac_analytic = get_grid_shear_jac(
        xinit, test, sarr=SARR, thetas=THETAS, zetas=ZETAS, volumes=volumes, normalize=normalize
    )
    jac_analytic = jac_analytic.reshape(-1, test.allglobal_mod.ngdof + 1)[:, dofs]

    jac_numeric = approx_derivative(
        residual,
        xinit,
        method="3-point",
        dofs=dofs + 0,
        abs_step=np.ones(len(dofs)) * 1e-8,
    )

    # the chosen DOFs lie in the bounding interface blocks, so the gradient is
    # genuinely non-trivial (guards against a vacuously-passing all-zero compare)
    assert np.max(np.abs(jac_analytic)) > 0

    maxelem = max(np.max(np.abs(jac_numeric)), np.max(np.abs(jac_analytic)))
    assert np.allclose(jac_numeric, jac_analytic, atol=1e-6 * maxelem, rtol=1e-6)

    remove_file(sp_fname + ".end")
