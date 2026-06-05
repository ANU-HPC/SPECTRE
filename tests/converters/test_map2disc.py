# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Tests for compute_interface_guesses_map2disc.

Does NOT require simsopt or a Vmec instance — only map2disc.
Boundary coefficients are hardcoded from the wout_iota.nc equilibrium
(nfp=2, SPECTRE angle-fix convention, modes with |value| > 0.01).
"""

import numpy as np
import pytest

# The module imports without map2disc (BCM is imported lazily). Tests that actually
# solve a BCM map are individually marked 'map2disc' and skip when it is missing.
from spectre.converters.map2disc import compute_interface_guesses_map2disc
from spectre.mesh_gen.map2disc import map_boundary_to_interfaces

# SPECTRE-convention (angle_fix=True) boundary modes from wout_iota.nc, nfp=2.
# Derived via vmec_boundary_to_fourier_coeffs(vmec.boundary, angle_fix=True),
# keeping only modes with |value| > 0.01.
_NFP = 2
_RBC = {
    "(0, 0)": 10.33980726,
    "(0, 1)": 0.99240485,
    "(0, 2)": 0.03438705,
    "(1, 1)": 0.26498366,
    "(1, 0)": 2.64781784,
    "(1, -1)": -1.45502752,
    "(1, -2)": -0.07117185,
    "(2, 1)": 0.12435637,
    "(2, 0)": 0.39914512,
    "(2, -1)": 0.75977593,
    "(2, -2)": 0.22120097,
    "(3, -1)": -0.03398074,
    "(3, -2)": -0.13283698,
    "(3, -3)": -0.03295907,
}
_ZBS = {
    "(0, 1)": -0.45379051,
    "(1, 1)": -0.34447216,
    "(1, 0)": -4.03575479,
    "(1, -1)": -1.39034957,
    "(2, 0)": -0.10502487,
    "(2, -1)": -0.36190999,
    "(2, -2)": 0.19379261,
    "(3, -1)": 0.11492154,
    "(3, -3)": -0.06133515,
}


@pytest.mark.map2disc
def test_map2disc_interface_guesses():
    """compute_interface_guesses_map2disc returns one entry per interface."""
    pytest.importorskip("map2disc")
    nvol = 2
    tfluxes = np.linspace(0.0, 1.0, nvol + 1, endpoint=True)[1:]
    guesses = compute_interface_guesses_map2disc(
        _RBC,
        _ZBS,
        _NFP,
        tfluxes,
        mpol=2,
        ntor=1,
        ntheta=32,
        nzeta=15,
        bcm_m=4,
    )
    # test that the shape is correct, and that they are named correctly
    assert len(guesses) == nvol
    assert all(f"interface_{i}" in guesses for i in range(1, nvol + 1))
    assert all("rbc" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))
    assert all("zbs" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))

    # test that the minor radius of the inner interface is smaller than that of the outer
    r00_inner = guesses["interface_1"]["rbc"].get("(1, 0)", 0.0)
    r00_outer = guesses["interface_2"]["rbc"].get("(1, 0)", 0.0)
    # Minor radius (m=1, n=0 mode) grows with flux surface index
    assert abs(r00_inner) < abs(r00_outer)


@pytest.mark.map2disc
def test_map_boundary_to_interfaces_circle():
    """map_boundary_to_interfaces maps a circle to nested, growing circles."""
    pytest.importorskip("map2disc")

    R0, a = 10.0, 1.0

    def make_boundary(zeta):
        def boundary(theta):
            th = np.asarray(theta, dtype=float)
            return R0 + a * np.cos(th), a * np.sin(th)

        return boundary

    rho_arr = np.array([0.4, 0.7, 1.0])
    th_arr = -np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    zeta_arr = np.linspace(0.0, np.pi, 4, endpoint=False)

    # without gradients
    R_arr, Z_arr = map_boundary_to_interfaces(make_boundary, rho_arr, th_arr, zeta_arr, bcm_m=4)
    shape = (rho_arr.size, zeta_arr.size, th_arr.size)
    assert R_arr.shape == shape
    assert Z_arr.shape == shape

    # The mapped minor radius grows with rho: spread of (R - R0, Z) increases.
    radius = np.sqrt((R_arr - R0) ** 2 + Z_arr**2)
    mean_radius = radius.mean(axis=(1, 2))
    assert np.all(np.diff(mean_radius) > 0)
    # Outermost interface recovers the boundary minor radius.
    np.testing.assert_allclose(mean_radius[-1], a, rtol=1e-2)

    # with gradients: four arrays of the same shape
    out = map_boundary_to_interfaces(
        make_boundary, rho_arr, th_arr, zeta_arr, bcm_m=4, with_grad=True
    )
    assert len(out) == 4
    assert all(arr.shape == shape for arr in out)
