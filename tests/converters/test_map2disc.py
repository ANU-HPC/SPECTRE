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

pytest.importorskip("map2disc")

from spectre.converters.vmec_helpers import compute_interface_guesses_map2disc  # noqa: E402

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


def test_map2disc_interface_guesses_shape():
    """compute_interface_guesses_map2disc returns one entry per interface."""
    nvol = 2
    tfluxes = np.linspace(0.0, 1.0, nvol + 1, endpoint=True)[1:]
    guesses = compute_interface_guesses_map2disc(
        _RBC, _ZBS, _NFP, tfluxes, mpol=2, ntor=1,
        ntheta=32, nzeta=15, bcm_m=4,
    )
    assert len(guesses) == nvol
    assert all(f"interface_{i}" in guesses for i in range(1, nvol + 1))
    assert all("rbc" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))
    assert all("zbs" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))


def test_map2disc_interface_modes_nonzero():
    """Interface RBC modes must be non-trivially populated."""
    tfluxes = np.array([0.5, 1.0])
    guesses = compute_interface_guesses_map2disc(
        _RBC, _ZBS, _NFP, tfluxes, mpol=2, ntor=1,
        ntheta=32, nzeta=15, bcm_m=4,
    )
    rbc_inner = guesses["interface_1"]["rbc"]
    assert "(0, 0)" in rbc_inner
    assert abs(rbc_inner["(0, 0)"]) > 1.0  # major radius should be substantial


def test_map2disc_inner_interface_smaller_than_outer():
    """Inner interface (tflux=0.25) major radius smaller than outer (tflux=1.0)."""
    tfluxes = np.array([0.25, 1.0])
    guesses = compute_interface_guesses_map2disc(
        _RBC, _ZBS, _NFP, tfluxes, mpol=2, ntor=1,
        ntheta=32, nzeta=15, bcm_m=4,
    )
    r00_inner = guesses["interface_1"]["rbc"].get("(1, 0)", 0.0)
    r00_outer = guesses["interface_2"]["rbc"].get("(1, 0)", 0.0)
    # Minor radius (m=1, n=0 mode) grows with flux surface index
    assert abs(r00_inner) < abs(r00_outer)
