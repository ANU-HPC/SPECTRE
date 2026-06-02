# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Tests for spectre.converters.mesh_gen.

The FFT/finite-difference helpers are exercised with synthetic numpy arrays and
need neither map2disc nor a built SPECTRE object. ``gen_infaces`` drives the full
map2disc path on a real SPECTRE equilibrium and is marked accordingly.
"""

import os.path
import types

import numpy as np
import pytest

from spectre.converters.mesh_gen import (
    gen_infaces,
    gen_mesh_calc_deriv2,
    gen_mesh_fft_rz,
)

curr_path = os.path.dirname(__file__) + "/"
WRAPPER_TOML = os.path.normpath(curr_path + "../wrapper/qa_7vol_test.toml")


def test_gen_mesh_calc_deriv2_linear():
    """Second radial derivative of a rho-linear first derivative is constant."""
    rho_arr = np.linspace(0.1, 1.0, 6)
    nmn = 4
    # first derivatives linear in rho -> np.gradient returns the (constant) slope.
    base = rho_arr[:, None] * np.arange(1, nmn + 1)[None, :]

    d2_re, d2_ro, d2_ze, d2_zo = gen_mesh_calc_deriv2(base, 2 * base, 3 * base, 4 * base, rho_arr)

    for arr, scale in ((d2_re, 1), (d2_ro, 2), (d2_ze, 3), (d2_zo, 4)):
        assert arr.shape == base.shape
        # d/drho of (scale * k * rho) = scale * k, independent of rho.
        np.testing.assert_allclose(arr, scale * np.arange(1, nmn + 1)[None, :], rtol=1e-6)


def test_gen_mesh_fft_rz_shapes():
    """gen_mesh_fft_rz returns eight harmonic arrays of matching shape."""
    mpol, ntor, nfp = 2, 1, 2
    nrho, nzeta, ntheta = 3, 12, 24
    th_arr = -np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    zeta_arr = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)

    rng = np.random.default_rng(0)
    R_arr = rng.standard_normal((nrho, nzeta, ntheta))
    Z_arr = rng.standard_normal((nrho, nzeta, ntheta))
    dR = rng.standard_normal((nrho, nzeta, ntheta))
    dZ = rng.standard_normal((nrho, nzeta, ntheta))

    test = types.SimpleNamespace(input_list_mod=types.SimpleNamespace(mpol=mpol, ntor=ntor))

    out = gen_mesh_fft_rz(R_arr, Z_arr, dR, dZ, test, th_arr, zeta_arr)
    assert len(out) == 8
    ref_shape = out[0].shape
    assert ref_shape[0] == nrho
    assert all(arr.shape == ref_shape for arr in out)


@pytest.mark.map2disc
def test_gen_infaces_from_spectre_state():
    """gen_infaces maps a real SPECTRE boundary inward, one entry per inner interface."""
    pytest.importorskip("map2disc")
    from spectre import SPECTRE, get_xinit_specwrap

    test = SPECTRE(WRAPPER_TOML, verbose=False)
    xin = get_xinit_specwrap(test)
    mvol = test.allglobal_mod.mvol

    R_even, R_odd, Z_even, Z_odd = gen_infaces(test, xin)

    # One row per internal interface (mvol - 1), identical mode count across arrays.
    for arr in (R_even, R_odd, Z_even, Z_odd):
        assert arr.shape[0] == mvol - 1
        assert np.all(np.isfinite(arr))
    assert R_even.shape == R_odd.shape == Z_even.shape == Z_odd.shape
