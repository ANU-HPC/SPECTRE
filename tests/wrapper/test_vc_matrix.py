# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Validation of the Laplace-virtual-casing matrix (lfreebsolver = 1, direct free-boundary solve).

The dense matrix dVC is checked two ways against independent implementations:

- ``get_vc_consistency`` returns the plasma-induced normal-field harmonics on the computational
  boundary computed matrix-free from the kernel quadrature and as ``dVC @ a``; the two must agree
  to machine precision (same discretization, different code paths).
- ``get_bnorm_plasmab`` evaluates the same integral with the mainline virtual-casing machinery
  (``surfacecurrent``/``casinggrid`` resolution cascade); agreement is limited only by the two
  quadratures, which both converge spectrally for the smooth periodic integrand.
"""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap
from spectre.free_boundary import bnorm_from_plasma

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def w7x_vc():
    """Solve the free-boundary W7-X fixture once with the direct (lfreebsolver=1) vacuum matrix."""
    sp_fname = f"{curr_path}test_freeb_w7x_vc.toml"
    test = SPECTRE(sp_fname, verbose=False)

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    yield test


def _vc_consistency(test):
    mn = int(test.allglobal_mod.mn)
    apply_e = np.zeros(mn, dtype=np.float64, order="F")
    apply_o = np.zeros_like(apply_e)
    dense_e = np.zeros_like(apply_e)
    dense_o = np.zeros_like(apply_e)
    test.wrapper_funcs_mod.get_vc_consistency(mn, apply_e, apply_o, dense_e, dense_o)
    return apply_e, apply_o, dense_e, dense_o


def test_dvc_apply_matches_dense(w7x_vc):
    """Matrix-free application of the kernel equals dVC times the packed vacuum solution."""
    apply_e, apply_o, dense_e, dense_o = _vc_consistency(w7x_vc)

    scale = np.max(np.abs(apply_o))
    assert scale > 0.0

    assert np.allclose(apply_o, dense_o, atol=1e-12 * scale, rtol=1e-12)
    assert np.allclose(apply_e, dense_e, atol=1e-12 * scale, rtol=1e-12)


def test_dvc_apply_vs_virtual_casing(w7x_vc):
    """The dVC rows reproduce the mainline virtual-casing quadrature of the same solution."""
    test = w7x_vc
    _, apply_o, _, _ = _vc_consistency(test)

    _, ofm = bnorm_from_plasma(test)

    scale = np.max(np.abs(ofm))
    assert scale > 0.0

    # two independent quadratures of the same integral; both spectrally convergent. At the
    # fixture's vcnt = vcnz = 128 source grid the dVC quadrature is converged to ~6e-5 relative.
    assert np.allclose(apply_o, ofm, atol=1e-4 * scale, rtol=0.0)
