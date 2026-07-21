# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for the xin-defaulting ``SPECTRE.force_*`` convenience methods."""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE
from spectre.force_targets import force_real as _force_real_free

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def sp():
    """Build a small fixed-boundary (qa_7vol) SPECTRE instance."""
    return SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)


def test_force_real_method_default_xin_matches_free_function(sp):
    """sp.force_real() must equal force_real(sp.xin, sp)."""
    expected = _force_real_free(sp.xin, sp)
    got = sp.force_real()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_force_real_method_passing_explicit_xin(sp):
    """Explicit xin argument routes through to the underlying function."""
    x = sp.xin.copy()
    # A second call with the same x should match a default-arg call.
    a = sp.force_real()
    b = sp.force_real(x)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)


def test_force_real_method_kwargs_pass_through(sp):
    """Keyword arguments (e.g. filter_stellsym) reach the wrapped function."""
    unfiltered = sp.force_real(filter_stellsym=False)
    filtered = sp.force_real(filter_stellsym=True)
    # filter_stellsym halves the residual vector (and the sum-of-squares
    # matches up to a factor of 2 by construction of filter_stellsym_pts).
    np.testing.assert_allclose(np.sum(unfiltered**2), np.sum(filtered**2) * 2, rtol=1e-10)


def test_force_modes_dforce_method_round_trip(sp):
    """sp.force_modes_dforce() must equal the free function at sp.xin."""
    from spectre.force_targets import force_modes_dforce as _free

    expected = _free(sp.xin, sp)
    got = sp.force_modes_dforce()
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
