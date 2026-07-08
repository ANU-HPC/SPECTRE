# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Tests for fixed-point finding and tracking functions."""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap
from spectre.fixed_points import FixedPointTracker

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def qa():
    """QA 7-volume vacuum test case."""
    return f"{curr_path}qa_7vol_test.toml", 12.546318091772426, 0.0


@pytest.fixture(scope="module")
def w7x():
    """Free-boundary W7-X test case."""
    return f"{curr_path}test_freeb_w7x.toml", 5.908497385241743, 0.0


@pytest.fixture(scope="module")
def qa_specout():
    """Solved SPECTREout for the QA 7-volume test case."""
    obj = SPECTRE(f"{curr_path}qa_7vol_test.toml")
    force_real(get_xinit_specwrap(obj), obj)
    return obj.create_spectreout()


@pytest.mark.parametrize("fixture", ["qa", "w7x"])
def test_find_axis(fixture, request):
    """Magnetic axis R, Z are recovered within 1e-7 of the reference values."""
    fixture = request.getfixturevalue(fixture)
    fname, r_expected, z_expected = fixture

    out_obj = SPECTRE(fname)
    xin = get_xinit_specwrap(out_obj)
    force_real(xin, out_obj)
    analysis_obj = out_obj.create_spectreout()

    r, z, s, theta = analysis_obj.find_axis(guess=[-0.9, 0.0], max_repeats=0, verbose=True)

    tol = 1e-7
    assert abs(r - r_expected) < tol
    assert abs(z - z_expected) < tol


def test_fp_tracker_axis_only_mode(qa_specout):
    """FixedPointTracker with repeat_period=None must track axis without raising."""
    tk = FixedPointTracker(repeat_period=None, volume=None, n=1)
    ctx = tk.step(qa_specout, 0)

    # Axis must have been located.
    assert not np.isnan(tk.axis_rz[0, 0]), "axis R should not be NaN in axis-only mode"
    # no_fp should be True because no island tracking was requested.
    assert ctx["no_fp"] is True
    assert ctx["nofp_reason"] == "axis-only mode"


# TODO: test_find_island_fixed_point — needs a small island-bearing equilibrium fixture
# (qa_7vol_test.toml has monotonic iota ~-0.2 to -0.25, no rational crossings).
# Uncomment and fill in when a suitable test case is available.
