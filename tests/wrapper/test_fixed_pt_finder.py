# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest
import os.path
from spectre import SPECTRE, force_real

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def qa():
    return f"{curr_path}qa_7vol_test.toml", 12.546318091772426, 0.0


@pytest.fixture(scope="module")
def w7x():
    return f"{curr_path}test_freeb_w7x.toml", 5.908497385241743, 0.0


# TODO: Add funcitonality to find island fixed points and add tests for that as well.
# def test_find_island_fixed_point(self):
#     """Test find_island_fixed_point with m=-5 n=2 in vol=4 where a large
#      island exists in this init config.
#      """
#     out_obj = SPECTREout("./examples/qa_vac_vol7/init.h5")

#     # find_island_fixed_point with n=2, m=-5 in vol=4.
#     r_fp, z_fp, s_fp, theta_fp = out_obj.find_island_fixed_point(guess=[-0.41, 2.29],
#         vol=4, n=2, m=-5, max_iter=0
#     )

#     # Should have found a fixed point
#     assert r_fp is not None


@pytest.mark.parametrize("fixture", ["qa", "w7x"])
def test_find_axis(fixture, request):

    fixture = request.getfixturevalue(fixture)

    fname, r_expected, z_expected = fixture

    out_obj = SPECTRE(fname)

    xin = out_obj.xin

    force_real(xin, out_obj)

    analysis_obj = out_obj.create_spectreout()

    r, z, s, theta = analysis_obj.find_axis(guess=[-0.9, 0.0], max_repeats=0, verbose=True)

    tol = 1e-7
    assert abs(r - r_expected) < tol
    assert abs(z - z_expected) < tol
