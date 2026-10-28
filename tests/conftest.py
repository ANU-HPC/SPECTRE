# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
#
# Authors: Christopher Berg Smiet <chris@bergtopfusion.com>
#          Erol Balkovic <erol.balkovic@epfl.ch>

"""Shared pytest configuration for the test suite."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg", force=True)  # headless: plt.show() must never block the suite

FIXED_POINTS_DIR = Path(__file__).parent / "fixed_points"
TRACK_FP_TEST = Path(__file__).parent / "scripts" / "test_track_fp.py"


def pytest_addoption(parser):
    """Add the '--run-fixed-points' option, which enables the fixed-point tests."""
    parser.addoption(
        "--run-fixed-points",
        action="store_true",
        default=False,
        help="run the tests in tests/fixed_points/ and tests/scripts/test_track_fp.py "
        "(skipped by default)",
    )


def pytest_collection_modifyitems(config, items):
    """Add the 'integration' marker to any test marked 'map2disc' or 'simsopt'.

    This lets tests be tagged with the specific dependency they exercise while
    'integration' acts as an umbrella selecting both (e.g. -m 'not integration').

    Tests in tests/fixed_points/ and tests/scripts/test_track_fp.py are skipped unless
    '--run-fixed-points' is passed.
    """
    run_fixed_points = config.getoption("--run-fixed-points")
    for item in items:
        if item.get_closest_marker("map2disc") or item.get_closest_marker("simsopt"):
            item.add_marker("integration")
        is_fixed_point_test = FIXED_POINTS_DIR in item.path.parents or item.path == TRACK_FP_TEST
        if not run_fixed_points and is_fixed_point_test:
            item.add_marker(pytest.mark.skip(reason="needs --run-fixed-points"))
