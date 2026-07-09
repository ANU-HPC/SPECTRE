# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Shared pytest configuration for the test suite."""

import matplotlib

matplotlib.use("Agg", force=True)  # headless: plt.show() must never block the suite


def pytest_collection_modifyitems(items):
    """Add the 'integration' marker to any test marked 'map2disc' or 'simsopt'.

    This lets tests be tagged with the specific dependency they exercise while
    'integration' acts as an umbrella selecting both (e.g. -m 'not integration').
    """
    for item in items:
        if item.get_closest_marker("map2disc") or item.get_closest_marker("simsopt"):
            item.add_marker("integration")
