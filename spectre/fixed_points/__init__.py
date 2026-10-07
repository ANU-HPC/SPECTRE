# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Fixed-point finding, classification, and tracking for SPECTRE equilibria."""

from spectre.fixed_points.classification import FixedPointClassifier, classify_point
from spectre.fixed_points.primitives import (
    FixedPointResult,
    find_axis,
    root_fixed_point,
    scan_fixed_points,
)
from spectre.fixed_points.tracker import FixedPointTracker

__all__ = [
    "FixedPointResult",
    "find_axis",
    "root_fixed_point",
    "scan_fixed_points",
    "FixedPointClassifier",
    "classify_point",
    "FixedPointTracker",
]
