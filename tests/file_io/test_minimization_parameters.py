# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Cross-field validation of the fixed-point tracking ``[minimization]`` parameters."""

import pytest
from pydantic import ValidationError

from spectre.file_io.input_parameters import MinimizationParameters


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        # A user who asked for island tracking must never silently get axis-only output, so an
        # incomplete configuration is rejected rather than degraded.
        (dict(track_islands=True, island_m=5), "island_volume"),
        (dict(track_islands=True, island_volume=3), "island_m"),
        (dict(track_islands=True), "island_volume"),
        # The map period (island count) must be a positive integer.
        (dict(track_islands=True, island_volume=3, island_m=0), "island_m"),
        (dict(track_islands=True, island_volume=3, island_m=-5), "island_m"),
        # Volume indices are 0-based and non-negative.
        (dict(track_islands=True, island_volume=-1, island_m=5), "island_volume"),
        # A non-positive numerator can never match n, a positive multiple of Nfp.
        (dict(track_islands=True, island_volume=3, island_m=5, island_n=0), "island_n"),
        # island_guess_rz is an (R, Z) pair; any other length is a config error.
        (
            dict(track_islands=True, island_volume=3, island_m=5, island_guess_rz=[1.5]),
            "island_guess_rz",
        ),
        (
            dict(track_islands=True, island_volume=3, island_m=5, island_guess_rz=[1.5, 0.0, 0.2]),
            "island_guess_rz",
        ),
    ],  # fmt: skip
)
def test_invalid_island_config_is_rejected(kwargs, match):
    """Each incomplete or out-of-range island-tracking configuration fails at load time."""
    with pytest.raises(ValidationError, match=match):
        MinimizationParameters(**kwargs)


def test_valid_island_configs_construct():
    """Complete island configs (with and without seed/numerator) and axis-only still validate."""
    MinimizationParameters(track_axis=True)
    MinimizationParameters(track_islands=True, island_volume=3, island_m=5)
    MinimizationParameters(
        track_islands=True, island_volume=3, island_m=5, island_n=3, island_guess_rz=[1.47, 0.0]
    )
