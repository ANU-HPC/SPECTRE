# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""SPECTRE converters package.

Convert continuous equilibria to SPECTRE and convert
a fixed boundary equilibrium to free boundary.
"""

from .compare_vmec_spectre import compare_vmec_spectre as compare_vmec_spectre
from .compute_guesses_map2disc import (
    compute_interface_guesses_map2disc as compute_interface_guesses_map2disc,
)
from .fix2free import fix2free as fix2free
from .fix2free import fix2free_refresh_infaces as fix2free_refresh_infaces
from .fix2free import params_to_simsopt_boundary as params_to_simsopt_boundary
from .fix2free import plot_freeboundary_modes as plot_freeboundary_modes
from .simsvmec2spectre import simsvmec2spectre as simsvmec2spectre
from .vmec_helpers import ContinuousPhysics as ContinuousPhysics
from .vmec_helpers import build_spectre_input as build_spectre_input
from .vmec_helpers import compute_interface_guesses as compute_interface_guesses
from .vmec_helpers import iota_function as iota_function
from .vmec_helpers import power_series as power_series
from .vmec_helpers import pressure_function as pressure_function
