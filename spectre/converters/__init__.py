# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""VMEC-to-SPECTRE converters package."""

from .compare_vmec_spectre import compare_vmec_spectre as compare_vmec_spectre
from .fixedbound_to_freeboundary import fixedbound_to_freeboundary as fixedbound_to_freeboundary
from .fixedbound_to_freeboundary import params_to_simsopt_boundary as params_to_simsopt_boundary
from .fixedbound_to_freeboundary import plot_freeboundary_modes as plot_freeboundary_modes
from .simsvmec2spectre import simsvmec2spectre as simsvmec2spectre
from .vmec_helpers import VmecPhysics as VmecPhysics
from .vmec_helpers import build_spectre_input as build_spectre_input
from .vmec_helpers import compute_interface_guesses as compute_interface_guesses
from .vmec_helpers import compute_interface_guesses_map2disc as compute_interface_guesses_map2disc
