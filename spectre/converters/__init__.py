# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""SPECTRE converters package for converting other formats to spectre and freeboundary to fixed boundary"""

from .compare_vmec_spectre import compare_vmec_spectre as compare_vmec_spectre
from .fix2free import fix2free as fix2free
from .fix2free import fixedbound_to_freeboundary as fixedbound_to_freeboundary  # deprecated
from .fix2free import params_to_simsopt_boundary as params_to_simsopt_boundary
from .fix2free import plot_freeboundary_modes as plot_freeboundary_modes
from .simsvmec2spectre import simsvmec2spectre as simsvmec2spectre
from .vmec_helpers import VmecPhysics as VmecPhysics
from .vmec_helpers import build_spectre_input as build_spectre_input
from .vmec_helpers import compute_interface_guesses as compute_interface_guesses
from .vmec_helpers import compute_interface_guesses_map2disc as compute_interface_guesses_map2disc
from .vmec_helpers import compute_iota_profile_from_iotaf as compute_iota_profile_from_iotaf
from .vmec_helpers import make_iota_interpolator as make_iota_interpolator
from .vmec_helpers import power_series as power_series
