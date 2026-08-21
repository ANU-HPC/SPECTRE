# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Free-boundary building blocks used by :meth:`spectre.core.SPECTRE.update_bns`."""

__all__ = [
    "MAX_FLUX_PASSES",
    "bnorm_from_plasma",
    "converge_dtflux",
    "refresh_sweep",
    "sync_tflux_pflux",
    "vacuum_field_at",
    "vacuum_field_derivative_at",
]

from .free_boundary import (
    bnorm_from_plasma,
    sync_tflux_pflux,
    vacuum_field_at,
    vacuum_field_derivative_at,
)
from .picard import MAX_FLUX_PASSES, converge_dtflux, refresh_sweep
