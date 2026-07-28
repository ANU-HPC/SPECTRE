# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Shared construction helpers for the wrapper tests."""

import os.path

from spectre import SPECTRE
from spectre.file_io import read_input_parameters_from_toml


def spectre_with_lconstraint(fname, lconstraint, verbose=False):
    """Build a SPECTRE instance running the requested constraint mode.

    ``lconstraint`` has to be set on the input, not on the live instance: the Fortran
    initialization derives state from it (``LocalConstraint``, the constraint-multiplier
    workspaces), which a post-construction assignment would leave inconsistent.

    Args:
        fname: Path to the TOML input file.
        lconstraint: Constraint mode to run (SPECTRE ``lconstraint``).
        verbose: Whether the instance prints its Fortran output.

    Returns:
        SPECTRE instance initialized with ``lconstraint``.
    """
    params = read_input_parameters_from_toml(fname)
    params.physics.lconstraint = lconstraint
    return SPECTRE.from_input_params(
        params, name=os.path.basename(fname)[: -len(".toml")], verbose=verbose
    )
