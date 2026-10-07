# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Shared construction helpers for the wrapper tests."""

import os.path

from spectre import SPECTRE
from spectre.file_io import read_input_parameters_from_toml
from spectre.file_io.toml_io import input_parameters_from_dict, input_parameters_to_dict


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


def spectre_freeb_w7x_reduced(lconstraint, verbose=False):
    """Build the free-boundary W7-X case truncated to a fast test resolution.

    Reads ``test_freeb_w7x.toml`` and drops every Fourier coefficient (boundary, wall,
    vns/bns and interface guesses) with ``m > 2`` or ``|n| > 1``, and lowers the radial
    resolution to ``lrad = [6, 5, 5]``. The truncated boundary is no longer an
    equilibrium, which is irrelevant for backend-consistency tests; the free-boundary
    structure — a vacuum volume bounded by the computational wall, with the (curtor,
    curpol) current constraint — is preserved.

    Args:
        lconstraint: Constraint mode to run.
        verbose: Whether the instance prints its Fortran output.

    Returns:
        SPECTRE instance at the reduced resolution.
    """
    mpol, ntor = 2, 1
    fname = os.path.dirname(__file__) + "/../fixed_points/test_freeb_w7x.toml"
    data = input_parameters_to_dict(read_input_parameters_from_toml(fname))
    phys = data["physics"]

    def truncated(coeffs):
        kept = {}
        for key, value in coeffs.items():
            m, n = (int(part) for part in key.strip("()").split(","))
            if m <= mpol and abs(n) <= ntor:
                kept[key] = value
        return kept

    for key in ("rbc", "zbs", "rbs", "zbc", "rwc", "zws", "rws", "zwc", "vns", "bns", "vnc", "bnc"):
        phys[key] = truncated(phys[key])
    for interface in phys["allrzrz"].values():
        for key in list(interface):
            interface[key] = truncated(interface[key])
    phys["mpol"] = mpol
    phys["ntor"] = ntor
    phys["lrad"] = [6, 5, 5]
    phys["lconstraint"] = lconstraint
    return SPECTRE.from_input_params(
        input_parameters_from_dict(data), name="test_freeb_w7x_reduced", verbose=verbose
    )
