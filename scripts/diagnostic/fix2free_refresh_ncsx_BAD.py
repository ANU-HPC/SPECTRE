# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Compare ``fix2free`` vs ``fix2free_refresh_infaces`` on the NCSX BAD case.

The free-boundary input written by ``make_freeboundary_ncsx.ipynb``
(``examples/utils/make_freeboundary/ncsx_BAD.toml``) has a computational boundary
that does not nest cleanly with the inherited plasma boundary / internal
interfaces, so the grid in the vacuum region is garbage.

This script reconstructs that case (its committed comp boundary + the NCSX coils)
and converts it two ways:

* ``plain``     – :func:`fix2free` (keeps the inherited interfaces — BAD grid).
* ``refreshed`` – :func:`fix2free_refresh_infaces` (regenerates the plasma
  boundary and every internal interface by mapping the comp boundary inward).

For each it builds a SPECTRE object, evaluates the real-space forces, prints the
field/force errors and saves a ``(s, theta)`` grid cross-section so the bad vs
clean vacuum grid is directly comparable.

Run:
    python scripts/diagnostic/fix2free_refresh_ncsx_BAD.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from simsopt.configs.zoo import get_data

from spectre import SPECTRE
from spectre.converters.fix2free import (
    fix2free,
    fix2free_refresh_infaces,
    params_to_simsopt_boundary,
)
from spectre.error_checking import print_all_errors
from spectre.file_io import read_input_parameters_from_toml

REPO = Path(__file__).resolve().parents[2]
BAD_TOML = REPO / "examples/utils/make_freeboundary/ncsx_BAD.toml"
OUT_DIR = Path(__file__).resolve().parent


def build_and_report(params, label):
    """Build a SPECTRE object from *params*, print errors, return the object."""
    import numpy as np

    nvol = params.physics.nvol
    tflux = np.asarray(params.physics.tflux, float)[: nvol + 1]
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    print(f"tflux[:nvol+1] = {tflux}")
    print(f"comp/plasma ratio (vacuum slot) = {tflux[nvol]:.4f}")
    spectre = SPECTRE.from_input_params(params, name=f"ncsx_{label}")
    spectre.force_real()
    print_all_errors(spectre)
    return spectre


def main():
    """Convert the NCSX BAD case both ways and compare grids / errors."""
    # The committed BAD input already carries the comp boundary (rwc/zws) and the
    # original plasma boundary; reuse it directly so we hit the exact BAD geometry
    # without re-running the slow QFM step from the notebook.
    params = read_input_parameters_from_toml(BAD_TOML)
    comp_boundary = params_to_simsopt_boundary(params, which="comp")

    # NCSX coils (full torus, all symmetry copies).
    *_, bs = get_data("ncsx")

    plain = fix2free(params, bs, comp_boundary)
    refreshed_sqrt = fix2free_refresh_infaces(params, bs, comp_boundary, flux_weighted_rho=False)
    refreshed = fix2free_refresh_infaces(params, bs, comp_boundary)  # flux-corrected rho

    s_plain = build_and_report(plain, "plain")
    s_sqrt = build_and_report(refreshed_sqrt, "refresh_sqrt_rho")
    s_flux = build_and_report(refreshed, "refresh_flux_rho")

    # --- grids (top) + Jacobians (bottom) for the three variants ---
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    cases = [
        (s_plain, "fix2free (keep interfaces)"),
        (s_sqrt, "refresh, rho=sqrt(tflux)"),
        (s_flux, "refresh, flux-corrected rho"),
    ]
    for col, (s, title) in enumerate(cases):
        s.plot_grid(s.xin, ax=axes[0, col])
        axes[0, col].set_title(f"{title} — grid")
        s.plot_jacobian(s.xin, ax=axes[1, col])
        axes[1, col].set_title(f"{title} — Jacobian")
    fig.tight_layout()
    out = OUT_DIR / "fix2free_refresh_ncsx_BAD_grids.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved grid/Jacobian comparison to {out}")


if __name__ == "__main__":
    main()
