#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Generate an NCSX free-boundary SPECTRE init TOML at a chosen (mpol, ntor).

Derived from `make_freeboundary_ncsx.ipynb`, which contains the two interface
paths the user manually switches between:

- ``--path normal``  → ``fix2free``: keeps the VMEC-inherited plasma boundary
  and internal interfaces; only adds the computational boundary + vacuum B·n.
- ``--path refresh`` → ``fix2free_refresh_infaces``: regenerates the plasma
  boundary and all internal interfaces by conformally mapping the comp boundary
  inward at flux-corrected radii (``flux_weighted_rho=True``) — the clean,
  well-nested grid.

The poloidal/toroidal resolution is set in the ``simsvmec2spectre`` call
(``mpol``, ``ntor``), which is what we sweep in the resolution study.

Usage:
  ./generate_ncsx_freeb.py --mpol 7  --ntor 6  --path refresh --out tomls/ncsx_refresh_m7n6.toml
  ./generate_ncsx_freeb.py --mpol 11 --ntor 10 --path normal  --out tomls/ncsx_normal_m11n10.toml
"""

from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser(
    prog="generate_ncsx_freeb",
    description="Generate an NCSX free-boundary init TOML at a given (mpol, ntor) via one of two paths.",
)
parser.add_argument("--mpol", type=int, required=True, help="Poloidal Fourier resolution.")
parser.add_argument(
    "--ntor", type=int, required=True, help="Toroidal Fourier resolution (no nfp factor)."
)
parser.add_argument("--nvol", type=int, default=2, help="Number of plasma volumes (default 2).")
parser.add_argument(
    "--path",
    choices=["normal", "refresh"],
    required=True,
    help="normal=fix2free (inherited interfaces); refresh=fix2free_refresh_infaces (regenerated clean grid).",
)
parser.add_argument(
    "--extend",
    type=float,
    default=-0.14,
    help="extend_via_normal distance for the comp boundary (negative = outward; default -0.14, from the notebook).",
)
parser.add_argument(
    "--wout",
    default=None,
    help="VMEC wout file. Default: wout_li383_low_res_reference.nc next to make_freeboundary/.",
)
parser.add_argument("--out", required=True, help="Output TOML path.")
args = parser.parse_args()

from simsopt.configs import get_data
from simsopt.mhd import Vmec

from spectre.converters import (
    fix2free,
    fix2free_refresh_infaces,
    params_to_simsopt_boundary,
    simsvmec2spectre,
)

# Default wout sits in the parent make_freeboundary/ directory, independent of CWD.
wout = args.wout or str(Path(__file__).resolve().parent.parent / "wout_li383_low_res_reference.nc")

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

print(f"[generate] mpol={args.mpol} ntor={args.ntor} nvol={args.nvol} path={args.path}")
print(f"[generate] wout = {wout}")

# Step 1 — fixed-boundary SPECTRE input from VMEC at the requested resolution.
vmec = Vmec(wout)
params = simsvmec2spectre(
    vmec, nvol=args.nvol, mpol=args.mpol, ntor=args.ntor, interface_guess="wout"
)

# Step 2 — NCSX coils.
_, _, _, _, bs = get_data("ncsx")

# Step 3 — computational boundary = plasma boundary extended outward.
plasma_boundary = params_to_simsopt_boundary(params, which="plasma")
comp_boundary = plasma_boundary.copy()
comp_boundary.extend_via_normal(args.extend)

# Step 4 — convert to free-boundary via the chosen path (writes the TOML).
if args.path == "normal":
    freeb = fix2free(params, bs, comp_boundary, filename=args.out)
else:
    freeb = fix2free_refresh_infaces(
        params, bs, comp_boundary, flux_weighted_rho=True, filename=args.out
    )

print(
    f"[generate] wrote {args.out}  "
    f"(lfreebound={freeb.physics.lfreebound}, curpol={freeb.physics.curpol:.4f}, "
    f"phiedge={freeb.physics.phiedge:.4f})"
)
