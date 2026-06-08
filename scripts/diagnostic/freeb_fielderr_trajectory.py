#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Trace per-volume Beltrami FIELD ERROR (avg AND max) alongside the interface
FORCE residual across the saved optimizer snapshots, to see whether the field
error grows as the optimizer reduces the force (i.e. the optimizer trades field
/ coordinate quality for force balance).

Loads each `xs_fixb/x_*.npy` (an xmn geometry snapshot), solves the field at it,
and prints one compact line per iterate.

Run (one MPI job at a time):
  OMP_NUM_THREADS=3 mpirun -n 3 ./freeb_fielderr_trajectory.py <toml> <snap_dir>
"""

from argparse import ArgumentParser
from glob import glob

parser = ArgumentParser(description="Field-error vs force across optimizer snapshots.")
parser.add_argument("inputfile", help="The .toml the snapshots were optimized from.")
parser.add_argument("snapdir", help="Directory of xs_fixb/x_*.npy snapshots.")
args = parser.parse_args()

import numpy as np

from spectre import SPECTRE
from spectre.error_checking import get_beltrami_errors
from spectre.force_targets import force_real

s = SPECTRE(args.inputfile)
mvol = s.allglobal_mod.mvol
rank0 = s.comm.rank == 0

snaps = sorted(glob(f"{args.snapdir.rstrip('/')}/x_*.npy"))
if rank0:
    print(f"\n{len(snaps)} snapshots; mvol={mvol} (vol {mvol - 1} = vacuum)\n")
    print(f"{'iter':>4}  {'force_rms':>10}  "
          + "  ".join(f"v{v}avg/max" for v in range(mvol)))

for snap in snaps:
    x = np.load(snap)
    force_real(x, s)  # collective solve; sets s.f_real_shpd
    force_rms = float(np.sqrt(np.mean(s.f_real_shpd**2)))
    avg, mx = get_beltrami_errors(s, print_res=False)
    if rank0:
        it = snap.split("_")[-1].split(".")[0]
        cols = "  ".join(
            f"{np.max(avg[v]):.1e}/{np.max(mx[v]):.1e}" for v in range(mvol)
        )
        print(f"{it:>4}  {force_rms:10.3e}  {cols}")

if rank0:
    print("\nDONE. Watch whether vacuum (last vol) avg/max field error RISES as "
          "force_rms FALLS — that would mean the optimizer warps the coordinates "
          "to satisfy force balance.")
