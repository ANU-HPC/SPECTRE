#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Plot the Beltrami coordinate grid + Jacobian at the INITIAL and FINAL geometry
of a free-boundary run, and report the per-volume FIELD ERROR (avg/max) for both.

The "final" geometry is the last `xs_fixb/x_*.npy` snapshot saved by
`Force_Minimizer`. This makes the warping of the (vacuum) coordinate grid
directly visible, and quantifies how the Beltrami field error changes from the
initial to the optimized state.

Single process (no mpirun); headless (Agg). Reuses `SPECTRE.inspect.plot_grid`
/ `plot_jacobian` (`spectre/core/inspect.py`) and `get_beltrami_errors`
(`spectre/error_checking.py`).

Usage:
  ./plot_init_final.py --toml run/ncsx_refresh_m7n6.toml --xsdir run/xs_fixb \
      --out run/grids_init_final.png --errout run/field_err.txt
"""

from argparse import ArgumentParser
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

parser = ArgumentParser(description="Init/final Beltrami grid + field-error plotter.")
parser.add_argument("--toml", required=True, help="The run's init TOML.")
parser.add_argument("--xsdir", required=True, help="xs_fixb directory of x_*.npy snapshots.")
parser.add_argument("--out", required=True, help="Output PNG path for the grid figure.")
parser.add_argument("--errout", default=None, help="Optional text file for the field-error table.")
parser.add_argument("--zeta", type=float, default=0.0, help="Toroidal angle of the cross-section.")
parser.add_argument("--title", default="", help="Title prefix (e.g. the run name).")
args = parser.parse_args()

from spectre import SPECTRE
from spectre.error_checking import get_beltrami_errors
from spectre.force_targets import force_real

s = SPECTRE(args.toml)
mvol = s.allglobal_mod.mvol

x_init = np.asarray(s.xin).copy()
snaps = sorted(glob(f"{args.xsdir.rstrip('/')}/x_*.npy"))
if not snaps:
    raise SystemExit(f"no x_*.npy snapshots in {args.xsdir}")
x_final = np.load(snaps[-1])


def _field_err(x):
    """Solve the field at x and return per-volume (avg, max) FIELD ERROR arrays."""
    force_real(x, s)
    return get_beltrami_errors(s, print_res=False)


# --- Grid + Jacobian figure: rows = (initial, final), cols = (grid, jacobian, vacuum jacobian) ---
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, (label, x) in enumerate([("initial", x_init), ("final", x_final)]):
    s.plot_grid(x, ax=axes[row, 0], zeta=args.zeta)
    axes[row, 0].set_title(f"{label}: (s,θ) grid")
    s.plot_jacobian(x, ax=axes[row, 1], zeta=args.zeta)
    axes[row, 1].set_title(f"{label}: Jacobian (all vols)")
    # vacuum (outermost) volume only — where the warping concentrates
    s.plot_jacobian(x, ax=axes[row, 2], zeta=args.zeta, volumes=[mvol - 1])
    axes[row, 2].set_title(f"{label}: Jacobian (vacuum vol {mvol - 1})")

fig.suptitle(f"{args.title}  Beltrami grid: initial vs final (ζ={args.zeta})")
fig.tight_layout()
fig.savefig(args.out, dpi=130, bbox_inches="tight")
print(f"[plot] wrote {args.out}")

# --- Field-error table: initial vs final, per volume ---
avg_i, max_i = _field_err(x_init)
avg_f, max_f = _field_err(x_final)

lines = [
    f"# FIELD ERROR (avg/max over s,theta,phi) — {args.title}",
    "# vol   initial_avg  initial_max  final_avg  final_max",
]
for v in range(mvol):
    tag = "vacuum" if v == mvol - 1 else f"plasma{v}"
    lines.append(
        f"{v} ({tag}):  {np.max(avg_i[v]):.3e}  {np.max(max_i[v]):.3e}   "
        f"{np.max(avg_f[v]):.3e}  {np.max(max_f[v]):.3e}"
    )
table = "\n".join(lines)
print(table)
if args.errout:
    with open(args.errout, "w") as fh:
        fh.write(table + "\n")
    print(f"[plot] wrote {args.errout}")
