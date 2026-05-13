#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Magnetic-axis evolution across ``.npy`` checkpoints (volume~0, ``find_axis`` / ``get_rz_coords``).

Writes ``{config_stem}_magnetic_axis_evolution.png`` in the same directory as the TOML.
Default Matplotlib backend is ``Agg`` (no display server); use ``--backend Qt5Agg`` only
where Qt is installed and a display is available.

Requires the same Python environment as a normal SPECTRE run (importable ``spectre`` package,
and any MPI / library setup your site uses). The script does not modify ``PATH`` or
activate a virtualenv.
"""

from argparse import ArgumentParser

parser = ArgumentParser(description="Plot evolution of the magnetic axis (fixed-point find_axis).")

parser.add_argument("config_file", help="SPECTRE configuration file")
parser.add_argument(
    "data_files",
    nargs="+",
    help='Data files to process. Use "o" to use original xinit from SPECTRE wrapper, '
    "or provide .npy files with data arrays",
)
parser.add_argument(
    "-s",
    "--secondary",
    nargs="+",
    default=None,
    help="Optional secondary set of data files to plot with different colors. "
    'Use "o" for original xinit or provide .npy files.',
)
parser.add_argument(
    "-z",
    "--zetan",
    type=float,
    default=0.0,
    help="Normalized toroidal label ζ_n: φ = ζ_n·(2π/N_fp) [rad], same convention as plot_evolution_coord_axis.",
)
parser.add_argument(
    "--max-repeats",
    type=int,
    default=50,
    help="Maximum random retries passed to find_axis (default: 50)",
)
parser.add_argument(
    "--verbose-axis",
    action="store_true",
    help="Verbose output from find_axis",
)
parser.add_argument(
    "--save",
    type=str,
    default=None,
    help="Save primary R coordinate column to text file (2 columns: iteration index, R [m])",
)
parser.add_argument(
    "--backend",
    default="Agg",
    choices=("Agg", "Qt5Agg"),
    help="Matplotlib backend: Agg for headless or batch schedulers (default); Qt5Agg when Qt and a display are available.",
)

args = parser.parse_args()

from pathlib import Path

from numpy import load
import numpy as np
import matplotlib

matplotlib.use(args.backend)

import matplotlib.pyplot as plt
from spectre import SPECTRE, get_xinit_specwrap
from spectre.magnetic_axis_rz import magnetic_axis_rz

test = SPECTRE(args.config_file, verbose=False)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

labels = []
r_vals = []
z_vals = []

for data_file in args.data_files:
    if data_file == "o":
        x = get_xinit_specwrap(test)
        label = "original"
    else:
        x = load(data_file)
        label = data_file

    r_axis, z_axis = magnetic_axis_rz(
        test,
        x,
        args.zetan,
        max_repeats=args.max_repeats,
        verbose=args.verbose_axis,
    )
    labels.append(label)
    r_vals.append(r_axis)
    z_vals.append(z_axis)

r_vals = np.asarray(r_vals, dtype=float)
z_vals = np.asarray(z_vals, dtype=float)
iterations = np.arange(len(labels))

labels_secondary = []
r_secondary = []
z_secondary = []

if args.secondary:
    for data_file in args.secondary:
        if data_file == "o":
            x = get_xinit_specwrap(test)
            label = "original (secondary)"
        else:
            x = load(data_file)
            label = f"{data_file} (secondary)"
        r_axis, z_axis = magnetic_axis_rz(
            test,
            x,
            args.zetan,
            max_repeats=args.max_repeats,
            verbose=args.verbose_axis,
        )
        labels_secondary.append(label)
        r_secondary.append(r_axis)
        z_secondary.append(z_axis)

    r_secondary = np.asarray(r_secondary, dtype=float)
    z_secondary = np.asarray(z_secondary, dtype=float)
    iterations_secondary = np.arange(len(labels_secondary))

fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

ax[0].plot(iterations, r_vals, "o-", lw=1.2, markersize=3, color=colors[0], label="primary")
ax[1].plot(iterations, z_vals, "o-", lw=1.2, markersize=3, color=colors[0])

if args.secondary:
    ax[0].plot(
        iterations_secondary,
        r_secondary,
        "s--",
        lw=1.2,
        markersize=3,
        color=colors[1],
        label="secondary",
    )
    ax[1].plot(iterations_secondary, z_secondary, "s--", lw=1.2, markersize=3, color=colors[1])
    ax[0].legend()

ax[1].set_xlabel("Iterations", fontsize=13)
ax[0].set_ylabel("R axis [m]", fontsize=13)
ax[1].set_ylabel("Z axis [m]", fontsize=13)
ax[0].set_title(rf"Magnetic axis position at $\zeta_n = {args.zetan:g}$", fontsize=12)
ax[0].grid(True, alpha=0.3)
ax[1].grid(True, alpha=0.3)

plt.tight_layout()

cfg_path = Path(args.config_file).expanduser().resolve()
figure_path = cfg_path.parent / f"{cfg_path.stem}_magnetic_axis_evolution.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to {figure_path}")

if args.save:
    save_data = np.column_stack((iterations, r_vals))
    header = "# Column 1: iteration index\n# Column 2: R on magnetic axis [m]"
    np.savetxt(args.save, save_data, fmt="%d %.8e", header=header, delimiter="\t")
    print(f"R axis data saved to {args.save}")

if args.backend != "Agg":
    plt.show()
else:
    plt.close(fig)
