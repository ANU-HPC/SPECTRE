#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Coordinate-axis evolution from interface data (Fourier ``get_coord_transform`` path).

Writes ``{config_stem}_coord_axis_evolution.png`` next to the TOML. Default backend is
``Agg`` for batch/headless runs; use ``--backend Qt5Agg`` only with a display and Qt.

Use the same Python environment as for other SPECTRE tooling (importable ``spectre``);
the script does not configure a virtualenv or site-specific wrappers.
"""

from argparse import ArgumentParser

parser = ArgumentParser(description="Plot evolution of the coordinate axis.")

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
    help="Zeta parameter for plotting in range [0, 1.0] (default: 0.0)",
)
parser.add_argument(
    "--save",
    type=str,
    default=None,
    help="Save R coordinate data to specified text file (2 columns: iteration, R position)",
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
from spectre.geometry import get_coord_transform
from spectre.utils import unpack_interfaces


def extract_axis_coords(test, xin, zetan=0.0):
    """
    Extract coordinate axis from configuration and data.

    Parameters
    ----------
    test : SPECTRE
        SPECTRE object with configuration
    xin : array
        Configuration data (interface coefficients)
    zetan : float
        Zeta parameter in range [0, 1.0]

    Returns
    -------
    rarr_axis : array
        R or x coordinate of axis
    zarr_axis : array
        Z or y coordinate of axis
    """
    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    zeta = zetan * (2 * np.pi / nfp)

    rarr_axis, zarr_axis = get_coord_transform(
        geometry,
        irbc[:, 0],
        irbc[:, 0 + 1],
        izbs[:, 0],
        izbs[:, 0 + 1],
        im,
        in_,
        -1.0,
        0.0,
        zeta,
        0,
    )

    return rarr_axis[0, 0, :, 0], zarr_axis[0, 0, :, 0]


# Load configuration
test = SPECTRE(args.config_file, verbose=False)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Collect all axis data
labels = []
r_axes = []
z_axes = []

# Process each data file and collect data
for i, data_file in enumerate(args.data_files):
    if data_file == "o":
        x = get_xinit_specwrap(test)
        label = "original"
    else:
        x = load(data_file)
        label = data_file

    r_axis, z_axis = extract_axis_coords(test, x, zetan=args.zetan)

    labels.append(label)
    r_axes.append(r_axis)
    z_axes.append(z_axis)

# Convert to numpy arrays
r_axes = np.array(r_axes)
z_axes = np.array(z_axes)
iterations = np.arange(len(labels))

# Process secondary data files if provided
labels_secondary = []
r_axes_secondary = []
z_axes_secondary = []

if args.secondary:
    for i, data_file in enumerate(args.secondary):
        if data_file == "o":
            x = get_xinit_specwrap(test)
            label = "original (secondary)"
        else:
            x = load(data_file)
            label = f"{data_file} (secondary)"

        r_axis, z_axis = extract_axis_coords(test, x, zetan=args.zetan)

        labels_secondary.append(label)
        r_axes_secondary.append(r_axis)
        z_axes_secondary.append(z_axis)

    r_axes_secondary = np.array(r_axes_secondary)
    z_axes_secondary = np.array(z_axes_secondary)
    iterations_secondary = np.arange(len(labels_secondary))

# Create figure and plot
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

geometry = test.input_list_mod.igeometry

ax[0].plot(iterations, r_axes, "o-", lw=1.2, markersize=3, color=colors[0], label="primary")
ax[1].plot(iterations, z_axes, "o-", lw=1.2, markersize=3, color=colors[0])

if args.secondary:
    ax[0].plot(
        iterations_secondary,
        r_axes_secondary,
        "s--",
        lw=1.2,
        markersize=3,
        color=colors[1],
        label="secondary",
    )
    ax[1].plot(iterations_secondary, z_axes_secondary, "s--", lw=1.2, markersize=3, color=colors[1])
    ax[0].legend()

if geometry == 3:
    ax[1].set_xlabel("Iteration", fontsize=13)
    ax[0].set_ylabel("R [m]", fontsize=13)
    ax[1].set_ylabel("Z [m]", fontsize=13)
elif geometry == 2:
    ax[1].set_xlabel("Iteration", fontsize=13)
    ax[0].set_ylabel("x [m]", fontsize=13)
    ax[1].set_ylabel("y [m]", fontsize=13)
elif geometry == 1:
    ax[0].set_ylabel("$\\theta$", fontsize=13)
    ax[1].set_ylabel("r", fontsize=13)
    ax[1].set_xlabel("Iteration", fontsize=13)

ax[0].grid(True, alpha=0.3)
ax[1].grid(True, alpha=0.3)

plt.tight_layout()

cfg_path = Path(args.config_file).expanduser().resolve()
figure_path = cfg_path.parent / f"{cfg_path.stem}_coord_axis_evolution.png"
fig.savefig(figure_path, dpi=200, bbox_inches="tight")
print(f"Figure saved to {figure_path}")

if args.save:
    save_data = np.column_stack((iterations, r_axes.flatten()))
    header = (
        "# Column 1: Iteration number\n# Column 2: R coordinate [m] (or x for cartesian geometry)"
    )
    np.savetxt(args.save, save_data, fmt="%d %.8e", header=header, delimiter="\t")
    print(f"R coordinate data saved to {args.save}")

if args.backend != "Agg":
    plt.show()
else:
    plt.close(fig)
