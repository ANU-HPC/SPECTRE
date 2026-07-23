#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

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

args = parser.parse_args()

from numpy import load
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

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
        if len(x) == test.allglobal_mod.ngdof + 1:  # legacy checkpoint with the dead slot 0
            x = x[1:]
        label = data_file
        # print(f"Loaded: x {x.shape=}")

    # Extract axis coordinates
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
            if len(x) == test.allglobal_mod.ngdof + 1:  # legacy checkpoint with the dead slot 0
                x = x[1:]
            # print(f"Loaded (secondary): x {x.shape=}")
            label = f"{data_file} (secondary)"

        # Extract axis coordinates
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

# Plot R/x coordinate vs iteration for all axis points (primary - blue tones)
ax[0].plot(iterations, r_axes, "o-", lw=1.2, markersize=3, color=colors[0], label="primary")

# Plot Z/y coordinate vs iteration for all axis points (primary - blue tones)
ax[1].plot(iterations, z_axes, "o-", lw=1.2, markersize=3, color=colors[0])

# Plot secondary data if provided (orange tones)
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

# Set labels
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

# # Set x-axis ticks to show file labels
# ax[1].set_xticks(iterations)
# ax[1].set_xticklabels(labels, rotation=45, ha="right")

plt.tight_layout()

# Save R coordinate data to text file if requested
if args.save:
    # Prepare data for saving (primary data)
    save_data = np.column_stack((iterations, r_axes.flatten()))

    # Add header comment to explain the columns
    header = (
        "# Column 1: Iteration number\n# Column 2: R coordinate [m] (or x for cartesian geometry)"
    )

    # Save to text file
    np.savetxt(args.save, save_data, fmt="%d %.8e", header=header, delimiter="\t")
    print(f"R coordinate data saved to {args.save}")

plt.show()
