#!/usr/bin/env python
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
    "-z",
    "--zetan",
    type=float,
    default=0.0,
    help="Zeta parameter for plotting in range [0, 1.0] (default: 0.0)",
)

args = parser.parse_args()

from numpy import load
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from spectre import SPECTRE, get_xinit_specwrap
from spectre.geometry import get_coord_transform
from spectre.misc import unpack_interfaces


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
test = SPECTRE.from_input_file(args.config_file, verbose=False)

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

    # Extract axis coordinates
    r_axis, z_axis = extract_axis_coords(test, x, zetan=args.zetan)

    labels.append(label)
    r_axes.append(r_axis)
    z_axes.append(z_axis)

# Convert to numpy arrays
r_axes = np.array(r_axes)
z_axes = np.array(z_axes)
iterations = np.arange(len(labels))

# Create figure and plot
fig, ax = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)

geometry = test.input_list_mod.igeometry

# Plot R/x coordinate vs iteration for all axis points
ax[0].plot(iterations, r_axes, "o-", lw=1.2, markersize=3)

# Plot Z/y coordinate vs iteration for all axis points
ax[1].plot(iterations, z_axes, "o-", lw=1.2, markersize=3)

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

plt.show()
