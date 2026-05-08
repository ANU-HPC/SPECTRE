#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

parser = ArgumentParser(description="Plot R coordinate evolution from saved text files.")

parser.add_argument(
    "data_files",
    nargs="+",
    help="One or more text files containing R coordinate evolution data",
)

args = parser.parse_args()

# Read and concatenate data from all files
all_iterations = []
all_r_coords = []
vlines = [0]

for data_file in args.data_files:
    # Load data from text file (skip comment lines starting with #)
    data = np.loadtxt(data_file)

    # Extract iterations and R coordinates
    iterations = data[:, 0]
    r_coords = data[:, 1]

    all_iterations.append(iterations)
    all_r_coords.append(r_coords)

    vlines.append(iterations[-1])

# Concatenate all data into single 1D arrays
all_iterations = np.concatenate(all_iterations)
all_r_coords = np.concatenate(all_r_coords)

# Create figure and plot
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(all_r_coords, "o-", lw=1.2, markersize=2)

ax.set_xlabel("Iteration", fontsize=13)
ax.set_ylabel("R coordinate [m]", fontsize=13)
ax.set_title("R Coordinate Evolution", fontsize=14)
ax.grid(True, alpha=0.3)

for ind in range(1, len(vlines) - 1):
    ax.axvline(x=vlines[ind] + sum(vlines[:ind]) + 1, color="red", linestyle="--", lw=1.2)

plt.tight_layout()
plt.show()
