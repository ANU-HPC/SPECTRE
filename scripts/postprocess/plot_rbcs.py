#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(description="Plot 2D boundary harmonics Rbc from a file.")
parser.add_argument("file_path", type=str, help="Path to the input file containing Rbc data.")
args = parser.parse_args()

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")  # must come before importing plt
import matplotlib.pyplot as plt

file_path = args.file_path

i_min, i_max = float('inf'), float('-inf')
j_min, j_max = float('inf'), float('-inf')

with open(file_path, "r") as file:
    for line in file:
        if "Rbc" in line:
            parts = line.split()
            indices = parts[0].strip("Rbc()").split(",")
            i = int(indices[0])
            j = int(indices[1])

            i_min = min(i_min, i)
            i_max = max(i_max, i)
            j_min = min(j_min, j)
            j_max = max(j_max, j)

n = i_max - i_min + 1  # Number of rows (i)
m = j_max - j_min + 1  # Number of columns (j)

Rbc = np.zeros((n, m))

with open(file_path, "r") as file:
    for line in file:
        if "Rbc(" in line:
            # Extract indices and value
            parts = line.split()
            indices = parts[0].strip("Rbc()").split(",")
            i = int(indices[0])
            j = int(indices[1])
            value = float(parts[2])

            i_shifted = i - i_min
            j_shifted = j - j_min

            Rbc[i_shifted, j_shifted] = value

# plt.pcolor(Rbc, cmap="viridis")
plt.pcolor(0.5*np.log10(Rbc**2), cmap="viridis")
plt.colorbar(label="$\log(R^2)/2$")

x_ticks = np.arange(m)  # Original x-tick positions
y_ticks = np.arange(n)  # Original y-tick positions

shifted_x_ticks = x_ticks + 0.5  # Shift x-ticks by 0.5
shifted_y_ticks = y_ticks + 0.5  # Shift y-ticks by 0.5

ax = plt.gca()

ax.set_xticks(x_ticks, minor=False)
ax.set_yticks(y_ticks, minor=False)
plt.grid(True, which="major", color="k", linestyle="-", linewidth=0.5)

ax.set_xticks(shifted_x_ticks, minor=True)
ax.set_yticks(shifted_y_ticks, minor=True)

ax.set_xticklabels(x_ticks + j_min, minor=True)  # Shift back to original j indices
ax.set_yticklabels(y_ticks + i_min, minor=True)  # Shift back to original i indices

ax.set_xticklabels([], minor=False)
ax.set_yticklabels([], minor=False)

ax.tick_params(axis='x', which='minor', bottom=False)
ax.tick_params(axis='y', which='minor', left=False)

plt.xlabel("M")
plt.ylabel("N")
plt.title("Boundary harmonics Rbc[m,n]")

plt.tight_layout()
plt.show()