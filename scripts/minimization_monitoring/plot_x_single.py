#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(description="Plot SPECTRE interfaces from configuration file and data files")

parser.add_argument("config_file", help="SPECTRE configuration file")
parser.add_argument(
    "data_files",
    nargs="+",
    help='Data files to plot. Use "o" to plot original xinit from SPECTRE wrapper, '
    "or provide .npy files with data arrays",
)
parser.add_argument(
    "-z",
    "--zetan",
    type=float,
    default=0.0,
    help="Zeta parameter for plotting in range [0, 1.0] (default: 0.0)",
)
parser.add_argument(
    "-i", "--infaces", type=int, nargs="+", help="Interfaces to plot (default: all)", default=None
)
args = parser.parse_args()

from numpy import load
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

from spectre import SPECTRE, plot_infaces, get_xinit_specwrap

test = SPECTRE.from_input_file(args.config_file, verbose=False)

fig, ax = plt.subplots(figsize=(11, 7))

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

for i, data_file in enumerate(args.data_files):
    if data_file == "o":
        x = get_xinit_specwrap(test)
    else:
        x = load(data_file)

    plot_infaces(
        x, test, ax=ax, c=colors[i % len(colors)], lw=1.2, zetan=args.zetan, label=data_file, surfaces=args.infaces
    )

ax.axis("equal")

han, lab = ax.get_legend_handles_labels()
plt.legend(han[-25:], lab[-25:])

plt.tight_layout()

plt.show()
