#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(description="Show evolution of KAM surfaces for an optimization run")
parser.add_argument("input_file", type=str, help="Path to the SPECTRE input file")
parser.add_argument(
    "data_files",
    nargs="+",
    help='Data files to process. Use "o" to use original xinit from SPECTRE wrapper, ',
)
parser.add_argument(
    "-n",
    "--num-iter",
    type=int,
    default=-1,
    help="Number of iteration steps to display (default all)",
)
parser.add_argument(
    "-s",
    "--step-iter",
    type=int,
    default=1,
    help="Step for iteration to display (default 1)",
)
parser.add_argument(
    "-z",
    "--zetan",
    type=float,
    default=0.0,
    help="Normalized zeta value for plotting (default: 0)",
)
parser.add_argument(
    "-i",
    "--interval",
    type=int,
    default=40,
    help="Interval for the animation in milliseconds (default 40)",
)
parser.add_argument(
    "--plot-axis",
    action="store_false",
    help="If set, do not plot axis (default True)",
)
parser.add_argument(
    "--plot-theta-grid",
    action="store_true",
    help="If set, plot axis (default False)",
)
parser.add_argument(
    "--save-html",
    action="store_false",
    help="If set, save the animation as an HTML file (default True)",
)
args = parser.parse_args()

from numpy import load
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, rcParams
from spectre import plot_infaces, SPECTRE, get_xinit_specwrap

rcParams["animation.embed_limit"] = 80.0

num_iter = args.num_iter if args.num_iter != -1 else len(args.data_files)

test = SPECTRE.from_input_file(args.input_file, verbose=False)

xinit = get_xinit_specwrap(test)

# Load data files
xs = np.empty((len(args.data_files), len(xinit)))
labels = []
for i, data_file in enumerate(args.data_files):
    if data_file == "o":
        x = xinit
        label = "original"
    else:
        x = load(data_file)
        label = data_file
    xs[i, :] = x
    labels.append(label)

fig, ax = plt.subplots(figsize=(10, 6))

def update(i):
    ax.clear()
    plot_infaces(
        xs[i], test, c="r", ax=ax, lw=1.2, zetan=args.zetan,
        plot_theta_grid=args.plot_theta_grid, plot_axis=args.plot_axis
    )

    ax.set_title(f"iter {i}: {labels[i]}")
    ax.axis("equal")
    plt.tight_layout()


ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(0, num_iter, args.step_iter),
    blit=False,
    interval=args.interval,
)

if args.save_html:
    with open("anim.html", "w") as f:
        print(ani.to_jshtml(10), file=f)

plt.show()
