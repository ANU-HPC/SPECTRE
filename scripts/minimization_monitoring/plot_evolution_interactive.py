#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Interactively slide through KAM surfaces for an optimization run"
)
parser.add_argument("input_file", type=str, help="Path to the SPECTRE input file")
parser.add_argument(
    "data_files",
    nargs="+",
    help='Data files to process. Use "o" to use original xinit from SPECTRE wrapper, ',
)
parser.add_argument(
    "-z",
    "--zetan",
    type=float,
    default=0.0,
    help="Initial normalized zeta value for plotting (default: 0)",
)
parser.add_argument(
    "--plot-axis",
    action="store_false",
    help="If set, do not plot axis (default True)",
)
parser.add_argument(
    "--plot-theta-grid",
    action="store_true",
    help="If set, plot theta grid (default False)",
)
args = parser.parse_args()

from numpy import load
import numpy as np
import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from spectre import plot_infaces, SPECTRE

test = SPECTRE(args.input_file, verbose=False)

xinit = test.xin

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

num_iter = len(args.data_files)

fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.22)


def draw(i, zetan):
    ax.clear()
    plot_infaces(
        xs[i],
        test,
        c="r",
        ax=ax,
        lw=1.2,
        zetan=zetan,
        plot_theta_grid=args.plot_theta_grid,
        plot_axis=args.plot_axis,
    )
    ax.set_title(f"iter {i}: {labels[i]}  (zetan={zetan:.3f})")
    ax.axis("equal")


# Sliders
ax_iter = plt.axes([0.15, 0.10, 0.7, 0.03])
ax_zeta = plt.axes([0.15, 0.05, 0.7, 0.03])

iter_max = max(num_iter - 1, 0)
slider_iter = Slider(
    ax_iter,
    "iter",
    0,
    iter_max,
    valinit=0,
    valstep=1,
    valfmt="%d",
)
slider_zeta = Slider(
    ax_zeta,
    "zetan",
    0.0,
    1.0,
    valinit=args.zetan,
)


def on_change(_val):
    i = int(slider_iter.val)
    zetan = float(slider_zeta.val)
    draw(i, zetan)
    fig.canvas.draw_idle()


slider_iter.on_changed(on_change)
slider_zeta.on_changed(on_change)

draw(0, args.zetan)

plt.show()
