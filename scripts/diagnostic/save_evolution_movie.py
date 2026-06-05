#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Render the evolution of KAM surfaces for an optimization run to an MP4 movie.

The output is encoded with H.264 (libx264) using the yuv420p pixel format, which
is required for the video to play inside Slack, browsers and QuickTime.

Example:
    python save_evolution_movie.py init.toml xs_freeb/* -o evolution.mp4
"""

from argparse import ArgumentParser

parser = ArgumentParser(description="Render evolution of KAM surfaces to an MP4 movie")
parser.add_argument("input_file", type=str, help="Path to the SPECTRE input file")
parser.add_argument(
    "data_files",
    nargs="+",
    help='Data files to process. Use "o" to use original xinit from SPECTRE wrapper.',
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="evolution.mp4",
    help="Output movie file (default: evolution.mp4)",
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
    "--fps",
    type=int,
    default=25,
    help="Frames per second for the movie (default 25)",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=150,
    help="Resolution of the movie in dots per inch (default 150)",
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

import numpy as np
from numpy import load
import matplotlib

# Headless backend: we are only saving a file, not showing a window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from spectre import plot_infaces, SPECTRE, get_xinit_specwrap

if not animation.writers.is_available("ffmpeg"):
    raise RuntimeError(
        "ffmpeg writer is not available to matplotlib. Install ffmpeg (e.g. `apt install ffmpeg`)."
    )

num_iter = args.num_iter if args.num_iter != -1 else len(args.data_files)

test = SPECTRE(args.input_file, verbose=False)

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
        xs[i],
        test,
        c="r",
        ax=ax,
        lw=1.2,
        zetan=args.zetan,
        plot_theta_grid=args.plot_theta_grid,
        plot_axis=args.plot_axis,
    )

    ax.set_title(f"iter {i}: {labels[i]}")
    ax.axis("equal")
    plt.tight_layout()


ani = animation.FuncAnimation(
    fig,
    update,
    frames=range(0, num_iter, args.step_iter),
    blit=False,
)

# H.264 + yuv420p so the file is playable in Slack/browsers/QuickTime.
# `-pix_fmt yuv420p` requires even frame dimensions; libx264 handles odd sizes
# poorly, so we let ffmpeg pad to even via the scale filter as a safety net.
writer = animation.FFMpegWriter(
    fps=args.fps,
    codec="libx264",
    extra_args=[
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-movflags",
        "+faststart",
    ],
)

print(f"Rendering {len(range(0, num_iter, args.step_iter))} frames to {args.output} ...")
ani.save(args.output, writer=writer, dpi=args.dpi)
print(f"Saved movie to {args.output}")
