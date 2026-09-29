#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import matplotlib

matplotlib.use("Qt5Agg")

import sys

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from re import compile

parser = ArgumentParser(prog="plotout", description="Plot SPECTRE optimization log files")
parser.add_argument("log_files", nargs="+", help="Path(s) to log file(s) to plot")
parser.add_argument(
    "-s", "--same-fig", action="store_true", help="Plot all files on the same figure"
)
args = parser.parse_args()

ms = 2.0
label_fontsize = 11
COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

N_PANELS = 7  # cost, <f>, freal, step norm, cost reduction, optimality, bnserr

if args.same_fig:
    fig, axes = plt.subplots(N_PANELS, 1, figsize=(10, 10.5), sharex=True)
    max_iters = 0

for log_file in args.log_files:
    try:
        with open(log_file, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found.")
        sys.exit(1)

    VAR_PATTERNS = [
        (compile(r"iter\s+(\d+)"), "iter"),
        (compile(r"nfev\s+(\d+)"), "nfev"),
        (compile(r"cost\s+([\d\.eE+-]+)"), "cost"),
        (compile(r"<f>\s+([\d\.eE+-]+)"), "f_avg"),  # handles <f> notation
        (compile(r"fr\s+([\d\.eE+-]+)"), "f_real"),  # handles freal notation
        (compile(r"f_real\s+([\d\.eE+-]+)"), "f_real"),  # alternate spelling
        (compile(r"stpnrm\s+([\d\.eE+-]+)"), "stpnrm"),
        (compile(r"costred\s+([\d\.eE+-]+)"), "costred"),
        (compile(r"optim\s+([\d\.eE+-]+)"), "optim"),
    ]

    # iBns refreshes are logged by utils.update_bnorm between inner solves, not on the
    # per-iteration lines, so bnserr carries the iteration it was last seen at.
    BNSERR_PATTERN = compile(r"bnserr\s+([\d\.eE+-]+)")

    SERIES = ("nfev", "cost", "f_avg", "f_real", "stpnrm", "costred", "optim")
    # Every series is padded to the length of `iterations` so all of them share that
    # x-axis. Iteration 0 reports no step norm or cost reduction, and an outer/inner run
    # reports gaps wherever a cycle restarts; nan leaves those points unplotted instead
    # of silently shifting the rest of the series by one.
    data = {name: [] for name in SERIES}
    iterations = []
    bnserr_iters, bnserr_vals = [], []
    last_iter = 0

    for line in lines:
        if not line.strip():
            continue

        if not line.startswith("iter"):
            match = BNSERR_PATTERN.search(line)
            if match:
                try:
                    bnserr_vals.append(float(match.group(1)))
                    bnserr_iters.append(last_iter)
                except ValueError:
                    pass
            continue

        found_values = {}
        for pattern, var_name in VAR_PATTERNS:
            match = pattern.search(line)
            if match and var_name not in found_values:
                value = match.group(1)
                try:
                    if "." in value or "e" in value.lower():
                        found_values[var_name] = float(value)
                    else:
                        found_values[var_name] = int(value)
                except ValueError:
                    pass

        if "iter" not in found_values:
            continue

        last_iter = found_values["iter"]
        iterations.append(last_iter)
        for name in SERIES:
            data[name].append(found_values.get(name, np.nan))

    cost, f_avg, f_real = data["cost"], data["f_avg"], data["f_real"]
    stpnrm, costred, optim = data["stpnrm"], data["costred"], data["optim"]

    color = COLORS[args.log_files.index(log_file) % len(COLORS)]
    label = log_file

    if not args.same_fig:
        fig, axes = plt.subplots(N_PANELS, 1, figsize=(10, 10.5), sharex=True, num=log_file)

    PANELS = [
        (0, iterations, cost, "Cost"),
        (1, iterations, f_avg, "<f>"),
        (2, iterations, f_real, "freal"),
        (3, iterations, stpnrm, "Step Norm"),
        (4, iterations, costred, "Cost Reduction"),
        (5, iterations, optim, "Optim"),
        (6, bnserr_iters, bnserr_vals, "bnserr"),
    ]

    for idx, xdata, ydata, ylabel in PANELS:
        if not ydata:
            continue
        c = COLORS[idx % len(COLORS)] if not args.same_fig else color
        axes[idx].semilogy(xdata, ydata, ms=ms, marker="o", color=c, label=label)
        axes[idx].set_ylabel(ylabel, fontsize=label_fontsize)

    axes[-1].set_xlabel("Iteration", fontsize=13)

    if args.same_fig:
        max_iters = max(max_iters, max(iterations, default=0))
    else:
        axes[-1].set_xlim(-2, max(iterations, default=0) + 2)
        axes[1].set_ylim(auto=True)
        plt.tight_layout()

if args.same_fig:
    axes[-1].set_xlim(-2, max_iters + 2)
    axes[1].set_ylim(auto=True)
    for ax in axes:
        ax.legend(fontsize=10)
    plt.tight_layout()


def _rescale_y_on_xlim_change(axes):
    """When any subplot's x-limits change (zoom/pan), rescale all subplots'
    y-axes to fit the visible data within the new x-range."""

    def on_xlim_changed(event_ax):
        for ax in axes:
            xlo, xhi = ax.get_xlim()
            y_visible = []
            for line in ax.get_lines():
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
                mask = (xdata >= xlo) & (xdata <= xhi)
                if mask.any():
                    y_visible.extend(ydata[mask])
            if y_visible:
                ymin, ymax = min(y_visible), max(y_visible)
                if ax.get_yscale() == "log":
                    if ymin > 0 and ymax > 0:
                        log_margin = (np.log10(ymax) - np.log10(ymin)) * 0.05
                        ax.set_ylim(
                            10 ** (np.log10(ymin) - log_margin), 10 ** (np.log10(ymax) + log_margin)
                        )
                else:
                    margin = (ymax - ymin) * 0.05
                    ax.set_ylim(ymin - margin, ymax + margin)
        ax.figure.canvas.draw_idle()

    for ax in axes:
        ax.callbacks.connect("xlim_changed", on_xlim_changed)


# Connect the rescale callback to all open figures
for fig_num in plt.get_fignums():
    fig = plt.figure(fig_num)
    _rescale_y_on_xlim_change(fig.axes)

plt.show()
