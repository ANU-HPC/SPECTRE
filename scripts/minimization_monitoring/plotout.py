#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from re import compile, search

parser = ArgumentParser(prog="plotout", description="Plot SPECTRE optimization log files")
parser.add_argument("log_files", nargs="+", help="Path(s) to log file(s) to plot")
parser.add_argument('-s', "--same-fig", action="store_true", help="Plot all files on the same figure")
args = parser.parse_args()

ms = 2.0
label_fontsize = 11
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

if args.same_fig:
    fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True)
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

    iterations, nfev, cost = [], [], []
    f_avg, f_real, stpnrm = [], [], []
    costred, optim = [], []

    for line in lines:
        if not line.strip() or not line.startswith("iter"):
            continue

        if search(r"iter\s+0\b", line):
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

        if "iter" in found_values:
            iterations.append(found_values["iter"])
        if "nfev" in found_values:
            nfev.append(found_values["nfev"])
        if "cost" in found_values:
            cost.append(found_values["cost"])
        if "f_avg" in found_values:
            f_avg.append(found_values["f_avg"])
        if "f_real" in found_values:
            f_real.append(found_values["f_real"])
        if "stpnrm" in found_values:
            stpnrm.append(found_values["stpnrm"])
        if "costred" in found_values:
            costred.append(found_values["costred"])
        if "optim" in found_values:
            optim.append(found_values["optim"])

    color = COLORS[args.log_files.index(log_file) % len(COLORS)]
    label = log_file

    if not args.same_fig:
        fig, axes = plt.subplots(6, 1, figsize=(10, 9), sharex=True, num=log_file)

    if cost:
        c = COLORS[0 % len(COLORS)] if not args.same_fig else color
        axes[0].semilogy(cost, ms=ms, marker="o", color=c, label=label)
        axes[0].set_ylabel("Cost", fontsize=label_fontsize)

    if f_avg:
        c = COLORS[1 % len(COLORS)] if not args.same_fig else color
        axes[1].semilogy(f_avg, ms=ms, marker="o", color=c, label=label)
        axes[1].set_ylabel("<f>", fontsize=label_fontsize)

    if f_real:
        c = COLORS[2 % len(COLORS)] if not args.same_fig else color
        axes[2].semilogy(f_real, ms=ms, marker="o", color=c, label=label)
        axes[2].set_ylabel("freal", fontsize=label_fontsize)

    if stpnrm:
        c = COLORS[3 % len(COLORS)] if not args.same_fig else color
        axes[3].semilogy(stpnrm, ms=ms, marker="o", color=c, label=label)
        axes[3].set_ylabel("Step Norm", fontsize=label_fontsize)

    if costred:
        c = COLORS[4 % len(COLORS)] if not args.same_fig else color
        axes[4].semilogy(costred, ms=ms, marker="o", color=c, label=label)
        axes[4].set_ylabel("Cost Reduction", fontsize=label_fontsize)

    if optim:
        c = COLORS[5 % len(COLORS)] if not args.same_fig else color
        axes[5].semilogy(optim, ms=ms, marker="o", color=c, label=label)
        axes[5].set_ylabel("Optim", fontsize=label_fontsize)

    axes[5].set_xlabel("Iteration", fontsize=13)

    if args.same_fig:
        max_iters = max(max_iters, len(iterations))
    else:
        axes[5].set_xlim(-2, len(iterations) + 2)
        axes[1].set_ylim(auto=True)
        plt.tight_layout()

if args.same_fig:
    axes[5].set_xlim(-2, max_iters + 2)
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
                        ax.set_ylim(10 ** (np.log10(ymin) - log_margin),
                                    10 ** (np.log10(ymax) + log_margin))
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
