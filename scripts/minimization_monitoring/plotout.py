#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import sys

import matplotlib
import numpy as np
from argparse import ArgumentParser
from re import compile, search

parser = ArgumentParser(prog="plotout", description="Plot SPECTRE optimization log files")
parser.add_argument("log_files", nargs="+", help="Path(s) to log file(s) to plot")
parser.add_argument(
    "-s", "--same-fig", action="store_true", help="Plot all files on the same figure"
)
parser.add_argument(
    "--save",
    default=None,
    help="Save the figure(s) to this PNG path (headless Agg) instead of showing interactively.",
)
args = parser.parse_args()

# Pick the backend AFTER parsing args: Agg for headless saving, Qt5Agg for
# interactive use. Must precede `import matplotlib.pyplot`.
matplotlib.use("Agg" if args.save else "Qt5Agg")

import matplotlib.pyplot as plt

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

if args.same_fig:
    # bnserr (axes[6]) updates only per outer refresh, but is tagged with the
    # iteration at which it fired so all 7 panels share the iteration x-axis.
    fig, axes = plt.subplots(7, 1, figsize=(10, 10.5))
    for ax in axes[1:]:
        ax.sharex(axes[0])
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

    # Each series keeps (x, y) where x is the actual iteration number, so panels
    # stay aligned on the shared x-axis even when a quantity is missing at some
    # iterations. iter 0 (the initial point) carries cost / <f> / freal but no
    # step / cost-reduction / optimality yet, so those series simply skip it.
    iterations = []
    series = {key: ([], []) for key in ("cost", "f_avg", "f_real", "stpnrm", "costred", "optim")}
    bnserr = ([], [])  # (iteration, value): one point per outer refresh
    last_iter = None

    for line in lines:
        if not line.strip():
            continue

        # bnserr is logged ("B_norm updated:  bnserr ...") once per outer refresh,
        # right after the last iter line of the cycle; tag it with that iteration
        # so it lands at the iteration where it was actually evaluated.
        if line.startswith("B_norm updated"):
            m = search(r"bnserr\s+([\d\.eE+-]+)", line)
            if m and last_iter is not None:
                try:
                    bnserr[0].append(last_iter)
                    bnserr[1].append(float(m.group(1)))
                except ValueError:
                    pass
            continue

        if not line.startswith("iter"):
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
        for key, (xs, ys) in series.items():
            if key in found_values:
                xs.append(last_iter)
                ys.append(found_values[key])

    color = COLORS[args.log_files.index(log_file) % len(COLORS)]
    label = log_file

    if not args.same_fig:
        fig, axes = plt.subplots(7, 1, figsize=(10, 10.5), num=log_file)
        for ax in axes[1:]:
            ax.sharex(axes[0])

    PANELS = [
        ("cost", "Cost"),
        ("f_avg", "<f>"),
        ("f_real", "freal"),
        ("stpnrm", "Step Norm"),
        ("costred", "Cost Reduction"),
        ("optim", "Optim"),
    ]
    for idx, (key, ylabel) in enumerate(PANELS):
        xs, ys = series[key]
        if ys:
            c = COLORS[idx % len(COLORS)] if not args.same_fig else color
            axes[idx].semilogy(xs, ys, ms=ms, marker="o", color=c, label=label)
        axes[idx].set_ylabel(ylabel, fontsize=label_fontsize)

    axes[5].set_xlabel("Iteration", fontsize=13)

    if bnserr[0]:
        c = COLORS[6 % len(COLORS)] if not args.same_fig else color
        axes[6].semilogy(bnserr[0], bnserr[1], ms=ms, marker="o", color=c, label=label)
    axes[6].set_ylabel("bnserr", fontsize=label_fontsize)
    axes[6].set_xlabel("Iteration", fontsize=13)

    if args.same_fig:
        max_iters = max(max_iters, max(iterations, default=0))
    else:
        axes[5].set_xlim(min(iterations, default=0) - 2, max(iterations, default=0) + 2)
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
                        ax.set_ylim(
                            10 ** (np.log10(ymin) - log_margin), 10 ** (np.log10(ymax) + log_margin)
                        )
                else:
                    margin = (ymax - ymin) * 0.05
                    ax.set_ylim(ymin - margin, ymax + margin)
        ax.figure.canvas.draw_idle()

    for ax in axes:
        ax.callbacks.connect("xlim_changed", on_xlim_changed)


if args.save:
    # Headless: save each open figure. One figure -> the given path; multiple
    # figures -> suffix by index.
    figs = plt.get_fignums()
    for i, fig_num in enumerate(figs):
        fig = plt.figure(fig_num)
        out = args.save if len(figs) == 1 else args.save.replace(".png", f"_{i}.png")
        fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved {len(figs)} figure(s) -> {args.save}")
else:
    # Connect the rescale callback to all open figures (interactive only)
    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        _rescale_y_on_xlim_change(fig.axes)
    plt.show()
