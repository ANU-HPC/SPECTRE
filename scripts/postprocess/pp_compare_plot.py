#!/usr/bin/env python
from argparse import ArgumentParser
parser = ArgumentParser(
    description="Compare surfaces (and poincare plots) of two or more SPECTRE equilibria."
)
parser.add_argument(
    "fnames",
    nargs="+",
    help="List of SPECTRE equilibria .h5 files (at least one required)",
)
parser.add_argument(
    "-z",
    "--zetan",
    nargs="+",
    type=float,
    default=[0.0],
    help="Normalized (0->1) zeta value for plotting interfaces (default: 0.0) (normalized to a field-period)",
)
parser.add_argument(
    "-p",
    "--poincare",
    help="Options to plot the poincare (e.g. f1l2u3 plots 1st eq fully, 2nd eq only lower part, 3rd eq only upper part)",
    action="store",
    const="all",
    nargs="?",
)
parser.add_argument(
    "-r",
    "--range",
    help="Options to plot the kam surfaces (e.g. f1l2u3 plots 1st eq fully, 2nd eq only lower part, 3rd eq only upper part)",
    action="store",
    const="all",
    nargs="?",
)
args = parser.parse_args()

import numpy as np

from spectre import SPECTREout

import matplotlib
matplotlib.use("Qt5Agg")  # must come before importing plt
plt = matplotlib.pyplot

num_files = len(args.fnames)

kam_opts = args.range
if kam_opts == 'all' or kam_opts is None:
    pranges_kam = ['full'] * num_files
else:
    pranges_kam = [None] * num_files
    for i in range(len(kam_opts)//2):
        flag = kam_opts[i*2+1]
        if flag=='f':
            flag = 'full'
        elif flag=='u':
            flag = 'upper'
        elif flag=='l':
            flag = 'lower'
        else:
            flag = None
        pranges_kam[int(kam_opts[i*2])] = flag

pncare_opts = args.poincare
if pncare_opts is None:
    pranges = [None]*num_files
elif pncare_opts == 'all':
    pranges = pranges_kam
else:
    pranges = [None]*num_files
    for i in range(len(pncare_opts)//2):
        flag = pncare_opts[i*2+1]
        if flag=='f':
            flag = 'full'
        elif flag=='u':
            flag = 'upper'
        elif flag=='l':
            flag = 'lower'
        else:
            flag = None
        pranges[int(pncare_opts[i*2])] = flag

colors =  matplotlib.colormaps['Set1_r'].colors[::-1]
colors = colors*2

colors_dark = []
for cr in range(20):
    colors_dark.append(tuple((np.array(colors[cr%len(colors)]) + (0.,0.,0.)) * 1.))

lss = ["-"] * 20

legends=[]

for iz in range(len(args.zetan)):
    fig, ax = plt.subplots(figsize=(8, 9))

    for i, f in enumerate(args.fnames):
        fname = f
        if fname[-5:] == ".toml":
            fname = fname[:-5] + ".h5"
        elif fname[-3:] != ".h5":
            raise ValueError("Bad fname argument!")

        o = SPECTREout(fname)

        if o.input.diagnostics.nPpts > 0 and pranges[i] is not None:
            o.plot_poincare(c=colors[i], ax=ax, prange=pranges[i], ms=1.2, zetan=args.zetan[iz], alpha=0.9, label=f)

        o.plot_infaces(
            color=colors_dark[i], ls=lss[i], ax=ax, prange=pranges_kam[i], zetan=args.zetan[iz], plot_axis=True, lw=1.3
        )

        legends.append(matplotlib.lines.Line2D([], [], color=colors_dark[i], marker='', markersize=15, label=f"Infaces {f}"))
        legends.append(matplotlib.lines.Line2D([], [], color=colors[i], marker='.', linewidth=0, markersize=7, label=f"Pncare {f}"))

    ax.legend(handles=legends, fontsize=12, title=f"@ $\\zeta_n$={args.zetan[iz]:.2f}")

    plt.tight_layout()

plt.show()
