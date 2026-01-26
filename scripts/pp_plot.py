#!/usr/bin/env python
from argparse import ArgumentParser
parser = ArgumentParser(
    description="Plot interfaces and poincare trace"
)
parser.add_argument(
    "filenames", type=str, nargs="+", help="Input .h5 file to process"
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
    action="store_true",
    help="Plot poincare traces if the tracing has been done (default: False)",
)
args = parser.parse_args()

from spectre import SPECout 

import matplotlib
matplotlib.use("Qt5Agg")  # must come before importing plt
plt = matplotlib.pyplot


for iz in range(len(args.zetan)):
    for fname in args.filenames:
        if fname[-5:] == ".toml":
            fname = fname[:-5] + ".h5"
        elif fname[-3:] != ".h5":
            raise ValueError("Input file must have .h5 extension")

        fig, ax = plt.subplots()
    
        o = SPECout(fname)
    
        o.plot_infaces(label=fname, zetan=args.zetan[iz], ax=ax, plot_axis=True)
    
        if args.poincare and o.input.diagnostics.nPpts > 0:
            o.plot_poincare(ax=ax, zetan=args.zetan[iz])
   
        ax.set_title(f"{fname}  @  $\\zeta_n$={args.zetan[iz]}")

        ax.axis('equal')

        plt.tight_layout()

plt.show()
