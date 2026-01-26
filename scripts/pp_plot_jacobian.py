#!/usr/bin/env python
from argparse import ArgumentParser

parser = ArgumentParser(
    description="Plot KAM surfaces and Poincare sections from .h5 files."
)
parser.add_argument(
    "filenames", metavar="F", type=str, nargs="+", help="Input .h5 files to process"
)
parser.add_argument(
    "-z",
    "--zetan",
    nargs="+",
    type=float,
    default=[0.0],
    help="Normalized (0->1) zeta value for plotting interfaces (default: 0.0) (normalized to a field-period)",
)

args = parser.parse_args()

from spectre import SPECout
import matplotlib
matplotlib.use("Qt5Agg")  # must come before importing plt
plt = matplotlib.pyplot

for iz in range(len(args.zetan)):
    for fname in args.filenames:    
        o = SPECout(fname)
    
        o.plot_jacobian(label=fname, zetan=args.zetan[iz])

    plt.tight_layout()
plt.show()
