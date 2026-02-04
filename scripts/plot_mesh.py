#!/usr/bin/env python

from argparse import ArgumentParser

parser = ArgumentParser(description="Plot SPECTRE grid and KAM surfaces.")
parser.add_argument("input_file", help="Path to the SPECTRE input file.")
parser.add_argument('-z', '--zetan', type=float, default=0.0, help='Zeta parameter for plotting in range [0, 1.0] (default: 0.0)')
parser.add_argument("-nt", "--ntheta", type=int, default=64, help="Number of theta points (default: 128).")
parser.add_argument(
    "-v", "--volumes",
    type=int,
    nargs='*',
    default=None,
    help="List of integer volume indices to label (e.g., --volumes 1 3 5). Default is None."
)
args = parser.parse_args()

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from spectre import SPECTRE, get_xinit_specwrap, plot_grid, plot_infaces

test = SPECTRE.from_input_file(args.input_file, verbose=True)

xin = get_xinit_specwrap(test)

fig, ax = plt.subplots(figsize=(10, 7))
plot_grid(
    xin,
    test,
    ax=ax,
    zeta=args.zetan * (2*np.pi/test.input_list_mod.nfp),
    thetas=np.linspace(0, 2 * np.pi, args.ntheta),
    volumes=args.volumes,
    lw=1.1,
    marker='',
    ms=2
)
plot_infaces(
    xin,
    test,
    zetan=args.zetan,
    ax=ax,
    plot_axis=True,
    color='lawngreen'
)

plt.axis('equal')
plt.tight_layout()
plt.show()
