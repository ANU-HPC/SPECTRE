#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(description="Initialize interfaces for a SPECTRE input file")
parser.add_argument("input_file", type=str, help="Path to the SPECTRE input file")
parser.add_argument(
    "-t",
    "--tflux",
    type=float,
    nargs="+",
    default=None,
    required=False,
    help="Target flux values (default: tflux in the input file)",
)
parser.add_argument(
    "-i",
    "--base_inface",
    type=int,
    default=None,
    help="Base inface to construct new infaces (default: boundary)",
)
parser.add_argument(
    "-o", "--fname_out", type=str, default=None, help="Output filename for generated interfaces"
)
parser.add_argument(
    "-p",
    "--plot",
    action="store_true",
    default=False,
    help="Flag to enable plotting of KAM surfaces (default: False)",
)
args = parser.parse_args()

from spectre import SPECTRE, gen_infaces, get_xinit_specwrap, pack_interfaces, plot_infaces

test = SPECTRE(args.input_file)

xin = get_xinit_specwrap(test)

reven, rodd, zeven, zodd = gen_infaces(
    test,
    xin,
    tflux=args.tflux,
    set_to_file=True,
    base_inface=args.base_inface,
    fname_out=args.fname_out,
)

xnew = pack_interfaces(reven.T, rodd.T, zeven.T, zodd.T, test)

if args.plot:
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt

    zeta = 0

    fig, ax = plt.subplots(figsize=(10, 7))
    plot_infaces(xin, test, ax=ax, c="b", zetan=zeta, surfaces=None, label="Initial")
    plot_infaces(xnew, test, ax=ax, c="r", zetan=zeta, surfaces=None, label="Generated m2d")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()
