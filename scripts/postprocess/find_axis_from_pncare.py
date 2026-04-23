#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(description="Change angular resolution of an sp file.")
parser.add_argument('in_name', help='Name of the .toml file to be analyzed')
parser.add_argument('-o', '--out_name', help='Output .toml file name (default: .temp.toml)')
parser.add_argument("-p", "--plot", action="store_true", help="Whether to plot the axis analysis")
args = parser.parse_args()

if args.out_name is None:
    out_name = '.temp.toml'
else:
    out_name = args.out_name

from spectre import find_axis_from_pncare

find_axis_from_pncare(args.in_name, out_name, args.plot)