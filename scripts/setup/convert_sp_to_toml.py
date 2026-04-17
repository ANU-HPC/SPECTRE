#!/usr/bin/env python
from argparse import ArgumentParser

parser = ArgumentParser(description="Convert an old-style SPEC input .sp file into a .toml file.")

parser.add_argument("spfile", help="The .sp input file")
parser.add_argument('-o',"--output", help="Name of the output TOML file (default: <spfile>.toml)", default=None)

args = parser.parse_args()

from spectre.file_io import specfile_to_toml

specfile_to_toml(args.spfile, args.output)