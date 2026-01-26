#!/usr/bin/env python

from argparse import ArgumentParser
parser = ArgumentParser(prog='runspec', description='A script for runing python wrapper of the SPECTRE code')
parser.add_argument('inputfile', help="Path to the input file")
parser.add_argument('-s', '--save-output', help="Save the HDF5 and restart files", type=bool, default=True)
parser.add_argument('-v', '--verbose', help="Enable verbose output", action="store_true")
args = parser.parse_args()

from time import time

time_start = time()

from spectre import SPECTRE

test = SPECTRE.from_input_file(args.inputfile, verbose=args.verbose)

rank = test.comm.rank

if rank == 0:
	print(f"Running SPECTRE with input from '{args.inputfile}'", flush=True)


test.run(save_output=args.save_output)

time_end = time()

if rank == 0:
	print(f"SPECTRE finished ({time_end-time_start:.2f} s)")
