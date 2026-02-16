#!/usr/bin/env python

from argparse import ArgumentParser

parser = ArgumentParser(prog='runspec', description='A script for runing python wrapper of the SPECTRE code')
parser.add_argument('inputfile', help="Path to the input file")
parser.add_argument('-s', '--save-output', help="Save the HDF5 and restart files", type=bool, default=True)
parser.add_argument('-v', '--verbose', help="Enable verbose output", action="store_true")
args = parser.parse_args()

from time import time
from h5py import File

time_start = time()

from spectre import SPECTRE, print_all_errors, force_real, get_xinit_specwrap, get_freal_fsa_metric, save_errors_to_h5

test = SPECTRE.from_input_file(args.inputfile, verbose=args.verbose)

rank = test.comm.rank

if rank == 0:
	print(f"Running calc_spectre_field with input '{args.inputfile}'", flush=True)

test.run(save_output=args.save_output)

time_end = time()

if rank == 0:
	print(f"Finished ({time_end-time_start:.2f} s)", flush=True)

xin = get_xinit_specwrap(test)

force_real(xin, test)
fsa_err = get_freal_fsa_metric(test)
if test.comm.Get_rank() == 0:
    print(f"\n  <f> = {fsa_err:.8e}")

beltrami_errors_avg, beltrami_errors_max, f_errs_rms, f_errs_max = print_all_errors(test, beltrami=True, force=True)

if rank == 0:        
	save_errors_to_h5(test.allglobal_mod.filename_h5, fsa_err, beltrami_errors_avg, beltrami_errors_max, f_errs_rms, f_errs_max)