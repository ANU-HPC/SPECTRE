#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="runspec", description="A script for setting-up the Henneberg surfaces for SPECTRE"
)
parser.add_argument("inputfile", help="Path to the input file")
parser.add_argument(
    "-o", "--output", help="Name of the output TOML file (default: <spfile>.toml)", default=None
)
args = parser.parse_args()

from time import time

time_start = time()

from spectre import SPECTRE, force_real, get_xinit_specwrap, get_freal_fsa_metric, print_all_errors
from spectre.surfaces.surf_henneberg import SurfaceHenneberg
from spectre.file_io.toml_io import write_input_parameters_to_toml

test = SPECTRE.from_input_file(args.inputfile)

if test.min_params.inface_repr != "henn":
    raise RuntimeError("Inface representation is not Henneberg")

rank = test.comm.rank

xin = get_xinit_specwrap(test)
bnd_rc = test.allglobal_mod.irbc[:, -1].copy()
bnd_zs = test.allglobal_mod.izbs[:, -1].copy()

mmax = test.min_params.mmax
nmax = test.min_params.nmax

surfaces = SurfaceHenneberg.surfaces_from_spectre_xmn(
    xin, bnd_rc, bnd_zs, test, alpha_fac=1, nmax=nmax, mmax=mmax, ntheta=64, nphi=64
)

xmn, bnd_new_rc, bnd_new_zs = SurfaceHenneberg.surfaces_to_spectre_xmn(surfaces, test)

test.allglobal_mod.irbc[:, -1] = bnd_new_rc.copy()
test.allglobal_mod.izbs[1:, -1] = bnd_new_zs.copy()

force_real(xmn, test)
fsa_err = get_freal_fsa_metric(test)
if test.comm.Get_rank() == 0:
    print(f"\n  <f> = {fsa_err:.8e}")

beltrami_errors_avg, beltrami_errors_max, f_errs_rms, f_errs_max = print_all_errors(
    test, beltrami=True, force=True
)

test.run(save_output=False)

if args.output is not None:
    test.wrapper_funcs_mod.prepare_endfile_writing()
    write_input_parameters_to_toml(
        test.parameters, args.output
    )  # write the final parameters to a toml file
