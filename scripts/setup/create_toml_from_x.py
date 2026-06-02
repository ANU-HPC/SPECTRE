#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

parser = ArgumentParser(
    description="Plot SPECTRE interfaces from configuration file and data files"
)
parser.add_argument("config_file", help="SPECTRE configuration file")
parser.add_argument("data_file", help="Npy file with the interface information")
args = parser.parse_args()

from numpy import load
from spectre import SPECTRE, check_intersection_wrapper
from spectre.file_io import write_input_parameters_to_toml

test = SPECTRE(args.config_file, verbose=False)

# test.run()

x = load(args.data_file)

# force_real(x, test)

test.xin = x

# test.allglobal_mod.skip_write = True
# test.lib.simulation_mod.final_diagnostics()

# fsa = get_freal_fsa_metric(test)

if test.comm.rank == 0:
    # print(f"<f>  {fsa:.4e}")

    # generate parameters from fortran memory and write to toml:
    test.wrapper_funcs_mod.prepare_endfile_writing()
    write_input_parameters_to_toml(
        test.parameters, f"temp.toml"
    )  # write the final parameters to a toml file
