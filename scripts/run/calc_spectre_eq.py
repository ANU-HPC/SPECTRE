#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## To run in the background and redirect output use `nohup ./calc_spectre_eq.py $FNAME > out.txt 2>&1 &`

from argparse import ArgumentParser

parser = ArgumentParser(
    prog="calc_spectre_eq", description="Find equilibrium using SPECTRE through force minimization"
)
parser.add_argument("inputfile", help="Path to the input .toml file")
args = parser.parse_args()

from spectre import Force_Minimizer

# The optimizer follows the deck: optimizer_picard (iBns frozen inside each inner force
# solve, refreshed only between solves) for conventional free-boundary runs, a single
# solve for fixed-boundary, direct virtual-casing (lvcvacuum=1) and lbnszero decks.
obj = Force_Minimizer(args.inputfile)

obj.minimize()

obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print("SPECTRE equilibrium calculation done!")
