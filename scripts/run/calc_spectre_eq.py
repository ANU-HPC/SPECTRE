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
from spectre.force_minimizer import optimizer_picard  # noqa: E402

# optimizer_picard keeps iBns frozen inside each inner force solve and refreshes it only
# between solves. It falls back to a single solve for fixed-boundary runs and for the
# direct virtual-casing solve (lvcvacuum=1), so it is the right default either way.
obj = Force_Minimizer(args.inputfile, optimizer=optimizer_picard)

obj.minimize()

obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print("SPECTRE equilibrium calculation done!")
