#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 Proxima Fusion
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

obj = Force_Minimizer(args.inputfile)

if obj.freeb_mode and obj.rank == 0:
    # This driver runs a single force solve with the default optimizer, which
    # never refreshes the free-boundary normal field iBns — so the result is
    # force-balanced against the initial-geometry iBns, not a self-consistent
    # free-boundary equilibrium. Use calc_spectre_eq_freeb.py for that.
    print(
        "WARNING: calc_spectre_eq.py does not self-consistently update the "
        "free-boundary normal field iBns (it is frozen at its initial value). "
        "For a converged free-boundary equilibrium use calc_spectre_eq_freeb.py "
        "(outer/inner loop with damped iBns refresh between cycles)."
    )

obj.minimize()

obj.postprocess(save_json=False, save_pkl=False)

if obj.rank == 0:
    print("SPECTRE equilibrium calculation done!")
