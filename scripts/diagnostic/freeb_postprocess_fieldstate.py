#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Isolate why the FINAL FIELD ERROR printout blows up (~1e10 in the vacuum)
while the optimizer's force balance is healthy (~1e-2).

Hypothesis: `Force_Minimizer.postprocess()` calls `SPECTRE.run()` (full
`simulation_mod.spectre()`), which reallocates Fortran arrays, but — unlike
`SPECTRE.parameters.setter` — does NOT clear the f90wrap pointer caches
afterward (see `core.py:_clear_fortran_array_caches`, l.296). The subsequent
`force_real` + `print_all_errors` then read through stale/dangling pointers,
producing garbage field errors that don't reflect the actual equilibrium.

This script loads the OPTIMIZED geometry and measures the per-volume Beltrami
FIELD ERROR at four stages to localize the cause:

  A) solve at the optimized geometry, NO run()           -> baseline truth
  B) right after run() (no re-solve)                      -> what run() left
  C) re-solve after run() (mirrors postprocess order)     -> the reported value
  D) clear f90wrap caches, then re-solve after run()      -> test the fix

Decision table:
  * A sane, C huge            -> run() corrupts the post-run reads (postproc bug)
  * C huge, D sane            -> CONFIRMS stale f90wrap pointer cache; fix is to
                                 clear caches after run()
  * A already huge            -> the optimized geometry's vacuum field is
                                 genuinely poorly resolved (not a state bug)

Run (one MPI job at a time):
  OMP_NUM_THREADS=3 mpirun -n 3 ./freeb_postprocess_fieldstate.py <opt.toml>
"""

from argparse import ArgumentParser

parser = ArgumentParser(description="Localize the postprocess FIELD ERROR blow-up.")
parser.add_argument(
    "inputfile",
    nargs="?",
    default="ncsx_TEST_cap30_opt.toml",
    help="Optimized .toml (default: ncsx_TEST_cap30_opt.toml in CWD).",
)
args = parser.parse_args()

import numpy as np

from spectre import SPECTRE
from spectre.error_checking import print_all_errors
from spectre.force_targets import force_real


def field_err_summary(s, label):
    """Collective; prints per-volume FIELD-ERROR max on rank 0 and returns it."""
    avg, mx, _frms, _fmax = print_all_errors(s, beltrami=True, force=True)
    if s.comm.rank == 0:
        mvol = s.allglobal_mod.mvol
        worst = [float(np.max(mx[v])) for v in range(mvol)]
        print(f"\n>>> [{label}] per-volume FIELD ERROR max: "
              + "  ".join(f"vol{v}={w:.3e}" for v, w in enumerate(worst)))
        print(f">>> [{label}] vacuum (vol {mvol - 1}) max = {worst[-1]:.3e}\n", flush=True)
    return mx


s = SPECTRE(args.inputfile)
x = s.xin.copy()

if s.comm.rank == 0:
    print("\n================ STAGE A: optimized geometry, NO run() ================")
force_real(x, s)
field_err_summary(s, "A: solve, pre-run")

if s.comm.rank == 0:
    print("\n================ STAGE B: immediately after run() (no re-solve) =======")
s.input_list_mod.linitialize = 0
s.input_list_mod.nppts = 0
s.run(save_output=False)
field_err_summary(s, "B: just after run()")

if s.comm.rank == 0:
    print("\n================ STAGE C: re-solve after run() (postprocess order) ====")
force_real(x, s)
field_err_summary(s, "C: solve, post-run, NO cache clear")

if s.comm.rank == 0:
    print("\n================ STAGE D: clear f90wrap caches + re-solve after run() =")
s._clear_fortran_array_caches()
force_real(x, s)
field_err_summary(s, "D: solve, post-run, WITH cache clear")

if s.comm.rank == 0:
    print("\nDONE. Compare A vs C (does run() corrupt?) and C vs D (does a cache "
          "clear fix it?).")
