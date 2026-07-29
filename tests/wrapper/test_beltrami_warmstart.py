# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Verify cross-evaluation warm starts of the matrix-free Beltrami solves.

Each iterative solve seeds its GMRES initial guess from the previous converged solution
held in the persistent Ate/Aze store (fortran_src/beltrami_solver_mod.F90). The seed is
an initial guess only -- convergence is always judged on the true residual of the current
system -- so it can never change the answer, only the iteration count.

The scenario tested is the realistic one: evaluate at x0, then at a slightly perturbed
geometry x1 (an optimizer-like small step). The warm evaluation at x1 is compared against
a cold evaluation at the same x1 from a fresh instance (whose store is zero). Identical
repeat evaluations are deliberately NOT used as the iteration probe: there the seed's
residual sits exactly at the convergence tolerance by construction and GMRES grinds at
the round-off floor, which measures floating-point noise, not the warm start.

Iteration counts are observed via the ``niters_gmres_total`` diagnostic counter.
"""

import os.path

import numpy as np
import pytest
import spectre4py_wrap as w4

from spectre import force_real, get_xinit_specwrap, remove_file

from helpers import spectre_with_lconstraint

curr_path = os.path.dirname(__file__) + "/"

# fixed mu/fluxes: no constraint iteration, so the solve sequence is deterministic
LCONSTRAINT = -1

FLAGS = dict(
    lfindzero=0,
    lhmatrix=False,
    lhevalues=False,
    lbeltramisolver=2,  # every eligible volume iterative
    nitergmres=5000,
    ngmresrestart=60,
    epsgmres=1e-12,
)


def _setup(fname):
    test = spectre_with_lconstraint(fname, LCONSTRAINT)
    for key, val in FLAGS.items():
        setattr(test.input_list_mod, key, val)
    return test


@pytest.mark.parametrize("fname", [f"{curr_path}qa_7vol_test.toml"])
def test_warmstart_fewer_iterations_same_force(fname):
    # warm path: eval at x0 fills the store, then eval at the stepped geometry x1
    test = _setup(fname)
    xin = np.asarray(get_xinit_specwrap(test))
    x1 = xin * (1.0 + 1e-8)  # small smooth geometry step, optimizer-like
    force_real(xin, test)
    c0 = int(w4.allglobal_mod.niters_gmres_total)
    f_warm = np.asarray(force_real(x1, test)).copy()
    k_warm = int(w4.allglobal_mod.niters_gmres_total) - c0

    # cold reference: fresh instance (zero store), same stepped geometry
    test2 = _setup(fname)
    c0 = int(w4.allglobal_mod.niters_gmres_total)
    f_cold = np.asarray(force_real(x1, test2))
    k_cold = int(w4.allglobal_mod.niters_gmres_total) - c0

    assert k_cold > 0, "no GMRES iterations recorded on the cold evaluation"
    assert k_warm < k_cold, (
        f"{os.path.basename(fname)}: warm-started evaluation took {k_warm} GMRES "
        f"iterations, not fewer than the cold reference's {k_cold}"
    )

    rel = np.max(np.abs(f_warm - f_cold)) / max(np.max(np.abs(f_cold)), 1e-300)
    assert rel < 1e-9, (
        f"{os.path.basename(fname)}: warm and cold evaluations of the same geometry "
        f"differ by rel {rel:.3e} -- the warm start must not affect the answer"
    )

    remove_file(fname + ".end")
