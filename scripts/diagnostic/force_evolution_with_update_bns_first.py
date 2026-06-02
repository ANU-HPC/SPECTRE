#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Follow-up to force_evolution_picard.py: does update_bns FIRST prevent
the stage 1a interface intersection?

Plan:
  0. initial (force ~ 1e+12)
  0.5. damped Picard until iBns converges (gBnbld=0.5, until |delta| <
       gbntol, or 12 iters max)
  1a. stage 1a fmodes (same as before)
  1b. stage 1b fmodes

At each snapshot, log force per interface + Picard contractivity.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import fix2free, params_to_simsopt_boundary
from spectre.force_targets import fmodes_obj
from spectre.least_squares.least_squares import least_squares as lstsq

REPO = Path(__file__).resolve().parents[2]
COMM = MPI.COMM_WORLD


def flush_print(*args):
    if COMM.rank == 0:
        print(*args, flush=True)


def build_ncsx_3vol():
    from simsopt.configs.zoo import get_data
    from simsopt.mhd import Vmec
    _, _, _, _, bs = get_data("ncsx")
    vmec = Vmec(str(REPO / "examples/utils/make_freeboundary/wout_li383_low_res_reference.nc"))
    fp = simsvmec2spectre(vmec, nvol=3, mpol=4, ntor=3, interface_guess="wout")
    nvol = fp.physics.nvol
    tflux = np.asarray(fp.physics.tflux[:nvol])
    pressure = np.array(fp.physics.pressure, dtype=float).copy()
    pressure[:nvol] = 0.1 + (0.001 - 0.1) * tflux
    fp = fp.model_copy(update={"physics": fp.physics.model_copy(update={"pressure": pressure})})
    plasma = params_to_simsopt_boundary(fp, which="plasma")
    comp = plasma.copy()
    comp.extend_via_normal(-0.15)
    return fix2free(fp, bs, comp)


def damped_picard_until_converged(sp, gbnbld, max_iter, tol):
    """Drive the iBns Picard with damping until |delta iBns| < tol or hit max_iter."""
    sp.input_list_mod.gbnbld = float(gbnbld)
    sp.input_list_mod.vcasingits = 8

    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force_buf = np.zeros(ngdof + 1, dtype=np.float64, order="F")
    sp.forces_mod.dforce(ngdof, x, force_buf, False, True)

    deltas = []
    for it in range(max_iter):
        before = np.asarray(sp.allglobal_mod.ibns).copy()
        sp.wrapper_funcs_mod.update_bnorm()
        after = np.asarray(sp.allglobal_mod.ibns)
        delta = float(np.max(np.abs(after - before)))
        deltas.append(delta)
        sp.forces_mod.dforce(ngdof, x, force_buf, False, True)
        if delta < tol:
            break
    return deltas


def evaluate_force(sp):
    sp.input_list_mod.gbnbld = 1.0
    sp.force_real()
    f_shpd = np.asarray(sp.f_real_shpd).copy()
    return f_shpd


def picard_classify(sp, gbnbld, max_iter=6):
    ibns_save = np.asarray(sp.allglobal_mod.ibns).copy()
    deltas = damped_picard_until_converged(sp, gbnbld, max_iter=max_iter, tol=0.0)
    sp.allglobal_mod.ibns[:] = ibns_save
    if len(deltas) >= 2 and deltas[0] > 0:
        r = deltas[-1] / deltas[0]
        cls = "CONTRACT" if r < 0.1 else ("DIVERGE" if r > 1.5 else "  flat ")
    else:
        cls = "TRIVIAL"
        r = float("nan")
    return cls, deltas, r


def snapshot(sp, label):
    flush_print(f"\n{'=' * 79}\n  SNAPSHOT: {label}\n{'=' * 79}")
    f = evaluate_force(sp)
    if COMM.rank == 0:
        for i in range(f.shape[0]):
            print(f"  iface {i}: rms = {np.sqrt(np.mean(f[i]**2)):.3e}  "
                  f"max = {np.max(np.abs(f[i])):.3e}")
    for gbnbld in (0.0, 0.5):
        cls, deltas, r = picard_classify(sp, gbnbld)
        flush_print(f"    gbnbld={gbnbld:.1f}  {cls}  "
                    f"[{' '.join(f'{d:.2e}' for d in deltas)}]  ratio={r:.3e}")


def main():
    flush_print("=" * 79)
    flush_print("FORCE EVOLUTION with update_bns BEFORE stage 1a")
    flush_print("=" * 79)

    if COMM.rank == 0:
        params = build_ncsx_3vol()
    else:
        params = None
    params = COMM.bcast(params, root=0)
    sp = SPECTRE.from_input_params(params, name="ncsx_evo_v2", verbose=False, comm=COMM)

    snapshot(sp, "00_initial")

    # Stage 0.5: damped Picard until converged.
    flush_print(f"\n{'=' * 79}\n  STAGE 0.5: damped Picard (gBnbld=0.5) until converged\n{'=' * 79}")
    gbntol = float(sp.input_list_mod.gbntol)
    deltas = damped_picard_until_converged(sp, gbnbld=0.5, max_iter=15, tol=gbntol)
    flush_print(f"  ran {len(deltas)} iter(s): "
                f"[{' '.join(f'{d:.2e}' for d in deltas)}]  "
                f"gbntol={gbntol:.1e}")

    snapshot(sp, "005_after_damped_picard")

    # Stage 1a
    flush_print(f"\n{'=' * 79}\n  STAGE 1a: fmodes, x_scale=1.0, 4 iters\n{'=' * 79}")
    min_params = sp.min_params
    min_params.objective = "fmodes"
    min_params.exp_fact = -0.05
    obj_func, jac_func = fmodes_obj(min_params)

    r1a = lstsq(
        obj_func, sp.xin, jac=jac_func,
        method="trf", verbose=0, args=[sp],
        max_niter=4, max_nfev=50,
        x_scale=1.0, tr_solver="lsmr",
        tr_options={"regularize": True, "damp": 1e-3},
        loss="soft_l1", f_scale=5.0,
    )
    flush_print(f"  stage1a done: cost={r1a.cost:.4e}, nfev={r1a.nfev}")
    sp.xin = r1a.x
    snapshot(sp, "01_after_stage1a")

    # Stage 1b
    flush_print(f"\n{'=' * 79}\n  STAGE 1b: fmodes, x_scale='jac', 10 iters\n{'=' * 79}")
    r1b = lstsq(
        obj_func, sp.xin, jac=jac_func,
        method="trf", verbose=0, args=[sp],
        max_niter=10, max_nfev=100,
        x_scale="jac", tr_solver="lsmr",
        tr_options={"regularize": True, "damp": 1e-3},
        loss="soft_l1", f_scale=5.0,
    )
    flush_print(f"  stage1b done: cost={r1b.cost:.4e}, nfev={r1b.nfev}")
    sp.xin = r1b.x
    snapshot(sp, "02_after_stage1b")


if __name__ == "__main__":
    main()
