#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Damping sweep only — fast. Reuses SPECTRE per geometry to avoid the cost
of constructing it per cell. Tests whether ANY damping makes the Picard
contract.

Output is line-by-line (flushed) so progress is visible during runs.
"""

import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import fix2free, params_to_simsopt_boundary

REPO = Path(__file__).resolve().parents[2]
COMM = MPI.COMM_WORLD


def flush_print(*args, **kwargs):
    if COMM.rank == 0:
        print(*args, **kwargs, flush=True)


def build(geom):
    from simsopt.configs.zoo import get_data
    from simsopt.mhd import Vmec
    if geom == "w7x":
        _, _, _, _, bs = get_data("w7x")
        vmec = Vmec(str(REPO / "examples/utils/make_freeboundary/input.W7-X_standard_configuration"))
        fp = simsvmec2spectre(vmec, nvol=3, mpol=8, ntor=8)
    elif geom == "ncsx":
        _, _, _, _, bs = get_data("ncsx")
        vmec = Vmec(str(REPO / "examples/utils/make_freeboundary/wout_li383_low_res_reference.nc"))
        fp = simsvmec2spectre(vmec, nvol=3, mpol=4, ntor=3, interface_guess="wout")
    else:
        raise ValueError(geom)

    nvol = fp.physics.nvol
    tflux = np.asarray(fp.physics.tflux[:nvol])
    pressure = np.array(fp.physics.pressure, dtype=float).copy()
    pressure[:nvol] = 0.1 + (0.001 - 0.1) * tflux
    fp = fp.model_copy(update={"physics": fp.physics.model_copy(update={"pressure": pressure})})

    plasma = params_to_simsopt_boundary(fp, which="plasma")
    comp = plasma.copy()
    comp.extend_via_normal(-0.15)
    return fix2free(fp, bs, comp)


def picard_with_damping(sp, gbnbld, max_iter, ngdof, x, force):
    """Reset SPECTRE state to baseline (iBns from TOML = 0), then iterate."""
    # Reset iBns to TOML (zeros) by resetting parameters from cached input
    # — easiest is to reset just iBns directly.
    mn = int(sp.allglobal_mod.mn)
    sp.allglobal_mod.ibns[:mn] = 0.0
    sp.input_list_mod.gbnbld = float(gbnbld)
    sp.input_list_mod.vcasingits = 8

    sp.forces_mod.dforce(ngdof, x, force, False, True)
    deltas = []
    for _ in range(max_iter):
        before = np.asarray(sp.allglobal_mod.ibns).copy()
        sp.wrapper_funcs_mod.update_bnorm()
        after = np.asarray(sp.allglobal_mod.ibns)
        deltas.append(float(np.max(np.abs(after - before))))
        sp.forces_mod.dforce(ngdof, x, force, False, True)
    return deltas


def classify(deltas):
    if len(deltas) < 2 or deltas[0] == 0:
        return "TRIVIAL"
    r = deltas[-1] / deltas[0]
    if r < 0.1:
        return "CONTRACT"
    elif r > 1.5:
        return "DIVERGE"
    else:
        return "  flat "


def fmt(deltas):
    return " ".join(f"{d:8.2e}" for d in deltas)


def run_geom(geom):
    flush_print(f"\n{'#' * 79}")
    flush_print(f"# {geom.upper()}")
    flush_print(f"{'#' * 79}")

    if COMM.rank == 0:
        params = build(geom)
        p = params.physics
        flush_print(f"  nvol={p.nvol}, mpol={p.mpol}, ntor={p.ntor}, nfp={p.nfp}, "
                    f"lconstraint={p.lconstraint}, lbnszero={p.lbnszero}")
        flush_print(f"  phiedge={p.phiedge:+.4f}, curpol={p.curpol:+8.2f}")
    else:
        params = None
    params = COMM.bcast(params, root=0)

    sp = SPECTRE.from_input_params(params, name=f"{geom}_damp", verbose=False, comm=COMM)
    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

    flush_print(f"  {'gbnbld':<8} {'class':<10} per-iter max|delta iBns|")
    flush_print(f"  {'-' * 78}")
    for gbnbld in (0.0, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0):
        try:
            deltas = picard_with_damping(sp, gbnbld, max_iter=6,
                                          ngdof=ngdof, x=x, force=force)
            cls = classify(deltas)
            flush_print(f"  {gbnbld:<8.2f} {cls:<10} [{fmt(deltas)}]")
        except Exception as e:
            flush_print(f"  {gbnbld:<8.2f} ERROR: {e}")


def main():
    flush_print("=" * 79)
    flush_print("Picard DAMPING-only sweep — both geometries, no SPECTRE re-construction")
    flush_print("=" * 79)
    for geom in ("ncsx", "w7x"):
        run_geom(geom)
    flush_print("\nDone.")


if __name__ == "__main__":
    main()
