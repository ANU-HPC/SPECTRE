#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Save 3-vol finite-pressure W7-X and NCSX TOMLs before and after update_bns.

Outputs (in the current working directory):
  <geom>_3vol_pressure_init.toml   — state right after construction (TOML iBns)
  <geom>_3vol_pressure_after_update_bns.toml  — after update_bns (iBns := ofmn)

Also prints, for each case:
  * geometry and constraint flags (lconstraint=3, lbnszero=False expected)
  * full ofmn vs iBns table after update_bns to verify they match modulo gbnbld
  * Picard 6-iter delta sequence

This addresses:
  1. "Save TOML before and after update_bns" — both files written.
  2. "iBns should be pretty dense" — print the full non-zero count + sample.
  3. "Compare iBns after update vs ofmn from VC" — they should be byte-equal
     with gbnbld=0 (full replacement); table verifies.
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import fix2free, params_to_simsopt_boundary

REPO = Path(__file__).resolve().parents[2]
COMM = MPI.COMM_WORLD


def build(geom):
    from simsopt.configs.zoo import get_data
    from simsopt.mhd import Vmec

    if geom == "w7x":
        _, _, _, _, bs = get_data("w7x")
        vmec = Vmec(str(
            REPO / "examples/utils/make_freeboundary/input.W7-X_standard_configuration"
        ))
        fp = simsvmec2spectre(vmec, nvol=3, mpol=11, ntor=11)
    elif geom == "ncsx":
        _, _, _, _, bs = get_data("ncsx")
        vmec = Vmec(str(
            REPO / "examples/utils/make_freeboundary/wout_li383_low_res_reference.nc"
        ))
        fp = simsvmec2spectre(vmec, nvol=3, mpol=4, ntor=3, interface_guess="wout")
    else:
        raise ValueError(geom)

    nvol = fp.physics.nvol
    tflux = np.asarray(fp.physics.tflux[:nvol])
    pressure = np.array(fp.physics.pressure, dtype=float).copy()
    pressure[:nvol] = 0.1 + (0.001 - 0.1) * tflux
    fp = fp.model_copy(update={
        "physics": fp.physics.model_copy(update={"pressure": pressure})
    })

    plasma_boundary = params_to_simsopt_boundary(fp, which="plasma")
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.15)
    return fix2free(fp, bs, comp_boundary)


def run_case(geom):
    if COMM.rank == 0:
        print("=" * 79)
        print(f"  {geom.upper()} 3-vol finite-pressure test case")
        print("=" * 79)
        params = build(geom)
        phys = params.physics
        print(f"  nvol={phys.nvol}, mpol={phys.mpol}, ntor={phys.ntor}, "
              f"nfp={phys.nfp}")
        print(f"  lfreebound={phys.lfreebound}, lconstraint={phys.lconstraint}, "
              f"lbnszero={phys.lbnszero}")
        print(f"  phiedge = {phys.phiedge:+.6f}, curpol = {phys.curpol:+8.3f}")
    else:
        params = None
    params = COMM.bcast(params, root=0)

    sp = SPECTRE.from_input_params(
        params, name=f"{geom}_3vol_pressure", verbose=False, comm=COMM
    )

    # Save the initial state TOML (after preset, before any dforce).
    init_toml = Path(f"{geom}_3vol_pressure_init.toml").resolve()
    sp.to_toml(init_toml)
    if COMM.rank == 0:
        print(f"  Wrote initial TOML → {init_toml}")

    # Dump initial iBns from the TOML.
    ibns_init = np.asarray(sp.allglobal_mod.ibns).copy()
    if COMM.rank == 0:
        nonzero_init = int(np.sum(np.abs(ibns_init) > 1e-14))
        print(f"  Initial iBns: {nonzero_init} non-zero of {len(ibns_init)} "
              f"total modes; max|iBns_init| = {float(np.max(np.abs(ibns_init))):.3e}")

    # Force gbnbld=0 and call one update_bnorm to replace iBns := ofmn.
    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")
    sp.input_list_mod.gbnbld = 0.0
    sp.forces_mod.dforce(ngdof, x, force, False, True)   # settle Beltrami
    sp.wrapper_funcs_mod.update_bnorm()                  # iBns := ofmn

    # Capture: after update_bnorm with gbnbld=0, iBns should equal ofmn(2:mn)
    # in the entries (2..mn). iBns(0) and iBns(1) are not updated (stell-sym).
    # NOTE: f90wrap exposes ofmn at its allocated buffer size, which may be
    # larger than mn; slice both to mn for comparison.
    mn = int(sp.allglobal_mod.mn)
    ibns_after = np.asarray(sp.allglobal_mod.ibns)[:mn].copy()
    ofmn = np.asarray(sp.allglobal_mod.ofmn)[:mn].copy()
    im = np.asarray(sp.allglobal_mod.im)[:mn].copy()
    in_ = np.asarray(sp.allglobal_mod.in_)[:mn].copy()
    ibns_init_mn = ibns_init[:mn].copy()
    nfp = int(sp.input_list_mod.nfp)
    n = in_ // nfp

    # Save the post-update TOML.
    after_toml = Path(f"{geom}_3vol_pressure_after_update_bns.toml").resolve()
    sp.to_toml(after_toml)
    if COMM.rank == 0:
        print(f"  Wrote post-update_bns TOML → {after_toml}")

    if COMM.rank == 0:
        nonzero_after = int(np.sum(np.abs(ibns_after) > 1e-14))
        print(f"  iBns after update_bns: {nonzero_after} non-zero of "
              f"{len(ibns_after)} total modes; "
              f"max|iBns_after| = {float(np.max(np.abs(ibns_after))):.3e}")
        nonzero_ofmn = int(np.sum(np.abs(ofmn) > 1e-14))
        print(f"  ofmn (this call): {nonzero_ofmn} non-zero of {len(ofmn)} "
              f"total modes; max|ofmn| = {float(np.max(np.abs(ofmn))):.3e}")

        # Verify iBns(2:mn) == ofmn(2:mn) byte-equal (gbnbld=0 → full replace).
        diff_2to_end = float(np.max(np.abs(ibns_after[1:] - ofmn[1:])))
        print(f"  max|iBns_after[1:] - ofmn[1:]| = {diff_2to_end:.3e}  "
              f"(should be 0 with gbnbld=0)")

        # Print the iBns vs ofmn table, sorted by (m, |n|).
        rows = []
        for ii in range(len(im)):
            m, nn = int(im[ii]), int(n[ii])
            rows.append((m, nn, float(ibns_init_mn[ii]),
                         float(ibns_after[ii]), float(ofmn[ii])))
        rows.sort(key=lambda r: (r[0], abs(r[1]), -r[1]))
        print(f"\n  {'(m,n)':<10}  {'iBns_init':>14}  "
              f"{'iBns_after':>14}  {'ofmn':>14}  {'delta_from_init':>14}")
        print("  " + "-" * 76)
        for m, nn, b_in, b_af, of in rows:
            print(f"  ({m:2d},{nn:+3d})  {b_in:>+14.4e}  {b_af:>+14.4e}  "
                  f"{of:>+14.4e}  {b_af - b_in:>+14.4e}")

    # Picard iteration trace.
    if COMM.rank == 0:
        print(f"\n  Picard iteration trace (gbnbld=0, fixed xin):")
    sp.input_list_mod.gbnbld = 0.0
    sp.forces_mod.dforce(ngdof, x, force, False, True)
    for it in range(1, 7):
        b_before = np.asarray(sp.allglobal_mod.ibns).copy()
        sp.wrapper_funcs_mod.update_bnorm()
        b_after = np.asarray(sp.allglobal_mod.ibns)
        delta = float(np.max(np.abs(b_after - b_before)))
        sp.forces_mod.dforce(ngdof, x, force, False, True)
        if COMM.rank == 0:
            print(f"    iter {it}: max|delta iBns| = {delta:.3e}")
    if COMM.rank == 0:
        print()

    del sp


def main():
    run_case("w7x")
    run_case("ncsx")


if __name__ == "__main__":
    main()
