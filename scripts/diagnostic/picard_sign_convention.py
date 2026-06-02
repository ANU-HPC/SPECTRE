#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Test whether iBns := -ofmn is the self-consistent Picard map.

User hypothesis: at convergence, iBns + ofmn = 0 (not iBns - ofmn = 0).
The flat 2-cycle observed at gBnbld=0 (|delta| ≈ 2·|iBns|) is consistent
with the iteration oscillating between +ofmn and -ofmn, which is what
you'd see if the "right" update were iBns := -ofmn but we're applying
iBns := +ofmn instead.

If correct:
  * Setting iBns := -ofmn once should give a self-consistent state.
  * A second `update_bnorm` from that state should produce
    ofmn' ≈ -iBns  (so iBns + ofmn' ≈ 0).

Tests on the same NCSX 3-vol fixture as picard_full_sweep.py.
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import fix2free, params_to_simsopt_boundary

REPO = Path(__file__).resolve().parents[2]
COMM = MPI.COMM_WORLD


def build_ncsx():
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


def main():
    if COMM.rank == 0:
        print("=" * 79)
        print("  SIGN CONVENTION TEST: is the self-consistent update iBns := -ofmn?")
        print("=" * 79)
        params = build_ncsx()
    else:
        params = None
    params = COMM.bcast(params, root=0)
    sp = SPECTRE.from_input_params(params, name="ncsx_sign", verbose=False, comm=COMM)

    mn = int(sp.allglobal_mod.mn)
    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

    sp.input_list_mod.gbnbld = 0.0   # full replace
    sp.input_list_mod.vcasingits = 8

    # Establish baseline: run update_bnorm once and capture ofmn1, iBns1.
    sp.forces_mod.dforce(ngdof, x, force, False, True)
    sp.wrapper_funcs_mod.update_bnorm()
    ofmn1 = np.asarray(sp.allglobal_mod.ofmn)[:mn].copy()
    ibns1 = np.asarray(sp.allglobal_mod.ibns)[:mn].copy()

    if COMM.rank == 0:
        diff_pos = float(np.max(np.abs(ibns1 - ofmn1)))
        diff_neg = float(np.max(np.abs(ibns1 + ofmn1)))
        print(f"\n  After 1st update_bnorm (gbnbld=0):")
        print(f"    max|iBns - ofmn| = {diff_pos:.3e}  (gbnbld=0 implies this is 0)")
        print(f"    max|iBns + ofmn| = {diff_neg:.3e}")

    # --- Test 1: re-run dforce + update_bnorm. Is the new ofmn ≈ -iBns?
    sp.forces_mod.dforce(ngdof, x, force, False, True)
    sp.wrapper_funcs_mod.update_bnorm()
    ofmn2 = np.asarray(sp.allglobal_mod.ofmn)[:mn].copy()
    ibns2 = np.asarray(sp.allglobal_mod.ibns)[:mn].copy()

    if COMM.rank == 0:
        diff_2_to_1 = float(np.max(np.abs(ibns2 - ibns1)))
        diff_2_to_minus_1 = float(np.max(np.abs(ibns2 + ibns1)))
        diff_ofmn = float(np.max(np.abs(ofmn2 - ofmn1)))
        diff_ofmn_neg = float(np.max(np.abs(ofmn2 + ofmn1)))
        print(f"\n  After 2nd update_bnorm (gbnbld=0):")
        print(f"    max|iBns2 - iBns1|  = {diff_2_to_1:.3e}")
        print(f"    max|iBns2 + iBns1|  = {diff_2_to_minus_1:.3e}")
        print(f"    max|ofmn2 - ofmn1|  = {diff_ofmn:.3e}")
        print(f"    max|ofmn2 + ofmn1|  = {diff_ofmn_neg:.3e}")
        print()
        print("  Interpretation:")
        if diff_2_to_minus_1 < 0.1 * diff_2_to_1:
            print("    → iBns flipped sign exactly between iters; supports 'should be -ofmn' hypothesis")
        elif diff_2_to_1 < 0.1 * diff_2_to_minus_1:
            print("    → iBns barely moved between iters; Picard IS converging at this geometry")
        else:
            print("    → mixed: not a clean 2-cycle, not converged")

    # --- Test 2: manually write iBns := -ofmn1 and see if the next update is a fixed point.
    if COMM.rank == 0:
        print()
        print(f"{'=' * 79}")
        print("  Test 2: manually set iBns := -ofmn1, then call update_bnorm.")
        print("           If hypothesis is right, ofmn3 should equal -iBns3 (≈ ofmn1).")
        print(f"{'=' * 79}")

    # Reset state by re-creating SPECTRE.
    del sp
    sp = SPECTRE.from_input_params(params, name="ncsx_sign_t2", verbose=False, comm=COMM)
    sp.input_list_mod.gbnbld = 0.0
    sp.input_list_mod.vcasingits = 8
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")
    sp.forces_mod.dforce(ngdof, x, force, False, True)

    # Manually overwrite iBns := -ofmn1
    ibns_arr = sp.allglobal_mod.ibns
    ibns_arr[:mn] = -ofmn1

    # Resolve Beltrami at the new (manually-set) iBns.
    sp.forces_mod.dforce(ngdof, x, force, False, True)

    # Capture state.
    ibns3 = np.asarray(sp.allglobal_mod.ibns)[:mn].copy()
    # Run update_bnorm to compute fresh ofmn3 (this also overwrites iBns
    # since gbnbld=0; capture ofmn3 BEFORE looking at iBns).
    sp.wrapper_funcs_mod.update_bnorm()
    ofmn3 = np.asarray(sp.allglobal_mod.ofmn)[:mn].copy()
    ibns3_after = np.asarray(sp.allglobal_mod.ibns)[:mn].copy()

    if COMM.rank == 0:
        # We set iBns := -ofmn1. If "should be -ofmn" hypothesis is right,
        # the new ofmn3 (computed from plasma at iBns = -ofmn1) should
        # roughly equal +ofmn1 = -iBns3. Check both.
        d_pos = float(np.max(np.abs(ofmn3 - ofmn1)))   # ofmn3 ≈ ofmn1?
        d_neg = float(np.max(np.abs(ofmn3 + ofmn1)))   # ofmn3 ≈ -ofmn1?
        d_neg_ibns = float(np.max(np.abs(ofmn3 + ibns3)))  # ofmn3 ≈ -iBns3?
        d_pos_ibns = float(np.max(np.abs(ofmn3 - ibns3)))  # ofmn3 ≈ +iBns3?
        print(f"\n  After setting iBns := -ofmn1 and running update_bnorm:")
        print(f"    max|ofmn3 - ofmn1|       = {d_pos:.3e}")
        print(f"    max|ofmn3 + ofmn1|       = {d_neg:.3e}")
        print(f"    max|ofmn3 - iBns3|       = {d_pos_ibns:.3e}  (ofmn3 ≈ +iBns3?)")
        print(f"    max|ofmn3 + iBns3|       = {d_neg_ibns:.3e}  (ofmn3 ≈ -iBns3?)")
        print()
        print("  Interpretation:")
        if d_neg < 0.1 * d_pos:
            print("    → ofmn flipped sign — hypothesis WRONG (negation just oscillates more)")
        elif d_pos < 0.1 * d_neg:
            print("    → ofmn stayed put — iBns := -ofmn1 IS the fixed point!")
            print("      That means the convention should be iBns := -ofmn, not +ofmn.")
        else:
            print("    → ofmn moved but not by sign flip — something else is going on.")


if __name__ == "__main__":
    main()
