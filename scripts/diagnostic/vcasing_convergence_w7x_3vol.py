#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Diagnose virtual-casing convergence on a finite-pressure 3-volume W7-X case.

Hypothesis under test: at the user's failing W7-X geometry, the
``compute_bnorm_plasmab`` virtual-casing integral is non-convergent
(``relvcerr`` ~ 1 at finest grid, *increasing* with refinement). The
resulting ``ofmn`` is dominated by noise concentrated at high ``(m,n)``.
Setting ``iBns := ofmn`` writes that noise into the vol-Mvol Dirichlet
BC, which the vacuum Beltrami amplifies.

This script generates a clean, reproducible 3-volume W7-X test case from
the canonical VMEC input + simsopt coil set (no committed TOML to drift
out of sync), then sweeps ``vcasingits`` and inspects the per-mode
stability of ``ofmn``.

Usage:
    python scripts/diagnostic/vcasing_convergence_w7x_3vol.py

Requires: simsopt, mpi4py, spectre installed in dev mode. The W7-X VMEC
input file must be present at the path referenced below.
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import (
    fix2free,
    params_to_simsopt_boundary,
)

REPO = Path(__file__).resolve().parents[2]
VMEC_INPUT = REPO / "examples/utils/make_freeboundary/input.W7-X_standard_configuration"
COMM = MPI.COMM_WORLD


def build_test_case():
    """Generate a finite-pressure 3-volume W7-X free-boundary InputParameters.

    Pattern mirrors examples/utils/make_freeboundary/test_vns_signs.py:132-148
    with two differences:
      * nvol = 3 (was 1)
      * pressure profile linear in tflux: ~0.1 inside → ~0.001 outside
    """
    from simsopt.configs.zoo import get_data
    from simsopt.mhd import Vmec

    _, _, _, _, bs = get_data("w7x")
    vmec = Vmec(str(VMEC_INPUT))
    fixed_params = simsvmec2spectre(vmec, nvol=3, mpol=11, ntor=11)

    # Linear-in-tflux pressure profile. tflux is monotonically increasing
    # to 1.0 at the plasma boundary; pressure[i] is the pressure in volume
    # i (no entry for the vacuum volume). The pressure array in
    # PhysicsParameters is sized to the max-Nvol bound, so write into a
    # copy of the existing array to preserve length and just overwrite the
    # first nvol entries.
    nvol = fixed_params.physics.nvol
    tflux = np.asarray(fixed_params.physics.tflux[:nvol])
    pressure = np.array(fixed_params.physics.pressure, dtype=float).copy()
    pressure[:nvol] = 0.1 + (0.001 - 0.1) * tflux
    fixed_params = fixed_params.model_copy(
        update={
            "physics": fixed_params.physics.model_copy(
                update={"pressure": pressure}
            ),
        }
    )

    plasma_boundary = params_to_simsopt_boundary(fixed_params, which="plasma")
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.15)

    params = fix2free(fixed_params, bs, comp_boundary)
    return params


def sweep_vcasingits(sp, its_list):
    """Run update_bnorm at each vcasingits in its_list, capture ofmn each time.

    Returns:
        dict[int, np.ndarray]: vcasingits → ofmn copy (shape (mn,)).
    """
    # Force gBnbld=0 so update_bnorm does a full replacement (iBns := ofmn) —
    # we never want the saved iBns to influence subsequent calls' ofmn.
    sp.input_list_mod.gbnbld = 0.0

    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

    # Settle Beltrami at the TOML's iBns once so compute_bnorm_plasmab has
    # surface currents to integrate.
    sp.forces_mod.dforce(ngdof, x, force, False, True)

    results = {}
    for its in its_list:
        sp.input_list_mod.vcasingits = its
        sp.wrapper_funcs_mod.update_bnorm()
        results[its] = np.asarray(sp.allglobal_mod.ofmn).copy()
    return results


def report(results, im, in_, nfp, label):
    """Print per-mode tables sorted by mode magnitude."""
    if COMM.rank != 0:
        return

    its_list = sorted(results.keys())
    n = in_ // nfp

    # Build the per-mode summary
    rows = []
    for ii in range(len(im)):
        m, nn = int(im[ii]), int(n[ii])
        vals = np.array([results[its][ii] for its in its_list])
        absvals = np.abs(vals)
        # Stability metric: (max - min) / mean of |ofmn| across the sweep.
        # >> 1 means the mode swings wildly with vcasingits; ~ 0 means stable.
        mean_abs = float(np.mean(absvals))
        range_abs = float(np.max(absvals) - np.min(absvals))
        stab = range_abs / mean_abs if mean_abs > 0 else 0.0
        rows.append((m, nn, vals, mean_abs, stab))

    # Sort by max(m, |n|), then by m, then by |n|
    rows.sort(key=lambda r: (max(r[0], abs(r[1])), r[0], abs(r[1])))

    header_its = "  ".join(f"its={its:>3}" for its in its_list)
    print(f"\n{'=' * 79}")
    print(f"  {label}")
    print(f"{'=' * 79}")
    print(f"  Mode      {header_its}   stab(range/mean)")
    print(f"  {'-' * 75}")
    for m, nn, vals, mean_abs, stab in rows:
        if mean_abs < 1e-14:
            continue
        vals_str = "  ".join(f"{v:+.2e}" for v in vals)
        flag = "  ← UNSTABLE" if stab > 0.5 else ""
        print(f"  m={m:2d} n={nn:+3d}  {vals_str}   {stab:6.3f}{flag}")
    print(f"{'=' * 79}\n")


def main():
    if COMM.rank == 0:
        print("=" * 79)
        print("VCASING CONVERGENCE DIAGNOSTIC — finite-pressure 3-vol W7-X")
        print("=" * 79)
        print(f"  VMEC input: {VMEC_INPUT}")
        print(f"  Building test case (nvol=3, pressure linear in tflux: 0.1 → 0.001) ...")
        params = build_test_case()
        phys = params.physics
        print(f"  Built. nvol={phys.nvol}, mpol={phys.mpol}, ntor={phys.ntor}, "
              f"nfp={phys.nfp}, lfreebound={phys.lfreebound}")
        print(f"  tflux:    {[round(t, 4) for t in phys.tflux]}")
        print(f"  pressure: {[round(p, 4) for p in phys.pressure]}")
        print(f"  phiedge = {phys.phiedge:.6f}, curpol = {phys.curpol:.6f}")
    else:
        params = None

    params = COMM.bcast(params, root=0)

    sp = SPECTRE.from_input_params(
        params, name="w7x_3vol_pressure", verbose=False, comm=COMM
    )

    nfp = int(sp.input_list_mod.nfp)
    im = np.asarray(sp.allglobal_mod.im).copy()
    in_ = np.asarray(sp.allglobal_mod.in_).copy()

    its_list = [4, 8, 16, 32, 64, 128]

    if COMM.rank == 0:
        print(f"\n  Sweeping vcasingits over {its_list} ...")
        print(f"  vcNt={int(sp.input_list_mod.vcnt)}, "
              f"vcNz={int(sp.input_list_mod.vcnz)}, "
              f"Lvcgrid={int(sp.input_list_mod.lvcgrid)}, "
              f"vcasingtol={float(sp.input_list_mod.vcasingtol):.1e}, "
              f"vcasingeps={float(sp.input_list_mod.vcasingeps):.1e}")

    results = sweep_vcasingits(sp, its_list)

    report(
        results,
        im,
        in_,
        nfp,
        label="ofmn per-mode at different vcasingits  "
        "(stab > 0.5 ⇒ noise-dominated)",
    )

    if COMM.rank == 0:
        # Coarse aggregate: how many modes are unstable?
        n_unstable = 0
        n_total = 0
        for ii in range(len(im)):
            vals = np.array([results[its][ii] for its in its_list])
            absvals = np.abs(vals)
            mean_abs = float(np.mean(absvals))
            if mean_abs < 1e-14:
                continue
            n_total += 1
            range_abs = float(np.max(absvals) - np.min(absvals))
            stab = range_abs / mean_abs
            if stab > 0.5:
                n_unstable += 1
        print(f"  vcasing-sweep summary: {n_unstable} / {n_total} non-zero "
              f"modes are noise-dominated (stab > 0.5).\n")

    # ----------------------------------------------------------------------
    # Picard convergence test: at fixed xin, does iBns converge across
    # successive update_bnorm calls?
    # ----------------------------------------------------------------------
    if COMM.rank == 0:
        print("=" * 79)
        print("  Picard convergence check at fixed xin")
        print("  (each row: max|iBns_after - iBns_before| for one update_bnorm)")
        print("=" * 79)

    # Restore the default vcasingits the user originally had.
    sp.input_list_mod.vcasingits = 8
    sp.input_list_mod.gbnbld = 0.0   # full replacement

    # Settle Beltrami first.
    sp.forces_mod.dforce(ngdof, x, force, False, True)

    for it in range(1, 9):
        ibns_before = np.asarray(sp.allglobal_mod.ibns).copy()
        sp.wrapper_funcs_mod.update_bnorm()
        ibns_after = np.asarray(sp.allglobal_mod.ibns)
        delta = float(np.max(np.abs(ibns_after - ibns_before)))
        # Also resolve Beltrami at the new iBns so the next ofmn reflects
        # the new equilibrium (not the old plasma + new iBns mismatch).
        sp.forces_mod.dforce(ngdof, x, force, False, True)
        bnserr = float(sp.allglobal_mod.bnserr)
        if COMM.rank == 0:
            print(f"  Picard iter {it}: max|delta iBns| = {delta:.3e}   "
                  f"bnserr_after_blend = {bnserr:.3e}")

    if COMM.rank == 0:
        print()
        print("  Interpretation:")
        print("    If max|delta iBns| → 0 fast, Picard is contractive at")
        print("    this geometry → past `bnserr ~ 1e-17` outcome is plausible.")
        print("    If it grows, the geometry is in the non-contractive regime")
        print("    that single-shot update_bns is designed for.")
        print()


if __name__ == "__main__":
    main()
