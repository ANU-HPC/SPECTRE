#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Diagnose virtual-casing convergence on a finite-pressure 3-volume NCSX case.

Hypothesis under test: at the user's failing geometry, the
``compute_bnorm_plasmab`` virtual-casing integral is non-convergent
(``relvcerr`` ~ 1 at the finest grid, *increasing* with refinement).
The resulting ``ofmn`` would then be dominated by noise concentrated at
high ``(m,n)``, which when written into iBns destabilizes the
vol-Mvol Beltrami.

The user has confirmed the failing case has ``lconstraint=3`` and
``lbnszero=False`` — both reproduced here.

This script generates a clean, reproducible 3-volume NCSX test case from
the canonical VMEC wout + simsopt coil set, then sweeps ``vcasingits``
and inspects the per-mode stability of ``ofmn``, and finally runs a
Picard iteration on iBns at fixed xin to test contractivity.

Usage:
    python scripts/diagnostic/vcasing_convergence_ncsx_3vol.py

Requires: simsopt, mpi4py, spectre installed in dev mode. The NCSX VMEC
wout file must be present at the path referenced below.
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
NCSX_WOUT = REPO / "examples/utils/make_freeboundary/wout_li383_low_res_reference.nc"
COMM = MPI.COMM_WORLD


def build_test_case():
    """Generate a finite-pressure 3-volume NCSX free-boundary InputParameters.

    Pattern mirrors examples/utils/make_freeboundary/make_freeboundary_ncsx.ipynb
    with two differences:
      * nvol = 3 (was 4)
      * pressure profile linear in tflux: ~0.1 inside → ~0.001 outside
    """
    from simsopt.configs.zoo import get_data
    from simsopt.mhd import Vmec

    _, _, _, _, bs = get_data("ncsx")
    vmec = Vmec(str(NCSX_WOUT))
    fixed_params = simsvmec2spectre(
        vmec, nvol=3, mpol=4, ntor=3, interface_guess="wout"
    )

    # Linear-in-tflux pressure profile. tflux is monotonically increasing
    # to 1.0 at the plasma boundary; pressure[i] is the pressure in volume
    # i. The pressure array in PhysicsParameters is padded to the max
    # bound, so write into a copy of the existing array and just overwrite
    # the first nvol entries.
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
    sp.input_list_mod.gbnbld = 0.0   # full replacement; isolates ofmn

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


def report_modes(results, im, in_, nfp, label):
    """Print per-mode tables sorted by mode magnitude."""
    if COMM.rank != 0:
        return

    its_list = sorted(results.keys())
    n = in_ // nfp

    rows = []
    for ii in range(len(im)):
        m, nn = int(im[ii]), int(n[ii])
        vals = np.array([results[its][ii] for its in its_list])
        absvals = np.abs(vals)
        mean_abs = float(np.mean(absvals))
        range_abs = float(np.max(absvals) - np.min(absvals))
        stab = range_abs / mean_abs if mean_abs > 0 else 0.0
        rows.append((m, nn, vals, mean_abs, stab))

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


def picard_check(sp, ngdof, x, force, max_iter=8):
    """Run iterated update_bnorm + dforce at fixed xin; print iBns drift per iter."""
    if COMM.rank == 0:
        print("=" * 79)
        print("  Picard convergence check at fixed xin")
        print(f"  vcasingits = {int(sp.input_list_mod.vcasingits)}, "
              f"gbnbld = {float(sp.input_list_mod.gbnbld)}")
        print("=" * 79)

    # Settle Beltrami first.
    sp.forces_mod.dforce(ngdof, x, force, False, True)

    for it in range(1, max_iter + 1):
        ibns_before = np.asarray(sp.allglobal_mod.ibns).copy()
        sp.wrapper_funcs_mod.update_bnorm()
        ibns_after = np.asarray(sp.allglobal_mod.ibns)
        delta = float(np.max(np.abs(ibns_after - ibns_before)))
        # Resolve Beltrami at the new iBns so the next ofmn reflects the
        # new equilibrium, not the old plasma + new iBns mismatch.
        sp.forces_mod.dforce(ngdof, x, force, False, True)
        bnserr = float(sp.allglobal_mod.bnserr)
        if COMM.rank == 0:
            print(f"  Picard iter {it}: max|delta iBns| = {delta:.3e}   "
                  f"bnserr_after_blend = {bnserr:.3e}")

    if COMM.rank == 0:
        print()
        print("  Interpretation:")
        print("    delta → 0 fast  ⇒ Picard contractive at this geometry")
        print("                       (matches the past `bnserr ~ 1e-17` observation)")
        print("    delta grows     ⇒ Picard non-contractive — vcasing noise")
        print("                       amplifies iter-over-iter")
        print()


def main():
    if COMM.rank == 0:
        print("=" * 79)
        print("VCASING CONVERGENCE DIAGNOSTIC — finite-pressure 3-vol NCSX")
        print("=" * 79)
        print(f"  VMEC wout: {NCSX_WOUT}")
        print(f"  Building test case (nvol=3, pressure linear in tflux: 0.1 → 0.001) ...")
        params = build_test_case()
        phys = params.physics
        print(f"  Built. nvol={phys.nvol}, mpol={phys.mpol}, ntor={phys.ntor}, "
              f"nfp={phys.nfp}")
        print(f"  lfreebound={phys.lfreebound}, lconstraint={phys.lconstraint}, "
              f"lbnszero={phys.lbnszero}")
        print(f"  pressure[:nvol] = "
              f"{[round(float(p), 6) for p in phys.pressure[:phys.nvol]]}")
        print(f"  phiedge = {phys.phiedge:.6f}, curpol = {phys.curpol:.6f}")
        if phys.lconstraint != 3:
            print(f"  WARNING: lconstraint={phys.lconstraint}, expected 3")
        if phys.lbnszero:
            print(f"  WARNING: lbnszero=True, expected False")
    else:
        params = None

    params = COMM.bcast(params, root=0)

    sp = SPECTRE.from_input_params(
        params, name="ncsx_3vol_pressure", verbose=False, comm=COMM
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

    report_modes(
        results,
        im,
        in_,
        nfp,
        label="ofmn per-mode at different vcasingits  "
        "(stab > 0.5 ⇒ noise-dominated)",
    )

    if COMM.rank == 0:
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

    # Picard convergence check at the default vcasingits.
    sp.input_list_mod.vcasingits = 8
    sp.input_list_mod.gbnbld = 0.0
    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")
    picard_check(sp, ngdof, x, force, max_iter=8)


if __name__ == "__main__":
    main()
