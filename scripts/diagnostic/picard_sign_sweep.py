#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Systematic sign-flip sweep: which of the physical parameters governs
whether the iBns Picard map is contractive at a given geometry?

Tests both W7-X and NCSX 3-vol finite-pressure test cases, then for each
applies sign flips to curpol, phiedge, vns, and combinations, and
measures whether `max|delta iBns|` shrinks (contractive → convergent),
oscillates flat, or grows (non-contractive → divergent) across a few
Picard iterations at fixed xin.

If W7-X converges but NCSX doesn't (or vice versa), and a sign flip
on one of the parameters fixes the failing case, that pinpoints the
convention mismatch.
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import fix2free, params_to_simsopt_boundary
from spectre.file_io.input_parameters import mmpol, mntor

REPO = Path(__file__).resolve().parents[2]
COMM = MPI.COMM_WORLD


def build_test_case(geom):
    """Generate a 3-vol finite-pressure free-boundary case for w7x or ncsx."""
    from simsopt.configs.zoo import get_data
    from simsopt.mhd import Vmec

    if geom == "w7x":
        _, _, _, _, bs = get_data("w7x")
        vmec_path = REPO / "examples/utils/make_freeboundary/input.W7-X_standard_configuration"
        vmec = Vmec(str(vmec_path))
        fixed_params = simsvmec2spectre(vmec, nvol=3, mpol=11, ntor=11)
    elif geom == "ncsx":
        _, _, _, _, bs = get_data("ncsx")
        vmec_path = REPO / "examples/utils/make_freeboundary/wout_li383_low_res_reference.nc"
        vmec = Vmec(str(vmec_path))
        fixed_params = simsvmec2spectre(
            vmec, nvol=3, mpol=4, ntor=3, interface_guess="wout"
        )
    else:
        raise ValueError(geom)

    # Linear pressure profile in tflux: 0.1 → 0.001 at the plasma boundary
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
    return fix2free(fixed_params, bs, comp_boundary)


def make_variant(base, **physics_updates):
    """Return a new InputParameters with physics fields updated."""
    new_phys = base.physics.model_copy(update=physics_updates)
    return base.model_copy(update={"physics": new_phys})


def picard_deltas(sp, max_iter=6):
    """Run iter rounds of (dforce + update_bnorm) and return max|delta iBns| per iter."""
    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

    sp.input_list_mod.gbnbld = 0.0   # full replacement to isolate Picard map
    sp.input_list_mod.vcasingits = 8

    # Settle Beltrami.
    sp.forces_mod.dforce(ngdof, x, force, False, True)

    deltas = []
    for _ in range(max_iter):
        ibns_before = np.asarray(sp.allglobal_mod.ibns).copy()
        sp.wrapper_funcs_mod.update_bnorm()
        ibns_after = np.asarray(sp.allglobal_mod.ibns)
        delta = float(np.max(np.abs(ibns_after - ibns_before)))
        deltas.append(delta)
        sp.forces_mod.dforce(ngdof, x, force, False, True)
    return deltas


def classify(deltas):
    """contractive: deltas[-1] < 0.1*deltas[0]
       divergent:   deltas[-1] > 1.5*deltas[0]
       flat:        otherwise
    """
    if len(deltas) < 2 or deltas[0] == 0:
        return "trivial"
    ratio = deltas[-1] / deltas[0]
    if ratio < 0.1:
        return "CONTRACTIVE"
    elif ratio > 1.5:
        return "DIVERGENT"
    else:
        return "flat"


def run_variant(label, params):
    """Build SPECTRE, run Picard test, print summary line."""
    if COMM.rank == 0:
        print(f"  {label:<40s} ", end="", flush=True)
    sp = SPECTRE.from_input_params(
        params, name=label.replace(" ", "_"), verbose=False, comm=COMM
    )
    try:
        deltas = picard_deltas(sp, max_iter=6)
    except Exception as e:
        if COMM.rank == 0:
            print(f"  ERROR: {e}")
        return
    if COMM.rank == 0:
        cls = classify(deltas)
        phys = params.physics
        deltas_str = "  ".join(f"{d:.2e}" for d in deltas)
        print(
            f"phi={phys.phiedge:+.3f}  cur={phys.curpol:+8.2f}  "
            f"[{deltas_str}]  → {cls}"
        )


def sign_flip_sweep(label, base):
    """Run a battery of sign-flip variants on base and report Picard behavior."""
    if COMM.rank == 0:
        print(f"\n{'=' * 95}")
        print(f"  {label}")
        print(f"{'=' * 95}")
        print(f"  {'variant':<40} {'phi':<8}  {'cur':<10}  "
              f"max|delta iBns| per Picard iter (gbnbld=0)  → classification")
        print(f"  {'-' * 91}")

    variants = [
        ("original", base),
        ("phiedge_flip", make_variant(base, phiedge=-base.physics.phiedge)),
        ("curpol_flip", make_variant(base, curpol=-base.physics.curpol)),
        ("phiedge+curpol_flip",
         make_variant(base, phiedge=-base.physics.phiedge,
                      curpol=-base.physics.curpol)),
        ("vns_sign_flip", make_variant(base, vns=-base.physics.vns)),
        ("vns_flip+phiedge_flip",
         make_variant(base, vns=-base.physics.vns,
                      phiedge=-base.physics.phiedge)),
        ("vns_flip+curpol_flip",
         make_variant(base, vns=-base.physics.vns,
                      curpol=-base.physics.curpol)),
        ("vns+phiedge+curpol_flip",
         make_variant(base, vns=-base.physics.vns,
                      phiedge=-base.physics.phiedge,
                      curpol=-base.physics.curpol)),
    ]

    for vlabel, vparams in variants:
        run_variant(vlabel, vparams)


def main():
    if COMM.rank == 0:
        print("Building W7-X 3-vol finite-pressure params ...")
    w7x_params = build_test_case("w7x") if COMM.rank == 0 else None
    w7x_params = COMM.bcast(w7x_params, root=0)

    if COMM.rank == 0:
        print("Building NCSX 3-vol finite-pressure params ...")
    ncsx_params = build_test_case("ncsx") if COMM.rank == 0 else None
    ncsx_params = COMM.bcast(ncsx_params, root=0)

    sign_flip_sweep("W7-X 3-vol  (Bphi sign determined by simsopt's W7-X coil set)",
                    w7x_params)
    sign_flip_sweep("NCSX 3-vol  (Bphi sign determined by simsopt's NCSX coil set)",
                    ncsx_params)

    if COMM.rank == 0:
        print()
        print("=" * 95)
        print("  Interpretation key:")
        print("  CONTRACTIVE  = max|delta| at iter 6 is < 10% of iter 1   (Picard converges)")
        print("  DIVERGENT    = max|delta| at iter 6 is > 150% of iter 1  (Picard diverges)")
        print("  flat         = neither (probably a 2-cycle or borderline)")
        print()
        print("  Reading: if W7-X is CONTRACTIVE and NCSX is DIVERGENT in 'original',")
        print("  but flipping curpol (or phiedge, or vns) on NCSX makes it CONTRACTIVE,")
        print("  that flip is the convention mismatch we're hunting.")
        print("=" * 95)


if __name__ == "__main__":
    main()
