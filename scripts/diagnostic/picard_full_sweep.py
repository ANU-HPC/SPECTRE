#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Full sweep: damping + sign flips on W7-X (mpol=8) and NCSX (mpol=4) 3-vol cases.

For each (geometry, variant, gbnbld) cell, runs 6 Picard iterations at
fixed xin and reports the per-iter ``max|delta iBns|`` sequence. Also
saves the initial and post-update_bns TOML for each baseline so the user
can inspect Bns content offline.

Outputs are written to the directory passed as --outdir (default: /tmp).
"""

import argparse
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
    fp = fp.model_copy(update={
        "physics": fp.physics.model_copy(update={"pressure": pressure})
    })

    plasma = params_to_simsopt_boundary(fp, which="plasma")
    comp = plasma.copy()
    comp.extend_via_normal(-0.15)
    return fix2free(fp, bs, comp)


def make_variant(base, **physics_updates):
    new_phys = base.physics.model_copy(update=physics_updates)
    return base.model_copy(update={"physics": new_phys})


def picard_run(sp, gbnbld, max_iter=6):
    """Run Picard with given gBnbld, return list of per-iter max|delta iBns|."""
    sp.input_list_mod.gbnbld = float(gbnbld)
    sp.input_list_mod.vcasingits = 8

    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")

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
        return "trivial"
    r = deltas[-1] / deltas[0]
    if r < 0.1:
        return "CONTRACT"
    elif r > 1.5:
        return "DIVERGE"
    else:
        return "  flat "


def fmt(deltas):
    return " ".join(f"{d:7.2e}" for d in deltas)


def run_cell(geom, variant_label, params, gbnbld):
    sp = SPECTRE.from_input_params(
        params, name=f"{geom}_{variant_label}_g{gbnbld:.1f}",
        verbose=False, comm=COMM
    )
    try:
        deltas = picard_run(sp, gbnbld, max_iter=6)
    except Exception as e:
        if COMM.rank == 0:
            print(f"  ERROR: {e}")
        return
    del sp
    if COMM.rank == 0:
        cls = classify(deltas)
        print(f"  gbnbld={gbnbld:.1f}  {cls}  [{fmt(deltas)}]")


def run_geometry(geom, outdir):
    if COMM.rank == 0:
        print(f"\n{'#' * 79}")
        print(f"# {geom.upper()}")
        print(f"{'#' * 79}")
        base = build(geom)
        p = base.physics
        print(f"  nvol={p.nvol}, mpol={p.mpol}, ntor={p.ntor}, nfp={p.nfp}, "
              f"lconstraint={p.lconstraint}, lbnszero={p.lbnszero}")
        print(f"  phiedge = {p.phiedge:+.4f}, curpol = {p.curpol:+8.2f}")

        # Save baseline initial TOML
        init_path = outdir / f"{geom}_3vol_pressure_init.toml"
        # We need a SPECTRE to write the TOML — build once for this.
    else:
        base = None
    base = COMM.bcast(base, root=0)

    # Save initial + post-update TOML for the baseline.
    sp = SPECTRE.from_input_params(
        base, name=f"{geom}_baseline", verbose=False, comm=COMM
    )
    init_path = outdir / f"{geom}_3vol_pressure_init.toml"
    sp.to_toml(init_path)

    sp.input_list_mod.gbnbld = 0.0
    sp.input_list_mod.vcasingits = 8
    ngdof = sp.allglobal_mod.ngdof
    x = sp.xin
    force = np.zeros(ngdof + 1, dtype=np.float64, order="F")
    sp.forces_mod.dforce(ngdof, x, force, False, True)
    sp.wrapper_funcs_mod.update_bnorm()
    after_path = outdir / f"{geom}_3vol_pressure_after_update_bns.toml"
    sp.to_toml(after_path)

    if COMM.rank == 0:
        print(f"  Wrote: {init_path.name}, {after_path.name}")
    del sp

    variants = {
        "original": base,
        "phiedge_flip": make_variant(base, phiedge=-base.physics.phiedge),
        "curpol_flip": make_variant(base, curpol=-base.physics.curpol),
        "phi+cur_flip": make_variant(base,
            phiedge=-base.physics.phiedge, curpol=-base.physics.curpol),
        "vns_sign_flip": make_variant(base, vns=-base.physics.vns),
        "vns_flip+phi_flip": make_variant(base,
            vns=-base.physics.vns, phiedge=-base.physics.phiedge),
        "vns_flip+cur_flip": make_variant(base,
            vns=-base.physics.vns, curpol=-base.physics.curpol),
        "vns+phi+cur_flip": make_variant(base,
            vns=-base.physics.vns,
            phiedge=-base.physics.phiedge, curpol=-base.physics.curpol),
    }

    if COMM.rank == 0:
        print()
        print(f"  {'variant':<22} damped Picard "
              f"(gbnbld=0 full replace … gbnbld=1 no update)")
        print(f"  {'-' * 91}")

    for vlabel, vparams in variants.items():
        if COMM.rank == 0:
            print(f"  {vlabel:<22}")
        for gbnbld in (0.0, 0.5, 0.8, 0.9, 1.0):
            run_cell(geom, vlabel, vparams, gbnbld)
        if COMM.rank == 0:
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="/tmp", help="Where to write the TOMLs")
    args = parser.parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if COMM.rank == 0:
        print("=" * 79)
        print(f"PICARD FULL SWEEP — damping × sign flips × geometry")
        print(f"  outdir = {outdir}")
        print("=" * 79)

    for geom in ("w7x", "ncsx"):
        run_geometry(geom, outdir)

    if COMM.rank == 0:
        print("=" * 79)
        print("Legend:")
        print("  CONTRACT = max|delta| at iter 6 is < 10% of iter 1 (converges)")
        print("  DIVERGE  = max|delta| at iter 6 is > 150% of iter 1 (diverges)")
        print("  flat     = neither (2-cycle or borderline)")
        print()
        print("Read: for each (geometry, variant), look across gbnbld values:")
        print("  → does any damping value make Picard CONTRACT?")
        print("  → does any sign-flip variant make 'original' damping work?")
        print("=" * 79)


if __name__ == "__main__":
    main()
