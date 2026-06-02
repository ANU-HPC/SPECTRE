#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""How does the iBns Picard's contractivity evolve as the geometry relaxes?

User hypothesis: the initial-state force on the plasma boundary is
'bonkers' because preset-time inputs (TOML iBns=0 from fix2free,
dpflux/dtflux at TOML guesses, interior interface guesses interpolated
rather than equilibrium) leave a state far from any physical
equilibrium. As the optimizer relaxes (stage 1a → 1b), the force
shrinks. The question this script answers: does the relaxation also
move us into the basin where the iBns Picard map is contractive?

For each of three snapshot states — initial, after stage 1a (4 iters
fmodes x_scale=1.0), and after stage 1b (10 iters fmodes x_scale='jac')
— we:
  1. Save a heatmap of force(θ, ζ) on every interior interface as a PNG.
  2. Run 6 Picard iterations at gBnbld=0 and gBnbld=0.5; report
     `max|delta iBns|` per iter and the contractivity class.

Geometry: NCSX 3-vol, mpol=4, ntor=3, linear pressure 0.1 → 0.001.
"""

import os
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


def evaluate_force(sp):
    """Settle Beltrami at current xin, compute force_real per interface.
    Returns (f_shpd shape (mvol-1, nz, nt), summary dict).
    """
    sp.input_list_mod.gbnbld = 1.0   # freeze iBns during force eval
    sp.force_real()                  # populates sp.f_real_shpd
    f_shpd = np.asarray(sp.f_real_shpd).copy()
    # Stats
    summary = {
        "rms_per_iface": [float(np.sqrt(np.mean(f_shpd[i] ** 2)))
                          for i in range(f_shpd.shape[0])],
        "max_per_iface": [float(np.max(np.abs(f_shpd[i])))
                          for i in range(f_shpd.shape[0])],
        "total_sumsq": float(np.sum(f_shpd ** 2)),
    }
    return f_shpd, summary


def plot_force_heatmap(f_shpd, label, outdir, sp):
    if COMM.rank != 0:
        return

    n_iface = f_shpd.shape[0]
    fig, axes = plt.subplots(1, n_iface, figsize=(4.5 * n_iface, 4.5))
    if n_iface == 1:
        axes = [axes]

    nfp = int(sp.input_list_mod.nfp)
    nz = f_shpd.shape[1]
    nt = f_shpd.shape[2]
    # Theta full circle, zeta one field period.
    theta = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zeta = np.linspace(0, 2 * np.pi / nfp, nz, endpoint=False)

    for i, ax in enumerate(axes):
        f = f_shpd[i]
        vmax = max(abs(f.min()), abs(f.max()))
        im = ax.imshow(
            f,
            origin="lower",
            extent=[theta[0], theta[-1], zeta[0], zeta[-1]],
            aspect="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(
            f"iface {i}\n"
            f"rms={np.sqrt(np.mean(f**2)):.3e}\n"
            f"max={np.max(np.abs(f)):.3e}"
        )
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel(r"$\zeta$  (one field period)")
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Force on plasma interfaces — {label}", fontsize=14)
    fig.tight_layout()
    path = outdir / f"force_{label}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  Saved {path}")


def picard_test(sp, label, max_iter=6):
    """For gbnbld ∈ {0, 0.5}, run Picard iterations and report deltas."""
    if COMM.rank == 0:
        print(f"\n  Picard test @ {label}:")

    results = {}
    for gbnbld in (0.0, 0.5):
        # Reset iBns to current preset state by re-running update_bnorm
        # once after this loop iteration's preceding settle.
        ngdof = sp.allglobal_mod.ngdof
        x = sp.xin
        force_buf = np.zeros(ngdof + 1, dtype=np.float64, order="F")

        # Snapshot iBns before our test perturbs it; we'll restore later.
        ibns_save = np.asarray(sp.allglobal_mod.ibns).copy()

        sp.input_list_mod.gbnbld = float(gbnbld)
        sp.input_list_mod.vcasingits = 8
        sp.forces_mod.dforce(ngdof, x, force_buf, False, True)

        deltas = []
        for _ in range(max_iter):
            before = np.asarray(sp.allglobal_mod.ibns).copy()
            sp.wrapper_funcs_mod.update_bnorm()
            after = np.asarray(sp.allglobal_mod.ibns)
            deltas.append(float(np.max(np.abs(after - before))))
            sp.forces_mod.dforce(ngdof, x, force_buf, False, True)

        results[gbnbld] = deltas

        # Restore iBns so the next Picard test (different gbnbld) starts
        # from the same baseline.
        sp.allglobal_mod.ibns[:] = ibns_save

        if COMM.rank == 0:
            ratio = deltas[-1] / deltas[0] if deltas[0] > 0 else float("nan")
            cls = "CONTRACT" if ratio < 0.1 else ("DIVERGE" if ratio > 1.5 else "  flat ")
            deltas_str = " ".join(f"{d:.2e}" for d in deltas)
            print(f"    gbnbld={gbnbld:.1f}  {cls}  [{deltas_str}]   ratio={ratio:.3f}")

    return results


def snapshot(sp, label, outdir):
    flush_print(f"\n{'=' * 79}\n  SNAPSHOT: {label}\n{'=' * 79}")
    f_shpd, summary = evaluate_force(sp)
    if COMM.rank == 0:
        for i, (rms, mx) in enumerate(zip(summary["rms_per_iface"],
                                           summary["max_per_iface"])):
            print(f"  iface {i}: rms = {rms:.3e}  max = {mx:.3e}")
        print(f"  total sum(f²) = {summary['total_sumsq']:.3e}")
    plot_force_heatmap(f_shpd, label, outdir, sp)
    picard_test(sp, label)


def main():
    outdir = Path("/tmp/force_evolution").resolve()
    if COMM.rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)

    flush_print("=" * 79)
    flush_print("FORCE EVOLUTION + PICARD CONTRACTIVITY across optimization stages")
    flush_print("=" * 79)

    if COMM.rank == 0:
        params = build_ncsx_3vol()
    else:
        params = None
    params = COMM.bcast(params, root=0)

    sp = SPECTRE.from_input_params(params, name="ncsx_force_evo", verbose=False, comm=COMM)
    flush_print(f"  nvol={sp.input_list_mod.nvol}, mpol={sp.input_list_mod.mpol}, "
                f"ntor={sp.input_list_mod.ntor}, nfp={sp.input_list_mod.nfp}")

    # -------------------- STAGE 00: initial --------------------
    snapshot(sp, "00_initial", outdir)

    # -------------------- STAGE 1a: 4 fmodes iters @ x_scale=1.0 --------------------
    flush_print(f"\n{'=' * 79}\n  Running STAGE 1a (4 iters, fmodes, x_scale=1.0) ...\n{'=' * 79}")
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
    flush_print(f"  stage1a done: cost={r1a.cost:.4e}, nfev={r1a.nfev}, "
                f"optim={r1a.optimality:.3e}")

    # Push the optimizer's final x into Fortran state.
    sp.xin = r1a.x

    snapshot(sp, "01_after_stage1a", outdir)

    # -------------------- STAGE 1b: 10 fmodes iters @ x_scale='jac' --------------------
    flush_print(f"\n{'=' * 79}\n  Running STAGE 1b (10 iters, fmodes, x_scale='jac') ...\n{'=' * 79}")
    r1b = lstsq(
        obj_func, sp.xin, jac=jac_func,
        method="trf", verbose=0, args=[sp],
        max_niter=10, max_nfev=100,
        x_scale="jac", tr_solver="lsmr",
        tr_options={"regularize": True, "damp": 1e-3},
        loss="soft_l1", f_scale=5.0,
    )
    flush_print(f"  stage1b done: cost={r1b.cost:.4e}, nfev={r1b.nfev}, "
                f"optim={r1b.optimality:.3e}")
    sp.xin = r1b.x

    snapshot(sp, "02_after_stage1b", outdir)

    flush_print(f"\n{'=' * 79}\n  Plots in {outdir}\n{'=' * 79}")


if __name__ == "__main__":
    main()
