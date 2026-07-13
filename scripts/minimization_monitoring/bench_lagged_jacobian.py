#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Benchmark the lagged-Jacobian trf mode (jac_lag) against the standard solver.

The dominant cost of the outer force minimisation is building the dense force
Jacobian. ``least_squares(..., jac_lag=k, jac_lag_rho=rho)`` reuses the
Jacobian across outer iterations while the trust-region ratio of accepted
steps stays >= rho times the ratio last achieved with a FRESH Jacobian (so
staleness, not the problem's intrinsic nonlinearity, triggers rebuilds); it
rebuilds after k consecutive reuses, on a rejected step, or before accepting
any convergence test; the trust radius is never adapted from a lagged-model
step. Pure Python change in spectre/least_squares.

This script runs the standard and the lagged solver on the given configs and
reports iterations, Jacobian builds (njev), residual evaluations (nfev), and
wall-clock, plus cost-vs-time trajectories.

Example
-------
    OMP_NUM_THREADS=6 python bench_lagged_jacobian.py \
        tests/wrapper/qa_7vol_test.toml tests/wrapper/test_freeb_w7x.toml \
        examples/analysis/finding_fixed_points/test_qa_nvol7.toml --no-show
"""

import os
import time
from argparse import ArgumentParser

_HERE = os.path.dirname(os.path.abspath(__file__))


def build_parser():
    p = ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("configs", nargs="+", help="SPECTRE .toml configuration files")
    p.add_argument("-n", "--max-niter", type=int, default=None,
                   help="Override outer-iteration cap (default: per-config)")
    p.add_argument("--jac-lag", type=int, default=6)
    p.add_argument("--jac-lag-rho", type=float, default=0.75)
    p.add_argument("--ftol", type=float, default=1e-10)
    p.add_argument("-o", "--outdir", type=str, default=_HERE)
    p.add_argument("--prefix", type=str, default="lagged_jacobian")
    p.add_argument("--no-show", action="store_true")
    return p


args = build_parser().parse_args()

import matplotlib
import numpy as np

from spectre import SPECTRE, force_real, force_real_jac, get_xinit_specwrap, least_squares

if args.no_show:
    matplotlib.use("Agg")
plt = matplotlib.pyplot

# expensive multi-volume cases get a lower default iteration cap
DEFAULT_NITER = {"test_qa_nvol7.toml": 8}


def run(config, jac_lag):
    test = SPECTRE(config, verbose=False)
    x0 = get_xinit_specwrap(test)
    force_real(x0, test)

    hist = []
    t0 = time.perf_counter()

    def cb(it, nfev, cost, cost_red, step_norm, opt, x, obj):
        hist.append((cost, time.perf_counter() - t0))

    def f(x, t):
        return force_real(x, t).flatten(order="C")

    def jac(x, t, *a):
        return force_real_jac(x, t).reshape(-1, t.allglobal_mod.ngdof + 1)

    niter = args.max_niter or DEFAULT_NITER.get(os.path.basename(config), 25)
    res = least_squares(f, x0, jac=jac, method="trf", verbose=2, args=[test],
                        max_niter=niter, ftol=args.ftol, xtol=1e-22, gtol=1e-13,
                        jac_lag=jac_lag, jac_lag_rho=args.jac_lag_rho, cb=cb)
    wall = time.perf_counter() - t0
    return np.array(hist), res, wall


def main():
    results = {}
    for config in args.configs:
        name = os.path.basename(config)
        print(f"===== {name} =====")
        for label, lag in [("standard", 0), (f"lagged (k={args.jac_lag}, rho={args.jac_lag_rho})", args.jac_lag)]:
            hist, res, wall = run(config, lag)
            results[(name, label)] = hist
            print(f"  {label:26s}: {len(hist) - 1:3d} iters  njev {res.njev:3d}  "
                  f"nfev {res.nfev:3d}  cost {hist[0][0]:.4e} -> {res.cost:.4e}  "
                  f"wall {wall:7.1f}s  (status {res.status})")

    ncfg = len(args.configs)
    fig, axes = plt.subplots(1, ncfg, figsize=(5.6 * ncfg, 4.6), squeeze=False)
    for k, config in enumerate(args.configs):
        name = os.path.basename(config)
        ax = axes[0, k]
        for (nm, label), hist in results.items():
            if nm != name:
                continue
            color = "C0" if label == "standard" else "C3"
            ax.semilogy(hist[:, 1], hist[:, 0], "o-", ms=4, color=color, label=label)
        ax.set(xlabel="wall-clock (s)", ylabel="cost  ½‖f‖²", title=name)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Lagged-Jacobian trf (jac_lag): identical steps, fewer Jacobian builds")
    fig.tight_layout()
    os.makedirs(args.outdir, exist_ok=True)
    png = os.path.join(args.outdir, f"{args.prefix}.png")
    fig.savefig(png, dpi=130)
    print(f"\nSaved {png}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
