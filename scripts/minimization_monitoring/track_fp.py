#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
r"""Post-processing driver: track the magnetic axis and one island chain across checkpoints.

Replays ``x_<a>_<b>.npy`` checkpoint files written by ``Force_Minimizer`` (or any SPECTRE
optimisation that saves ``xs_fixb/``), re-solves the field at each checkpoint, and calls
``FixedPointTracker.step`` per checkpoint. Results are saved to a ``.npz`` file.

Usage example::

    python track_fp.py xs_fixb/ case.toml 5 4 \
        --poloidal-occurrences 4 --discover --output fp_metrics.npz

Send ``SIGUSR1`` (or ``scancel --signal=USR1 <SLURM_JOB_ID>``) to stop cleanly after the
current checkpoint and save the partial result.
"""

import argparse
import os
import re
import signal
from pathlib import Path
from time import perf_counter

import numpy as np

_XS_FIXB_RE = re.compile(r"^x_(\d+)_(\d+)\.npy$")


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("xs_dir", help="Folder containing x_<a>_<b>.npy checkpoint files")
    p.add_argument("toml", help="SPECTRE TOML matching the checkpoints")
    p.add_argument("repeat_period", type=int, help="Island chain period (denominator of iota)")
    p.add_argument("volume", type=int, help="Volume index hosting the chain (>= 1)")
    p.add_argument(
        "--poloidal-occurrences",
        type=int,
        default=None,
        help="Toroidal mode number (numerator of iota); pins chain identity",
    )
    p.add_argument(
        "--guess",
        nargs=2,
        type=float,
        metavar=("R", "Z"),
        default=None,
        help="(R, Z) seed at phi=0 for the first checkpoint",
    )
    p.add_argument(
        "--discover",
        action="store_true",
        help="Self-seed by FP-first scan (required when no --guess)",
    )
    p.add_argument(
        "--start-at",
        type=int,
        default=0,
        metavar="ITER",
        help="Skip checkpoints with iteration < ITER",
    )
    p.add_argument("--stride", type=int, default=1, help="Process every STRIDE-th checkpoint")
    p.add_argument(
        "--limit", type=int, default=0, help="Process at most this many checkpoints (0 = all)"
    )
    p.add_argument(
        "--reverse",
        action="store_true",
        help="Process high->low iteration (track backward from converged state)",
    )
    p.add_argument(
        "--output", default=None, help="Output .npz path (default: fp_metrics.npz next to the TOML)"
    )
    p.add_argument("--verbose", action="store_true", help="Print per-checkpoint diagnostic block")
    return p.parse_args()


def main():  # noqa: D103
    args = _parse_args()

    # SPECTRE imports are deferred (heavy core; keeps --help instant).
    from spectre import SPECTRE
    from spectre.fixed_points import FixedPointTracker
    from spectre.force_targets import force_real
    from spectre.utils import get_xinit_specwrap, unpack_interfaces

    xs_dir = Path(args.xs_dir)
    toml_path = Path(args.toml)
    if not xs_dir.is_dir():
        raise FileNotFoundError(f"xs_fixb directory not found: {xs_dir}")
    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    out = Path(args.output) if args.output else toml_path.parent / "fp_metrics.npz"
    if out.parent == Path("."):
        out = toml_path.parent / out.name

    stride = max(1, int(args.stride))

    # List checkpoints sorted by iteration number.
    all_paths = sorted(
        ((int(m.group(2)), q) for q in xs_dir.iterdir() if (m := _XS_FIXB_RE.match(q.name))),
        key=lambda t: t[0],
    )
    if not all_paths:
        raise FileNotFoundError(f"No x_***_***.npy files found in {xs_dir}")
    paths = [(it, q) for it, q in all_paths if it >= args.start_at][::stride]
    if args.reverse:
        paths = paths[::-1]
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        raise ValueError(
            f"No checkpoints with iter >= {args.start_at} "
            f"(have {all_paths[0][0]}..{all_paths[-1][0]})"
        )
    n = len(paths)

    guess = args.guess  # None or [R, Z]
    discover = args.discover or (guess is None)  # default to discover when no guess

    tk = FixedPointTracker(
        repeat_period=args.repeat_period,
        volume=args.volume,
        n=n,
        guess=guess,
        discover=discover,
        poloidal_occurrences=args.poloidal_occurrences,
    )

    # Header.
    seed_str = (
        f"guess{tuple(guess)} + re-discover on loss"
        if guess is not None and discover
        else ("discover" if discover else f"guess{tuple(guess)}")
    )
    po_str = (
        f"{args.poloidal_occurrences} ({args.poloidal_occurrences}/{args.repeat_period})"
        if args.poloidal_occurrences
        else "any k*Nfp"
    )
    print("track_fp - island + axis tracker")
    print(f"  toml     : {toml_path}")
    print(f"  period   : {args.repeat_period} · po {po_str} · volume {args.volume}")
    print(f"  seed     : {seed_str}")
    print(
        f"  run      : {n} checkpoints (iters {paths[0][0]}..{paths[-1][0]}, "
        f"{'reverse' if args.reverse else 'forward'}, stride {stride})"
    )
    print(f"  output   : {out.name}")
    if sid := os.environ.get("SLURM_JOB_ID"):
        print(f"  stop     : scancel --signal=USR1 {sid}")
    if discover and args.poloidal_occurrences is None:
        print(
            "  ** WARNING: discover without poloidal_occurrences - the first cold lock pins the "
            "chain; pass --poloidal-occurrences to avoid a silent wrong-chain lock."
        )
    print(flush=True)

    iter_indices = np.array([it for it, _ in paths], dtype=int)

    # SIGUSR1 -> stop after current checkpoint.
    stop = [False]
    old_handler = signal.signal(
        signal.SIGUSR1,
        lambda *_: (
            stop.__setitem__(0, True),
            print("\n  [SIGUSR1] stop after this checkpoint.", flush=True),
        ),
    )

    obj = SPECTRE(str(toml_path), verbose=False)

    try:
        for i, (it, npy) in enumerate(paths):
            if stop[0]:
                print(f"  Stopping before iter {it} (SIGUSR1).", flush=True)
                break
            t0 = perf_counter()

            try:
                unpack_interfaces(np.load(str(npy)), obj)
                force_real(get_xinit_specwrap(obj), obj)
                specout = obj.create_spectreout()
            except Exception as exc:
                print(f"[iter {it:3d}]  ! checkpoint load FAILED - {exc}; skipping.", flush=True)
                continue
            tk.t_recon[i] = perf_counter() - t0

            ctx = tk.step(specout, i)

            tk.t_total[i] = perf_counter() - t0
            done = i + 1
            elapsed = float(np.nansum(tk.t_total[:done]))
            if args.verbose:
                tk.log_row(i, ctx, it=it, done=done, n=n, elapsed=elapsed)

            np.savez(str(out), **tk.result_dict(iter_indices, args.start_at, stride, args.reverse))
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    found = ~np.isnan(tk.island_rz[:, 0])
    n_cold = int(np.sum(tk.provenance == "cold"))
    n_follow = int(np.sum(tk.provenance == "follow"))
    res_rng = (
        f"{tk.residual[found].min():.1e}..{np.nanmax(tk.residual[found]):.1e}"
        if found.any()
        else "n/a"
    )
    print(
        f"\nDone. {int(found.sum())}/{n} loc · cold {n_cold}/follow {n_follow} "
        f"· no_fp {int(tk.no_fp.sum())} · res {res_rng}"
    )
    print(flush=True)
    return tk.result_dict(iter_indices, args.start_at, stride, args.reverse)


if __name__ == "__main__":
    main()
