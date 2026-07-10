#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
r"""Post-processing driver: track the magnetic axis and one island chain across checkpoints.

Replays ``x_<a>_<b>.npy`` checkpoint files written by ``Force_Minimizer`` (or any SPECTRE
optimisation that saves ``xs_fixb/``), re-solves the field at each checkpoint, and calls
``FixedPointTracker.step`` per checkpoint. Results are saved to a ``.npz`` file.

Usage example::

    python track_fp.py case.toml 5 4 \
        --data-files xs_fixb/*.npy --poloidal-occurrences 4 --discover --output fp_metrics.npz

(Pass ``--data-files`` last so the shell-expanded glob does not swallow the positionals.)

Send ``SIGUSR1`` to stop cleanly after the current checkpoint and save the partial result --
``kill -USR1 <pid>`` on a local run, or ``scancel --signal=USR1 <SLURM_JOB_ID>`` under SLURM
(the banner prints the right one for you).
"""

import argparse
import os
import signal
from pathlib import Path
from time import perf_counter

import numpy as np


def _parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("toml", help="SPECTRE TOML matching the checkpoints")
    p.add_argument("repeat_period", type=int, help="Island chain period (denominator of iota)")
    p.add_argument("volume", type=int, help="Volume index hosting the chain (0 = axis volume)")
    p.add_argument(
        "--data-files",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Checkpoint x_<seg>_<iter>.npy files (shell glob, e.g. xs_fixb/*.npy). Pass last.",
    )
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


def select_checkpoints(data_files, start_at, stride, reverse, limit):
    """Order the given checkpoint files by optimizer iteration and apply start/stride/limit/reverse.

    The optimizer writes files named ``x_<segment>_<iteration>.npy``; the trailing integer is the
    iteration used for ordering and labelling. Returns a list of ``(iteration, path)`` pairs.
    """

    def iteration_of(path):
        return int(Path(path).stem.split("_")[-1])

    selected = sorted(
        ((iteration_of(f), Path(f)) for f in data_files if iteration_of(f) >= start_at),
        key=lambda pair: pair[0],
    )[::stride]
    if reverse:
        selected = selected[::-1]
    if limit > 0:
        selected = selected[:limit]
    return selected


def load_xins(paths):
    """Load each checkpoint file into its packed interface-DOF (xin) array."""
    return [np.load(str(p)) for p in paths]


def reconstruct_field(obj, xin):
    """Load one checkpoint's interface DOFs into ``obj``, re-solve the field, return the specout."""
    from spectre.force_targets import force_real
    from spectre.utils import get_xinit_specwrap, unpack_interfaces

    unpack_interfaces(xin, obj)
    force_real(get_xinit_specwrap(obj), obj)
    return obj.create_spectreout()


def print_header(toml_path, args, iters, stride, out, discover, guess):
    """Print the run banner and the platform-appropriate clean-stop command."""
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
        f"  run      : {len(iters)} checkpoints (iters {iters[0]}..{iters[-1]}, "
        f"{'reverse' if args.reverse else 'forward'}, stride {stride})"
    )
    print(f"  output   : {out.name}")
    if sid := os.environ.get("SLURM_JOB_ID"):
        print(f"  stop     : scancel --signal=USR1 {sid}")
    else:
        print(f"  stop     : kill -USR1 {os.getpid()}")
    if discover and args.poloidal_occurrences is None:
        print(
            "  ** WARNING: discover without poloidal_occurrences - the first cold lock pins the "
            "chain; pass --poloidal-occurrences to avoid a silent wrong-chain lock."
        )
    print(flush=True)


def main():  # noqa: D103
    args = _parse_args()

    # SPECTRE imports are deferred (heavy core; keeps --help instant).
    from spectre import SPECTRE
    from spectre.fixed_points import FixedPointTracker

    toml_path = Path(args.toml)
    if not toml_path.is_file():
        raise FileNotFoundError(f"TOML file not found: {toml_path}")

    out = Path(args.output) if args.output else toml_path.parent / "fp_metrics.npz"
    if out.parent == Path("."):
        out = toml_path.parent / out.name

    stride = max(1, int(args.stride))

    selected = select_checkpoints(args.data_files, args.start_at, stride, args.reverse, args.limit)
    if not selected:
        raise ValueError(
            f"No checkpoints with iter >= {args.start_at} among the {len(args.data_files)} "
            "given --data-files"
        )
    iters = [it for it, _ in selected]
    xins = load_xins([p for _, p in selected])
    n = len(selected)

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

    print_header(toml_path, args, iters, stride, out, discover, guess)

    iter_indices = np.array(iters, dtype=int)

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

    t_start = perf_counter()
    try:
        for i, (it, xin) in enumerate(zip(iters, xins, strict=True)):
            if stop[0]:
                print(f"  Stopping before iter {it} (SIGUSR1).", flush=True)
                break

            specout = reconstruct_field(obj, xin)
            report = tk.step(specout, i)

            if args.verbose:
                print(f"[{i + 1}/{n}] ({(perf_counter() - t_start) / 60:.1f} min elapsed)")
                tk.log(i, report, verbose=True, it=it)

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
