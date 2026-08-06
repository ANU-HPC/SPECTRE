#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Summarize perf_picard_vs_direct runs: wall time to reach <f> thresholds per config.

Reads every ``runs/perf_freeb/<tag>.csv`` (+ ``<tag>.json`` metadata), prints one row per
run with its lowest reached ``<f>`` and the wall time at which it first crossed each
threshold of a shared ladder, so "time to a given quality" is read off directly.
"""

import csv
import json
from pathlib import Path

import numpy as np

OUTDIR = Path(__file__).resolve().parents[2] / "runs" / "perf_freeb"


def main():
    """Print the cross-run comparison table."""
    runs = {}
    for csv_path in sorted(OUTDIR.glob("*.csv")):
        tag = csv_path.stem
        with open(csv_path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        meta = {}
        meta_path = OUTDIR / f"{tag}.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
        runs[tag] = (rows, meta)

    if not runs:
        print(f"no runs found under {OUTDIR}")
        return

    # Shared threshold ladder: log-spaced between the worst initial and best final <f>.
    fsa_all = {
        tag: np.array([float(r["fsa"]) for r in rows], dtype=float)
        for tag, (rows, _) in runs.items()
    }
    best = min(f.min() for f in fsa_all.values())
    start = min(f[0] for f in fsa_all.values())
    thresholds = 10.0 ** np.arange(np.floor(np.log10(start)), np.floor(np.log10(best)) - 1, -1)
    thresholds = [t for t in thresholds if t < start]

    header = ["run", "iters", "total_s", "best <f>"] + [f"t(<f><{t:.0e})" for t in thresholds]
    print("  ".join(f"{h:>16}" for h in header))
    for tag, (rows, meta) in runs.items():
        fsa = fsa_all[tag]
        wall = np.array([float(r["wall_s"]) for r in rows], dtype=float)
        cells = [
            tag,
            str(len(rows)),
            f"{meta.get('t_total_s', float('nan')):.0f}",
            f"{fsa.min():.3e}",
        ]
        for t in thresholds:
            crossed = np.nonzero(fsa < t)[0]
            cells.append(f"{wall[crossed[0]]:.1f}s" if crossed.size else "--")
        print("  ".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    main()
