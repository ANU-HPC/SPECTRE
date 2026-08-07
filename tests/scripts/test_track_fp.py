# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Test the offline fixed-point tracking driver ``track_fp.py``."""

import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_FP = REPO_ROOT / "scripts" / "minimization_monitoring" / "track_fp.py"
# The NCSX fixed-point example TOML and three consecutive calc_spectre_eq checkpoints of it (the
# 3/5 chain in volume 3). Replaying them is exactly the "run track_fp.py on a folder of
# calc_spectre_eq output" workflow. Iterations 7-9 are committed rather than the first three: the
# opening iterations move the interfaces enough that every step cold-rediscovers, so only a later
# window exercises the anchor-follow path.
EXAMPLE_TOML = REPO_ROOT / "examples" / "analysis" / "finding_fixed_points" / "init.toml"
CHECKPOINT_DIR = REPO_ROOT / "tests" / "test_files" / "ncsx_checkpoints"


def test_track_fp_replays_checkpoints(tmp_path):
    """``track_fp.py`` replays saved checkpoints into an ``fp_metrics.npz`` with the 3/5 island.

    Runs the offline driver as a subprocess on the committed NCSX checkpoints and checks it exits
    cleanly, writes the ``.npz``, locates the 3/5 chain (iota = 3/5) in every checkpoint, and
    anchor-follows the member discovered in the first one.
    """
    checkpoints = sorted(str(p) for p in CHECKPOINT_DIR.glob("x_*.npy"))
    assert checkpoints, f"no committed checkpoints in {CHECKPOINT_DIR}"
    out_npz = tmp_path / "fp_metrics.npz"

    result = subprocess.run(
        [
            sys.executable,
            str(TRACK_FP),
            str(EXAMPLE_TOML),
            "-v",
            "3",
            "-m",
            "5",
            "-n",
            "3",  # no --guess -> self-discover
            "--output",
            str(out_npz),
            "--data-files",
            *checkpoints,  # glob last, so it does not swallow the positionals
        ],  # fmt: skip
        cwd=tmp_path,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg"),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert out_npz.is_file(), "track_fp.py did not write the output .npz"

    d = np.load(out_npz)
    for key in (
        "iteration_indices",
        "island_rz",
        "iota",
        "greene_residue",
        "kind",
        "no_fp",
        "provenance",
        "displacement",
        "greene_residue_delta",
    ):
        assert key in d, f"fp_metrics.npz is missing the '{key}' array"
    assert len(d["iteration_indices"]) == len(checkpoints), "one row per checkpoint expected"

    # The 3/5 chain is present in all three committed checkpoints: the first acquisition is
    # necessarily a cold discovery, and the later checkpoints must anchor-follow the same member
    # (this is what the three-checkpoint sequence pays for).
    found = ~d["no_fp"]
    assert found.all(), "the 3/5 island should be located in every committed checkpoint"
    assert d["provenance"][0] == "cold", "the first acquisition is necessarily a cold discovery"
    assert (d["provenance"][1:] == "follow").all(), (
        f"checkpoints 2-3 must anchor-follow the member found in checkpoint 1, "
        f"got {d['provenance'].tolist()}"
    )
    # displacement is the geometric convergence metric: defined (finite) exactly on follow steps.
    assert np.isnan(d["displacement"][0]), "a cold acquisition has no displacement"
    assert np.isfinite(d["displacement"][1:]).all(), "follow steps must record the displacement"
    assert (np.abs(np.abs(d["iota"][found]) - 3 / 5) < 5e-3).all(), (
        f"located iota should be 3/5, got {d['iota'][found]}"
    )


def test_track_fp_sigusr1_saves_partial_output(tmp_path):
    """SIGUSR1 stops the replay cleanly after the current checkpoint and keeps the partial npz.

    Starts the replay, waits for the first per-checkpoint report on stdout, sends SIGUSR1, and
    checks the driver exits 0 with an npz holding fewer rows than checkpoints -- all keys present
    and no NaN tail (the tracker slices its arrays to the processed rows).
    """
    checkpoints = sorted(str(p) for p in CHECKPOINT_DIR.glob("x_*.npy"))
    out_npz = tmp_path / "fp_metrics_partial.npz"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(TRACK_FP),
            str(EXAMPLE_TOML),
            "-v",
            "3",
            "-m",
            "5",
            "-n",
            "3",
            "--verbose",  # the per-checkpoint progress lines drive the stop timing below
            "--output",
            str(out_npz),
            "--data-files",
            *checkpoints,
        ],  # fmt: skip
        cwd=tmp_path,
        env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg"),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = []
    try:
        for line in proc.stdout:  # wait for the first completed checkpoint, then ask for a stop
            lines.append(line)
            if line.startswith("[1/"):
                proc.send_signal(signal.SIGUSR1)
                break
        else:
            pytest.fail("never saw the first per-checkpoint report:\n" + "".join(lines))
        lines.extend(proc.stdout)  # drain to EOF so the subprocess can finish
        assert proc.wait(timeout=180) == 0, "".join(lines)[-2000:]
    finally:
        proc.kill()

    d = np.load(out_npz)
    n_rows = len(d["iteration_indices"])
    assert 1 <= n_rows < len(checkpoints), (
        f"expected a partial replay, got {n_rows}/{len(checkpoints)} rows"
    )
    for key in ("island_rz", "iota", "no_fp", "provenance", "displacement"):
        assert key in d and len(d[key]) == n_rows, f"partial npz key {key!r} missing or untrimmed"
    assert np.isfinite(d["island_rz"]).all(), "an interrupted run must export no NaN tail"
