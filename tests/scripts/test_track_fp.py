# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Test the offline fixed-point tracking driver ``track_fp.py``."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACK_FP = REPO_ROOT / "scripts" / "minimization_monitoring" / "track_fp.py"
# The NCSX fixed-point example TOML and a few of its calc_spectre_eq checkpoints (the 3/5 chain in
# volume 3). Replaying them is exactly the "run track_fp.py on a folder of calc_spectre_eq output"
# workflow.
EXAMPLE_TOML = REPO_ROOT / "examples" / "analysis" / "finding_fixed_points" / "init.toml"
CHECKPOINT_DIR = REPO_ROOT / "tests" / "test_files" / "ncsx_checkpoints"


def test_track_fp_replays_checkpoints(tmp_path):
    """``track_fp.py`` replays saved checkpoints into an ``fp_metrics.npz`` with the 3/5 island.

    Runs the offline driver as a subprocess on the committed NCSX checkpoints and checks it exits
    cleanly, writes the ``.npz``, and locates the 3/5 chain (iota = 3/5) in at least one checkpoint.
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
    for key in ("iteration_indices", "island_rz", "iota", "greene_residue", "kind", "no_fp"):
        assert key in d, f"fp_metrics.npz is missing the '{key}' array"
    assert len(d["iteration_indices"]) == len(checkpoints), "one row per checkpoint expected"

    found = ~d["no_fp"]
    assert found.any(), "the 3/5 island should be located in at least one checkpoint"
    assert abs(abs(np.nanmedian(d["iota"][found])) - 3 / 5) < 5e-3, (
        f"located iota should be 3/5, got {np.nanmedian(d['iota'][found]):.4f}"
    )
