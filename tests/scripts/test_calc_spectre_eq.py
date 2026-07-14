"""End-to-end test of the scripts/run/calc_spectre_eq.py driver script."""

import os
import subprocess
import sys
from pathlib import Path

import tomlkit

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run" / "calc_spectre_eq.py"
BASE_TOML = REPO_ROOT / "tests" / "wrapper" / "qa_7vol_test.toml"


def test_calc_spectre_eq_script(tmp_path):
    """Run the driver script on a small 3D case and check it completes.

    Runs ``scripts/run/calc_spectre_eq.py`` as a subprocess on a copy of the
    qa_7vol wrapper case (Igeometry=3, so the coordinate-singular Zernike
    volume is exercised) capped at two minimizer iterations, from a temporary
    working directory (the solver writes its output ``.h5`` and ``xs_fixb``
    artifacts to the current working directory).
    """
    doc = tomlkit.parse(BASE_TOML.read_text())
    doc["minimization"] = {"max_niter": 2}
    case = tmp_path / "qa_7vol_smoke.toml"
    case.write_text(tomlkit.dumps(doc))

    env = dict(
        os.environ,
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MPLBACKEND="Agg",
    )
    result = subprocess.run(
        [sys.executable, str(DRIVER), str(case)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "SPECTRE equilibrium calculation done!" in result.stdout
    assert (tmp_path / "qa_7vol_smoke.h5").is_file()
