# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""End-to-end test of the scripts/run/calc_spectre_eq.py driver script."""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

from spectre.file_io import read_input_parameters_from_toml, write_input_parameters_to_toml
from spectre.file_io.input_parameters import MinimizationParameters

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run" / "calc_spectre_eq.py"
BASE_TOML = REPO_ROOT / "tests" / "wrapper" / "qa_3vol_light.toml"


def test_calc_spectre_eq_script(tmp_path):
    """Run the driver script on a small 3D case and check it completes.

    Runs ``scripts/run/calc_spectre_eq.py`` as a subprocess on a copy of the
    qa_3vol_light wrapper case (Igeometry=3, so the coordinate-singular Zernike
    volume is exercised) capped at two minimizer iterations, from a temporary
    working directory (the solver writes its output ``.h5`` and ``xs_fixb``
    artifacts to the current working directory).
    """
    params = read_input_parameters_from_toml(str(BASE_TOML))
    params = params.model_copy(update={"minimization": MinimizationParameters(max_niter=2)})
    case = tmp_path / "qa_3vol_smoke.toml"
    write_input_parameters_to_toml(params, str(case))

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
        timeout=180,
    )

    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "SPECTRE equilibrium calculation done!" in result.stdout
    assert (tmp_path / "qa_3vol_smoke.h5").is_file()

    # Regression check for the interface field output: Btemn is filled per interface by
    # interface_bfield_harmonics in final_diagnostics. A stale-scratch bug once left every
    # interface with an identical copied vector, so pin that the harmonics are finite and
    # actually differ between the outer faces of volumes 1 and 2.
    with h5py.File(tmp_path / "qa_3vol_smoke.h5") as h5:
        btemn = h5["output/Btemn"][()]  # (Mvol, iocons, mn)
    assert np.isfinite(btemn).all()
    assert not np.array_equal(btemn[0, 1, :], btemn[1, 1, :])


def test_calc_spectre_eq_writes_fp_tracking_slots(tmp_path):
    """With ``track_axis`` on, the driver writes ``/iterations/fp`` once, into the final h5 only.

    Runs the driver on the qa_3vol_light case (monotonic iota, so no island -- axis-only tracking is
    enough to exercise the write path) with in-loop fixed-point tracking enabled for two
    iterations, then checks that the ``/iterations/fp`` group lands in the final ``_opt.h5``
    (with the expected axis dataset and scalar metadata attributes) and that the intermediate
    ``<stem>.h5`` stays free of it -- nothing is written to disk during the iterations.
    """
    params = read_input_parameters_from_toml(str(BASE_TOML))
    params = params.model_copy(
        update={"minimization": MinimizationParameters(max_niter=2, track_axis=True)}
    )
    case = tmp_path / "qa_3vol_fp.toml"
    write_input_parameters_to_toml(params, str(case))

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(DRIVER), str(case)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    # One writer, one file: fp lands in the final _opt.h5 and nowhere else.
    final_h5 = tmp_path / "qa_3vol_fp_opt.h5"
    intermediate_h5 = tmp_path / "qa_3vol_fp.h5"
    assert final_h5.is_file(), f"final dump missing; found {list(tmp_path.glob('*.h5'))}"
    if intermediate_h5.is_file():
        with h5py.File(intermediate_h5, "r") as h5:
            assert "iterations/fp" not in h5, "intermediate h5 must not carry /iterations/fp"

    with h5py.File(final_h5, "r") as h5:
        fp = h5["iterations/fp"]
        axis_rz = fp["axis_rz"][:]  # (n_steps, 2)
        assert axis_rz.shape[1] == 2
        # At least the first accepted iterate must have a finite axis location.
        assert np.isfinite(axis_rz[0]).all(), "axis position not recorded"
        # Scalar run metadata is stored as group attributes.
        for key in ("nfp", "volume_start"):
            assert key in fp.attrs, f"missing /iterations/fp attribute {key!r}"
