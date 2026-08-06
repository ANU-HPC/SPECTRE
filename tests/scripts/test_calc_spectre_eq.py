# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""End-to-end test of the scripts/run/calc_spectre_eq.py driver script under mpirun."""

import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import tomlkit
from mpi4py import MPI

REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER = REPO_ROOT / "scripts" / "run" / "calc_spectre_eq.py"
BASE_TOML = REPO_ROOT / "tests" / "wrapper" / "qa_7vol_test.toml"

# A child inheriting these joins (and corrupts) the parent's MPI job instead of
# starting its own; strip them so the nested mpirun below is a fresh job.
MPI_JOB_ENV_PREFIXES = ("OMPI_", "PMIX_", "PMI_", "PRTE_")


def test_calc_spectre_eq_script(tmp_path):
    """Run the driver under mpirun on a small 3D case and check it completes.

    Launches ``mpirun -np 2 python calc_spectre_eq.py`` — the production invocation — on a
    copy of the qa_7vol wrapper case (Igeometry=3, so the coordinate-singular Zernike
    volume is exercised) capped at two minimizer iterations, from a temporary working
    directory (the solver writes its output ``.h5`` and ``xs_fixb`` artifacts to the
    current working directory).

    When pytest itself runs under mpirun, only world-rank 0 launches the child job and the
    results are broadcast, so every rank asserts the same outcome (rank-dependent outcomes
    would desynchronize the ranks' MPI schedules).
    """
    comm = MPI.COMM_WORLD

    payload = None
    if comm.rank == 0:
        doc = tomlkit.parse(BASE_TOML.read_text())
        doc["minimization"] = {"max_niter": 2}
        case = tmp_path / "qa_7vol_smoke.toml"
        case.write_text(tomlkit.dumps(doc))

        env = {k: v for k, v in os.environ.items() if not k.startswith(MPI_JOB_ENV_PREFIXES)}
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")

        cmd = ["mpirun", "--oversubscribe", "-np", "2", sys.executable, str(DRIVER), str(case)]
        try:
            result = subprocess.run(
                cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=600
            )
            h5_path = tmp_path / "qa_7vol_smoke_opt.h5"
            btemn = None
            if result.returncode == 0 and h5_path.is_file():
                with h5py.File(h5_path) as h5:
                    btemn = h5["output/Btemn"][()]  # (Mvol, iocons, mn)
            payload = {
                "returncode": result.returncode,
                "tail": result.stdout[-2000:] + result.stderr[-2000:],
                "stdout": result.stdout,
                "btemn": btemn,
            }
        except Exception as exc:  # never leave the other ranks hanging in the bcast
            payload = {"exception": repr(exc)}

    payload = comm.bcast(payload, root=0)

    assert "exception" not in payload, payload.get("exception")
    assert payload["returncode"] == 0, payload["tail"]
    assert "SPECTRE equilibrium calculation done!" in payload["stdout"]

    # Regression check for the interface field output: Btemn is filled per interface by
    # interface_bfield_harmonics in final_diagnostics. A stale-scratch bug once left every
    # interface with an identical copied vector, so pin that the harmonics are finite and
    # actually differ between the outer faces of volumes 1 and 2.
    btemn = payload["btemn"]
    assert btemn is not None, "driver did not write qa_7vol_smoke_opt.h5"
    assert np.isfinite(btemn).all()
    assert not np.array_equal(btemn[0, 1, :], btemn[1, 1, :])
