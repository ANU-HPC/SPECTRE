# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""``SPECTRE.save_output`` owns the output duties the Fortran driver used to have.

While ``run()`` still existed its output was the golden reference, and ``save_output`` was
pinned dataset-by-dataset against it (bitwise for the state, one-ulp tolerant for derived
outputs). With the driver excised these tests pin the content directly: the HDF5 file must
carry the input mirror, grid, output groups and vector potential, and the mirror must agree
with the ``_end.toml`` restart file it is written alongside.
"""

import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from mpi4py import MPI

from spectre import SPECTRE

CURR_PATH = Path(__file__).parent
QA_DECK = CURR_PATH / "qa_7vol_test.toml"

# One representative dataset per group run() used to write; losing any of these silently
# would break every downstream reader (SPECTREout, plotout, restart tooling).
REQUIRED_DATASETS = [
    "version",
    "grid/Rij",
    "grid/Zij",
    "grid/sg",
    "grid/BR",
    "input/physics/tflux",
    "input/physics/pflux",
    "input/physics/helicity",
    "input/physics/Lrad",
    "output/Btemn",
    "output/ForceErr",
    "output/volume",
    "output/Rbc",
    "vector_potential/Ate",
    "vector_potential/Aze",
]


def _h5_datasets(path):
    """Return {name: value} for every dataset in the HDF5 file."""
    items = {}
    with h5py.File(path, "r") as f:

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                items[name] = obj[()]

        f.visititems(visit)
    return items


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """Solve the qa case (one dforce completes the diagnostics state) and save the output."""
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("save_output")) if comm.Get_rank() == 0 else None
    d = Path(comm.bcast(path, root=0))  # every rank works in the same directory
    cwd = os.getcwd()
    try:
        if comm.Get_rank() == 0:
            shutil.copy(QA_DECK, d / "case.toml")
        comm.Barrier()
        os.chdir(d)
        sp = SPECTRE("case.toml", verbose=False)
        sp.force_modes_dforce()
        sp.save_output()
        comm.Barrier()  # rank 0 finishes writing before any rank reads
    finally:
        os.chdir(cwd)
    return d


def test_h5_has_the_required_datasets(saved):
    """Every output group a downstream reader relies on is present and finite."""
    data = _h5_datasets(saved / "case.h5")
    missing = [name for name in REQUIRED_DATASETS if name not in data]
    assert not missing, f"save_output lost datasets: {missing}"

    for name in REQUIRED_DATASETS:
        values = np.asarray(data[name])
        if np.issubdtype(values.dtype, np.number):
            assert np.isfinite(values).all(), name


def test_solved_state_is_written(saved):
    """The dforce products (volumes, force error) are written, not left zeroed."""
    data = _h5_datasets(saved / "case.h5")
    assert np.all(np.asarray(data["output/volume"]) > 0.0)
    assert np.asarray(data["output/ForceErr"]).item() > 0.0


def test_vector_potential_group_is_written(saved):
    """SPECTREout._postprocess reads /vector_potential unguarded; it has to exist."""
    with h5py.File(saved / "case.h5", "r") as f:
        assert "vector_potential" in f


def test_input_mirror_matches_the_end_toml(saved):
    """The state-synced inputs in the HDF5 mirror equal what the end-TOML records.

    Both are written by the same save_output call, after the flux write-back, so the
    restart file and the archived input group must agree with each other.
    """
    import tomlkit

    data = _h5_datasets(saved / "case.h5")
    end = tomlkit.parse((saved / "case_end.toml").read_text())

    mvol = np.asarray(end["physics"]["tflux"]).size
    for name in ("tflux", "pflux"):
        np.testing.assert_array_equal(
            np.asarray(data[f"input/physics/{name}"])[:mvol],
            np.asarray(end["physics"][name], dtype=np.float64),
            err_msg=name,
        )
