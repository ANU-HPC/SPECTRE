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
FREEB_DECK = CURR_PATH / "test_freeb_w7x.toml"

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
    """Save the qa case with a force evaluation (the dforce completes the diagnostics state)."""
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
        sp.save_output(evaluate_force=True)
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


@pytest.fixture(scope="module")
def saved_freeb(tmp_path_factory):
    """Free-boundary case: solve, move iBns through update_bns, then save twice.

    ``freeb.h5`` is written by the default path right after the refresh -- which must warn,
    since the tracked field no longer matches the free-boundary state, and must not solve --
    and ``freeb_force.h5`` by ``save_output(evaluate_force=True)``. Returns the directory and
    the live state at the writes.
    """
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("save_output_freeb")) if comm.Get_rank() == 0 else None
    d = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    try:
        if comm.Get_rank() == 0:
            shutil.copy(FREEB_DECK, d / "freeb.toml")
        comm.Barrier()
        os.chdir(d)
        sp = SPECTRE("freeb.toml", verbose=False)
        sp.compute_fields()
        sp.update_bns(mode="refresh")
        with pytest.warns(UserWarning, match="not known"):
            sp.save_output()
        mn = int(sp.allglobal_mod.mn)
        state = {
            "ibns": np.asarray(sp.allglobal_mod.ibns)[:mn].copy(),
            "im": np.asarray(sp.allglobal_mod.im)[:mn].copy(),
            "in": np.asarray(sp.allglobal_mod.in_)[:mn] // int(sp.input_list_mod.nfp),
            "mpol": int(sp.input_list_mod.mpol),
            "ntor": int(sp.input_list_mod.ntor),
            "nvol": int(sp.input_list_mod.nvol),
        }
        sp.allglobal_mod.filename_h5 = "freeb_force.h5"
        sp.save_output(evaluate_force=True)
        state["vvolume"] = np.asarray(sp.allglobal_mod.vvolume).copy()
        state["forceerr"] = float(sp.allglobal_mod.forceerr)
        comm.Barrier()  # rank 0 finishes writing before any rank reads
    finally:
        os.chdir(cwd)
    return d, state


def test_input_mirror_is_prepared_from_the_live_free_boundary_state(saved_freeb):
    """The HDF5 input mirror, /output/Bns and the end-TOML all carry the refreshed iBns.

    prepare_endfile_writing copies the live iBns into the input_list Bns array that both the
    mirror and the TOML are written from; mirroring before that copy archived the deck's
    Bns next to a refreshed /output/Bns. Layouts: the mirror is Bns(-Ntor:Ntor, -Mpol:Mpol)
    read transposed by h5py, the TOML a sparse "(m, n)" map.
    """
    import tomlkit

    d, state = saved_freeb
    m, n = state["im"], state["in"]
    assert np.max(np.abs(state["ibns"])) > 1e-3  # a refreshed, non-trivial state

    end = tomlkit.parse((d / "freeb_end.toml").read_text())["physics"]["bns"]
    toml = np.array([float(end.get(f"({mm}, {nn})", 0.0)) for mm, nn in zip(m, n, strict=True)])
    np.testing.assert_array_equal(toml, state["ibns"])
    for name in ("freeb.h5", "freeb_force.h5"):
        with h5py.File(d / name, "r") as f:
            mirror = np.asarray(f["input/physics/Bns"])[m + state["mpol"], n + state["ntor"]]
            output = np.asarray(f["output/Bns"])
        np.testing.assert_array_equal(output, state["ibns"], err_msg=name)
        np.testing.assert_array_equal(mirror, state["ibns"], err_msg=name)


def test_volumes_are_written_on_both_paths(saved_freeb):
    """/output/volume is the recomputed plasma volume even without a force evaluation.

    The volume is a geometry integral and is always recomputed; ForceErr is only complete
    after the dforce that evaluate_force=True runs at the written geometry.
    """
    d, state = saved_freeb
    plasma = state["vvolume"][: state["nvol"]]
    assert np.all(plasma > 0.0)
    for name in ("freeb.h5", "freeb_force.h5"):
        with h5py.File(d / name, "r") as f:
            assert np.asarray(f["output/volume"]).item() == pytest.approx(np.sum(plasma)), name
    with h5py.File(d / "freeb_force.h5", "r") as f:
        assert np.asarray(f["output/ForceErr"]).item() == state["forceerr"] > 0.0


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
