# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Resume bookkeeping of ``Force_Minimizer`` (`[minimization] resume = true`)."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tomlkit
from mpi4py import MPI

from spectre import SPECTRE, Force_Minimizer

CURR_PATH = Path(__file__).parent
FREEB_DECK = CURR_PATH / "test_freeb_w7x.toml"


def test_iter_arrays_sized_for_resumed_run(tmp_path_factory):
    """A resumed run sizes the iter_* arrays to the global iteration budget.

    On resume, the callback indexes the iter_* diagnostic lists by the *global* iteration
    counter, which reaches ``iteration_start + max_niter``. Sizing the lists to only
    ``max_niter + 1`` overflows them whenever ``iteration_start > 0``, crashing the run
    with an IndexError after the optimization already did useful work. This guards the
    array length against that regression.
    """
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("fm_resume")) if comm.Get_rank() == 0 else None
    directory = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        n_snapshots = 3
        if comm.Get_rank() == 0:
            doc = tomlkit.parse(FREEB_DECK.read_text())
            doc["minimization"] = {"resume": True, "max_niter": 5}  # must be > n_snapshots
            (directory / "resume_case.toml").write_text(tomlkit.dumps(doc))
        comm.Barrier()

        # Fabricate xs_fixb snapshots so _set_init_iter takes the resume path. The last
        # snapshot is loaded into the interfaces, so save the real xin to keep the shape.
        sp = SPECTRE("resume_case.toml", verbose=False)
        if comm.Get_rank() == 0:
            os.mkdir("xs_fixb")
            for i in range(n_snapshots):
                np.save(f"xs_fixb/x_000_{i:04}.npy", sp.xin)
        comm.Barrier()

        fm = Force_Minimizer("resume_case.toml")
    finally:
        os.chdir(cwd)

    # Resume path actually taken: iteration_start = num_infaces - 1 > 0.
    assert fm.iteration_start == n_snapshots - 1
    if fm.rank != 0:
        return  # the iter_* history lives on rank 0 only

    expected_len = fm.iteration_start + fm.min_params.max_niter + 1
    for name in (
        "iter_iter",
        "iter_nfev",
        "iter_cost",
        "iter_cost_red",
        "iter_step_norm",
        "iter_opt",
        "iter_fsa",
        "iter_freal",
    ):
        assert len(getattr(fm, name)) == expected_len, name
