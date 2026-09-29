# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""The state a free-boundary run ends on must be the state it writes out.

``iBns`` is half of the free-boundary state and is not part of the geometry checkpoint, so
a run whose end file carries an ``iBns`` that no force evaluation ever used cannot be
reloaded, resumed or compared against anything. Fortran ``spectre()`` used to step it on
every ``run()`` -- once from the constructor and once from ``postprocess`` -- which left
the reported force, the written file and the HDF5 field describing different states.

These tests pin the two properties that make the written file a faithful record: the
outer/inner driver is the only thing that steps ``iBns``, and reloading what a completed
run writes reproduces the force that run reported.
"""

import os.path
from pathlib import Path

import numpy as np
import pytest
import tomlkit
from mpi4py import MPI

from spectre import SPECTRE, Force_Minimizer, force_real
from spectre.force_minimizer import optimizer_picard
from spectre.utils import get_freal_fsa_metric

curr_path = os.path.dirname(__file__) + "/"
BASE_TOML = curr_path + "test_freeb_w7x.toml"


def _case(directory, name, **min_params):
    """Copy the free-boundary W7-X case into *directory* with minimization overrides."""
    doc = tomlkit.parse(Path(BASE_TOML).read_text())
    doc["minimization"] = min_params
    path = directory / f"{name}.toml"
    if MPI.COMM_WORLD.Get_rank() == 0:
        path.write_text(tomlkit.dumps(doc))
    MPI.COMM_WORLD.Barrier()
    return str(path)


@pytest.fixture(scope="module")
def fm(tmp_path_factory):
    """A constructed (not yet minimized) free-boundary Force_Minimizer, in its own cwd."""
    directory = tmp_path_factory.mktemp("freeb_takeover")
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        yield Force_Minimizer(_case(directory, "takeover", max_niter=1), optimizer=optimizer_picard)
    finally:
        os.chdir(cwd)


def test_driver_takes_over_the_picard_iteration(fm):
    """The Fortran free-boundary loop is off, so update_bns is the only mutator left."""
    assert fm.picard_freeb
    assert int(fm.field_obj.input_list_mod.lautoinitbn) == 0
    assert int(fm.field_obj.input_list_mod.mfreeits) == 0
    # lautoinitbn=1 in the input is honoured, as the Python bootstrap.
    assert fm._bootstrap_bns


def test_run_leaves_ibns_alone(fm):
    """A run() must not slip a Picard step in behind the optimizer's back.

    This is what made the end file unreloadable: the step lands after the last force
    evaluation, so the geometry was never scored against the iBns that gets written.
    """
    ibns_before = np.asarray(fm.field_obj.allglobal_mod.ibns).copy()
    fm.field_obj.run(save_output=False)
    np.testing.assert_array_equal(np.asarray(fm.field_obj.allglobal_mod.ibns), ibns_before)


def test_a_restart_from_the_written_file_lands_on_the_same_state(tmp_path):
    """A completed run's end file has to be a faithful restart point.

    The geometry checkpoints in ``xs_fixb`` carry no iBns, so the end file's ``bns`` is the
    only record of half the state. Reloading it has to honour that record. It did not:
    ``lautoinitbn=1`` -- which the Fortran documents as "user-supplied Bns will only be
    considered if LautoinitBn = 0" -- propagated into the written file, so constructing a
    minimizer from it re-derived iBns from the geometry and discarded the run's own value.
    A resumed run then silently continued from a different problem than the one it stopped.

    Handing the iteration to the driver fixes both halves: the file records the run's iBns,
    and it carries the flags that stop a reload from overwriting it.
    """
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        case = _case(tmp_path, "roundtrip", max_niter=2, inner_iter_cap=1)
        obj = Force_Minimizer(case, optimizer=optimizer_picard)
        obj.minimize()
        obj.postprocess(save_json=False, save_pkl=False, run_spec=True)

        ibns_final = np.asarray(obj.field_obj.allglobal_mod.ibns).copy()
        reported = get_freal_fsa_metric(obj.field_obj)

        written = str(tmp_path / "roundtrip_opt.toml")

        # A bare reload: no minimizer, so nothing can touch iBns either way.
        reloaded = SPECTRE(written, verbose=False)
        force_real(reloaded.xin, reloaded, filter_stellsym=obj.min_params.filter_stellsym)
        bare = get_freal_fsa_metric(reloaded)

        # The restart path proper, which runs spectre() on the way in.
        restart = Force_Minimizer(written, optimizer=optimizer_picard)
        ibns_restart = np.asarray(restart.field_obj.allglobal_mod.ibns).copy()
        resumed = get_freal_fsa_metric(restart.field_obj)
    finally:
        os.chdir(cwd)

    if MPI.COMM_WORLD.Get_rank() == 0:
        np.testing.assert_allclose(ibns_restart, ibns_final, rtol=1e-12, atol=1e-14)
        assert bare == pytest.approx(reported, rel=1e-8)
        assert resumed == pytest.approx(reported, rel=1e-8)
