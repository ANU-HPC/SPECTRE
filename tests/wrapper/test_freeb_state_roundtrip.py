# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""The free-boundary state a calculation ends on must be the state it writes out.

``iBns`` is half of the free-boundary state and is not part of the geometry checkpoint, so
an end file whose ``bns`` does not match what the run converged cannot be reloaded, resumed
or compared against anything. These tests pin that ``update_bns`` + ``save_output`` write a
faithful record: reloading the end-TOML reproduces the converged ``iBns`` and fluxes.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import tomlkit
from mpi4py import MPI

from spectre import SPECTRE, Force_Minimizer
from spectre.force_minimizer import optimizer_picard

CURR_PATH = Path(__file__).parent
FREEB_DECK = CURR_PATH / "test_freeb_w7x.toml"


@pytest.fixture(scope="module")
def roundtrip(tmp_path_factory):
    """Converge the Picard state, save the output, and reload the written end-TOML."""
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("freeb_roundtrip")) if comm.Get_rank() == 0 else None
    d = Path(comm.bcast(path, root=0))  # every rank works in the same directory
    cwd = os.getcwd()
    try:
        if comm.Get_rank() == 0:
            shutil.copy(FREEB_DECK, d / "case.toml")
        comm.Barrier()
        os.chdir(d)
        sp = SPECTRE("case.toml", verbose=False)
        bnserr = sp.update_bns(mode="converge", tol=1e-8, max_sweeps=30)
        state = {
            "bnserr": bnserr,
            "ibns": np.asarray(sp.allglobal_mod.ibns).copy(),
            "tflux": np.asarray(sp.input_list_mod.tflux).copy(),
            "dtflux": np.asarray(sp.allglobal_mod.dtflux).copy(),
        }
        sp.save_output()
        comm.Barrier()  # rank 0 finishes writing before any rank reads
        reloaded = SPECTRE("case_end.toml", verbose=False)
    finally:
        os.chdir(cwd)
    return state, reloaded


def test_converged_before_writing(roundtrip):
    """The fixture state must actually be self-consistent, or the roundtrip proves nothing."""
    state, _ = roundtrip
    assert state["bnserr"] < 1e-8


def test_ibns_roundtrips_through_the_end_toml(roundtrip):
    """Reloading the written file reproduces the converged iBns exactly."""
    state, reloaded = roundtrip
    np.testing.assert_allclose(
        np.asarray(reloaded.allglobal_mod.ibns), state["ibns"], rtol=1e-12, atol=1e-14
    )


def test_fluxes_roundtrip_through_the_end_toml(roundtrip):
    """The converged tflux survives the write + reload + preset rebuild."""
    state, reloaded = roundtrip
    mvol = int(reloaded.allglobal_mod.mvol)
    np.testing.assert_allclose(
        np.asarray(reloaded.input_list_mod.tflux)[:mvol],
        state["tflux"][:mvol],
        rtol=1e-12,
        atol=1e-14,
    )
    # preset rebuilt dtflux from the written tflux; it must match the converged increments.
    np.testing.assert_allclose(
        np.asarray(reloaded.allglobal_mod.dtflux), state["dtflux"], rtol=1e-10, atol=1e-14
    )


def test_reloaded_state_is_already_self_consistent(roundtrip):
    """A refresh on the reloaded state reports (nearly) the residual the run converged to.

    This is the operational meaning of a faithful restart file: continuing the Picard
    iteration from the written file starts where the previous run stopped.
    """
    _, reloaded = roundtrip
    bnserr = reloaded.update_bns(mode="refresh")
    assert bnserr < 1e-6


# --------------------------------------------------------------------------- driver takeover
# The tests below construct Force_Minimizer instances, which re-initialize the process-global
# Fortran state; they must stay after the `roundtrip`-fixture tests above.


def _case(directory, name, **min_params):
    """Copy the free-boundary W7-X case into *directory* with minimization overrides."""
    doc = tomlkit.parse(FREEB_DECK.read_text())
    doc["minimization"] = min_params
    path = directory / f"{name}.toml"
    if MPI.COMM_WORLD.Get_rank() == 0:
        path.write_text(tomlkit.dumps(doc))
    MPI.COMM_WORLD.Barrier()
    return str(path)


@pytest.fixture(scope="module")
def fm(tmp_path_factory):
    """A constructed (not yet minimized) free-boundary Force_Minimizer, in its own cwd.

    ``initialize_bns_from_current`` exercises the collective iBns bootstrap that runs in
    the constructor -- the Python replacement for the removed ``lautoinitbn`` input.
    """
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("freeb_takeover")) if comm.Get_rank() == 0 else None
    directory = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        yield Force_Minimizer(
            _case(directory, "takeover", max_niter=1, initialize_bns_from_current=True),
            optimizer=optimizer_picard,
        )
    finally:
        os.chdir(cwd)


def test_driver_owns_the_picard_iteration(fm):
    """update_bns is the only iBns mutator, and the Python bootstrap request is honoured."""
    assert fm.picard_freeb
    assert fm._bootstrap_bns
    # The bootstrap converged at construction: the state is already near self-consistent.
    assert float(fm.field_obj.allglobal_mod.bnserr) < float(fm.field_obj.input_list_mod.gbntol)


def test_a_restart_from_the_written_file_lands_on_the_same_state(tmp_path_factory):
    """A completed run's end file has to be a faithful restart point.

    The geometry checkpoints in ``xs_fixb`` carry no iBns, so the end file's ``bns`` is
    the only record of half the state. Reloading it has to honour that record: with the
    Fortran loop disabled the file records the run's own iBns, and it carries the flags
    that stop a reload from overwriting it.
    """
    from spectre import force_real
    from spectre.utils import get_freal_fsa_metric

    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("freeb_restart")) if comm.Get_rank() == 0 else None
    directory = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        case = _case(directory, "roundtrip", max_niter=2, inner_iter_cap=1)
        obj = Force_Minimizer(case, optimizer=optimizer_picard)
        obj.minimize()
        obj.postprocess(save_json=False, save_pkl=False, run_spec=True)

        ibns_final = np.asarray(obj.field_obj.allglobal_mod.ibns).copy()
        reported = get_freal_fsa_metric(obj.field_obj)

        comm.Barrier()
        written = str(directory / "roundtrip_opt.toml")

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

    if comm.Get_rank() == 0:
        np.testing.assert_allclose(ibns_restart, ibns_final, rtol=1e-12, atol=1e-14)
        assert bare == pytest.approx(reported, rel=1e-8)
        assert resumed == pytest.approx(reported, rel=1e-8)
