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
from spectre.force_minimizer import optimizer_default, optimizer_picard

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
        # The last sweep solved the field at the pre-blend iBns; the dforce inside re-solves
        # at the blended one (nudging dtflux within its tolerance) before the write, so the
        # state is recorded after it -- exactly what went into the file.
        sp.save_output(evaluate_force=True)
        state = {
            "bnserr": bnserr,
            "ibns": np.asarray(sp.allglobal_mod.ibns).copy(),
            "tflux": np.asarray(sp.input_list_mod.tflux).copy(),
            "dtflux": np.asarray(sp.allglobal_mod.dtflux).copy(),
        }
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


def _case(directory, name, physics=None, **min_params):
    """Copy the free-boundary W7-X case into *directory* with minimization overrides."""
    doc = tomlkit.parse(FREEB_DECK.read_text())
    doc["minimization"] = min_params
    if physics:
        doc["physics"].update(physics)
    path = directory / f"{name}.toml"
    if MPI.COMM_WORLD.Get_rank() == 0:
        path.write_text(tomlkit.dumps(doc))
    MPI.COMM_WORLD.Barrier()
    return str(path)


@pytest.fixture(scope="module")
def fm(tmp_path_factory):
    """A constructed (not yet minimized) free-boundary Force_Minimizer, in its own cwd.

    No optimizer is passed and ``initialize_bns_from_current`` is left at its default, so
    the constructor's own gate selects the driver and runs the collective iBns bootstrap
    (the Python replacement for the removed ``lautoinitbn`` input); ``gbnbld`` is the
    driver-level damping override.
    """
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("freeb_takeover")) if comm.Get_rank() == 0 else None
    directory = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        obj = Force_Minimizer(_case(directory, "takeover", max_niter=1, gbnbld=0.5))
        obj.end_toml = directory / "takeover_end.toml"  # written by the constructor
        yield obj
    finally:
        os.chdir(cwd)


def test_driver_owns_the_picard_iteration(fm):
    """A conventional free-boundary deck selects Picard and bootstraps iBns by default."""
    assert fm.picard_freeb
    assert fm.optimizer is optimizer_picard
    assert fm._bootstrap_bns
    # The bootstrap converged at construction: the state is already near self-consistent.
    assert float(fm.field_obj.allglobal_mod.bnserr) < float(fm.field_obj.input_list_mod.gbntol)


def test_refresh_damping_is_the_minimization_override(fm, monkeypatch):
    """minimization.gbnbld drives every refresh; global.gbnbld stays the namelist value.

    The blend runs in numpy on every rank, so the damping is resolved once and passed to
    update_bns explicitly, and the end-TOML records both inputs as given rather than a
    damping the run never used.
    """
    assert fm._bns_damping == 0.5
    assert float(fm.field_obj.input_list_mod.gbnbld) == 0.666  # the deck value

    end = tomlkit.parse(fm.end_toml.read_text())
    assert float(end["global"]["gbnbld"]) == 0.666
    assert float(end["minimization"]["gbnbld"]) == 0.5

    if fm.comm.Get_size() == 1:  # refresh_bns also hands the peers a worker-loop command
        calls = []
        monkeypatch.setattr(
            fm.field_obj, "update_bns", lambda mode, **kw: calls.append((mode, kw)) or 0.0
        )
        fm.refresh_bns()
        assert calls == [("refresh", {"damping": 0.5, "verbose": True})]


def test_a_restart_from_the_written_file_lands_on_the_same_state(tmp_path_factory):
    """A completed run's end file has to be a faithful restart point.

    The geometry checkpoints in ``xs_fixb`` carry no iBns, so the end file's ``bns`` is
    the only record of half the state. Reloading it has to honour that record: the file
    records the run's own iBns, and a reload must not overwrite it. The bootstrap is off
    for the run (the deck's Bns is already consistent, and the written file carries the
    flag): switched on, the restart would re-derive iBns from the plasma and land near,
    not on, the record.
    """
    from spectre import force_real
    from spectre.utils import get_freal_fsa_metric

    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("freeb_restart")) if comm.Get_rank() == 0 else None
    directory = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        case = _case(
            directory, "roundtrip", max_niter=2, inner_iter_cap=1, initialize_bns_from_current=False
        )
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

        # The restart path proper; the written file carries initialize_bns_from_current = false.
        restart = Force_Minimizer(written)
        ibns_restart = np.asarray(restart.field_obj.allglobal_mod.ibns).copy()
        resumed = get_freal_fsa_metric(restart.field_obj)
    finally:
        os.chdir(cwd)

    if comm.Get_rank() == 0:
        np.testing.assert_allclose(ibns_restart, ibns_final, rtol=1e-12, atol=1e-14)
        assert bare == pytest.approx(reported, rel=1e-8)
        assert resumed == pytest.approx(reported, rel=1e-8)


def test_default_optimizer_follows_the_free_boundary_gate(tmp_path_factory):
    """Only lfreebound=1, lfreebsolver=0, lbnszero=false decks get Picard by default.

    lbnszero pins iBns = 0 (update_bns returns nan, so bnserr could never reach its tol),
    the direct solve never reads iBns, and a fixed boundary has none. An explicit optimizer
    is kept as given; optimizer_default on a Picard deck is allowed but warned about.
    """
    comm = MPI.COMM_WORLD
    path = str(tmp_path_factory.mktemp("freeb_gate")) if comm.Get_rank() == 0 else None
    directory = Path(comm.bcast(path, root=0))
    cwd = os.getcwd()
    os.chdir(directory)
    try:
        obj = Force_Minimizer(_case(directory, "bnszero", physics={"lbnszero": True}))
        assert not obj.picard_freeb and not obj._bootstrap_bns
        assert obj.optimizer is optimizer_default

        obj = Force_Minimizer(str(CURR_PATH / "test_freeb_w7x_vc.toml"))
        assert not obj.picard_freeb and obj.optimizer is optimizer_default

        obj = Force_Minimizer(str(CURR_PATH / "qa_7vol_test.toml"), optimizer=optimizer_picard)
        assert not obj.picard_freeb and obj.optimizer is optimizer_picard

        with pytest.warns(UserWarning, match="optimizer_picard"):
            obj = Force_Minimizer(
                _case(directory, "frozen", initialize_bns_from_current=False),
                optimizer=optimizer_default,
            )
        assert obj.picard_freeb and obj.optimizer is optimizer_default
    finally:
        os.chdir(cwd)
