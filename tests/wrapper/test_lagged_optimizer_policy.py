# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Policy of the lagged-banded driver in ``optimizer_default``.

The lagged mode switches on a process-global Beltrami factor cache and consumes
Jacobian-vector products that only some configurations can produce. Both are checked here
without a SPECTRE instance: the driver is exercised against a stub exposing exactly the
Fortran state it touches, so the tests stay on the policy (which configurations enable
lagging, and what happens to the cache afterwards) rather than on the physics.
"""

import numpy as np
import pytest

import spectre.force_minimizer as fm
from spectre.input_dict import input_dict

MVOL = 4


class _AllGlobal:
    def __init__(self, cache_on=False):
        self.mvol = MVOL
        self.lbeltrami_factor_cache = cache_on
        self.force_jac_status = np.zeros(MVOL, dtype=np.int32)


class _BeltramiFactorCacheMod:
    def __init__(self):
        self.clear_calls = 0

    def factor_cache_clear(self):
        self.clear_calls += 1


class _FieldObj:
    def __init__(self, lconstraint=3, cache_on=False):
        self.allglobal_mod = _AllGlobal(cache_on)
        self.input_list_mod = input_dict(dict(mpol=4, ntor=4, nfp=1, lconstraint=lconstraint))
        self.lib = input_dict(dict(beltrami_factor_cache_mod=_BeltramiFactorCacheMod()))


class _Comm:
    def __init__(self, size=1):
        self.size = size
        self.sent = []

    def Get_size(self):
        return self.size

    def Send(self, payload, dest):
        self.sent.append((np.array(payload[0], copy=True), payload[1], dest))


class _Obj:
    """Stand-in for Force_Minimizer, holding what optimizer_default reads."""

    def __init__(self, lconstraint=3, cache_on=False, jvp_lag=3, comm_size=1):
        self.field_obj = _FieldObj(lconstraint, cache_on)
        self.comm = _Comm(comm_size)
        self.freeb_mode = 0
        self.picard_freeb = False
        self.iteration_start = 0
        self.callback = None
        self.Dxmn_Dxdof_at_xdof = lambda xdof, field_obj: np.eye(len(xdof))
        self.xdof_to_xmn = lambda xdof, field_obj: xdof
        self.min_params = input_dict(
            dict(
                gtol=1e-13,
                xtol=1e-22,
                ftol=1e-13,
                max_nfev=10,
                max_niter=10,
                method="trf",
                rcond_lstsq=None,
                tr_solver="banded",
                jvp_lag=jvp_lag,
                objective="freal",
                inface_repr="fourier",
                mpol_rz=None,
                ntor_rz=None,
                nt_dbb=None,
                nz_dbb=None,
                filter_stellsym=True,
            )
        )


@pytest.fixture()
def captured_lstsq(monkeypatch):
    """Replace the least-squares driver with a recorder; return the recorded calls."""
    calls = {}

    def fake_lstsq(f, x_init, **kwargs):
        calls.update(kwargs)
        return "result"

    monkeypatch.setattr(fm, "lstsq", fake_lstsq)
    return calls


def test_lagging_restores_and_clears_the_factor_cache(captured_lstsq):
    obj = _Obj()

    assert fm.optimizer_default(obj, np.zeros(3), None, None) == "result"

    assert "jvp" in captured_lstsq["tr_options"], "lagged mode should have installed a JVP"
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False, (
        "the caller's factor-cache setting was not restored"
    )
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_factor_cache_is_restored_after_a_failed_optimization(monkeypatch):
    def exploding_lstsq(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(fm, "lstsq", exploding_lstsq)
    obj = _Obj()

    with pytest.raises(RuntimeError, match="boom"):
        fm.optimizer_default(obj, np.zeros(3), None, None)

    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_a_caller_that_had_the_cache_on_keeps_it(captured_lstsq):
    obj = _Obj(cache_on=True)

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is True
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_factor_cache_scope_coordinates_all_rank_lifetimes():
    root = _FieldObj(cache_on=False)
    worker = _FieldObj(cache_on=True)
    worker_scope = fm._BeltramiFactorCacheScope(worker)
    commands = []

    def worker_command(command):
        commands.append(command)
        if command is fm._WorkerCommand.CACHE_ENTER:
            worker_scope.__enter__()
        elif command is fm._WorkerCommand.CACHE_CLEAR:
            worker_scope.clear()
        elif command is fm._WorkerCommand.CACHE_EXIT:
            worker_scope.__exit__(None, None, None)

    root_scope = fm._BeltramiFactorCacheScope(root, worker_command)
    with root_scope:
        assert root.allglobal_mod.lbeltrami_factor_cache is True
        assert worker.allglobal_mod.lbeltrami_factor_cache is True
        root_scope.clear()
        assert root.allglobal_mod.lbeltrami_factor_cache is True
        assert worker.allglobal_mod.lbeltrami_factor_cache is True

    assert root.allglobal_mod.lbeltrami_factor_cache is False
    assert worker.allglobal_mod.lbeltrami_factor_cache is True
    assert root.lib.beltrami_factor_cache_mod.clear_calls == 2
    assert worker.lib.beltrami_factor_cache_mod.clear_calls == 2
    assert commands == [
        fm._WorkerCommand.CACHE_ENTER,
        fm._WorkerCommand.CACHE_CLEAR,
        fm._WorkerCommand.CACHE_EXIT,
    ]


def test_factor_cache_scope_restores_all_ranks_after_exception():
    root = _FieldObj(cache_on=True)
    worker = _FieldObj(cache_on=False)
    worker_scope = fm._BeltramiFactorCacheScope(worker)

    def worker_command(command):
        if command is fm._WorkerCommand.CACHE_ENTER:
            worker_scope.__enter__()
        elif command is fm._WorkerCommand.CACHE_EXIT:
            worker_scope.__exit__(RuntimeError, RuntimeError("boom"), None)

    root_scope = fm._BeltramiFactorCacheScope(root, worker_command)
    with pytest.raises(RuntimeError, match="boom"), root_scope:
        raise RuntimeError("boom")

    assert root.allglobal_mod.lbeltrami_factor_cache is True
    assert worker.allglobal_mod.lbeltrami_factor_cache is False
    assert root.lib.beltrami_factor_cache_mod.clear_calls == 1
    assert worker.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_multi_rank_lagging_sends_cache_scope_protocol(monkeypatch, captured_lstsq):
    mpi_int = object()
    mpi_double = object()
    monkeypatch.setattr(fm, "MPI_INT", mpi_int, raising=False)
    monkeypatch.setattr(fm, "MPI_DOUBLE", mpi_double, raising=False)
    obj = _Obj(comm_size=2)

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" in captured_lstsq["tr_options"]
    commands = [
        int(payload)
        for payload, datatype, destination in obj.comm.sent
        if payload.ndim == 0 and datatype is mpi_int and destination == 1
    ]
    assert commands == [fm._WorkerCommand.CACHE_ENTER, fm._WorkerCommand.CACHE_EXIT]
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_degraded_jvp_retry_clears_and_refills_all_rank_caches(monkeypatch):
    import spectre.force_targets as force_targets

    mpi_int = object()
    mpi_double = object()
    monkeypatch.setattr(fm, "MPI_INT", mpi_int, raising=False)
    monkeypatch.setattr(fm, "MPI_DOUBLE", mpi_double, raising=False)
    obj = _Obj(comm_size=2)
    attempts = 0

    def force_real_jvp(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        status = np.ones(MVOL, dtype=np.int32) if attempts == 1 else np.zeros(MVOL, dtype=np.int32)
        return np.zeros((2, 1, 1)), status

    def invoking_lstsq(f, x_init, **kwargs):
        result = kwargs["tr_options"]["jvp"](np.zeros(3), np.ones((3, 1)))
        np.testing.assert_array_equal(result, 0.0)
        return "result"

    monkeypatch.setattr(force_targets, "force_real_jvp", force_real_jvp)
    monkeypatch.setattr(fm, "lstsq", invoking_lstsq)

    assert fm.optimizer_default(obj, np.zeros(3), None, None) == "result"
    commands = [
        int(payload)
        for payload, datatype, destination in obj.comm.sent
        if payload.ndim == 0 and datatype is mpi_int and destination == 1
    ]
    assert commands == [
        fm._WorkerCommand.CACHE_ENTER,
        fm._WorkerCommand.JVP,
        1,
        fm._WorkerCommand.CACHE_CLEAR,
        fm._WorkerCommand.JVP,
        1,
        fm._WorkerCommand.CACHE_EXIT,
    ]
    assert attempts == 2
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 2
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False


def test_rank_zero_stops_workers_after_optimizer_exception(monkeypatch):
    mpi_int = object()
    mpi_double = object()
    monkeypatch.setattr(fm, "MPI_INT", mpi_int, raising=False)
    monkeypatch.setattr(fm, "MPI_DOUBLE", mpi_double, raising=False)
    minimizer = fm.Force_Minimizer.__new__(fm.Force_Minimizer)
    minimizer.rank = 0
    minimizer.comm = _Comm(size=2)
    minimizer.freeb_mode = 0
    minimizer.x_init = np.zeros(3)
    minimizer.obj_func = None
    minimizer.jac_func = None

    def exploding_optimizer(*args):
        raise RuntimeError("boom")

    minimizer.optimizer = exploding_optimizer
    with pytest.raises(RuntimeError, match="boom"):
        minimizer.minimize()

    commands = [
        int(payload)
        for payload, datatype, destination in minimizer.comm.sent
        if payload.ndim == 0 and datatype is mpi_int and destination == 1
    ]
    assert commands == [fm._WorkerCommand.STOP]


def test_factor_cache_scope_is_replaceable_by_a_fake(monkeypatch, captured_lstsq):
    events = []

    class FakeCacheScope:
        def __enter__(self):
            events.append("enter")
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append("exit")

        def clear(self):
            events.append("clear")

    fake_scope = FakeCacheScope()
    monkeypatch.setattr(fm, "_make_beltrami_factor_cache_scope", lambda field_obj: fake_scope)

    obj = _Obj()
    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert events == ["enter", "exit"]


def test_lconstraint_2_enables_lagging(captured_lstsq):
    """force_real_jvp linearises the fixed-boundary helicity constraint, so lagging engages."""
    obj = _Obj(lconstraint=2)

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" in captured_lstsq["tr_options"], (
        "force_real_jvp supports fixed-boundary lconstraint = 2 and should be lagged against"
    )
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 1


def test_picard_free_boundary_does_not_enable_lagging(captured_lstsq, capsys):
    obj = _Obj(comm_size=2)
    obj.freeb_mode = 1
    obj.picard_freeb = True

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" not in captured_lstsq["tr_options"], (
        "the Picard free-boundary residual changes between iterations"
    )
    assert "free-boundary" in capsys.readouterr().out
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 0
    assert obj.comm.sent == []


def test_direct_free_boundary_enables_lagging(captured_lstsq):
    """Direct VC has a fixed residual definition, so lagged JVP remains eligible."""
    obj = _Obj()
    obj.freeb_mode = 1
    obj.picard_freeb = False

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" in captured_lstsq["tr_options"]


@pytest.mark.parametrize(
    "rejection",
    ["nonbanded", "wrong_objective", "henneberg", "truncated_mpol", "truncated_ntor"],
)
def test_other_lagging_rejections_never_touch_any_rank_cache(rejection, captured_lstsq, capsys):
    obj = _Obj(comm_size=2)
    if rejection == "nonbanded":
        obj.min_params.tr_solver = "exact"
    elif rejection == "wrong_objective":
        obj.min_params.objective = "freal,len"
    elif rejection == "henneberg":
        obj.min_params.inface_repr = "henn"
    elif rejection == "truncated_mpol":
        obj.min_params.mpol_rz = 3
    elif rejection == "truncated_ntor":
        obj.min_params.ntor_rz = 3

    fm.optimizer_default(obj, np.zeros(3), None, None)

    assert "jvp" not in captured_lstsq["tr_options"]
    assert "ignored" in capsys.readouterr().out
    assert obj.field_obj.allglobal_mod.lbeltrami_factor_cache is False
    assert obj.field_obj.lib.beltrami_factor_cache_mod.clear_calls == 0
    assert obj.comm.sent == []


def test_degraded_jacobian_rebuild_warns_and_continues(captured_lstsq, capsys):
    obj = _Obj()
    built = []

    def jac(xdof, obj_):
        built.append(xdof)
        return np.zeros((2, 3))

    fm.optimizer_default(obj, np.zeros(3), None, jac)
    jac_used = captured_lstsq["jac"]

    # a clean build is passed through silently
    obj.field_obj.allglobal_mod.force_jac_status[:] = 0
    jac_used(np.zeros(3), obj)
    assert len(built) == 1
    assert "degraded linear solves" not in capsys.readouterr().out

    # a build whose linearised solves degraded is still returned, with a warning
    obj.field_obj.allglobal_mod.force_jac_status[2] = 1
    result = jac_used(np.zeros(3), obj)
    assert len(built) == 2
    assert result.shape == (2, 3)
    assert "WARNING: the force Jacobian was built from degraded linear solves" in (
        capsys.readouterr().out
    )
