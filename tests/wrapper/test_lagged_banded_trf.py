# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Verify the lagged-banded matrix-free trust-region iteration (tr_options 'jvp'/'lag').

Between dense Jacobian rebuilds the iteration steps from a fresh-JVP reduced model over
a lagged subspace (least_squares/lagged.py, wired in trf.trf_no_bounds). It must reach
the same minimizer as the exact SVD path with strictly fewer dense Jacobian builds, and
must never declare convergence from lagged quantities.
"""

import importlib
import os.path
from types import SimpleNamespace

import numpy as np
import pytest
from helpers import spectre_with_lconstraint
from numpy.linalg import norm

from spectre.least_squares.lagged import (
    BandedTrOptions,
    JvpResult,
    JvpStatus,
    LaggedAction,
    LaggedJacobianController,
    ReducedTrustRegionModel,
    orthonormal_columns,
)
from spectre.least_squares.trf import trf_no_bounds

curr_path = os.path.dirname(__file__) + "/"

N_BLOCKS = 6
RPB = 40  # residual rows per interface block
CPB = 7  # geometry DOFs per interface block


def _block_tridiagonal_lsq_problem(seed):
    """Mildly nonlinear least squares with an exactly block-tridiagonal Jacobian."""
    rng = np.random.default_rng(seed)
    m, n = N_BLOCKS * RPB, N_BLOCKS * CPB
    A = np.zeros((m, n))
    for r in range(N_BLOCKS):
        c_lo = max(r - 1, 0) * CPB
        c_hi = min(r + 2, N_BLOCKS) * CPB
        A[r * RPB : (r + 1) * RPB, c_lo:c_hi] = rng.standard_normal((RPB, c_hi - c_lo))
    # Small residual at the minimizer, so the gradient can be driven to gtol cleanly.
    x_star = 0.3 * rng.standard_normal(n)
    b = A @ (x_star + 0.05 * x_star**3) + 1e-3 * rng.standard_normal(m)
    x0 = x_star + 0.5 * rng.standard_normal(n)

    def fun(x):
        return A @ (x + 0.05 * x**3) - b

    def jac(x, f=None):
        return A * (1.0 + 0.15 * x**2)  # column scaling keeps the block structure

    def jvp(x, dirs):
        return jac(x) @ dirs

    return fun, jac, jvp, x0


def _run_trf(fun, jac, x0, tr_solver, tr_options):
    f0 = fun(x0)
    J0 = jac(x0, f0)
    counter = {"njev": 1}

    def jac_counted(x, f):
        counter["njev"] += 1
        return jac(x, f)

    result = trf_no_bounds(
        fun,
        jac_counted,
        x0,
        f0,
        J0,
        ftol=1e-15,
        xtol=1e-15,
        gtol=1e-8,
        max_nfev=400,
        max_niter=100,
        x_scale=1.0,
        loss_function=None,
        tr_solver=tr_solver,
        tr_options=tr_options,
        verbose=0,
        cb=None,
    )
    assert result.njev == counter["njev"]
    return result


def test_reduced_model_validates_basis_and_jvp_shapes():
    with pytest.raises(ValueError, match="equal shape"):
        orthonormal_columns([np.ones(2), np.ones(3)])
    with pytest.raises(ValueError, match="positive"):
        orthonormal_columns([np.ones(2)], drop_tol=0.0)

    Q = np.eye(2)
    f = np.ones(3)
    with pytest.raises(ValueError, match="W must have shape"):
        ReducedTrustRegionModel(Q, np.ones((3, 1)), f)
    with pytest.raises(ValueError, match="orthonormal"):
        ReducedTrustRegionModel(2.0 * Q, np.ones((3, 2)), f)


def test_banded_options_translate_the_public_mapping():
    def jvp(x, directions):
        return directions

    options = BandedTrOptions.from_mapping({"n_blocks": np.int64(3), "lag": "2", "jvp": jvp})
    assert options == BandedTrOptions(n_blocks=3, lag=2, jvp=jvp, lag_hist=4)
    with pytest.raises(KeyError, match="n_blocks"):
        BandedTrOptions.from_mapping({})
    assert BandedTrOptions.from_mapping({"n_blocks": 3, "lag": -1}).lag == -1
    assert BandedTrOptions.from_mapping({"n_blocks": 3, "lag": 8}).lag_hist == 8
    explicit = BandedTrOptions.from_mapping({"n_blocks": 3, "lag": 2, "lag_hist": 1})
    assert explicit.lag_hist == 1


def test_lagged_controller_transitions_with_fake_evaluators():
    calls = []
    current_jacobian = np.array([[2.0, 0.0], [0.0, 3.0], [1.0, -1.0]])

    def fake_jvp(x, directions):
        calls.append((x.copy(), directions.copy()))
        return current_jacobian @ directions

    options = BandedTrOptions.from_mapping({"n_blocks": 1, "lag": 2, "jvp": fake_jvp})
    controller = LaggedJacobianController(options)
    assert controller.initialize_fresh_jacobian(current_jacobian).available
    assert controller.request_iteration(1.0, 1e-8).action is LaggedAction.USE_FRESH_MODEL

    next_action = controller.record_step_accepted(np.array([0.1, -0.2]), ratio=0.8)
    assert next_action is LaggedAction.USE_LAGGED_MODEL
    assert controller.request_iteration(1.0, 1e-8).action is LaggedAction.USE_LAGGED_MODEL

    residual = np.array([0.5, -0.25, 0.1])
    decision = controller.request_model(
        np.array([0.3, 0.2]),
        np.array([1.0, -0.5]),
        Delta=0.4,
        alpha=0.0,
        scale=np.ones(2),
        residual=residual,
    )
    assert decision.action is LaggedAction.USE_LAGGED_MODEL
    assert decision.jvp_result.status is JvpStatus.SUCCESS
    assert decision.model is not None
    assert len(calls) == 1

    controller.record_trial_started()
    assert controller.request_fresh_convergence_verification(2) is None
    assert controller.request_iteration(1.0, 1e-8).action is LaggedAction.REBUILD_JACOBIAN
    assert controller.consume_restore_requirement()
    controller.record_rebuild_complete()
    assert controller.request_iteration(1.0, 1e-8).action is LaggedAction.USE_FRESH_MODEL


def test_lagged_subspace_widens_before_spending_jvp_columns():
    """Diverse history and three radius samples widen the post-QR reduced model."""
    m, n = 8, 6
    rng = np.random.default_rng(2)
    J = rng.standard_normal((m, n))
    captured = {}

    def probing_jvp(x, directions):
        captured["directions"] = directions.copy()
        return J @ directions

    controller = LaggedJacobianController(
        BandedTrOptions.from_mapping({"n_blocks": 1, "lag": 4, "jvp": probing_jvp})
    )
    assert controller.initialize_fresh_jacobian(J).available
    for step in rng.standard_normal((3, n)):
        controller.record_step_accepted(step, ratio=0.9)

    decision = controller.request_model(
        np.zeros(n),
        rng.standard_normal(n),
        Delta=0.5,
        alpha=0.0,
        scale=1.0,
        residual=rng.standard_normal(m),
    )
    assert decision.action is LaggedAction.USE_LAGGED_MODEL
    assert decision.model.k >= 4
    assert captured["directions"].shape[1] == decision.model.k


def test_lagged_candidates_use_three_radii_and_drop_collinear_history(monkeypatch):
    """The same factorization supplies all radii and QR removes redundant candidates."""
    calls = []
    captured = {}
    gradient = np.array([1.0, 2.0, -1.0, 0.5])

    def solve(g_h, radius, initial_alpha=None):
        calls.append(radius)
        return SimpleNamespace(
            available=True,
            step=SimpleNamespace(step=(1.0 + radius) * gradient),
        )

    def jvp(x, directions):
        captured["width"] = directions.shape[1]
        return np.vstack([directions, directions[:1]])

    controller = LaggedJacobianController(BandedTrOptions(n_blocks=1, lag=4, jvp=jvp, lag_hist=2))
    controller._banded_system = SimpleNamespace(solve=solve)
    controller.record_step_accepted(2.0 * gradient, ratio=0.9)
    controller.record_step_accepted(-3.0 * gradient, ratio=0.9)

    decision = controller.request_model(
        np.zeros(4),
        gradient,
        Delta=0.75,
        alpha=0.0,
        scale=1.0,
        residual=np.ones(5),
    )
    assert calls == [0.75, 0.375, 3.0]
    assert decision.model.k == 1
    assert captured["width"] == 1
    assert controller.diagnostics.history_length == 2


def test_step_history_is_newest_first_and_capped(monkeypatch):
    """Candidate construction offers only the configured recent steps, newest first."""
    captured = {}

    def capture_columns(columns, drop_tol=1e-12):
        captured["columns"] = [np.asarray(column).copy() for column in columns]
        return np.eye(3, 1)

    lagged_module = importlib.import_module("spectre.least_squares.lagged")
    monkeypatch.setattr(lagged_module, "orthonormal_columns", capture_columns)
    controller = LaggedJacobianController(
        BandedTrOptions(
            n_blocks=1,
            lag=4,
            jvp=lambda x, directions: np.ones((4, directions.shape[1])),
            lag_hist=2,
        )
    )
    controller._banded_system = SimpleNamespace(
        solve=lambda *args, **kwargs: SimpleNamespace(available=False)
    )
    steps = [np.full(3, value) for value in (1.0, 2.0, 3.0)]
    for step in steps:
        controller.record_step_accepted(step, ratio=0.9)
    controller.request_model(np.zeros(3), np.array([4.0, 0.0, 0.0]), 1.0, 0.0, 1.0, np.ones(4))

    np.testing.assert_array_equal(captured["columns"][0], steps[2])
    np.testing.assert_array_equal(captured["columns"][1], steps[1])
    assert controller.diagnostics.history_length == 2


def test_release_stale_model_drops_all_large_model_state():
    """Pre-rebuild release makes stale factors and reduced arrays unavailable."""
    J = np.eye(3)
    controller = LaggedJacobianController(
        BandedTrOptions(n_blocks=1, lag=2, jvp=lambda x, directions: J @ directions)
    )
    assert controller.initialize_fresh_jacobian(J).available
    controller.record_step_accepted(np.ones(3), ratio=0.9)
    assert (
        controller.request_model(np.zeros(3), np.ones(3), 1.0, 0.0, 1.0, np.ones(3)).model
        is not None
    )

    controller.release_stale_model()
    diagnostics = controller.diagnostics
    assert not diagnostics.banded_model_available
    assert diagnostics.reduced_subspace_dimension == 0
    assert controller.request_iteration(1.0, 1e-8).action is LaggedAction.USE_LAGGED_MODEL
    assert (
        controller.request_model(np.zeros(3), np.ones(3), 1.0, 0.0, 1.0, np.ones(3)).action
        is LaggedAction.REBUILD_JACOBIAN
    )


def test_trf_releases_stale_model_before_rebuild_callback(monkeypatch):
    """The rebuild callback never overlaps the outgoing factorization and reduced model."""
    fun, jac, jvp, x0 = _block_tridiagonal_lsq_problem(5)
    release_state = {"released": False, "jac_calls": 0, "verified": 0}
    original_release = LaggedJacobianController.release_stale_model

    def tracked_release(controller):
        original_release(controller)
        release_state["released"] = True

    def checked_jac(x, f=None):
        release_state["jac_calls"] += 1
        if release_state["jac_calls"] > 1:
            assert release_state["released"]
            release_state["verified"] += 1
            release_state["released"] = False
        return jac(x, f)

    monkeypatch.setattr(LaggedJacobianController, "release_stale_model", tracked_release)
    result = _run_trf(
        fun,
        checked_jac,
        x0,
        "banded",
        {"n_blocks": N_BLOCKS, "jvp": jvp, "lag": 1},
    )
    assert result.njev > 1
    assert release_state["verified"] == release_state["jac_calls"] - 1


def test_failed_dense_rebuild_cannot_reuse_released_stale_state(monkeypatch):
    """A rebuild exception propagates after stale state has already been released."""
    fun, jac, jvp, x0 = _block_tridiagonal_lsq_problem(7)
    f0 = fun(x0)
    J0 = jac(x0, f0)
    released = {"value": False}
    original_release = LaggedJacobianController.release_stale_model

    def tracked_release(controller):
        original_release(controller)
        released["value"] = True

    def failed_rebuild(x, f):
        assert released["value"]
        raise RuntimeError("synthetic dense rebuild failure")

    monkeypatch.setattr(LaggedJacobianController, "release_stale_model", tracked_release)
    with pytest.raises(RuntimeError, match="synthetic dense rebuild failure"):
        trf_no_bounds(
            fun,
            failed_rebuild,
            x0,
            f0,
            J0,
            ftol=1e-15,
            xtol=1e-15,
            gtol=1e-8,
            max_nfev=100,
            max_niter=20,
            x_scale=1.0,
            loss_function=None,
            tr_solver="banded",
            tr_options={"n_blocks": N_BLOCKS, "jvp": jvp, "lag": 1},
            verbose=0,
            cb=None,
        )


def test_model_quality_report_contains_age_subspace_and_radius(capsys):
    """Verbose diagnostics expose all quantities needed to classify a poor model."""
    controller = LaggedJacobianController(
        BandedTrOptions(n_blocks=1, lag=2, jvp=lambda x, directions: directions)
    )
    controller._jacobian_age = 2
    controller._reduced_model = SimpleNamespace(k=3)
    controller.report_model_quality(1.0, 0.75, 0.75, 0.4, 0.5)
    output = capsys.readouterr().out
    for token in ("age 2", "subspace k=3", "pred", "actual", "ratio", "step", "Delta"):
        assert token in output


def test_lagged_controller_reports_an_unusable_fake_jvp():
    def nonfinite_jvp(x, directions):
        return JvpResult.from_backend_failure(
            (3, directions.shape[1]), backend_status=np.array([0, 2])
        )

    controller = LaggedJacobianController(BandedTrOptions(n_blocks=1, lag=1, jvp=nonfinite_jvp))
    assert controller.initialize_fresh_jacobian(np.eye(3, 2)).available
    controller.record_step_accepted(np.array([0.1, -0.2]), ratio=0.8)
    decision = controller.request_model(
        np.zeros(2),
        np.array([1.0, -0.5]),
        Delta=0.4,
        alpha=0.0,
        scale=1.0,
        residual=np.ones(3),
    )
    assert decision.action is LaggedAction.REBUILD_JACOBIAN
    assert decision.jvp_result.status is JvpStatus.BACKEND_FAILURE
    assert decision.jvp_result.backend_status == (0, 2)
    assert controller.diagnostics.jvp_evaluations == 1


@pytest.mark.parametrize("seed", [3, 17])
def test_lagged_reaches_exact_minimizer_with_fewer_jacobians(seed):
    fun, jac, jvp, x0 = _block_tridiagonal_lsq_problem(seed)

    res_exact = _run_trf(fun, jac, x0, "exact", {})
    res_lag = _run_trf(fun, jac, x0, "banded", {"n_blocks": N_BLOCKS, "jvp": jvp, "lag": 3})

    assert res_exact.status == 1 and res_lag.status == 1  # both converged on the gradient
    rel = norm(res_lag.x - res_exact.x) / norm(res_exact.x)
    assert rel < 1e-6, f"lagged minimizer deviates from exact by rel {rel:.3e}"
    assert res_lag.njvp > 0
    assert res_lag.njev < res_exact.njev, (
        f"lagged mode did not save dense Jacobian builds "
        f"(njev {res_lag.njev} vs exact {res_exact.njev})"
    )


def test_lag_without_jvp_warns_and_falls_back(capsys):
    fun, jac, jvp, x0 = _block_tridiagonal_lsq_problem(3)

    res_exact = _run_trf(fun, jac, x0, "exact", {})
    res_lag = _run_trf(fun, jac, x0, "banded", {"n_blocks": N_BLOCKS, "lag": 3})

    assert "disabled" in capsys.readouterr().out
    assert res_lag.njvp == 0
    rel = norm(res_lag.x - res_exact.x) / norm(res_exact.x)
    assert rel < 1e-10  # plain banded: identical trajectory to the SVD path


def test_trf_lagged_on_qa_7vol():
    """Four trf iterations of the real force minimization, lagged-banded vs exact SVD."""
    from spectre import (
        force_real,
        force_real_jac,
        force_real_jvp,
        get_xinit_specwrap,
        remove_file,
    )
    from spectre.least_squares.least_squares import least_squares as lstsq

    fname = f"{curr_path}qa_7vol_test.toml"

    def run(tr_solver, lag):
        test = spectre_with_lconstraint(fname, 3)
        ngdof = int(test.allglobal_mod.ngdof)
        n_blocks = int(test.allglobal_mod.mvol) - 1

        def obj_fn(xdof):
            return force_real(np.concatenate(([0.0], xdof)), test).flatten()

        def jac_fn(xdof):
            fjac = force_real_jac(np.concatenate(([0.0], xdof)), test)
            return fjac.reshape(-1, ngdof + 1)[:, 1:]

        def jvp_fn(xdof, dirs):
            dirs_full = np.vstack([np.zeros((1, dirs.shape[1])), dirs])
            jvp = force_real_jvp(np.concatenate(([0.0], xdof)), dirs_full, test)
            return jvp.reshape(-1, dirs.shape[1])

        tr_options = {}
        if tr_solver == "banded":
            tr_options = {"n_blocks": n_blocks}
            if lag:
                tr_options.update({"jvp": jvp_fn, "lag": lag})

        x0 = np.asarray(get_xinit_specwrap(test))[1:]
        result = lstsq(
            obj_fn,
            x0,
            jac=jac_fn,
            method="trf",
            tr_solver=tr_solver,
            tr_options=tr_options,
            max_niter=4,
            gtol=1e-15,
            ftol=1e-15,
            xtol=1e-22,
            verbose=0,
        )
        return result

    res_exact = run("exact", 0)
    res_lag = run("banded", 2)

    # Different (fresh vs reduced-model) steps, so trajectories differ; the lagged run
    # must still make comparable progress with strictly fewer dense Jacobian builds.
    assert res_lag.njev < res_exact.njev
    assert res_lag.njvp > 0
    assert res_lag.cost < 10.0 * res_exact.cost, (
        f"lagged cost {res_lag.cost:.3e} fell far behind exact {res_exact.cost:.3e}"
    )

    remove_file(fname + ".end")


def test_optimizer_default_lagged_matches_replicated_control_flow_under_mpi():
    """The rank-0/worker protocol reproduces the replicated lagged control flow under MPI."""
    from mpi4py import MPI

    import spectre.force_minimizer as force_minimizer
    from spectre import (
        force_real,
        force_real_jac,
        force_real_jvp,
        get_xinit_specwrap,
        remove_file,
    )
    from spectre.force_minimizer import (
        Force_Minimizer,
        _BeltramiFactorCacheScope,
        _send_int_array,
        _WorkerCommand,
        optimizer_default,
    )
    from spectre.input_dict import input_dict
    from spectre.least_squares.least_squares import least_squares as lstsq

    fname = f"{curr_path}qa_7vol_test.toml"
    field_obj = spectre_with_lconstraint(fname, 3)
    comm = field_obj.comm
    rank = comm.Get_rank()
    force_minimizer.MPI_INT = MPI.INT
    force_minimizer.MPI_DOUBLE = MPI.DOUBLE
    ngdof = int(field_obj.allglobal_mod.ngdof)
    n_blocks = int(field_obj.allglobal_mod.mvol) - 1
    x0 = np.asarray(get_xinit_specwrap(field_obj))[1:]

    def to_xmn(xdof):
        return np.concatenate(([0.0], xdof))

    def objective(xdof):
        return force_real(to_xmn(xdof), field_obj).flatten()

    def jacobian(xdof):
        return force_real_jac(to_xmn(xdof), field_obj).reshape(-1, ngdof + 1)[:, 1:]

    def jvp(xdof, dirs):
        dirs_mn = np.vstack([np.zeros((1, dirs.shape[1])), dirs])
        return force_real_jvp(to_xmn(xdof), dirs_mn, field_obj).reshape(-1, dirs.shape[1])

    cache = field_obj.lib.beltrami_factor_cache_mod
    cache.factor_cache_clear()
    field_obj.allglobal_mod.lbeltrami_factor_cache = False
    with _BeltramiFactorCacheScope(field_obj):
        reference = lstsq(
            objective,
            x0,
            jac=jacobian,
            method="trf",
            tr_solver="banded",
            tr_options={"n_blocks": n_blocks, "jvp": jvp, "lag": 2},
            max_niter=4,
            gtol=1e-15,
            ftol=1e-15,
            xtol=1e-15,
            verbose=0,
        )

    params = input_dict(
        dict(
            gtol=1e-15,
            xtol=1e-15,
            ftol=1e-15,
            max_nfev=50,
            max_niter=4,
            method="trf",
            rcond_lstsq=None,
            tr_solver="banded",
            jvp_lag=2,
            objective="freal",
            inface_repr="fourier",
            mpol_rz=None,
            ntor_rz=None,
            nt_dbb=None,
            nz_dbb=None,
            filter_stellsym=False,
        )
    )

    minimizer = Force_Minimizer.__new__(Force_Minimizer)
    minimizer.rank = rank
    minimizer.comm = comm
    minimizer.field_obj = field_obj
    minimizer.freeb_mode = 0
    minimizer.min_params = params
    minimizer.optimizer = optimizer_default
    minimizer.callback = None
    minimizer.iteration_start = 0
    minimizer.x_init = x0.copy()
    minimizer.x_curr = x0.copy()
    minimizer.xdof_to_xmn = lambda xdof, test: to_xmn(xdof)
    minimizer.Dxmn_Dxdof_at_xdof = lambda xdof, test: np.identity(ngdof + 1)[:, 1:]

    def protocol_objective(xdof, owner):
        if rank == 0:
            _send_int_array(xdof, _WorkerCommand.FORCE, comm)
        return objective(xdof)

    def protocol_jacobian(xdof, owner):
        if rank == 0:
            _send_int_array(xdof, _WorkerCommand.JACOBIAN, comm)
        return jacobian(xdof)

    minimizer.obj_func = protocol_objective
    minimizer.jac_func = protocol_jacobian
    minimizer.minimize()

    if rank == 0:
        assert minimizer.result.njvp > 0
        np.testing.assert_allclose(minimizer.result.x, reference.x, rtol=2e-12, atol=2e-12)
        assert minimizer.result.cost == pytest.approx(reference.cost, rel=1e-10, abs=1e-10)

    assert not bool(field_obj.allglobal_mod.lbeltrami_factor_cache)
    assert cache.factor_cache_diagnostics()[:2] == (0, 0)
    if rank == 0:
        remove_file(fname + ".end")
    comm.Barrier()
