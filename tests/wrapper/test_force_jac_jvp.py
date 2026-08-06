# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Golden test of the force-Jacobian JVP against the dense Jacobian build.

``force_real_jvp`` computes J @ d with one general-direction linearised solve per volume
(Fortran ``solve_field_deriv_dir``); the dense build computes one column per interface
harmonic. Both are exact linearisations of the same discretisation, so they must agree to
rounding for unit directions (single dense columns) and for random directions spanning
every interface, R and Z, cosine and sine, including the constraint (dmupfdx) coupling.
"""

import os.path

import numpy as np
import pytest
import spectre4py_wrap
from helpers import spectre_with_lconstraint

from spectre import (
    SPECTRE,
    force_real,
    force_real_jac,
    force_real_jvp,
    get_xinit_specwrap,
    remove_file,
)
from spectre.file_io import read_input_parameters_from_toml
from spectre.force_targets import _process_input_nt_nz

curr_path = os.path.dirname(__file__) + "/"


def test_force_jvp_context_activation_reset_and_status_isolation():
    """The private context owns activation identity and persistent batch diagnostics."""
    context = spectre4py_wrap.force_jvp_context_mod
    context.jvp_context_initialize(3)

    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)
    np.testing.assert_array_equal([context.jvp_context_status(vvol) for vvol in range(1, 4)], 0)

    scope = context.jvp_context_open()
    context.jvp_context_reserve(5, 12)
    context.jvp_context_activate(5, 8, 2)
    assert context.jvp_context_is_active()
    assert context.jvp_context_identity() == (2, 5, 8)

    context.jvp_context_raise_status(2, 4)
    context.jvp_context_raise_status(2, 1)
    assert context.jvp_context_status(2) == 4

    context.jvp_context_reset()
    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)
    assert context.jvp_context_status(2) == 4

    context.jvp_context_activate(5, 12, 3)
    assert context.jvp_context_is_active()
    context.finalize_jvp_scope(scope)
    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)
    context.jvp_context_reset_status()
    np.testing.assert_array_equal([context.jvp_context_status(vvol) for vvol in range(1, 4)], 0)


def test_force_jvp_context_cleanup_after_scope_exit():
    """The scope finalizer deactivates and releases an active JVP context."""
    context = spectre4py_wrap.force_jvp_context_mod
    context.jvp_context_initialize(3)

    scope = context.jvp_context_open()
    context.jvp_context_reserve(4, 8)
    context.jvp_context_activate(4, 8, 1)
    assert context.jvp_context_is_active()
    context.finalize_jvp_scope(scope)

    assert not context.jvp_context_is_active()
    assert context.jvp_context_identity() == (0, 0, 0)


@pytest.mark.parametrize("lcons", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "fname",
    [
        f"{curr_path}qa_7vol_test.toml",
        f"{curr_path}cyl_manyvol_test.toml",
        f"{curr_path}../compare/G2V32L1Fi/input.toml",
    ],
)
def test_force_real_jvp_matches_dense_jacobian(fname, lcons):
    test = spectre_with_lconstraint(fname, lcons)
    assert int(test.input_list_mod.lconstraint) == lcons
    mvol = int(test.allglobal_mod.mvol)
    if test.comm.Get_size() > mvol:
        pytest.skip(f"requires at most one MPI rank per volume ({test.comm.Get_size()=} > {mvol=})")

    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J, jac_status = force_real_jac(
        xinit, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(jac_status, 0)
    J = J.reshape(-1, ngdof)
    rank = test.comm.Get_rank()
    field = test.field_mod
    assert int(field.last_deriv_response_count_written) == int(
        field.last_deriv_response_count_expected
    )
    assert bool(field.last_dense_jacobian_assembled) is (rank == 0)
    expected_slices = (
        mvol
        if rank == 0
        else sum((volume - 1) % test.comm.Get_size() == rank for volume in range(1, mvol + 1))
    )
    assert int(field.last_dense_jacobian_slices_touched) == expected_slices
    if not bool(test.allglobal_mod.localconstraint) and int(test.input_list_mod.lconstraint) != 2:
        assert 0 < int(field.last_jacobian_max_chunk_columns) <= 512
    assert np.all(np.asarray(field.last_jacobian_phase_seconds) >= 0.0)

    # unit directions (single dense columns) and a random direction across all interfaces
    rng = np.random.default_rng(11)
    dirs = np.zeros((ngdof, 3), order="F")
    dirs[0, 0] = 1.0
    dirs[min(4, ngdof - 1), 1] = 1.0
    dirs[:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
    jvp = jvp.reshape(-1, dirs.shape[1])

    for rank_jvp, rank_status in zip(
        test.comm.allgather(jvp), test.comm.allgather(status), strict=True
    ):
        np.testing.assert_array_equal(rank_status, status)
        np.testing.assert_array_equal(rank_jvp, jvp)

    ref = np.ascontiguousarray(J @ dirs) if rank == 0 else np.empty(jvp.shape)
    test.comm.Bcast(ref, root=0)
    for k in range(dirs.shape[1]):
        scale = max(np.max(np.abs(ref[:, k])), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
        assert rel < 1e-10, (
            f"{os.path.basename(fname)} lcons={lcons}: JVP direction {k} deviates from the "
            f"dense Jacobian by rel {rel:.3e}"
        )

    if test.comm.Get_rank() == 0:
        remove_file(fname + ".end")
    test.comm.Barrier()


def test_iterative_dense_jacobian_is_thread_count_independent():
    """Fixed iterative stripes preserve results, statuses, and iteration counts."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 3)
    test.input_list_mod.lbeltramisolver = 2
    test.input_list_mod.nadofgmres_sing = 1
    test.input_list_mod.nitergmres = 3000
    test.input_list_mod.ngmresrestart = 60
    test.input_list_mod.epsgmres = 1e-12
    test.input_list_mod.epsgmres_jac = 1e-10
    xinit = np.asarray(get_xinit_specwrap(test))
    support = test.lib.matrix_free_test_support_mod
    original_threads = int(support.get_test_omp_max_threads())

    try:
        support.set_test_omp_threads(1)
        force_real(xinit, test, filter_stellsym=False)
        results = []
        # Never raise the team size above the value used by preset_mod to allocate the
        # per-thread Ate/Aze slots. Multi-thread topology jobs set OMP_NUM_THREADS before import.
        for nthreads in (1, original_threads):
            support.set_test_omp_threads(nthreads)
            jacobian, status = force_real_jac(
                xinit, test, filter_stellsym=False, return_status=True
            )
            results.append(
                (
                    np.array(jacobian, copy=True),
                    np.array(status, copy=True),
                    int(test.field_mod.last_jacobian_total_iterations),
                )
            )
    finally:
        support.set_test_omp_threads(original_threads)

    jacobian_1, status_1, iterations_1 = results[0]
    jacobian_n, status_n, iterations_n = results[1]
    np.testing.assert_array_equal(status_1, status_n)
    np.testing.assert_array_equal(status_1, 0)
    np.testing.assert_array_equal(jacobian_1, jacobian_n)
    assert iterations_1 > 0
    assert iterations_1 == iterations_n

    if test.comm.Get_rank() == 0:
        remove_file(fname + ".end")
    test.comm.Barrier()


def test_nonsymmetric_truncated_jacobian_matches_jvp():
    """Rank-local counting covers asymmetric modes and a truncated perturbation spectrum."""
    fname = f"{curr_path}qa_7vol_test.toml"
    params = read_input_parameters_from_toml(fname)
    params.physics.lconstraint = 3
    params.physics.enforce_stell_sym = False
    test = SPECTRE.from_input_params(params, name="qa_nonsym_truncated", verbose=False)
    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, filter_stellsym=False)
    jacobian, jac_status = force_real_jac(
        xinit,
        test,
        mpol_rz=1,
        ntor_rz=0,
        filter_stellsym=False,
        return_status=True,
    )
    np.testing.assert_array_equal(jac_status, 0)
    assert int(test.field_mod.last_deriv_response_count_written) == int(
        test.field_mod.last_deriv_response_count_expected
    )

    rank = test.comm.Get_rank()
    jacobian = jacobian.reshape(-1, ngdof)
    if rank == 0:
        active = np.flatnonzero(np.linalg.norm(jacobian, axis=0) > 1e-13)
        assert 0 < active.size < ngdof
    else:
        active = None
    active = test.comm.bcast(active, root=0)

    dirs = np.zeros((ngdof, 2), order="F")
    dirs[active[0], 0] = 1.0
    dirs[active[: min(4, active.size)], 1] = np.random.default_rng(31).standard_normal(
        min(4, active.size)
    )
    jvp, jvp_status = force_real_jvp(xinit, dirs, test, filter_stellsym=False, return_status=True)
    np.testing.assert_array_equal(jvp_status, 0)
    jvp = jvp.reshape(-1, dirs.shape[1])

    reference = np.ascontiguousarray(jacobian @ dirs) if rank == 0 else np.empty(jvp.shape)
    test.comm.Bcast(reference, root=0)
    np.testing.assert_allclose(jvp, reference, rtol=1e-10, atol=1e-11)

    if rank == 0:
        remove_file(fname + ".end")
    test.comm.Barrier()


def test_force_real_jvp_collective_reduces_degraded_status():
    """A failure raised by a volume owner is visible on every rank and leaves no active context."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 1)
    test.input_list_mod.lbeltramisolver = 2
    test.input_list_mod.ngmresrestart = 1
    test.input_list_mod.nitergmres = 1
    test.input_list_mod.epsgmres = 1e-14
    test.input_list_mod.epsgmres_jac = 1e-30
    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, filter_stellsym=False)
    dirs = np.zeros((ngdof, 1), order="F")
    dirs[:, 0] = np.random.default_rng(19).standard_normal(ngdof)
    jvp, status = force_real_jvp(xinit, dirs, test, filter_stellsym=False, return_status=True)

    assert np.any(status > 0)
    assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
    for rank_jvp, rank_status in zip(
        test.comm.allgather(jvp), test.comm.allgather(status), strict=True
    ):
        np.testing.assert_array_equal(rank_status, status)
        np.testing.assert_array_equal(rank_jvp, jvp)

    if test.comm.Get_rank() == 0:
        remove_file(fname + ".end")
    test.comm.Barrier()


@pytest.mark.parametrize("lbeltramisolver", [0, 2])
def test_force_real_jvp_lcons2_findiff_conserves_helicity(lbeltramisolver):
    """Finite-difference validation of the helicity-constrained (lconstraint=2) JVP.

    At a perturbed geometry the forward solve re-iterates mu until each volume matches its
    helicity target again, so central differences of ``force_real`` differentiate along the
    constrained manifold. The JVP's Schur-complement response (da = y + dmu*z with
    <dMD.a0, da> = 0) must reproduce them, and every perturbed state must conserve
    helicity. ``lbeltramisolver=2`` repeats the check with the matrix-free (GMRES) backend
    behind both the forward solves and the JVP solves.
    """
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 2)
    test.input_list_mod.lbeltramisolver = lbeltramisolver
    if lbeltramisolver:
        test.input_list_mod.nitergmres = 3000
        test.input_list_mod.ngmresrestart = 60
        test.input_list_mod.epsgmres = 1e-13
        test.input_list_mod.epsgmres_jac = 1e-13
    nt, nz = _process_input_nt_nz(None, None, test)
    nvol = int(test.input_list_mod.nvol)
    k_target = np.array(test.input_list_mod.helicity[:nvol], copy=True)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    def collective_labintegral():
        values = np.array(test.allglobal_mod.labintegral[:nvol], copy=True)
        if test.comm.Get_size() > 1:
            from mpi4py import MPI

            test.comm.Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
        return values

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    np.testing.assert_allclose(collective_labintegral(), k_target, rtol=0, atol=1e-9)

    # one harmonic per interface plus a random direction across all interfaces
    rng = np.random.default_rng(3)
    dirs = np.zeros((ngdof, 3), order="F")
    dirs[0, 0] = 1.0
    dirs[min(36, ngdof - 1), 1] = 1.0
    dirs[:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    jvp = jvp.reshape(-1, dirs.shape[1])

    eps = 1e-8
    mu_response = np.zeros((dirs.shape[1], nvol))
    for k in range(dirs.shape[1]):
        f_pm = []
        mu_pm = []
        for sign in (+1.0, -1.0):
            f = force_real(
                xinit + sign * eps * dirs[:, k], test, nt=nt, nz=nz, filter_stellsym=False
            )
            f_pm.append(np.asarray(f).flatten())
            mu_pm.append(np.array(test.input_list_mod.mu[:nvol], copy=True))
            # every constrained re-solve conserves the helicity targets
            np.testing.assert_allclose(collective_labintegral(), k_target, rtol=0, atol=1e-9)
        fd = (f_pm[0] - f_pm[1]) / (2 * eps)
        mu_response[k] = (mu_pm[0] - mu_pm[1]) / (2 * eps)
        scale = max(np.max(np.abs(fd)), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - fd)) / scale
        assert rel < 5e-6, (
            f"lbeltramisolver={lbeltramisolver}: JVP direction {k} deviates from finite "
            f"differences by rel {rel:.3e}"
        )

    # the constraint is active: mu responds to the geometry perturbations, so the JVP would
    # be wrong without the helicity Schur-complement term
    assert np.max(np.abs(mu_response)) > 1e-3

    remove_file(fname + ".end")


def test_force_real_jvp_lcons2_freeb_conserves_plasma_helicity():
    """Free-boundary lconstraint=2: plasma volumes conserve their helicity while the vacuum
    volume varies its fluxes to match (curtor, curpol) and its helicity stays free.

    The JVP must match the dense Jacobian (both carry the vacuum flux response through
    dmupfdx / dmupf_dir) and central finite differences of ``force_real``. Every perturbed
    re-solve must hold the plasma helicity targets and the current constraints, while the
    vacuum helicity responds freely to the geometry.
    """
    fname = f"{curr_path}test_freeb_w7x.toml"
    test = spectre_with_lconstraint(fname, 2)
    nt, nz = _process_input_nt_nz(None, None, test)
    nvol = int(test.input_list_mod.nvol)
    mvol = int(test.allglobal_mod.mvol)
    k_target = np.array(test.input_list_mod.helicity[:nvol], copy=True)
    curtor = float(test.input_list_mod.curtor)
    curpol = float(test.input_list_mod.curpol)

    def check_constrained_state():
        # plasma helicity at target; vacuum currents at target (mupftol = 1e-16 here)
        labintegral = np.array(test.allglobal_mod.labintegral, copy=True)
        if test.comm.Get_size() > 1:
            from mpi4py import MPI

            test.comm.Allreduce(MPI.IN_PLACE, labintegral, op=MPI.SUM)
        np.testing.assert_allclose(labintegral[:nvol], k_target, rtol=0, atol=1e-12)

        vacuum_owner = (mvol - 1) % test.comm.Get_size()
        ditgp = np.array(test.allglobal_mod.ditgpdxtp)
        vacuum_currents = test.comm.bcast(
            ditgp[:, 1, mvol - 1] if test.comm.Get_rank() == vacuum_owner else None,
            root=vacuum_owner,
        )
        np.testing.assert_allclose(vacuum_currents[0], curtor, rtol=0, atol=1e-10)
        np.testing.assert_allclose(vacuum_currents[1], curpol, rtol=0, atol=1e-8)
        return labintegral[mvol - 1]

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    check_constrained_state()

    J, jac_status = force_real_jac(
        xinit, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(jac_status, 0)
    J = J.reshape(-1, ngdof)

    # an inner-interface harmonic, a plasma-boundary harmonic (exercising the vacuum flux
    # response), and a random direction across both interfaces
    rng = np.random.default_rng(5)
    dirs = np.zeros((ngdof, 3), order="F")
    dirs[0, 0] = 1.0
    dirs[ngdof - 1, 1] = 1.0
    dirs[:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    jvp = jvp.reshape(-1, dirs.shape[1])

    rank = test.comm.Get_rank()
    ref = np.ascontiguousarray(J @ dirs) if rank == 0 else np.empty(jvp.shape)
    test.comm.Bcast(ref, root=0)
    for k in range(dirs.shape[1]):
        scale = max(np.max(np.abs(ref[:, k])), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
        assert rel < 1e-10, f"freeb lcons2: JVP direction {k} deviates from dense by rel {rel:.3e}"

    # finite differences along the plasma-boundary harmonic and the random direction
    eps = 1e-7
    for k in (1, 2):
        f_pm = []
        mu_pm = []
        k_vac_pm = []
        for sign in (+1.0, -1.0):
            f = force_real(
                xinit + sign * eps * dirs[:, k], test, nt=nt, nz=nz, filter_stellsym=False
            )
            f_pm.append(np.asarray(f).flatten())
            mu_values = np.array(test.input_list_mod.mu[:nvol], copy=True)
            if test.comm.Get_size() > 1:
                from mpi4py import MPI

                test.comm.Allreduce(MPI.IN_PLACE, mu_values, op=MPI.SUM)
            mu_pm.append(mu_values)
            k_vac_pm.append(check_constrained_state())
        fd = (f_pm[0] - f_pm[1]) / (2 * eps)
        scale = max(np.max(np.abs(fd)), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - fd)) / scale
        assert rel < 5e-6, (
            f"freeb lcons2: JVP direction {k} deviates from finite differences by rel {rel:.3e}"
        )
        # the plasma helicity constraint is active and the vacuum helicity responds freely
        assert np.max(np.abs(mu_pm[0] - mu_pm[1])) / (2 * eps) > 1e-3
        assert abs(k_vac_pm[0] - k_vac_pm[1]) / (2 * eps) > 1e-3

    if rank == 0:
        remove_file(fname + ".end")
    test.comm.Barrier()


@pytest.mark.parametrize("lcons", [1, 2, 3])
def test_force_real_jvp_with_factor_cache(lcons):
    """With Lbeltrami_factor_cache the JVP reuses the factorizations deposited by the
    preceding force evaluation instead of rebuilding them; the products must be identical
    to rounding, including on a second batch (which takes them back from the put-back)."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, lcons)
    assert int(test.input_list_mod.lconstraint) == lcons
    cache = test.lib.beltrami_factor_cache_mod
    cache.factor_cache_clear()
    test.allglobal_mod.lbeltrami_factor_cache = True
    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J = force_real_jac(xinit, test, nt=nt, nz=nz, filter_stellsym=False).reshape(-1, ngdof)

    rng = np.random.default_rng(7)
    dirs = np.zeros((ngdof, 2), order="F")
    dirs[:, 0] = rng.standard_normal(ngdof)
    dirs[:, 1] = rng.standard_normal(ngdof)
    rank = test.comm.Get_rank()
    ref = np.ascontiguousarray(J @ dirs) if rank == 0 else np.empty((J.shape[0], dirs.shape[1]))
    test.comm.Bcast(ref, root=0)

    hits = cache.factor_cache_diagnostics()[2]
    for label in ("cached", "put-back"):
        jvp = force_real_jvp(xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False)
        assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
        jvp = jvp.reshape(-1, dirs.shape[1])
        diagnostics = cache.factor_cache_diagnostics()
        assert diagnostics[0] == int(test.allglobal_mod.mvol)
        assert diagnostics[1] > 0
        assert diagnostics[2] > hits
        hits = diagnostics[2]
        for k in range(dirs.shape[1]):
            scale = max(np.max(np.abs(ref[:, k])), 1e-300)
            rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
            assert rel < 1e-10, (
                f"lcons={lcons} ({label}): JVP direction {k} deviates from the dense "
                f"Jacobian by rel {rel:.3e}"
            )

    test.allglobal_mod.lbeltrami_factor_cache = False
    # the cache is process-global; do not leak it to the next test
    cache.factor_cache_clear()
    assert cache.factor_cache_diagnostics()[:2] == (0, 0)
    if rank == 0:
        remove_file(fname + ".end")
    test.comm.Barrier()
