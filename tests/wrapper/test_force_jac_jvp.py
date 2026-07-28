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
    force_real,
    force_real_jac,
    force_real_jvp,
    get_xinit_specwrap,
    remove_file,
)
from spectre.force_minimizer import _clear_beltrami_factor_cache
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


def test_force_jvp_context_cleanup_after_unsupported_early_return():
    """The scope finalizer deactivates the context when the JVP driver returns early."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, 2)
    xinit = np.asarray(get_xinit_specwrap(test))
    dirs = np.zeros((int(test.allglobal_mod.ngdof) + 1, 1), order="F")

    jvp, status = force_real_jvp(xinit, dirs, test, filter_stellsym=False, return_status=True)

    np.testing.assert_array_equal(jvp, 0.0)
    np.testing.assert_array_equal(status, -2)
    assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
    assert test.lib.force_jvp_context_mod.jvp_context_identity() == (0, 0, 0)


@pytest.mark.parametrize("lcons", [0, 1, 3])
@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_force_real_jvp_matches_dense_jacobian(fname, lcons):
    test = spectre_with_lconstraint(fname, lcons)
    assert int(test.input_list_mod.lconstraint) == lcons
    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J, jac_status = force_real_jac(
        xinit, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(jac_status, 0)
    J = J.reshape(-1, ngdof + 1)

    # unit directions (single dense columns) and a random direction across all interfaces
    rng = np.random.default_rng(11)
    dirs = np.zeros((ngdof + 1, 3), order="F")
    dirs[1, 0] = 1.0
    dirs[min(5, ngdof), 1] = 1.0
    dirs[1:, 2] = rng.standard_normal(ngdof)

    jvp, status = force_real_jvp(
        xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False, return_status=True
    )
    np.testing.assert_array_equal(status, 0)
    assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
    jvp = jvp.reshape(-1, dirs.shape[1])

    ref = J @ dirs
    for k in range(dirs.shape[1]):
        scale = max(np.max(np.abs(ref[:, k])), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
        assert rel < 1e-10, (
            f"{os.path.basename(fname)} lcons={lcons}: JVP direction {k} deviates from the "
            f"dense Jacobian by rel {rel:.3e}"
        )

    remove_file(fname + ".end")


@pytest.mark.parametrize("lcons", [1, 3])
def test_force_real_jvp_with_factor_cache(lcons):
    """With Lbeltrami_factor_cache the JVP reuses the factorizations deposited by the
    preceding force evaluation instead of rebuilding them; the products must be identical
    to rounding, including on a second batch (which takes them back from the put-back)."""
    fname = f"{curr_path}qa_7vol_test.toml"
    test = spectre_with_lconstraint(fname, lcons)
    assert int(test.input_list_mod.lconstraint) == lcons
    test.allglobal_mod.lbeltrami_factor_cache = True
    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J = force_real_jac(xinit, test, nt=nt, nz=nz, filter_stellsym=False).reshape(-1, ngdof + 1)

    rng = np.random.default_rng(7)
    dirs = np.zeros((ngdof + 1, 2), order="F")
    dirs[1:, 0] = rng.standard_normal(ngdof)
    dirs[1:, 1] = rng.standard_normal(ngdof)
    ref = J @ dirs

    for label in ("cached", "put-back"):
        jvp = force_real_jvp(xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False)
        assert not test.lib.force_jvp_context_mod.jvp_context_is_active()
        jvp = jvp.reshape(-1, dirs.shape[1])
        for k in range(dirs.shape[1]):
            scale = max(np.max(np.abs(ref[:, k])), 1e-300)
            rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
            assert rel < 1e-10, (
                f"lcons={lcons} ({label}): JVP direction {k} deviates from the dense "
                f"Jacobian by rel {rel:.3e}"
            )

    test.allglobal_mod.lbeltrami_factor_cache = False
    _clear_beltrami_factor_cache(
        test
    )  # the cache is process-global; do not leak it to the next test
    remove_file(fname + ".end")
