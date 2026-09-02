# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for the ``SPECTRE.xin`` property/setter and ``unpacked_interfaces``."""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def sp():
    """Build a small fixed-boundary (qa_7vol) SPECTRE instance for exercising xin."""
    return SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)


def test_xin_returns_fortran_contiguous_array_of_correct_size(sp):
    """Reading xin gives an F-contiguous float64 array of length ngdof."""
    x = sp.xin
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    assert x.shape == (sp.allglobal_mod.ngdof,)
    assert x.flags["F_CONTIGUOUS"]


def test_xin_setter_round_trips(sp):
    """Reading xin, writing it back unchanged, and reading again matches exactly."""
    x0 = sp.xin.copy()
    sp.xin = x0
    x1 = sp.xin
    np.testing.assert_array_equal(x0, x1)


def test_xin_setter_actually_updates_fortran_arrays(sp):
    """Setting xin propagates a perturbed DOF into the Fortran iRbc/iZbs arrays."""
    x0 = sp.xin.copy()
    irbc_before = sp.allglobal_mod.irbc.copy()

    x_perturbed = x0.copy()
    # Every entry of xin is a real DOF; perturb the first one.
    x_perturbed[0] += 1.0e-6
    sp.xin = x_perturbed

    irbc_after = sp.allglobal_mod.irbc.copy()
    # The perturbed degree of freedom changes at least one irbc entry.
    assert not np.array_equal(irbc_before, irbc_after), (
        "Setting sp.xin should propagate into iRbc/iZbs/iRbs/iZbc."
    )

    # Restore for any downstream tests in the same module.
    sp.xin = x0


def test_unpacked_interfaces_returns_fortran_array_refs(sp):
    """unpacked_interfaces property returns the (irbc, izbs, irbs, izbc) tuple."""
    irbc, izbs, irbs, izbc = sp.unpacked_interfaces
    # Identity (`is`) check: must be the exact Fortran-backed arrays, not copies.
    assert irbc is sp.allglobal_mod.irbc
    assert izbs is sp.allglobal_mod.izbs
    assert irbs is sp.allglobal_mod.irbs
    assert izbc is sp.allglobal_mod.izbc


def test_unpack_interfaces_is_observational(sp):
    """Unpacking an alternate state returns copies without changing live interfaces."""
    original = sp.xin.copy()
    alternate = original.copy()
    alternate[1] += 1.0e-6

    unpacked = sp.unpack_interfaces(alternate)

    np.testing.assert_array_equal(sp.xin, original)
    assert all(array.flags["F_CONTIGUOUS"] for array in unpacked)
    assert not np.array_equal(unpacked[0], sp.allglobal_mod.irbc)


def test_unpack_interfaces_current_state_matches_live_arrays(sp):
    """Unpacking the current state preserves fixed and inactive coefficients."""
    unpacked = sp.unpack_interfaces()

    for copied, live in zip(unpacked, sp.unpacked_interfaces, strict=True):
        np.testing.assert_array_equal(copied, live)


def test_unpack_interfaces_free_function_remains_compatible(sp):
    """The legacy free function continues to update the live interfaces."""
    from spectre.utils import unpack_interfaces

    original = sp.xin.copy()
    alternate = original.copy()
    alternate[1] += 1.0e-6
    unpack_interfaces(alternate, sp)
    np.testing.assert_array_equal(sp.xin, alternate)
    sp.xin = original


def test_xin_setter_after_off_grid_force_evaluation_keeps_working_grid(sp):
    """Force evaluations on another grid must hand the working grid back before the setter.

    ``force_real`` and ``force_real_jac`` evaluate on the ``nt``/``nz`` grid they are given
    and resize the Fortran grid arrays and the FFT plan to it. The ``xin`` setter recomputes
    the axis on the working ``Nt x Nz`` grid, so a stale coarser grid overran those arrays
    (heap corruption at the next malloc) and a stale finer grid gave a wrong axis.
    """
    from spectre.force_targets import force_real, force_real_jac

    nt, nz = int(sp.allglobal_mod.nt), int(sp.allglobal_mod.nz)
    x = np.asarray(sp.xin).copy()
    sp.xin = x
    axis_ref = np.asarray(sp.allglobal_mod.irbc[:, 0]).copy()

    for evaluate, grid in ((force_real, (nt // 3, nz // 2)), (force_real_jac, (2 * nt, 2 * nz))):
        evaluate(x, sp, nt=grid[0], nz=grid[1])
        assert (int(sp.field_mod.nt_alloc), int(sp.field_mod.nz_alloc)) == (nt, nz)
        sp.xin = x
        np.testing.assert_allclose(sp.allglobal_mod.irbc[:, 0], axis_ref, rtol=0, atol=1e-12)


def test_freal_fsa_after_off_grid_force_evaluation(sp):
    """The callback's flux-surface average sizes from the force array, not the module grid.

    ``force_real`` hands the working grid back on exit, so ``get_fsa_wrapper`` has to take
    the force grid from ``f_real_shpd`` and ``compute_coordinates`` has to put the module grid
    on it for the Jacobian (the inverse FFT fills ``nt_plan * nz_plan`` values of its output).
    On a grid that subsamples the working one the Jacobian is the same function at the same
    angles, and the working grid is back in place afterwards.
    """
    from spectre.force_targets import force_real
    from spectre.utils import get_freal_fsa_metric

    nt, nz = int(sp.allglobal_mod.nt), int(sp.allglobal_mod.nz)
    mn = int(sp.allglobal_mod.mn)
    x = np.asarray(sp.xin).copy()

    force_real(x, sp, nt=nt // 2, nz=nz // 2)
    assert sp.f_real_shpd.shape[1:] == (nz // 2, nt // 2)
    assert np.isfinite(get_freal_fsa_metric(sp))
    assert (int(sp.field_mod.nt_alloc), int(sp.field_mod.nz_alloc)) == (nt, nz)
    assert (int(sp.fft_backend_mod.nt_plan), int(sp.fft_backend_mod.nz_plan)) == (nt, nz)

    _, _, sg_fine, _, _ = sp.compute_coordinates(2, 1.0, 1, nt, nz, mn)
    _, _, sg_coarse, _, _ = sp.compute_coordinates(2, 1.0, 1, nt // 2, nz // 2, mn)
    fine = sg_fine[:, 0].reshape((nt, nz), order="F")[::2, ::2]
    coarse = sg_coarse[:, 0].reshape((nt // 2, nz // 2), order="F")
    np.testing.assert_allclose(coarse, fine, rtol=1e-12, atol=0)
    assert (int(sp.field_mod.nt_alloc), int(sp.field_mod.nz_alloc)) == (nt, nz)
