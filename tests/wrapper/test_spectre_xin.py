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
    """Reading xin gives an F-contiguous float64 array of length ngdof+1."""
    x = sp.xin
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    assert x.shape == (sp.allglobal_mod.ngdof + 1,)
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
    # Index 1 is the first real DOF (index 0 is the unused [0] slot in the
    # F-indexed (0:ngdof) layout). Perturb it.
    x_perturbed[1] += 1.0e-6
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
