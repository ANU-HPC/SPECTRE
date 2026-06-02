# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def sp():
    # qa_7vol_test is a small fixed-boundary case; we don't need free-boundary
    # to exercise xin.
    return SPECTRE(curr_path + "qa_7vol_test.toml", verbose=False)


def test_xin_returns_fortran_contiguous_array_of_correct_size(sp):
    x = sp.xin
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    assert x.shape == (sp.allglobal_mod.ngdof + 1,)
    assert x.flags["F_CONTIGUOUS"]


def test_xin_setter_round_trips(sp):
    # Read, write back unchanged, read again — must match exactly.
    x0 = sp.xin.copy()
    sp.xin = x0
    x1 = sp.xin
    np.testing.assert_array_equal(x0, x1)


def test_xin_setter_actually_updates_fortran_arrays(sp):
    # Perturb one element of xin, push, read back — that element should reflect
    # the change. (We use a small perturbation so we don't break Fortran-side
    # invariants in subsequent tests.)
    x0 = sp.xin.copy()
    irbc_before = sp.allglobal_mod.irbc.copy()

    x_perturbed = x0.copy()
    # Index 1 is the first real DOF (index 0 is the unused [0] slot in the
    # F-indexed (0:ngdof) layout). Perturb it.
    x_perturbed[1] += 1.0e-6
    sp.xin = x_perturbed

    irbc_after = sp.allglobal_mod.irbc.copy()
    # At least one of the irbc entries should differ now.
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


def test_unpack_interfaces_helper_emits_deprecation_warning(sp):
    """The free function unpack_interfaces is deprecated; calling it warns."""
    import pytest

    from spectre.utils import unpack_interfaces

    x = sp.xin.copy()
    with pytest.deprecated_call(match="unpack_interfaces is deprecated"):
        unpack_interfaces(x, sp)
