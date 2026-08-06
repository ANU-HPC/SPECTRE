# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Essential direct tests for the core ``SPECTRE`` class (``spectre.core.core``).

Covers the coordinate facade and field-consistency tracking used by
``create_spectreout``.
"""

import pytest

from spectre import (
    SPECTRE,
    SPECTREout,
    remove_file,
)

FNAME = "tests/wrapper/qa_7vol_test.toml"


@pytest.fixture
def test():
    """A fresh (light) SPECTRE instance per test, output files cleaned up afterwards."""
    obj = SPECTRE(FNAME, verbose=False)
    yield obj
    remove_file(FNAME + ".end")


def test_instantiation_has_no_tracked_field(test):
    """A freshly instantiated SPECTRE has no tracked field solve."""
    assert test._field_fingerprint is None
    assert test.field_is_current is False


def test_compute_coordinates_returns_caller_owned_arrays(test):
    """The coordinate facade returns all explicit Fortran output arrays."""
    nt = int(test.allglobal_mod.nt)
    nz = int(test.allglobal_mod.nz)
    mn = int(test.allglobal_mod.mn)

    rij, zij, sg, inv_jac, guvij = test.compute_coordinates(1, 1.0, 1, nt, nz, mn)

    assert rij.shape == zij.shape == (nt * nz, 4, 4)
    assert sg.shape == (nt * nz, 4)
    assert inv_jac.shape == (nt * nz,)
    assert guvij.shape == (nt * nz, 4, 4, 5)


def test_create_spectreout_solve_field(test):
    """solve_field=True computes the Beltrami field and marks it consistent."""
    out = test.create_spectreout(solve_field=True)
    assert isinstance(out, SPECTREout)
    assert test.field_is_current is True


def test_stale_field_detected_and_warns(test):
    """Changing the geometry after a tracked solve is detected and warned about."""
    test.create_spectreout(solve_field=True)

    xin = test.xin
    perturbed = xin.copy()
    perturbed[1:] *= 1.0001
    test.xin = perturbed

    assert test.field_is_current is False
    with pytest.warns(UserWarning, match="stale"):
        test.create_spectreout()


def test_parameters_setter_invalidates_field(test):
    """Setting parameters reallocates/zeros the field, clearing the fingerprint."""
    test.create_spectreout(solve_field=True)
    assert test._field_fingerprint is not None

    test.parameters = test.parameters  # round-trip set

    assert test._field_fingerprint is None
    assert test.field_is_current is False
