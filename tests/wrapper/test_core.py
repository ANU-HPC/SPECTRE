# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Essential direct tests for the core ``SPECTRE`` class (``spectre.core.core``).

Covers the coordinate facade and field-consistency tracking used by
``create_spectreout``.
"""

import numpy as np
import pytest

from spectre import (
    SPECTRE,
    SPECTREout,
    remove_file,
)

FNAME = "tests/wrapper/qa_7vol_test.toml"
PRESSURE_DECK = "tests/compare/G2V32L1Fi/input.toml"  # Ladiabatic = 0, pscale = 1, adiabatic = 0
FREEB_DECK = "tests/wrapper/test_freeb_w7x.toml"


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


def test_dforce_defeats_the_compute_fields_skip(test):
    """A raw force evaluation elsewhere must not leave a skippable solve behind.

    ``compute_fields`` skips a repeated solve at one ``xin``; ``dforce`` re-solves the field
    (and repacks the geometry) without going through it, so after ``force_modes_dforce`` at
    another geometry the next ``force_real`` at the original ``xin`` must solve again. Left
    unchecked this returned a force 8.7 % off on the current-ring deck.
    """
    x1 = test.xin.copy()
    reference = test.force_real(x1).copy()  # fresh instance: the true force at x1

    f = test.dforce(1.02 * x1)
    assert f.shape == x1.shape  # plain ngdof vectors on the Python side, no dead slot
    again = test.force_real(x1)

    np.testing.assert_allclose(again, reference, rtol=1e-12, atol=1e-12)


def test_adiabatic_initialized_from_pressure():
    """Construction sets the adiabatic constants the removed Fortran driver used to set.

    Every dforce-based consumer (ForceErr, the spectral force, the Jacobian's gamma term)
    reads ``adiabatic``, so a ``pscale != 0`` deck that only specifies ``pressure`` would
    otherwise run at zero pressure; the free-boundary vacuum slot is reset to zero.
    """
    sp = SPECTRE(PRESSURE_DECK, verbose=False)
    nvol = int(sp.input_list_mod.nvol)
    pressure = np.asarray(sp.input_list_mod.pressure)[:nvol]
    vvolume = np.asarray(sp.allglobal_mod.vvolume)[:nvol]
    adiabatic = np.asarray(sp.input_list_mod.adiabatic)[:nvol]
    assert np.all(pressure != 0.0) and np.all(vvolume > 0.0)
    np.testing.assert_allclose(
        adiabatic, pressure * vvolume ** float(sp.input_list_mod.gamma), rtol=1e-14
    )

    freeb = SPECTRE(FREEB_DECK, verbose=False)
    mvol = int(freeb.allglobal_mod.mvol)
    assert float(freeb.input_list_mod.adiabatic[mvol - 1]) == 0.0
    assert float(freeb.input_list_mod.pressure[mvol - 1]) == 0.0
    remove_file(FREEB_DECK + ".end")


def test_parameters_setter_invalidates_field(test):
    """Setting parameters reallocates/zeros the field, clearing the fingerprint."""
    test.create_spectreout(solve_field=True)
    assert test._field_fingerprint is not None

    test.parameters = test.parameters  # round-trip set

    assert test._field_fingerprint is None
    assert test.field_is_current is False
