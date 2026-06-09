# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Essential direct tests for the core ``SPECTRE`` class (``spectre.core.core``).

Covers instantiation, the ``parameters`` get/set round-trip, and the
field-consistency tracking used by ``create_spectreout``. Elsewhere ``SPECTRE``
is only used as setup for force / fixed-point / converter tests.
"""

from pathlib import Path

import pytest

from spectre import (
    SPECTRE,
    SPECTREout,
    get_xinit_specwrap,
    remove_file,
    unpack_interfaces,
)
from spectre.file_io import InputParameters

TEST_FILES = Path(__file__).parent.parent / "test_files"

# Free-boundary, toroidal (igeometry=3) rotating-ellipse case, reduced to
# mpol=4/ntor=3 for fast tests.
FNAME_LIGHT = str(TEST_FILES / "rot_ell_light.toml")


@pytest.fixture
def test():
    """A fresh (light) SPECTRE instance per test, output files cleaned up afterwards."""
    obj = SPECTRE(FNAME_LIGHT, verbose=False)
    yield obj
    remove_file(FNAME_LIGHT + ".end")


def test_instantiation_state(test):
    """A freshly instantiated SPECTRE has validated parameters and a non-current field."""
    assert test.name == "rot_ell_light"
    assert isinstance(test.parameters, InputParameters)
    assert test.parameters.physics.nvol == 7
    # nothing solved yet -> negative
    assert test._field_fingerprint is None
    assert test.field_is_current is False


def test_parameters_get_set_roundtrip(test):
    """The parameters getter returns the Fortran state; the setter writes it back."""
    params = test.parameters

    params.physics.pscale = params.physics.pscale + 0.5
    test.parameters = params  # invokes the setter

    updated = test.parameters
    assert updated.physics.pscale == pytest.approx(params.physics.pscale)
    # structural quantities are unchanged by the round-trip
    assert updated.physics.nvol == 7
    assert updated.physics.mpol == params.physics.mpol


def test_create_spectreout_solve_field(test):
    """solve_field=True computes the Beltrami field and marks it consistent."""
    out = test.create_spectreout(solve_field=True)
    assert isinstance(out, SPECTREout)
    assert test.field_is_current is True


def test_stale_field_detected_and_warns(test):
    """Changing the geometry after a tracked solve is detected and warned about."""
    test.create_spectreout(solve_field=True)

    xin = get_xinit_specwrap(test)
    perturbed = xin.copy()
    perturbed[1:] *= 1.0001
    unpack_interfaces(perturbed, test)  # no `xin` setter exists; repack via packxi

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
