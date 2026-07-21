# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Essential direct tests for the core ``SPECTRE`` class (``spectre.core.core``).

Covers instantiation, the ``parameters`` get/set round-trip, and the
field-consistency tracking used by ``create_spectreout``. Elsewhere ``SPECTRE``
is only used as setup for force / fixed-point / converter tests.
"""

from pathlib import Path

import numpy as np
import pytest

from spectre import (
    SPECTRE,
    SPECTREout,
    get_xinit_specwrap,
    remove_file,
    unpack_interfaces,
    update_bnorm,
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
    assert test._solved_xin is None
    assert test.field_is_current is False


def test_field_matches_after_force_real(test):
    """force_real stamps the raw argument; field_matches ties the solve to it."""
    x = test.xin
    test.force_real(x)

    assert test.field_matches(x) is True
    assert test.field_is_current is True

    perturbed = x.copy()
    perturbed[1:] *= 1.0001
    assert test.field_matches(perturbed) is False


def test_field_matches_tracks_latest_solve(test):
    """A later solve at a different x supersedes the earlier stamp.

    This is the rejected-trial analogue: the field in memory belongs to the
    last evaluated x, not to an earlier (accepted) one.
    """
    x1 = test.xin
    test.force_real(x1)

    x2 = x1.copy()
    x2[1:] *= 1.0001  # small non-overlapping perturbation, as above
    test.force_real(x2)

    assert test.field_matches(x2) is True
    assert test.field_matches(x1) is False


def test_unpack_without_solve_invalidates_field_matches(test):
    """Unpacking a different geometry after a solve breaks the match for both vectors."""
    x1 = test.xin
    test.force_real(x1)

    x2 = x1.copy()
    x2[1:] *= 1.0001
    unpack_interfaces(x2, test)  # geometry moved, no solve

    assert test.field_matches(x1) is False  # field_is_current is now False
    assert test.field_matches(x2) is False  # never solved at x2


def test_update_bnorm_clears_tracked_solve(test):
    """An iBns refresh invalidates the tracked solve.

    The field in memory no longer solves the problem defined by the current
    inputs (in free-boundary the vacuum-volume solve uses iBns as boundary
    data), so both the fingerprint and the raw-argument stamp must go.
    """
    x = test.xin
    test.force_real(x)
    assert test.field_matches(x) is True

    update_bnorm(test, first_free_bound=True)

    assert test._field_fingerprint is None
    assert test._solved_xin is None
    assert test.field_is_current is False
    assert test.field_matches(x) is False


def test_overlap_path_does_not_touch_stamps(test, monkeypatch):
    """The interfaces-overlap early return neither stamps nor clears the tracked solve."""
    from spectre import force_targets

    x = test.xin
    test.force_real(x)
    assert test.field_matches(x) is True

    monkeypatch.setattr(force_targets, "check_intersection_wrapper", lambda *a, **k: True)
    x_bad = x.copy()
    x_bad[1:] *= 1.0001
    f = test.force_real(x_bad)
    assert np.all(f == 1e10)

    # Stamps survive: the early return leaves the Fortran state untouched, so
    # the field genuinely still corresponds to the earlier solve.
    assert test.field_matches(x) is True
    assert test.field_matches(x_bad) is False


def test_overlap_path_does_not_stamp_fresh_object(test, monkeypatch):
    """An overlap evaluation on a fresh object leaves both stamps at None."""
    from spectre import force_targets

    monkeypatch.setattr(force_targets, "check_intersection_wrapper", lambda *a, **k: True)
    test.force_real()

    assert test._field_fingerprint is None
    assert test._solved_xin is None


def test_compute_fields_stamps_solved_xin(test):
    """compute_fields records the raw vector it solved at, like force_real."""
    x = test.xin
    test.compute_fields()

    assert test.field_matches(x) is True
