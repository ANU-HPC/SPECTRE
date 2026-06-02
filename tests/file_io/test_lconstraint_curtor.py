# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest

from spectre.file_io.input_parameters import PhysicsParameters


def _base_kwargs(lconstraint, curtor):
    """Minimal PhysicsParameters kwargs covering the validators for the lconstraint we care about."""
    if lconstraint == 1:
        return dict(
            nvol=3, mpol=1, ntor=1,
            lconstraint=1, curtor=curtor,
            tflux=[0.1, 0.2, 0.3],
            iota=[0.0, 0.25, 0.5, 0.75],
            oita=[0.0, 0.25, 0.5, 0.75],
        )
    if lconstraint == 3:
        return dict(
            nvol=3, mpol=1, ntor=1,
            lconstraint=3, curtor=curtor,
            tflux=[0.1, 0.2, 0.3],
            ivolume=[0.0, 0.0, 0.0],
            isurf=[0.0, 0.0],
        )
    raise ValueError(lconstraint)


def test_small_curtor_clamped_to_zero_when_lconstraint_1():
    p = PhysicsParameters(**_base_kwargs(1, 1e-15))
    assert p.curtor == 0.0


def test_negative_small_curtor_clamped_to_zero_when_lconstraint_1():
    p = PhysicsParameters(**_base_kwargs(1, -3e-17))
    assert p.curtor == 0.0


def test_exactly_zero_curtor_stays_zero_when_lconstraint_1():
    p = PhysicsParameters(**_base_kwargs(1, 0.0))
    assert p.curtor == 0.0


@pytest.mark.parametrize("curtor", [1e-8, 1.0, -2.5, 1e-9])
def test_above_threshold_curtor_unchanged_when_lconstraint_1(curtor):
    p = PhysicsParameters(**_base_kwargs(1, curtor))
    assert p.curtor == curtor


def test_small_curtor_unchanged_when_lconstraint_3():
    # Clamp must NOT fire for other lconstraints — those may genuinely need
    # to carry a tiny curtor through.
    p = PhysicsParameters(**_base_kwargs(3, 1e-15))
    assert p.curtor == 1e-15
