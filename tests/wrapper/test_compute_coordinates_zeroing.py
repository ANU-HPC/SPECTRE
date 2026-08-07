# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Check that compute_coordinates' selective array zeroing is safe.

compute_coordinates zeros only the output slots (Rij/Zij/sg/inv_jac/guvij) that are
returned as 0 on the current Lcurvature path; slots it overwrites are not pre-zeroed.
The risk: a slot that is neither zeroed nor written silently returns caller garbage.

To catch that, we make two calls that both start from NaN-filled arrays - one with a
blanket zeroing of every output slot applied first, one without. If the selective
zeroing covers every slot it must, the two agree (NaN==NaN for slots nobody touches).
A missing zero shows up as NaN in one run and 0.0 in the other.

Swept over both geometries, on- and off-axis, with/without stellarator symmetry, and
over the Lcurvature values that are pure geometry (0, 1, 2). Lcurvature >= 3 reads the
DOF-perturbation globals (dRodR/dBdX/...), which are only valid mid-Jacobian, so those
paths can't be called standalone - their zeroing is covered by the finite-difference
Jacobian test instead.
"""

import os.path

import numpy as np
import pytest

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"

LCURVATURES = [0, 1, 2]


def _alloc(ntz, fill):
    def a(shape):
        arr = np.empty(shape, dtype=np.float64, order="F")
        arr[...] = fill
        return arr

    # Fortran bounds: Rij/Zij(1:Ntz,0:3,0:3), sg(1:Ntz,0:3), inv_jac(1:Ntz),
    # guvij(1:Ntz,0:3,0:3,-1:3) -> last axis is 5 wide, python index 0 is Fortran -1.
    return dict(
        rij=a((ntz, 4, 4)),
        zij=a((ntz, 4, 4)),
        sg=a((ntz, 4)),
        inv_jac=a((ntz,)),
        guvij=a((ntz, 4, 4, 5)),
    )


def _apply_blanket_zeroing(arrays, lcurvature):
    """Zero every output slot relevant at this Lcurvature - the reference the selective
    zeroing is checked against. guvij's last axis is -1:3, so Fortran kk=0 is python
    index 1 and kk=1:3 is python 2:5."""
    arrays["rij"][:] = 0.0
    arrays["zij"][:] = 0.0
    if lcurvature > 0:
        arrays["sg"][:, 0] = 0.0
        arrays["inv_jac"][:] = 0.0
        arrays["guvij"][:, 1:4, 1:4, 1] = 0.0
    if lcurvature > 1:
        arrays["sg"][:, 1:4] = 0.0
        arrays["guvij"][:, 1:4, 1:4, 2:5] = 0.0


def _call(test, lvol, lss, lcurvature, nt, nz, mn, lcoord, arrays):
    test.geometry_mod.compute_coordinates(
        lvol,
        lss,
        lcurvature,
        nt,
        nz,
        mn,
        lcoord,
        arrays["rij"],
        arrays["zij"],
        arrays["sg"],
        arrays["inv_jac"],
        arrays["guvij"],
    )


def _cases(test):
    mvol = test.allglobal_mod.mvol
    igeom = test.input_list_mod.igeometry
    yield (2 if mvol >= 2 else 1, 0.3, False)  # some interior interface
    if igeom != 1:
        yield (1, -0.7, True)  # innermost volume, on the axis


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_3vol_light.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lcurvature", LCURVATURES)
@pytest.mark.parametrize("stellsym", [True, False])
def test_selective_zeroing_matches_blanket(fname, lcurvature, stellsym):
    test = SPECTRE(fname, verbose=False)

    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    mn = test.allglobal_mod.mn
    ntz = nt * nz

    test.input_list_mod.enforce_stell_sym = stellsym

    for lvol, lss, lcoord in _cases(test):
        blanket = _alloc(ntz, np.nan)
        _apply_blanket_zeroing(blanket, lcurvature)
        _call(test, lvol, lss, lcurvature, nt, nz, mn, lcoord, blanket)

        selective = _alloc(ntz, np.nan)
        _call(test, lvol, lss, lcurvature, nt, nz, mn, lcoord, selective)

        for name in ("rij", "zij", "sg", "inv_jac", "guvij"):
            a, b = blanket[name], selective[name]
            both_nan = np.isnan(a) & np.isnan(b)
            diff = ~(both_nan | (a == b))
            assert not diff.any(), (
                f"{name}: {int(diff.sum())} slots differ "
                f"(Lcurv={lcurvature}, stellsym={stellsym}, lvol={lvol}, lcoord={lcoord}); "
                f"{int((np.isnan(b) & ~np.isnan(a)).sum())} left as NaN by the selective path"
            )
