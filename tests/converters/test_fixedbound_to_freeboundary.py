# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Tests for spectre.converters.fixedbound_to_freeboundary.

Requires simsopt; skipped automatically when simsopt is not installed.
Test data: tests/test_files/wout_li383_low_res_reference.nc  (NCSX, nfp=3).
The NCSX coil set is loaded from simsopt.configs.zoo.get_data('ncsx').

All expensive fixtures (VMEC load, Biot-Savart setup, freeboundary conversion)
are module-scoped so the computation runs only once per pytest session.
"""

from pathlib import Path

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")
from simsopt.mhd import Vmec  # noqa: E402

from spectre.converters.fixedbound_to_freeboundary import (  # noqa: E402
    fixedbound_to_freeboundary,
    params_to_simsopt_boundary,
)

TEST_FILES = Path(__file__).parents[1] / "test_files"
WOUT_LI383 = TEST_FILES / "wout_li383_low_res_reference.nc"

# Resolution knobs kept small throughout so tests run quickly.
_MPOL = 2
_NTOR = 1
_NTHETA = 32
_NZETA = 32
_NVNS_MPOL = 2
_NVNS_NTOR = 2


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ncsx_setup():
    """Fixed-boundary params + NCSX BiotSavart at minimal resolution."""
    from simsopt.configs.zoo import get_data

    from spectre.converters import simsvmec2spectre

    _, _, _, _, bs = get_data("ncsx")

    vmec = Vmec(str(WOUT_LI383))
    params = simsvmec2spectre(vmec, nvol=2, mpol=_MPOL, ntor=_NTOR)

    return params, bs


@pytest.fixture(scope="module")
def freeb_result(ncsx_setup):
    """Run fixedbound_to_freeboundary at minimal resolution and return the result."""
    params, bs = ncsx_setup

    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    result = fixedbound_to_freeboundary(
        params,
        bs,
        comp_boundary,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
    )
    return result, params


# ---------------------------------------------------------------------------
# params_to_simsopt_boundary
# ---------------------------------------------------------------------------


def test_params_to_simsopt_boundary_plasma_shape(ncsx_setup):
    params, *_ = ncsx_setup
    surf = params_to_simsopt_boundary(params, which="plasma", ntheta=16, nzeta_per_period=16)
    assert surf.nfp == params.physics.nfp
    assert surf.mpol == params.physics.mpol
    assert surf.ntor == params.physics.ntor


def test_params_to_simsopt_boundary_plasma_modes_nonzero(ncsx_setup):
    params, *_ = ncsx_setup
    surf = params_to_simsopt_boundary(params, which="plasma", ntheta=16, nzeta_per_period=16)
    assert np.any(surf.rc != 0.0)


def test_params_to_simsopt_boundary_invalid_which(ncsx_setup):
    params, *_ = ncsx_setup
    with pytest.raises(ValueError, match="which"):
        params_to_simsopt_boundary(params, which="bogus")


# ---------------------------------------------------------------------------
# fixedbound_to_freeboundary – output flags and fields
# ---------------------------------------------------------------------------


def test_lfreebound_set(freeb_result):
    result, _ = freeb_result
    assert result.physics.lfreebound == 1


def test_lconstraint_set(freeb_result):
    result, _ = freeb_result
    assert result.physics.lconstraint == 3


def test_lvcgrid_set(freeb_result):
    result, _ = freeb_result
    assert result.numeric.lvcgrid == 1


# ---------------------------------------------------------------------------
# fixedbound_to_freeboundary – boundary fields
# ---------------------------------------------------------------------------


def test_plasma_boundary_rbc_unchanged(freeb_result):
    """Plasma boundary (rbc) must be identical before and after conversion."""
    result, params_orig = freeb_result
    np.testing.assert_array_equal(result.physics.rbc, params_orig.physics.rbc)


def test_comp_boundary_rwc_populated(freeb_result):
    result, _ = freeb_result
    assert np.any(result.physics.rwc != 0.0)


def test_plasma_and_comp_boundaries_differ(freeb_result):
    """The computational boundary must differ from the plasma boundary."""
    result, _ = freeb_result
    assert np.any(result.physics.rbc != result.physics.rwc)


def test_vns_populated(freeb_result):
    result, _ = freeb_result
    assert np.any(result.physics.vns != 0.0)


# ---------------------------------------------------------------------------
# fixedbound_to_freeboundary – tflux / phiedge
# ---------------------------------------------------------------------------


def test_tflux_vacuum_slot_positive(freeb_result):
    result, _ = freeb_result
    nvol = result.physics.nvol
    assert result.physics.tflux[nvol] > 0.0
    assert np.isfinite(result.physics.tflux[nvol])


def test_tflux_vacuum_slot_greater_than_plasma(freeb_result):
    """Comp boundary encloses more flux than plasma boundary."""
    result, _ = freeb_result
    nvol = result.physics.nvol
    assert result.physics.tflux[nvol] > result.physics.tflux[nvol - 1]


def test_phiedge_finite_nonzero(freeb_result):
    result, _ = freeb_result
    assert np.isfinite(result.physics.phiedge)
    assert result.physics.phiedge != 0.0


# ---------------------------------------------------------------------------
# fixedbound_to_freeboundary – file I/O
# ---------------------------------------------------------------------------


def test_writes_toml_when_filename_given(ncsx_setup, tmp_path):
    params, bs = ncsx_setup

    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    out = tmp_path / "freeb.toml"
    result = fixedbound_to_freeboundary(
        params,
        bs,
        comp_boundary,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
        filename=str(out),
    )
    assert out.exists()
    assert result.physics.lfreebound == 1


# ---------------------------------------------------------------------------
# fixedbound_to_freeboundary – idempotency on already-freeboundary input
# ---------------------------------------------------------------------------


def test_accepts_freeboundary_input_replaces_comp_boundary(freeb_result, ncsx_setup):
    """Passing already-freeboundary params replaces comp boundary without error."""
    result_first, _ = freeb_result
    params, bs = ncsx_setup

    plasma_boundary = params_to_simsopt_boundary(
        result_first, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary2 = plasma_boundary.copy()
    comp_boundary2.extend_via_normal(-0.08)

    result2 = fixedbound_to_freeboundary(
        result_first,
        bs,
        comp_boundary2,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
    )
    assert result2.physics.lfreebound == 1
    # Larger extension → different rwc modes
    assert np.any(result2.physics.rwc != result_first.physics.rwc)
