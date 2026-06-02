# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Tests for spectre.converters.fix2free.

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

from spectre.converters.fix2free import (  # noqa: E402
    fix2free,
    params_to_simsopt_boundary,
)

pytestmark = pytest.mark.simsopt

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
    """Run fix2free at minimal resolution and return the result."""
    params, bs = ncsx_setup

    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    result = fix2free(
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
# fix2free – output flags and fields
# ---------------------------------------------------------------------------


def test_lfreebound_set(freeb_result):
    result, _ = freeb_result
    assert result.physics.lfreebound == 1


def test_lconstraint_preserved(freeb_result):
    """Non-vacuum conversion preserves the input lconstraint."""
    result, params_orig = freeb_result
    assert result.physics.lconstraint == params_orig.physics.lconstraint


def test_lconstraint_vacuum_is_3(ncsx_setup):
    """vacuum=True forces lconstraint=3 regardless of the input lconstraint."""
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)
    result = fix2free(
        params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR, vacuum=True
    )
    assert result.physics.lconstraint == 3


def test_vacuum_zeros_pressure_and_currents(ncsx_setup):
    """vacuum=True zeros pressure/pscale/curtor and the volume+interface
    current profiles, but keeps curpol (the coil poloidal linking current)."""
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)
    result = fix2free(
        params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR, vacuum=True
    )
    phys = result.physics
    assert phys.pscale == 0.0
    assert phys.curtor == 0.0
    assert np.all(phys.pressure == 0.0)
    assert np.all(phys.ivolume == 0.0)
    assert np.all(phys.isurf == 0.0)
    assert phys.lbnszero is True
    assert phys.curpol != 0.0


@pytest.mark.parametrize("current_mode", ["volume", "sheet"])
def test_nonvacuum_carries_currents_over_unchanged(ncsx_setup, current_mode, tmp_path):
    """Non-vacuum conversion must preserve isurf/ivolume from the input.

    Regression for a bug where fix2free zeroed both the volume (ivolume) and
    interface (isurf) current profiles in the free-boundary output, dropping
    the lconstraint=3 currents produced by simsvmec2spectre.  The plasma-volume
    entries must equal the input; the vacuum slot at index nvol is the extended
    Fortran constraint ivolume[nvol] = ivolume[nvol-1].  Values must also
    survive a TOML write/read round-trip.
    """
    from spectre.converters import simsvmec2spectre

    _, bs = ncsx_setup

    vmec = Vmec(str(WOUT_LI383))
    params = simsvmec2spectre(vmec, nvol=2, mpol=_MPOL, ntor=_NTOR, current_mode=current_mode)
    nvol = params.physics.nvol

    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    out = tmp_path / "freeb_currents.toml"
    result = fix2free(
        params,
        bs,
        comp_boundary,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
        filename=str(out),
    )

    # isurf is carried over completely unchanged.
    np.testing.assert_array_equal(result.physics.isurf, params.physics.isurf)
    # ivolume: plasma-volume entries unchanged; vacuum slot = previous interface.
    np.testing.assert_array_equal(result.physics.ivolume[:nvol], params.physics.ivolume[:nvol])
    assert result.physics.ivolume[nvol] == params.physics.ivolume[nvol - 1]

    # The relevant mode must actually be populated for this to be meaningful.
    if current_mode == "volume":
        assert np.any(params.physics.ivolume[:nvol] != 0.0)
    else:
        assert np.any(params.physics.isurf[:nvol] != 0.0)

    # Round-trip through TOML preserves the currents.
    from spectre.file_io.toml_io import read_input_parameters_from_toml

    reloaded = read_input_parameters_from_toml(str(out))
    np.testing.assert_array_equal(reloaded.physics.isurf, params.physics.isurf)
    np.testing.assert_array_equal(reloaded.physics.ivolume, result.physics.ivolume)


def test_lvcgrid_set(freeb_result):
    result, _ = freeb_result
    assert result.numeric.lvcgrid == 1


# ---------------------------------------------------------------------------
# fix2free – boundary fields
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
# fix2free – tflux / phiedge
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
# fix2free – file I/O
# ---------------------------------------------------------------------------


def test_writes_toml_when_filename_given(ncsx_setup, tmp_path):
    params, bs = ncsx_setup

    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    out = tmp_path / "freeb.toml"
    result = fix2free(
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
# Physics: absolute toroidal flux is preserved from input (VMEC), and the
# comp/plasma ratio comes from the coils
# ---------------------------------------------------------------------------
#
# After Fortran's tflux ← tflux/tflux(Nvol) normalization, the physical flux at
# interface i is  phiedge * tflux[i] / tflux[nvol-1].  fix2free preserves the
# input phiedge (it comes from the original equilibrium / VMEC, not from the
# coils — see fix2free docstring) and uses the coil-derived ratio
# ToroidalFlux(comp)/ToroidalFlux(plasma) only for the relative comp slot.


def _absolute_flux_at_interface(params, i):
    """Phiedge * tflux[i] / tflux[nvol-1] — what Fortran sees after normalization."""
    phys = params.physics
    return phys.phiedge * phys.tflux[i] / phys.tflux[phys.nvol - 1]


def test_phiedge_preserved_from_input(freeb_result):
    """Output phiedge must equal the input phiedge (taken from VMEC, not coils)."""
    result, params_orig = freeb_result
    np.testing.assert_allclose(
        result.physics.phiedge,
        params_orig.physics.phiedge,
        rtol=1e-12,
    )


def test_plasma_boundary_absolute_flux_equals_input_phiedge(freeb_result):
    """Plasma-boundary absolute flux must equal input phiedge after fix2free.

    With tflux[nvol-1] renormalized to 1.0 and phiedge inherited from input,
    the absolute flux at the plasma boundary is exactly phys.phiedge.
    """
    result, params_orig = freeb_result
    nvol = result.physics.nvol
    np.testing.assert_allclose(
        _absolute_flux_at_interface(result, nvol - 1),
        params_orig.physics.phiedge,
        rtol=1e-12,
    )


def test_comp_slot_equals_coil_flux_ratio(freeb_result, ncsx_setup):
    """tflux[nvol] in the output must equal ToroidalFlux(comp)/ToroidalFlux(plasma)."""
    from simsopt.geo import ToroidalFlux

    result, _ = freeb_result
    _, bs = ncsx_setup
    plasma = params_to_simsopt_boundary(
        result, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp = params_to_simsopt_boundary(result, which="comp", ntheta=_NTHETA, nzeta_per_period=_NZETA)
    expected_ratio = float(ToroidalFlux(comp, bs).J()) / float(ToroidalFlux(plasma, bs).J())
    nvol = result.physics.nvol
    actual_ratio = result.physics.tflux[nvol] / result.physics.tflux[nvol - 1]
    np.testing.assert_allclose(actual_ratio, expected_ratio, rtol=1e-8)


def test_fix2free_invariant_under_input_tflux_rescaling(ncsx_setup):
    """Multiplying the input tflux by an arbitrary scalar must not change anything.

    Fortran normalizes by tflux(Nvol), so only ratios matter.  After fix2free,
    both phiedge and the normalized tflux profile must be identical regardless
    of how the input tflux array was scaled.
    """
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    r1 = fix2free(params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)

    # Rescale phys.tflux by 0.3; phiedge stays the same (VMEC truth).
    nvol = params.physics.nvol
    scaled_tflux = params.physics.tflux.copy()
    scaled_tflux[:nvol] *= 0.3
    new_phys = params.physics.model_copy(update={"tflux": scaled_tflux})
    params_scaled = params.model_copy(update={"physics": new_phys})
    r2 = fix2free(params_scaled, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)

    # phiedge identical
    np.testing.assert_allclose(r2.physics.phiedge, r1.physics.phiedge, rtol=1e-12)
    # absolute flux at every interface identical
    for i in range(r1.physics.nvol + 1):
        np.testing.assert_allclose(
            _absolute_flux_at_interface(r2, i),
            _absolute_flux_at_interface(r1, i),
            rtol=1e-10,
            err_msg=f"interface {i} absolute flux changed under rescaling",
        )


def test_curpol_sign_matches_phiedge_sign(freeb_result):
    """Curpol and phiedge must carry the same sign.

    Both quantities are tied to sign(B_φ): phiedge = ∫B_φ dA on the boundary;
    curpol = μ₀ × I_pol_linked = Ampère integral on a poloidal loop.  A
    sign mismatch is unphysical (SPECTRE would try to enforce a vacuum-region
    B_φ pointing opposite to the coil field).
    """
    result, _ = freeb_result
    assert result.physics.phiedge != 0.0
    assert result.physics.curpol != 0.0
    assert np.sign(result.physics.curpol) == np.sign(result.physics.phiedge)


def test_curpol_sign_flips_with_phiedge(ncsx_setup):
    """Flipping the sign of input phiedge must flip the auto-computed curpol."""
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    # Baseline (positive phiedge from VMEC)
    r_pos = fix2free(params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)

    # Flip phiedge
    new_phys = params.physics.model_copy(update={"phiedge": -params.physics.phiedge})
    params_flipped = params.model_copy(update={"physics": new_phys})
    r_neg = fix2free(params_flipped, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)

    np.testing.assert_allclose(r_neg.physics.curpol, -r_pos.physics.curpol, rtol=1e-12)
    assert np.sign(r_neg.physics.phiedge) == np.sign(r_neg.physics.curpol)


def test_fix2free_iota_preserved_at_same_absolute_flux(ncsx_setup):
    """Iota values must stay attached to the same absolute toroidal flux.

    Because fix2free preserves phiedge from the input and renormalizes tflux so
    tflux[nvol-1]=1.0, the absolute flux at every plasma-side interface is
    unchanged.  iota[i+1] continues to describe the rotational transform at
    the same physical surface it did before conversion.
    """
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    nvol = params.physics.nvol
    iota_in = params.physics.iota.copy()
    iota_in[1 : nvol + 1] = np.linspace(0.3, 0.45, nvol)
    new_phys = params.physics.model_copy(update={"iota": iota_in, "lconstraint": 1})
    params_iota = params.model_copy(update={"physics": new_phys})

    result = fix2free(
        params_iota,
        bs,
        comp_boundary,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
    )

    np.testing.assert_array_equal(
        result.physics.iota[1 : nvol + 1],
        iota_in[1 : nvol + 1],
    )
    for i in range(nvol):
        flux_in = (
            params_iota.physics.phiedge
            * params_iota.physics.tflux[i]
            / params_iota.physics.tflux[nvol - 1]
        )
        flux_out = _absolute_flux_at_interface(result, i)
        np.testing.assert_allclose(
            flux_out,
            flux_in,
            rtol=1e-10,
            err_msg=f"interface {i}: iota now at different absolute flux",
        )


# ---------------------------------------------------------------------------
# fix2free – idempotency on already-freeboundary input
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

    result2 = fix2free(
        result_first,
        bs,
        comp_boundary2,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
    )
    assert result2.physics.lfreebound == 1
    # Larger extension → different rwc modes
    assert np.any(result2.physics.rwc != result_first.physics.rwc)
