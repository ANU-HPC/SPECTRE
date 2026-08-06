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

import warnings
from pathlib import Path

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")
from simsopt.mhd import Vmec  # noqa: E402

from spectre.converters.fix2free import (  # noqa: E402
    _parse_mn_dict,
    fix2free,
    fix2free_refresh_infaces,
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
_MU0 = 4.0 * np.pi * 1e-7

WOUT_LHD = TEST_FILES / "wout_LHD_090493t2330.nc"
INPUT_W7X = TEST_FILES / "input.W7-X_standard_configuration"

# Per-device workflow configs: each seeds its IC from a VMEC-facing file (a wout,
# or an input.* namelist for W7X) and uses the matching simsopt-zoo coils.
# The plasma boundary is interior (tfluxes[-1] < 1).
_DEVICE_CONFIGS = {
    "ncsx": dict(
        source=WOUT_LI383,
        zoo="ncsx",
        tfluxes=[0.2, 0.7],
        interface_guess="wout",
        profile_type=None,
        extend=-0.15,
        explicit_curpol=False,
        needs_map2disc=False,
    ),
    "lhd": dict(
        source=WOUT_LHD,
        zoo="lhd_like",
        tfluxes=[0.2, 0.8],
        interface_guess="wout",
        profile_type="current",
        extend=-0.20,
        explicit_curpol=True,
        needs_map2disc=False,
    ),
    "w7x": dict(
        source=INPUT_W7X,
        zoo="w7x",
        tfluxes=[0.2, 0.8],
        interface_guess="map2disc",
        profile_type=None,
        extend=-0.20,
        explicit_curpol=False,
        needs_map2disc=True,
    ),
}
_DEVICE_CACHE = {}

# Sign of phiedge each device's stock workflow should produce.
_EXPECTED_PHIEDGE_SIGN = {"lhd": -1.0, "w7x": -1.0, "ncsx": +1.0}


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    """Keep VMEC/SPECTRE scratch files (e.g. W7X PARVMEC threed1.*) out of the tree."""
    monkeypatch.chdir(tmp_path)


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
    current profiles, but keeps curpol (the coil poloidal linking current).
    """
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


def test_phiedge_magnitude_preserved_sign_from_coils(freeb_result):
    """Output phiedge keeps the input magnitude; its sign follows the coil field.

    fix2free no longer copies the input phiedge *sign* (the VMEC convention may
    disagree with the coils).  The magnitude is preserved from the input and the
    sign matches the coil-field-derived curpol.
    """
    result, params_orig = freeb_result
    np.testing.assert_allclose(
        abs(result.physics.phiedge),
        abs(params_orig.physics.phiedge),
        rtol=1e-12,
    )
    assert np.sign(result.physics.phiedge) == np.sign(result.physics.curpol)


def test_plasma_boundary_absolute_flux_matches_input_magnitude(freeb_result):
    """Plasma-boundary absolute flux magnitude must equal |input phiedge|.

    With tflux[nvol-1] renormalized to 1.0 and the phiedge magnitude inherited
    from the input, the |absolute flux| at the plasma boundary is exactly
    |phys.phiedge| (the sign is set by the coil field).
    """
    result, params_orig = freeb_result
    nvol = result.physics.nvol
    np.testing.assert_allclose(
        abs(_absolute_flux_at_interface(result, nvol - 1)),
        abs(params_orig.physics.phiedge),
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


def test_curpol_sign_independent_of_input_phiedge_sign(ncsx_setup):
    """Curpol's sign comes from the coil field, NOT from the input phiedge sign.

    Flipping the *input* phiedge must leave both the auto-computed curpol and the
    output phiedge unchanged: their sign is set by the Biot-Savart B_phi direction
    (which the coils determine, not the VMEC convention), and only the phiedge
    *magnitude* is carried over from the input.
    """
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    r_pos = fix2free(params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)

    # Flip the *input* phiedge sign; the coils are untouched.
    new_phys = params.physics.model_copy(update={"phiedge": -params.physics.phiedge})
    params_flipped = params.model_copy(update={"physics": new_phys})
    r_neg = fix2free(params_flipped, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)

    # curpol unchanged: it depends only on the (untouched) coil field.
    np.testing.assert_allclose(r_neg.physics.curpol, r_pos.physics.curpol, rtol=1e-12)
    # output phiedge unchanged too (sign from coils, magnitude is |input|).
    np.testing.assert_allclose(r_neg.physics.phiedge, r_pos.physics.phiedge, rtol=1e-12)
    np.testing.assert_allclose(abs(r_neg.physics.phiedge), abs(params.physics.phiedge), rtol=1e-12)


def test_curpol_auto_warns_assumes_modular_coils(ncsx_setup):
    """Omitting curpol warns that the auto-formula assumes all coils are modular."""
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)
    with pytest.warns(UserWarning, match="modular"):
        fix2free(params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR)


def test_curpol_given_suppresses_warning(ncsx_setup):
    """Passing curpol explicitly does not emit the auto-computation warning."""
    params, bs = ncsx_setup
    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        fix2free(params, bs, comp_boundary, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR, curpol=1.0)
    assert not any("modular" in str(w.message) for w in record)


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


# ---------------------------------------------------------------------------
# Per-device workflow sign check (LHD/W7X negative phiedge, NCSX positive),
# with curpol consistent with phiedge and with B_phi at the magnetic axis.
# ---------------------------------------------------------------------------


def _device_setup(name):
    """Build (params, bs, base_currents, ma) for a device, cached per module."""
    if name not in _DEVICE_CACHE:
        from simsopt.configs.zoo import get_data

        from spectre.converters import simsvmec2spectre

        cfg = _DEVICE_CONFIGS[name]
        _, base_currents, ma, _, bs = get_data(cfg["zoo"])
        vmec = Vmec(str(cfg["source"]))
        params = simsvmec2spectre(
            vmec,
            tfluxes=cfg["tfluxes"],
            mpol=_MPOL,
            ntor=_NTOR,
            interface_guess=cfg["interface_guess"],
            profile_type=cfg["profile_type"],
        )
        _DEVICE_CACHE[name] = (params, bs, base_currents, ma)
    return _DEVICE_CACHE[name]


def _bphi_at(bs, point_xyz):
    """Toroidal field B_phi from Biot-Savart at a Cartesian point (x, y, z)."""
    p = np.asarray(point_xyz, dtype=float).reshape(1, 3)
    bs.set_points(p)
    bx, by, _bz = bs.B()[0]
    phi = np.arctan2(p[0, 1], p[0, 0])
    return -bx * np.sin(phi) + by * np.cos(phi)


@pytest.mark.parametrize(
    "name",
    ["ncsx", "lhd", pytest.param("w7x", marks=pytest.mark.map2disc)],
)
def test_workflow_phiedge_and_curpol_signs(name):
    """fix2free gives the expected phiedge sign per device, with curpol consistent.

    LHD and W7-X have negative phiedge, NCSX positive.  curpol must carry the same
    sign as phiedge, and that sign must match the coil toroidal field B_phi at the
    magnetic axis ``ma.gamma()[0]`` — a point distinct from the boundary centre
    fix2free itself uses, so it independently confirms the field direction.
    """
    cfg = _DEVICE_CONFIGS[name]
    if cfg["needs_map2disc"]:
        pytest.importorskip("map2disc")

    params, bs, base_currents, ma = _device_setup(name)
    comp = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp = comp.copy()
    comp.extend_via_normal(cfg["extend"])

    curpol = 2.0 * 5.0 * base_currents[-1].get_value() * _MU0 if cfg["explicit_curpol"] else None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fix2free(
            params, bs, comp, nvns_mpol=_NVNS_MPOL, nvns_ntor=_NVNS_NTOR, curpol=curpol, vacuum=True
        )

    phiedge = result.physics.phiedge
    curpol_out = result.physics.curpol

    # Expected phiedge sign for this device, and curpol carries the same sign.
    assert np.sign(phiedge) == _EXPECTED_PHIEDGE_SIGN[name]
    assert np.sign(curpol_out) == np.sign(phiedge)

    # curpol agrees with B_phi at the magnetic axis (independent of fix2free's point).
    bphi_axis = _bphi_at(bs, ma.gamma()[0])
    assert bphi_axis != 0.0
    assert np.sign(curpol_out) == np.sign(bphi_axis)


# ---------------------------------------------------------------------------
# fix2free_refresh_infaces – regenerated interface geometry
# ---------------------------------------------------------------------------


def _eval_fourier_surface(modes, nfp, theta, zeta):
    """Evaluate a {'rbc': {...}, 'zbs': {...}, ...} surface at (theta, zeta)."""
    th = np.asarray(theta, dtype=float)
    R = np.zeros_like(th)
    Z = np.zeros_like(th)
    for m, n, v in _parse_mn_dict(modes.get("rbc", {})):
        R += v * np.cos(m * th - n * nfp * zeta)
    for m, n, v in _parse_mn_dict(modes.get("rbs", {})):
        R += v * np.sin(m * th - n * nfp * zeta)
    for m, n, v in _parse_mn_dict(modes.get("zbs", {})):
        Z += v * np.sin(m * th - n * nfp * zeta)
    for m, n, v in _parse_mn_dict(modes.get("zbc", {})):
        Z += v * np.cos(m * th - n * nfp * zeta)
    return R, Z


def _refreshed_surfaces(result):
    """Interface (and plasma-boundary) Fourier dicts of a refreshed result.

    Returns the list [interface_1, ..., interface_nvol]; interface_nvol is the
    regenerated plasma boundary (identical to physics rbc/zbs by construction).
    """
    from spectre.file_io import input_parameters_to_dict

    data = input_parameters_to_dict(result)
    allrzrz = data["physics"]["allrzrz"]
    nvol = result.physics.nvol
    return [allrzrz[f"interface_{i}"] for i in range(1, nvol + 1)]


def _mean_minor_radius(modes, nfp, ntheta=64, nzeta=8):
    """Mean distance to the per-zeta-cut centroid, averaged over zeta cuts."""
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    radii = []
    for zeta in np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False):
        R, Z = _eval_fourier_surface(modes, nfp, theta, zeta)
        radii.append(np.mean(np.hypot(R - R.mean(), Z - Z.mean())))
    return float(np.mean(radii))


@pytest.fixture(scope="module")
def refreshed_pair(ncsx_setup):
    """fix2free_refresh_infaces on NCSX with and without flux-weighted rho."""
    pytest.importorskip("map2disc")
    params, bs = ncsx_setup

    plasma_boundary = params_to_simsopt_boundary(
        params, which="plasma", ntheta=_NTHETA, nzeta_per_period=_NZETA
    )
    comp_boundary = plasma_boundary.copy()
    comp_boundary.extend_via_normal(-0.05)

    result_fw = fix2free_refresh_infaces(
        params,
        bs,
        comp_boundary,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
        flux_weighted_rho=True,
    )
    result_uw = fix2free_refresh_infaces(
        params,
        bs,
        comp_boundary,
        nvns_mpol=_NVNS_MPOL,
        nvns_ntor=_NVNS_NTOR,
        flux_weighted_rho=False,
    )
    return result_fw, result_uw


@pytest.mark.map2disc
@pytest.mark.parametrize("which", ["flux_weighted", "sqrt_tflux"])
def test_refresh_infaces_validates_and_nests(refreshed_pair, which):
    """Result validates as free-boundary and the regenerated surfaces nest.

    For each zeta cut, the radius about the innermost interface's centroid must
    be strictly monotone across consecutive interfaces at every theta — the
    conformal map preserves the theta correspondence between surfaces, so any
    crossing would show up as a pointwise ordering violation.  The comp
    boundary (rwc/zws) must in turn enclose the regenerated plasma boundary.
    """
    result = refreshed_pair[0] if which == "flux_weighted" else refreshed_pair[1]
    assert result.physics.lfreebound == 1

    nfp = result.physics.nfp
    surfaces = _refreshed_surfaces(result)
    # The serialized comp boundary uses the same "(m, n)" dict convention.
    from spectre.file_io import input_parameters_to_dict

    phys_dict = input_parameters_to_dict(result)["physics"]
    surfaces.append({"rbc": phys_dict["rwc"], "zbs": phys_dict["zws"]})

    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    for zeta in np.linspace(0.0, 2.0 * np.pi / nfp, 8, endpoint=False):
        rz = [_eval_fourier_surface(modes, nfp, theta, zeta) for modes in surfaces]
        # Common interior point: centroid of the innermost interface.
        R0c, Z0c = rz[0][0].mean(), rz[0][1].mean()
        radii = np.array([np.hypot(R - R0c, Z - Z0c) for R, Z in rz])
        assert np.all(np.diff(radii, axis=0) > 0.0), f"interfaces cross at zeta={zeta} ({which})"


@pytest.mark.map2disc
def test_refresh_infaces_flux_weighted_boundary_lies_outside(refreshed_pair):
    """Flux weighting must push the plasma boundary outward.

    The vacuum field B_phi = G/R is weaker at large R, so reaching the plasma
    flux fraction requires a larger conformal radius than rho = sqrt(tflux):
    the flux-weighted plasma boundary sits closer to the comp boundary.
    """
    result_fw, result_uw = refreshed_pair
    nfp = result_fw.physics.nfp

    boundary_fw = _refreshed_surfaces(result_fw)[-1]
    boundary_uw = _refreshed_surfaces(result_uw)[-1]

    assert _mean_minor_radius(boundary_fw, nfp) > _mean_minor_radius(boundary_uw, nfp)
