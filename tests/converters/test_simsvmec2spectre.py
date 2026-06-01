# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Integration tests for spectre.converters.simsvmec2spectre.

Requires simsopt; skipped automatically when simsopt is not installed.
Test data: tests/test_files/{wout_iota.nc, input.iota, input.qabeta_curr}.
"""

from pathlib import Path

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")
from simsopt.mhd import Vmec  # noqa: E402

from spectre.converters.simsvmec2spectre import simsvmec2spectre
from spectre.converters.vmec_helpers import (  # noqa: E402
    MU0,
    VmecPhysics,
    build_spectre_input,
    compute_interface_guesses,
    compute_iota_profile,
    compute_iota_profile_from_iotaf,
    compute_pressure_profile,
    make_iota_interpolator,
    vmec_boundary_to_fourier_coeffs,
)

TEST_FILES = Path(__file__).parents[1] / "test_files"
WOUT_IOTA = TEST_FILES / "wout_iota.nc"
INPUT_IOTA = TEST_FILES / "input.iota"
INPUT_CURR = TEST_FILES / "input.qabeta_curr"

TFLUXES = np.linspace(0.0, 1.0, 6, endpoint=True)[1:]  # 5 volumes


# ---------------------------------------------------------------------------
# Fixtures: VmecPhysics constructed directly from simsopt + vmec_helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def iota_physics():
    vmec = Vmec(str(WOUT_IOTA))
    src = vmec.wout
    rbc, zbs = vmec_boundary_to_fourier_coeffs(vmec.boundary)
    return VmecPhysics(
        curtor=src.ctor,
        phiedge=src.phi[-1],
        pres_scale=src.presf[0],
        am=src.am,
        ai=src.ai,
        ac=src.ac,
        nfp=src.nfp,
        rbc=rbc,
        zbs=zbs,
        pcurr_type="power_series",
        profile_type="iota",
    )


@pytest.fixture
def current_physics():
    vmec = Vmec(str(INPUT_CURR))
    src = vmec.indata
    rbc, zbs = vmec_boundary_to_fourier_coeffs(vmec.boundary)
    return VmecPhysics(
        curtor=src.curtor,
        phiedge=src.phiedge,
        pres_scale=src.pres_scale,
        am=src.am,
        ai=src.ai,
        ac=src.ac,
        nfp=src.nfp,
        rbc=rbc,
        zbs=zbs,
        pcurr_type="power_series",
        profile_type="current",
    )


# ---------------------------------------------------------------------------
# simsvmec2spectre – profile type and source dispatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", [WOUT_IOTA, INPUT_IOTA])
def test_iota_equilibrium_sets_lconstraint_1(path, tmp_path):
    vmec = Vmec(str(path))
    result = simsvmec2spectre(vmec, nvol=3, filename=str(tmp_path / "out.toml"))
    assert result.physics.lconstraint == 1


def test_current_equilibrium_sets_lconstraint_3(tmp_path):
    vmec = Vmec(str(INPUT_CURR))
    result = simsvmec2spectre(vmec, nvol=3, filename=str(tmp_path / "out.toml"))
    assert result.physics.lconstraint == 3


def test_bad_vmec_object_raises():
    with pytest.raises(RuntimeError, match="indata.*wout"):
        simsvmec2spectre(object(), nvol=3)


# ---------------------------------------------------------------------------
# compute_pressure_profile
# ---------------------------------------------------------------------------


def test_pressure_profile_normalization(iota_physics):
    profile = compute_pressure_profile(iota_physics.am, iota_physics.pres_scale, TFLUXES)
    assert profile.shape == (len(TFLUXES),)
    assert np.all(profile >= 0.0)


# ---------------------------------------------------------------------------
# compute_iota_profile
# ---------------------------------------------------------------------------


def test_iota_profile_at_interfaces(iota_physics):
    iota = compute_iota_profile(iota_physics.ai, TFLUXES)
    assert iota.shape == (len(TFLUXES),)


# ---------------------------------------------------------------------------
# make_iota_interpolator / compute_iota_profile_from_iotaf
# ---------------------------------------------------------------------------


def test_make_iota_interpolator_wout():
    vmec = Vmec(str(WOUT_IOTA))
    f = make_iota_interpolator(vmec)
    assert f(TFLUXES).shape == (len(TFLUXES),)
    assert np.ndim(f(0.5)) == 0


def test_make_iota_interpolator_indata():
    vmec = Vmec(str(INPUT_IOTA))
    f = make_iota_interpolator(vmec)
    assert f(TFLUXES).shape == (len(TFLUXES),)
    assert np.ndim(f(0.5)) == 0


def test_iota_profile_from_iotaf_shape():
    vmec = Vmec(str(WOUT_IOTA))
    iota = compute_iota_profile_from_iotaf(vmec.wout.iotaf, TFLUXES)
    assert iota.shape == (len(TFLUXES),)


def test_wout_iota_uses_iotaf():
    """simsvmec2spectre from a wout uses iotaf interpolation, not the ai polynomial."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, tfluxes=TFLUXES)
    nvol = len(TFLUXES)
    iota_from_converter = result.physics.iota[1 : nvol + 1]
    iota_from_polynomial = compute_iota_profile(vmec.wout.ai, TFLUXES)
    iota_from_iotaf = compute_iota_profile_from_iotaf(vmec.wout.iotaf, TFLUXES)
    np.testing.assert_allclose(iota_from_converter, iota_from_iotaf, rtol=1e-6)
    # iotaf and polynomial should differ at least slightly (they need not be identical)
    assert not np.allclose(iota_from_iotaf, iota_from_polynomial)


# ---------------------------------------------------------------------------
# profile_type override
# ---------------------------------------------------------------------------


def test_profile_type_override_iota_explicit():
    """Explicitly passing profile_type='iota' on a wout still sets lconstraint=1."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, profile_type="iota")
    assert result.physics.lconstraint == 1


def test_profile_type_override_current_on_iota_wout():
    """Forcing profile_type='current' on an iota wout flips lconstraint to 3."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, profile_type="current")
    assert result.physics.lconstraint == 3


# ---------------------------------------------------------------------------
# build_spectre_input – iota case
# ---------------------------------------------------------------------------


def test_build_spectre_input_iota(iota_physics):
    dict_pydantic, profiles = build_spectre_input(
        iota_physics, TFLUXES, mpol=3, ntor=2, current_mode="volume"
    )
    assert dict_pydantic.physics.nvol == len(TFLUXES)
    assert dict_pydantic.physics.lconstraint == 1
    assert "iota" in profiles
    # iota is stored in a fixed-size padded array; check the populated slice
    nvol = len(TFLUXES)
    assert dict_pydantic.physics.iota[0] == 0.0  # axis value
    assert np.all(dict_pydantic.physics.iota[1 : nvol + 1] != 0.0)


# ---------------------------------------------------------------------------
# build_spectre_input – current cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["volume", "sheet"])
def test_build_spectre_input_current(current_physics, mode):
    dict_pydantic, profiles = build_spectre_input(
        current_physics, TFLUXES, mpol=3, ntor=2, current_mode=mode
    )
    assert dict_pydantic.physics.nvol == len(TFLUXES)
    assert dict_pydantic.physics.lconstraint == 3


def test_volume_current_total_matches_curtor(current_physics):
    """The outermost interface should carry mu0 * curtor."""
    _, profiles = build_spectre_input(
        current_physics, TFLUXES, mpol=3, ntor=2, current_mode="volume"
    )
    expected = -current_physics.curtor * MU0
    np.testing.assert_allclose(profiles["volume_current"][-1], expected, rtol=1e-4)


def test_curtor_equals_ivolume_last(current_physics):
    """dict_physics.curtor must equal ivolume[-1] so they are always consistent."""
    result, profiles = build_spectre_input(
        current_physics, TFLUXES, mpol=3, ntor=2, current_mode="volume"
    )
    np.testing.assert_allclose(result.physics.curtor, profiles["volume_current"][-1], rtol=1e-10)


def test_curtor_zero_when_profiles_zero(iota_physics):
    """Forcing current mode on an iota equilibrium (ac=0) must give curtor=0."""
    zero_current_physics = iota_physics._replace(profile_type="current")
    result, _ = build_spectre_input(
        zero_current_physics, TFLUXES, mpol=3, ntor=2, current_mode="volume"
    )
    assert result.physics.curtor == 0.0


# ---------------------------------------------------------------------------
# simsvmec2spectre – end-to-end
# ---------------------------------------------------------------------------


def test_writes_toml_when_filename_given(tmp_path):
    vmec = Vmec(str(WOUT_IOTA))
    out = tmp_path / "gen.toml"
    result = simsvmec2spectre(vmec, nvol=4, filename=str(out))
    assert out.exists()
    assert result.physics.nvol == 4


def test_no_file_written_without_filename(tmp_path):
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=4)
    assert result.physics.nvol == 4
    assert not any(tmp_path.iterdir())  # nothing written


def test_tfluxes_and_nvol_mutually_exclusive():
    vmec = Vmec(str(WOUT_IOTA))
    with pytest.raises(RuntimeError):
        simsvmec2spectre(vmec)


def test_boundary_modes_present(tmp_path):
    """At least some RBC boundary modes should be non-zero."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, filename=str(tmp_path / "gen.toml"))
    assert np.any(result.physics.rbc != 0.0)


# ---------------------------------------------------------------------------
# Interface guesses and angle fix
# ---------------------------------------------------------------------------


def test_interface_guesses_set_when_wout():
    """Allrzrz should be populated when interface_guess='wout'."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, interface_guess="wout")
    assert result.physics.allrzrz is not None


def test_no_interface_guesses_by_default():
    """Allrzrz should be None by default (interface_guess=None)."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3)
    assert result.physics.allrzrz is None


def test_linitialize_0_when_guesses_provided():
    """Linitialize must be 0 when interface guesses are supplied."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, interface_guess="wout")
    assert result.numeric.linitialize == 0


def test_linitialize_1_without_guesses():
    """Linitialize must be 1 when no interface guesses (default)."""
    vmec = Vmec(str(INPUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3)
    assert result.numeric.linitialize == 1


def test_wout_guess_from_indata_raises():
    """interface_guess='wout' must raise when no wout is available."""
    vmec = Vmec(str(INPUT_IOTA))
    with pytest.raises(RuntimeError, match="wout"):
        simsvmec2spectre(vmec, nvol=3, interface_guess="wout")


def test_invalid_interface_guess_raises():
    """interface_guess with an unknown value must raise ValueError."""
    vmec = Vmec(str(WOUT_IOTA))
    with pytest.raises(ValueError, match="interface_guess"):
        simsvmec2spectre(vmec, nvol=3, interface_guess="cubic")


def test_angle_fix_flips_boundary_n_sign():
    """With angle_fix=True, an m>0, n>0 mode appears at (m, -n) in the output."""
    vmec = Vmec(str(WOUT_IOTA))
    rbc_fixed, _ = vmec_boundary_to_fourier_coeffs(vmec.boundary, angle_fix=True)
    rbc_raw, _ = vmec_boundary_to_fourier_coeffs(vmec.boundary, angle_fix=False)
    for key, val in rbc_raw.items():
        m, n = [int(x) for x in key.strip("()").split(",")]
        if m > 0 and n > 0 and val != 0.0:
            assert f"({m}, {-n})" in rbc_fixed
            assert rbc_fixed[f"({m}, {-n})"] == pytest.approx(val)
            break


def test_compute_interface_guesses_shape():
    """compute_interface_guesses returns one entry per interface."""
    vmec = Vmec(str(WOUT_IOTA))
    nvol = 4
    tfluxes = np.linspace(0.0, 1.0, nvol + 1, endpoint=True)[1:]
    guesses = compute_interface_guesses(vmec.wout, tfluxes, mpol=4, ntor=3)
    assert len(guesses) == nvol
    assert all(f"interface_{i}" in guesses for i in range(1, nvol + 1))
    assert all("rbc" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))


def test_interface_guesses_include_m0_zbs():
    """zbs[(0, n)] modes must appear when zmns(0, n) is non-zero in the wout."""
    vmec = Vmec(str(WOUT_IOTA))
    wout = vmec.wout
    nfp = int(wout.nfp)
    # wout_iota.nc is a 3D stellarator (nfp=2); zmns(0, 1) ≈ 1.3 m is non-zero.
    # Find the reduced index n for the first non-zero m=0 zmns mode.
    n_nonzero = None
    for j in range(len(wout.xm)):
        m = int(wout.xm[j])
        n = int(round(float(wout.xn[j]) / nfp))
        if m == 0 and n > 0 and np.max(np.abs(wout.zmns[j, :])) > 1e-10:
            n_nonzero = n
            break
    assert n_nonzero is not None, "test wout has no non-zero zmns(0,n) — update test data"
    tfluxes = np.array([0.5])
    guesses = compute_interface_guesses(wout, tfluxes, mpol=2, ntor=n_nonzero)
    assert f"(0, {n_nonzero})" in guesses["interface_1"]["zbs"]


# ---------------------------------------------------------------------------
# Physics: absolute toroidal flux at each interface matches VMEC
# ---------------------------------------------------------------------------
#
# Fortran normalizes tflux ← tflux / tflux(Nvol) on read, so the physical
# toroidal flux at interface i is  phiedge * tflux[i] / tflux[nvol-1].
# When tfluxes[-1] < 1, the SPECTRE plasma boundary geometry is replaced
# with the VMEC surface at that interior s-value; phiedge must scale to
# keep every interface's absolute flux at VMEC.phi[-1] * tfluxes[i],
# otherwise iota/oita/isurf — evaluated at s=tfluxes[i] — end up at
# the wrong physical-flux positions.


def _absolute_flux_at_interface(params, i):
    """phiedge * tflux[i] / tflux[nvol-1] — what Fortran sees after normalization."""
    phys = params.physics
    return phys.phiedge * phys.tflux[i] / phys.tflux[phys.nvol - 1]


def test_phiedge_unchanged_when_tfluxes_end_at_one():
    """Round-trip: tfluxes[-1]=1.0 leaves phiedge equal to VMEC.phi[-1]."""
    vmec = Vmec(str(WOUT_IOTA))
    tfluxes = np.array([0.25, 0.5, 0.75, 1.0])
    result = simsvmec2spectre(vmec, tfluxes=tfluxes, interface_guess="wout")
    np.testing.assert_allclose(result.physics.phiedge, vmec.wout.phi[-1], rtol=1e-12)


def test_phiedge_scaled_when_tfluxes_end_below_one():
    """When the plasma boundary sits inside VMEC, phiedge scales by tfluxes[-1]."""
    vmec = Vmec(str(WOUT_IOTA))
    tfluxes = np.array([0.2, 0.45, 0.75])
    result = simsvmec2spectre(vmec, tfluxes=tfluxes, interface_guess="wout")
    expected = vmec.wout.phi[-1] * tfluxes[-1]
    np.testing.assert_allclose(result.physics.phiedge, expected, rtol=1e-12)


@pytest.mark.parametrize(
    "tfluxes",
    [
        np.array([0.25, 0.5, 0.75, 1.0]),
        np.array([0.2, 0.45, 0.75]),
        np.array([0.3, 0.6, 0.9]),
    ],
)
def test_interface_absolute_flux_matches_vmec(tfluxes):
    """Each interface's absolute toroidal flux must equal VMEC.phi[-1] * tfluxes[i]."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, tfluxes=tfluxes, interface_guess="wout")
    nvol = len(tfluxes)
    for i in range(nvol):
        absolute_flux = _absolute_flux_at_interface(result, i)
        expected = vmec.wout.phi[-1] * tfluxes[i]
        np.testing.assert_allclose(
            absolute_flux,
            expected,
            rtol=1e-10,
            err_msg=f"interface {i} (tflux={tfluxes[i]}) absolute flux mismatch",
        )


def test_iota_attached_to_correct_absolute_flux():
    """iota[i+1] must equal VMEC iota at the s-value where the interface physically sits.

    The interface i sits at absolute flux phiedge*tflux[i]/tflux[nvol-1]; in VMEC
    s-coordinates that is (phiedge*tflux[i]/tflux[nvol-1]) / VMEC.phi[-1].
    iota[i+1] in the SPECTRE input must equal VMEC_iota at that s.
    """
    vmec = Vmec(str(WOUT_IOTA))
    tfluxes = np.array([0.2, 0.45, 0.75])
    result = simsvmec2spectre(vmec, tfluxes=tfluxes, interface_guess="wout")
    iota_interp = make_iota_interpolator(vmec)
    nvol = len(tfluxes)
    for i in range(nvol):
        absolute_flux = _absolute_flux_at_interface(result, i)
        s_in_vmec = absolute_flux / vmec.wout.phi[-1]
        # iota is stored shifted: index 0 is the axis, index i+1 is interface i.
        np.testing.assert_allclose(
            result.physics.iota[i + 1],
            iota_interp(s_in_vmec),
            rtol=1e-10,
            err_msg=f"interface {i}: iota at wrong absolute flux position",
        )


@pytest.mark.slow
def test_spectre_runs_from_wout_iota():
    """Generated InputParameters from wout_iota.nc must produce a runnable SPECTRE object."""
    from spectre import SPECTRE

    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, mpol=4, ntor=3)
    spectre = SPECTRE.from_input_params(result, verbose=False)
    spectre.run()  # segfaults if geometry is invalid (e.g. negative volume)


# ---------------------------------------------------------------------------
# map2disc interface guesses
# ---------------------------------------------------------------------------

pytest.importorskip("map2disc")


def test_simsvmec2spectre_use_map2disc_from_indata():
    """interface_guess='map2disc' should produce allrzrz even from an indata source."""
    vmec = Vmec(str(INPUT_IOTA))
    result = simsvmec2spectre(
        vmec,
        nvol=2,
        mpol=2,
        ntor=1,
        interface_guess="map2disc",
        map2disc_kwargs={"ntheta": 32, "nzeta": 15, "bcm_m": 4},
    )
    assert result.physics.allrzrz is not None
    assert result.numeric.linitialize == 0


@pytest.mark.slow
def test_map2disc_interfaces_not_self_intersecting():
    """Interfaces generated by map2disc must not intersect each other in SPECTRE geometry."""
    from spectre import SPECTRE

    vmec = Vmec(str(INPUT_IOTA))
    params = simsvmec2spectre(
        vmec,
        nvol=2,
        mpol=2,
        ntor=1,
        interface_guess="map2disc",
        map2disc_kwargs={"ntheta": 32, "nzeta": 15, "bcm_m": 4},
    )
    spectre = SPECTRE.from_input_params(params, verbose=False)
    spectreout = spectre.create_spectreout()
    assert not spectreout.check_intersection_interfaces()
