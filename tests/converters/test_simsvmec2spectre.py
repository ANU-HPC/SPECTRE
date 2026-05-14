# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Integration tests for spectre.converters.simsvmec2spectre.

Requires simsopt; skipped automatically when simsopt is not installed.
Test data: examples/utils/vmec2spectre/{wout_iota.nc, input.iota, input.qabeta_curr}.
"""

from pathlib import Path

import numpy as np
import pytest

simsopt = pytest.importorskip("simsopt")
from simsopt.mhd import Vmec  # noqa: E402

from spectre.converters.simsvmec2spectre import simsvmec2spectre
from spectre.converters.vmec_helpers import (
    MU0,
    VmecPhysics,
    build_spectre_input,
    compute_interface_guesses,
    compute_iota_profile,
    compute_pressure_profile,
    vmec_boundary_to_fourier_coeffs,
)

EXAMPLES = Path(__file__).parents[2] / "examples" / "utils" / "vmec2spectre"
WOUT_IOTA = EXAMPLES / "wout_iota.nc"
INPUT_IOTA = EXAMPLES / "input.iota"
INPUT_CURR = EXAMPLES / "input.qabeta_curr"

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
        curtor=src.ctor, phiedge=src.phi[-1], pres_scale=src.presf[0],
        am=src.am, ai=src.ai, ac=src.ac, nfp=src.nfp,
        rbc=rbc, zbs=zbs,
        pcurr_type="power_series", profile_type="iota",
    )


@pytest.fixture
def current_physics():
    vmec = Vmec(str(INPUT_CURR))
    src = vmec.indata
    rbc, zbs = vmec_boundary_to_fourier_coeffs(vmec.boundary)
    return VmecPhysics(
        curtor=src.curtor, phiedge=src.phiedge, pres_scale=src.pres_scale,
        am=src.am, ai=src.ai, ac=src.ac, nfp=src.nfp,
        rbc=rbc, zbs=zbs,
        pcurr_type="power_series", profile_type="current",
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


def test_map2disc_interface_guesses_shape():
    """compute_interface_guesses_map2disc returns one entry per interface."""
    from spectre.converters.vmec_helpers import compute_interface_guesses_map2disc

    vmec = Vmec(str(WOUT_IOTA))
    rbc, zbs = vmec_boundary_to_fourier_coeffs(vmec.boundary, angle_fix=True)
    nvol = 3
    tfluxes = np.linspace(0.0, 1.0, nvol + 1, endpoint=True)[1:]
    guesses = compute_interface_guesses_map2disc(
        rbc, zbs, int(vmec.wout.nfp), tfluxes, mpol=4, ntor=3
    )
    assert len(guesses) == nvol
    assert all(f"interface_{i}" in guesses for i in range(1, nvol + 1))
    assert all("rbc" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))
    assert all("zbs" in guesses[f"interface_{i}"] for i in range(1, nvol + 1))


def test_simsvmec2spectre_use_map2disc_from_indata():
    """interface_guess='map2disc' should produce allrzrz even from an indata source."""
    vmec = Vmec(str(INPUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, interface_guess="map2disc")
    assert result.physics.allrzrz is not None
    assert result.numeric.linitialize == 0


def test_simsvmec2spectre_use_map2disc_from_wout():
    """interface_guess='map2disc' should produce allrzrz from a wout source."""
    vmec = Vmec(str(WOUT_IOTA))
    result = simsvmec2spectre(vmec, nvol=3, interface_guess="map2disc")
    assert result.physics.allrzrz is not None
    assert result.numeric.linitialize == 0
