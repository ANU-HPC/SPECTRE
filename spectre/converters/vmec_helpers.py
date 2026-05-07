# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""VMEC physics helpers and generic VMEC-to-SPECTRE builder.

Contains everything that is shared across VMEC adapters:

- Physical constants and polynomial utilities (:func:`power_series`)
- Profile discretisation functions (pressure, iota, volume/sheet current)
- Boundary Fourier-coefficient extraction from a simsopt surface object
- :class:`VmecPhysics` – the common data structure that every VMEC adapter
  must produce
- :func:`build_spectre_input` – turns a :class:`VmecPhysics` into a validated
  SPECTRE :class:`~spectre.file_io.input_parameters.InputParameters` object

Adding a new VMEC interface
---------------------------
Write a new adapter module that imports :class:`VmecPhysics` from here and
implements its own extraction function::

    def extract_<name>_vmec_physics(source) -> VmecPhysics:
        ...
        return VmecPhysics(
            curtor=..., phiedge=..., pres_scale=...,
            am=..., ai=..., ac=..., nfp=...,
            rbc={...}, zbs={...},
            pcurr_type='power_series',
            profile_type='iota',   # or 'current'
        )

Then call :func:`build_spectre_input` directly — no other file in this package
needs to change.
"""

from typing import NamedTuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

from spectre.input_dict import input_dict
from spectre.file_io.toml_io import input_parameters_from_dict

# Permeability of free space [H/m]; used to convert pressures and currents to
# the internal SPECTRE units (mu0 * SI value).
MU0 = 4 * np.pi * 1e-7


# ---------------------------------------------------------------------------
# Polynomial profile utilities
# ---------------------------------------------------------------------------


def power_series(coeffs):
    """Return a callable that evaluates ``sum_i coeffs[i] * s**i``.

    VMEC stores pressure, iota, and current profiles as power-series
    coefficients in normalised toroidal flux s ∈ [0, 1].

    Args:
        coeffs: sequence of polynomial coefficients ordered from degree 0.

    Returns:
        Callable ``f(s)`` that accepts scalars or arrays.
    """
    def f(s):
        # Build each monomial term separately, then sum along the coefficient
        # axis so the result shape matches s for both scalar and array inputs.
        return np.sum([coeffs[i] * s**i for i in range(len(coeffs))], axis=0)
    return f


def compute_pressure_profile(am, pres_scale, tfluxes):
    """Volume-averaged mu0*p in each sub-volume between consecutive interfaces.

    SPECTRE stores a single pressure value per volume.  We obtain it by
    integrating the continuous VMEC pressure profile p(s) over the s-interval
    [s_{i-1}, s_i] of each volume and dividing by the interval width, giving
    the volume average.  The result is multiplied by mu0 to match SPECTRE's
    internal units.

    Args:
        am: VMEC pressure polynomial coefficients (array-like).
        pres_scale: VMEC pressure scale factor (multiplies the polynomial).
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].

    Returns:
        1-D array of length ``len(tfluxes)`` with mu0 * <p> per volume.
    """
    nvol = len(tfluxes)
    # Prepend s=0 so that volume i spans [tfluxes_ext[i], tfluxes_ext[i+1]].
    tfluxes_ext = np.concatenate([[0.0], tfluxes])
    pres_func = power_series(am)

    profile = np.zeros(nvol)
    for i in range(nvol):
        a, b = tfluxes_ext[i], tfluxes_ext[i + 1]
        # Integrate p(s) over the volume interval, then normalise to get <p>.
        profile[i] = MU0 * pres_scale * quad(pres_func, a, b)[0] / (b - a)
    return profile


def compute_iota_profile(ai, tfluxes):
    """Evaluate the VMEC rotational transform at each interface location.

    SPECTRE's iota constraint stores iota directly on the interfaces, so we
    simply evaluate the VMEC polynomial at each interface s-value.

    Args:
        ai: VMEC iota polynomial coefficients (array-like).
        tfluxes: 1-D array of interface toroidal-flux values.

    Returns:
        1-D array of iota values, one per interface.
    """
    return power_series(ai)(tfluxes)


def compute_volume_current_profile(ac, curtor, tfluxes):
    """Cumulative mu0*I through each interface, normalised to match VMEC curtor.

    VMEC stores the current-density profile as j(s) = sum_i ac[i] * s**i.
    We integrate j over each sub-volume to get the incremental current, then
    accumulate across volumes to obtain the total enclosed current up to each
    interface.  The sign and scale are chosen so that the outermost interface
    carries exactly mu0 * curtor.

    Args:
        ac: VMEC current-density polynomial coefficients (array-like).
        curtor: total toroidal current from VMEC [A].
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].

    Returns:
        1-D array of length ``len(tfluxes)`` with mu0 * I_enclosed per interface.
    """
    nvol = len(tfluxes)
    current_density_func = power_series(ac)
    profile = np.zeros(nvol)

    # Integrate j(s) over each sub-volume.  The outermost volume extends to
    # s=1 (full plasma) rather than stopping at the last interface.
    for i in range(nvol):
        a = 0.0 if i == 0 else tfluxes[i - 1]
        b = 1.0 if i == (nvol - 1) else tfluxes[i]
        profile[i] = quad(current_density_func, a, b)[0]

    # Accumulate: profile[i] should be the total current inside interface i.
    for i in range(1, nvol):
        profile[i] += profile[i - 1]

    # Apply sign convention (SPECTRE uses -curtor) and convert to SPECTRE units.
    profile *= -curtor * MU0
    return profile


def compute_sheet_current_profile(ac, curtor, tfluxes):
    """Sheet-current mu0*K at each interface, normalised to match VMEC curtor.

    In sheet-current mode, each interface carries a surface current K.  We
    approximate K_i by evaluating the VMEC current-density j at s_i, then
    rescale the whole array so that the discrete sum reproduces curtor.  The
    final entry is zero by convention — no sheet current sits on the plasma
    boundary itself.

    Args:
        ac: VMEC current-density polynomial coefficients (array-like).
        curtor: total toroidal current from VMEC [A].
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].

    Returns:
        1-D array of length ``len(tfluxes)`` with mu0 * K per interface.
    """
    nvol = len(tfluxes)
    current_density_func = power_series(ac)
    profile = np.zeros(nvol)

    # Evaluate j at each interior interface; the last slot stays zero.
    profile[:-1] = current_density_func(tfluxes[:-1])

    # Normalise so the discrete sum equals curtor, preserving the profile shape.
    total = np.sum(profile)
    if total != 0.0:
        profile *= curtor / total

    # Sign convention and unit conversion.
    profile *= -MU0
    return profile


# ---------------------------------------------------------------------------
# Boundary Fourier-coefficient extraction (simsopt surface)
# ---------------------------------------------------------------------------


def vmec_boundary_to_fourier_coeffs(boundary):
    """Extract RBC and ZBS Fourier coefficients from a simsopt boundary surface.

    A simsopt ``SurfaceRZFourier`` stores the full (m, n) mode spectrum
    including negative-m modes.  We iterate over the positive-m half
    (num_modes = len(boundary.m) // 2 + 1) and read each cosine (R) and
    sine (Z) coefficient via the surface's accessor methods.

    Args:
        boundary: simsopt ``SurfaceRZFourier`` object (e.g. ``vmec.boundary``).

    Returns:
        Tuple ``(rbc, zbs)`` where each is a dict keyed by ``'(m, n)'`` strings
        mapping to the corresponding Fourier coefficient.
    """
    # The mode arrays contain both ±m; only the first half are unique.
    num_modes = len(boundary.m) // 2 + 1
    m_modes = boundary.m[:num_modes]
    n_modes = boundary.n[:num_modes]

    rbc = {}
    zbs = {}
    for m, n in zip(m_modes, n_modes):
        rbc[f"({m}, {n})"] = boundary.get_rc(m, n)
        zbs[f"({m}, {n})"] = boundary.get_zs(m, n)

    return rbc, zbs


# ---------------------------------------------------------------------------
# Common data structure: VmecPhysics
# ---------------------------------------------------------------------------


class VmecPhysics(NamedTuple):
    """SPECTRE-relevant physics extracted from a VMEC equilibrium.

    This is the contract between a VMEC adapter and :func:`build_spectre_input`.
    Every adapter must populate all fields; the builder has no knowledge of
    where the data came from.

    Attributes:
        curtor: total toroidal current [A].
        phiedge: toroidal magnetic flux at the plasma boundary [Wb].
        pres_scale: VMEC pressure normalisation factor.
        am: pressure polynomial coefficients (power series in s).
        ai: rotational-transform polynomial coefficients (power series in s).
        ac: current-density polynomial coefficients (power series in s).
        nfp: number of toroidal field periods.
        rbc: cosine boundary coefficients as ``{'(m, n)': value}`` dict.
        zbs: sine boundary coefficients as ``{'(m, n)': value}`` dict.
        pcurr_type: VMEC current-profile type string (must be
            ``'power_series'``).
        profile_type: ``'iota'`` for a rotational-transform constraint,
            ``'current'`` for an integrated-current constraint.
    """

    curtor: float
    phiedge: float
    pres_scale: float
    am: np.ndarray
    ai: np.ndarray
    ac: np.ndarray
    nfp: int
    rbc: dict
    zbs: dict
    pcurr_type: str
    profile_type: str


# ---------------------------------------------------------------------------
# Generic SPECTRE input builder
# ---------------------------------------------------------------------------


def build_spectre_input(physics, tfluxes, mpol, ntor, current_mode):
    """Assemble a SPECTRE :class:`InputParameters` object from VMEC physics.

    This function has no dependency on any VMEC interface library; it operates
    purely on :class:`VmecPhysics` and standard numpy arrays.

    Args:
        physics: :class:`VmecPhysics` named tuple.
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        mpol: poloidal mode resolution for SPECTRE.
        ntor: toroidal mode resolution for SPECTRE.
        current_mode: ``'volume'`` or ``'sheet'``; only used when
            ``physics.profile_type == 'current'``.

    Returns:
        Tuple ``(dict_pydantic, profiles)`` where ``profiles`` is a dict of
        intermediate arrays (``'pressure'``, ``'iota'``, ``'volume_current'``,
        or ``'sheet_current'``) useful for plotting or inspection.

    Raises:
        RuntimeError: if ``profile_type`` or ``current_mode`` is unrecognised.
    """
    nvol = len(tfluxes)

    pressure_profile = compute_pressure_profile(physics.am, physics.pres_scale, tfluxes)
    # Collect intermediate profiles so the caller can inspect or plot them.
    profiles = {"pressure": pressure_profile}

    dict_physics = input_dict()
    dict_numeric = input_dict()

    # --- global geometry and resolution ---
    dict_physics.nfp = physics.nfp
    dict_physics.nvol = nvol
    dict_physics.mpol = mpol
    dict_physics.ntor = ntor
    dict_physics.phiedge = physics.phiedge

    # --- interface positions and pressure ---
    dict_physics.tflux = tfluxes
    dict_physics.pressure = pressure_profile
    dict_physics.pscale = 1.0  # pressure is already in SPECTRE units

    # lrad controls the radial Chebyshev resolution in each volume; a modest
    # excess over mpol is standard practice.
    dict_physics.lrad = np.array([mpol + 4] * nvol)

    # --- plasma boundary Fourier modes ---
    dict_physics.rbc = physics.rbc
    dict_physics.zbs = physics.zbs

    # linitialize=1 tells SPECTRE to linearly interpolate the initial RZ_mn
    # guess across volumes rather than using the boundary as a flat initial
    # condition for all volumes.
    dict_numeric.linitialize = 1

    if physics.profile_type == "iota":
        iota_profile = compute_iota_profile(physics.ai, tfluxes)
        profiles["iota"] = iota_profile

        # lconstraint=1: SPECTRE fixes iota on interfaces (Taylor-state volumes).
        dict_physics.lconstraint = 1
        # The iota array has nvol+1 entries; index 0 is the magnetic axis value,
        # which we set to zero (SPECTRE extrapolates it internally).
        dict_physics.iota = np.concatenate([[0.0], iota_profile])
        # oita mirrors iota; both must be consistent.
        dict_physics.oita = dict_physics.iota

    elif physics.profile_type == "current":
        # lconstraint=3: SPECTRE fixes total enclosed current on interfaces.
        dict_physics.lconstraint = 3

        if current_mode == "volume":
            volume_current_profile = compute_volume_current_profile(
                physics.ac, physics.curtor, tfluxes
            )
            profiles["volume_current"] = volume_current_profile
            dict_physics.ivolume = volume_current_profile
            dict_physics.isurf = np.zeros(nvol)   # no sheet currents in this mode

        elif current_mode == "sheet":
            sheet_current_profile = compute_sheet_current_profile(
                physics.ac, physics.curtor, tfluxes
            )
            profiles["sheet_current"] = sheet_current_profile
            dict_physics.isurf = sheet_current_profile
            dict_physics.ivolume = np.zeros(nvol)  # no volume currents in this mode

        else:
            raise RuntimeError(
                f"Unrecognised current_mode '{current_mode}'; expected 'volume' or 'sheet'."
            )

    else:
        raise RuntimeError(
            f"Unrecognised profile_type '{physics.profile_type}'; expected 'iota' or 'current'."
        )

    dict_total = dict(physics=dict_physics, numeric=dict_numeric)
    dict_pydantic = input_parameters_from_dict(dict_total)

    return dict_pydantic, profiles


# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------


def plot_vmec2spectre(
    tfluxes,
    pressure_func,
    pressure_profile,
    profile_type,
    iota_func=None,
    iota_profile=None,
    current_density_func=None,
    current_mode=None,
    volume_current_profile=None,
    sheet_current_profile=None,
    curtor=None,
):
    """Plot VMEC continuous profiles alongside the SPECTRE discretised values.

    Produces a two-panel figure: pressure on the left and either rotational
    transform or current on the right.  Vertical lines mark the interface
    positions.

    Args:
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        pressure_func: callable ``p(s)`` returning the VMEC pressure profile.
        pressure_profile: 1-D array of mu0*<p> per volume (SPECTRE values).
        profile_type: ``'iota'`` or ``'current'``.
        iota_func: callable ``iota(s)``; required when ``profile_type='iota'``.
        iota_profile: 1-D array of iota values at interfaces.
        current_density_func: callable ``j(s)``; required for current profiles.
        current_mode: ``'volume'`` or ``'sheet'``; required for current profiles.
        volume_current_profile: cumulative mu0*I array (volume-current mode).
        sheet_current_profile: mu0*K array (sheet-current mode).
        curtor: total toroidal current [A]; used to detect the zero-current case.

    Returns:
        ``(fig, ax)`` matplotlib figure and axes array of shape ``(2,)``.
    """
    # Dense s-grid for plotting the continuous VMEC profile.
    s = np.linspace(0.0, 1.0, 256)
    # Extended tflux array with s=0 prepended, used for the pressure hlines.
    tfluxes_ext = np.concatenate([[0.0], tfluxes])

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))

    # --- left panel: pressure ---
    ax[0].plot(s, pressure_func(s), label="VMEC")
    # Show the volume-averaged SPECTRE pressure as horizontal segments.
    # Divide by MU0 to plot in the same units as the continuous profile.
    ax[0].hlines(pressure_profile / MU0, tfluxes_ext[1:], tfluxes_ext[:-1], "red", lw=2)
    ax[0].set_title("Pressure")

    # --- right panel: rotational transform or current ---
    if profile_type == "iota":
        ax[1].plot(s, iota_func(s))
        # Discrete iota values that SPECTRE will enforce on each interface.
        ax[1].plot(tfluxes, iota_profile, "ro")
        ax[1].set_title("Rotational transform")

    elif profile_type == "current":
        if current_mode == "volume":
            # Plot the cumulative integral of j(s), which represents the
            # enclosed current as a function of s.
            integrated = [quad(current_density_func, 0.0, lim)[0] for lim in s]
            ax[1].plot(s, integrated)
            # Normalise the discrete values to the same scale for comparison.
            ax[1].plot(tfluxes, volume_current_profile / np.max(volume_current_profile), "ro")
            ax[1].set_title("Integrated volume current (normalized)")

        elif current_mode == "sheet":
            j_continuous = current_density_func(s)
            # Normalise to unit amplitude so the profile shape is clear; skip
            # normalisation if curtor is zero to avoid division by zero.
            j_max = np.max(j_continuous) if curtor != 0.0 else 1.0
            ax[1].plot(s, j_continuous / j_max if curtor != 0.0 else j_continuous)

            # Show the sampled j values at interior interface positions.
            j_at_interfaces = current_density_func(tfluxes[:-1])
            ax[1].plot(
                tfluxes[:-1],
                j_at_interfaces / j_max if curtor != 0.0 else j_at_interfaces,
                "ro",
            )
            ax[1].set_title("Sheet current (normalized)")

    # --- shared formatting ---
    for a in ax:
        # Mark interface positions as vertical lines on both panels.
        a.vlines(tfluxes, *a.get_ylim(), color="k", lw=0.9, label="Interfaces")
        a.set_xlabel("s / tflux")
        a.autoscale(enable=None, axis="y", tight=True)
        a.autoscale(enable=None, axis="x", tight=False)

    ax[0].legend(["VMEC", "SPECTRE", "Interfaces"])
    plt.tight_layout(w_pad=2)

    return fig, ax
