# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

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
from scipy.integrate import quad
from scipy.interpolate import interp1d

from spectre.file_io.toml_io import input_parameters_from_dict
from spectre.input_dict import input_dict

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


def vmec_boundary_to_fourier_coeffs(boundary, angle_fix=True):
    """Extract RBC and ZBS Fourier coefficients from a simsopt boundary surface.

    A simsopt ``SurfaceRZFourier`` stores the full (m, n) mode spectrum
    including negative-m modes.  We iterate over the positive-m half
    (num_modes = len(boundary.m) // 2 + 1) and read each cosine (R) and
    sine (Z) coefficient via the surface's accessor methods.

    Args:
        boundary: simsopt ``SurfaceRZFourier`` object (e.g. ``vmec.boundary``).
        angle_fix: if ``True`` (default), apply the θ→-θ handedness correction
            for m>0 modes: the toroidal index is negated (n→-n) and the ZBS
            coefficient is negated.  This pre-applies what ``lchangeangle=True``
            would do at Fortran runtime, so the output TOML needs no
            ``lchangeangle`` flag.  Set to ``False`` to keep the raw VMEC
            convention.

    Returns:
        Tuple ``(rbc, zbs)`` where each is a dict keyed by ``'(m, n)'`` strings
        mapping to the corresponding Fourier coefficient.
    """
    rbc = {}
    zbs = {}
    # boundary.m/n list every (±m, n) pair; skip m<0 (redundant by stellarator symmetry)
    # and (m=0, n<0) which SPECTRE does not store.
    for m, n in zip(boundary.m, boundary.n):  # noqa: B905
        if m < 0 or (m == 0 and n < 0):
            continue
        rval = boundary.get_rc(m, n)
        zval = boundary.get_zs(m, n)
        if angle_fix and m > 0:
            # θ→-θ: n → -n; cos(mθ-nφ)→cos(mθ+nφ)=cos(mθ-(-n)φ), sin negated
            rbc[f"({m}, {-n})"] = rval
            zbs[f"({m}, {-n})"] = -zval
        else:
            rbc[f"({m}, {n})"] = rval
            zbs[f"({m}, {n})"] = zval

    return rbc, zbs


# ---------------------------------------------------------------------------
# Interface geometry guesses from VMEC wout
# ---------------------------------------------------------------------------


def compute_interface_guesses(wout, tfluxes, mpol, ntor):
    """Interpolate VMEC wout surfaces at each interface position.

    Reads ``rmnc``/``zmns`` from the simsopt wout object, interpolates
    cubically on the VMEC radial grid, and evaluates at each value in
    ``tfluxes``.  The θ→-θ angle fix is applied (n→-n for m>0, ZBS negated)
    so the returned coefficients are in SPECTRE's right-handed coordinate
    convention — identical to what ``lchangeangle=True`` would produce at
    Fortran runtime.

    Only handles stellarator-symmetric equilibria (``lasym=False``).

    Args:
        wout: simsopt wout object (``vmec_obj.wout``).
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        mpol: maximum poloidal mode number to include in the output.
        ntor: maximum toroidal mode number (reduced, without nfp) to include.

    Returns:
        dict suitable for ``PhysicsParameters.allrzrz``:
        ``{'interface_1': {'rbc': {...}, 'zbs': {...}, 'rbs': {}, 'zbc': {}},
        ...}``
    """
    ns = wout.ns
    nfp = int(wout.nfp)
    xm = wout.xm        # shape (num_modes,), integer m values
    xn = wout.xn        # shape (num_modes,), n * nfp (full toroidal index)
    rmnc = wout.rmnc    # shape (ns, num_modes), R cosine coefficients
    zmns = wout.zmns    # shape (ns, num_modes), Z sine coefficients

    # rmnc/zmns are stored as (num_modes, ns) in the simsopt wout, so
    # the radial grid axis is axis=1.
    s_grid = np.linspace(0.0, 1.0, ns)
    rmnc_interp = interp1d(s_grid, rmnc, kind="cubic", axis=1)
    zmns_interp = interp1d(s_grid, zmns, kind="cubic", axis=1)

    allrzrz = {}
    for ivol, s in enumerate(tfluxes, 1):
        rbc_vals = rmnc_interp(float(s))
        zbs_vals = zmns_interp(float(s))

        rbc, zbs = {}, {}
        for j in range(len(xm)):
            m = int(xm[j])
            n = int(round(float(xn[j]) / nfp))  # reduced toroidal index
            if m > mpol or abs(n) > ntor:
                continue

            rval = float(rbc_vals[j])
            zval = float(zbs_vals[j])

            if m == 0:
                # cos(0·θ - n·nfp·φ) is independent of θ, so no angle fix needed
                if n < 0:
                    continue  # SPECTRE convention: skip (0, n<0)
                if rval != 0.0:
                    rbc[f"(0, {n})"] = rval
            else:
                # Apply θ→-θ angle fix: n → -n, ZBS negated
                n_out = -n
                if rval != 0.0:
                    rbc[f"({m}, {n_out})"] = rval
                if zval != 0.0:
                    zbs[f"({m}, {n_out})"] = -zval

        allrzrz[f"interface_{ivol}"] = {"rbc": rbc, "zbs": zbs, "rbs": {}, "zbc": {}}

    return allrzrz


def compute_interface_guesses_map2disc(
    rbc, zbs, nfp, tfluxes, mpol, ntor, ntheta=124, nzeta=78, bcm_m=6
):
    """Generate interface geometry guesses using conformal mapping (map2disc).

    Uses the ``map2disc`` BCM solver to conformally map the plasma boundary
    inward to each interface position, producing geometrically smooth nested
    surfaces.  Only the boundary Fourier harmonics are required — no VMEC wout
    or SPECTRE object is needed — so this works from both ``input.*`` and
    ``wout_*.nc`` sources.

    The ``rbc`` / ``zbs`` dicts must be in SPECTRE convention (angle-fix applied,
    i.e. the output of :func:`vmec_boundary_to_fourier_coeffs` with
    ``angle_fix=True``).  The same sign convention (``th_arr = -th_arr``) as
    :func:`~spectre.mesh_gen.gen_infaces` is used internally so the FFT output
    is directly compatible with ``allrzrz``.

    Args:
        rbc: boundary R cosine harmonics as ``{'(m, n)': value}`` dict
            (SPECTRE convention, angle-fix applied).
        zbs: boundary Z sine harmonics as ``{'(m, n)': value}`` dict
            (SPECTRE convention, angle-fix applied).
        nfp: number of toroidal field periods.
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        mpol: maximum poloidal mode number for the output.
        ntor: maximum toroidal mode number (reduced, without nfp) for the output.
        ntheta: number of poloidal quadrature points (default 124).
        nzeta: number of toroidal quadrature points per field period (default 78).
        bcm_m: number of Fourier modes used by BCM (default 6).

    Returns:
        dict suitable for ``PhysicsParameters.allrzrz``:
        ``{'interface_1': {'rbc': {...}, 'zbs': {...}, 'rbs': {...}, 'zbc': {...}},
        ...}``

    Raises:
        ImportError: if ``map2disc`` is not installed.
    """
    try:
        from map2disc import BCM
    except ImportError as exc:
        raise ImportError(
            "map2disc is required for interface_guess='map2disc'. "
            "Install it with: pip install map2disc"
        ) from exc

    from spectre.fourier import spec_fft_fast

    # Pre-parse the dict keys once to avoid repeated string splitting inside the zeta loop.
    # Each entry is (m, n, coefficient) where m is poloidal and n is the reduced toroidal index.
    rbc_modes = [
        (int(k.strip("()").split(",")[0]), int(k.strip("()").split(",")[1]), v)
        for k, v in rbc.items()
    ]
    zbs_modes = [
        (int(k.strip("()").split(",")[0]), int(k.strip("()").split(",")[1]), v)
        for k, v in zbs.items()
    ]

    num_ifaces = len(tfluxes)
    # BCM works in the unit disc with radial coordinate rho = sqrt(tflux).
    rho_arr = np.sqrt(np.asarray(tfluxes, dtype=float))

    # Negative poloidal angle: SPECTRE's internal θ sign convention (same as gen_infaces).
    # The FFT in spec_fft_fast expects this orientation to produce the correct even/odd split.
    th_arr = -np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    # One toroidal field period is sufficient; BCM solves each ζ slice independently.
    zeta_arr = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)

    R_arr = np.zeros((num_ifaces, nzeta, ntheta))
    Z_arr = np.zeros_like(R_arr)

    # Outer loop: iterate over toroidal cross-sections (ζ slices).
    # BCM is a 2-D conformal map that operates on a single poloidal cross-section at a time,
    # so the full 3-D geometry is built up one ζ value at a time.
    for iz, zeta in enumerate(zeta_arr):
        # _boundary evaluates the plasma boundary curve at the fixed toroidal angle zeta.
        # It sums the Fourier harmonics R = Σ rbc(m,n) cos(mθ - n·nfp·ζ) and similarly for Z.
        # The default-argument capture (_zeta=float(zeta)) freezes the loop variable.
        def _boundary(theta, _zeta=float(zeta)):
            th = np.asarray(theta, dtype=float)
            R = np.zeros_like(th)
            Z = np.zeros_like(th)
            for m, n, v in rbc_modes:
                R += v * np.cos(m * th - n * nfp * _zeta)
            for m, n, v in zbs_modes:
                Z += v * np.sin(m * th - n * nfp * _zeta)
            return R, Z

        # BCM conformally maps the unit disc to the cross-section defined by _boundary.
        # After solve(), eval_rt_1d returns (R, Z) on a 2-D grid of (rho, theta) points,
        # giving all interfaces at this ζ simultaneously.
        bcm = BCM(_boundary, M=bcm_m)
        bcm.solve()
        x, y = bcm.eval_rt_1d(rho_arr, th_arr)
        R_arr[:, iz, :] = x   # shape: (num_ifaces, ntheta)
        Z_arr[:, iz, :] = y

    # FFT step: decompose R and Z into Fourier harmonics over (θ, ζ).
    # spec_fft_fast expects shape (..., nzeta, ntheta) and returns (..., nmn).
    # R_even/Z_odd are the cos/sin harmonics (RBC/ZBS); R_odd/Z_even are the sin/cos (RBS/ZBC).
    R_even, R_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, R_arr, mpol, ntor)
    Z_even, Z_odd,  _,   _  = spec_fft_fast(th_arr, zeta_arr, Z_arr, mpol, ntor)
    # in_ is the reduced toroidal index (n, not n·nfp) — use it directly in the key strings.
    nmn = len(im_)

    # Pack into the allrzrz dict format expected by PhysicsParameters.
    # One entry per interface, each containing the four Fourier coefficient dicts.
    allrzrz = {}
    for ivol in range(num_ifaces):
        allrzrz[f"interface_{ivol + 1}"] = {
            "rbc": {f"({im_[j]}, {in_[j]})": float(R_even[ivol, j]) for j in range(nmn)},
            "zbs": {f"({im_[j]}, {in_[j]})": float(Z_odd [ivol, j]) for j in range(nmn)},
            "rbs": {f"({im_[j]}, {in_[j]})": float(R_odd [ivol, j]) for j in range(nmn)},
            "zbc": {f"({im_[j]}, {in_[j]})": float(Z_even[ivol, j]) for j in range(nmn)},
        }

    return allrzrz


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


def build_spectre_input(physics, tfluxes, mpol, ntor, current_mode, interface_guesses=None):
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
        interface_guesses: optional dict in ``allrzrz`` format returned by
            :func:`compute_interface_guesses`.  When provided, it is stored as
            the initial geometry guess and ``linitialize`` is set to 0 (use
            supplied surfaces).  When ``None``, ``linitialize=1`` causes SPECTRE
            to linearly interpolate an initial guess from the boundary.

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

    if interface_guesses is not None:
        dict_physics.allrzrz = interface_guesses
        dict_numeric.linitialize = 0  # use the supplied surface guesses
    else:
        dict_numeric.linitialize = 1  # linearly interpolate from boundary

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


