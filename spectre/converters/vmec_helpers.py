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


def make_iota_interpolator(vmec_obj):
    """Return a callable that evaluates iota at any normalised toroidal flux.

    Dispatches to the appropriate backend depending on what the simsopt
    ``Vmec`` object exposes:

    - **wout source**: cubic interpolation of the ``iotaf`` full-mesh grid.
      Use this when high-fidelity iota values from a converged VMEC run are
      available; the polynomial fit may not reproduce ``iotaf`` exactly.
    - **indata source**: polynomial evaluation via :func:`power_series` using
      the ``ai`` coefficients.  This matches VMEC's own profile evaluation.

    Args:
        vmec_obj: simsopt ``Vmec`` instance (loaded from either ``input.*``
            or ``wout_*.nc``).

    Returns:
        Callable ``f(s)`` that accepts scalars or arrays of normalised
        toroidal flux s ∈ [0, 1] and returns the corresponding iota value(s).

    Raises:
        RuntimeError: if ``vmec_obj`` has neither ``indata`` nor ``wout``.
    """
    if hasattr(vmec_obj, "wout") and hasattr(vmec_obj.wout, "iotaf"):
        iotaf = vmec_obj.wout.iotaf
        ns = len(iotaf)
        s_grid = np.linspace(0.0, 1.0, ns)
        return interp1d(s_grid, iotaf, kind="cubic")
    elif hasattr(vmec_obj, "indata"):
        return power_series(vmec_obj.indata.ai)
    else:
        raise RuntimeError("vmec_obj must have either an 'indata' or 'wout' attribute.")


def compute_iota_profile_from_iotaf(iotaf, tfluxes):
    """Evaluate the VMEC full-mesh iota profile at each interface location.

    Provided as a convenience parallel to :func:`compute_iota_profile` for
    polynomial sources.  Operates directly on the raw ``iotaf`` array rather
    than a simsopt object; use :func:`make_iota_interpolator` for the
    user-facing API.

    Args:
        iotaf: 1-D array of iota values on the VMEC full radial grid (ns points, s=0..1).
        tfluxes: 1-D array of interface toroidal-flux values.

    Returns:
        1-D array of iota values, one per interface.
    """
    ns = len(iotaf)
    s_grid = np.linspace(0.0, 1.0, ns)
    return interp1d(s_grid, iotaf, kind="cubic")(tfluxes)


def normalized_enclosed_current(ac, s):
    """Normalised cumulative enclosed toroidal current Î(s) at flux value(s) ``s``.

    VMEC's ``power_series`` current profile stores ``ac`` as the *current
    density* j(s) = dI/ds = sum_i ac[i] * s**i.  The enclosed current is its
    integral I(s) = ∫₀ˢ j ds', and VMEC normalises so that I(1) corresponds to
    ``curtor``.  This function returns the dimensionless, normalised shape

        Î(s) = (∫₀ˢ ac ds') / (∫₀¹ ac ds')

    which runs from 0 at the axis to 1 at the plasma boundary.  Dividing by the
    total ∫₀¹ ac is essential: ``ac`` is only a *shape*, and its overall scale is
    arbitrary (it is commonly — but not always — normalised so that ∫₀¹ ac = 1).
    Omitting this division is what made the old volume-current path wrong by the
    factor ∫₀¹ ac.

    Args:
        ac: VMEC current-density polynomial coefficients (array-like).
        s: scalar or 1-D array of normalised toroidal-flux values.

    Returns:
        Î(s) with the same shape as ``s`` (scalar in → scalar out).  Returns all
        zeros when ∫₀¹ ac ds == 0 (e.g. a zero current profile).
    """
    j = power_series(ac)
    total = quad(j, 0.0, 1.0)[0]
    s_arr = np.atleast_1d(np.asarray(s, dtype=float))
    if total == 0.0:
        out = np.zeros_like(s_arr)
    else:
        out = np.array([quad(j, 0.0, float(si))[0] for si in s_arr]) / total
    return out.reshape(np.shape(s))


def distribute_current_to_interfaces(ac, curtor, tfluxes, vol_to_interface_ratio):
    """Map a continuous VMEC current profile onto SPECTRE interfaces.

    .. warning::

        **Provisional / not final.**  How to map a *continuous* (VMEC) current
        profile onto the *discrete* interfaces of an MRxMHD (SPEC/SPECTRE)
        equilibrium is a fundamental open question.  The control-volume scheme
        below is one physically-motivated choice; expect it to change.

    Two representations of the same enclosed current are produced and blended by
    ``vol_to_interface_ratio`` r (``1.0`` = all volume, ``0.0`` = all sheet),
    both in SPECTRE units of μ₀·A:

    - **Volume current** ``ivolume[i] = r · (−curtor·μ₀) · Î(sᵢ)`` — the
      *cumulative* enclosed current at interface i (Î from
      :func:`normalized_enclosed_current`).
    - **Sheet current** ``isurf[i] = (1−r) · (−curtor·μ₀) · (Î(eᵢ) − Î(eᵢ₋₁))`` —
      the current density integrated over the *control volume* owned by
      interface i, whose edges sit at the midpoints to its neighbours::

          e₀ = 0                     # magnetic axis
          eᵢ = (sᵢ + sᵢ₊₁)/2         # midpoint to the next interface
          e_N = s_N                  # outer edge of the last cell = last interface

      So each interface collects current from halfway to its inner neighbour to
      halfway to its outer neighbour — an *average of inside and outside*, not
      just the current of the volume just inside it.  For equidistant interfaces
      this reduces to the spot-evaluation j(sᵢ)·Δ; for non-equidistant
      interfaces a sparsely-spaced interface automatically owns a wider band and
      carries more of the surrounding current.

    Current is conserved: the control-volume increments telescope to Î(s_N), so
    ``ivolume[-1] + sum(isurf) = (−curtor·μ₀)·Î(s_N)`` for any r.  When the last
    interface is the true plasma boundary (s_N = 1) this is the full −curtor·μ₀;
    when tfluxes[-1] < 1 the total is reduced by the appropriate amount, because
    the current flowing beyond the (interior) SPECTRE boundary is not enclosed.

    The last interface carries ``K_N = (−curtor·μ₀)(Î(s_N) − Î(e_{N-1}))`` — the
    current from the last midpoint out to the boundary.  In a fixed-boundary run
    SPECTRE ignores ``Isurf(Mvol)``, but this value is preserved so that
    :func:`~spectre.converters.fix2free.fix2free` carries it over: when the
    plasma boundary is promoted to an interior interface in the free-boundary
    equilibrium, the slot becomes physical.

    Note: in a split (0 < r < 1) the sheet cumulative lands on the cell-edge
    midpoints while the volume cumulative lands on the interfaces, so the two
    pieces are not a trivial cumulative sum of one another — inherent to
    concentrating distributed current onto sheets.

    Args:
        ac: VMEC current-density polynomial coefficients (array-like).
        curtor: total toroidal current from VMEC [A].
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        vol_to_interface_ratio: fraction r ∈ [0, 1] carried as volume current;
            the remainder (1 − r) is carried as interface sheet current.

    Returns:
        Tuple ``(ivolume, isurf)`` of 1-D arrays of length ``len(tfluxes)``,
        both in μ₀·A.
    """
    s = np.asarray(tfluxes, dtype=float)
    nvol = len(s)
    r = vol_to_interface_ratio

    scale = -curtor * MU0

    # Volume current: cumulative enclosed current at each interface.
    enclosed_nodes = normalized_enclosed_current(ac, s)
    ivolume = r * scale * enclosed_nodes

    # Sheet current: control-volume increments between consecutive cell edges.
    edges = np.empty(nvol + 1)
    edges[0] = 0.0
    edges[1:nvol] = 0.5 * (s[:-1] + s[1:])
    edges[nvol] = s[-1]
    enclosed_edges = normalized_enclosed_current(ac, edges)
    isurf = (1.0 - r) * scale * np.diff(enclosed_edges)

    return ivolume, isurf


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
    xm = wout.xm  # shape (num_modes,), integer m values
    xn = wout.xn  # shape (num_modes,), n * nfp (full toroidal index)
    rmnc = wout.rmnc  # shape (ns, num_modes), R cosine coefficients
    zmns = wout.zmns  # shape (ns, num_modes), Z sine coefficients

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
                if zval != 0.0:
                    zbs[f"(0, {n})"] = zval
            else:
                # Apply θ→-θ angle fix: n → -n, ZBS negated
                n_out = -n
                if rval != 0.0:
                    rbc[f"({m}, {n_out})"] = rval
                if zval != 0.0:
                    zbs[f"({m}, {n_out})"] = -zval

        allrzrz[f"interface_{ivol}"] = {"rbc": rbc, "zbs": zbs, "rbs": {}, "zbc": {}}

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
        iotaf: iota on the VMEC full radial grid (ns points, s=0..1), or
            ``None`` when only polynomial coefficients are available (e.g.
            indata source).  When set, :func:`build_spectre_input` uses
            :func:`make_iota_interpolator` instead of the ``ai`` polynomial.
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
    iotaf: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Generic SPECTRE input builder
# ---------------------------------------------------------------------------


def build_spectre_input(
    physics, tfluxes, mpol, ntor, vol_to_interface_ratio=1.0, interface_guesses=None
):
    """Assemble a SPECTRE :class:`InputParameters` object from VMEC physics.

    This function has no dependency on any VMEC interface library; it operates
    purely on :class:`VmecPhysics` and standard numpy arrays.

    Args:
        physics: :class:`VmecPhysics` named tuple.
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        mpol: poloidal mode resolution for SPECTRE.
        ntor: toroidal mode resolution for SPECTRE.
        vol_to_interface_ratio: fraction r ∈ [0, 1] of the toroidal current
            represented as distributed volume current (``ivolume``); the
            remainder (1 − r) is placed on the interfaces as sheet current
            (``isurf``).  ``1.0`` (default) = all volume, ``0.0`` = all sheet.
            Only used when ``physics.profile_type == 'current'``.  See
            :func:`distribute_current_to_interfaces` for the (provisional)
            discretisation scheme.
        interface_guesses: optional dict in ``allrzrz`` format returned by
            :func:`compute_interface_guesses`.  When provided, it is stored as
            the initial geometry guess and ``linitialize`` is set to 0 (use
            supplied surfaces).  When ``None``, ``linitialize=1`` causes SPECTRE
            to linearly interpolate an initial guess from the boundary.

    Returns:
        Tuple ``(dict_pydantic, profiles)`` where ``profiles`` is a dict of
        intermediate arrays (``'pressure'``, ``'iota'``, ``'ivolume'``,
        ``'isurf'``) useful for plotting or inspection.

    Raises:
        ValueError: if ``vol_to_interface_ratio`` is outside [0, 1].
        RuntimeError: if ``profile_type`` is unrecognised.
    """
    if not 0.0 <= vol_to_interface_ratio <= 1.0:
        raise ValueError(f"vol_to_interface_ratio must be in [0, 1]; got {vol_to_interface_ratio}.")

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
    # When tfluxes[-1] < 1, the SPECTRE plasma boundary sits inside the VMEC
    # plasma surface; the physical toroidal flux at that location is
    # tfluxes[-1] * VMEC.phiedge. SPECTRE's phiedge is the absolute flux at
    # tflux(Nvol), so scale accordingly to keep all interfaces at their
    # correct absolute-flux positions.
    dict_physics.phiedge = physics.phiedge * float(tfluxes[-1])

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
        if physics.iotaf is not None:
            iota_profile = compute_iota_profile_from_iotaf(physics.iotaf, tfluxes)
        else:
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

        ivolume, isurf = distribute_current_to_interfaces(
            physics.ac, physics.curtor, tfluxes, vol_to_interface_ratio
        )
        profiles["ivolume"] = ivolume
        profiles["isurf"] = isurf
        dict_physics.ivolume = ivolume
        dict_physics.isurf = isurf
        # curtor is the total enclosed current this profile represents (in μ₀·A):
        # the cumulative volume current at the boundary plus every sheet current.
        # Setting it from the profiles keeps Fortran's rescaling a no-op (and
        # gives curtor=0 for a zero profile).
        dict_physics.curtor = float(ivolume[-1] + np.sum(isurf))

    else:
        raise RuntimeError(
            f"Unrecognised profile_type '{physics.profile_type}'; expected 'iota' or 'current'."
        )

    dict_total = dict(physics=dict_physics, numeric=dict_numeric)
    dict_pydantic = input_parameters_from_dict(dict_total)

    return dict_pydantic, profiles
