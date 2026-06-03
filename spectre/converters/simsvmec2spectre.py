# SPDX-FileCopyrightText: 2026 EPFL
#
# SPDX-License-Identifier: MIT
#
# Authors:  Erol Balkovic <erol.balkovic@epfl.ch>
#           Christopher Berg Smiet <chris@bergtopfusion.com>

"""VMEC-to-SPECTRE converter using the simsopt Vmec interface."""

import numpy as np

from spectre.file_io.toml_io import write_input_parameters_to_toml

from .compare_vmec_spectre import compare_vmec_spectre
from .map2disc import compute_interface_guesses_map2disc
from .vmec_helpers import (
    VmecPhysics,
    build_spectre_input,
    compute_interface_guesses,
    vmec_boundary_to_fourier_coeffs,
)


def simsvmec2spectre(
    vmec_obj,
    tfluxes=None,
    nvol=None,
    mpol=None,
    ntor=None,
    vol_to_interface_ratio=1.0,
    profile_type=None,
    filename=None,
    plot=False,
    angle_fix=True,
    interface_guess=None,
    map2disc_kwargs=None,
):
    """Generate a SPECTRE input from a simsopt VMEC equilibrium.

    Reads pressure, rotational-transform or current profiles and the plasma
    boundary from a simsopt :class:`~simsopt.mhd.Vmec` object, discretises
    them onto a set of SPECTRE interfaces, and optionally writes the result to
    a TOML file.

    Supports both VMEC ``input.*`` files (``vmec_obj.indata``) and
    ``wout_*.nc`` output files (``vmec_obj.wout``).  Only ``power_series``
    current profiles are supported.

    Args:
        vmec_obj: simsopt ``Vmec`` instance.
        tfluxes: 1-D array-like of interface toroidal-flux values in (0, 1].
            Mutually exclusive with ``nvol``.  If ``tfluxes[-1] < 1``, the
            outermost SPECTRE interface is interior to the VMEC plasma
            boundary; the boundary coefficients (``rbc``/``zbs``) are then
            replaced with the interpolated geometry at ``tfluxes[-1]``.  This
            requires ``interface_guess`` to be ``'wout'`` or ``'map2disc'``.
        nvol: number of SPECTRE volumes; interfaces are placed uniformly.
            Mutually exclusive with ``tfluxes``.
        mpol: poloidal Fourier resolution.  Defaults to ``vmec_obj.boundary.mpol``.
        ntor: toroidal Fourier resolution.  Defaults to ``vmec_obj.boundary.ntor``.
        vol_to_interface_ratio: fraction r ∈ [0, 1] of the toroidal current
            represented as distributed volume current (``ivolume``); the
            remainder (1 − r) is placed on the interfaces as sheet current
            (``isurf``).  ``1.0`` (default) = all volume, ``0.0`` = all sheet,
            intermediate = a blend.  Only relevant when the VMEC equilibrium
            uses a current profile.  The continuous-to-discrete current mapping
            is **provisional** — see
            :func:`~spectre.converters.vmec_helpers.distribute_current_to_interfaces`.
        profile_type: ``'iota'`` or ``'current'``, overriding auto-detection.
            When ``None`` (default), the type is inferred from the source:
            ``ncurr`` for ``indata``, or whether ``ai`` is nonzero for ``wout``.
        filename: path for the output TOML file.  If ``None``, no file is
            written and the result is returned only.
        plot: if ``True``, show a figure comparing VMEC continuous profiles
            with the SPECTRE-discretised values.
        angle_fix: if ``True`` (default), apply the θ→-θ handedness correction
            to boundary and interface coefficients (n→-n for m>0, ZBS negated).
            This replaces the need for ``lchangeangle=True`` in the TOML.
            Set to ``False`` only for equilibria that run correctly without
            ``lchangeangle``.
        interface_guess: strategy for generating initial interface geometry.
            ``None`` (default) — no guesses; SPECTRE linearly interpolates
            (``linitialize=1``).
            ``'wout'`` — cubic interpolation of VMEC flux surfaces from a wout
            file (requires the ``Vmec`` object to have been loaded from a
            ``wout_*.nc`` file).
            ``'map2disc'`` — conformal mapping via the ``map2disc`` library;
            works from both ``input.*`` and ``wout_*.nc`` sources.
            When guesses are provided, ``linitialize=0``.
        map2disc_kwargs: optional dict of keyword arguments forwarded to
            :func:`~spectre.converters.map2disc.compute_interface_guesses_map2disc`
            (e.g. ``{"ntheta": 32, "nzeta": 15, "bcm_m": 4}``).
            Only used when ``interface_guess='map2disc'``.

    Returns:
        Validated :class:`~spectre.file_io.input_parameters.InputParameters`
        pydantic object.

    Raises:
        RuntimeError: if neither ``tfluxes`` nor ``nvol`` is provided.
        RuntimeError: if ``vmec_obj`` has neither ``indata`` nor ``wout``.
        RuntimeError: if the current-profile type is not ``'power_series'``.
        RuntimeError: if ``interface_guess='wout'`` but the Vmec object has no wout.
        ValueError: if ``interface_guess`` is not ``None``, ``'wout'``, or ``'map2disc'``.
        ValueError: if ``tfluxes[-1] < 1`` but ``interface_guess`` is ``None``.
    """
    # --- resolve interface positions ---
    if tfluxes is None and nvol is None:
        raise RuntimeError("Provide either 'tfluxes' or 'nvol'.")
    if tfluxes is None:
        tfluxes = np.linspace(0.0, 1.0, nvol + 1, endpoint=True)[1:]
    tfluxes = np.asarray(tfluxes)

    # --- extract physics from the simsopt Vmec object ---
    # simsopt exposes two different attribute layouts depending on whether the
    # object was loaded from an input file (indata) or a wout netCDF file (wout).
    if hasattr(vmec_obj, "indata"):
        src = vmec_obj.indata
        curtor = src.curtor
        phiedge = src.phiedge
        pres_scale = src.pres_scale
        am, ai, ac = src.am, src.ai, src.ac
        nfp = src.nfp
        pcurr_type = src.pcurr_type.decode()
        # ncurr=0 means iota is prescribed; ncurr=1 means current is prescribed.
        auto_profile_type = "iota" if src.ncurr == 0 else "current"
        iotaf = None  # indata has no precomputed iota grid

    elif hasattr(vmec_obj, "wout"):
        src = vmec_obj.wout
        curtor = src.ctor
        phiedge = src.phi[-1]  # last entry is the boundary flux
        # am in the wout stores the original polynomial coefficients (in Pa);
        # presf[0] = pres_scale_input × am[0], so recover the true scale factor.
        pres_scale = src.presf[0] / src.am[0] if abs(src.am[0]) > 1e-10 else 1.0
        am, ai, ac = src.am, src.ai, src.ac
        nfp = src.nfp
        # pcurr_type is stored as a byte array in wout; join and strip whitespace.
        pcurr_type = "".join([s.decode() for s in src.pcurr_type]).strip()
        # In wout files there is no ncurr flag; infer from whether ai is nonzero.
        auto_profile_type = "current" if np.max(np.abs(src.ai)) < 1e-8 else "iota"
        iotaf = src.iotaf  # iota on full radial grid (ns points, s=0..1)

    else:
        raise RuntimeError("vmec_obj must have either an 'indata' or 'wout' attribute.")

    effective_profile_type = profile_type if profile_type is not None else auto_profile_type

    if pcurr_type != "power_series":
        raise RuntimeError(f"Only 'power_series' current profile is supported; got '{pcurr_type}'.")

    # Resolve mpol/ntor from the VMEC boundary surface if not specified.
    if mpol is None:
        mpol = vmec_obj.boundary.mpol
    if ntor is None:
        ntor = vmec_obj.boundary.ntor

    # Extract boundary Fourier coefficients from the simsopt surface object.
    rbc, zbs = vmec_boundary_to_fourier_coeffs(vmec_obj.boundary, angle_fix=angle_fix)

    physics = VmecPhysics(
        curtor=curtor,
        phiedge=phiedge,
        pres_scale=pres_scale,
        am=am,
        ai=ai,
        ac=ac,
        nfp=nfp,
        rbc=rbc,
        zbs=zbs,
        pcurr_type=pcurr_type,
        profile_type=effective_profile_type,
        iotaf=iotaf,
    )

    # Generate interface geometry guesses based on the requested strategy.
    if interface_guess is None:
        interface_guesses = None
    elif interface_guess == "map2disc":
        interface_guesses = compute_interface_guesses_map2disc(
            physics.rbc,
            physics.zbs,
            physics.nfp,
            tfluxes,
            mpol,
            ntor,
            **(map2disc_kwargs or {}),
        )
    elif interface_guess == "wout":
        if not (hasattr(vmec_obj, "wout") and hasattr(vmec_obj.wout, "ns")):
            raise RuntimeError(
                "interface_guess='wout' requires a Vmec object loaded from a wout_*.nc file."
            )
        interface_guesses = compute_interface_guesses(vmec_obj.wout, tfluxes, mpol, ntor)
    else:
        raise ValueError(
            f"interface_guess must be None, 'wout', or 'map2disc'; got {interface_guess!r}."
        )

    # If the outermost requested interface sits inside the VMEC plasma boundary
    # (tflux < 1), replace the boundary coefficients with the interpolated
    # geometry at tfluxes[-1]. This requires a method that can evaluate VMEC
    # surfaces at arbitrary tflux — i.e. interface_guess in {'wout', 'map2disc'}.
    if not np.isclose(tfluxes[-1], 1.0):
        if interface_guesses is None:
            raise ValueError(
                "asked to put the boundary at an internal interface but no method "
                "for generating these specified"
            )
        last = interface_guesses[f"interface_{len(tfluxes)}"]
        physics = physics._replace(rbc=last["rbc"], zbs=last["zbs"])
        print(
            f"tfluxes[-1] = {tfluxes[-1]:.6f} != 1.0: replacing plasma boundary "
            f"with interpolated surface at this tflux (via interface_guess={interface_guess!r})."
        )

    # --- build SPECTRE input ---
    dict_pydantic, profiles = build_spectre_input(
        physics,
        tfluxes,
        mpol,
        ntor,
        vol_to_interface_ratio,
        interface_guesses=interface_guesses,
    )

    if filename is not None:
        write_input_parameters_to_toml(dict_pydantic, filename)

    if plot:
        compare_vmec_spectre(vmec_obj, dict_pydantic)

    return dict_pydantic
