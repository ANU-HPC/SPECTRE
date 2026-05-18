# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""VMEC-to-SPECTRE converter using the simsopt Vmec interface."""

import numpy as np

from spectre.file_io.toml_io import write_input_parameters_to_toml

from .compare_vmec_spectre import compare_vmec_spectre
from .vmec_helpers import (
    VmecPhysics,
    build_spectre_input,
    compute_interface_guesses,
    compute_interface_guesses_map2disc,
    vmec_boundary_to_fourier_coeffs,
)


def simsvmec2spectre(
    vmec_obj,
    tfluxes=None,
    nvol=None,
    mpol=None,
    ntor=None,
    current_mode="volume",
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
            Mutually exclusive with ``nvol``.
        nvol: number of SPECTRE volumes; interfaces are placed uniformly.
            Mutually exclusive with ``tfluxes``.
        mpol: poloidal Fourier resolution.  Defaults to ``vmec_obj.boundary.mpol``.
        ntor: toroidal Fourier resolution.  Defaults to ``vmec_obj.boundary.ntor``.
        current_mode: ``'volume'`` (integrated volume currents, default) or
            ``'sheet'`` (surface sheet currents).  Only relevant when the VMEC
            equilibrium uses a current profile.
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
            :func:`~spectre.converters.vmec_helpers.compute_interface_guesses_map2disc`
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
        profile_type = "iota" if src.ncurr == 0 else "current"

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
        profile_type = "current" if np.max(np.abs(src.ai)) < 1e-8 else "iota"

    else:
        raise RuntimeError("vmec_obj must have either an 'indata' or 'wout' attribute.")

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
        profile_type=profile_type,
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

    # --- build SPECTRE input ---
    dict_pydantic, profiles = build_spectre_input(
        physics,
        tfluxes,
        mpol,
        ntor,
        current_mode,
        interface_guesses=interface_guesses,
    )

    if filename is not None:
        write_input_parameters_to_toml(dict_pydantic, filename)

    if plot:
        compare_vmec_spectre(vmec_obj, dict_pydantic)

    return dict_pydantic
