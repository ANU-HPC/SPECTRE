# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""VMEC-to-SPECTRE converter using the simsopt Vmec interface."""

import numpy as np

from spectre.file_io.toml_io import write_input_parameters_to_toml

from .vmec_helpers import (
    VmecPhysics,
    build_spectre_input,
    vmec_boundary_to_fourier_coeffs,
    power_series,
    plot_vmec2spectre,
)


def simsvmec2spectre(
    vmec_obj,
    tfluxes=None,
    nvol=None,
    mpol=6,
    ntor=5,
    current_mode="volume",
    filename=None,
    plot=False,
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
        mpol: poloidal Fourier resolution (default 6).
        ntor: toroidal Fourier resolution (default 5).
        current_mode: ``'volume'`` (integrated volume currents, default) or
            ``'sheet'`` (surface sheet currents).  Only relevant when the VMEC
            equilibrium uses a current profile.
        filename: path for the output TOML file.  If ``None``, no file is
            written and the result is returned only.
        plot: if ``True``, show a figure comparing VMEC continuous profiles
            with the SPECTRE-discretised values.

    Returns:
        Validated :class:`~spectre.file_io.input_parameters.InputParameters`
        pydantic object.

    Raises:
        RuntimeError: if neither ``tfluxes`` nor ``nvol`` is provided.
        RuntimeError: if ``vmec_obj`` has neither ``indata`` nor ``wout``.
        RuntimeError: if the current-profile type is not ``'power_series'``.
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
        phiedge = src.phi[-1]   # last entry is the boundary flux
        pres_scale = src.presf[0]
        am, ai, ac = src.am, src.ai, src.ac
        nfp = src.nfp
        # pcurr_type is stored as a byte array in wout; join and strip whitespace.
        pcurr_type = "".join([s.decode() for s in src.pcurr_type]).strip()
        # In wout files there is no ncurr flag; infer from whether ai is nonzero.
        profile_type = "current" if np.max(np.abs(src.ai)) < 1e-8 else "iota"

    else:
        raise RuntimeError("vmec_obj must have either an 'indata' or 'wout' attribute.")

    if pcurr_type != "power_series":
        raise RuntimeError(
            f"Only 'power_series' current profile is supported; got '{pcurr_type}'."
        )

    # Extract boundary Fourier coefficients from the simsopt surface object.
    rbc, zbs = vmec_boundary_to_fourier_coeffs(vmec_obj.boundary)

    physics = VmecPhysics(
        curtor=curtor,
        phiedge=phiedge,
        pres_scale=pres_scale,
        am=am, ai=ai, ac=ac,
        nfp=nfp,
        rbc=rbc, zbs=zbs,
        pcurr_type=pcurr_type,
        profile_type=profile_type,
    )

    # --- build SPECTRE input ---
    dict_pydantic, profiles = build_spectre_input(physics, tfluxes, mpol, ntor, current_mode)

    if filename is not None:
        write_input_parameters_to_toml(dict_pydantic, filename)

    if plot:
        plot_vmec2spectre(
            tfluxes=tfluxes,
            pressure_func=power_series(physics.am),
            pressure_profile=profiles["pressure"],
            profile_type=physics.profile_type,
            iota_func=power_series(physics.ai) if physics.profile_type == "iota" else None,
            iota_profile=profiles.get("iota"),
            current_density_func=power_series(physics.ac) if physics.profile_type == "current" else None,
            current_mode=current_mode,
            volume_current_profile=profiles.get("volume_current"),
            sheet_current_profile=profiles.get("sheet_current"),
            curtor=physics.curtor,
        )

    return dict_pydantic
