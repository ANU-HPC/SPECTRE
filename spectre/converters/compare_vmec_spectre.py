# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Comparison plots: VMEC continuous profiles vs SPECTRE discretised values."""

import matplotlib.pyplot as plt
import numpy as np

from .vmec_helpers import (
    iota_function,
    normalized_enclosed_current,
    power_series,
    pressure_function,
)


def plot_pressure_comparison(ax, pressure, tfluxes, pressure_spectre):
    """Overlay VMEC continuous pressure and SPECTRE volume-averaged pressure.

    Both curves are in SPECTRE's internal (mu0-scaled) pressure units: the
    continuous VMEC profile as a line, the SPECTRE values as horizontal segments
    spanning each volume.

    Args:
        ax: matplotlib ``Axes`` to draw into.
        pressure: callable ``pressure(s)`` in SPECTRE units (mu0-scaled).
        tfluxes: 1-D array of interface toroidal-flux values.
        pressure_spectre: 1-D array of mu0*<p> per volume from SPECTRE.
    """
    s = np.linspace(0.0, 1.0, 256)
    tfluxes_ext = np.concatenate([[0.0], tfluxes])

    ax.plot(s, pressure(s), label="VMEC")
    ax.hlines(
        pressure_spectre,
        tfluxes_ext[:-1],
        tfluxes_ext[1:],
        colors="C1",
        lw=2,
        label="SPECTRE",
    )
    ax.vlines(tfluxes, *ax.get_ylim(), color="k", lw=0.8, alpha=0.5)
    ax.set_title("Pressure")
    ax.set_xlabel("radial parameter (s or tflux)")
    ax.legend()


def plot_iota_comparison(ax, iota, tfluxes, iota_spectre=None):
    """Overlay VMEC continuous iota and optionally the SPECTRE interface values.

    Args:
        ax: matplotlib ``Axes`` to draw into.
        iota: callable ``iota(s)`` returning the VMEC rotational transform.
        tfluxes: 1-D array of interface toroidal-flux values.
        iota_spectre: 1-D array of iota values at interfaces from SPECTRE;
            omit to show only the VMEC profile (e.g. for current equilibria).
    """
    s = np.linspace(0.0, 1.0, 256)

    ax.plot(s, iota(s), label="VMEC")
    if iota_spectre is not None:
        ax.plot(tfluxes, iota_spectre, "o", color="C1", label="SPECTRE")
    ax.vlines(tfluxes, *ax.get_ylim(), color="k", lw=0.8, alpha=0.5)
    ax.set_title("Rotational transform")
    ax.set_xlabel("radial parameter (s or tflux)")
    ax.legend()


def plot_current_comparison(ax, current_density, tfluxes, enclosed_spectre=None):
    """Overlay the VMEC and SPECTRE *enclosed* toroidal current vs flux.

    Plots the normalised cumulative integral of the VMEC current-density profile
    (the enclosed current Î(s) ∈ [0, 1]) and, when given, the SPECTRE enclosed
    current — the combined contribution of the volume (``ivolume``) and sheet
    (``isurf``) representations, ``ivolume + cumsum(isurf)`` — normalised by its
    boundary value so the two curves are directly comparable.

    Args:
        ax: matplotlib ``Axes`` to draw into.
        current_density: callable ``j(s) = dI/ds`` for the VMEC current profile.
        tfluxes: 1-D array of interface toroidal-flux values.
        enclosed_spectre: 1-D array of SPECTRE enclosed current per interface
            (``ivolume + cumsum(isurf)``); omit to show only the VMEC profile
            (e.g. for iota equilibria).
    """
    s = np.linspace(0.0, 1.0, 256)

    vmec_curve = normalized_enclosed_current(current_density, s)
    ax.set_title("Enclosed current (normalised)")
    ax.plot(s, vmec_curve, label="VMEC")

    if enclosed_spectre is not None and np.any(enclosed_spectre != 0.0):
        scale = enclosed_spectre[-1]
        if scale == 0.0:
            scale = np.max(np.abs(enclosed_spectre))
        if scale != 0.0:
            ax.plot(tfluxes, enclosed_spectre / scale, "o", color="C1", label="SPECTRE")

    ax.vlines(tfluxes, *ax.get_ylim(), color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("radial parameter (s or tflux)")
    ax.legend()


def compare_vmec_spectre(vmec, spectre_params):
    """Three-panel comparison of VMEC profiles vs SPECTRE discretised values.

    Produces a figure with three subplots — pressure, rotational transform, and
    toroidal current — by calling the three individual plot functions.  The
    VMEC continuous profiles are read directly from the simsopt object; the
    SPECTRE discretised values are extracted from the pydantic
    :class:`~spectre.file_io.input_parameters.InputParameters`.  The current
    mode (volume or sheet) is determined automatically from the InputParameters.

    Args:
        vmec: simsopt ``Vmec`` object (loaded from either an input file or a
            wout netCDF file).
        spectre_params: :class:`~spectre.file_io.input_parameters.InputParameters`
            returned by :func:`~spectre.converters.simsvmec2spectre`.

    Returns:
        Tuple ``(fig, axes)`` — matplotlib figure and array of 3 axes.
    """
    # The VMEC profiles are read via the same factories the converter uses, so
    # the indata/wout dispatch lives in one place (vmec_helpers).  Only the
    # coefficients pressure/current still need (am, ac, pres_scale) are read
    # here; iota dispatch is fully handled by iota_function.
    if hasattr(vmec, "indata"):
        src = vmec.indata
        am, ac = src.am, src.ac
        pres_scale = float(src.pres_scale)
    elif hasattr(vmec, "wout"):
        src = vmec.wout
        am, ac = src.am, src.ac
        pres_scale = float(src.presf[0] / src.am[0]) if abs(src.am[0]) > 1e-10 else 1.0
    else:
        raise RuntimeError("vmec must have either an 'indata' or 'wout' attribute.")

    pressure = pressure_function(am, pres_scale)
    iota = iota_function(vmec)
    current_density = power_series(ac)

    phys = spectre_params.physics
    nvol = phys.nvol
    tfluxes = np.asarray(phys.tflux[:nvol])
    pressure_spectre = np.asarray(phys.pressure[:nvol])
    lconstraint = phys.lconstraint

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plot_pressure_comparison(axes[0], pressure, tfluxes, pressure_spectre)

    if lconstraint == 1:
        iota_spectre = np.asarray(phys.iota[1 : nvol + 1])
        plot_iota_comparison(axes[1], iota, tfluxes, iota_spectre)
    else:
        plot_iota_comparison(axes[1], iota, tfluxes)

    if lconstraint == 3:
        ivolume = np.asarray(phys.ivolume[:nvol])
        isurf = np.asarray(phys.isurf[:nvol])
        # Combined enclosed current: cumulative volume current plus the running
        # sum of the interface sheet currents.
        enclosed_spectre = ivolume + np.cumsum(isurf)
        plot_current_comparison(axes[2], current_density, tfluxes, enclosed_spectre)
    else:
        plot_current_comparison(axes[2], current_density, tfluxes)

    plt.tight_layout(w_pad=2)
    return fig, axes
