# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Comparison plots: VMEC continuous profiles vs SPECTRE discretised values."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from .vmec_helpers import MU0, power_series


def plot_pressure_comparison(ax, am, pres_scale, tfluxes, pressure_spectre):
    """Overlay VMEC continuous pressure and SPECTRE volume-averaged pressure.

    The VMEC profile is plotted as a line; the SPECTRE values (stored internally
    as mu0*<p>) are shown as horizontal segments spanning each volume.

    Args:
        ax: matplotlib ``Axes`` to draw into.
        am: VMEC pressure polynomial coefficients.
        pres_scale: VMEC pressure scale factor.
        tfluxes: 1-D array of interface toroidal-flux values.
        pressure_spectre: 1-D array of mu0*<p> per volume from SPECTRE.
    """
    s = np.linspace(0.0, 1.0, 256)
    tfluxes_ext = np.concatenate([[0.0], tfluxes])

    ax.plot(s, pres_scale * power_series(am)(s), label="VMEC")
    ax.hlines(
        pressure_spectre / MU0,
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


def plot_iota_comparison(ax, ai, tfluxes, iota_spectre=None):
    """Overlay VMEC continuous iota and optionally the SPECTRE interface values.

    Args:
        ax: matplotlib ``Axes`` to draw into.
        ai: VMEC rotational-transform polynomial coefficients.
        tfluxes: 1-D array of interface toroidal-flux values.
        iota_spectre: 1-D array of iota values at interfaces from SPECTRE;
            omit to show only the VMEC profile (e.g. for current equilibria).
    """
    s = np.linspace(0.0, 1.0, 256)

    ax.plot(s, power_series(ai)(s), label="VMEC")
    if iota_spectre is not None:
        ax.plot(tfluxes, iota_spectre, "o", color="C1", label="SPECTRE")
    ax.vlines(tfluxes, *ax.get_ylim(), color="k", lw=0.8, alpha=0.5)
    ax.set_title("Rotational transform")
    ax.set_xlabel("radial parameter (s or tflux)")
    ax.legend()


def plot_current_comparison(ax, ac, curtor, tfluxes, current_spectre=None, spectre_mode="volume"):
    """Overlay VMEC continuous current and optionally the SPECTRE discrete values.

    For ``spectre_mode='volume'``: plots the cumulative integral of the VMEC
    current-density profile (enclosed current vs s) and the SPECTRE ivolume
    values, both normalised to [0, 1].

    For ``spectre_mode='sheet'``: plots the VMEC current-density profile j(s)
    and the SPECTRE isurf values, both normalised by their respective maxima.

    Args:
        ax: matplotlib ``Axes`` to draw into.
        ac: VMEC current-density polynomial coefficients.
        curtor: total toroidal current from VMEC [A].
        tfluxes: 1-D array of interface toroidal-flux values.
        current_spectre: 1-D array of SPECTRE current values (ivolume or isurf);
            omit to show only the VMEC profile (e.g. for iota equilibria).
        spectre_mode: ``'volume'`` (default) or ``'sheet'``.
    """
    s = np.linspace(0.0, 1.0, 256)
    j_func = power_series(ac)

    if spectre_mode == "volume":
        total = quad(j_func, 0.0, 1.0)[0]
        vmec_curve = (
            np.array([quad(j_func, 0.0, si)[0] / total for si in s])
            if total != 0.0
            else np.zeros_like(s)
        )
        ax.set_title("Enclosed current (normalised)")
    else:
        j_vals = j_func(s)
        j_max = np.max(np.abs(j_vals)) if curtor != 0.0 else 1.0
        vmec_curve = j_vals / j_max if j_max != 0.0 else j_vals
        ax.set_title("Current density (normalised)")

    ax.plot(s, vmec_curve, label="VMEC")

    if current_spectre is not None and np.any(current_spectre != 0.0):
        scale = np.max(np.abs(current_spectre))
        if scale != 0.0:
            ax.plot(tfluxes, current_spectre / scale, "o", color="C1", label="SPECTRE")

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
    if hasattr(vmec, "indata"):
        src = vmec.indata
        am, ai, ac = src.am, src.ai, src.ac
        pres_scale = float(src.pres_scale)
        curtor = float(src.curtor)
    elif hasattr(vmec, "wout"):
        src = vmec.wout
        am, ai, ac = src.am, src.ai, src.ac
        pres_scale = float(src.presf[0] / src.am[0]) if abs(src.am[0]) > 1e-10 else 1.0
        curtor = float(src.ctor)
    else:
        raise RuntimeError("vmec must have either an 'indata' or 'wout' attribute.")

    phys = spectre_params.physics
    nvol = phys.nvol
    tfluxes = np.asarray(phys.tflux[:nvol])
    pressure_spectre = np.asarray(phys.pressure[:nvol])
    lconstraint = phys.lconstraint

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    plot_pressure_comparison(axes[0], am, pres_scale, tfluxes, pressure_spectre)

    if lconstraint == 1:
        iota_spectre = np.asarray(phys.iota[1 : nvol + 1])
        plot_iota_comparison(axes[1], ai, tfluxes, iota_spectre)
    else:
        plot_iota_comparison(axes[1], ai, tfluxes)

    if lconstraint == 3:
        ivolume = np.asarray(phys.ivolume[:nvol])
        isurf = np.asarray(phys.isurf[:nvol])
        if np.any(ivolume != 0.0):
            plot_current_comparison(axes[2], ac, curtor, tfluxes, ivolume, spectre_mode="volume")
        else:
            plot_current_comparison(axes[2], ac, curtor, tfluxes, isurf, spectre_mode="sheet")
    else:
        plot_current_comparison(axes[2], ac, curtor, tfluxes)

    plt.tight_layout(w_pad=2)
    return fig, axes
