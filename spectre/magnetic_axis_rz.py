# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Magnetic axis position as ``(R, Z)`` from ``get_rz_coords`` for a loaded ``SPECTRE`` state."""

import numpy as np

from spectre.force_targets import force_real
from spectre.utils import get_xinit_specwrap, unpack_interfaces


def magnetic_axis_rz(test, x, zetan, *, max_repeats=50, verbose=False):
    """
    Return magnetic-axis ``(R, Z)`` for one toroidal cut (``zetan`` scalar per call).

    ``φ = \\texttt{zetan} \\cdot (2\\pi/N_{\\mathrm{fp}})`` radians, i.e.
    ``\\texttt{zetan} = \\phi\\,N_{\\mathrm{fp}}/(2\\pi)`` for this normalized label.
    Returns two floats ``(R, Z)`` for that cut (callers stack per iteration themselves).

    Unpacks ``x`` into ``test``, runs ``force_real``, then ``test.create_spectreout().find_axis`` (vol.~0).
    The **geometric toroidal angle** ``φ`` matches ``plot_evolution_coord_axis``.
    ``get_rz_coords`` names this angle ``phi``; in ``spectre.geometry.get_coord_transform``
    the corresponding array argument is ``zarr`` (same angle in the
    ``im * theta - in * zarr`` Fourier sums for toroidal geometry).
    """
    unpack_interfaces(x, test)
    xin = get_xinit_specwrap(test)
    force_real(xin, test)

    # Create a spectreout object and find the magnetic axis
    out = test.create_spectreout()
    rax, zax, sax, thax = out.find_axis(max_repeats=int(max_repeats), verbose=bool(verbose))
    rax0, zax0 = float(rax), float(zax)
    sax_f, thax_f = float(sax), float(thax)

    # Release the tracer after finding the axis
    if hasattr(out, "_tracer_single"):
        try:
            delattr(out, "_tracer_single")
        except Exception:
            pass

    nfp = int(test.input_list_mod.nfp)
    phi = float(zetan) * (2.0 * np.pi / float(nfp))

    # Case phi=0
    if np.isclose(phi, 0.0):
        return rax0, zax0

    # Case phi!=0
    try:
        rv, zv = out.get_rz_coords(0, float(sax_f), float(thax_f), float(phi))
        R = float(np.asarray(rv, dtype=float).ravel()[0])
        Z = float(np.asarray(zv, dtype=float).ravel()[0])
        return R, Z
    except Exception as err_rz:
        if hasattr(out, "get_coord_transform"):  # If get_rz_coords fails, try get_coord_transform
            try:
                rarr, zarr = out.get_coord_transform(0, sax_f, thax_f, phi)
                R = float(np.asarray(rarr, dtype=float).ravel()[0])
                Z = float(np.asarray(zarr, dtype=float).ravel()[0])
                return R, Z
            except Exception as err_ct:
                raise RuntimeError(
                    "magnetic_axis_rz: could not map magnetic axis to (R, Z) at this "
                    "toroidal angle (get_rz_coords and get_coord_transform both failed)."
                ) from err_ct
        raise RuntimeError(
            "magnetic_axis_rz: get_rz_coords failed to map magnetic axis to (R, Z) at "
            "this toroidal angle."
        ) from err_rz
