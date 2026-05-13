# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Magnetic axis position as ``(R, Z)`` from ``get_rz_coords`` for a loaded ``SPECTRE`` state."""

import contextlib

import numpy as np

from spectre.force_targets import force_real
from spectre.utils import get_xinit_specwrap, unpack_interfaces


def magnetic_axis_rz(test, x, zetan, *, max_repeats=50, verbose=False):
    r"""Return magnetic-axis ``(R, Z)`` for one toroidal cut (``zetan`` scalar per call).

    ``φ = \texttt{zetan} \cdot (2\pi/N_{\mathrm{fp}})`` radians, i.e.
    ``\texttt{zetan} = \phi\,N_{\mathrm{fp}}/(2\pi)`` for this normalized label.
    Returns two floats ``(R, Z)`` for that cut (callers stack per iteration themselves).

    Unpacks ``x`` into ``test``, runs ``force_real``, then
    ``test.create_spectreout().find_axis`` (vol.~0).
    The **geometric toroidal angle** ``φ`` matches ``plot_evolution_coord_axis``.
    ``get_rz_coords`` uses the same angle as its ``phi`` argument.
    """
    unpack_interfaces(x, test)
    xin = get_xinit_specwrap(test)
    force_real(xin, test)

    # SPECTREout carries find_axis / get_rz_coords used for the axis solve and mapping.
    out = test.create_spectreout()
    rax, zax, sax, thax = out.find_axis(max_repeats=int(max_repeats), verbose=bool(verbose))
    rax0, zax0 = float(rax), float(zax)
    sax_f, thax_f = float(sax), float(thax)

    # Release the tracer after finding the axis
    if hasattr(out, "_tracer_single"):
        with contextlib.suppress(Exception):
            delattr(out, "_tracer_single")

    nfp = int(test.input_list_mod.nfp)
    phi = float(zetan) * (2.0 * np.pi / float(nfp))

    # phi = 0: find_axis already returned (R, Z) at that cut
    if np.isclose(phi, 0.0):
        return rax0, zax0

    # phi != 0: map (s, theta) at the axis to (R, Z) at this toroidal angle
    rv, zv = out.get_rz_coords(0, float(sax_f), float(thax_f), float(phi))
    r0 = float(np.asarray(rv, dtype=float).ravel()[0])
    z0 = float(np.asarray(zv, dtype=float).ravel()[0])
    return r0, z0
