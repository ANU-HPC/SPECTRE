# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Magnetic axis position as ``(R, Z)`` from ``get_rz_coords`` for a loaded ``SPECTRE`` state."""

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

    out = test.create_spectreout()
    rax, zax, sax, thax = out.find_axis(max_repeats=max_repeats, verbose=verbose)

    nfp = test.input_list_mod.nfp
    phi = zetan * (2.0 * np.pi / nfp)

    # phi = 0: find_axis already returned (R, Z) at that cut
    if np.isclose(phi, 0.0):
        return rax, zax

    # phi != 0: map (s, theta) at the axis to (R, Z) at this toroidal angle
    rv, zv = out.get_rz_coords(0, sax, thax, phi)
    r0 = np.asarray(rv).ravel()[0]
    z0 = np.asarray(zv).ravel()[0]
    return r0, z0
