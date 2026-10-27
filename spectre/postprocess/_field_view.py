# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Extract field and geometry data from a live SPECTRE instance.

The extracted namespaces contain the small Fourier-coefficient arrays and per-volume vector
potential needed by :meth:`SPECTREout.from_spectre`.

Note: these namespaces hold only small Fourier coefficients and the vector potential — none
of the large field/Poincare grids a SPECTREout loaded from an ``.h5`` may carry.
"""

import contextlib
import types

import numpy as np


def _spectre_field_namespaces(spectre):
    """Build the ``(input, output, vector_potential)`` namespaces for field/geometry eval.

    Pulls only the small data — Fourier mode numbers, interface coefficients and the
    per-volume vector potential — from a live SPECTRE instance.
    """
    from spectre.utils import get_vec_pot

    lib = spectre.input_list_mod
    glob = spectre.allglobal_mod

    nvol = int(lib.nvol)
    lfreebound = int(lib.lfreebound)
    mvol = nvol + 1 if lfreebound else nvol

    physics = types.SimpleNamespace()
    physics.Lrad = np.array(lib.lrad[:mvol])
    physics.Mpol = int(lib.mpol)
    physics.Igeometry = int(lib.igeometry)
    physics.Nvol = nvol
    physics.Lfreebound = lfreebound
    physics.Mvol = mvol
    physics.Nfp = int(lib.nfp)
    physics.rpol = float(lib.rpol)
    physics.rtor = float(lib.rtor)
    physics.pscale = float(lib.pscale)
    physics.pressure = np.array(lib.pressure[:nvol])
    with contextlib.suppress(AttributeError):
        physics.enforce_stell_sym = bool(lib.enforce_stell_sym)

    input_ = types.SimpleNamespace(physics=physics)

    output = types.SimpleNamespace()
    output.im = np.array(glob.im)
    output.in_ = np.array(glob.in_)
    output.Rbc = np.array(glob.irbc.T)
    output.Zbs = np.array(glob.izbs.T)
    output.Rbs = np.array(glob.irbs.T)
    output.Zbc = np.array(glob.izbc.T)
    output.tflux = np.array(lib.tflux[:mvol])
    ate_list, aze_list, ato_list, azo_list = get_vec_pot(spectre)
    vp = types.SimpleNamespace(Ate=ate_list, Aze=aze_list, Ato=ato_list, Azo=azo_list)

    return input_, output, vp
