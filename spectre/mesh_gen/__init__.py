# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Generate SPECTRE meshes and interface guesses from a SPECTRE object's state.

The two public generators are :func:`gen_infaces` (nested internal interface
guesses) and :func:`gen_mesh_external` (external mesh harmonics).  All routines
are re-exported here so they can be reached as ``spectre.mesh_gen.XXX``; the
implementation is split across two submodules by concern:

- :mod:`spectre.mesh_gen.map2disc` — the BCM conformal map and the SPECTRE-state
  boundary mapping (:func:`~spectre.mesh_gen.map2disc.map_boundary_to_interfaces`,
  :func:`~spectre.mesh_gen.map2disc.get_inface_func`,
  :func:`~spectre.mesh_gen.map2disc.gen_mesh_calc_rz`).
- :mod:`spectre.mesh_gen.generators` — the spectral decomposition and mesh
  assembly (:func:`~spectre.mesh_gen.generators.gen_mesh_fft_rz`,
  :func:`~spectre.mesh_gen.generators.gen_mesh_calc_deriv2`,
  :func:`~spectre.mesh_gen.generators.gen_mesh_set_remn`) and the two generators
  above.

Note: the top-level :mod:`spectre` package deliberately exposes only
:func:`gen_infaces` and :func:`gen_mesh_external`.
"""

from .generators import (
    gen_infaces as gen_infaces,
    gen_mesh_calc_deriv2 as gen_mesh_calc_deriv2,
    gen_mesh_external as gen_mesh_external,
    gen_mesh_fft_rz as gen_mesh_fft_rz,
    gen_mesh_set_remn as gen_mesh_set_remn,
)
from .map2disc import (
    gen_mesh_calc_rz as gen_mesh_calc_rz,
    get_inface_func as get_inface_func,
    map_boundary_to_interfaces as map_boundary_to_interfaces,
)

__all__ = [
    "gen_infaces",
    "gen_mesh_external",
    "gen_mesh_fft_rz",
    "gen_mesh_calc_deriv2",
    "gen_mesh_set_remn",
    "gen_mesh_calc_rz",
    "get_inface_func",
    "map_boundary_to_interfaces",
]
