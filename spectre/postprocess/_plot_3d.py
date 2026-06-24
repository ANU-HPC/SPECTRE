# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""3D interface-surface plotting for :class:`~spectre.postprocess.SPECTREout`.

These methods render the plasma interfaces as closed toroidal surfaces with ``pyvista``,
coloured by a force-balance quantity. The geometry and all colour scalars are evaluated in
pure Python from the stored vector potential, so the same routine is reused by the live
``spectre_obj.inspect`` namespace (which builds a transient ``SPECTREout`` first).
"""

import numpy as np

# Per-quantity colorbar labels.
_INFACES3D_LABELS = {
    "force": "force residual  [[p + B^2/2]]",
    "pressure_jump": "pressure jump  dp",
    "b2_inner": "B^2 inner",
    "b2_outer": "B^2 outer",
    "current": "|B_in x B_out|",
    "angle": "angle(B_in, B_out) [deg]",
}


def _infaces3d_scalar(self, color_by, i, mvol, pressure, tarr, zarr):
    """Per-cell colour scalar on interface ``i`` (0-based) over the ``(theta, zeta)`` grid.

    Interface ``i`` is the ``s=+1`` surface of volume ``i`` (between volume ``i`` and ``i+1``).
    The inner side is volume ``i`` at ``s=+1``; the outer side is volume ``i+1`` at ``s=-1``.
    Returns an array of shape ``(len(tarr), len(zarr))``.
    """
    if color_by == "pressure_jump":
        return np.full((len(tarr), len(zarr)), float(pressure[i + 1] - pressure[i]))

    if color_by == "b2_inner":
        return self.get_field_mod(i, [1.0], tarr, zarr)[0] ** 2

    if color_by == "b2_outer":
        return self.get_field_mod(i + 1, [-1.0], tarr, zarr)[0] ** 2

    if color_by == "force":
        b2_in = self.get_field_mod(i, [1.0], tarr, zarr)[0] ** 2
        b2_out = self.get_field_mod(i + 1, [-1.0], tarr, zarr)[0] ** 2
        return float(pressure[i + 1] - pressure[i]) + 0.5 * (b2_out - b2_in)

    # current / angle need the field vectors on both sides (cylindrical [B_R, B_Z, B_phi]).
    b_in = self.get_field_cylindrical(i, [1.0], tarr, zarr)[:, 0]  # (3, nt, nz)
    b_out = self.get_field_cylindrical(i + 1, [-1.0], tarr, zarr)[:, 0]
    if color_by == "current":
        return np.linalg.norm(np.cross(b_in, b_out, axis=0), axis=0)
    # angle between the inner and outer field solutions, in degrees.
    dot = np.sum(b_in * b_out, axis=0)
    norms = np.linalg.norm(b_in, axis=0) * np.linalg.norm(b_out, axis=0)
    return np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))


def infaces3d_surfaces(self, color_by="force", surfaces=None, n_theta=64, n_zeta=192):
    """Build the 3D interface surfaces and their colour scalar (no plotting).

    Returns ``(data, tarr, zarr)`` where ``data`` is a list of
    ``(interface_index, X, Y, Z, scalar)`` with each array of shape ``(n_theta, n_zeta)``.
    The Cartesian coordinates close into a full torus (``in_`` already encodes ``nfp``, so
    ``zeta`` is sampled over the whole ``[0, 2*pi]``).
    """
    if color_by not in _INFACES3D_LABELS:
        raise ValueError(
            f"color_by must be one of {sorted(_INFACES3D_LABELS)}; got {color_by!r}"
        )

    mvol = self.input.physics.Mvol
    needs_outer = color_by != "b2_inner"  # everything except the inner B^2 needs volume i+1

    if surfaces is None:
        surfaces = list(range(mvol - 1))  # internal force-balance interfaces
    surfaces = [mvol - 1 if s == -1 else int(s) for s in np.atleast_1d(surfaces)]

    if needs_outer:
        bad = [s for s in surfaces if s >= mvol - 1]
        if bad:
            raise ValueError(
                f"color_by={color_by!r} needs an outer volume; interface(s) {bad} are the "
                f"outer boundary (no volume beyond). Choose interfaces < {mvol - 1}."
            )

    pscale = self.input.physics.pscale
    pressure = np.asarray(self.input.physics.pressure, dtype=float)[:mvol] * pscale

    tarr = np.linspace(0.0, 2.0 * np.pi, n_theta)
    zarr = np.linspace(0.0, 2.0 * np.pi, n_zeta)
    cosz = np.cos(zarr)[None, :]
    sinz = np.sin(zarr)[None, :]

    data = []
    for i in surfaces:
        rarr, zcoord = self.get_coord_transform(i, 1.0, tarr, zarr)
        r = rarr[0, 0]  # (n_theta, n_zeta)
        z = zcoord[0, 0]
        x = r * cosz
        y = r * sinz
        scalar = self._infaces3d_scalar(color_by, i, mvol, pressure, tarr, zarr)
        data.append((i, x, y, z, scalar))

    return data, tarr, zarr


def plot_infaces3D(
    self,
    color_by="force",
    surfaces=None,
    n_theta=64,
    n_zeta=192,
    cmap=None,
    show=True,
    plotter=None,
    show_boundary=True,
    opacity=1.0,
    **kwargs,
):
    """Render the plasma interfaces as 3D toroidal surfaces coloured by a force quantity.

    Args:
        color_by (str): Quantity to colour by — one of ``'force'`` (the optimiser target,
            ``[[p + B^2/2]]``), ``'pressure_jump'`` (``dp``), ``'b2_inner'``, ``'b2_outer'``,
            ``'current'`` (``|B_in x B_out|``), or ``'angle'`` (angle between the inner and
            outer field solutions, degrees). Defaults to ``'force'``.
        surfaces (array-like, optional): Interface indices to draw. Defaults to the internal
            force-balance interfaces ``0..mvol-2``. ``-1`` is the outermost interface.
        n_theta, n_zeta (int): Surface grid resolution (``n_zeta`` spans the full torus).
        cmap (str, optional): Colormap; defaults to ``'coolwarm'`` for the signed ``'force'``
            and ``'viridis'`` otherwise.
        show (bool): Open the interactive window. Defaults to True.
        plotter (pyvista.Plotter, optional): Draw into an existing plotter.
        show_boundary (bool): Draw the outer boundary as a faint grey surface for context.
        opacity (float): Surface opacity.
        **kwargs: Forwarded to ``pyvista``'s ``add_mesh``.

    Returns:
        pyvista.Plotter: the plotter (so the caller can further customise / screenshot).
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # optional dependency
        raise ImportError(
            "plot_infaces3D requires pyvista; install it with `pip install pyvista`."
        ) from exc

    data, tarr, zarr = self.infaces3d_surfaces(color_by, surfaces, n_theta, n_zeta)
    label = _INFACES3D_LABELS[color_by]

    allvals = np.concatenate([scalar.ravel() for *_, scalar in data])
    vmin, vmax = float(np.nanmin(allvals)), float(np.nanmax(allvals))
    if color_by == "force":  # signed residual -> symmetric range about 0
        amp = max(abs(vmin), abs(vmax)) or 1.0
        vmin, vmax = -amp, amp
    if cmap is None:
        cmap = "coolwarm" if color_by == "force" else "viridis"

    if plotter is None:
        plotter = pv.Plotter()

    for k, (_idx, x, y, z, scalar) in enumerate(data):
        grid = pv.StructuredGrid(x, y, z)
        grid[label] = scalar.ravel(order="F")  # StructuredGrid points vary i (theta) fastest
        plotter.add_mesh(
            grid,
            scalars=label,
            cmap=cmap,
            clim=(vmin, vmax),
            opacity=opacity,
            smooth_shading=True,
            show_scalar_bar=(k == 0),
            **({"scalar_bar_args": {"title": label}} if k == 0 else {}),
            **kwargs,
        )

    mvol = self.input.physics.Mvol
    drawn = {idx for idx, *_ in data}
    if show_boundary and (mvol - 1) not in drawn:
        cosz = np.cos(zarr)[None, :]
        sinz = np.sin(zarr)[None, :]
        rarr, zcoord = self.get_coord_transform(mvol - 1, 1.0, tarr, zarr)
        rb, zb = rarr[0, 0], zcoord[0, 0]
        bgrid = pv.StructuredGrid(rb * cosz, rb * sinz, zb)
        plotter.add_mesh(bgrid, color="lightgray", opacity=0.12, show_scalar_bar=False)

    plotter.add_axes()
    if show:
        plotter.show()
    return plotter
