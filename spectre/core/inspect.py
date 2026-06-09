# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Grid/mesh inspection helpers for :class:`~spectre.SPECTRE`.

These thin wrappers expose the standalone plotting routines in
:mod:`spectre.plotting` through the ``spectre_obj.inspect`` namespace, e.g.
``spectre_obj.inspect.plot_true_mesh()``. The geometry argument ``xin`` defaults
to the instance's current ``xin`` so the methods can be called with no
arguments. The underlying implementations in :mod:`spectre.plotting` are left
untouched.
"""

import spectre.plotting as _plotting


class _Inspect:
    """Namespace of grid/mesh plotting helpers bound to a ``SPECTRE`` instance.

    Accessed as ``spectre_obj.inspect``. Each method delegates to the matching
    routine in :mod:`spectre.plotting`, passing the instance as the ``test``
    argument and defaulting ``xin`` to ``spectre_obj.xin``.
    """

    def __init__(self, spectre):
        self._spectre = spectre

    def plot_infaces(
        self,
        xin=None,
        ax=None,
        surfaces=None,
        zetan=0.0,
        thetas=None,
        figsize=None,
        plot_theta_grid=False,
        plot_axis=False,
        **kwargs,
    ):
        """Plot the interfaces of the geometry described by ``xin``.

        Args:
            xin (ndarray, optional): Packed interface degrees of freedom to plot.
                Defaults to the instance's current ``xin``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure and
                axes are created if not given.
            surfaces (array-like, optional): Indices of the interfaces to draw.
                Defaults to all interfaces. ``-1`` refers to the outer boundary.
            zetan (float, optional): Toroidal cross-section, in units of the field
                period (multiplied internally by ``2*pi/nfp``). Defaults to ``0.0``.
            thetas (array-like, optional): Poloidal angles at which to sample each
                interface. Defaults to an evenly spaced grid over ``[0, 2*pi]``.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            plot_theta_grid (bool, optional): Overlay the poloidal grid points.
                Defaults to False.
            plot_axis (bool, optional): Mark the magnetic axis. Defaults to False.
            **kwargs: Forwarded to the underlying ``matplotlib`` plot calls
                (e.g. ``c``/``color``, ``ls``/``linestyle``).
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_infaces(
            xin,
            self._spectre,
            ax=ax,
            surfaces=surfaces,
            zetan=zetan,
            thetas=thetas,
            figsize=figsize,
            plot_theta_grid=plot_theta_grid,
            plot_axis=plot_axis,
            **kwargs,
        )

    def plot_jacobian(
        self,
        xin=None,
        ax=None,
        volumes=None,
        zeta=0.0,
        thetas=None,
        sarr=None,
        figsize=None,
        **kwargs,
    ):
        """Plot the coordinate Jacobian on a cross-section of the geometry ``xin``.

        Args:
            xin (ndarray, optional): Packed interface degrees of freedom defining
                the geometry. Defaults to the instance's current ``xin``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure and
                axes are created if not given.
            volumes (array-like, optional): Indices of the volumes to draw. Defaults
                to all volumes. ``-1`` refers to the outermost volume.
            zeta (float, optional): Toroidal angle of the cross-section. Defaults to
                ``0.0``.
            thetas (array-like, optional): Poloidal angles to sample. Defaults to an
                evenly spaced grid over ``[0, 2*pi]``.
            sarr (array-like, optional): Radial coordinate samples in ``[-1, 1]``.
                Defaults to an evenly spaced grid.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            **kwargs: Forwarded to ``matplotlib``'s ``pcolor``.
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_jacobian(
            xin,
            self._spectre,
            ax=ax,
            volumes=volumes,
            zeta=zeta,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            **kwargs,
        )

    def plot_grid(
        self,
        xin=None,
        ax=None,
        volumes=None,
        zeta=0.0,
        thetas=None,
        sarr=None,
        figsize=None,
        **kwargs,
    ):
        """Plot the ``(s, theta)`` coordinate grid on a cross-section of ``xin``.

        Args:
            xin (ndarray, optional): Packed interface degrees of freedom defining
                the geometry. Defaults to the instance's current ``xin``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure and
                axes are created if not given.
            volumes (array-like, optional): Indices of the volumes to draw. Defaults
                to all volumes. ``-1`` refers to the outermost volume.
            zeta (float, optional): Toroidal angle of the cross-section. Defaults to
                ``0.0``.
            thetas (array-like, optional): Poloidal angles to sample. Defaults to an
                evenly spaced grid over ``[0, 2*pi]``.
            sarr (array-like, optional): Radial coordinate samples in ``[-1, 1]``.
                Defaults to an evenly spaced grid.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            **kwargs: Forwarded to the underlying ``matplotlib`` plot calls
                (e.g. ``c``/``color``, ``ls``/``linestyle``, ``lw``, ``ms``).
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_grid(
            xin,
            self._spectre,
            ax=ax,
            volumes=volumes,
            zeta=zeta,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            **kwargs,
        )

    def plot_true_mesh(
        self,
        vvol=0,
        ax=None,
        ind_z=0,
        ind_i=0,
        ind_j=0,
        figsize=None,
        color_r="k",
        color_th="r",
        **kwargs,
    ):
        """Plot the true (Fortran-computed) mesh of a single volume.

        Unlike :meth:`plot_grid`, the mesh is read from the Fortran ``get_mesh``
        routine for the geometry currently in memory rather than reconstructed from
        a supplied ``xin``.

        Args:
            vvol (int, optional): Volume index to plot. Defaults to ``0``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure and
                axes are created if not given.
            ind_z (int, optional): Toroidal slice index. Defaults to ``0``.
            ind_i (int, optional): Mesh index passed to ``get_mesh``. Defaults to ``0``.
            ind_j (int, optional): Mesh index passed to ``get_mesh``. Defaults to ``0``.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            color_r (str, optional): Colour of the radial grid lines. Defaults to ``"k"``.
            color_th (str, optional): Colour of the poloidal grid lines. Defaults to ``"r"``.
            **kwargs: Forwarded to the underlying ``matplotlib`` plot calls.

        Returns:
            tuple: ``(rij_shpd, zij_shpd)`` arrays of the mesh coordinates.
        """
        return _plotting.plot_true_mesh(
            self._spectre,
            vvol=vvol,
            ax=ax,
            ind_z=ind_z,
            ind_i=ind_i,
            ind_j=ind_j,
            figsize=figsize,
            color_r=color_r,
            color_th=color_th,
            **kwargs,
        )
