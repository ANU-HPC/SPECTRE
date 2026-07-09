# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Grid/mesh inspection helpers for :class:`~spectre.SPECTRE`.

These thin wrappers expose the standalone routines in :mod:`spectre.plotting`
and :mod:`spectre.error_checking` through the ``spectre_obj.inspect`` namespace,
e.g. ``spectre_obj.inspect.plot_true_mesh()`` or ``spectre_obj.inspect.print_errors()``.
The geometry argument ``xin`` defaults to the instance's current ``xin`` so the
methods can be called with no arguments. The underlying implementations are left
untouched.
"""

import spectre.plotting as _plotting
import spectre.error_checking as _error_checking


class _Inspect:
    """Namespace of grid/mesh plotting helpers bound to a ``SPECTRE`` instance.

    Accessed as ``spectre_obj.inspect``. Each method delegates to the matching
    routine in :mod:`spectre.plotting`, passing the instance as the ``test``
    argument and defaulting ``xin`` to ``spectre_obj.xin``.
    """

    def __init__(self, spectre):
        self._spectre = spectre

    def print_errors(self, beltrami=True, force=True):
        """Print the per-volume field (Beltrami) error and per-interface force error.

        Delegates to :func:`spectre.error_checking.print_all_errors`.

        Both read the instance's in-memory results, so populate them first: call
        ``spectre_obj.force_real()`` (it solves the field *and* computes the force
        residual). The field/Beltrami half alone only needs a solved field.

        Args:
            beltrami (bool, optional): Include the field/Beltrami error. Defaults to True.
            force (bool, optional): Include the per-interface force error. Defaults to True.

        Returns:
            tuple: ``(beltrami_avg, beltrami_max, f_rms, f_max)`` (entries are ``None`` when
            the corresponding section is disabled or off-rank).
        """
        if force and not hasattr(self._spectre, "f_real_shpd"):
            raise RuntimeError(
                "Force errors require a force evaluation; call spectre_obj.force_real() "
                "first (it solves the field and computes the force residual)."
            )
        return _error_checking.print_all_errors(self._spectre, beltrami=beltrami, force=force)

    def beltrami_errors(self, error_type=0, print_res=True):
        """Per-volume Beltrami (field) error ``|curl B - mu B|`` (avg and max per component).

        Delegates to :func:`spectre.error_checking.get_beltrami_errors`. Requires a solved
        field (e.g. ``spectre_obj.compute_fields()`` or ``spectre_obj.force_real()``).

        Args:
            error_type (int, optional): 0 = contravariant residual; 1 = Cartesian
                (Igeometry==3 only). Defaults to 0.
            print_res (bool, optional): Print the errors in addition to returning them.
                Defaults to True.

        Returns:
            tuple: ``(beltrami_errors_avg, beltrami_errors_max)``, each shape ``(mvol, 3)``.
        """
        return _error_checking.get_beltrami_errors(
            self._spectre, error_type=error_type, print_res=print_res
        )

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
        interactive=False,
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
            interactive (bool, optional): Add a ``matplotlib`` slider sweeping ``zetan``
                from 0 to 0.5, drawing into ``ax`` when given and returning ``(fig, mainax)``.
                The view expands to the interfaces' extent over the angles visited. Defaults
                to False.
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
            interactive=interactive,
            **kwargs,
        )

    def plot_jacobian(
        self,
        xin=None,
        ax=None,
        vvol="all",
        zetan=0.0,
        thetas=None,
        sarr=None,
        figsize=None,
        interactive=False,
        min_val=None,
        **kwargs,
    ):
        """Plot the coordinate Jacobian on a cross-section, flagging ill-conditioned cells.

        By default the Jacobian is sampled on exactly the Beltrami solve grid (each
        volume's Gauss-Legendre radial nodes x the native ``Nt`` poloidal grid), so the
        picture shows precisely what the Beltrami solve and the ``freal,jbar`` barrier
        see. The colormap runs blue (healthy, large Jacobian) to red as the Jacobian
        approaches zero, with negative (folded) cells drawn in magenta. ``min_val`` sets
        the lower bound of the colour scale (default ``0``).

        Args:
            xin (ndarray, optional): Packed interface degrees of freedom defining
                the geometry. Defaults to the instance's current ``xin``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure and
                axes are created if not given.
            vvol (str | int | array-like, optional): Volumes to draw. ``'all'`` (default)
                draws every volume; an int or sequence selects specific volumes; ``-1``
                refers to the outermost volume.
            zetan (float, optional): Toroidal cross-section, in units of the field period
                (multiplied internally by ``2*pi/nfp``). Defaults to ``0.0``.
            thetas (array-like, optional): Poloidal angles to sample. Defaults to the
                native ``Nt`` nodes (plus the wrap point for a closed picture).
            sarr (array-like, optional): Radial coordinate samples in ``[-1, 1]``.
                Defaults to each volume's Gauss-Legendre quadrature nodes.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            interactive (bool, optional): Add a ``matplotlib`` slider sweeping ``zetan``
                from 0 to 0.5, drawing into ``ax`` when given and returning ``(fig, mainax)``.
                Defaults to False.
            **kwargs: Forwarded to ``matplotlib``'s ``pcolor``.
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_jacobian(
            xin,
            self._spectre,
            ax=ax,
            vvol=vvol,
            zetan=zetan,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            interactive=interactive,
            min_val=min_val,
            **kwargs,
        )

    def plot_beltrami_error(
        self,
        xin=None,
        ax=None,
        vvol="all",
        zetan=0.0,
        component=None,
        error_type=0,
        figsize=None,
        interactive=False,
        min_val=None,
        **kwargs,
    ):
        """Plot the Beltrami error ``|curl B - mu B|`` on a cross-section.

        Requires a solved field (e.g. ``spectre_obj.compute_fields()`` or
        ``spectre_obj.force_real()``).

        Args:
            xin (ndarray, optional): Packed interface degrees of freedom defining the
                geometry. Defaults to the instance's current ``xin`` (which should match
                the solved field).
            ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure and axes
                are created if not given.
            vvol (str | int | array-like, optional): Volumes to draw. ``'all'`` by default.
            zetan (float, optional): Toroidal cross-section in field-period units. Snapped
                to the nearest native toroidal grid plane. Defaults to ``0.0``.
            component (int | str, optional): Error component to plot: ``0/1/2`` or the
                strings ``'s'``/``'t'``/``'z'`` (s/theta/zeta). If ``None`` (default), the
                component attaining the max-norm of the error vector on the plane is selected
                automatically; the component plotted is stated in the title.
            error_type (int, optional): 0 = contravariant residual; 1 = Cartesian
                (Igeometry==3 only). Defaults to ``0``.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            interactive (bool, optional): Add a ``matplotlib`` slider sweeping ``zetan``
                from 0 to 0.5, drawing into ``ax`` when given and returning ``(fig, mainax)``.
                Defaults to False.
            min_val (float, optional): Lower bound of the (log) colour scale; smaller,
                non-problematic errors are clipped to the bottom colour. Defaults to the
                smallest positive value (whole range).
            **kwargs: Forwarded to ``matplotlib``'s ``pcolor``.
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_beltrami_error(
            xin,
            self._spectre,
            ax=ax,
            vvol=vvol,
            zetan=zetan,
            component=component,
            error_type=error_type,
            figsize=figsize,
            interactive=interactive,
            min_val=min_val,
            **kwargs,
        )

    def plot_worst_beltrami(
        self,
        xin=None,
        ax=None,
        vvol="all",
        component=None,
        error_type=0,
        figsize=None,
        min_val=None,
        **kwargs,
    ):
        """Plot the toroidal cross-section containing the largest Beltrami error.

        Scans all native toroidal planes across ``vvol`` and draws the worst one, marking
        the offending cell. Requires a solved field.

        Args:
            xin (ndarray, optional): Packed interface DOF. Defaults to current ``xin``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on.
            vvol (str | int | array-like, optional): Volumes to consider. ``'all'`` by default.
            component (int | str, optional): Error component (``0/1/2`` or ``'s'``/``'t'``/
                ``'z'``); ``None`` (default) ranks cells by the max-norm of the error vector
                and plots the responsible component.
            error_type (int, optional): 0 = contravariant; 1 = Cartesian. Defaults to ``0``.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            min_val (float, optional): Lower bound of the colour scale (clips small errors).
                Defaults to the smallest positive value.
            **kwargs: Forwarded to the underlying plot call.
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_worst_beltrami(
            xin,
            self._spectre,
            ax=ax,
            vvol=vvol,
            component=component,
            error_type=error_type,
            figsize=figsize,
            min_val=min_val,
            **kwargs,
        )

    def plot_worst_jacobian(
        self,
        xin=None,
        ax=None,
        vvol="all",
        nzeta=None,
        thetas=None,
        sarr=None,
        figsize=None,
        skip_center=True,
        min_val=None,
        **kwargs,
    ):
        """Plot the toroidal cross-section where the Jacobian is most ill-conditioned.

        By default samples exactly the Beltrami solve grid (per-volume Gauss radial
        nodes x native ``Nt`` thetas, scanning the native ``Nz`` planes), so what is
        flagged here is precisely what the Beltrami solve and the ``freal,jbar``
        barrier see. A plane with any negative (folded) Jacobian beats an all-positive
        one; among planes with negatives the one with the most negative cells is chosen,
        otherwise the plane with the smallest (closest to zero) Jacobian. The
        most-negative cell is marked.

        Args:
            xin (ndarray, optional): Packed interface DOF. Defaults to current ``xin``.
            ax (matplotlib.axes.Axes, optional): Axes to draw on.
            vvol (str | int | array-like, optional): Volumes to consider. ``'all'`` by default.
            nzeta (int, optional): Number of toroidal planes to scan over one field period.
                Defaults to the native ``Nz`` (the solve's toroidal grid).
            thetas (array-like, optional): Poloidal angles to sample. Defaults to the
                native ``Nt`` nodes (plus the wrap point).
            sarr (array-like, optional): Radial coordinate samples in ``[-1, 1]``.
                Defaults to each volume's Gauss-Legendre quadrature nodes.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            skip_center (bool, optional): Exclude the central volume (index 0) from the
                worst-cell search. Its Jacobian vanishes at the magnetic-axis coordinate
                singularity and axis-volume folds have not mattered in practice; it is
                still drawn for context. Defaults to True.
            min_val (float, optional): Lower bound of the colour scale. Defaults to ``0``.
            **kwargs: Forwarded to the underlying plot call.
        """
        if xin is None:
            xin = self._spectre.xin
        return _plotting.plot_worst_jacobian(
            xin,
            self._spectre,
            ax=ax,
            vvol=vvol,
            nzeta=nzeta,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            skip_center=skip_center,
            min_val=min_val,
            **kwargs,
        )

    def plot_grid(
        self,
        xin=None,
        ax=None,
        volumes=None,
        zeta=0.0,
        zetan=None,
        thetas=None,
        sarr=None,
        figsize=None,
        interactive=False,
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
            zeta (float, optional): Toroidal angle of the cross-section (radians).
                Defaults to ``0.0``.
            zetan (float, optional): Toroidal cross-section in field-period units; takes
                precedence over ``zeta`` when given.
            thetas (array-like, optional): Poloidal angles to sample. Defaults to an
                evenly spaced grid over ``[0, 2*pi]``.
            sarr (array-like, optional): Radial coordinate samples in ``[-1, 1]``.
                Defaults to an evenly spaced grid.
            figsize (tuple, optional): Figure size, used only when ``ax`` is None.
            interactive (bool, optional): Add a ``matplotlib`` slider sweeping ``zetan``
                from 0 to 0.5, drawing into ``ax`` when given and returning ``(fig, mainax)``.
                Defaults to False.
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
            zetan=zetan,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            interactive=interactive,
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

    def plot_infaces_3d(
        self,
        color_by="force",
        surfaces=None,
        n_theta=64,
        n_zeta=192,
        cmap=None,
        show=True,
        show_boundary=True,
        **kwargs,
    ):
        """Render the plasma interfaces as 3D toroidal surfaces (pyvista), coloured by a
        force-balance quantity.

        Requires a solved field (e.g. ``spectre_obj.compute_fields()`` or
        ``spectre_obj.force_real()``); raises ``RuntimeError`` otherwise. Runs the
        ``SPECTREout`` interface routines directly on the live instance via a lightweight
        field view (no full ``SPECTREout`` is constructed), so the same routine backs both
        entry points.

        Args:
            color_by (str): ``'force'`` (the optimiser target ``[[p + B^2/2]]``),
                ``'pressure_jump'``, ``'b2_inner'``, ``'b2_outer'``, ``'current'``
                (``|B_in x B_out|``), or ``'angle'`` (angle between the inner/outer field
                solutions). Defaults to ``'force'``.
            surfaces (array-like, optional): Interface indices; defaults to the internal
                force-balance interfaces.
            n_theta, n_zeta (int): Surface grid resolution (``n_zeta`` spans the full torus).
            cmap (str, optional): Colormap (default depends on ``color_by``).
            show (bool): Open the interactive window. Defaults to True.
            show_boundary (bool): Draw the outer boundary as faint context. Defaults to True.
            **kwargs: Forwarded to ``pyvista``'s ``add_mesh``.

        Returns:
            pyvista.Plotter: the plotter.
        """
        if not self._spectre.field_is_current:
            raise RuntimeError(
                "plot_infaces_3d requires a solved field; call spectre_obj.compute_fields() "
                "(or spectre_obj.force_real()) first."
            )
        from spectre.postprocess._field_view import SpectreFieldView

        view = SpectreFieldView(self._spectre)
        return view.plot_infaces3D(
            color_by=color_by,
            surfaces=surfaces,
            n_theta=n_theta,
            n_zeta=n_zeta,
            cmap=cmap,
            show=show,
            show_boundary=show_boundary,
            **kwargs,
        )
