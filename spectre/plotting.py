# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spectre.geometry import get_coord_transform, get_jacobian
from spectre.utils import unpack_interfaces
from spectre.error_checking import beltrami_error_grid


def plot_fmn(
    fmn,
    im_,
    in_,
    mpol_box=None,
    ntor_box=None,
    figsize=(7, 5),
    ax=None,
    inset_kam_surfs=None,
    vmin=-9,
    vmax=1,
):
    mpol = int(np.max(im_))
    ntor = int(np.max(in_))

    plot_msh = np.empty((mpol + 1, ntor * 2 + 1))
    plot_msh[:] = np.nan

    in_axis = np.array(in_ + ntor, dtype=int)
    plot_msh[(im_, in_axis)] = fmn

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    cmap = mpl.colormaps.get_cmap("cividis")
    cmap.set_bad(color="white")

    img = ax.pcolormesh(
        np.arange(mpol + 1), np.arange(-ntor, ntor + 1), plot_msh.T, cmap=cmap, vmin=vmin, vmax=vmax
    )

    plt.colorbar(
        img,
        ax=ax,
    )

    ax.grid(which="minor", c="k", snap=False, lw=0.5)

    if mpol_box is not None and ntor_box is not None:
        ax.vlines(
            mpol_box + 0.5,
            ymin=-ntor_box - 0.5,
            ymax=ntor_box + 0.5,
            colors="k",
            lw=3,
        )
        ax.hlines(
            [-ntor_box - 0.5, ntor_box + 0.6],
            colors="k",
            xmin=-0.5,
            xmax=mpol_box + 0.5,
            lw=3,
        )

    ax.set_xticks(np.arange(mpol) + 0.5, minor=True)
    ax.set_yticks(np.arange(-ntor, ntor + 1)[:-1] + 0.5, minor=True)

    ax.set_xticks(np.arange(mpol + 1))
    ax.set_yticks(np.arange(-ntor, ntor + 1))

    ax.set_xlabel("m", fontsize=13)
    ax.set_ylabel("n", fontsize=13)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")

    # if inset_kam_surfs is not None:

    #     axin = ax.inset_axes([0.67, -0.11, 0.32, 0.45])

    #     inset_kam_surfs(np.arange(len(fmn)), axin, lw=0.5, c="k")
    #     inset_kam_surfs([iface], axin, c="r")

    #     axin.set(
    #         xticklabels=[], yticklabels=[], xticks=[], yticks=[], ylabel="", xlabel=""
    #     )
    #     axin.set_aspect("equal", adjustable="box")
    #     axin.set_ylim(0, axin.get_ylim()[1])

    plt.tight_layout()


def plot_fmn_anim(*args, **kwargs):
    from ipywidgets import IntSlider, interact

    if "figsize" not in kwargs.keys():
        kwargs["figsize"] = None

    f = plt.figure(figsize=kwargs["figsize"])

    def plot(i):
        f.clf()
        ax = f.gca()
        plot_fmn(args[0], args[1], args[2], i, args[3], args[4], ax=ax, **kwargs)

    return interact(plot, i=IntSlider(0, 0, len(args[0]) - 1))


def plot_infaces(
    xin,
    test,
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
    """Plot the interfaces on a cross-section.

    With ``interactive=True`` the figure gets a ``zetan`` slider sweeping 0 -> 0.5 of a field
    period, the plot is drawn into ``ax`` when given, and the call returns ``(fig, mainax)``.
    """
    if interactive:
        return _zetan_slider(
            plot_infaces,
            xin,
            test,
            ax=ax,
            surfaces=surfaces,
            thetas=thetas,
            figsize=figsize,
            plot_theta_grid=plot_theta_grid,
            plot_axis=plot_axis,
            zetan_init=zetan,
            **kwargs,
        )

    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    mvol = test.allglobal_mod.mvol

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    plot_infaces_helper(
        geometry,
        nfp,
        im,
        in_,
        mvol,
        irbc,
        izbs,
        irbs,
        izbc,
        ax=ax,
        surfaces=surfaces,
        zetan=zetan,
        thetas=thetas,
        figsize=figsize,
        plot_theta_grid=plot_theta_grid,
        plot_axis=plot_axis,
        **kwargs,
    )


def plot_infaces_helper(
    geometry,
    nfp,
    im,
    in_,
    mvol,
    irbc,
    izbs,
    irbs,
    izbc,
    ax=None,
    surfaces=None,
    zetan=0.0,
    thetas=None,
    figsize=None,
    plot_theta_grid=False,
    plot_axis=False,
    **kwargs,
):
    zeta = zetan * (2 * np.pi / nfp)

    fontsize_labels = 13

    kwargs.setdefault("c", None)
    kwargs.setdefault("ls", "-")

    if "color" in kwargs.keys():
        kwargs["c"] = kwargs.pop("color")
    if "linestyle" in kwargs.keys():
        kwargs["ls"] = kwargs.pop("linestyle")

    if surfaces is None:
        surfaces = np.arange(mvol)
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 92)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        _ = plt.gcf()

    for ind in surfaces:
        if ind == -1:
            ind = mvol - 1

        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, ind],
            irbc[:, ind + 1],
            izbs[:, ind],
            izbs[:, ind + 1],
            im,
            in_,
            1.0,
            thetas,
            zeta,
            ind,
        )

        if geometry == 3:
            pts_x_next = rarr[0, 0, :, 0]
            pts_y_next = zarr[0, 0, :, 0]
        elif geometry == 2:
            r = rarr[0, 0, :, 0]
            pts_x_next = r * np.cos(thetas)
            pts_y_next = r * np.sin(thetas)
        elif geometry == 1:
            r = rarr[0, 0, :, 0]
            pts_x_next = thetas
            pts_y_next = r

        if ind == mvol - 1:
            c = kwargs["c"]
            ls = kwargs["ls"]
            kwargs["c"] = "k"
            kwargs["ls"] = "-"
            ax.plot(pts_x_next, pts_y_next, **kwargs)
            kwargs["c"] = c
            kwargs["ls"] = ls
            continue

        ax.plot(pts_x_next, pts_y_next, **kwargs)
        kwargs["label"] = ""

        if plot_theta_grid:
            # thgrid = np.array([0.07, 0.4, 1.5, 5.4])
            thgrid = np.linspace(0, 2 * np.pi, 57, endpoint=False)

            rarr, zarr = get_coord_transform(
                geometry,
                irbc[:, ind],
                irbc[:, ind + 1],
                izbs[:, ind],
                izbs[:, ind + 1],
                im,
                in_,
                1.0,
                thgrid,
                zeta,
                ind,
            )

            thgrid_pts_x = rarr[0, 0, :, 0]
            thgrid_pts_y = zarr[0, 0, :, 0]

            ax.plot(thgrid_pts_x, thgrid_pts_y, "gx", ms=3)

    if plot_axis:
        rarr_axis, zarr_axis = get_coord_transform(
            geometry,
            irbc[:, 0],
            irbc[:, 0 + 1],
            izbs[:, 0],
            izbs[:, 0 + 1],
            im,
            in_,
            -1.0,
            0.0,
            zeta,
            0,
        )
        ax.plot(rarr_axis[0, 0, :, 0], zarr_axis[0, 0, :, 0], "+", ms=4)

    if geometry == 3:
        ax.set_xlabel("R [m]", fontsize=fontsize_labels)
        ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
    elif geometry == 2:
        ax.set_xlabel("x [m]", fontsize=fontsize_labels)
        ax.set_ylabel("y [m]", fontsize=fontsize_labels)
    elif geometry == 1:
        ax.set_xlabel("$\\theta$", fontsize=fontsize_labels)
        ax.set_ylabel("r", fontsize=fontsize_labels)

    # ax.axis("equal")
    # ax.set_adjustable("datalim")
    # plt.tight_layout()


def _resolve_vvol(vvol, mvol):
    """Normalise a ``vvol`` argument to an integer array of volume indices.

    ``'all'`` expands to every volume; ``-1`` maps to the outermost volume; an int or a
    sequence is taken as-is.
    """
    if isinstance(vvol, str):
        if vvol == "all":
            return np.arange(mvol)
        raise ValueError(f"Invalid vvol={vvol!r}; expected 'all', an int, or a sequence")
    vols = np.atleast_1d(vvol).astype(int)
    vols[vols == -1] = mvol - 1
    return vols


def _jac_plot_grid(test, thetas, sarr):
    """Sample grid for the Jacobian plots: the Beltrami solve grid by default.

    Returns ``(thetas, sarr_of)`` where ``sarr_of(lvol)`` gives the radial sample
    points of volume ``lvol``. With ``thetas``/``sarr`` None these are the native
    poloidal nodes ``2 pi j / Nt`` (with the ``2 pi`` wrap point appended so the
    drawn cross-section closes) and the per-volume Gauss-Legendre abscissae -- the
    exact points the Beltrami matrices are quadratured over. Explicit arguments are
    used as sample points as given (shared by all volumes).
    """
    if thetas is None:
        nt = int(test.allglobal_mod.nt)
        thetas = np.linspace(0, 2 * np.pi, nt + 1)
    if sarr is None:
        from spectre.angle_targets import _solvegrid

        snodes, _, _, _ = _solvegrid(test)
        return thetas, lambda lvol: snodes[lvol]
    sarr = np.atleast_1d(sarr) + 0.0
    return thetas, lambda lvol: sarr


def _set_geometry_labels(ax, geometry, fontsize_labels=13):
    """Set the cross-section axis labels appropriate to ``Igeometry``."""
    if geometry == 3:
        ax.set_xlabel("R [m]", fontsize=fontsize_labels)
        ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
    elif geometry == 2:
        ax.set_xlabel("x [m]", fontsize=fontsize_labels)
        ax.set_ylabel("y [m]", fontsize=fontsize_labels)
    elif geometry == 1:
        ax.set_xlabel("$\\theta$", fontsize=fontsize_labels)
        ax.set_ylabel("r", fontsize=fontsize_labels)


def _beltrami_comp_names(error_type):
    """Names of the three Beltrami-error components for the given ``error_type``."""
    return ("R", "Z", "φ") if error_type == 1 else ("s", "θ", "ζ")


def _resolve_component(component):
    """Normalise a component selector to an int index (0/1/2) or ``None``.

    Accepts ``None`` (use the max-norm of the error vector), an int ``0/1/2``, or the
    strings ``'s'``/``'t'``/``'z'`` (s / theta / zeta).
    """
    if component is None:
        return None
    if isinstance(component, str):
        try:
            return {"s": 0, "t": 1, "z": 2}[component.lower()]
        except KeyError:
            raise ValueError(f"component string must be 's', 't', or 'z'; got {component!r}")
    component = int(component)
    if component not in (0, 1, 2):
        raise ValueError(f"component must be 0, 1, 2, 's', 't', 'z', or None; got {component!r}")
    return component


def _beltrami_planes(test, vvol, error_type=0):
    """Per-volume Beltrami-error grids for the requested volumes.

    Returns a list of ``(lvol, err4, s_nodes)`` with ``err4`` shape ``(lquad+1, 3, nz, nt)``.
    """
    mvol = test.allglobal_mod.mvol
    planes = []
    for lvol in _resolve_vvol(vvol, mvol):
        lvol = int(lvol)
        err4, s_nodes = beltrami_error_grid(test, lvol, error_type=error_type)
        planes.append((lvol, err4, s_nodes))
    return planes


def _reduce_beltrami(err4, component):
    """Reduce a ``(lquad+1, 3, nz, nt)`` error grid to a scalar field.

    ``component is None`` -> max-norm over the three components; otherwise that component.
    """
    return np.max(err4, axis=1) if component is None else err4[:, component, :, :]


def _argmax_beltrami_cell(planes, k=None, component=None):
    """Locate the largest-error cell across ``planes`` and the component responsible.

    Ranks cells by the max-norm of the error vector (``component is None``) or by a single
    component. When ``k`` is given the search is restricted to that toroidal plane. Returns
    ``(value, lvol, k, jquad, jtheta, comp)`` where ``comp`` is the dominant component at the
    winning cell (equal to ``component`` when one was requested).
    """
    best = None
    for lvol, err4, _ in planes:
        red = _reduce_beltrami(err4, component)  # (lquad+1, nz, nt)
        kk = int(np.argmax(red.max(axis=(0, 2)))) if k is None else k
        sl = red[:, kk, :]
        value = float(np.max(sl))
        if best is None or value > best[0]:
            jquad, jtheta = np.unravel_index(np.argmax(sl), sl.shape)
            comp = (
                component if component is not None else int(np.argmax(err4[jquad, :, kk, jtheta]))
            )
            best = (value, lvol, kk, int(jquad), int(jtheta), comp)
    return best


def _zetan_slider(plotfunc, *args, ax=None, zetan_init=0.0, figsize=None, **kwargs):
    """Wrap a zetan-driven plotter with a ``matplotlib`` slider over ``zetan`` in [0, 0.5].

    The cross-section is drawn into ``ax`` (a new figure and axes are created when ``ax`` is
    ``None``); a slider axes is added at the bottom of that figure. Each redraw clears the
    main axes, restores its position and removes the previous frame's colorbar (so colorbars
    neither stack nor progressively shrink the axes), then re-runs ``plotfunc``. The slider
    is stashed on the figure to keep it responsive.

    The view's data bounding box accumulates over the angles visited and is only ever
    expanded (never shrunk), so the plot doesn't jump around while sliding — it grows towards
    the cross-section's extent over all toroidal angles, respecting the plot's aspect ratio.
    The plotted scalar's colour scale accumulates expand-only the same way, so colours stay
    comparable between angles.

    Returns:
        tuple: ``(fig, mainax)`` — the figure and the (reused) axes the plot is drawn in.
    """
    from matplotlib.widgets import Slider

    if ax is None:
        fig = plt.figure(figsize=figsize)
        mainax = fig.add_axes([0.13, 0.22, 0.78, 0.70])  # leave room for the slider below
    else:
        mainax = ax
        fig = mainax.figure

    ax_zeta = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    slider = Slider(ax_zeta, "zetan", 0.0, 0.5, valinit=zetan_init)

    main_pos = mainax.get_position()  # restored each redraw to undo colorbar "stealing"
    keep = set(fig.axes)  # main axes, slider axes, and any pre-existing panels

    # Expand-only data bounding box accumulated over the angles visited.
    bbox = None
    # Expand-only colour scale of the plotted scalar (accumulated over the angles visited).
    clim = None

    def draw(zetan):
        nonlocal bbox, clim
        for a in list(fig.axes):
            if a not in keep:
                fig.delaxes(a)  # drop the previous frame's colorbar
        mainax.clear()
        mainax.set_position(main_pos)
        plt.sca(mainax)  # make this the current axes/figure so plt.colorbar lands here
        plotfunc(*args, ax=mainax, zetan=zetan, **kwargs)

        # Grow the data bounding box (expand-only) and rescale to it; going through dataLim +
        # autoscale_view respects the plot's aspect ratio (set_xlim/ylim would be overridden).
        x0, x1 = mainax.dataLim.intervalx
        y0, y1 = mainax.dataLim.intervaly
        if np.isfinite((x0, x1, y0, y1)).all():
            if bbox is not None:
                x0, y0 = min(x0, bbox[0]), min(y0, bbox[1])
                x1, y1 = max(x1, bbox[2]), max(y1, bbox[3])
            bbox = (x0, y0, x1, y1)
            mainax.dataLim.update_from_data_xy(np.array([[x0, y0], [x1, y1]]), ignore=True)
            mainax.autoscale_view()

        # Keep the scalar's colour scale consistent across angles (expand-only).
        mappables = [
            m
            for m in (*mainax.collections, *mainax.images)
            if m.norm.vmin is not None and m.norm.vmax is not None
        ]
        if mappables:
            cmin = min(m.norm.vmin for m in mappables)
            cmax = max(m.norm.vmax for m in mappables)
            if clim is not None:
                cmin, cmax = min(cmin, clim[0]), max(cmax, clim[1])
            clim = (cmin, cmax)
            for m in mappables:
                m.set_clim(cmin, cmax)

        fig.canvas.draw_idle()

    slider.on_changed(lambda val: draw(float(val)))
    draw(zetan_init)

    fig._zetan_slider = slider  # keep a reference alive so the widget keeps working
    return fig, mainax


def plot_jacobian(
    xin,
    test,
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
    """Plot the coordinate Jacobian on a cross-section, highlighting ill-conditioned cells.

    By default the Jacobian is sampled on exactly the Beltrami solve grid: each volume's
    Gauss-Legendre radial nodes (``gaussianabscissae``) times the native ``Nt`` poloidal
    grid (plus the wrap point for a closed picture) -- the same points the Beltrami
    matrices are quadratured over (and that the ``freal,jbar`` barrier guards). Explicit
    ``sarr``/``thetas`` override this and are used as sample points as given.

    The colormap runs blue (healthy, large positive Jacobian) -> red as the Jacobian
    approaches zero, with any negative (folded) cell drawn in magenta. This assumes the
    healthy Jacobian is positive. ``min_val`` sets the lower bound of the colour scale
    (default ``0``); cells below it fall to the magenta "under" colour. With
    ``interactive=True`` the figure gets a ``zetan`` slider sweeping 0 -> 0.5 of a field
    period (the colour scale then accumulates the range over all angles visited), the plot
    is drawn into ``ax`` when given, and the call returns ``(fig, mainax)``.
    """
    if interactive:
        return _zetan_slider(
            plot_jacobian,
            xin,
            test,
            ax=ax,
            vvol=vvol,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            min_val=min_val,
            zetan_init=zetan,
            **kwargs,
        )

    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    nvol = test.allglobal_mod.mvol

    zeta = zetan * (2 * np.pi / nfp)
    volumes = _resolve_vvol(vvol, nvol)

    thetas, sarr_of = _jac_plot_grid(test, thetas, sarr)

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    own_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        _ = plt.gcf()

    jacs = []
    min_jac = 1.0e10
    max_jac = -1.0e10
    for ind in volumes:
        jac = get_jacobian(
            geometry,
            irbc[:, ind],
            irbc[:, ind + 1],
            izbs[:, ind],
            izbs[:, ind + 1],
            im,
            in_,
            sarr_of(ind),
            thetas,
            zeta,
            ind,
        )
        jacs.append(jac)

        min_jac = min(min_jac, np.min(jac))
        max_jac = max(max_jac, np.max(jac))

    # blue (large J) -> red (J -> 0), magenta for negative (folded) cells.
    kwargs.setdefault("cmap", mpl.colormaps["RdBu"].with_extremes(under="magenta"))
    if "norm" not in kwargs and "vmin" not in kwargs and "vmax" not in kwargs:
        vlo = 0.0 if min_val is None else float(min_val)
        kwargs["norm"] = mpl.colors.Normalize(vmin=vlo, vmax=max(max_jac, vlo))

    image = None
    for i, ind in enumerate(volumes):
        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, ind],
            irbc[:, ind + 1],
            izbs[:, ind],
            izbs[:, ind + 1],
            im,
            in_,
            sarr_of(ind),
            thetas,
            zeta,
            ind,
        )

        image = ax.pcolor(
            rarr[0, :, :, 0],
            zarr[0, :, :, 0],
            jacs[i][:, :, 0],
            shading="nearest",
            **kwargs,
        )

    _set_geometry_labels(ax, geometry)

    plt.colorbar(image, ax=ax, extend="min")

    title = f"min J = {min_jac:.2e}"
    if min_jac < 0.0:
        title += "   NEGATIVE!"
    ax.set_title(title)

    ax.axis("equal")
    ax.set_adjustable("datalim")

    if own_fig:
        plt.tight_layout()


def plot_beltrami_error(
    xin,
    test,
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

    Requires a solved field. The residual has three components (s/theta/zeta, or R/Z/phi for
    ``error_type=1``). ``component`` may be ``0/1/2`` or the strings ``'s'``/``'t'``/``'z'``
    to isolate one; if ``None`` (default) the component attaining the max-norm of the error
    vector on this plane is selected automatically. The component actually plotted is stated
    in the title. Colours use a log scale shared across volumes; ``min_val`` sets the lower
    bound of that scale (default: the smallest positive value) so small, non-problematic
    errors below it are clipped to the bottom colour.

    The error is only defined on the native toroidal grid, so ``zetan`` is snapped to the
    nearest grid plane (reported in the title). With ``interactive=True`` the figure gets a
    ``zetan`` slider sweeping 0 -> 0.5 of a field period (the colour scale then accumulates
    the range over all angles visited), the plot is drawn into ``ax`` when given, and the
    call returns ``(fig, mainax)``.
    """
    if interactive:
        return _zetan_slider(
            plot_beltrami_error,
            xin,
            test,
            ax=ax,
            vvol=vvol,
            component=component,
            error_type=error_type,
            figsize=figsize,
            min_val=min_val,
            zetan_init=zetan,
            **kwargs,
        )

    if not test.field_is_current:
        raise RuntimeError(
            "plot_beltrami_error requires a solved field; call test.compute_fields() "
            "(or test.force_real()) first."
        )

    component = _resolve_component(component)

    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz

    # The Beltrami error only exists on the native toroidal grid; snap zetan to the nearest plane.
    k = int(round(zetan * nz)) % nz
    zeta = k * (2 * np.pi / nfp) / nz
    zetan_snapped = k / nz

    thetas = np.arange(nt + 1) * (2 * np.pi / nt)  # +1 closes the poloidal seam

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    planes = _beltrami_planes(test, vvol, error_type=error_type)

    # A single component is shown: the requested one, or the one attaining the max-norm of the
    # error vector on this plane (so the plot is unambiguous and can be labelled).
    if component is None:
        disp_comp = _argmax_beltrami_cell(planes, k=k, component=None)[5]
    else:
        disp_comp = component

    own_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        _ = plt.gcf()

    # Shared log colour scale across volumes (errors span orders of magnitude).
    slices = [err4[:, disp_comp, k, :] for _, err4, _ in planes]
    vmax = max(np.max(s) for s in slices)
    if vmax <= 0.0:
        vmax = 1.0
    if min_val is not None:
        vmin = float(min_val)
    else:
        positives = (
            np.concatenate([s[s > 0] for s in slices if np.any(s > 0)])
            if any(np.any(s > 0) for s in slices)
            else None
        )
        vmin = positives.min() if positives is not None else vmax * 1e-12
        vmin = max(vmin, vmax * 1e-12)
    if vmax <= vmin:
        vmax = vmin * 10.0
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    kwargs.setdefault("cmap", "magma")

    image = None
    for (lvol, _, s_nodes), err2d in zip(planes, slices):
        err2d = np.concatenate([err2d, err2d[:, :1]], axis=1)  # close the seam

        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, lvol],
            irbc[:, lvol + 1],
            izbs[:, lvol],
            izbs[:, lvol + 1],
            im,
            in_,
            s_nodes,
            thetas,
            zeta,
            lvol,
        )

        image = ax.pcolor(
            rarr[0, :, :, 0],
            zarr[0, :, :, 0],
            err2d,
            shading="nearest",
            norm=norm,
            **kwargs,
        )

    _set_geometry_labels(ax, geometry)

    if image is not None:
        plt.colorbar(image, ax=ax)

    comp_label = _beltrami_comp_names(error_type)[disp_comp]
    ax.set_title(f"Beltrami error |E^{comp_label}|   ζn = {zetan_snapped:.3f}")

    ax.axis("equal")
    ax.set_adjustable("datalim")

    if own_fig:
        plt.tight_layout()


def plot_worst_beltrami(
    xin,
    test,
    ax=None,
    vvol="all",
    component=None,
    error_type=0,
    figsize=None,
    min_val=None,
    **kwargs,
):
    """Plot the toroidal cross-section that contains the largest Beltrami error.

    The worst cell is found by the max-norm of the error vector, i.e. the largest of
    ``|E^s|, |E^θ|, |E^ζ|`` over every cell (or, if ``component`` is given, that component
    alone). ``component`` may be ``0/1/2`` or the strings ``'s'``/``'t'``/``'z'``. The
    cross-section of the responsible component is then drawn via :func:`plot_beltrami_error`,
    the offending cell is marked, and the component plotted is stated in the title.
    ``min_val`` clips the bottom of the colour scale (see :func:`plot_beltrami_error`).
    """
    if not test.field_is_current:
        raise RuntimeError(
            "plot_worst_beltrami requires a solved field; call test.compute_fields() "
            "(or test.force_real()) first."
        )

    component = _resolve_component(component)

    nfp = test.input_list_mod.nfp
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz

    # Rank cells by the max-norm of the error vector (or the chosen component) over all planes.
    planes = _beltrami_planes(test, vvol, error_type=error_type)
    value, lvol_w, k_w, jquad_w, jtheta_w, comp_w = _argmax_beltrami_cell(
        planes, k=None, component=component
    )
    zetan_w = k_w / nz
    s_nodes_w = next(s_nodes for lvol, _, s_nodes in planes if lvol == lvol_w)

    own_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    plot_beltrami_error(
        xin,
        test,
        ax=ax,
        vvol=vvol,
        zetan=zetan_w,
        component=comp_w,
        error_type=error_type,
        min_val=min_val,
        **kwargs,
    )

    # Mark the worst cell.
    geometry = test.input_list_mod.igeometry
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)
    zeta = k_w * (2 * np.pi / nfp) / nz
    theta_w = jtheta_w * (2 * np.pi / nt)
    rarr, zarr = get_coord_transform(
        geometry,
        irbc[:, lvol_w],
        irbc[:, lvol_w + 1],
        izbs[:, lvol_w],
        izbs[:, lvol_w + 1],
        im,
        in_,
        s_nodes_w[jquad_w],
        theta_w,
        zeta,
        lvol_w,
    )
    ax.plot(rarr[0, 0, 0, 0], zarr[0, 0, 0, 0], marker="o", mfc="none", mec="lime", ms=16, mew=2)
    comp_name = _beltrami_comp_names(error_type)[comp_w]
    ax.set_title(
        f"worst Beltrami error |E^{comp_name}| = {value:.2e}   vol {lvol_w}   ζn = {zetan_w:.3f}"
    )

    if own_fig:
        plt.tight_layout()


def plot_worst_jacobian(
    xin,
    test,
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

    By default the Jacobian is sampled on exactly the Beltrami solve grid (per-volume
    Gauss radial nodes x native ``Nt`` thetas; see :func:`plot_jacobian`) and the scan
    visits the native ``Nz`` toroidal planes -- so what this plot flags is precisely what
    the Beltrami solve (and the ``freal,jbar`` barrier) sees. A plane with any negative
    (folded) Jacobian is always considered worse than an all-positive one; among planes
    with negatives the one with the **most** negative cells wins, otherwise the plane with
    the smallest (closest to zero) Jacobian is chosen. That plane is drawn via
    :func:`plot_jacobian` and its most-negative cell is marked.

    The axis volume (index 0) always participates in the negative-cell search -- a fold
    there is a genuine problem. With ``skip_center=True`` (default) it is only excluded
    from the smallest-positive fallback, where its Jacobian legitimately vanishes toward
    the coordinate singularity and would otherwise always win. Set ``skip_center=False``
    to include it there too. ``min_val`` sets the lower bound of the colour scale (see
    :func:`plot_jacobian`).
    """
    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    nvol = test.allglobal_mod.mvol

    if nzeta is None:
        nzeta = test.allglobal_mod.nz  # native planes = the solve's toroidal grid
    thetas, sarr_of = _jac_plot_grid(test, thetas, sarr)

    volumes = _resolve_vvol(vvol, nvol)
    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    # Rank planes so that any negative (folded) Jacobian beats an all-positive plane, and
    # among planes with negatives the one with the *most* negative cells wins. With no
    # negatives anywhere this falls back to the smallest (most ill-conditioned) Jacobian.
    best = None  # (rank_key, min_jac, neg_count, zetan, lvol, js, jt)
    for zetan in np.linspace(0.0, 1.0, nzeta, endpoint=False):
        zeta = zetan * (2 * np.pi / nfp)
        neg_count = 0
        plane_min = None  # (min_jac, lvol, js, jt)
        fallback0 = None  # axis-volume min cell, used only if no other candidate exists
        for ind in volumes:
            jac = get_jacobian(
                geometry,
                irbc[:, ind],
                irbc[:, ind + 1],
                izbs[:, ind],
                izbs[:, ind + 1],
                im,
                in_,
                sarr_of(ind),
                thetas,
                zeta,
                ind,
            )[:, :, 0]
            has_neg = bool(np.any(jac < 0.0))
            neg_count += int(np.sum(jac < 0.0))
            mn = float(np.min(jac))
            js, jt = np.unravel_index(np.argmin(jac), jac.shape)
            cand = (mn, int(ind), int(js), int(jt))
            if ind == 0 and skip_center and not has_neg:
                # benign vanishing toward the coordinate singularity: not a candidate
                # (a negative cell in volume 0 IS a candidate -- folds are never benign)
                fallback0 = cand
                continue
            if plane_min is None or mn < plane_min[0]:
                plane_min = cand
        if plane_min is None:  # only the (healthy) axis volume was requested
            plane_min = fallback0

        # higher negative-cell count first, then more-negative Jacobian.
        rank_key = (neg_count, -plane_min[0])
        if best is None or rank_key > best[0]:
            best = (rank_key, plane_min[0], neg_count, float(zetan), *plane_min[1:])

    _, min_jac, neg_count, zetan_w, lvol_w, js_w, jt_w = best
    zeta_w = zetan_w * (2 * np.pi / nfp)
    rarr, zarr = get_coord_transform(
        geometry,
        irbc[:, lvol_w],
        irbc[:, lvol_w + 1],
        izbs[:, lvol_w],
        izbs[:, lvol_w + 1],
        im,
        in_,
        sarr_of(lvol_w)[js_w],
        thetas[jt_w],
        zeta_w,
        lvol_w,
    )
    rw, zw = rarr[0, 0, 0, 0], zarr[0, 0, 0, 0]

    own_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    plot_jacobian(
        xin,
        test,
        ax=ax,
        vvol=vvol,
        zetan=zetan_w,
        thetas=thetas,
        sarr=sarr,
        min_val=min_val,
        **kwargs,
    )

    ax.plot(rw, zw, marker="o", mfc="none", mec="lime", ms=16, mew=2)
    title = f"worst J = {min_jac:.2e}   vol {lvol_w}   ζn = {zetan_w:.3f}"
    if neg_count > 0:
        title += f"   ({neg_count} negative cells)"
    ax.set_title(title)

    if own_fig:
        plt.tight_layout()


def plot_grid(
    xin,
    test,
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
    """Plot the ``(s, theta)`` coordinate grid on a cross-section.

    ``zetan`` (toroidal cut in field-period units) takes precedence over ``zeta`` (radians)
    when given. With ``interactive=True`` the figure gets a ``zetan`` slider sweeping 0 -> 0.5
    of a field period, the plot is drawn into ``ax`` when given, and the call returns
    ``(fig, mainax)``.
    """
    if interactive:
        return _zetan_slider(
            plot_grid,
            xin,
            test,
            ax=ax,
            volumes=volumes,
            thetas=thetas,
            sarr=sarr,
            figsize=figsize,
            zetan_init=(zetan or 0.0),
            **kwargs,
        )

    if zetan is not None:
        zeta = zetan * (2 * np.pi / test.input_list_mod.nfp)

    kwargs.setdefault("c", "k")
    kwargs.setdefault("ls", "-")
    kwargs.setdefault("lw", 0.5)
    kwargs.setdefault("ms", 3)
    kwargs.setdefault("marker", ".")

    if "color" in kwargs.keys():
        kwargs["c"] = kwargs.pop("color")
    if "linestyle" in kwargs.keys():
        kwargs["ls"] = kwargs.pop("linestyle")

    geometry = test.input_list_mod.igeometry
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    nvol = test.allglobal_mod.mvol

    if volumes is None:
        volumes = np.arange(nvol)
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 92)
    if sarr is None:
        sarr = np.linspace(-1.0, 1.0, 10)

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    own_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        _ = plt.gcf()

    for i, ind in enumerate(volumes):
        if ind == -1:
            ind = nvol - 1

        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, ind],
            irbc[:, ind + 1],
            izbs[:, ind],
            izbs[:, ind + 1],
            im,
            in_,
            sarr,
            thetas,
            zeta,
            ind,
        )

        kwargs["c"] = "k"
        ax.plot(rarr[0, :, :, 0], zarr[0, :, :, 0], **kwargs)

        kwargs["c"] = "r"
        ax.plot(rarr[0, :, :, 0].T, zarr[0, :, :, 0].T, **kwargs)

    _set_geometry_labels(ax, geometry)

    ax.axis("equal")
    ax.set_adjustable("datalim")

    if own_fig:
        plt.tight_layout()


def plot_true_mesh(
    test,
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
    if vvol < 0 or vvol > (test.allglobal_mod.mvol - 1):
        raise RuntimeError(f"Invalid {vvol=}")

    kwargs.setdefault("ls", "-")
    kwargs.setdefault("lw", 0.5)
    kwargs.setdefault("ms", 2)
    kwargs.setdefault("marker", ".")

    if "color" in kwargs.keys():
        kwargs["c"] = kwargs.pop("color")
    if "linestyle" in kwargs.keys():
        kwargs["ls"] = kwargs.pop("linestyle")

    fontsize_labels = 13

    size = test.allglobal_mod.iquad[vvol]

    geometry = test.input_list_mod.igeometry
    ntz = test.allglobal_mod.ntz
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz

    # if np.sum(test.input_list_mod.lrad[:nvol]+1) > size:
    #     raise RuntimeError('Size has to be larger than sum(lrad)')

    rij = np.zeros((size, ntz), order="F", dtype=np.float64)
    zij = np.zeros_like(rij)

    test.wrapper_funcs_mod.get_mesh(vvol + 1, rij, zij, ind_i, ind_j)

    rij_shpd = np.empty((rij.shape[0], nz + 1, nt + 1))
    zij_shpd = np.empty_like(rij_shpd)
    rij_shpd[:, :nz, :nt] = rij.reshape(-1, nz, nt, order="C")
    zij_shpd[:, :nz, :nt] = zij.reshape(-1, nz, nt, order="C")
    rij_shpd[:, -1, :] = rij_shpd[:, 0, :]
    rij_shpd[:, :, -1] = rij_shpd[:, :, 0]
    zij_shpd[:, -1, :] = zij_shpd[:, 0, :]
    zij_shpd[:, :, -1] = zij_shpd[:, :, 0]

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        _ = plt.gcf()

    kwargs["c"] = color_r
    ax.plot(rij_shpd[:, ind_z].T, zij_shpd[:, ind_z].T, **kwargs)

    kwargs["c"] = color_th
    ax.plot(rij_shpd[:, ind_z], zij_shpd[:, ind_z], **kwargs)

    if geometry == 3:
        ax.set_xlabel("R [m]", fontsize=fontsize_labels)
        ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
    elif geometry == 2:
        ax.set_xlabel("x [m]", fontsize=fontsize_labels)
        ax.set_ylabel("y [m]", fontsize=fontsize_labels)
    elif geometry == 1:
        ax.set_xlabel("$\\theta$", fontsize=fontsize_labels)
        ax.set_ylabel("r", fontsize=fontsize_labels)

    # ax.axis("equal")
    # ax.set_adjustable("datalim")
    # plt.tight_layout()

    return rij_shpd, zij_shpd
