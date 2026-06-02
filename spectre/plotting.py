# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from spectre.geometry import get_coord_transform, get_jacobian
from spectre.utils import _unpack_interfaces_impl


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
    **kwargs,
):
    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    mvol = test.allglobal_mod.mvol

    irbc, izbs, irbs, izbc = _unpack_interfaces_impl(xin, test)

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


def plot_jacobian(
    xin, test, ax=None, volumes=None, zeta=0.0, thetas=None, sarr=None, figsize=None, **kwargs
):
    fontsize_labels = 13

    geometry = test.input_list_mod.igeometry
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    nvol = test.allglobal_mod.mvol

    if volumes is None:
        volumes = np.arange(nvol)
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 92)
    if sarr is None:
        sarr = np.linspace(-1.0, 1.0, 14)

    sarr = (sarr[1:] + sarr[:-1]) * 0.5  # for plotting take 's' at mid-points

    irbc, izbs, irbs, izbc = _unpack_interfaces_impl(xin, test)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    else:
        _ = plt.gcf()

    jacs = []
    min_jac = 1.0e10
    max_jac = -1.0e10
    for ind in volumes:
        if ind == -1:
            ind = nvol - 1

        jac = get_jacobian(
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
        jacs.append(jac)

        min_jac = min(min_jac, np.min(jac))
        max_jac = max(max_jac, np.max(jac))

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

        image = ax.pcolor(
            rarr[0, :, :, 0],
            zarr[0, :, :, 0],
            jacs[i][:, :, 0],
            shading="nearest",
            vmin=min_jac,
            vmax=max_jac,
            **kwargs,
        )

    if geometry == 3:
        ax.set_xlabel("R [m]", fontsize=fontsize_labels)
        ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
    elif geometry == 2:
        ax.set_xlabel("x [m]", fontsize=fontsize_labels)
        ax.set_ylabel("y [m]", fontsize=fontsize_labels)
    elif geometry == 1:
        ax.set_xlabel("$\\theta$", fontsize=fontsize_labels)
        ax.set_ylabel("r", fontsize=fontsize_labels)

    plt.colorbar(image, ax=ax)

    ax.axis("equal")
    ax.set_adjustable("datalim")

    plt.tight_layout()


def plot_grid(
    xin, test, ax=None, volumes=None, zeta=0.0, thetas=None, sarr=None, figsize=None, **kwargs
):
    kwargs.setdefault("c", "k")
    kwargs.setdefault("ls", "-")
    kwargs.setdefault("lw", 0.5)
    kwargs.setdefault("ms", 3)
    kwargs.setdefault("marker", ".")

    if "color" in kwargs.keys():
        kwargs["c"] = kwargs.pop("color")
    if "linestyle" in kwargs.keys():
        kwargs["ls"] = kwargs.pop("linestyle")

    fontsize_labels = 13

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

    irbc, izbs, irbs, izbc = _unpack_interfaces_impl(xin, test)

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

    if geometry == 3:
        ax.set_xlabel("R [m]", fontsize=fontsize_labels)
        ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
    elif geometry == 2:
        ax.set_xlabel("x [m]", fontsize=fontsize_labels)
        ax.set_ylabel("y [m]", fontsize=fontsize_labels)
    elif geometry == 1:
        ax.set_xlabel("$\\theta$", fontsize=fontsize_labels)
        ax.set_ylabel("r", fontsize=fontsize_labels)

    ax.axis("equal")
    ax.set_adjustable("datalim")

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
