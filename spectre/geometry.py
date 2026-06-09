# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np

from spectre.config import run_decorator
from spectre.utils import infaces_unflatten


def get_interface_geometry(xin, test, volumes=None, zeta=0.0, thetas=None, sarr=None):
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

    test.xin = xin
    irbc, izbs, irbs, izbc = test.unpacked_interfaces

    for ind in volumes:
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

    return rarr[0, :, :, :], zarr[0, :, :, :]


def get_coord_transform(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta
                    (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
    Returns:
                    Rarr
                    Zarr

    Stell. nonsymmetric eqilibria not supported!
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    Rarr, Zarr = get_coord_transform_helper(
        geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
    )

    return Rarr, Zarr


@run_decorator
def get_coord_transform_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta
                                    (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
    Returns:
                                    Rarr
                                    Zarr
    """

    sbar = (sarr + 1.0) / 2.0
    ns = len(sarr)
    nt = len(tarr)
    nz = len(zarr)

    Rarr = np.empty((4, ns, nt, nz))
    Zarr = np.empty((4, ns, nt, nz))

    fac0 = np.empty((len(im), ns))
    fac1 = np.empty((len(im), ns))

    if lvol > 0 or geometry == 1:  # no coordinate singularity
        fac0[:] = sbar[None, :]
        fac1[:] = 0.5

    else:  # with coordinate singularity
        if geometry == 2:  # cylindrical
            for s in range(ns):
                fac0[:, s] = sbar[s] ** (im + 1.0)
                fac1[:, s] = (im + 1.0) * 0.5 * sbar[s] ** (im)
            fac0[im == 0] = sbar
            fac1[im == 0] = 0.5
        elif geometry == 3:  # toroidal
            for s in range(ns):
                fac0[:, s] = sbar[s] ** im
                fac1[:, s] = im * 0.5 * sbar[s] ** (im - 1.0)
            fac0[im == 0] = sbar**2
            fac1[im == 0] = sbar

    if geometry == 3:
        for s in range(ns):
            dR1 = Rac + fac0[:, s] * (Rbc - Rac)
            dZ1 = Zas + fac0[:, s] * (Zbs - Zas)
            for t in range(nt):
                for z in range(nz):
                    cos = np.cos(im * tarr[t] - in_ * zarr[z])
                    sin = np.sin(im * tarr[t] - in_ * zarr[z])

                    Rarr[0, s, t, z] = np.sum(dR1 * cos)
                    Rarr[1, s, t, z] = np.sum(fac1[:, s] * (Rbc - Rac) * cos)
                    Rarr[2, s, t, z] = np.sum(-im * dR1 * sin)
                    Rarr[3, s, t, z] = np.sum(in_ * dR1 * sin)

                    Zarr[0, s, t, z] = np.sum(dZ1 * sin)
                    Zarr[1, s, t, z] = np.sum(fac1[:, s] * (Zbs - Zas) * sin)
                    Zarr[2, s, t, z] = np.sum(im * dZ1 * cos)
                    Zarr[3, s, t, z] = np.sum(-in_ * dZ1 * cos)
    else:
        for s in range(ns):
            dR1 = Rac + fac0[:, s] * (Rbc - Rac)
            for t in range(nt):
                for z in range(nz):
                    cos = np.cos(im * tarr[t] - in_ * zarr[z])
                    sin = np.sin(im * tarr[t] - in_ * zarr[z])

                    Rarr[0, s, t, z] = np.sum(dR1 * cos)
                    Rarr[1, s, t, z] = np.sum(fac1[:, s] * (Rbc - Rac) * cos)
                    Rarr[2, s, t, z] = np.sum(-im * dR1 * sin)
                    Rarr[3, s, t, z] = np.sum(in_ * dR1 * sin)

    return Rarr, Zarr


def get_coord_deriv(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta
                    (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
    Returns:
                    Rarr
                    Zarr

    Stell. nonsymmetric eqilibria not supported!
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    Rarr, Zarr, dRarr, dZarr = get_coord_deriv_helper(
        geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
    )

    return Rarr, Zarr, dRarr, dZarr


@run_decorator
def get_coord_deriv_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta
                                    (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
    Returns:
                                    Rarr
                                    Zarr
    """

    sbar = (sarr + 1.0) / 2.0
    ns = len(sarr)
    nt = len(tarr)
    nz = len(zarr)

    Rarr = np.empty((ns, nt, nz))
    Zarr = np.empty((ns, nt, nz))

    dRarr = np.empty((ns, nt, nz, len(im)))
    dZarr = np.empty((ns, nt, nz, len(im)))

    fac0 = np.empty((len(im), ns))

    if lvol > 0 or geometry == 1:  # no coordinate singularity
        fac0[:] = sbar[None, :]

    else:  # with coordinate singularity
        if geometry == 3:  # toroidal
            for s in range(ns):
                fac0[:, s] = sbar[s] ** im
            fac0[im == 0] = sbar**2

    if geometry == 3:
        for s in range(ns):
            dR1 = Rac + fac0[:, s] * (Rbc - Rac)
            d2_dR1 = 1.0 + fac0[:, s]
            dZ1 = Zas + fac0[:, s] * (Zbs - Zas)

            for t in range(nt):
                for z in range(nz):
                    cos = np.cos(im * tarr[t] - in_ * zarr[z])
                    sin = np.sin(im * tarr[t] - in_ * zarr[z])

                    Rarr[s, t, z] = np.sum(dR1 * cos)
                    Zarr[s, t, z] = np.sum(dZ1 * sin)

                    dRarr[s, t, z] = 0.5 * d2_dR1 * cos
                    dZarr[s, t, z] = 0.5 * d2_dR1 * sin

    return Rarr, Zarr, dRarr, dZarr


def get_jacobian(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):
    if geometry == 1:
        raise ValueError("Not implemented for geometry==1")

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    return get_jacobian_helper_v2(geometry, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol)


@run_decorator
def get_jacobian_helper_v2(geometry, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol):
    jac = np.empty((len(sarr), len(tarr), len(zarr)))

    Rarr, Zarr = get_coord_transform_helper(
        geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
    )

    if geometry == 2:  # cylindrical
        jac = Rarr[1] * Rarr[0]

    elif geometry == 3:  # toroidal
        jac = Rarr[0] * (Rarr[2] * Zarr[1] - Rarr[1] * Zarr[2])

    return jac


def get_rz_drzdxin(xin, test, thetas=None, zetas=None):
    nvol = test.allglobal_mod.mvol
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp

    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 48)
    if zetas is None:
        zetas = np.linspace(0, 2 * np.pi / nfp, 7)

    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    r_infaces = np.zeros((nvol, thetas.size, zetas.size))
    z_infaces = np.zeros_like(r_infaces)

    dr_drbc = np.zeros((nvol, thetas.size, zetas.size, len(im)))
    dz_dzbs = np.zeros((nvol, thetas.size, zetas.size, len(im)))

    for i in range(nvol):
        rarr, zarr, drarr, dzarr = get_coord_deriv(
            geometry,
            rbc[i],
            rbc[i + 1],
            zbs[i],
            zbs[i + 1],
            im,
            in_,
            1.0,
            thetas,
            zetas,
            i,
        )

        r_infaces[i] = rarr[0, :, :]
        z_infaces[i] = zarr[0, :, :]
        dr_drbc[i] = drarr[0, :, :, :]
        dz_dzbs[i] = dzarr[0, :, :, :]

    return r_infaces, z_infaces, dr_drbc, dz_dzbs


@run_decorator
def sth_to_RZ(Rac, Rbc, Zas, Zbs, Ras, Rbs, Zac, Zbc, im, in_, s, theta, zeta, singular_vol):
    """Convert (s, theta) of a point to its (R, Z) description.

    Returns:
        ndarray[float]: R, Z coordinates in an array of shape (2, len(s))
    """
    sbar = (s + 1.0) / 2.0
    npts = len(s)

    RZarr = np.empty((2, npts))

    fac0 = np.empty((len(im), npts))

    if not singular_vol:
        fac0[:] = sbar[None, :]
    else:
        for s in range(npts):
            fac0[:, s] = sbar[s] ** im
        fac0[im == 0] = sbar**2

    for s in range(npts):
        dR1c = Rac + fac0[:, s] * (Rbc - Rac)
        dZ1s = Zas + fac0[:, s] * (Zbs - Zas)
        dR1s = Ras + fac0[:, s] * (Rbs - Ras)
        dZ1c = Zac + fac0[:, s] * (Zbc - Zac)

        cos = np.cos(im * theta[s] - in_ * zeta[s])
        sin = np.sin(im * theta[s] - in_ * zeta[s])

        RZarr[0, s] = np.sum(dR1c * cos + dR1s * sin)
        RZarr[1, s] = np.sum(dZ1s * sin + dZ1c * cos)

    return RZarr


@run_decorator
def at_volume_edge(phi, sth, Aze, Ate, Azo, Ato, im, in_, lrad, mpol):
    """If s > 1.0 or s < -1.0, returns 0.0, else returns 1.0. For use in CyRK edge detection (argument layout by CyRK).

    Returns:
        float: Indicator if outside of the domain
    """
    s, _ = sth

    # make sure we don't leave the domain, since by definition -1.0 < s < 1.0
    if s > 1.0 or s < -1.0:
        return 0.0
    else:
        return 1.0
