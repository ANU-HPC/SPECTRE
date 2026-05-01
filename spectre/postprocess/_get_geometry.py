# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## Calculate geometric quantities for a SPECTRE equilibrium
## gives metric, jacobian, grid...

import numpy as np
from scipy import integrate

from spectre.postprocess.config import run_decorator


def get_jacobian(self, lvol, sarr, tarr, zarr):
    """Computes the Jacobian of the coordinate transformation at given coordinates.

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Jacobian values with shape [s, t, z].
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    geometry = self.input.physics.Igeometry
    rpol = self.input.physics.rpol
    rtor = self.input.physics.rtor
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    # Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    # Zac, Zbc = self.output.Zbc[lvol : lvol + 2]

    return get_jacobian_helper_v2(
        geometry, rpol, rtor, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol
    )


@run_decorator
def get_jacobian_helper_v1(
    geometry, rpol, rtor, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol
):

    jac = np.zeros((len(sarr), len(tarr), len(zarr)))

    sbar = np.divide(np.add(sarr, 1.0), 2.0)

    if geometry == 1:  # slab/cartesian
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                Rarr1 = np.sum(0.5 * (Rbc - Rac) * cos, axis=0)

                jac[:, t, z] = Rarr1 * rpol * rtor

    elif geometry == 2:  # cylindrical
        pass
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):
                    if lvol > 0:
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 = 0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** (im + 1.0)
                        fac1 = (im + 1.0) / 2.0 * sbar[s] ** (im)
                        fac0[im == 0] = sbar[s]
                        fac1[im == 0] = 0.5

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr1 = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)

                    jac[s, t, z] = Rarr1 * Rarr0

    elif geometry == 3:
        # pass
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):
                    if lvol > 0:
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 = 0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** im
                        fac1 = (im / 2.0) * sbar[s] ** (im - 1.0)
                        fac0[im == 0] = sbar[s] ** 2
                        fac1[im == 0] = sbar[s]

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr1 = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)
                    Rarr2 = np.sum(-im * dR1 * sin, axis=0)

                    dZ1 = Zas + fac0 * (Zbs - Zas)
                    Zarr1 = np.sum(fac1 * (Zbs - Zas) * sin, axis=0)
                    Zarr2 = np.sum(im * dZ1 * cos, axis=0)

                    jac[s, t, z] = Rarr0 * (Rarr2 * Zarr1 - Rarr1 * Zarr2)

    return jac


@run_decorator
def get_jacobian_helper_v2(
    geometry, rpol, rtor, im, in_, Rac, Rbc, Zas, Zbs, sarr, tarr, zarr, lvol
):

    jac = np.empty((len(sarr), len(tarr), len(zarr)))

    Rarr, Zarr = get_coord_transform_helper(
        geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
    )

    if geometry == 1:  # slab/cartesian
        jac = Rarr[1] * (rpol * rtor)

    elif geometry == 2:  # cylindrical
        jac = Rarr[1] * Rarr[0]

    elif geometry == 3:  # toroidal
        jac = Rarr[0] * (Rarr[2] * Zarr[1] - Rarr[1] * Zarr[2])

    return jac


def get_metric(self, lvol, sarr, tarr, zarr):
    """Computes the covariant metric tensor g_ij = e_i · e_j.

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Metric tensor with shape [3, 3, s, t, z].
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    geometry = self.input.physics.Igeometry
    rpol = self.input.physics.rpol
    rtor = self.input.physics.rtor
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    # Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    # Zac, Zbc = self.output.Zbc[lvol : lvol + 2]
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)

    # return get_metric_helper(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol)
    return get_metric_helper_v2(
        geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
    )


# slightly faster than v2
@run_decorator
def get_metric_helper_v1(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):

    sbar = np.divide(np.add(sarr, 1.0), 2.0)

    metric = np.zeros((3, 3, len(sarr), len(tarr), len(zarr)))

    if geometry == 1:  # slab/cartesian
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):
                    dR1 = Rac + sbar[s] * (Rbc - Rac)
                    Rarr = np.empty((3))
                    Rarr[0] = np.sum(0.5 * (Rbc - Rac) * cos, axis=0)
                    Rarr[1] = np.sum(-im * dR1 * sin, axis=0)
                    Rarr[2] = np.sum(in_ * dR1 * sin, axis=0)

                    for i in range(3):
                        for j in range(3):
                            metric[i, j, s, t, z] = Rarr[i] * Rarr[j]

        metric[1, 1] += rpol**2
        metric[2, 2] += rtor**2

    elif geometry == 2:  # cylindrical
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):
                    if lvol > 0:
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 = 0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** (im + 1.0)
                        fac1 = (im + 1.0) / 2.0 * sbar[s] ** (im)
                        fac0[im == 0] = sbar[s]
                        fac1[im == 0] = 0.5

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr = np.empty((3))
                    Rarr[0] = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)
                    Rarr[1] = np.sum(-im * dR1 * sin, axis=0)
                    Rarr[2] = np.sum(in_ * dR1 * sin, axis=0)

                    for i in range(3):
                        for j in range(3):
                            metric[i, j, s, t, z] = Rarr[i] * Rarr[j]

                    metric[1, 1, s, t, z] += Rarr0**2.0

        metric[2, 2] += 1.0

    elif geometry == 3:  # toroidal
        for t in range(len(tarr)):
            for z in range(len(zarr)):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])
                for s in range(len(sarr)):
                    if lvol > 0:
                        fac0 = sbar[s] ** np.ones_like(im)
                        fac1 = 0.5 ** np.ones_like(im)
                    else:
                        fac0 = sbar[s] ** im
                        fac1 = (im / 2.0) * sbar[s] ** (im - 1.0)
                        fac0[im == 0] = sbar[s] ** 2
                        fac1[im == 0] = sbar[s]

                    dR1 = Rac + fac0 * (Rbc - Rac)
                    Rarr = np.empty((3))
                    Rarr0 = np.sum(dR1 * cos, axis=0)
                    Rarr[0] = np.sum(fac1 * (Rbc - Rac) * cos, axis=0)
                    Rarr[1] = np.sum(-im * dR1 * sin, axis=0)
                    Rarr[2] = np.sum(in_ * dR1 * sin, axis=0)

                    dZ1 = Zas + fac0 * (Zbs - Zas)
                    # Zarr0 = np.sum(dZ1 * sin, axis=0)
                    Zarr = np.empty((3))
                    Zarr[0] = np.sum(fac1 * (Zbs - Zas) * sin, axis=0)
                    Zarr[1] = np.sum(im * dZ1 * cos, axis=0)
                    Zarr[2] = np.sum(-in_ * dZ1 * cos, axis=0)

                    for i in range(3):
                        for j in range(3):
                            metric[i, j, s, t, z] = Rarr[i] * Rarr[j] + Zarr[i] * Zarr[j]

                    metric[2, 2, s, t, z] += Rarr0**2.0

    return metric


@run_decorator
def get_metric_helper_v2(geometry, rpol, rtor, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):

    metric = np.empty((3, 3, len(sarr), len(tarr), len(zarr)))

    Rarr, Zarr = get_coord_transform_helper(
        geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
    )

    for i in range(3):
        for j in range(3):
            metric[i, j] = Rarr[i + 1] * Rarr[j + 1]

    if geometry == 1:  # slab/cartesian
        metric[1, 1] += rpol**2
        metric[2, 2] += rtor**2

    elif geometry == 2:  # cylindrical
        metric[1, 1] += Rarr[0] ** 2
        metric[2, 2] += 1.0

    elif geometry == 3:  # toroidal
        for i in range(3):
            for j in range(3):
                metric[i, j] += Zarr[i + 1] * Zarr[j + 1]
        metric[2, 2] += Rarr[0] ** 2

    return metric


def get_coord_transform(self, lvol, sarr, tarr, zarr):
    """Returns R (and Z for toroidal geometry) and their partial derivatives w.r.t. (s, theta, zeta).

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        tuple: (Rarr, Zarr) — each with shape [4, s, t, z], where index 0 is the coordinate value and indices 1–3 are the partial derivatives w.r.t. s, theta, zeta.
    """
    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    geometry = self.input.physics.Igeometry
    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)

    try:
        stell_sym = self.input.physics.enforce_stell_sym
    except AttributeError as _:
        stell_sym = True

    if stell_sym:
        Rarr, Zarr = get_coord_transform_helper(
            geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
        )
    else:
        Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
        Zac, Zbc = self.output.Zbc[lvol : lvol + 2]
        Rarr, Zarr = get_coord_transform_geom3nonstellsym_helper(
            Rac, Rbc, Zas, Zbs, Ras, Rbs, Zac, Zbc, im, in_, sarr, tarr, zarr, lvol
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
            # for t in range(nt):
            # 	for z in range(nz):
            # 		cos = np.cos(im * tarr[t] - in_ * zarr[z])
            # 		sin = np.sin(im * tarr[t] - in_ * zarr[z])

            # 		Rarr[0,s,t,z] = np.sum(dR1 * cos)
            # 		Rarr[1,s,t,z] = np.sum(fac1[:,s] * (Rbc - Rac) * cos)
            # 		Rarr[2,s,t,z] = np.sum(-im * dR1 * sin)
            # 		Rarr[3,s,t,z] = np.sum(in_ * dR1 * sin)

            # 		Zarr[0,s,t,z] = np.sum(dZ1 * sin)
            # 		Zarr[1,s,t,z] = np.sum(fac1[:,s] * (Zbs - Zas) * sin)
            # 		Zarr[2,s,t,z] = np.sum(im * dZ1 * cos)
            # 		Zarr[3,s,t,z] = np.sum(-in_ * dZ1 * cos)

            for t in range(nt):
                for z in range(nz):
                    r0, r1, r2, r3, z0, z1, z2, z3 = (
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    for m in range(len(im)):
                        angle = im[m] * tarr[t] - in_[m] * zarr[z]
                        cos, sin = np.cos(angle), np.sin(angle)

                        r0 += dR1[m] * cos
                        r1 += fac1[m, s] * (Rbc[m] - Rac[m]) * cos
                        r2 += -im[m] * dR1[m] * sin
                        r3 += in_[m] * dR1[m] * sin

                        z0 += dZ1[m] * sin
                        z1 += fac1[m, s] * (Zbs[m] - Zas[m]) * sin
                        z2 += im[m] * dZ1[m] * cos
                        z3 += -in_[m] * dZ1[m] * cos

                    Rarr[0, s, t, z] = r0
                    Rarr[1, s, t, z] = r1
                    Rarr[2, s, t, z] = r2
                    Rarr[3, s, t, z] = r3

                    Zarr[0, s, t, z] = z0
                    Zarr[1, s, t, z] = z1
                    Zarr[2, s, t, z] = z2
                    Zarr[3, s, t, z] = z3
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


@run_decorator
def get_coord_transform_geom3nonstellsym_helper(
    Rac, Rbc, Zas, Zbs, Ras, Rbs, Zac, Zbc, im, in_, sarr, tarr, zarr, lvol
):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta
            (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
            For non-stellarator symmetric equilibria
    Returns:
            Rarr
            Zarr
    """

    sbar = (sarr + 1.0) / 2.0
    ns = len(sarr)
    nt = len(tarr)
    nz = len(zarr)

    Rarr = np.empty((4, ns, nt, nz))
    Zarr = np.zeros((4, ns, nt, nz))

    fac0 = np.empty((len(im), ns))
    fac1 = np.empty((len(im), ns))

    if lvol > 0:  # no coordinate singularity
        fac0[:] = sbar[None, :]
        fac1[:] = 0.5

    else:  # with coordinate singularity
        for s in range(ns):
            fac0[:, s] = sbar[s] ** im
            fac1[:, s] = im * 0.5 * sbar[s] ** (im - 1.0)
        fac0[im == 0] = sbar**2
        fac1[im == 0] = sbar

    for s in range(ns):
        dR1c = Rac + fac0[:, s] * (Rbc - Rac)
        dZ1s = Zas + fac0[:, s] * (Zbs - Zas)
        dR1s = Ras + fac0[:, s] * (Rbs - Ras)
        dZ1c = Zac + fac0[:, s] * (Zbc - Zac)
        for t in range(nt):
            for z in range(nz):
                cos = np.cos(im * tarr[t] - in_ * zarr[z])
                sin = np.sin(im * tarr[t] - in_ * zarr[z])

                Rarr[0, s, t, z] = np.sum(dR1c * cos + dR1s * sin)
                Rarr[1, s, t, z] = np.sum(
                    fac1[:, s] * (Rbc - Rac) * cos + fac1[:, s] * (Rbs - Ras) * sin
                )
                Rarr[2, s, t, z] = np.sum(-im * dR1c * sin + im * dR1s * cos)
                Rarr[3, s, t, z] = np.sum(in_ * dR1c * sin - in_ * dR1s * cos)

                Zarr[0, s, t, z] = np.sum(dZ1s * sin + dZ1c * cos)
                Zarr[1, s, t, z] = np.sum(
                    fac1[:, s] * (Zbs - Zas) * sin + fac1[:, s] * (Zbc - Zac) * cos
                )
                Zarr[2, s, t, z] = np.sum(im * dZ1s * cos - im * dZ1c * sin)
                Zarr[3, s, t, z] = np.sum(-in_ * dZ1s * cos + in_ * dZ1c * sin)

    return Rarr, Zarr


def get_coord_transform_list(self, lvol, sarr, tarr, zarr):
    """Returns R and Z coordinate transforms for a list of (s, theta, zeta) points (non-stellarator-symmetric).

    Unlike get_coord_transform, all three coordinate arrays must be the same length and are evaluated point-wise rather than on a grid.

    Args:
        lvol (int): Volume index.
        sarr (array-like): Radial coordinate values.
        tarr (array-like): Poloidal coordinate values (same length as sarr).
        zarr (array-like): Toroidal coordinate values (same length as sarr).

    Raises:
        ValueError: sarr, tarr and zarr do not have the same length.

    Returns:
        tuple: (Rarr, Zarr) — each with shape [4, N], where index 0 is the coordinate value and indices 1–3 are the partial derivatives w.r.t. s, theta, zeta.
    """

    sarr = np.atleast_1d(sarr) + 0.0
    tarr = np.atleast_1d(tarr) + 0.0
    zarr = np.atleast_1d(zarr) + 0.0

    if len(sarr) != len(tarr) or len(sarr) != len(zarr):
        raise ValueError("sarr, tarr and zarr must have the same length")

    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]
    im = np.array(self.output.im, dtype=int)
    in_ = np.array(self.output.in_, dtype=int)

    Ras, Rbs = self.output.Rbs[lvol : lvol + 2]
    Zac, Zbc = self.output.Zbc[lvol : lvol + 2]
    Rarr, Zarr = get_coord_transform_geom3nonstellsym_list_helper(
        Rac, Rbc, Zas, Zbs, Ras, Rbs, Zac, Zbc, im, in_, sarr, tarr, zarr, lvol
    )

    return Rarr, Zarr


@run_decorator
def get_coord_transform_geom3nonstellsym_list_helper(
    Rac, Rbc, Zas, Zbs, Ras, Rbs, Zac, Zbc, im, in_, sarr, tarr, zarr, lvol
):
    """Return coordinate transformations: R, dR/ds, dR/dtheta, dR/dzeta
            (and for toroidal geometry: Z, dZ/ds, dZ/dtheta, dZ/dzeta)
            For non-stellarator symmetric equilibria
    Returns:
            Rarr
            Zarr
    """

    sbar = (sarr + 1.0) / 2.0
    npts = len(sarr)

    Rarr = np.empty((4, npts))
    Zarr = np.zeros((4, npts))

    fac0 = np.empty((len(im), npts))
    fac1 = np.empty((len(im), npts))

    if lvol > 0:  # no coordinate singularity
        fac0[:] = sbar[None, :]
        fac1[:] = 0.5

    else:  # with coordinate singularity
        for s in range(npts):
            fac0[:, s] = sbar[s] ** im
            fac1[:, s] = im * 0.5 * sbar[s] ** (im - 1.0)
        fac0[im == 0] = sbar**2
        fac1[im == 0] = sbar

    for s in range(npts):
        dR1c = Rac + fac0[:, s] * (Rbc - Rac)
        dZ1s = Zas + fac0[:, s] * (Zbs - Zas)
        dR1s = Ras + fac0[:, s] * (Rbs - Ras)
        dZ1c = Zac + fac0[:, s] * (Zbc - Zac)

        cos = np.cos(im * tarr[s] - in_ * zarr[s])
        sin = np.sin(im * tarr[s] - in_ * zarr[s])

        Rarr[0, s] = np.sum(dR1c * cos + dR1s * sin)
        Rarr[1, s] = np.sum(fac1[:, s] * (Rbc - Rac) * cos + fac1[:, s] * (Rbs - Ras) * sin)
        Rarr[2, s] = np.sum(-im * dR1c * sin + im * dR1s * cos)
        Rarr[3, s] = np.sum(in_ * dR1c * sin - in_ * dR1s * cos)

        Zarr[0, s] = np.sum(dZ1s * sin + dZ1c * cos)
        Zarr[1, s] = np.sum(fac1[:, s] * (Zbs - Zas) * sin + fac1[:, s] * (Zbc - Zac) * cos)
        Zarr[2, s] = np.sum(im * dZ1s * cos - im * dZ1c * sin)
        Zarr[3, s] = np.sum(-in_ * dZ1s * cos + in_ * dZ1c * sin)

    return Rarr, Zarr


def get_infaces_norm(out, tarr, zarr):
    """Calculates the unit normal vector to each interface surface.

    The normal is computed as n = (dr/dtheta) × (dr/dzeta), then normalised,
    where r = R R^ + Z Z^.

    Args:
        out (SPECTREout): SPECTRE output object.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Unit normal vectors with shape [Nvol-1, t, z, 3].
    """

    nvol = out.input.physics.Nvol

    n = np.empty((nvol - 1, len(tarr), len(zarr), 3))
    temp_n = np.empty((len(tarr), len(zarr), 3))
    # r =  np.empty((nvol-1, len(tarr), len(zarr), 3))

    sinzarr = np.sin(zarr)[None, None, :]
    coszarr = np.cos(zarr)[None, None, :]

    for il in range(nvol - 1):
        Rarr, Zarr = out.get_coord_transform(il, 1.0, tarr, zarr)

        fac1 = Rarr[2] * Zarr[3] - Rarr[3] * Zarr[2]
        fac2 = Rarr[0] * Zarr[2]

        temp_n[:, :, 0] = sinzarr * fac1 - coszarr * fac2
        temp_n[:, :, 1] = -coszarr * fac1 - sinzarr * fac2
        temp_n[:, :, 2] = Rarr[0] * Rarr[2]

        n[il] = temp_n / np.linalg.norm(temp_n, axis=-1)[..., None]

        # tempr = np.empty((len(tarr), len(zarr), 3))
        # tempr[:,:,0] = Rarr[0] * coszarr
        # tempr[:,:,1] = Rarr[0] * sinzarr
        # tempr[:,:,2] = Zarr[0]
        # r[l] = tempr

    return n


def get_infaces_dAvec(out, surfaces, tarr, zarr):
    """Calculates the (unnormalised) area element vector dA for each requested interface.

    dA = (dr/dtheta) × (dr/dzeta), where r = R R^ + Z Z^.

    Args:
        out (SPECTREout): SPECTRE output object.
        surfaces (array-like): Indices of the interfaces to evaluate.
        tarr (array-like): Poloidal coordinate values.
        zarr (array-like): Toroidal coordinate values.

    Returns:
        ndarray: Area element vectors with shape [len(surfaces), t, z, 3].
    """

    dA = np.empty((len(surfaces), len(tarr), len(zarr), 3))
    temp_dA = np.empty((len(tarr), len(zarr), 3))
    # r =  np.empty((nvol-1, len(tarr), len(zarr), 3))

    sinzarr = np.sin(zarr)[None, None, :]
    coszarr = np.cos(zarr)[None, None, :]

    for i, surf in enumerate(surfaces):
        Rarr, Zarr = out.get_coord_transform(surf, 1.0, tarr, zarr)

        fac1 = Rarr[2] * Zarr[3] - Rarr[3] * Zarr[2]
        fac2 = Rarr[0] * Zarr[2]

        temp_dA[:, :, 0] = sinzarr * fac1 - coszarr * fac2
        temp_dA[:, :, 1] = -coszarr * fac1 - sinzarr * fac2
        temp_dA[:, :, 2] = Rarr[0] * Rarr[2]

        dA[i] = temp_dA

    return dA


def calc_interfaces_area(self, surfaces, tarr, zarr):
    """Computes the surface area of each requested interface.

    Args:
        surfaces (array-like or None): Indices of the interfaces to evaluate. Defaults to all volumes if None.
        tarr (array-like): Poloidal coordinate grid for integration.
        zarr (array-like): Toroidal coordinate grid for integration.

    Raises:
        ValueError: A surface index exceeds Nvol - 1.

    Returns:
        ndarray: Surface areas with shape [len(surfaces)].
    """

    surfaces = np.arange(self.input.physics.Nvol) if surfaces is None else surfaces

    if np.any(surfaces > self.input.physics.Nvol - 1):
        raise ValueError("Invalid surfaces given")

    areas = np.empty(len(surfaces))

    for i, surf in enumerate(surfaces):
        g = self.get_metric(surf, 1.0, tarr, zarr)

        dA = np.sqrt(g[1, 1] * g[2, 2] - g[1, 2] ** 2)

        areas[i] = integrate.trapezoid(integrate.trapezoid(dA, zarr), tarr)

    return areas


def periodic_integrator_thzt(f, t, z):
    """Integrates strictly periodic f where t, z are grids with 0, d, 2d, ..., 2pi-d"""

    f_mod = np.concatenate([f, f[:, 0][:, None]], axis=1)
    t_mod = np.concatenate([t, [t[0] + 2 * np.pi]])
    z_mod = np.concatenate([z, [z[0] + 2 * np.pi]])

    # f_mod = f
    # t_mod = t
    # z_mod = z

    z_integral = integrate.trapezoid(f_mod, z_mod)

    z_integral = np.concatenate([z_integral, [z_integral[0]]])

    tot_integral = integrate.trapezoid(z_integral, t_mod)

    return tot_integral


def calc_flux_surface_average(self, farr, surfaces, tarr, zarr):
    """Computes the flux-surface average <f> = ∮ f J dθ dζ / ∮ J dθ dζ for each requested surface.

    Args:
        farr (ndarray): Scalar field values on the surface with shape [t, z].
        surfaces (array-like): Indices of the surfaces to evaluate.
        tarr (array-like): Poloidal coordinate grid matching the first axis of farr.
        zarr (array-like): Toroidal coordinate grid matching the second axis of farr.

    Raises:
        ValueError: A surface index exceeds Nvol - 1.

    Returns:
        ndarray: Flux-surface averages with shape [len(surfaces)].
    """

    if np.any(surfaces > self.input.physics.Nvol - 1):
        raise ValueError("Invalid surfaces given")

    fs_avgs = np.zeros(len(surfaces))

    for i, surf in enumerate(surfaces):
        jac = self.get_jacobian(surf, 1.0, tarr, zarr)[0]

        numerator = periodic_integrator_thzt(jac * farr, tarr, zarr)
        denominator = periodic_integrator_thzt(jac, tarr, zarr)

        fs_avgs[i] = numerator / denominator

    return fs_avgs


# def get_infaces_norm(out, tarr, zarr):
# 	# calculates normal vector n(theta. zeta) for each interface

# 	# n = d r^> / d theta * d r^> / d zeta
# 	# r^> = R * R^ + Z * Z^

# 	nvol = out.input.physics.Nvol
# 	Rarrs = np.empty((nvol-1, 4, len(tarr), len(zarr)))
# 	Zarrs = np.empty((nvol-1, 4, len(tarr), len(zarr)))

# 	for l in range(nvol-1):
# 		Rt, Zt = out.get_coord_transform(l, 1.0, tarr, zarr)
# 		Rarrs[l] = Rt[:,0]
# 		Zarrs[l] = Zt[:,0]

# 	return get_infaces_norm_helper(Rarrs, Zarrs, nvol, tarr, zarr)


# @run_decorator
# def get_infaces_norm_helper(Rarrs, Zarrs, nvol, tarr, zarr):

# 	nt = len(tarr)
# 	nz = len(zarr)

# 	n =  np.empty((nvol-1, nt, nz, 3))
# 	# r =  np.empty((nvol-1, len(tarr), len(zarr), 3))

# 	sinzarr = np.sin(zarr)
# 	coszarr = np.cos(zarr)

# 	for l in range(nvol-1):
# 		Rarr = Rarrs[l]
# 		Zarr = Zarrs[l]

# 		temp_n = n[l]

# 		for it in range(nt):
# 			for iz in range(nz):
# 				fac1 = Rarr[2,it,iz] * Zarr[3,it,iz] - Rarr[3,it,iz] * Zarr[2,it,iz]
# 				fac2 = Rarr[0,it,iz] * Zarr[2,it,iz]

# 				x = sinzarr[iz] * fac1 - coszarr[iz] * fac2
# 				y = - coszarr[iz] * fac1 - sinzarr[iz] * fac2
# 				z = Rarr[0,it,iz] * Rarr[2,it,iz]

# 				norm = np.sqrt(x**2 + y**2 + z**2)

# 				temp_n[it,iz,0] = x / norm
# 				temp_n[it,iz,1] = y / norm
# 				temp_n[it,iz,2] = z / norm
# 		# temp_n[:,:,0] = sinzarr * fac1 - coszarr * fac2
# 		# temp_n[:,:,1] = - coszarr * fac1 - sinzarr * fac2
# 		# temp_n[:,:,2] = Rarr[0] * Rarr[2]

# 		# n[l] = temp_n / np.linalg.norm(temp_n, axis=-1)[...,None]

# 		# tempr = np.empty((len(tarr), len(zarr), 3))
# 		# tempr[:,:,0] = Rarr[0] * coszarr
# 		# tempr[:,:,1] = Rarr[0] * sinzarr
# 		# tempr[:,:,2] = Zarr[0]
# 		# r[l] = tempr

# 	return n


def get_rz_coords(self, lvol, s, theta, phi):
    """Returns (R, Z) cylindrical coordinates for given flux coordinates (s, theta, zeta).

    Args:
        lvol (int): Volume index.
        s (array-like): Radial coordinate values.
        theta (array-like): Poloidal coordinate values.
        phi (array-like): Toroidal coordinate values.

    Returns:
        tuple: (Rarr, Zarr) — R and Z values with shape matching the input coordinate arrays.
    """

    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]

    return get_rz_coords_helper(
        self.input.physics.Igeometry,
        Rac,
        Rbc,
        Zas,
        Zbs,
        self.output.im,
        self.output.in_,
        s,
        theta,
        phi,
        lvol,
    )


def get_st_coords(self, lvol, r, z, phi, s_guess=0.0, t_guess=0.0):
    """Returns flux coordinates (s, theta) for given cylindrical coordinates (R, Z, phi) via root-finding.

    Args:
        lvol (int): Volume index.
        r (float or array-like): Cylindrical R coordinate.
        z (float or array-like): Cylindrical Z coordinate.
        phi (float or array-like): Toroidal angle.
        s_guess (float, optional): Initial guess for s. Defaults to 0.
        t_guess (float, optional): Initial guess for theta. Defaults to 0.

    Returns:
        tuple: (sarr, tarr) — s and theta values at the given (R, Z, phi) point.
    """

    Rac, Rbc = self.output.Rbc[lvol : lvol + 2]
    Zas, Zbs = self.output.Zbs[lvol : lvol + 2]

    return get_st_coords_helper(
        self.input.physics.Igeometry,
        Rac,
        Rbc,
        Zas,
        Zbs,
        self.output.im,
        self.output.in_,
        r,
        z,
        phi,
        lvol,
        s_guess,
        t_guess,
    )


@run_decorator
def get_rz_coords_helper(geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol):

    sbar = (sarr + 1.0) / 2.0

    fac0 = np.empty((len(im)))

    if lvol > 0 or geometry == 1:  # no coordinate singularity
        fac0[:] = sbar
    else:  # with coordinate singularity
        if geometry == 2:  # cylindrical
            fac0[:] = sbar ** (im + 1.0)
            fac0[im == 0] = sbar
        elif geometry == 3:  # toroidal
            fac0[:] = sbar**im
            fac0[im == 0] = sbar**2

    r0 = 0.0
    z0 = 0.0

    if geometry == 3:
        dR1 = Rac + fac0[:] * (Rbc - Rac)
        dZ1 = Zas + fac0[:] * (Zbs - Zas)

        for m in range(len(im)):
            angle = im[m] * tarr - in_[m] * zarr
            r0 += dR1[m] * np.cos(angle)
            z0 += dZ1[m] * np.sin(angle)
    else:
        dR1 = Rac + fac0[:] * (Rbc - Rac)

        for m in range(len(im)):
            angle = im[m] * tarr - in_[m] * zarr
            r0 += dR1[m] * np.cos(angle)

    return r0, z0


# @run_decorator
def get_st_coords_helper(
    geometry, Rac, Rbc, Zas, Zbs, im, in_, r, z, phi, lvol, s_guess=0.0, t_guess=0.0
):
    from scipy.optimize import root

    def residual(x):
        sarr, tarr = x
        r_calc, z_calc = get_rz_coords_helper(
            geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, phi, lvol
        )
        return [r_calc - r, z_calc - z]

    max_attempts = 15
    for attempt in range(max_attempts):
        if attempt == 0:
            guess = [s_guess, t_guess]
        else:
            # Random guess: s in [-1, 1], theta in [0, 2*pi]
            guess = [np.random.uniform(-1, 1), np.random.uniform(1e-5, 2 * np.pi - 1e-5)]

        sol = root(residual, guess, options=dict(xtol=1e-13))

        if sol.success:
            s_found, t_found = sol.x[0], sol.x[1]
            # Check if s is in valid range
            if -1 <= s_found <= 1:
                # print(f"bfr {t_found=} {t_found==2*np.pi=}")
                # Do theta range checking, and a dirty 2*pi equivalence check
                while t_found > 2 * np.pi:
                    t_found -= 2 * np.pi
                while t_found <= 0:
                    t_found += 2 * np.pi
                t_found = t_found if abs(t_found - 2 * np.pi) > 1e-15 else 0.0
                # print(f"after {t_found=}")
                return s_found, t_found

    raise RuntimeError(
        f"Root finding failed to find solution with s in [-1, 1] after {max_attempts} attempts"
    )


def find_which_volume(self, R, Z, phi):
    """Finds the volume containing a point in cylindrical coordinates.

    Attempts root-finding sequentially in each volume until the point
    is successfully placed. 

    Args:
        R (float): Cylindrical radial coordinate.
        Z (float): Cylindrical vertical coordinate.
        phi (float): Toroidal angle in radians.

    Returns:
        int: Index of the volume containing the point.

    Raises:
        RuntimeError: If the point ``(R, Z, phi)`` could not be placed
            in any volume, either because it lies outside the domain or
            root-finding failed in all volumes.
    """

    for lvol in range(self.input.physics.Nvol):
        try:
            s, t = self.get_st_coords(lvol, R, Z, phi)
        except RuntimeError:
            continue

        return lvol

    raise RuntimeError("Volume could not be found")
