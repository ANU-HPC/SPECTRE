# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np

from spectre.fourier import spec_fft_fast
from spectre.utils import infaces_unflatten, infaces_flatten
from spectre.geometry import get_rz_drzdxin


def angular_force_real(xin, test, refind_fields=True):
    nvol = test.input_list_mod.nvol
    ntz = test.allglobal_mod.ntz

    if refind_fields:
        ngdof = test.allglobal_mod.ngdof

        f = np.zeros([ngdof + 1], dtype=np.float64, order="F")
        test.forces_mod.dforce(ngdof, xin, f, False, True)

    II = np.zeros((nvol, ntz), dtype=np.float64, order="F")  # [0,1,2,3]
    LL = np.zeros((nvol, 2, ntz), dtype=np.float64, order="F")

    test.wrapper_funcs_mod.get_angle_constraint(II, LL)

    return II, LL


def angular_force_likespec(xin, test, refind_fields=False, epsilon=None, upsilon=None):
    II, LL = angular_force_real(xin, test, refind_fields)

    mn = test.allglobal_mod.mn
    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor
    nvol = test.input_list_mod.nvol
    nz = test.allglobal_mod.nz
    nt = test.allglobal_mod.nt

    tarr = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zarr = np.linspace(0, 2 * np.pi, nz, endpoint=False)

    II_shpd = II.reshape(nvol, nz, nt)
    _, iomn_custom, _, _ = spec_fft_fast(tarr, zarr, II_shpd, mpol, ntor)

    LL_shpd = LL.reshape(nvol, 2, nz, nt)

    _, somn_custom, _, _ = spec_fft_fast(tarr, zarr, LL_shpd, mpol, ntor)

    force = np.zeros((nvol - 1, mn - 1))

    if epsilon is None:
        epsilon = test.input_list_mod.epsilon

    sweightmod = test.allglobal_mod.sweight
    if upsilon is not None:
        sweightmod *= upsilon / test.input_list_mod.upsilon

    for i in range(nvol - 1):
        force[i, :] = (
            epsilon * iomn_custom[i, 1:]
            + somn_custom[i, 1, 1:] * sweightmod[i]
            - somn_custom[i + 1, 0, 1:] * sweightmod[i + 1]
        )

    return force.flatten()


def angular_force_real_jac(
    xin, test, add_psifactor=False, add_sweightfactor=False, refind_fields=False
):
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    ntz = test.allglobal_mod.ntz
    nvol = test.allglobal_mod.mvol
    ngdof = test.allglobal_mod.ngdof

    if refind_fields:
        f = np.zeros([ngdof + 1], dtype=np.float64, order="F")
        test.forces_mod.dforce(ngdof, xin, f, False, True)

    ang_constr_jac = np.zeros((2, ngdof, nvol - 1, ntz), dtype=np.float64, order="F")

    test.wrapper_funcs_mod.angconst_dfp200(
        ang_constr_jac, add_psifactor=add_psifactor, add_sweightfactor=add_sweightfactor
    )

    dII, dLL = ang_constr_jac.reshape((2, ngdof, nvol - 1, nz, nt))

    return dII, dLL


def angular_force_jac_likespec(xin, test, refind_fields=False, epsilon=None, upsilon=None):
    dII, dLL = angular_force_real_jac(
        xin,
        test,
        add_psifactor=True,
        add_sweightfactor=True,
        refind_fields=refind_fields,
    )

    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor
    ngdof = test.allglobal_mod.ngdof

    tarr = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zarr = np.linspace(0, 2 * np.pi, nz, endpoint=False)
    _, dII_mode, im_, in_ = spec_fft_fast(tarr, zarr, dII, mpol, ntor)
    _, dLL_mode, im_, in_ = spec_fft_fast(tarr, zarr, dLL, mpol, ntor)

    # remove harmonic 0 since that's what SPECTRE does
    dII_mode_final = dII_mode[:, :, 1:]
    dLL_mode_final = dLL_mode[:, :, 1:]

    if epsilon is None:
        epsilon = test.input_list_mod.epsilon
    if upsilon is None:
        upsilon = test.input_list_mod.upsilon

    target_shp = (ngdof, -1)

    angular_constr_jac_custom = epsilon * dII_mode_final.reshape(
        target_shp
    ) + upsilon * dLL_mode_final.reshape(target_shp)

    return angular_constr_jac_custom


def len_cons_real(x, test, add_sweightfactor=True, refind_fields=False):
    _, LL = angular_force_real(x, test, refind_fields=refind_fields)

    n_inface = test.allglobal_mod.mvol - 1
    ntz = test.allglobal_mod.ntz

    len_cons = np.zeros((n_inface * LL.shape[2]))

    sweightmod = test.allglobal_mod.sweight
    if not add_sweightfactor:
        sweightmod[:] = 1.0

    for i in range(n_inface):
        len_cons[i * ntz : (i + 1) * ntz] = (
            LL[i, 1] * sweightmod[i] - LL[i + 1, 0] * sweightmod[i + 1]
        )

    return len_cons


def len_cons_real_jac(x, test, add_psifactor=True, add_sweightfactor=True, refind_fields=False):
    upsilon = test.input_list_mod.upsilon

    _, len_jac_init = angular_force_real_jac(
        x,
        test,
        add_psifactor=add_psifactor,
        add_sweightfactor=add_sweightfactor,
        refind_fields=refind_fields,
    )

    len_jac = np.zeros(
        (
            len_jac_init.shape[1] * len_jac_init.shape[2] * len_jac_init.shape[3],
            len_jac_init.shape[0] + 1,
        )
    )
    len_jac[:, 1:] = (
        len_jac_init.reshape((len_jac_init.shape[0], len_jac_init.shape[1], -1))
        .reshape(len_jac_init.shape[0], -1)
        .T
    )

    len_jac *= upsilon

    return len_jac


def eval_spectral_width(x, test, p=None):
    num_inf = test.allglobal_mod.mvol - 1

    if p is None:
        p = test.input_list_mod.pcondense

    r, z = infaces_unflatten(x, test)

    im = test.allglobal_mod.im

    M = np.zeros((num_inf))

    for i in range(num_inf):
        nom = np.sum((im**p) * (r[i] ** 2 + z[i] ** 2))
        denom = np.sum(r[i] ** 2 + z[i] ** 2)
        M[i] = nom / denom

    return M


def eval_spectral_width_jac(x, test, p=None):
    num_inf = test.allglobal_mod.mvol - 1
    mn = test.allglobal_mod.mn

    if p is None:
        p = test.input_list_mod.pcondense

    r, z = infaces_unflatten(x, test)

    im = test.allglobal_mod.im

    dMdx = np.zeros((num_inf, 1 + num_inf * (2 * mn - 1)))

    rtemp = np.zeros_like(r)
    ztemp = np.zeros_like(z)

    for i in range(num_inf):
        nom = np.sum((im**p) * (r[i] ** 2 + z[i] ** 2))
        denom = np.sum(r[i] ** 2 + z[i] ** 2)
        # M[i] = nom / denom

        rtemp[:] = 0.0
        ztemp[:] = 0.0

        rtemp[i] = 2 * r[i] * im**p
        ztemp[i] = 2 * z[i] * im**p
        temp = infaces_flatten(rtemp, ztemp, test)

        rtemp[i] = 2 * r[i]
        ztemp[i] = 2 * z[i]
        temp2 = infaces_flatten(rtemp, ztemp, test)

        dMdx[i, :] = denom**-2 * (denom * temp - nom * temp2)

    return dMdx


def get_theta_chord_length(xin, test, thetas=None, zetas=None):
    nfp = test.input_list_mod.nfp

    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 48)
    if zetas is None:
        zetas = np.linspace(0, 2 * np.pi / nfp, 7)

    r_infaces, z_infaces, _, _ = get_rz_drzdxin(xin, test, thetas, zetas)

    length_sqrd = (r_infaces[1:] - r_infaces[:-1]) ** 2 + (z_infaces[1:] - z_infaces[:-1]) ** 2

    return np.sqrt(length_sqrd)


def get_theta_chord_length_jac(xin, test, thetas=None, zetas=None):
    nvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    nfp = test.input_list_mod.nfp
    nfp = test.input_list_mod.nfp

    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 48)
    if zetas is None:
        zetas = np.linspace(0, 2 * np.pi / nfp, 7)

    r_infaces, z_infaces, dr_drbc, dz_dzbs = get_rz_drzdxin(xin, test, thetas, zetas)

    length_sqrd = (r_infaces[1:] - r_infaces[:-1]) ** 2 + (z_infaces[1:] - z_infaces[:-1]) ** 2
    length = np.sqrt(length_sqrd)

    dR_dx = np.empty((nvol - 1, len(thetas), len(zetas), (nvol - 1) * (mn * 2 - 1) + 1))
    dZ_dx = np.empty_like(dR_dx)

    psifactor = test.allglobal_mod.psifactor.T[0:]

    rtemp = np.zeros((nvol - 1, mn))
    ztemp = np.zeros_like(rtemp)
    temp_zero = np.zeros_like(rtemp)

    for idofz in range(len(zetas)):
        for idof in range(len(thetas)):
            for indl in range(nvol - 1):
                if not (indl + 3) > nvol:
                    rtemp[indl + 1] = 0.0
                    ztemp[indl + 1] = 0.0

                rtemp[indl] = dr_drbc[indl][idof, idofz] * psifactor[indl]
                dR_dx[indl, idof, idofz] = infaces_flatten(rtemp, temp_zero, test)

                ztemp[indl] = dz_dzbs[indl][idof, idofz] * psifactor[indl]
                dZ_dx[indl, idof, idofz] = infaces_flatten(temp_zero, ztemp, test)

                if (indl + 3) > nvol:
                    continue

                rtemp[indl] = 0.0
                ztemp[indl] = 0.0

                rtemp[indl + 1] = dr_drbc[indl + 1][idof, idofz] * psifactor[indl + 1]
                dR_dx[indl, idof, idofz] -= infaces_flatten(rtemp, temp_zero, test)

                ztemp[indl + 1] = dz_dzbs[indl + 1][idof, idofz] * psifactor[indl + 1]
                dZ_dx[indl, idof, idofz] -= infaces_flatten(temp_zero, ztemp, test)

    dlength_dxin = (1 / length)[:, :, :, None] * (
        (r_infaces[:-1] - r_infaces[1:])[:, :, :, None] * dR_dx
        + (z_infaces[:-1] - z_infaces[1:])[:, :, :, None] * dZ_dx
    )

    return dlength_dxin


def _volume_metric_terms(rac, rbc, zas, zbs, im, in_, sbar, thetas, zetas, singular=False):
    """Covariant metric components on the (s, theta, zeta) grid for one volume.

    Computed for a single volume bounded by interface coefficients (rac/zas inner,
    rbc/zbs outer). Non-singular volumes use the linear radial map
    R(s) = Rac + sbar*(Rbc - Rac); with singular=True the regularized map of the
    axis volume is used instead, R(s) = Rac + fac0*(Rbc - Rac) with fac0 = sbar^m
    (sbar^2 for m = 0), mirroring compute_coordinates (geometry_mod.F90, igeometry=3).

    Returns R, Rs, Rt, Zs, Zt -- each shaped (ns, nt, nz): the R value and dR/ds,
    dR/dtheta, dZ/ds, dZ/dtheta -- plus cosA, sinA shaped (nt, nz, mn) and the
    per-mode radial factors fac0, fac1 = dfac0/ds shaped (ns, mn) for analytic
    Jacobians.
    """
    ns = sbar.size
    mn = im.size

    angle = im[None, None, :] * thetas[:, None, None] - in_[None, None, :] * zetas[None, :, None]
    cosA = np.cos(angle)  # (nt, nz, mn)
    sinA = np.sin(angle)

    if singular:
        fac0 = np.empty((ns, mn))
        fac1 = np.empty((ns, mn))
        mpos = im > 0
        fac0[:, mpos] = sbar[:, None] ** im[None, mpos]
        fac1[:, mpos] = 0.5 * im[None, mpos] * sbar[:, None] ** (im[None, mpos] - 1)
        fac0[:, ~mpos] = (sbar**2)[:, None]
        fac1[:, ~mpos] = sbar[:, None]
    else:
        fac0 = np.broadcast_to(sbar[:, None], (ns, mn))
        fac1 = np.broadcast_to(np.full(1, 0.5), (ns, mn))

    dRdiff = rbc - rac  # (mn,)
    dZdiff = zbs - zas
    dR1 = rac[None, :] + fac0 * dRdiff[None, :]  # (ns, mn)
    dZ1 = zas[None, :] + fac0 * dZdiff[None, :]

    Rs = np.tensordot(fac1 * dRdiff[None, :], cosA, axes=([1], [2]))  # (ns, nt, nz)
    Zs = np.tensordot(fac1 * dZdiff[None, :], sinA, axes=([1], [2]))

    Rt = np.tensordot(-im[None, :] * dR1, sinA, axes=([1], [2]))  # (ns, nt, nz)
    Zt = np.tensordot(im[None, :] * dZ1, cosA, axes=([1], [2]))
    R = np.tensordot(dR1, cosA, axes=([1], [2]))  # R value, needed for sqrt(g)

    return R, Rs, Rt, Zs, Zt, cosA, sinA, fac0, fac1


# --- sqrt(g) barrier on the Beltrami solve grid ------------------------------
#
# One-sided hinge h = max(0, s_floor - sqrt g), sampled on exactly the grid the
# Beltrami matrices are quadratured over: each volume's
# Gauss-Legendre radial nodes (allglobal_mod.gaussianabscissae) times the native
# (Nt, Nz) angular grid. A fold anywhere on that grid corrupts the Beltrami
# matrix elements (the solve integrates the metric over precisely these points),
# so a guard on any other grid can miss exactly the cells that matter. Stacking
# one residual per cell (~5e5 for a typical free-boundary run) into
# least_squares is prohibitive, so the hinge is aggregated per
# (volume, zeta-plane),
#   r[v, k] = sqrt( sum_{s,theta} w_s * w_ang * h^2 ),
# with w_s the Gauss weights and w_ang = (2 pi/Nt) * (2 pi/(nfp*Nz)), making
# sum(r^2) the integral of h^2 over (s, theta, zeta) summed over volumes: the
# identical scalar objective and exact gradient as per-cell stacking, just with
# coarser Gauss-Newton curvature.
#
# The coordinate-singular axis volume is excluded by default: folds there have
# not mattered in practice, so penalizing them only biases the optimization.
# When volume 0 is requested explicitly, its healthy sqrt g vanishes linearly at
# the coordinate singularity (sqrt g ~ c*sbar with the m=1 modes dominating), so
# a flat floor would keep the innermost Gauss nodes permanently active; there
# the floor is scaled, h = max(0, s_floor*sbar - sqrt g), bounding the slope
# sqrt(g)/sbar instead.


def _solvegrid(test):
    """Beltrami solve grid: per-volume Gauss nodes/weights plus the native angles.

    Returns (snodes, sweights, thetas, zetas) where snodes/sweights are lists of
    per-volume arrays of length Iquad[lvol] and thetas/zetas are the native
    real-space angular grids theta_j = 2 pi j / Nt, zeta_k = (2 pi / nfp) k / Nz
    (matching initialize_internal_arrays in preset_mod.F90).
    """
    nfp = test.input_list_mod.nfp
    nt = int(test.allglobal_mod.nt)
    nz = int(test.allglobal_mod.nz)
    thetas = 2.0 * np.pi * np.arange(nt) / nt
    zetas = (2.0 * np.pi / nfp) * np.arange(nz) / nz

    iquad = np.atleast_1d(test.allglobal_mod.iquad).astype(int)
    absc = test.allglobal_mod.gaussianabscissae  # (maxIquad, Mvol)
    wght = test.allglobal_mod.gaussianweight
    mvol = int(test.allglobal_mod.mvol)
    snodes = [absc[: iquad[v], v].copy() for v in range(mvol)]
    sweights = [wght[: iquad[v], v].copy() for v in range(mvol)]
    return snodes, sweights, thetas, zetas


def get_solvegrid_jbar(xin, test, s_floor=1e-3, volumes=None):
    """sqrt(g)-barrier residual on the Beltrami solve grid, aggregated per zeta-plane.

    Evaluates h = max(0, s_floor - sqrt g) on exactly the grid the Beltrami solver
    quadratures over (see module comment above) and aggregates it per volume and
    zeta-plane, r[v, k] = sqrt(sum_{s,theta} w h^2), so that sum(r**2) equals the
    (s, theta, zeta)-integral of h^2 summed over volumes. By default the
    coordinate-singular axis volume is excluded: folds there have not mattered in
    practice, so penalizing them only biases the optimization. Pass volumes
    explicitly to include volume 0 (regularized radial map); its floor is then
    scaled to s_floor*sbar because the healthy sqrt g vanishes linearly at the
    axis (s_floor there bounds the slope sqrt(g)/sbar).

    Args:
        xin (ndarray): Packed interface DOF vector.
        test (SPECTRE): SPECTRE instance (geometry state must correspond to ``xin``
            for the axis volume, which reads the coordinate axis from ``irbc``).
        s_floor (float): Hinge threshold; cells with sqrt g >= s_floor (axis
            volume: sqrt g >= s_floor*sbar) contribute 0.
        volumes (list, optional): Volumes to include; defaults to all non-axis
            volumes, range(1, Mvol).

    Returns:
        ndarray: Aggregated hinge residuals shaped (len(volumes), Nz).
    """
    if volumes is None:
        volumes = list(range(1, int(test.allglobal_mod.mvol)))
    snodes, sweights, thetas, zetas = _solvegrid(test)
    nz = zetas.size
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp

    if geometry != 3:
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print("WARNING: get_solvegrid_jbar is only defined for toroidal geometry (igeometry=3)")
        return np.zeros((len(volumes), nz))

    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    w_ang = (2.0 * np.pi / thetas.size) * (2.0 * np.pi / (nfp * nz))
    out = np.empty((len(volumes), nz))
    for iv, lvol in enumerate(volumes):
        sbar = (snodes[lvol] + 1.0) / 2.0
        R, Rs, Rt, Zs, Zt, *_ = _volume_metric_terms(
            rbc[lvol],
            rbc[lvol + 1],
            zbs[lvol],
            zbs[lvol + 1],
            im,
            in_,
            sbar,
            thetas,
            zetas,
            singular=(lvol == 0),
        )
        sqrtg = R * (Rt * Zs - Rs * Zt)
        floor = s_floor * sbar[:, None, None] if lvol == 0 else s_floor
        h = np.maximum(0.0, floor - sqrtg)
        out[iv] = np.sqrt(np.einsum("ijk,i->k", h * h, sweights[lvol] * w_ang))
    return out


def get_solvegrid_jbar_jac(xin, test, s_floor=1e-3, volumes=None):
    """Analytic Jacobian of get_solvegrid_jbar w.r.t. the interface Fourier DOFs.

    Returns (len(volumes), Nz, (Mvol-1)*(2*mn-1)+1). Rows with zero residual have a
    zero gradient (one-sided hinge subgradient); elsewhere
    dr/dX = -(1/r) * sum_{s,theta} w h d(sqrt g)/dX, assembled without materializing
    the per-cell Jacobian: weighted mode contractions produce (Nz, mn) blocks that
    are scattered into the packed DOF layout via psifactor and infaces_flatten.

    volumes defaults to all non-axis volumes, matching get_solvegrid_jbar. When
    the axis volume is included explicitly, its coordinate axis is treated as
    frozen: packxi recomputes the axis from interface 1 (rzaxis) on every unpack,
    and that d(axis)/d(interface 1) chain is neglected here; the Jacobian is
    exact for all other volumes.
    """
    if volumes is None:
        volumes = list(range(1, int(test.allglobal_mod.mvol)))
    snodes, sweights, thetas, zetas = _solvegrid(test)
    nz = zetas.size
    nvol = int(test.allglobal_mod.mvol)
    mn = test.allglobal_mod.mn
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry
    nfp = test.input_list_mod.nfp
    ndof = (nvol - 1) * (mn * 2 - 1) + 1

    if geometry != 3:
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print(
                "WARNING: get_solvegrid_jbar_jac is only defined for toroidal geometry "
                "(igeometry=3)"
            )
        return np.zeros((len(volumes), nz, ndof))

    # psifactor.T row i-1 holds interface i's DOF scaling (matches infaces_unflatten)
    psifac = test.allglobal_mod.psifactor.T

    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    def contract(field, gfac, trig):
        # sum_{s,theta} field[s,t,z] * gfac[s,m] * trig[t,z,m] -> (nz, mn)
        T = np.tensordot(gfac, field, axes=(0, 0))  # (mn, nt, nz)
        return np.einsum("mjk,jkm->km", T, trig)

    w_ang = (2.0 * np.pi / thetas.size) * (2.0 * np.pi / (nfp * nz))
    jac = np.zeros((len(volumes), nz, ndof))
    for iv, lvol in enumerate(volumes):
        sbar = (snodes[lvol] + 1.0) / 2.0
        R, Rs, Rt, Zs, Zt, cosA, sinA, fac0, fac1 = _volume_metric_terms(
            rbc[lvol],
            rbc[lvol + 1],
            zbs[lvol],
            zbs[lvol + 1],
            im,
            in_,
            sbar,
            thetas,
            zetas,
            singular=(lvol == 0),
        )
        J2 = Rt * Zs - Rs * Zt
        sqrtg = R * J2
        floor = s_floor * sbar[:, None, None] if lvol == 0 else s_floor
        h = np.maximum(0.0, floor - sqrtg)
        W = sweights[lvol] * w_ang  # (ns,)
        r = np.sqrt(np.einsum("ijk,i->k", h * h, W))  # (nz,)

        # r dr/dX = sum W h dh/dX = -sum (W h) d(sqrt g)/dX (the floor is
        # x-independent); h = 0 off the hinge, so the active-cell mask is
        # implicit in A.
        A = W[:, None, None] * h  # (ns, nt, nz)
        F1 = A * J2
        F2 = A * R * Zs
        F3 = A * R * Zt
        F4 = A * R * Rt
        F5 = A * R * Rs
        g0a = 1.0 - fac0  # inner-interface radial factor, (ns, mn)
        g0b = fac0
        im_g0a = im[None, :] * g0a
        im_g0b = im[None, :] * g0b

        # d(sqrt g)/dX per mode: dR*J2 + R*(dRt*Zs - dRs*Zt) for R-coeffs,
        # R*(Rt*dZs - Rs*dZt) for Z-coeffs, with dR = fac*cosA, dRs = +-fac1*cosA,
        # dRt = -+ m*fac*sinA, dZs = +-fac1*sinA, dZt = +- m*fac*cosA.
        dRa = contract(F1, g0a, cosA) + contract(F3, fac1, cosA) - contract(F2, im_g0a, sinA)
        dRb = contract(F1, g0b, cosA) - contract(F3, fac1, cosA) - contract(F2, im_g0b, sinA)
        dZa = -contract(F4, fac1, sinA) - contract(F5, im_g0a, cosA)
        dZb = contract(F4, fac1, sinA) - contract(F5, im_g0b, cosA)

        inv_r = np.divide(1.0, r, out=np.zeros_like(r), where=r > 0.0)
        rtemp = np.zeros((nz, nvol - 1, mn))
        ztemp = np.zeros((nz, nvol - 1, mn))
        if lvol >= 1:  # inner interface lvol is a DOF (the axis is not)
            rtemp[:, lvol - 1, :] = -dRa * inv_r[:, None] * psifac[lvol - 1][None, :]
            ztemp[:, lvol - 1, :] = -dZa * inv_r[:, None] * psifac[lvol - 1][None, :]
        if lvol + 1 <= nvol - 1:  # outer interface lvol+1 is a DOF (not the CB)
            rtemp[:, lvol, :] += -dRb * inv_r[:, None] * psifac[lvol][None, :]
            ztemp[:, lvol, :] += -dZb * inv_r[:, None] * psifac[lvol][None, :]

        jac[iv] = infaces_flatten(rtemp, ztemp, test)

    return jac
