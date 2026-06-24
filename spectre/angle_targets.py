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


# --- s-theta grid shear (off-diagonal covariant metric) penalty -------------
#
# Penalizes the s-theta shear of the generated grid in the (non-singular) outer
# volume(s): the normalized off-diagonal covariant metric
#   cos = g_st / sqrt(g_ss * g_tt),
# which is 0 when the radial coordinate lines are perpendicular to the surfaces
# (no shear/fold) and -> +/-1 as e_s and e_theta align (sqrt(g) -> 0, a fold).
# Unlike the per-surface condensers (chord length, spectral width, SPEC eps
# force) this is a *pairwise* grid metric measured on the grid that terminates
# on the fixed computational boundary, so it can only align the moving surface
# toward CB-consistency, never away from it.
#
# IMPORTANT: the residual reconstructs the grid with the *linear* radial map
# R(s) = Rac + sbar*(Rbc - Rac) (fac1 = 0.5) used by the Beltrami grid in
# non-singular volumes; the analytic Jacobian below differentiates that map. If
# the external/map2disc mesh is active (allglobal_mod.use_ext_mesh == True) the
# radial map is different and both the residual and its Jacobian are invalid;
# freal_shear_obj warns in that case.


def _shear_metric_terms(rac, rbc, zas, zbs, im, in_, sbar, thetas, zetas):
    """Covariant metric components on the (s, theta, zeta) grid for a single
    non-singular volume bounded by interface coefficients (rac/zas inner,
    rbc/zbs outer), using the linear radial map R(s) = Rac + sbar*(Rbc - Rac).

    Returns R, Rs, Rt, Zs, Zt -- each shaped (ns, nt, nz): the R value and dR/ds,
    dR/dtheta, dZ/ds, dZ/dtheta -- plus cosA, sinA shaped (nt, nz, mn) for the
    analytic Jacobian. Mirrors get_coord_transform_helper (geometry.py) for lvol > 0.
    """
    ns = sbar.size
    nt = thetas.size
    nz = zetas.size

    angle = im[None, None, :] * thetas[:, None, None] - in_[None, None, :] * zetas[None, :, None]
    cosA = np.cos(angle)  # (nt, nz, mn)
    sinA = np.sin(angle)

    dRdiff = rbc - rac  # (mn,)
    dZdiff = zbs - zas
    dR1 = rac[None, :] + sbar[:, None] * dRdiff[None, :]  # (ns, mn)
    dZ1 = zas[None, :] + sbar[:, None] * dZdiff[None, :]

    # dR/ds, dZ/ds are s-independent for the linear map (fac1 = 0.5).
    Rs = np.broadcast_to(0.5 * np.tensordot(cosA, dRdiff, axes=([2], [0])), (ns, nt, nz))
    Zs = np.broadcast_to(0.5 * np.tensordot(sinA, dZdiff, axes=([2], [0])), (ns, nt, nz))

    Rt = np.tensordot(-im[None, :] * dR1, sinA, axes=([1], [2]))  # (ns, nt, nz)
    Zt = np.tensordot(im[None, :] * dZ1, cosA, axes=([1], [2]))
    R = np.tensordot(dR1, cosA, axes=([1], [2]))  # R value, needed for sqrt(g)

    return R, Rs, Rt, Zs, Zt, cosA, sinA


def _shear_safe_denom(denom, test, floor=1e-12):
    """Floor sqrt(g_ss*g_tt) away from zero so the normalized shear (and its
    Jacobian) stay finite at a near-degenerate grid point. Warns and continues
    rather than raising."""
    if np.any(denom < floor):
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print(
                f"WARNING: get_grid_shear: near-degenerate metric "
                f"(min sqrt(g_ss*g_tt) = {float(np.min(denom)):.2e}); "
                f"clamping denominator to {floor:.0e}"
            )
        denom = np.maximum(denom, floor)
    return denom


def _shear_defaults(test, sarr, thetas, zetas, volumes):
    nvol = test.allglobal_mod.mvol
    nfp = test.input_list_mod.nfp
    if thetas is None:
        thetas = np.linspace(0, 2 * np.pi, 24)
    if zetas is None:
        zetas = np.linspace(0, 2 * np.pi / nfp, 5)
    if sarr is None:
        # interior grid of the volume; drop the s=+1 (outer) endpoint, where the
        # vacuum-volume map depends only on the frozen CB coeffs (zero gradient).
        sarr = np.linspace(-1.0, 1.0, 9)[:-1]
    if volumes is None:
        volumes = [nvol - 1]  # outer / vacuum volume only
    return np.atleast_1d(sarr) + 0.0, thetas, zetas, list(volumes)


def get_grid_shear(xin, test, sarr=None, thetas=None, zetas=None, volumes=None, normalize=True):
    """Normalized s-theta grid shear cos = g_st/sqrt(g_ss*g_tt) (or raw g_st if
    normalize=False) evaluated on the interior (s, theta, zeta) grid of each
    target volume. Returns an array shaped (len(volumes), ns, nt, nz).

    volumes defaults to the outer/vacuum volume [Mvol-1]; the coordinate-singular
    axis volume 0 must be excluded (different radial basis)."""
    sarr, thetas, zetas, volumes = _shear_defaults(test, sarr, thetas, zetas, volumes)
    ns, nt, nz = sarr.size, thetas.size, zetas.size

    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry

    if geometry != 3:
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print("WARNING: get_grid_shear is only defined for toroidal geometry (igeometry=3)")
        return np.zeros((len(volumes), ns, nt, nz))

    sbar = (sarr + 1.0) / 2.0
    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    out = np.empty((len(volumes), ns, nt, nz))
    for iv, lvol in enumerate(volumes):
        _, Rs, Rt, Zs, Zt, _, _ = _shear_metric_terms(
            rbc[lvol], rbc[lvol + 1], zbs[lvol], zbs[lvol + 1], im, in_, sbar, thetas, zetas
        )
        g_st = Rs * Rt + Zs * Zt
        if normalize:
            denom = _shear_safe_denom(np.sqrt((Rs**2 + Zs**2) * (Rt**2 + Zt**2)), test)
            out[iv] = g_st / denom
        else:
            out[iv] = g_st
    return out


def get_grid_shear_jac(xin, test, sarr=None, thetas=None, zetas=None, volumes=None, normalize=True):
    """Analytic Jacobian of get_grid_shear w.r.t. the interface Fourier DOFs.
    Returns an array shaped (len(volumes), ns, nt, nz, (Mvol-1)*(2*mn-1)+1).

    For a non-singular volume the linear radial map is linear in the bounding
    interface coeffs, so the per-mode derivatives of the metric components are
    closed-form (fac0=sbar, fac1=0.5). Contributions are scattered into the DOF
    layout via infaces_flatten; the computational boundary (interface Mvol) is
    never a DOF, so the outer term is dropped when the outer interface is the CB."""
    sarr, thetas, zetas, volumes = _shear_defaults(test, sarr, thetas, zetas, volumes)
    ns, nt, nz = sarr.size, thetas.size, zetas.size
    npts = ns * nt * nz

    nvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry
    ndof = (nvol - 1) * (mn * 2 - 1) + 1

    if geometry != 3:
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print("WARNING: get_grid_shear_jac is only defined for toroidal geometry (igeometry=3)")
        return np.zeros((len(volumes), ns, nt, nz, ndof))

    sbar = (sarr + 1.0) / 2.0
    one_minus = 1.0 - sbar
    psifac = test.allglobal_mod.psifactor.T  # (nvol+1, mn), indexed by interface

    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    jac = np.zeros((len(volumes), ns, nt, nz, ndof))
    for iv, lvol in enumerate(volumes):
        _, Rs, Rt, Zs, Zt, cosA, sinA = _shear_metric_terms(
            rbc[lvol], rbc[lvol + 1], zbs[lvol], zbs[lvol + 1], im, in_, sbar, thetas, zetas
        )
        g_st = Rs * Rt + Zs * Zt
        g_ss = Rs**2 + Zs**2
        g_tt = Rt**2 + Zt**2

        if normalize:
            denom = _shear_safe_denom(np.sqrt(g_ss * g_tt), test)
            invD = 1.0 / denom
            coef2 = g_st / (2.0 * denom**3)

        # per-mode derivatives of the metric components, shape (ns, nt, nz, mn)
        cosA_b = cosA[None]  # broadcast over s
        sinA_b = sinA[None]
        imb = im[None, None, None, :]
        sbar_b = sbar[:, None, None, None]
        oneb = one_minus[:, None, None, None]

        dRs_dRac, dRs_dRbc = -0.5 * cosA_b, 0.5 * cosA_b
        dRt_dRac, dRt_dRbc = -imb * oneb * sinA_b, -imb * sbar_b * sinA_b
        dZs_dZas, dZs_dZbs = -0.5 * sinA_b, 0.5 * sinA_b
        dZt_dZas, dZt_dZbs = imb * oneb * cosA_b, imb * sbar_b * cosA_b

        def dres(da_s, da_t, A_s, A_t):
            # chain rule for cos = g_st/sqrt(g_ss*g_tt) through one channel
            # (R or Z): A_s, A_t are that channel's (Rs,Rt) or (Zs,Zt).
            dg_st = A_t[..., None] * da_s + A_s[..., None] * da_t
            if not normalize:
                return dg_st
            dg_ss = 2.0 * A_s[..., None] * da_s
            dg_tt = 2.0 * A_t[..., None] * da_t
            return invD[..., None] * dg_st - coef2[..., None] * (
                dg_ss * g_tt[..., None] + g_ss[..., None] * dg_tt
            )

        dcos_dRac = dres(dRs_dRac, dRt_dRac, Rs, Rt)
        dcos_dRbc = dres(dRs_dRbc, dRt_dRbc, Rs, Rt)
        dcos_dZas = dres(dZs_dZas, dZt_dZas, Zs, Zt)
        dcos_dZbs = dres(dZs_dZbs, dZt_dZbs, Zs, Zt)

        # scatter into the flattened DOF layout; physical coeff = xin*psifactor,
        # so d(res)/d(xin) = d(res)/d(coeff) * psifactor[interface].
        rtemp = np.zeros((npts, nvol - 1, mn))
        ztemp = np.zeros((npts, nvol - 1, mn))
        if lvol >= 1:  # inner interface lvol is a DOF -> flat row lvol-1
            rtemp[:, lvol - 1, :] = dcos_dRac.reshape(npts, mn) * psifac[lvol][None, :]
            ztemp[:, lvol - 1, :] = dcos_dZas.reshape(npts, mn) * psifac[lvol][None, :]
        if lvol + 1 <= nvol - 1:  # outer interface lvol+1 is a DOF (not the CB)
            rtemp[:, lvol, :] += dcos_dRbc.reshape(npts, mn) * psifac[lvol + 1][None, :]
            ztemp[:, lvol, :] += dcos_dZbs.reshape(npts, mn) * psifac[lvol + 1][None, :]

        jac[iv] = infaces_flatten(rtemp, ztemp, test).reshape(ns, nt, nz, ndof)

    return jac


# --- grid-Jacobian (sqrt g) barrier penalty ---------------------------------
#
# One-sided hinge r = max(0, s_floor - sqrt(g)) on the (s, theta, zeta) grid:
# zero where the grid is healthy (sqrt g >= s_floor), growing as sqrt g drops
# toward 0 and large when sqrt g < 0 (folded). sqrt(g) (toroidal) = R*(R_t Z_s -
# R_s Z_t). It steers the optimizer away from the near-degenerate / folded grids
# where the Beltrami solution error explodes (~sqrt(g)^-1.5; see the correlation
# study), which a *radial pinch* or *theta bunching* can cause without the s-theta
# shear cos ever flagging it (cos sees the basis-vector angle; sqrt g also sees
# their magnitudes). Being one-sided it is inactive -- and so unbiasing -- on a
# healthy grid. Same linear-radial-map assumption / use_ext_mesh caveat as the
# shear penalty.


def _jbar_default_volumes(test, volumes):
    if volumes is None:
        return list(range(1, int(test.allglobal_mod.mvol)))  # all non-singular volumes
    return volumes


def get_grid_jbar(xin, test, sarr=None, thetas=None, zetas=None, volumes=None, s_floor=1e-3):
    """sqrt(g)-barrier residual r = max(0, s_floor - sqrt(g)) on the interior grid
    of each target volume. Returns (len(volumes), ns, nt, nz). volumes defaults to
    all non-singular volumes (a pinch/bunch can occur in any of them; the axis
    volume 0 is excluded)."""
    volumes = _jbar_default_volumes(test, volumes)
    sarr, thetas, zetas, volumes = _shear_defaults(test, sarr, thetas, zetas, volumes)
    ns, nt, nz = sarr.size, thetas.size, zetas.size
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry

    if geometry != 3:
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print("WARNING: get_grid_jbar is only defined for toroidal geometry (igeometry=3)")
        return np.zeros((len(volumes), ns, nt, nz))

    sbar = (sarr + 1.0) / 2.0
    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    out = np.empty((len(volumes), ns, nt, nz))
    for iv, lvol in enumerate(volumes):
        R, Rs, Rt, Zs, Zt, _, _ = _shear_metric_terms(
            rbc[lvol], rbc[lvol + 1], zbs[lvol], zbs[lvol + 1], im, in_, sbar, thetas, zetas
        )
        sqrtg = R * (Rt * Zs - Rs * Zt)
        out[iv] = np.maximum(0.0, s_floor - sqrtg)
    return out


def get_grid_jbar_jac(xin, test, sarr=None, thetas=None, zetas=None, volumes=None, s_floor=1e-3):
    """Analytic Jacobian of get_grid_jbar w.r.t. the interface Fourier DOFs.
    Returns (len(volumes), ns, nt, nz, (Mvol-1)*(2*mn-1)+1).

    r = max(0, s_floor - sqrt g) ⇒ dr/dX = -d(sqrt g)/dX where sqrt g < s_floor
    (hinge subgradient), else 0. sqrt(g) = R*J2 with J2 = R_t Z_s - R_s Z_t, so the
    product-rule derivatives reuse the shear per-mode terms (dRs,dRt,dZs,dZt) plus
    dR; scatter via infaces_flatten with the psifactor chain factor and the
    CB-exclusion guard, identical to get_grid_shear_jac."""
    volumes = _jbar_default_volumes(test, volumes)
    sarr, thetas, zetas, volumes = _shear_defaults(test, sarr, thetas, zetas, volumes)
    ns, nt, nz = sarr.size, thetas.size, zetas.size
    npts = ns * nt * nz

    nvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    geometry = test.input_list_mod.igeometry
    ndof = (nvol - 1) * (mn * 2 - 1) + 1

    if geometry != 3:
        rank = getattr(getattr(test, "comm", None), "rank", 0)
        if rank == 0:
            print("WARNING: get_grid_jbar_jac is only defined for toroidal geometry (igeometry=3)")
        return np.zeros((len(volumes), ns, nt, nz, ndof))

    sbar = (sarr + 1.0) / 2.0
    one_minus = 1.0 - sbar
    psifac = test.allglobal_mod.psifactor.T

    rbc = test.allglobal_mod.irbc.copy().T
    zbs = test.allglobal_mod.izbs.copy().T
    rbc[1:-1], zbs[1:-1] = infaces_unflatten(xin, test, add_psifactor=True)

    jac = np.zeros((len(volumes), ns, nt, nz, ndof))
    for iv, lvol in enumerate(volumes):
        R, Rs, Rt, Zs, Zt, cosA, sinA = _shear_metric_terms(
            rbc[lvol], rbc[lvol + 1], zbs[lvol], zbs[lvol + 1], im, in_, sbar, thetas, zetas
        )
        J2 = Rt * Zs - Rs * Zt  # 2D (s,theta)->(R,Z) Jacobian; sqrt g = R*J2
        sqrtg = R * J2
        active = -(sqrtg < s_floor).astype(float)  # dr/d(sqrt g) = -1 where active

        cosA_b = cosA[None]
        sinA_b = sinA[None]
        imb = im[None, None, None, :]
        sbar_b = sbar[:, None, None, None]
        oneb = one_minus[:, None, None, None]

        dR_dRac, dR_dRbc = oneb * cosA_b, sbar_b * cosA_b
        dRs_dRac, dRs_dRbc = -0.5 * cosA_b, 0.5 * cosA_b
        dRt_dRac, dRt_dRbc = -imb * oneb * sinA_b, -imb * sbar_b * sinA_b
        dZs_dZas, dZs_dZbs = -0.5 * sinA_b, 0.5 * sinA_b
        dZt_dZas, dZt_dZbs = imb * oneb * cosA_b, imb * sbar_b * cosA_b

        def dres_R(dR, dRs, dRt):
            # d(sqrt g)/dX for an R-coeff: only R, R_s, R_t depend on it
            dsqrtg = dR * J2[..., None] + R[..., None] * (dRt * Zs[..., None] - dRs * Zt[..., None])
            return active[..., None] * dsqrtg

        def dres_Z(dZs, dZt):
            # d(sqrt g)/dX for a Z-coeff: only Z_s, Z_t depend on it
            dsqrtg = R[..., None] * (Rt[..., None] * dZs - Rs[..., None] * dZt)
            return active[..., None] * dsqrtg

        dr_dRac = dres_R(dR_dRac, dRs_dRac, dRt_dRac)
        dr_dRbc = dres_R(dR_dRbc, dRs_dRbc, dRt_dRbc)
        dr_dZas = dres_Z(dZs_dZas, dZt_dZas)
        dr_dZbs = dres_Z(dZs_dZbs, dZt_dZbs)

        rtemp = np.zeros((npts, nvol - 1, mn))
        ztemp = np.zeros((npts, nvol - 1, mn))
        if lvol >= 1:
            rtemp[:, lvol - 1, :] = dr_dRac.reshape(npts, mn) * psifac[lvol][None, :]
            ztemp[:, lvol - 1, :] = dr_dZas.reshape(npts, mn) * psifac[lvol][None, :]
        if lvol + 1 <= nvol - 1:
            rtemp[:, lvol, :] += dr_dRbc.reshape(npts, mn) * psifac[lvol + 1][None, :]
            ztemp[:, lvol, :] += dr_dZbs.reshape(npts, mn) * psifac[lvol + 1][None, :]

        jac[iv] = infaces_flatten(rtemp, ztemp, test).reshape(ns, nt, nz, ndof)

    return jac
