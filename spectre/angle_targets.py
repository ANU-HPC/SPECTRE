import numpy as np

from spectre.spec_fft import spec_fft_fast
from spectre.misc import infaces_unflatten, infaces_flatten
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
