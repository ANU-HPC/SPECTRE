# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
from os import system

from spectre.utils import unpack_interfaces, pack_interfaces
from spectre.fourier import spec_fft_fast
from spectre.geometry import get_coord_transform
from spectre.force_targets import force_real
from spectre.file_io import write_input_parameters_to_toml


def get_inface_func(xin, test, zeta, inface):
    geometry = test.input_list_mod.igeometry
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    mvol = test.allglobal_mod.mvol

    if inface > (mvol - 1) or inface < 0:
        raise RuntimeError(f"Invalid {inface=}")

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    def func(t):
        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, inface],
            irbc[:, inface + 1],
            izbs[:, inface],
            izbs[:, inface + 1],
            im,
            in_,
            1.0,
            t,
            zeta,
            inface,
        )

        return rarr[0, 0, :, 0], zarr[0, 0, :, 0]

    return func


def gen_mesh_calc_rz(xin, test, inface, bcm_m, rho_arr, th_arr, zeta_arr):
    # generate mesh: R, Z, dR/ds, dZ/ds
    # 330 ms

    R_arr = np.empty((rho_arr.size, zeta_arr.size, th_arr.size))
    Z_arr = np.empty_like(R_arr)
    dRdrho_arr = np.empty_like(R_arr)
    dZdrho_arr = np.empty_like(R_arr)

    cos_th_arr = np.cos(th_arr)[None, :]
    sin_th_arr = np.sin(th_arr)[None, :]

    from map2disc import BCM

    for z in range(zeta_arr.size):
        rz_spec_func = get_inface_func(xin, test, zeta=zeta_arr[z], inface=inface)

        # maymap2discbe speedup with numba
        bcm = BCM(rz_spec_func, M=bcm_m)  # 100ms
        bcm.solve()  # 200ms

        x, y = bcm.eval_rt_1d(rho_arr, th_arr)

        rgrad = bcm.eval_grad_rt_1d(rho_arr, th_arr)
        dxy_dr = rgrad[:, 0] * cos_th_arr + rgrad[:, 1] * sin_th_arr

        R_arr[:, z, :] = x
        Z_arr[:, z, :] = y

        dRdrho_arr[:, z, :] = dxy_dr[0]
        dZdrho_arr[:, z, :] = dxy_dr[1]

    return R_arr, Z_arr, dRdrho_arr, dZdrho_arr  # , dxy_dt


def gen_mesh_fft_rz(R_arr, Z_arr, dRdrho_arr, dZdrho_arr, test, th_arr, zeta_arr):
    # now time for some heavey FTs
    # 6.5 ms

    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor

    R_even, R_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, R_arr, mpol, ntor)
    Z_even, Z_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, Z_arr, mpol, ntor)

    dRdrho_even, dRdrho_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, dRdrho_arr, mpol, ntor)
    dZdrho_even, dZdrho_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, dZdrho_arr, mpol, ntor)

    return R_even, dRdrho_even, R_odd, dRdrho_odd, Z_even, dZdrho_even, Z_odd, dZdrho_odd


def gen_mesh_calc_deriv2(dRdrho_even, dRdrho_odd, dZdrho_even, dZdrho_odd, rho_arr):
    # finite difference for finding d^2R/drho^2...

    dRdrho2_even = np.gradient(dRdrho_even, rho_arr, axis=0)
    dRdrho2_odd = np.gradient(dRdrho_odd, rho_arr, axis=0)
    dZdrho2_even = np.gradient(dZdrho_even, rho_arr, axis=0)
    dZdrho2_odd = np.gradient(dZdrho_odd, rho_arr, axis=0)

    return dRdrho2_even, dRdrho2_odd, dZdrho2_even, dZdrho2_odd


def gen_mesh_set_remn(
    R_even,
    dRdrho_even,
    dRdrho2_even,
    R_odd,
    dRdrho_odd,
    dRdrho2_odd,
    Z_even,
    dZdrho_even,
    dZdrho2_even,
    Z_odd,
    dZdrho_odd,
    dZdrho2_odd,
    test,
):
    # set in spectre
    # 23 us

    test.wrapper_funcs_mod.alloc_remn_ext()

    test.allglobal_mod.remn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([R_even, dRdrho_even, dRdrho2_even]), 0, 2), 0, 1
    )
    test.allglobal_mod.romn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([R_odd, dRdrho_odd, dRdrho2_odd]), 0, 2), 0, 1
    )
    test.allglobal_mod.zemn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([Z_even, dZdrho_even, dZdrho2_even]), 0, 2), 0, 1
    )
    test.allglobal_mod.zomn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([Z_odd, dZdrho_odd, dZdrho2_odd]), 0, 2), 0, 1
    )


def gen_mesh_external(xin, test, bcm_m=6, inface=0, ntheta=101, nzeta=32):
    ntheta = 401
    nzeta = 120
    bcm_m = 6

    nfp = test.input_list_mod.nfp

    zeta_arr = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
    th_arr = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    th_arr = -th_arr  # must be negative due to some convention in spectre

    rho_spec_abscis = (test.allglobal_mod.gaussianabscissae[:, 0] + 1.0) * 0.5

    R_arr, Z_arr, dRdrho_arr, dZdrho_arr = gen_mesh_calc_rz(
        xin, test, inface, bcm_m, rho_spec_abscis, th_arr, zeta_arr
    )

    # rho in SPECTRE: -1 -> 1, map2disc: 0 -> 1
    dRdrho_arr /= 2.0
    dZdrho_arr /= 2.0

    R_even, dRdrho_even, R_odd, dRdrho_odd, Z_even, dZdrho_even, Z_odd, dZdrho_odd = (
        gen_mesh_fft_rz(R_arr, Z_arr, dRdrho_arr, dZdrho_arr, test, th_arr, zeta_arr)
    )

    dRdrho2_even, dRdrho2_odd, dZdrho2_even, dZdrho2_odd = gen_mesh_calc_deriv2(
        dRdrho_even, dRdrho_odd, dZdrho_even, dZdrho_odd, rho_spec_abscis
    )

    # rho in SPECTRE: -1 -> 1, map2disc: 0 -> 1
    dRdrho2_even /= 2.0
    dRdrho2_odd /= 2.0
    dZdrho2_even /= 2.0
    dZdrho2_odd /= 2.0

    gen_mesh_set_remn(
        R_even,
        dRdrho_even,
        dRdrho2_even,
        R_odd,
        dRdrho_odd,
        dRdrho2_odd,
        Z_even,
        dZdrho_even,
        dZdrho2_even,
        Z_odd,
        dZdrho_odd,
        dZdrho2_odd,
        test,
    )


def gen_infaces(test, xin, tflux=None, set_to_file=False, base_inface=None, fname_out=None):
    mvol = test.allglobal_mod.mvol
    nfp = test.input_list_mod.nfp

    if tflux is None:
        tflux = test.input_list_mod.tflux[: mvol - 1] / test.input_list_mod.tflux[mvol - 1]
    base_inface = mvol - 1 if base_inface is None else base_inface

    num_infaces = len(tflux)

    if mvol != (num_infaces + 1):
        raise RuntimeError("Input equilibria has to have mvol=(num_infaces+1)")

    ntheta = 124
    nzeta = 78
    bcm_m = 6

    # convert tflux to radius assuming tflux~=r^2
    rho_arr = np.sqrt(tflux)

    zeta_arr = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
    th_arr = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    th_arr = -th_arr  # must be negative due to some convention in spectre

    R_arr, Z_arr, _, _ = gen_mesh_calc_rz(xin, test, base_inface, bcm_m, rho_arr, th_arr, zeta_arr)

    R_even, _, R_odd, _, Z_even, _, Z_odd, _ = gen_mesh_fft_rz(
        R_arr, Z_arr, np.empty((nzeta, ntheta)), np.empty((nzeta, ntheta)), test, th_arr, zeta_arr
    )

    if set_to_file:
        xnew = pack_interfaces(R_even.T, R_odd.T, Z_even.T, Z_odd.T, test)

        force_real(xnew, test)

        test.input_list_mod.linitialize = 0

        test.allglobal_mod.skip_write = True
        nppts = test.input_list_mod.nppts
        test.input_list_mod.nppts = 0
        test.lib.simulation_mod.final_diagnostics()
        test.input_list_mod.nppts = nppts

        # test.allglobal_mod.wrtend()
        if test.comm.rank == 0:
            # generate parameters from fortran memory and write to toml:
            test.wrapper_funcs_mod.prepare_endfile_writing()
            write_input_parameters_to_toml(
                test.parameters, f"{test.name}_end.toml"
            )  # write the final parameters to a toml file

        fname_out = f"{test.name}_init.toml" if fname_out is None else fname_out
        system(f"cp {test.name}_end.toml {fname_out}")

    return R_even, R_odd, Z_even, Z_odd
