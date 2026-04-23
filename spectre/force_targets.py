# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np

from spectre.isect_tester import check_intersection_wrapper
from spectre.fourier import spec_fft_fast
from spectre.utils import reshape_fjac, filter_stellsym_pts


def freal_obj(min_params):

    def obj_func(x, t):
        return force_real(
            x,
            t,
            nt=min_params.nt_dbb,
            nz=min_params.nz_dbb,
            filter_stellsym=min_params.filter_stellsym,
        ).flatten()

    def jac_func(x, t):
        return force_real_jac(
            x,
            t,
            nt=min_params.nt_dbb,
            nz=min_params.nz_dbb,
            filter_stellsym=min_params.filter_stellsym,
            mpol_rz=min_params.mpol_rz,
            ntor_rz=min_params.ntor_rz,
        ).reshape(-1, t.allglobal_mod.ngdof + 1)

    return obj_func, jac_func


def freal_len_obj(min_params, nfp):
    from spectre import get_theta_chord_length, get_theta_chord_length_jac

    thetas = np.linspace(0, 2 * np.pi, 27)
    zetas = np.linspace(0, 2 * np.pi / nfp, 5)

    def obj_func(x, t):
        fr = force_real(
            x,
            t,
            nt=min_params.nt_dbb,
            nz=min_params.nz_dbb,
            filter_stellsym=min_params.filter_stellsym,
        ).flatten()

        length = min_params.len_factor * get_theta_chord_length(x, t, thetas=thetas, zetas=zetas)
        if t.comm.rank == 0:
            print(f"func:  val  {np.sum(fr**2) / 2:.3e}  val_len {np.sum(length**2) / 2:.3e}")
        return np.concatenate([fr, length.reshape(-1)])

    def jac_func(x, t):
        frjac = force_real_jac(
            x,
            t,
            nt=min_params.nt_dbb,
            nz=min_params.nz_dbb,
            filter_stellsym=min_params.filter_stellsym,
            mpol_rz=min_params.mpol_rz,
            ntor_rz=min_params.ntor_rz,
        ).reshape(-1, t.allglobal_mod.ngdof + 1)
        length_jac = min_params.len_factor * get_theta_chord_length_jac(
            x, t, thetas=thetas, zetas=zetas
        )
        return np.vstack([frjac, length_jac.reshape(-1, t.allglobal_mod.ngdof + 1)])

    return obj_func, jac_func


def _process_input_nt_nz(nt, nz, test):

    if nt is None:
        nt = test.input_list_mod.mpol * test.input_list_mod.ndiscrete * 4
    elif isinstance(nt, str) and nt[0] == "m":
        nt = test.input_list_mod.mpol * int(nt[1:])
    elif isinstance(nt, int):
        pass
    else:
        raise ValueError("nt has to be an integer or string e.g. 'm4' (for nt=4*mpol)")

    if nz is None:
        nz = test.input_list_mod.ntor * test.input_list_mod.ndiscrete * 4
        nz = 1 if nz == 0 else nz
    elif isinstance(nz, str) and nz[0] == "n":
        nz = test.input_list_mod.ntor * int(nz[1:])
    elif isinstance(nz, int):
        pass
    else:
        raise ValueError("nz has to be an integer or string e.g. 'n3' (for nz=3*ntor)")

    return nt, nz


def force_real(
    xin, test, pert_axis=True, add_pressure=True, nt=None, nz=None, filter_stellsym=False
):
    """Get force in real space on a given theta-zeta grid. Uses the fortran wrappers to compute the field and force."""

    nt, nz = _process_input_nt_nz(nt, nz, test)
    ntz = nt * nz

    # print(f'force_real: {nt=} {nz=}', flush=True)

    rank = test.comm.Get_rank()

    mvol = test.allglobal_mod.mvol

    if check_intersection_wrapper(xin, test, nt=nt, nz=nz, verbose=True):
        if rank == 0:
            print("Interfaces overlap!")
        return np.ones((mvol - 1, ntz)) * 1e10

    test.field_mod.solve_field(xin, pert_axis)

    f_total = np.zeros([mvol - 1, ntz], dtype=np.float64, order="F")
    test.field_mod.calc_b2mag_jump(f_total, nt, nz)

    # # Old approach: Call dforce to calculate field, and then extract field using lforce
    # ngdof = test.allglobal_mod.ngdof
    # f = np.zeros([ngdof + 1], dtype=np.float64, order="F")
    # test.forces_mod.dforce(ngdof, xin, f, False, True)  # force w/ axis calculation
    # test.fmn_spec = np.sum(f**2)
    # test.fmn_spec_orig = np.sqrt(np.sum(f**2) / test.allglobal_mod.ngdof)
    # dbb1 = np.zeros([ntz, 4], dtype=np.float64, order="F")
    # dbb2 = np.zeros([ntz, 4], dtype=np.float64, order="F")
    # xx = np.zeros([ntz], dtype=np.float64, order="F")
    # yy = np.zeros([ntz], dtype=np.float64, order="F")
    # length = np.zeros([ntz], dtype=np.float64, order="F")
    # f_total = np.zeros((mvol - 1, ntz))
    #
    # for inface in range(mvol - 1):
    #     # inside face
    #     lvol = inface + 1  # inside
    #     iocons = 1  # 0 in, 1 out
    #     lregion(test, lvol)
    #     test.forces_mod.lforce(lvol, iocons, 0, ntz, dbb1, xx, yy, length, 0.0, 0.0, 0)
    #     # outside face
    #     lvol = lvol + 1
    #     iocons = 0  # 0 in, 1 out
    #     lregion(test, lvol)
    #     test.forces_mod.lforce(lvol, iocons, 0, ntz, dbb2, xx, yy, length, 0.0, 0.0, 0)
    #     f_total[inface] = dbb2[:, 1] - dbb1[:, 1]

    if add_pressure:
        p_scaled = test.input_list_mod.pressure[:mvol] * test.input_list_mod.pscale
        f_total += np.diff(p_scaled)[:, None]

    test.f_real_shpd = f_total.reshape(-1, nz, nt, order="C")

    if filter_stellsym and test.input_list_mod.enforce_stell_sym:
        f_total = filter_stellsym_pts(f_total, nt, nz)

    return f_total


def force_real_jac(
    xin, test, pert_axis=True, mpol_rz=None, ntor_rz=None, nt=None, nz=None, filter_stellsym=False
):

    nt, nz = _process_input_nt_nz(nt, nz, test)
    ntz = nt * nz

    if mpol_rz is None:
        mpol_rz = test.input_list_mod.mpol
    if ntor_rz is None:
        ntor_rz = test.input_list_mod.ntor * test.input_list_mod.nfp
    if not (isinstance(mpol_rz, int) and isinstance(mpol_rz, int)):
        raise ValueError("mpol_rz and ntor_rz need to be integers")

    ngdof = test.allglobal_mod.ngdof
    mvol = test.allglobal_mod.mvol

    test.field_mod.solve_field_deriv(xin, mpol_rz, ntor_rz, pert_axis)

    freal_jac = np.zeros(((mvol - 1) * ntz, ngdof + 1), dtype=np.float64, order="F")
    test.field_mod.calc_b2mag_jump_deriv(freal_jac, nt, nz)

    freal_jac_shpd = freal_jac.reshape(mvol - 1, ntz, ngdof + 1)

    if filter_stellsym and test.input_list_mod.enforce_stell_sym:
        freal_jac_shpd = filter_stellsym_pts(freal_jac_shpd, nt, nz)

    return freal_jac_shpd


def force_modes_dforce(x, test):
    # Returns the standard truncated fmn spectrum, directly from dforce() in SPECTRE

    ngdof = test.allglobal_mod.ngdof

    f = np.zeros([ngdof + 1], dtype=np.float64, order="F")
    test.forces_mod.dforce(ngdof, x, f, False, True)

    fmn_sum = np.sqrt(np.sum(f**2) / ngdof)

    test.fmn_spec = np.sum(f**2)
    test.fmn_spec_orig = fmn_sum

    return f


def force_modes_dforce_with_grad(x, test):
    # Returns the standard truncated fmn spectrum, directly from dforce() in SPECTRE
    # with the gradient

    ngdof = test.allglobal_mod.ngdof
    ntz = test.allglobal_mod.ntz

    f = np.empty([ngdof + 1], dtype=np.float64, order="F")

    test.wrapper_funcs_mod.alloc_for_hessian(ntz)

    test.forces_mod.dforce(ngdof, x, f, True, False)

    f_grad = test.allglobal_mod.hessian.copy()

    test.wrapper_funcs_mod.dealloc_hessian()

    return f, f_grad


def lregion(test, lvol):
    # Some SPECTRE Hokus-Pokus

    if test.input_list_mod.igeometry == 1 or lvol > 1:
        test.allglobal_mod.lcoordinatesingularity = False
    else:
        test.allglobal_mod.lcoordinatesingularity = True

    test.allglobal_mod.lplasmaregion = True
    test.allglobal_mod.lvacuumregion = False


def force_modes(xin, test, add_pressure=True, which_modes="all", exp_fact=0, return_indices=False):
    # Returns force spectrum, Calculated manually

    mvol = test.allglobal_mod.mvol
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    mn = test.allglobal_mod.mn
    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor

    test.comm.Barrier()

    rank = test.comm.Get_rank()

    if nz == 1:
        nz_mod = 3
    else:
        nz_mod = nz

    # Pick number of modes
    if which_modes == "all":
        num_modes = (
            int(np.floor(nz_mod / 2) - 1)
            + 1
            + int(np.floor(nt / 2)) * (2 * int(np.floor(nz_mod / 2) - 1) + 1)
        )
        mpol_max = None
        ntor_max = None
    elif which_modes == "spectre":
        num_modes = mn  # use as many modes as in spectre fmn...
        mpol_max = mpol
        ntor_max = ntor
    else:
        raise RuntimeError('which_modes has to be "all" or "spectre"')

    freal = force_real(xin, test, add_pressure=add_pressure)

    if np.all(freal > 9.9e3):
        if rank == 0:
            print("Interfaces overlap!")
        if return_indices:
            return np.ones((mvol - 1, num_modes)) * 1e4, np.array([]), np.array([])
        else:
            return np.ones((mvol - 1, num_modes)) * 1e4

    if nz == 1:
        freal_shaped = np.broadcast_to(freal[:, None, :], (freal.shape[0], nz_mod, freal.shape[1]))
    else:
        freal_shaped = freal.reshape(freal.shape[0], nz_mod, nt, order="C")

    fmodes_total = np.zeros((mvol - 1, num_modes))

    thetas = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zetas = np.linspace(0, 2 * np.pi, nz_mod, endpoint=False)

    for lvol in range(mvol - 1):
        fmodes, fmodes_odd, _im, _in = spec_fft_fast(
            thetas, zetas, freal_shaped[lvol], mpol_max, ntor_max
        )

        if nz == 1:
            nz0_mask = _in == 0
            fmodes = fmodes[nz0_mask]

            _im = _im[nz0_mask]
            _in = _in[nz0_mask]
            # fmodes_odd = fmodes_odd[nz0_mask]

        if which_modes == "spectre":
            mask = (_im < mpol + 1) & (-ntor - 1 < _in) & (_in < ntor + 1)
        else:
            mask = _im > -1

        fmodes_total[lvol] = fmodes[mask]

    test.fmn_full = np.sum(fmodes_total**2)  # diagnostics

    # Add an exponential weighting factor, when it works it works beautifully (but the exponent has to be "tuned")
    weight_factors = 1.0 * np.exp(exp_fact * (_im[mask] ** 2 + _in[mask] ** 2))
    fmodes_total = fmodes_total * weight_factors[None, :]

    test.comm.Barrier()

    if return_indices:
        return fmodes_total, _im[mask], _in[mask]

    return fmodes_total


def force_modes_targetmn(xin, test, target_mns, exp_fact=0):
    ## Return only certain modes of the full force_modes

    if not (
        (isinstance(target_mns, list) and isinstance(target_mns[0], tuple))
        or (isinstance(target_mns, np.ndarray) and len(target_mns.shape) == 2)
    ):
        raise ValueError('"target_mns" has to be a list of tuples...')

    num_targets = len(target_mns)

    fmn, _im, _in = force_modes(
        xin, test, which_modes="all", exp_fact=exp_fact, return_indices=True
    )

    fmn_target = np.zeros((len(fmn), num_targets))

    for i in range(num_targets):
        ind = (_im == target_mns[i][0]) & (_in == target_mns[i][1])
        fmn_target[:, i] = fmn[:, ind][:, 0]

    return fmn_target


def force_modes_jac(xin, test, freal_jac=None, which_modes="all"):
    # Returns jacobian of the force spectrum
    # Calculated manually

    test.comm.Barrier()

    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    ngdof = test.allglobal_mod.ngdof
    thetas = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zetas = np.linspace(0, 2 * np.pi, nz, endpoint=False)

    if freal_jac is None:
        freal_jac = force_real_jac(xin, test).reshape(-1, ngdof + 1)

    if which_modes == "all":
        mpol_max = None
        ntor_max = None
    elif which_modes == "spectre":
        mpol_max = test.input_list_mod.mpol
        ntor_max = test.input_list_mod.ntor
    else:
        raise RuntimeError('which_modes has to be "all" or "spectre"')

    fjac_shpd = reshape_fjac(freal_jac, nt, nz)

    fjac_ft, _, im_full, in_full = spec_fft_fast(thetas, zetas, fjac_shpd, mpol_max, ntor_max)

    return fjac_ft


def force_modes_speclike(xin, test):

    fmn_custom = force_modes(xin, test, which_modes="spectre")

    return fmn_custom.flatten()


def force_modes_jac_speclike(xin, test, freal_jac=None):

    ngdof = test.allglobal_mod.ngdof

    fjac_ft = force_modes_jac(xin, test, freal_jac=freal_jac, which_modes="spectre")

    fmn_jac_custom = fjac_ft[1:].reshape(ngdof, -1)

    return fmn_jac_custom
