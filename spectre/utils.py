# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
from time import perf_counter, strftime, gmtime
from inspect import getsource
from scipy import integrate

from spectre.config import run_decorator


def find_res_infaces(q, qres):
    """
    Ex use: 'surfaces=find_res_infaces(1/o.input.physics.iota[1:], 2.0)'
    """
    arr = q - qres
    inds = np.full(len(q), False)
    inds[arr < 0.0] = True

    ind_min = np.argmin(arr[inds] ** 2.0)
    ind_max = np.argmin(arr[~inds] ** 2.0) + np.sum(inds)

    return np.array([ind_min, ind_max])


def get_xinit_specwrap(test):
    ngdof = test.allglobal_mod.ngdof
    x = np.zeros([ngdof + 1], dtype=np.float64, order="F")
    mvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    irbc = test.allglobal_mod.irbc
    izbs = test.allglobal_mod.izbs
    irbs = test.allglobal_mod.irbs
    izbc = test.allglobal_mod.izbc

    test.packing_mod.packxi(ngdof, x, mvol, mn, nt, nz, irbc, izbs, irbs, izbc, "P", False, True)

    return x


class Timer:
    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start

        self.readout = strftime("Time elapsed  %H:%M:%S", gmtime(self.time))
        if self.time < 1.0:
            self.readout += f".{int(self.time * 1000):03d}"


def create_runfile(name, ext, run):
    sourcefile = getsource(run)

    s = "def run():\n"
    sourcefile = (
        sourcefile.replace(s, s + f"\n    ext = '{ext}'\n    name = '{name}'\n") + "run()\n"
    )

    with open(f".tmpcode_{name}", "w") as f:
        f.writelines(sourcefile)


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


def get_fsa_wrapper(f, test):
    nt = test.field_mod.nt_alloc
    nz = test.field_mod.nz_alloc
    mn = test.allglobal_mod.mn
    nvol = test.allglobal_mod.mvol

    tarr = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    zarr = np.linspace(0, 2 * np.pi, nz, endpoint=False)

    fsa_f = np.empty(nvol - 1)

    for i in np.arange(nvol - 1):
        _, _, sg, _, _ = test.compute_coordinates(i + 1, 1.0, 1, nt, nz, mn)
        jac = sg[:, 0].reshape((nt, nz), order="F")

        # if test.comm.Get_rank() == 0 and i==2:
        #     print(f"{(jac**2).sum():.3e}")
        # jac2 = np.empty(ntz+1, dtype=np.float64, order="F")
        # test.wrapper_funcs_mod.get_jac(jac2, i + 1, 1.0, 1, ntz, mn)
        # jac2 = jac2[1:].reshape((nt, nz), order="F")
        # if(test.comm.Get_rank() == 0):
        #     print(f"{jac.shape=}  {np.max(jac)=:.3e}")
        #     print(f"{jac2.shape=}  {np.max(jac2)=:.3e}")

        numerator = periodic_integrator_thzt(jac * (f[i]).T, tarr, zarr)
        denominator = periodic_integrator_thzt(jac, tarr, zarr)

        fsa_f[i] = numerator / denominator

    return fsa_f


def get_freal_fsa(test):
    try:
        freal_sqrd = test.f_real_shpd**2
        freal_sqrd_fsa = get_fsa_wrapper(freal_sqrd, test)

    except AttributeError:
        freal_sqrd_fsa = 1.0e10
        test.f_real_shpd = np.array([[[1.0e10]]])
    test.freal_sqrd_fsa = freal_sqrd_fsa

    return freal_sqrd_fsa


def get_freal_fsa_metric(test):
    freal_sqrd_fsa = get_freal_fsa(test)

    return np.sqrt(np.mean(freal_sqrd_fsa))


def check_fsa_vols(test):
    if not hasattr(test, "freal_sqrd_fsa"):
        freal_sqrd = test.f_real_shpd**2
        test.freal_sqrd_fsa = get_fsa_wrapper(freal_sqrd, test)

    f_fsa = np.sqrt(test.freal_sqrd_fsa)

    print(f"<f> on infaces: {np.array2string(f_fsa, formatter={'float': lambda x: f'{x:.3e} '})}")

    return f_fsa


def get_xmn(x, test, m, n, lvol=None):
    ## Extract a specific interface harmonic

    nvol = test.allglobal_mod.mvol
    in_ = test.allglobal_mod.in_
    im = test.allglobal_mod.im

    if lvol is None:
        lvol = np.arange(nvol - 1)
    else:
        lvol = np.atleast_1d(lvol)

    # ind = (mpol+1) * 2 + 1
    ind = (m == im) & (n == in_)

    xsh = x[1:].reshape(nvol - 1, -1)

    return xsh[lvol, ind]


@run_decorator
def reshape_fjac(fjac, nt, nz):
    dim, num_modes = fjac.shape
    ntz = nt * nz
    ninfaces = dim // ntz

    fjac_shpd = np.empty((num_modes, ninfaces, nz, nt))
    for i in range(num_modes):
        for inface in range(ninfaces):
            for t in range(nt):
                for z in range(nz):
                    fjac_shpd[i, inface, z, t] = fjac[inface * ntz + nt * z + t, i]

    return fjac_shpd


@run_decorator
def reshape_modes_fjac(fjac, im_, in_, mpol, ntor, nvol, nt, nz):
    mn = len(im_)

    fjac_shpd = reshape_fjac(fjac, nt, nz)
    fjac_shpd_again = np.empty((mpol + 1, 2 * ntor + 1, nvol - 1, nvol - 1, nz, nt))

    for ind in range(fjac_shpd.shape[0]):
        m_mode = im_[ind % mn]
        n_mode = in_[ind % mn] + ntor
        l_mode = ind // mn

        temp = fjac_shpd[ind]
        for ivol in range(nvol - 1):
            for t in range(nt):
                for z in range(nz):
                    fjac_shpd_again[m_mode, n_mode, l_mode, ivol, z, t] = temp[ivol, z, t]

    return fjac_shpd_again


def get_fmn_force_mask(test):
    # returns mask for indexing only components of SPECTRE force from B^2
    # inverting the mask gives spectral compoments

    geometry = test.input_list_mod.igeometry
    ngdof = test.allglobal_mod.ngdof
    mvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn

    mask = np.full(ngdof, True)

    if geometry < 3:
        return mask

    mnboth = 2 * mn - 1
    for sl in range(mvol - 1):
        ind = sl * mnboth + mn
        mask[ind : ind + (mn - 1)] = False

    return mask


def remove_file(fname):
    from os import remove, path

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_rank() == 0:
        if path.exists(fname):
            remove(fname)


def infaces_unflatten(x, test, add_psifactor=False):
    nvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn

    if x.ndim == 1:
        x_shpd = x.copy()[1:].reshape(nvol - 1, -1, order="C")
        r = np.hstack([x_shpd[:, 0][:, None], x_shpd[:, 1::2]])
        z = np.hstack([np.zeros(nvol - 1)[:, None], x_shpd[:, 2::2]])

        if add_psifactor:
            psifactor = test.allglobal_mod.psifactor.T[0:-1]
            r *= psifactor
            z *= psifactor

        return r, z

    else:
        batch_size = x.shape[0]

        x_batch = x.copy()[:, 1:]
        x_shpd = x_batch.reshape(batch_size, nvol - 1, 2 * mn - 1, order="C")

        r = np.concatenate([x_shpd[:, :, 0:1], x_shpd[:, :, 1::2]], axis=2)

        z = np.concatenate([np.zeros((batch_size, nvol - 1, 1)), x_shpd[:, :, 2::2]], axis=2)

        if add_psifactor:
            psifactor = test.allglobal_mod.psifactor.T[0:-1]
            r *= psifactor[None, :, :]
            z *= psifactor[None, :, :]

        return r, z


def infaces_flatten(r, z, test, add_psifactor=False):
    if test.input_list_mod.igeometry != 3:
        raise ValueError("Works only for G=3")

    if r.ndim == 2:
        if add_psifactor:
            psifactor = test.allglobal_mod.psifactor.T[0:-1]
            r /= psifactor
            z /= psifactor

        x_comb = np.zeros((r.shape[0], r.shape[1] + (z.shape[1] - 1)))
        x_comb[:, 0] = r[:, 0].copy()
        x_comb[:, 1::2] = r[:, 1:].copy()
        x_comb[:, 2::2] = z[:, 1:].copy()
        x = np.concatenate(([0.0], x_comb.flatten()))
        return x

    else:
        if add_psifactor:
            psifactor = test.allglobal_mod.psifactor.T[0:-1]
            r /= psifactor[None, :, :]
            z /= psifactor[None, :, :]

        batch_size = r.shape[0]
        x_comb = np.zeros((batch_size, r.shape[1], r.shape[2] + (z.shape[2] - 1)))
        x_comb[:, :, 0] = r[:, :, 0].copy()
        x_comb[:, :, 1::2] = r[:, :, 1:].copy()
        x_comb[:, :, 2::2] = z[:, :, 1:].copy()

        x = np.zeros((batch_size, x_comb.shape[1] * x_comb.shape[2] + 1))
        x[:, 1:] = x_comb.reshape(batch_size, -1)

        return x


def _set_interfaces_from_xin(xin, test):
    """Push packed interface degrees of freedom into the Fortran arrays."""
    ngdof = test.allglobal_mod.ngdof
    mn = test.allglobal_mod.mn
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz
    nvol = test.allglobal_mod.mvol

    irbc = test.allglobal_mod.irbc
    irbs = test.allglobal_mod.irbs
    izbs = test.allglobal_mod.izbs
    izbc = test.allglobal_mod.izbc
    test.packing_mod.packxi(ngdof, xin, nvol, mn, nt, nz, irbc, izbs, irbs, izbc, "U", False, True)

    return irbc, izbs, irbs, izbc


def unpack_interfaces(xin, test):
    """Unpack interface degrees of freedom into the Fortran arrays."""
    return _set_interfaces_from_xin(xin, test)


def pack_interfaces(r_even, r_odd, z_even, z_odd, test):
    rbc = test.allglobal_mod.irbc.copy()
    rbs = test.allglobal_mod.irbs.copy()
    zbc = test.allglobal_mod.izbc.copy()
    zbs = test.allglobal_mod.izbs.copy()

    rbc[:, 1:-1] = r_even
    rbs[:, 1:-1] = r_odd
    zbc[:, 1:-1] = z_even
    zbs[:, 1:-1] = z_odd

    ngdof = test.allglobal_mod.ngdof
    xnew = np.zeros([ngdof + 1], dtype=np.float64, order="F")
    mvol = test.allglobal_mod.mvol
    mn = test.allglobal_mod.mn
    nt = test.allglobal_mod.nt
    nz = test.allglobal_mod.nz

    test.packing_mod.packxi(ngdof, xnew, mvol, mn, nt, nz, rbc, zbs, rbs, zbc, "P", False, False)

    return xnew


def get_vec_pot_flat(test):
    """Get vector potential from the current solution. Found using `test.wrapper_funcs_mod.get_vec_pot`

    Args:
        test (SPECTRE): Wrapper object

    Returns:
        ndarray[float]: At cosine coefficients, shape (sum(lrad+1), mn)
        ndarray[float]: Az cosine coefficients, shape (sum(lrad+1), mn)
        ndarray[float]: At sine coefficients, shape (sum(lrad+1), mn)
        ndarray[float]: Az sine coefficients, shape (sum(lrad+1), mn)
    """

    lrad = test.input_list_mod.lrad
    mvol = test.allglobal_mod.mvol
    size = np.sum(lrad[:mvol] + 1)

    ate = np.zeros((size, test.allglobal_mod.mn), order="F", dtype=np.float64)
    aze = np.zeros_like(ate)
    ato = np.zeros_like(ate)
    azo = np.zeros_like(ate)

    test.wrapper_funcs_mod.get_vec_pot(size, ate, aze, ato, azo)

    return ate, aze, ato, azo


def get_vec_pot(test):
    """Get vector potential from the current solution. Found using `test.wrapper_funcs_mod.get_vec_pot`

    Args:
        test (SPECTRE): Wrapper object

    Returns:
        List[ndarray[float]]: At cosine coefficients, list of arrays with shape (lrad, mn)
        List[ndarray[float]]: Az cosine coefficients, list of arrays with shape (lrad, mn)
        List[ndarray[float]]: At sine coefficients, list of arrays with shape (lrad, mn)
        List[ndarray[float]]: Az sine coefficients, list of arrays with shape (lrad, mn)
    """
    ate, aze, ato, azo = get_vec_pot_flat(test)

    ate_list = []
    aze_list = []
    ato_list = []
    azo_list = []

    ind = 0
    for iv in range(test.allglobal_mod.mvol):
        ind_next = ind + test.input_list_mod.lrad[iv] + 1
        ate_list.append(np.atleast_2d(ate[ind:ind_next]).T)
        aze_list.append(np.atleast_2d(aze[ind:ind_next]).T)
        ato_list.append(np.atleast_2d(ato[ind:ind_next]).T)
        azo_list.append(np.atleast_2d(azo[ind:ind_next]).T)
        ind = ind_next

    return ate_list, aze_list, ato_list, azo_list


def update_bnorm(test, first_free_bound, print_info=False):
    """
    Update B.n on computational boundary from plasma using virtual-casing principle

    first_free_bound: Sets which surface current is used for the virtual-casing
        True: the surface current on the inner face of plasma boundary
        False: the surface current on the outer face of the plasma boundary
            (requires a solution of vacuum field which in turn requires B.n from plasma)
    """

    test.allglobal_mod.first_free_bound = first_free_bound

    test.wrapper_funcs_mod.update_bnorm()

    if print_info:
        with np.printoptions(formatter={"all": lambda x: f"{x:.4e}"}):
            print(
                f"B_norm updated:  bnserr {test.allglobal_mod.bnserr:.4e}  ibns[1:6] {test.allglobal_mod.ibns[1:6]}"
            )


def filter_stellsym_pts(arr, nt, nz):
    """Takes an array on the full (nt, nz) grid and returns only the unique stellarator-symmetric components.
    Conserves the sum of squared arr (up to a constant factor), e.g. sum(arr**2) == 2 * sum(filter_stellsym_pts(arr**2,...))."""
    num_pts = nz // 2 + 1

    shape = list(arr.shape)
    shape.insert(1, nz)
    shape[2] = nt

    arr_shpd = arr.reshape(*shape, order="C")[:, :num_pts].copy()
    arr_shpd[:, [0, -1]] *= np.sqrt(
        0.5
    )  # sqrt(0.5) is used to preserve sum of squared array (since we use least squares minimization)

    shape.pop(1)
    shape[1] = nt * num_pts
    return arr_shpd.reshape(*shape)


def save_errors_to_h5(
    filename_h5, fsa_err, beltrami_errors_avg, beltrami_errors_max, f_errs_rms, f_errs_max
):
    from h5py import File

    with File(filename_h5, "a") as f:
        grp_errs = f.create_group("errors")

        grp_errs.create_dataset("fsa", data=[fsa_err])
        grp_errs.create_dataset("beltrami_avg", data=beltrami_errors_avg)
        grp_errs.create_dataset("beltrami_max", data=beltrami_errors_max)
        grp_errs.create_dataset("force_rms", data=f_errs_rms)
        grp_errs.create_dataset("force_max", data=f_errs_max)
