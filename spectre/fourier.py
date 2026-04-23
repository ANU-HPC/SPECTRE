# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## Written by Erol Balkovic 8/9/2024
## erol.balkovic@epfl.ch
from importlib.util import find_spec

import numpy as np
from scipy.fft import fft2

from spectre.config import run_decorator


@run_decorator
def check_equidistant(arr):
    dm = arr[1] - arr[0]
    for a in range(len(arr) - 1):
        dp = arr[a + 1] - arr[a]
        if np.abs(dp - dm) > 1e-14:
            raise ValueError("Points should be equidistant in tarr/zarr")
        dm = dp


@run_decorator
def shift_fftfreq(arr):
    num_rows = arr.shape[0]
    num_rows_div2 = num_rows // 2
    diff = num_rows - num_rows_div2

    tarr = np.empty_like(arr)

    for i in range(diff):
        tarr[i + num_rows_div2] = arr[i]

    for i in range(diff, num_rows):
        tarr[i - diff] = arr[i]

    return tarr


# @njit(cache=True, fastmath=True)
def spec_fft(tarr: np.array, zarr: np.array, freal: np.array, Mpol: int = None, Ntor: int = None):
    """
    Fourier transform as in SPECTRE fortran source

    Args:
     - tarr: 1D numpy array of size nt. Poloidal angle coordinate, each point
             should be equidistant
     - zarr: 1D numpy array of size nz. Toroidal angle coordinate, each point
             should be equidistant
     - freal: 2D numpy array of size (nz,nt). Function, in real space, for which
             Fourier harmonics should be evaluated
     - Mpol: integer. Poloidal resolution. Default is nt/2.
     - Ntor: integer. Toroidal resolution. Default is nz/2.
    """

    # if not isinstance(tarr, np.ndarray):
    #     raise ValueError("tarr should be a numpy array")
    if tarr.ndim > 1:
        raise ValueError("tarr should be one-dimensional")

    nt = tarr.size
    if nt < 3:
        raise ValueError("Not enough points in tarr")

    # zarr
    # if not isinstance(zarr, np.ndarray):
    #     raise ValueError("zarr should be a numpy array")
    if zarr.ndim > 1:
        raise ValueError("zarr should be one-dimensional")

    nz = zarr.size
    if nz < 3:
        raise ValueError("Not enough points in zarr")

    # if not isinstance(freal, np.ndarray):
    #     raise ValueError('freal should be a numpy array')
    if freal.ndim != 2:
        raise ValueError("freal should be two-dimensional")
    if freal.shape != (nz, nt):
        raise ValueError("freal should be of size (nz,nt)")

    if tarr[0] + 2 * np.pi == tarr[-1]:  # not a good check for floats!
        print("tarr[-1] should not be equal to tarr[0]. Removing last element...")
        tarr = tarr[:-1]
        nt -= 1
        freal = freal[:, :-1]

    if zarr[0] + 2 * np.pi == zarr[-1]:  # not a good check for floats!
        print("zarr[-1] should not be equal to zarr[0]. Removing last element...")
        zarr = zarr[:-1]
        nz -= 1
        freal = freal[:-1, :]

    check_equidistant(tarr)
    check_equidistant(zarr)

    M = int(np.floor(nt / 2))
    if Mpol is None:
        Mpol = M if nt % 2 == 0 else M
    # if not isinstance(Mpol, int) and not isinstance(Mpol, np.integer):
    #     raise ValueError('Mpol should be an integer')
    if Mpol < 1 or Mpol > nt / 2:
        raise ValueError("Mpol should be at least 1 and smaller than num_theta/2")

    N = int(np.floor(nz / 2))
    if Ntor is None:
        Ntor = N - 1 if nz % 2 == 0 else N
    # if not isinstance(Ntor, int) and not isinstance(Ntor, np.integer):
    #     raise ValueError('Ntor should be an integer')
    if Ntor < 0 or Ntor > nz / 2:
        raise ValueError("Ntor should be positive and smaller than num_zeta/2")

    # Actual fft
    fmn = np.fft.fft2(freal) / (nt * nz)

    # Now construct 2D array. Negative m-modes have to be added to positive
    # m-modes, i.e. (m,n)+(-m,-n) for even modes, and (m,n)-(-m,-n) for odd
    # modes.
    # Difference if the number of elements (either nt or nz) is odd - this is
    # due to the internal implementation of numpy.fft.fft.
    efmn = fmn.copy()
    efmn[1:, 1:] = fmn[1:, 1:] + fmn[1:, 1:][::-1, ::-1]

    ofmn = fmn.copy()
    ofmn[1:, 1:] = fmn[1:, 1:] - fmn[1:, 1:][::-1, ::-1]

    if np.mod(nt, 2) == 0:
        efmn[0, 1 : M + 1] += fmn[0, M:][::-1]  # n=0
        ofmn[0, 1 : M + 1] -= fmn[0, M:][::-1]
    else:
        efmn[0, 1 : M + 1] += fmn[0, M + 1 :][::-1]  # n=0
        ofmn[0, 1 : M + 1] -= fmn[0, M + 1 :][::-1]

    if np.mod(nz, 2) == 0:
        efmn[1 : N + 1, 0] += fmn[N:, 0][::-1]  # m=0
        ofmn[1 : N + 1, 0] -= fmn[N:, 0][::-1]
    else:
        efmn[1 : N + 1, 0] += fmn[N + 1 :, 0][::-1]  # m=0
        ofmn[1 : N + 1, 0] -= fmn[N + 1 :, 0][::-1]

    efmn = (np.absolute(efmn) * np.cos(np.angle(efmn)))[:, 0 : M + 1]
    ofmn = (np.absolute(ofmn) * np.sin(np.angle(ofmn)))[:, 0 : M + 1]

    # Change sign of m>0 odd modes
    ofmn[:, 1:] = -ofmn[:, 1:]

    efmn = shift_fftfreq(efmn)
    ofmn = shift_fftfreq(ofmn)

    # Invert n -> -n
    if np.mod(nz, 2) == 0:
        efmn[1:, 1:] = efmn[1:, 1:][::-1]
        ofmn[1:, 1:] = ofmn[1:, 1:][::-1]
    else:
        efmn[:, 1:] = efmn[::-1, 1:]
        ofmn[:, 1:] = ofmn[::-1, 1:]

    # 1D - construct array mode by mode
    nmn = Ntor + 1 + Mpol * (2 * Ntor + 1)
    even_out = np.zeros((nmn,))
    odd_out = np.zeros((nmn,))
    _im = np.zeros((nmn,), dtype="i")
    _in = np.zeros((nmn,), dtype="i")

    ind = -1
    for mm in range(0, Mpol + 1):
        for nn in range(-Ntor, Ntor + 1):
            if mm == 0 and nn < 0:
                continue

            ind += 1

            _im[ind] = mm
            _in[ind] = nn
            even_out[ind] = efmn[N + nn, mm]
            odd_out[ind] = ofmn[N + nn, mm]

    return even_out, odd_out, _im, _in


# if rocket_fft is installed, use it for fast fft from numba
has_rocket_fft = find_spec("rocket_fft") is not None
if has_rocket_fft:
    run_decorator_fft = run_decorator
else:

    def run_decorator_fft(func):
        return func


@run_decorator_fft
def spec_fft_fast(
    tarr: np.array, zarr: np.array, freal_in: np.array, Mpol: int = None, Ntor: int = None
):
    """
    Fourier transform as in SPECTRE fortran source

    Args:
     - tarr: 1D numpy array of size nt. Poloidal angle coordinate, each point
                     should be equidistant
     - zarr: 1D numpy array of size nz. Toroidal angle coordinate, each point
                     should be equidistant
     - freal: 2D numpy array of size (nz,nt). Function, in real space, for which
                     Fourier harmonics should be evaluated
     - Mpol: integer. Poloidal resolution. Default is nt/2.
     - Ntor: integer. Toroidal resolution. Default is nz/2.
    """

    if tarr.ndim > 1:
        raise ValueError("tarr should be one-dimensional")
    nt = tarr.size

    if zarr.ndim > 1:
        raise ValueError("zarr should be one-dimensional")
    nz = zarr.size

    transform_1d_nt = False
    transform_1d_nz = False

    if nt == 1:
        freal = np.stack((freal_in, freal_in, freal_in), axis=-1)[..., 0, :]
        nt = 3
        transform_1d_nt = True
        Mpol = None

    if nz == 1:
        freal = np.stack((freal_in, freal_in, freal_in), axis=-2)[..., 0, :, :]
        nz = 3
        transform_1d_nz = True
        Ntor = None

    if not (transform_1d_nt or transform_1d_nz):
        freal = freal_in

    freal_shp = freal.shape

    if freal_shp[-2] != nz:
        raise ValueError("freal should be of size (...,nz,nt)")
    if freal_shp[-1] != nt:
        raise ValueError("freal should be of size (...,nt)")

    if tarr[0] + 2 * np.pi == tarr[-1]:  # not a good check for floats!
        print("tarr[-1] should not be equal to tarr[0]. Removing last element...")
        tarr = tarr[:-1]
        nt -= 1
        freal = freal[..., :-1]

    if zarr[0] + 2 * np.pi == zarr[-1]:  # not a good check for floats!
        print("zarr[-1] should not be equal to zarr[0]. Removing last element...")
        zarr = zarr[:-1]
        nz -= 1
        freal = freal[..., :-1, :]

    check_equidistant(tarr)
    check_equidistant(zarr)

    M = int(np.floor(nt / 2))
    if Mpol is None:
        Mpol = M if nt % 2 == 0 else M
    # if not isinstance(Mpol, int) and not isinstance(Mpol, np.integer):
    #     raise ValueError('Mpol should be an integer')
    if Mpol < 1 or Mpol > nt / 2:
        raise ValueError("Mpol should be at least 1 and smaller than num_theta/2")

    N = int(np.floor(nz / 2))
    if Ntor is None:
        Ntor = N - 1 if nz % 2 == 0 else N
    # if not isinstance(Ntor, int) and not isinstance(Ntor, np.integer):
    #     raise ValueError('Ntor should be an integer')
    if Ntor < 0 or Ntor > nz / 2:
        raise ValueError("Ntor should be positive and smaller than num_zeta/2")

        # 1D - construct array mode by mode
    nmn = Ntor + 1 + Mpol * (2 * Ntor + 1)
    even_out = np.zeros((nmn,))
    odd_out = np.zeros((nmn,))
    _im = np.zeros((nmn,), dtype="i")
    _in = np.zeros((nmn,), dtype="i")

    ind = -1
    for mm in range(0, Mpol + 1):
        for nn in range(-Ntor, Ntor + 1):
            if mm == 0 and nn < 0:
                continue

            ind += 1

            _im[ind] = mm
            _in[ind] = nn

    ntz = nt * nz

    nt_even = np.mod(nt, 2) == 0
    nz_even = np.mod(nz, 2) == 0

    even_out_full = np.empty((freal.shape[:-2] + (nmn,)))
    odd_out_full = np.empty_like(even_out_full)

    for idx in np.ndindex(freal.shape[:-2]):
        fmn = fft2(freal[idx]) / ntz
        # fmn = np.empty(freal.shape[-2:], dtype=np.float64)

        efmn = fmn.copy()
        efmn[1:, 1:] = fmn[1:, 1:] + fmn[1:, 1:][::-1, ::-1]

        ofmn = fmn.copy()
        ofmn[1:, 1:] = fmn[1:, 1:] - fmn[1:, 1:][::-1, ::-1]

        if nt_even:
            efmn[0, 1 : M + 1] += fmn[0, M:][::-1]  # n=0
            ofmn[0, 1 : M + 1] -= fmn[0, M:][::-1]
        else:
            efmn[0, 1 : M + 1] += fmn[0, M + 1 :][::-1]  # n=0
            ofmn[0, 1 : M + 1] -= fmn[0, M + 1 :][::-1]

        if nz_even:
            efmn[1 : N + 1, 0] += fmn[N:, 0][::-1]  # m=0
            ofmn[1 : N + 1, 0] -= fmn[N:, 0][::-1]
        else:
            efmn[1 : N + 1, 0] += fmn[N + 1 :, 0][::-1]  # m=0
            ofmn[1 : N + 1, 0] -= fmn[N + 1 :, 0][::-1]

        efmn = efmn[:, 0 : M + 1]
        ofmn = ofmn[:, 0 : M + 1]

        efmn = np.abs(efmn) * np.cos(np.angle(efmn))
        ofmn = np.abs(ofmn) * np.sin(np.angle(ofmn))

        ## most expensive
        # efmn = (np.absolute(efmn) * np.cos(np.angle(efmn)))[:, 0 : M + 1]
        # ofmn = (np.absolute(ofmn) * np.sin(np.angle(ofmn)))[:, 0 : M + 1]

        # Change sign of m>0 odd modes
        ofmn[:, 1:] = -ofmn[:, 1:]

        efmn = shift_fftfreq(efmn)
        ofmn = shift_fftfreq(ofmn)

        # Invert n -> -n
        if nz_even:
            efmn[1:, 1:] = efmn[1:, 1:][::-1]
            ofmn[1:, 1:] = ofmn[1:, 1:][::-1]
        else:
            efmn[:, 1:] = efmn[::-1, 1:]
            ofmn[:, 1:] = ofmn[::-1, 1:]

        ind = -1
        for mm in range(0, Mpol + 1):
            for nn in range(-Ntor, Ntor + 1):
                if mm == 0 and nn < 0:
                    continue

                ind += 1

                even_out[ind] = efmn[N + nn, mm]
                odd_out[ind] = ofmn[N + nn, mm]

        even_out_full[idx] = even_out
        odd_out_full[idx] = odd_out

    mask = _im > -1
    if transform_1d_nt:
        mask = _im == 0
    if transform_1d_nz:
        mask = _in == 0

    return even_out_full[..., mask], odd_out_full[..., mask], _im[mask], _in[mask]
