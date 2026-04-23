# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from numpy import cos, sin

from spectre.config import run_decorator
from spectre.basis import get_cheby_basis_single, get_zernike_basis_single


@run_decorator
def bfield_flow_nonsing(d_sth, phi, sth, Aze, Ate, Azo, Ato, im, in_, lrad, mpol):
    """Calculates magnetic field direction at the given point (sth[0], sth[1], phi) in a non-singular volume.
        For use in CyRK field integration. Outputs [sdot = Bs / Bz, thetadot = Bt / Bz].

    Args:
        d_sth (List[float]): placeholder for the caculted field direction
        phi (float): phi coordinate
        sth (List[float]): s and theta coordinates
        Aze (ndarray[float]): A_z vector potential cosine coefficients
        Ate (ndarray[float]): A_t vector potential cosine coefficients
        Azo (ndarray[float]): A_z vector potential sine coefficients
        Ato (ndarray[float]): A_t vector potential sine coefficients
        im (ndarray[int]): indices for poloidal modes
        in_ (ndarray[int]): indices for toroidal modes
        lrad (int): radial resolution
        mpol (int): poloidal Fourier resolution
    """

    s, theta = sth

    Bs = 0.0
    Bt = 0.0
    Bz = 0.0

    basis = get_cheby_basis_single(s, lrad)

    for ii in range(len(im)):  # CHECK CHANGE
        mi = im[ii]
        ni = in_[ii]
        arg = mi * theta - ni * phi
        carg = cos(arg)
        sarg = sin(arg)

        for il in range(lrad + 1):
            Bs += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, 0] * sarg
            Bt += (-Aze[ii, il] * basis[il, 1]) * carg
            Bz += (Ate[ii, il] * basis[il, 1]) * carg

            # # non-stell-symmetric component (NOT TESTED)
            # Bs += (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, 0] * carg
            # Bt += (- Azo[ii, l] * basis[l, 1]) * sarg
            # Bz += (Ato[ii, l] * basis[l, 1]) * sarg

    d_sth[0] = Bs / Bz
    d_sth[1] = Bt / Bz


@run_decorator
def bfield_flow_sing(d_sth, phi, sth, Aze, Ate, Azo, Ato, im, in_, lrad, mpol):
    """Calculates magnetic field direction at the given point (sth[0], sth[1], phi) in a singular volume.
        For use in CyRK field integration. Outputs [sdot = Bs / Bz, thetadot = Bt / Bz].

    Args:
        d_sth (List[float]): placeholder for the caculted field direction
        phi (float): phi coordinate
        sth (List[float]): s and theta coordinates
        Aze (ndarray[float]): A_z vector potential cosine coefficients
        Ate (ndarray[float]): A_t vector potential cosine coefficients
        Azo (ndarray[float]): A_z vector potential sine coefficients
        Ato (ndarray[float]): A_t vector potential sine coefficients
        im (ndarray[int]): indices for poloidal modes
        in_ (ndarray[int]): indices for toroidal modes
        lrad (int): radial resolution
        mpol (int): poloidal Fourier resolution
    """

    s, theta = sth

    Bs = 0.0
    Bt = 0.0
    Bz = 0.0

    basis = get_zernike_basis_single(s, lrad, mpol)

    for ii in range(len(im)):
        mi = im[ii]
        ni = in_[ii]
        arg = mi * theta - ni * phi
        carg = cos(arg)
        sarg = sin(arg)

        for il in range(lrad + 1):
            Bs += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, mi, 0] * sarg
            Bt += (-Aze[ii, il] * basis[il, mi, 1]) * 0.5 * carg
            Bz += (Ate[ii, il] * basis[il, mi, 1]) * 0.5 * carg

            # # non-stell-symmetric component (NOT TESTED)
            # Bs += (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, mi, 0] * carg
            # Bt += (- Azo[ii, l] * basis[l, mi, 1]) * 0.5 * sarg
            # Bz += (Ato[ii, l] * basis[l, mi, 1]) * 0.5 * sarg

    d_sth[0] = Bs / Bz
    d_sth[1] = Bt / Bz
