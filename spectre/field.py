# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from numpy import cos, sin

from spectre.config import run_decorator
from spectre.basis import (
    get_cheby_basis_single,
    get_cheby_basis_single_w2derivs,
    get_zernike_basis_single,
    get_zernike_basis_single_w2derivs,
)


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
def bfield_flow_tan_nonsing(d_y, phi, y, Aze, Ate, Azo, Ato, im, in_, lrad, mpol):
    """Calculate the field direction and its tangent-map derivative in a non-singular volume.

    For use in CyRK integration of the variational equation dJ/dphi = A J with
    A = d(Bs/Bz, Bt/Bz)/d(s, theta) at the point (y[0], y[1], phi). The 6-component state is
    y = (s, theta, J11, J21, J12, J22) (J in column-major order). Only the
    stellarator-symmetric part is implemented (matching ``bfield_flow_nonsing``); Azo, Ato
    and mpol are unused placeholders kept for arg-tuple compatibility.

    Args:
        d_y (List[float]): placeholder for the calculated 6-component derivative
        phi (float): phi coordinate
        y (List[float]): s, theta coordinates and the 4 tangent-map components
        Aze (ndarray[float]): A_z vector potential cosine coefficients
        Ate (ndarray[float]): A_t vector potential cosine coefficients
        Azo (ndarray[float]): A_z vector potential sine coefficients
        Ato (ndarray[float]): A_t vector potential sine coefficients
        im (ndarray[int]): indices for poloidal modes
        in_ (ndarray[int]): indices for toroidal modes
        lrad (int): radial resolution
        mpol (int): poloidal Fourier resolution

    Returns:
        None: the result is written in place into ``d_y`` as the 6-component derivative
        ``(ds/dphi, dtheta/dphi, dJ11/dphi, dJ21/dphi, dJ12/dphi, dJ22/dphi)`` -- the field
        direction plus the tangent-map derivative ``dJ/dphi = A J`` -- for CyRK integration.
    """
    s = y[0]
    theta = y[1]

    Bs = 0.0
    Bt = 0.0
    Bz = 0.0
    Bs_s = 0.0
    Bt_s = 0.0
    Bz_s = 0.0
    Bs_t = 0.0
    Bt_t = 0.0
    Bz_t = 0.0

    basis = get_cheby_basis_single_w2derivs(s, lrad)

    for ii in range(len(im)):
        mi = im[ii]
        ni = in_[ii]
        arg = mi * theta - ni * phi
        carg = cos(arg)
        sarg = sin(arg)

        for il in range(lrad + 1):
            Bs += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, 0] * sarg
            Bt += (-Aze[ii, il] * basis[il, 1]) * carg
            Bz += (Ate[ii, il] * basis[il, 1]) * carg

            # d/ds shifts the radial basis one derivative up; d/dtheta: sin -> m cos, cos -> -m sin
            Bs_s += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, 1] * sarg
            Bt_s += (-Aze[ii, il] * basis[il, 2]) * carg
            Bz_s += (Ate[ii, il] * basis[il, 2]) * carg
            Bs_t += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, 0] * mi * carg
            Bt_t += (Aze[ii, il] * basis[il, 1]) * mi * sarg
            Bz_t += (-Ate[ii, il] * basis[il, 1]) * mi * sarg

    invBz2 = 1.0 / (Bz * Bz)
    A11 = (Bs_s * Bz - Bs * Bz_s) * invBz2
    A12 = (Bs_t * Bz - Bs * Bz_t) * invBz2
    A21 = (Bt_s * Bz - Bt * Bz_s) * invBz2
    A22 = (Bt_t * Bz - Bt * Bz_t) * invBz2

    d_y[0] = Bs / Bz
    d_y[1] = Bt / Bz
    d_y[2] = A11 * y[2] + A12 * y[3]
    d_y[3] = A21 * y[2] + A22 * y[3]
    d_y[4] = A11 * y[4] + A12 * y[5]
    d_y[5] = A21 * y[4] + A22 * y[5]


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


@run_decorator
def bfield_flow_tan_sing(d_y, phi, y, Aze, Ate, Azo, Ato, im, in_, lrad, mpol):
    """Calculate the field direction and its tangent-map derivative in a singular volume.

    Zernike (axis-volume) analogue of ``bfield_flow_tan_nonsing``: same 6-component state
    y = (s, theta, J11, J21, J12, J22) and variational equation dJ/dphi = A J, with the
    radial contraction per poloidal mode m. The Zernike basis derivatives are taken w.r.t.
    sbar = (s+1)/2, so d/ds carries a factor 0.5 per derivative order (0.5 on the first
    derivative as in ``bfield_flow_sing``, 0.25 on the second). Only the
    stellarator-symmetric part is implemented; Azo and Ato are unused placeholders kept for
    arg-tuple compatibility.

    Args:
        d_y (List[float]): placeholder for the calculated 6-component derivative
        phi (float): phi coordinate
        y (List[float]): s, theta coordinates and the 4 tangent-map components
        Aze (ndarray[float]): A_z vector potential cosine coefficients
        Ate (ndarray[float]): A_t vector potential cosine coefficients
        Azo (ndarray[float]): A_z vector potential sine coefficients
        Ato (ndarray[float]): A_t vector potential sine coefficients
        im (ndarray[int]): indices for poloidal modes
        in_ (ndarray[int]): indices for toroidal modes
        lrad (int): radial resolution
        mpol (int): poloidal Fourier resolution

    Returns:
        None: the result is written in place into ``d_y`` as the 6-component derivative
        ``(ds/dphi, dtheta/dphi, dJ11/dphi, dJ21/dphi, dJ12/dphi, dJ22/dphi)`` -- the field
        direction plus the tangent-map derivative ``dJ/dphi = A J`` -- for CyRK integration.
    """
    s = y[0]
    theta = y[1]

    Bs = 0.0
    Bt = 0.0
    Bz = 0.0
    Bs_s = 0.0
    Bt_s = 0.0
    Bz_s = 0.0
    Bs_t = 0.0
    Bt_t = 0.0
    Bz_t = 0.0

    basis = get_zernike_basis_single_w2derivs(s, lrad, mpol)

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

            # d/ds shifts the radial basis one derivative (x0.5 per order, sbar chain rule);
            # d/dtheta: sin -> m cos, cos -> -m sin
            Bs_s += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, mi, 1] * 0.5 * sarg
            Bt_s += (-Aze[ii, il] * basis[il, mi, 2]) * 0.25 * carg
            Bz_s += (Ate[ii, il] * basis[il, mi, 2]) * 0.25 * carg
            Bs_t += (-mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, mi, 0] * mi * carg
            Bt_t += (Aze[ii, il] * basis[il, mi, 1]) * 0.5 * mi * sarg
            Bz_t += (-Ate[ii, il] * basis[il, mi, 1]) * 0.5 * mi * sarg

    invBz2 = 1.0 / (Bz * Bz)
    A11 = (Bs_s * Bz - Bs * Bz_s) * invBz2
    A12 = (Bs_t * Bz - Bs * Bz_t) * invBz2
    A21 = (Bt_s * Bz - Bt * Bz_s) * invBz2
    A22 = (Bt_t * Bz - Bt * Bz_t) * invBz2

    d_y[0] = Bs / Bz
    d_y[1] = Bt / Bz
    d_y[2] = A11 * y[2] + A12 * y[3]
    d_y[3] = A21 * y[2] + A22 * y[3]
    d_y[4] = A11 * y[4] + A12 * y[5]
    d_y[5] = A21 * y[4] + A22 * y[5]
