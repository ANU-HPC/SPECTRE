import numpy as np

from spectre.config import run_decorator

@run_decorator
def get_cheby_basis_single(s, lrad):
    """Evaluate Chebyshev basis functions at a single point with coordinate (s)

    Args:
        s (float): s value of the point
        lrad (int): radial resolution

    Returns:
        np.ndarray [lrad+1, 2]: modes (and derivatives) evaluated at point (s)
    """

    basis = np.zeros((lrad+1, 2)) 
        
    basis[0, 0] = 1.0
    basis[1, 0] = s
    basis[1, 1] = 1.0

    for il in range(2, lrad+1):
        basis[il, 0] = 2.0 * s * basis[il-1, 0] - basis[il-2, 0]
        basis[il, 1] = 2.0 * basis[il-1, 0] + 2.0 * s * basis[il-1, 1] - basis[il-2, 1]

    # basis recombination
    basis[:, 0] -= (-1)**np.arange(lrad+1)

    # rescale for conditioning
    for il in range(0, lrad+1):
        basis[il] /= (il + 1.0)
    
    return basis


@run_decorator
def get_zernike_basis_single(sarr, lrad, mpol):
    """Evaluate Zernike basis functions at a single point with coordinate (s)

    Args:
        s (float): s value of the point
        lrad (int): radial resolution
        mpol (int): poloidal Fourier resolution

    Returns:
        np.ndarray[lrad+1, mpol+1, 2]: modes (and derivatives) evaluated at point (s)
    """

    basis = np.zeros((lrad+1, mpol+1, 2)) 

    sbar = 0.5 * (sarr + 1.0)
    r = sbar
    rm = 1.0
    rm1 = 0.0
    for m in range(mpol+1):

        if(m <= lrad):
            basis[m, m, 0] = rm
            basis[m, m, 1] = m * rm1

        if(m+2 <= lrad):
            basis[m+2, m, 0] = (m+2) * rm * r**2.0 - (m+1) * rm
            basis[m+2, m, 1] = (m+2)**2.0 * rm * r - (m+1) * m * rm1 
        
        for il in range(m+4, lrad+1, 2):
            fac1 = il / (il**2.0 - m**2.0)
            fac2 = 4.0 * (il - 1)
            fac3 = (il - 2 + m) **2.0 / (il - 2) + (il - m)**2.0 / il
            fac4 = ((il - 2)**2.0 - m**2.0) / (il - 2.0)

            basis[il, m, 0] = fac1 * ( (fac2 * r**2.0 - fac3) * basis[il-2, m, 0] - fac4 * basis[il-4, m, 0])
            basis[il, m, 1] = fac1 * ( 2.0 * fac2 * r * basis[il-2, m, 0] 
                                        + (fac2 * r**2.0 - fac3) * basis[il-2, m, 1] - fac4 * basis[il-4, m, 1])
            
        rm1 = rm
        rm = rm * r
    
    # basis recombination
    for il in range(2, lrad+1, 2):
        basis[il, 0, 0] -= (-1)**(il/2.0)

    if(mpol >= 1):
        for il in range(3, lrad+1, 2):
            basis[il, 1, 0] -= (-1)**((il-1)*0.5) * (il+1) * 0.5 * sbar
            basis[il, 1, 1] -= (-1)**((il-1)*0.5) * (il+1) * 0.5
    
    # rescaling
    for m in range(0, mpol+1):
        for il in range(m, lrad+1, 2):
            basis[il, m, :] /= (il + 1.0)

    return basis


@run_decorator
def get_cheby_basis_single_w2derivs(s, lrad):

    basis = np.zeros((lrad+1, 3)) 
        
    basis[0, 0] = 1.0
    basis[1, 0] = s
    basis[1, 1] = 1.0

    for il in range(2, lrad+1):
        basis[il, 0] = 2.0 * s * basis[il-1, 0] - basis[il-2, 0]
        basis[il, 1] = 2.0 * basis[il-1, 0] + 2.0 * s * basis[il-1, 1] - basis[il-2, 1]
        basis[il, 2] = 2.0 * basis[il-1, 1] + 2.0 * basis[il-1, 1] - basis[il-2, 2] + 2.0 * s * basis[il-1, 2]

    # basis recombination
    basis[:, 0] -= (-1)**np.arange(lrad+1)

    # rescale for conditioning
    for il in range(0, lrad+1):
        basis[il] /= (il + 1.0)
    
    return basis


@run_decorator
def get_zernike_basis_single_w2derivs(sarr, lrad, mpol):

    basis = np.zeros((lrad+1, mpol+1, 3)) 

    sbar = 0.5 * (sarr + 1.0)
    r = sbar
    rm = 1.0
    rm1 = 0.0
    rm2 = 0.0
    for m in range(mpol+1):

        if(m <= lrad):
            basis[m, m, 0] = rm
            basis[m, m, 1] = m * rm1
            basis[m, m, 2] = m * (m - 1) * rm2

        if(m+2 <= lrad):
            basis[m+2, m, 0] = (m+2) * rm * r**2.0 - (m+1) * rm
            basis[m+2, m, 1] = (m+2)**2.0 * rm * r - (m+1) * m * rm1 
            basis[m+2, m, 2] = (m+2)**2.0 * rm * (m+1) - (m+1)*m*(m-1) * rm2
        
        for il in range(m+4, lrad+1, 2):
            fac1 = il / (il**2.0 - m**2.0)
            fac2 = 4.0 * (il - 1)
            fac3 = (il - 2 + m) **2.0 / (il - 2) + (il - m)**2.0 / il
            fac4 = ((il - 2)**2.0 - m**2.0) / (il - 2.0)

            basis[il, m, 0] = fac1 * ( (fac2 * r**2.0 - fac3) * basis[il-2, m, 0] - fac4 * basis[il-4, m, 0])
            basis[il, m, 1] = fac1 * ( 2.0 * fac2 * r * basis[il-2, m, 0] 
                                        + (fac2 * r**2.0 - fac3) * basis[il-2, m, 1] - fac4 * basis[il-4, m, 1])
            basis[il, m, 2] = fac1 * (
                2.0 * fac2 * (2.0 * r * basis[il-2, m, 1] + basis[il-2, m, 0]) + (fac2 * r**2.0 - fac3) * basis[il-2, m, 2] - fac4 * basis[il-4, m, 2]
            )
        
        rm2 = rm1
        rm1 = rm
        rm = rm * r
    
    # basis recombination
    for il in range(2, lrad+1, 2):
        basis[il, 0, 0] -= (-1)**(il/2.0)

    if(mpol >= 1):
        for il in range(3, lrad+1, 2):
            basis[il, 1, 0] -= (-1)**((il-1)*0.5) * (il+1) * 0.5 * sbar
            basis[il, 1, 1] -= (-1)**((il-1)*0.5) * (il+1) * 0.5
    
    # rescaling
    for m in range(0, mpol+1):
        for il in range(m, lrad+1, 2):
            basis[il, m, :] /= (il + 1.0)

    return basis