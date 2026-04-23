# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np



def allrzrz_from_interface_arrays(irbc, izbs, irbs, izbc, mn, im, in_, mvol, nfp):
    """
    construct the allRZRZ (interface guess) array from the interfaces and helper 
    arrays
    Args: 
        irbc: Even Fourier coefficients for the R-value of the interfaces
        izbs: Odd Fourier coefficients for the Z-valueof the interfaces
        irbs: Odd Fourier coefficients for the R-value of the interfaces
        izbc: Even Fourier coefficients for the Z-value of the interfaces
        mn: number of poloidal/toroidal modes
        im: array of poloidal mode numbers
        in_: array of toroidal mode numbers
        mvol: number of volumes
    """
    allrzrz = dict()
    for vvol in range(1, mvol+1):
        interface = dict()
        rbc = {}
        zbs = {}
        rbs = {}
        zbc = {}
        for imn in range(mn):
            rbc[f"({im[imn]}, {in_[imn]//nfp})"] = irbc[imn, vvol]
            zbs[f"({im[imn]}, {in_[imn]//nfp})"] = izbs[imn, vvol]
            rbs[f"({im[imn]}, {in_[imn]//nfp})"] = irbs[imn, vvol]
            zbc[f"({im[imn]}, {in_[imn]//nfp})"] = izbc[imn, vvol]
        interface["rbc"] = rbc
        interface["zbs"] = zbs
        interface["rbs"] = rbs
        interface["zbc"] = zbc
        allrzrz[f'interface_{vvol}'] = interface
    return allrzrz