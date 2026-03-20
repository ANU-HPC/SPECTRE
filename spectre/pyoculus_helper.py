import numpy as np
from numpy import cos, sin
from typing import Union

from spectre.basis import get_cheby_basis_single_w2derivs, get_zernike_basis_single_w2derivs
from spectre.postprocess.config import run_decorator
from spectre.postprocess.main import SPECTREout
from spectre.core.core import SPECTRE

class PyocFields:
	"""
	class for lazy generation of the pyoculus fields in each volume. The pyoculus fields are only generated when accessed, and then cached for future use.
	"""
	def __init__(self, spectre: Union[SPECTRE, SPECTREout], mvol: int):
		self.spectre = spectre
		self._size = mvol
		self._fields = [None] * self._size
	
	def __getitem__(self, index):
		if self._fields[index] is None:
			self._fields[index] = self.generate_field(lvol=index)
		return self._fields[index]

	def __len__(self):
		return self._size
	
	def generate_field(self, lvol):
		"""
		Generate the pyoculus map for the given volume.
		"""
		from pyoculus.fields import SpectreBfield
		bfield = SpectreBfield(self.spectre.pyoculus_helpers[lvol], lvol=lvol)
		return bfield


class PyocHelpers:
	"""
	class for lazy generation of the pyoculus helpers in each volume. The pyoculus fields are only generated when accessed, and then cached for future use.
	"""
	def __init__(self, spectre: Union[SPECTRE, SPECTREout], mvol: int):
		self.spectre = spectre
		self._size = mvol
		self._helpers = [None] * self._size
	
	def __getitem__(self, index):
		if self._helpers[index] is None:
			self._helpers[index] = self.generate_helper(lvol=index)
		return self._helpers[index]

	def __len__(self):
		return self._size
	
	def generate_helper(self, lvol):
		"""
		Generate the pyoculus helper for the given volume.
		"""
		from spectre.pyoculus_helper import SPECTRE_pyoculus_helper
		helper = SPECTRE_pyoculus_helper(self.spectre, lvol)
		return helper
     
class SPECTRE_pyoculus_helper:

    def __init__(self, obj, lvol = 0):
        """Created from a SPECTREout object"""

        self.lvol = lvol

        if isinstance(obj, SPECTREout):
            self.stellsym = obj.input.physics.enforce_stell_sym
            self.lrad = obj.input.physics.Lrad[lvol]
            self.mpol = obj.input.physics.Mpol
            self.im = obj.output.im
            self.in_ = obj.output.in_
            self.Aze = np.atleast_2d(obj.vector_potential.Aze[lvol])
            self.Ate = np.atleast_2d(obj.vector_potential.Ate[lvol])
            self.Azo = np.atleast_2d(obj.vector_potential.Azo[lvol])
            self.Ato = np.atleast_2d(obj.vector_potential.Ato[lvol])
            self.geometry = obj.input.physics.Igeometry
            self.Rac, self.Rbc = obj.output.Rbc[lvol : lvol + 2]
            self.Zas, self.Zbs = obj.output.Zbs[lvol : lvol + 2]
            self.Nfp = obj.input.physics.Nfp

        elif isinstance(obj, SPECTRE):
            from spectre.misc import get_vec_pot
            self.stellsym = obj.input_list_mod.enforce_stell_sym
            self.lrad = obj.input_list_mod.lrad[lvol]
            self.mpol = obj.input_list_mod.mpol
            self.im = obj.allglobal_mod.im
            self.in_ = obj.allglobal_mod.in_
            ate_list, aze_list, ato_list, azo_list = get_vec_pot(obj)
            self.Aze = np.atleast_2d(aze_list[lvol])
            self.Ate = np.atleast_2d(ate_list[lvol])
            self.Azo = np.atleast_2d(azo_list[lvol])
            self.Ato = np.atleast_2d(ato_list[lvol])
            self.geometry = obj.input_list_mod.igeometry
            self.Rac, self.Rbc = obj.allglobal_mod.irbc.T[lvol : lvol + 2]
            self.Zas, self.Zbs = obj.allglobal_mod.izbs.T[lvol : lvol + 2]
            self.Nfp = obj.input_list_mod.nfp

        else:
            raise RuntimeError("obj must be a SPECTREout or SPECTRE object")  
        
        self.singularity = (self.geometry > 1 and self.lvol == 0) 
        
    def B(self, coords, *args):
        """! Returns magnetic fields
        @param coords with s, theta, zeta
        @param *args extra parameters
        @returns the contravariant magnetic fields
        """

        return bfield_noderiv(
            coords,
            self.Aze, 
            self.Ate, 
            self.Azo, 
            self.Ato, 
            self.im, 
            self.in_, 
            self.lrad, 
            self.mpol,
            self.singularity
        )
        
    def dBdX(self, coords, *args):
        """! Returns the magnetic field and derivative wrt. s, theta, zeta
        @param coords with s, theta, zeta
        @param *args extra parameters
        @returns B, dBdX, the contravariant magnetic fields, the derivatives of them
        """

        return bfield_wderiv(
            coords,
            self.Aze, 
            self.Ate, 
            self.Azo, 
            self.Ato, 
            self.im, 
            self.in_, 
            self.lrad, 
            self.mpol,
            self.singularity
        )
        
    def convert_coords(self, coords):
        """! Python wrapper for getting the R, phi, Z coordinates from s, theta, zeta
        @param coords with s, theta, zeta
        @returns R, phi, Z
        """

        r, z = get_rz_coords(
            self.geometry, self.Rac, self.Rbc, self.Zas, self.Zbs, self.im, self.in_, *coords, self.lvol
        )

        return np.array([r, coords[2], z]) # R, phi, Z

    def A(self, coords):
        """! Returns the vector potential at the given coordinates
        @param coords with s, theta, zeta
        @returns At, Az
        """

        return vecpot_noderiv(
            coords,
            self.Aze, 
            self.Ate, 
            self.Azo, 
            self.Ato, 
            self.im, 
            self.in_, 
            self.lrad, 
            self.mpol,
            self.stellsym
        )
    
@run_decorator
def bfield_wderiv(coords, Aze, Ate, Azo, Ato, im, in_, lrad, mpol, singularity):

    s, theta, phi = coords
    
    Bs = 0.0
    Bt = 0.0
    Bz = 0.0
    dBs_ds = 0.0
    dBs_dt = 0.0
    dBs_dp = 0.0
    dBt_ds = 0.0
    dBt_dt = 0.0
    dBt_dp = 0.0
    dBz_ds = 0.0
    dBz_dt = 0.0
    dBz_dp = 0.0
    
    if singularity:
        basis = get_zernike_basis_single_w2derivs(s, lrad, mpol)
        basis[:, :, 1] *= 0.5
        basis[:, :, 2] *= 0.25
    else:
        basis = get_cheby_basis_single_w2derivs(s, lrad)[:,None,:]

    for ii in range(len(im)):
        mi = im[ii]
        ni = in_[ii]
        arg = mi * theta - ni * phi
        carg = cos(arg)
        sarg = sin(arg)
        
        for il in range(lrad+1):
            
            term0 = - mi * Aze[ii, il] - ni * Ate[ii, il]
            term1 = Aze[ii, il] * basis[il, mi, 1]
            term2 = Ate[ii, il] * basis[il, mi, 1]

            Bs += term0 * basis[il, mi, 0] * sarg
            Bt += - term1 * carg
            Bz += term2 * carg

            dBs_ds += term0 * basis[il, mi, 1] * sarg
            dBt_ds += (- Aze[ii, il] * basis[il, mi, 2]) * carg
            dBz_ds += (Ate[ii, il] * basis[il, mi, 2]) * carg

            dBs_dt += term0 * basis[il, mi, 0] * carg * mi
            dBt_dt += term1 * sarg * mi
            dBz_dt += - term2 * sarg * mi

            dBs_dp += - term0 * basis[il, mi, 0] * carg * ni
            dBt_dp += - term1 * sarg * ni
            dBz_dp += term2 * sarg * ni

            # # non-stell-symmetric component (NOT TESTED)
            # Bs += (- mi * Azo[ii, l] - ni * Ato[ii, l]) * basis[l, mi, 0] * carg
            # Bt += (- Azo[ii, l] * basis[l, mi, 1]) * sarg
            # Bz += (Ato[ii, l] * basis[l, mi, 1]) * sarg
            # dBs_ds, dBs_dt, dBs_dp, dBt_ds, dBt_dt, dBt_dp, dBz_ds, dBz_dt, dBz_dp ...
    
    return np.array([Bs, Bt, Bz]), np.array([[dBs_ds, dBs_dt, dBs_dp], [dBt_ds, dBt_dt, dBt_dp], [dBz_ds, dBz_dt, dBz_dp]]).T

@run_decorator
def bfield_noderiv(coords, Aze, Ate, Azo, Ato, im, in_, lrad, mpol, singularity):

    s, theta, phi = coords
    
    Bs = 0.0
    Bt = 0.0
    Bz = 0.0
    
    if singularity:
        basis = get_zernike_basis_single_w2derivs(s, lrad, mpol)
        basis[:, :, 1] *= 0.5
    else:
        basis = get_cheby_basis_single_w2derivs(s, lrad)[:,None,:]

    for ii in range(len(im)):
        mi = im[ii]
        ni = in_[ii]
        arg = mi * theta - ni * phi
        carg = cos(arg)
        sarg = sin(arg)
        
        for il in range(lrad+1):
            Bs += (- mi * Aze[ii, il] - ni * Ate[ii, il]) * basis[il, mi, 0] * sarg
            Bt += (- Aze[ii, il] * basis[il, mi, 1]) * carg
            Bz += (Ate[ii, il] * basis[il, mi, 1]) * carg
    
    return np.array([Bs, Bt, Bz])


@run_decorator
def get_rz_coords(
    geometry, Rac, Rbc, Zas, Zbs, im, in_, sarr, tarr, zarr, lvol
):

    sbar = (sarr + 1.0) / 2.0

    fac0 = np.empty((len(im)))

    if lvol > 0 or geometry == 1:  # no coordinate singularity
        fac0[:] = sbar
    else:  # with coordinate singularity
        if geometry == 2:  # cylindrical
            fac0[:] = sbar ** (im + 1.0)
            fac0[im == 0] = sbar
        elif geometry == 3:  # toroidal
            fac0[:] = sbar ** im
            fac0[im == 0] = sbar**2

    r0 = 0.0
    z0 = 0.0

    if geometry == 3:
        dR1 = Rac + fac0[:] * (Rbc - Rac)
        dZ1 = Zas + fac0[:] * (Zbs - Zas)

        for m in range(len(im)):
            angle = im[m] * tarr - in_[m] * zarr
            r0 += dR1[m] * np.cos(angle)
            z0 += dZ1[m] * np.sin(angle)
    else:
        dR1 = Rac + fac0[:] * (Rbc - Rac)

        for m in range(len(im)):
            angle = im[m] * tarr - in_[m] * zarr
            r0 += dR1[m] * np.cos(angle)

    return r0, z0

@run_decorator
def vecpot_noderiv(coords, Aze, Ate, Azo, Ato, im, in_, lrad, mpol, singularity):

    s, theta, phi = coords
    
    # Gauge in SPECTRE: As = 0.0
    At = 0.0
    Az = 0.0
    
    if singularity:
        basis = get_zernike_basis_single_w2derivs(s, lrad, mpol)
        basis[:, :, 1] *= 0.5
    else:
        basis = get_cheby_basis_single_w2derivs(s, lrad)[:,None,:]

    for ii in range(len(im)):
        mi = im[ii]
        ni = in_[ii]
        arg = mi * theta - ni * phi
        carg = cos(arg)
        
        for il in range(lrad+1):            
            At += Ate[ii, il] * basis[il, mi, 0] * carg
            Az += Aze[ii, il] * basis[il, mi, 0] * carg

            # # + non-stell-symmetric components...
            # At += Ato[ii, il] * basis[il, mi, 0] * sarg
            # Az += Azo[ii, il] * basis[il, mi, 0] * sarg
    
    return np.array([At, Az])