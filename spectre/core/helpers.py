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


class PyocFields:
	"""
	class for lazy generation of the pyoculus fields in each volume. The pyoculus fields are only generated when accessed, and then cached for future use.
	"""
	def __init__(self, SPECTREout):
		self.spectre = SPECTREout
		self._size = SPECTREout.input.physics.Mvol
		self._maps = [None] * self._size
	
	def __getitem__(self, index):
		if self._maps[index] is None:
			self._maps[index] = self.generate_map(lvol=index)
		return self._maps[index]

	def __len__(self):
		return self._size
	
	def generate_map(self, lvol):
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
	def __init__(self, spectreOUT):
		self.spectre = spectreOUT
		self._size = self.spectre.input.physics.Mvol
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