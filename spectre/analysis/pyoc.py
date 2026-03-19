from __future__ import annotations

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
	from spectre.core import SPECTRE
	from spectre.postprocess.main import SPECTREout


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