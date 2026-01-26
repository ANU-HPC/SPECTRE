import h5py
import numpy as np
import os.path
import keyword
import matplotlib.pyplot as plt

from spectre.field_tracer import Field_Tracer

class SPECout:
	
	from ._get_field import get_field_contrav, get_field_cart, get_field_mod, get_cheby_basis_single, calc_avg_beta_vol
	from ._get_geometry import get_metric, get_jacobian, get_coord_transform, get_coord_transform_list, get_infaces_norm, \
		calc_interfaces_area, get_infaces_dAvec, calc_flux_surface_average
	from ._check_isect import check_intersection_interfaces
	
	def __init__(self, *args, **kwargs):
		# args[0] should always be the name of a file or an item inside the root object
		# if args[0] is not a filename, kwargs['content'] should be the content to be added
		# as self.`args[0]`

		_content = None
		if kwargs.get("content") is None:
			# assume arg[0] is a filename
			if args[0][-3:] != ".h5":
				raise RuntimeError(f"Input file ({args[0]}) must be a .h5 file")
			_content = h5py.File(args[0], "r")

			# keep track of which file this object corresponds to
			self.filename = os.path.abspath(args[0])

		elif isinstance(kwargs["content"], h5py.Group):
			_content = kwargs["content"]

		if _content is not None:
			for key in _content:
				if isinstance(_content[key], h5py.Group):
					# recurse into group
					if key in keyword.kwlist:  # avoid assign python keywords
						setattr(self, key + "1", SPECout(content=_content[key]))
					else:
						setattr(self, key, SPECout(content=_content[key]))
				elif isinstance(_content[key], h5py.Dataset):  # read dataset
					if (
						key in keyword.kwlist
					):  # add underscore avoiding assigning python keywords
						setattr(self, key + "_", _content[key][()])
					else:
						if (
							len(_content[key][()]) == 1
						):  # if just one element, use the value directly
							setattr(self, key, _content[key][0])
						else:
							setattr(self, key, _content[key][()])

		if isinstance(_content, h5py.File):
			_content.close()

			# make sure that Lrad is always an array
			if np.isscalar(self.input.physics.Lrad):
				self.input.physics.Lrad = np.array([self.input.physics.Lrad])
			# make sure that im always an array
			if np.isscalar(self.output.im):
				self.output.im = np.array([self.output.im])
			# make sure that in_ is always an array
			if np.isscalar(self.output.in_):
				self.output.in_ = np.array([self.output.in_])

			# these define the target dimensions in the radial direction
			Nvol = self.input.physics.Nvol
			Mvol = Nvol
			if self.input.physics.Lfreebound:
				Mvol += 1
			self.input.physics.Mvol = Mvol

			Lrad = self.input.physics.Lrad

			# lists for vector potential
			cAte = []
			cAto = []
			cAze = []
			cAzo = []

			# lists for grid
			cRij = []
			cZij = []
			csg = []
			cBR = []
			cBp = []
			cBZ = []

			# split up radial matrix dimension into list of matrices for each of the nested volumes
			start = 0
			for i in range(Mvol):
				# vector potential
				cAte.append(
					np.atleast_2d(self.vector_potential.Ate)[
						:, start : start + Lrad[i] + 1
					]
				)
				cAto.append(
					np.atleast_2d(self.vector_potential.Ato)[
						:, start : start + Lrad[i] + 1
					]
				)
				cAze.append(
					np.atleast_2d(self.vector_potential.Aze)[
						:, start : start + Lrad[i] + 1
					]
				)
				cAzo.append(
					np.atleast_2d(self.vector_potential.Azo)[
						:, start : start + Lrad[i] + 1
					]
				)

				# grid
				if hasattr(self.grid, 'Rij') :
					cRij.append(
						np.atleast_2d(self.grid.Rij)[:, start : start + Lrad[i] + 1]
					)
				if hasattr(self.grid, 'Zij'):
					cZij.append(
						np.atleast_2d(self.grid.Zij)[:, start : start + Lrad[i] + 1]
					)
				if hasattr(self.grid, 'sg'):
					csg.append(np.atleast_2d(self.grid.sg)[:, start : start + Lrad[i] + 1])
				if hasattr(self.grid, 'BR'):
					cBR.append(np.atleast_2d(self.grid.BR)[:, start : start + Lrad[i] + 1])
				if hasattr(self.grid, 'Bp'):
					cBp.append(np.atleast_2d(self.grid.Bp)[:, start : start + Lrad[i] + 1])
				if hasattr(self.grid, 'BZ'):
					cBZ.append(np.atleast_2d(self.grid.BZ)[:, start : start + Lrad[i] + 1])

				# move along the merged array dimension
				start = start + Lrad[i] + 1

			# replace original content in data structure
			self.vector_potential.Ate = cAte
			self.vector_potential.Ato = cAto
			self.vector_potential.Aze = cAze
			self.vector_potential.Azo = cAzo

			if hasattr(self.grid, 'Rij') :
				self.grid.Rij = cRij
			if hasattr(self.grid, 'Zij'):
				self.grid.Zij = cZij
			if hasattr(self.grid, 'sg'):
				self.grid.sg = csg
			if hasattr(self.grid, 'BR'):
				self.grid.BR = cBR
			if hasattr(self.grid, 'Bp'):
				self.grid.Bp = cBp
			if hasattr(self.grid, 'BZ'):
				self.grid.BZ = cBZ

			if hasattr(self, "poincare"):
				# remove unsuccessful Poincare trajectories
				self.poincare.R = self.poincare.R[self.poincare.success == 1, :, :]
				self.poincare.Z = self.poincare.Z[self.poincare.success == 1, :, :]
				self.poincare.t = self.poincare.t[self.poincare.success == 1, :, :]
				self.poincare.s = self.poincare.s[self.poincare.success == 1, :, :]

		return

	# needed for iterating over the contents of the file
	def __iter__(self):
		return iter(self.__dict__)

	def __next__(self):
		return next(self.__dict__)

	# needed for using SPECout with 'with' statement
	def __enter__(self):
		return self

	def __exit__(self, t, v, tb):
		return

	# print a list of items contained in this object
	def inventory(self, prefix=""):
		_prefix = ""
		if prefix != "":
			_prefix = prefix + "/"

		for a in self:
			try:
				# recurse into member
				getattr(self, a).inventory(prefix=_prefix + a)
			except Exception:
				# print item name
				print(_prefix + a)

	def plot_iota(self, xaxis="R", yaxis="i", reverse_y_axis=False, ax=None, plot=True, **kwargs):
		"""Iota plots
		@param xaxis the choose of the xaxis, 'R'(default) or 's'
		@param yaxis the choose of the yaxis, 'i'(default) or 'q'
		@param ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
		@param kwargs (dict, optional): keyword arguments. Matplotlib.pyplot.scatter keyword arguments.
		@raises ValueError: xaxis or yaxis illegal
		@returns pyplot.scatter: Matplotlib.pyplot.scatter returns
		"""

		if kwargs.get("marker") is None:
			kwargs.update({"marker": "*"})
		if kwargs.get("c") is None:
			pass
			# kwargs.update({"c": "gray"})

		if xaxis == "s":
			xdata = self.transform.fiota[0, :]
			ydata = self.transform.fiota[1, :]
			xlabel = r"s"
		elif xaxis == "R":
			xdata = self.poincare.R[:, 0, 0]
			ydata = self.transform.fiota[1, self.poincare.success == 1]
			xlabel = r"R"
		else:
			raise ValueError("xaxis should be one of ['R', 's'].")

		if yaxis == "i":
			ylabel = r"$\iota$"
		elif yaxis == "q":
			ydata = 1.0 / ydata
			ylabel = r"q"
		else:
			raise ValueError("yaxis should be one of ['i', 'q'].")
		
		if reverse_y_axis:
			ydata = -ydata
		
		if(plot is False):
			return xdata, ydata
		
		if ax is None:
			_, ax = plt.subplots()

		ax.plot(xdata, ydata, **kwargs)

		ax.set_ylabel(ylabel)
		ax.set_xlabel(xlabel)

		return xdata, ydata

	def plot_jacobian(self, surfaces=None, num_theta=400, zetan=0.0, ax=None, shift=0.0, color='r', prange='full', plot_axis=False, plot_comp_bndry=True, **kwargs):
		
		nfp = self.input.physics.Nfp		
		zeta = zetan * (2*np.pi/nfp)

		if ax is None:
			_, ax = plt.subplots(figsize=(8,6))
		else:
			plt.sca(ax)
		
		sarr = np.linspace(-.999,1,48)
		tarr = np.linspace(0, 2*np.pi, 148)
		jacs = []
		for lvol in range(self.input.physics.Nvol):
			jac = self.get_jacobian(sarr, tarr, zeta, lvol)[:,:,0]
			jacs.append(jac)
			
			rarr, zarr = self.get_coord_transform(lvol, sarr, tarr, zeta)
			rarr = rarr[0,:,:,0]
			zarr = zarr[0,:,:,0]
			
			ax.plot(rarr[-1], zarr[-1], 'k-', lw=1.)
			im = ax.pcolormesh(rarr, zarr, jac, shading='gouraud')
		
		plt.colorbar(im, ax=ax)

		ax.axis('equal')
		jac_total = np.array(jacs)
		plt.title(f"Jacobian($\\zeta$={zeta:.2f}):  max {np.max(jac_total):.4e}   min {np.min(jac_total):.4e}")

	def plot_infaces(self, surfaces=None, num_theta=400, zetan=0.0, ax=None, shift=0.0, color='r', prange='full', plot_axis=False, plot_comp_bndry=True, **kwargs):

		fontsize_labels = 13
		
		if(ax is None):
			fig, ax = plt.subplots()
			newaxis = True
		else:
			fig = plt.gcf()
			newaxis = False
		
		thetas = np.linspace(0, 2*np.pi, num_theta)
		figsize = 6.0
		
		geometry = self.input.physics.Igeometry
		nvol = self.input.physics.Nvol
		nfp = self.input.physics.Nfp

		plot_comp_bndry = plot_comp_bndry and self.input.physics.Lfreebound

		if(surfaces is None):
			surfaces = np.arange(nvol)

		if('label' not in kwargs.keys()):
			kwargs['label'] = None

		zeta = zetan * (2*np.pi/nfp)

		if(geometry == 1):
			for s in surfaces:
				rarr, _ = self.get_coord_transform(s, 1.0, thetas, zeta)
				r = rarr[0,0,:,0]
				
				ax.plot(thetas, r, color=color, **kwargs)

				if(kwargs['label'] is not None):
					kwargs['label'] = None
				
			ax.set_xlabel('R [m]', fontsize=fontsize_labels)
			ax.set_ylabel('Z [m]', fontsize=fontsize_labels)
			ax.set_xlim(0, 2*np.pi)
			ax.set_ylim(0, 2*np.pi)
			
			if(newaxis):
				fig.set_size_inches(figsize, figsize)

		elif(geometry == 2):
			for s in surfaces:
				rarr, _ = self.get_coord_transform(s, 1.0, thetas, zeta)
				r = rarr[0,0,:,0]

				x_data = r*np.cos(thetas) + shift
				y_data = r*np.sin(thetas)

				if prange == "full":
					pass
				elif prange == "upper":
					mask = y_data > 0.
					x_data = x_data[mask]
					y_data = y_data[mask]
				elif prange == "lower":
					mask = y_data < 0.
					x_data = x_data[mask]
					y_data = y_data[mask]
				else:
					raise ValueError("Bad prange input")
				
				if(s == (nvol-1)):
					ax.plot(x_data, y_data, color='k', **kwargs)
					continue    

				if(s == (nvol-1)):
					ax.plot(x_data, y_data, color='k', **kwargs)
					continue
				
				ax.plot(x_data, y_data, color=color, **kwargs)

				if(kwargs['label'] is not None):
					kwargs['label'] = None
				
			ax.set_xlabel('x [m]', fontsize=fontsize_labels)
			ax.set_ylabel('y [m]', fontsize=fontsize_labels)
			
			ax.axis('equal')
		
		elif(geometry == 3):
			for s in surfaces:
				rarr, zarr = self.get_coord_transform(s, 1.0, thetas, zeta)
				r = rarr[0,0,:,0]
				z = zarr[0,0,:,0]

				x_data = r + shift
				y_data = z

				if prange == "full":
					pass
				elif prange == "upper":
					mask = y_data > 0.
					x_data = x_data[mask][1:]
					y_data = y_data[mask][1:]
				elif prange == "lower":
					mask = y_data < 1e-5
					x_data = x_data[mask][:-1]
					y_data = y_data[mask][:-1]
				else:
					raise ValueError("Bad prange input")
				
				if(s == (nvol-1)):
					ax.plot(x_data, y_data, color=color, **kwargs)
					continue            
				
				ax.plot(x_data, y_data, color=color, **kwargs)

				if(kwargs['label'] is not None):
					kwargs['label'] = None
				
			if plot_axis:
				rarr_axis, zarr_axis = self.get_coord_transform(0, -1.0, 0., zeta)
				ax.plot(rarr_axis[0,0,:,0], zarr_axis[0,0,:,0], '+', ms=7, c=color)

			if plot_comp_bndry:   
				rarr, zarr = self.get_coord_transform(nvol, 1.0, thetas, zeta)
				r = rarr[0,0,:,0]
				z = zarr[0,0,:,0]
				x_data = r + shift
				y_data = z

				if prange == "full":
					pass
				elif prange == "upper":
					mask = y_data > 0.
					x_data = x_data[mask]
					y_data = y_data[mask]
				elif prange == "lower":
					mask = y_data < 0.
					x_data = x_data[mask]
					y_data = y_data[mask]
				else:
					raise ValueError("Bad prange input")          
				
				kwargs['ls'] = '--'
				ax.plot(x_data, y_data, color='g', **kwargs)

			if prange != 'full':
				ax.axhline(y=0, xmin=0.02, xmax=.98, c='k', lw=1)
			
			ax.set_xlabel('R [m]', fontsize=fontsize_labels)
			ax.set_ylabel('Z [m]', fontsize=fontsize_labels)
			
			ax.axis('equal')
			if(newaxis):
				aspect_ratio = (z.max() - z.min()) / (r.max() - r.min())
				fig.set_size_inches(figsize, figsize * aspect_ratio)
			
		else:
			raise ValueError(f"Invalid geometry {geometry}")

	def plot_infaces3D(self):

		raise NotImplementedError("3D interface plotting not implemented yet")

	def plot_modB(
		self,
		sarr=np.linspace(-0.999, 1, 4),
		tarr=np.linspace(0, 2 * np.pi, 5),
		zarr=np.linspace(0, 0, 2),
		lvol=None,
		ax=None,
		colorbar=True,
		**kwargs,
	):
		"""[summary]
		Args:
			sarr ([type], optional): [description]. Defaults to np.linspace(-0.999, 1, 4).
			tarr ([type], optional): [description]. Defaults to np.linspace(0, 2 * np.pi, 5).
			zarr ([type], optional): [description]. Defaults to np.linspace(0, 0, 2).
			lvol (int, optional): [description]. Defaults to 0.
			ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
		"""

		Nvol = self.input.physics.Nvol
		if lvol is None:
			lvollist = np.arange(0, Nvol, dtype=int).tolist()
		elif np.isscalar(lvol):
			lvollist = [lvol]
		else:
			lvollist = lvol
		# get axis data
		if ax is None:
			fig, ax = plt.subplots()
		plt.sca(ax)

		nr = len(sarr)
		nt = len(tarr)

		plotR = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
		plotZ = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
		plotB = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)

		for i, ivol in enumerate(lvollist):
			R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
				lvol=ivol, sarr=sarr, tarr=tarr, zarr=zarr
			)
			Bcontrav = self.get_B(
				lvol=ivol, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr
			)
			modB = self.get_modB(Bcontrav, g)

			Igeometry = self.input.physics.Igeometry

			if Igeometry == 1:
				plotR[i * nr : (i + 1) * nr, :] = tarr[None, :]
				plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0]
			if Igeometry == 2:
				plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.cos(tarr[None, :])
				plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.sin(tarr[None, :])
			if Igeometry == 3:
				plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0]
				plotZ[i * nr : (i + 1) * nr, :] = Z[:, :, 0]

			plotB[i * nr : (i + 1) * nr, :] = modB[:, :, 0]
		plot = ax.pcolormesh(plotR[:, :], plotZ[:, :], plotB[:, :], **kwargs)
		if Igeometry == 1:
			ax.set_xlabel(r"$\theta$")
			ax.set_ylabel(r"$R$")
		if Igeometry == 2:
			ax.set_aspect("equal")
			ax.set_xlabel(r"$X$")
			ax.set_ylabel(r"$Y$")
		if Igeometry == 3:
			#ax.set_aspect("equal")
			ax.set_xlabel(r"$R$")
			ax.set_ylabel(r"$Z$")

		if colorbar:
			cbar = plt.colorbar(plot, ax=ax)
			cbar.set_label(r"$|\vec B|$", rotation=0)
		return

	def plot_fmn(self, intf, ntheta, **kwargs):  

		tarr=np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
		zarr=np.linspace(0, 0, 1)

		sarr = np.array([1.0])
		R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
				lvol=intf, sarr=sarr, tarr=tarr, zarr=zarr)
		# Bcontrav = self.get_B(
		#     lvol=intf, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr)
		
		Bcontrav = self.get_field_contrav(intf, sarr, tarr, zarr)
		
		f_th_zeta_1 = np.sqrt(np.einsum("i...,...ji,j...->...", Bcontrav, g, Bcontrav))[0,:,0]

		sarr = np.array([-1.0])
		R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
				lvol=intf+1, sarr=sarr, tarr=tarr, zarr=zarr)
		# Bcontrav = self.get_B(
		#     lvol=intf+1, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr)
		
		Bcontrav = self.get_field_contrav(intf+1, sarr, tarr, zarr)

		f_th_zeta_2 = np.sqrt(np.einsum("i...,...ji,j...->...", Bcontrav, g, Bcontrav))[0,:,0]

		f_jump = f_th_zeta_2**2 - f_th_zeta_1**2

		f_mn = np.real(np.fft.rfft(f_jump))

		return f_jump, f_mn


	def plot_cartB(
		self,
		sarr=np.linspace(-0.999, 1, 4),
		tarr=np.linspace(0, 2 * np.pi, 5),
		zarr=np.linspace(0, 0, 2),
		lvol=None,
		ax=None,
		colorbar=True,
		comp=0,
		**kwargs,
	):
		"""[summary]
		Args:
			sarr ([type], optional): [description]. Defaults to np.linspace(-0.999, 1, 4).
			tarr ([type], optional): [description]. Defaults to np.linspace(0, 2 * np.pi, 5).
			zarr ([type], optional): [description]. Defaults to np.linspace(0, 0, 2).
			lvol (int, optional): [description]. Defaults to 0.
			ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
		"""

		Nvol = self.input.physics.Nvol
		if lvol is None:
			lvollist = np.arange(0, Nvol, dtype=int).tolist()
		elif np.isscalar(lvol):
			lvollist = [lvol]
		else:
			lvollist = lvol

		if ax is None:
			fig, ax = plt.subplots()
		plt.sca(ax)

		nr = len(sarr)
		nt = len(tarr)

		plotR = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
		plotZ = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)
		plotB = np.zeros([len(lvollist) * nr, nt], dtype=np.float64)

		for i, ivol in enumerate(lvollist):
			R, Z, jacobian, g = self.get_grid_and_jacobian_and_metric(
				lvol=ivol, sarr=sarr, tarr=tarr, zarr=zarr
			)
			Bcontrav = self.get_B(
				lvol=ivol, jacobian=jacobian, sarr=sarr, tarr=tarr, zarr=zarr
			)

			Igeometry = self.input.physics.Igeometry

			if Igeometry == 1:
				plotR[i * nr : (i + 1) * nr, :] = tarr[None, :]
				plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0]
			if Igeometry == 2:
				plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.cos(tarr[None, :])
				plotZ[i * nr : (i + 1) * nr, :] = R[:, :, 0] * np.sin(tarr[None, :])
			if Igeometry == 3:
				plotR[i * nr : (i + 1) * nr, :] = R[:, :, 0]
				plotZ[i * nr : (i + 1) * nr, :] = Z[:, :, 0]

			plotB[i * nr : (i + 1) * nr, :] = Bcontrav[:, :, 0, comp]
		plot = ax.pcolormesh(plotR[:, :], plotZ[:, :], plotB[:, :], **kwargs)
		if Igeometry == 1:
			ax.set_xlabel(r"$\theta$")
			ax.set_ylabel(r"$R$")
		if Igeometry == 2:
			ax.set_aspect("equal")
			ax.set_xlabel(r"$X$")
			ax.set_ylabel(r"$Y$")
		if Igeometry == 3:
			ax.set_xlabel(r"$R$")
			ax.set_ylabel(r"$Z$")

		if colorbar:
			cbar = plt.colorbar(plot, ax=ax)
			cbar.set_label(r"$|\vec B|$", rotation=0)
		return

	def plot_poincare(self, zetan=0, prange="full", ax=None, **kwargs):
		"""Poincare plots
		Args:
			toroidalIdx (int, optional): The index of toroidal cross-section to be plotted. Defaults to 0.
			prange (str, optional): Range of plotted points, one of ['full', 'upper', 'lower']. Defaults to 'full'.
			ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
			kwargs (dict, optional): keyword arguments. Matplotlib.pyplot.scatter keyword arguments.
		Returns:
			pyplot.scatter: Matplotlib.pyplot.scatter returns
		"""
		
		fontsize_labels = None

		nfp = self.input.physics.Nfp
		Igeometry = self.input.physics.Igeometry
		
		zeta = zetan * (2*np.pi/nfp)
		
		if(Igeometry == 3):
			zeta = zeta - (2*np.pi/nfp) * int(zeta/(2*np.pi/nfp)) # move zeta into one field period

			num_tor_planes = self.poincare.R.shape[2]
			zetas = np.linspace(0, 2 * np.pi/nfp, num_tor_planes, endpoint=False)
			print(f"Normalized zetas: {zetas/(2*np.pi/nfp)}")
			toroidalIdx = np.argmin(np.abs(zetas - zeta))
			if abs(zetas[0]+6.28/nfp-zeta) < abs(zetas[toroidalIdx]-zeta):
				toroidalIdx = 0

			rr = self.poincare.R[:, :, toroidalIdx]
			zz = self.poincare.Z[:, :, toroidalIdx]
		elif(Igeometry == 2):

			num_tor_planes = self.poincare.R.shape[2]
			zetas = np.linspace(0, 2 * np.pi, num_tor_planes, endpoint=False)
			toroidalIdx = np.argmin(np.abs(zetas - zeta))

			rr = self.poincare.R[:, :, toroidalIdx] * np.cos(
				self.poincare.t[:, :, toroidalIdx]
			)
			zz = self.poincare.R[:, :, toroidalIdx] * np.sin(
				self.poincare.t[:, :, toroidalIdx]
			)
		elif(Igeometry == 1):

			num_tor_planes = self.poincare.R.shape[2]
			zetas = np.linspace(0, 2 * np.pi, num_tor_planes, endpoint=False)
			toroidalIdx = np.argmin(np.abs(zetas - zeta))

			rr = np.mod(self.poincare.t[:, :, toroidalIdx], np.pi * 2)
			zz = self.poincare.R[:, :, toroidalIdx]

		if ax is None:
			_, ax = plt.subplots()
		else:
			plt.sca(ax)
			_ = plt.gcf()

		# if kwargs.get("marker") is None:
		#     kwargs.update({"marker": "."})
		# use gray color
		if kwargs.get("c") is None:
			kwargs.update({"c": "black"})
		# size of marker
		if kwargs.get("ms") is None:
			kwargs.update({"ms": 0.8})
			
		# kwargs.update({"alpha": 1.0})
		kwargs.update({"lw": None})
		kwargs.update({"markeredgewidth": 0})

		# Remove the failed point w/ R=0
		if Igeometry==3:
			row_indices, col_indices = np.where(rr**2 < 1e-16)
			if len(row_indices) > 1 and np.all(row_indices==row_indices[0]):
				rr[row_indices] = None

		zz[rr<1e-4] = None
		rr[rr<1e-4] = None

		if prange == "full":
			dots = ax.plot(rr, zz, 'o', **kwargs)
		elif prange == "upper":
			dots = ax.plot(rr[zz >= 0], zz[zz >= 0], '.', **kwargs)
		elif prange == "lower":
			dots = ax.plot(rr[zz <= 0], zz[zz <= 0], '.', **kwargs)
		else:
			raise ValueError("prange should be one of ['full'(default), 'upper', 'lower'].")
		
		if self.input.physics.Igeometry == 3:
			plt.xlabel("R [m]", fontsize=fontsize_labels)
			plt.ylabel("Z [m]", fontsize=fontsize_labels)
		if self.input.physics.Igeometry == 2:
			plt.xlabel("X [m]", fontsize=fontsize_labels)
			plt.ylabel("Y [m]", fontsize=fontsize_labels)
		if self.input.physics.Igeometry == 1:
			plt.ylabel("Y [m]", fontsize=fontsize_labels)
			plt.xlabel("X [m]", fontsize=fontsize_labels)
			plt.xlim([0, 2*np.pi])
		
		return dots

	def plot_pressure(self, normalize=True, xaxis=None, tarr=None, zarr=None, ax=None, scalar=1.0, **kwargs):
		"""Plot stepped pressure profile
		Args:
			normalize (bool, optional): Normalize the pressure with mu_0. Defaults to True.
			ax (Matplotlib axis, optional): Matplotlib axis to be plotted on. Defaults to None.
			kwargs (dict, optional): Keyword arguments. Matplotlib.pyplot.plot keyword arguments.
		"""
		
		nvol = self.input.physics.Nvol
		
		pressure = self.input.physics.pressure * self.input.physics.pscale
		if not normalize:
			pressure /= 4 * np.pi * 1.0e-7
		
		xaxis = "tflux" if xaxis is None else xaxis
		
		if kwargs.get("lw") is None:
				kwargs.update({"lw": 2.0})  # prefer thicker lines
		if kwargs.get("ms") is None:
			kwargs.update({"ms": 4})  # default label
		
		if(ax is None):
			_, ax = plt.subplots()
		
		if(xaxis == "tflux"):
			tflux = self.output.tflux[: len(pressure)]
			
			_tflux = np.insert(tflux, 0, 0)
			_pressure = np.append(pressure, 0)
			x_tflux = np.zeros(2 * len(tflux) + 1)
			x_tflux[0::2] = _tflux
			x_tflux[1::2] = tflux
			y_pressure = np.zeros(2 * len(pressure) + 1)
			y_pressure[0::2] = _pressure
			y_pressure[1::2] = pressure
			
			ax.plot(x_tflux[:-1], y_pressure[:-1]*scalar, **kwargs)

			ax.set_xlabel("Normalized flux")
			ax.set_ylabel("Pressure")
		
		elif(xaxis == "x"):
			tarr = np.array([3.0]) if tarr is None else tarr
			zarr = np.array([0.0]) if zarr is None else zarr

			for lvol in range(nvol):
				sarr = np.linspace(-1,1,2) if lvol==nvol//2 else np.array([-1., 0.0, 1.])
				R, _ = self.get_coord_transform(lvol, sarr, tarr, zarr)
				R = R[0,:,0,0]
				ax.plot(R, pressure[lvol]*np.ones_like(R)*scalar, 'r-', label="SPECTRE" if lvol==0 else None, **kwargs)       
		
			ax.set_xlabel("x")
			ax.set_ylabel("Pressure")

	def trace_fieldlines(self, init_pts, num_phi_planes=32, num_transits=1, num_threads=1, batch_size=1, method='RK45', rtol=1.0e-7, atol=1.0e-7):
		"""Generate Poincare using custom field line tracing in Python. Sets the self.tracer to the generated Field_Tracer object.

		Args:
			init_pts (List[mvol]): list of ndarray[num_trajectories, 2] with initial (s, theta) points for tracing
			num_phi_planes (int, optional): Number of phi planes to record the poincare. Defaults to 32.
			num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
			num_threads (int, optional): Number of threads for multithreaded integration. Defaults to 1.
			batch_size (int, optional): Batch size for multithreaded integration. Defaults to 1.
			method (str, optional): Method for integrating the ode. Defaults to 'RK45'. Options: 'RK23', 'RK45', 'DOP853'
			rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
			atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.
		"""
		
		mvol = self.input.physics.Mvol    
		igeometry = self.input.physics.Igeometry
		mpol = self.input.physics.Mpol
		im = self.output.im
		in_ = self.output.in_
		nfp = self.input.physics.Nfp
		lrad_arr = self.input.physics.Lrad

		Aze = [np.atleast_2d(self.vector_potential.Aze[ivol]) for ivol in range(mvol)]
		Ate = [np.atleast_2d(self.vector_potential.Ate[ivol]) for ivol in range(mvol)]
		Azo = [np.atleast_2d(self.vector_potential.Azo[ivol]) for ivol in range(mvol)]
		Ato = [np.atleast_2d(self.vector_potential.Ato[ivol]) for ivol in range(mvol)]

		Rbc = self.output.Rbc
		Zbs = self.output.Zbs
		Rbs = self.output.Rbs
		Zbc = self.output.Zbc

		tracer = Field_Tracer(mvol, igeometry, nfp, mpol, im, in_, Ate, Aze, Ato, Azo, Rbs, Rbc, Zbs, Zbc, lrad_arr, 
			num_threads=num_threads, batch_size=batch_size, method=method, rtol=rtol, atol=atol)

		tracer.trace_field(init_pts, num_phi_planes=num_phi_planes, num_transits=num_transits)

		self._tracer = tracer

	def trace_one_fieldline(self, sth_init, ivol, num_phi_planes=32, num_transits=1, method='RK45', rtol=1.0e-7, atol=1.0e-7):
		"""Generate field line trace for a single point in Python. Sets the self.tracer to the generated Field_Tracer object, for the first call.

		Args:
			sth_init (List[float]): initial (s, theta) of the point.
			ivol (int): Index of the volume where tracing is done.
			num_phi_planes (int, optional): Number of phi planes to record the poincare. Defaults to 32.
			num_transits (int, optional): Number of toroidal transits to perform. Defaults to 1.
			method (str, optional): Method for integrating the ode. Defaults to 'RK45'. Options: 'RK23', 'RK45', 'DOP853'
			rtol (float, optional): Relative tolerance of the integrator. Defaults to 1.0e-6.
			atol (float, optional): Absolute tolerance of the integrator. Defaults to 1.0e-6.

		Returns:
			List[float]: Final (s, theta) coordinates.
			List[float]: Final (R, Z) coordinates.
            float: Final iota.
			bool: True if integration did not reach the end, False otherwise.
			
		"""
		
		if not hasattr(self, '_tracer_single'):
			mvol = self.input.physics.Mvol    
			igeometry = self.input.physics.Igeometry
			mpol = self.input.physics.Mpol
			im = self.output.im
			in_ = self.output.in_
			nfp = self.input.physics.Nfp
			lrad_arr = self.input.physics.Lrad

			Aze = [np.atleast_2d(self.vector_potential.Aze[ivol]) for ivol in range(mvol)]
			Ate = [np.atleast_2d(self.vector_potential.Ate[ivol]) for ivol in range(mvol)]
			Azo = [np.atleast_2d(self.vector_potential.Azo[ivol]) for ivol in range(mvol)]
			Ato = [np.atleast_2d(self.vector_potential.Ato[ivol]) for ivol in range(mvol)]

			Rbc = self.output.Rbc
			Zbs = self.output.Zbs
			Rbs = self.output.Rbs
			Zbc = self.output.Zbc

			tracer = Field_Tracer(mvol, igeometry, nfp, mpol, im, in_, Ate, Aze, Ato, Azo, Rbs, Rbc, Zbs, Zbc, lrad_arr, method=method, rtol=rtol, atol=atol)

			self._tracer_single = tracer

		sth, RZ, iota, flag = self._tracer_single.trace_field_single(sth_init, ivol, num_phi_planes=num_phi_planes, num_transits=num_transits)

		return sth, RZ, iota, flag

	@property
	def tracer(self):
		if not hasattr(self, '_tracer'):
			raise AttributeError("Field tracing has not been done, run 'trace_fieldlines()' first")
		return self._tracer