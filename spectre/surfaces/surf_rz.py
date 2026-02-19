# Fourier-RZ surface representation. 
# Current implementation borrowed from simsopt (https://github.com/hiddenSymmetries/simsopt/tree/master/src/simsopt/geo/surfacerzfourier.py).
# Modified here for use in spectre interface representation.

import numpy as np
from spectre.surfaces.surf import Surface

class SurfaceRZFourier(Surface):
    r"""
    ``SurfaceRZFourier`` is a surface that is represented in
    cylindrical coordinates using the following Fourier series:

    .. math::
           r(\theta, \phi) = \sum_{m=0}^{m_{\text{pol}}}
               \sum_{n=-n_{\text{tor}}}^{n_\text{tor}} [
               r_{c,m,n} \cos(m \theta - n_{\text{fp}} n \phi)
               + r_{s,m,n} \sin(m \theta - n_{\text{fp}} n \phi) ]

    and the same for :math:`z(\theta, \phi)`.

    Here, :math:`(r,\phi, z)` are standard cylindrical coordinates, and theta
    is any poloidal angle.

    Note that for :math:`m=0` we skip the :math:`n<0` term for the cos terms,
    and the :math:`n \leq 0` for the sin terms.

    In addition, in the ``stellsym=True`` case, we skip the sin terms for
    :math:`r`, and the cos terms for :math:`z`.

    Args:
        nfp: The number of field periods.
        stellsym: Whether the surface is stellarator-symmetric, i.e.
          symmetry under rotation by :math:`\pi` about the x-axis.
        mpol: Maximum poloidal mode number included.
        ntor: Maximum toroidal mode number included, divided by ``nfp``.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, nfp=1, stellsym=True, mpol=1, ntor=0,
                 quadpoints_phi=None, quadpoints_theta=None,
                 dofs=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym

        self.rc = np.zeros((self.mpol + 1, 2*self.ntor + 1))
        self.zs = np.zeros((self.mpol + 1, 2*self.ntor + 1))
        if not stellsym:
            self.rs = np.zeros((self.mpol + 1, 2*self.ntor + 1))
            self.zc = np.zeros((self.mpol + 1, 2*self.ntor + 1))
        
        self.rc[0, ntor] = 1.0
        self.rc[1, ntor] = 0.1
        self.zs[1, ntor] = 0.1
        if dofs is None:
            Surface.__init__(self, x0=self.get_dofs(),
                             external_dof_setter=SurfaceRZFourier.set_dofs_impl,
                             names=self._make_names())
        else:
            Surface.__init__(self, dofs=dofs,
                             external_dof_setter=SurfaceRZFourier.set_dofs_impl)
        self._make_mn()

    def get_dofs(self):
        """
        Return the dofs associated to this surface.
        """

        ntor = self.ntor
        shift = (self.mpol+1) * (2*self.ntor + 1)

        if self.stellsym:
            dofs = np.concatenate((self.rc.flat[ntor:shift], self.zs.flat[ntor+1:shift]))
        else:
            dofs = np.concatenate((self.rc.flat[ntor:shift], self.rs.flat[ntor+1:shift],
                                   self.zc.flat[ntor:shift], self.zs.flat[ntor+1:shift]))
        
        return dofs    

    def set_dofs(self, dofs):
        self.local_full_x = dofs

    def set_dofs_impl(self, dofs):

        # void set_dofs_impl(const vector<double>& dofs) override {
        #     int shift = (mpol+1)*(2*ntor+1);
        #     int counter = 0;
        #     if(stellsym) {
        #         for (int i = ntor; i < shift; ++i)
        #             rc.data()[i] = dofs[counter++];
        #         for (int i = ntor+1; i < shift; ++i)
        #             zs.data()[i] = dofs[counter++];

        #     } else {
        #         for (int i = ntor; i < shift; ++i)
        #             rc.data()[i] = dofs[counter++];
        #         for (int i = ntor+1; i < shift; ++i)
        #             rs.data()[i] = dofs[counter++];
        #         for (int i = ntor; i < shift; ++i)
        #             zc.data()[i] = dofs[counter++];
        #         for (int i = ntor+1; i < shift; ++i)
        #             zs.data()[i] = dofs[counter++];
        #     }
        # }

        ntor = self.ntor
        shift = (self.mpol+1) * (2*self.ntor + 1)

        if self.stellsym:
            self.rc.flat[ntor:shift] = dofs[:shift-ntor]
            self.zs.flat[ntor+1:shift] = dofs[shift-ntor:]
        else:
            self.rc.flat[ntor:shift] = dofs[:shift-ntor]
            self.rs.flat[ntor+1:shift] = dofs[shift-ntor:2*(shift-ntor)]
            self.zc.flat[ntor:shift] = dofs[2*(shift-ntor):2*(shift-ntor)+shift-ntor]
            self.zs.flat[ntor+1:shift] = dofs[2*(shift-ntor)+shift-ntor:]


    def _make_names(self):
        """
        Form a list of names of the ``rc``, ``zs``, ``rs``, or ``zc``
        array elements.  The order of these four arrays here must
        match the order in ``set_dofs_impl()`` and ``get_dofs()``
        """
        if self.stellsym:
            names = self._make_names_helper('rc', True) + self._make_names_helper('zs', False)
        else:
            names = self._make_names_helper('rc', True) \
                + self._make_names_helper('rs', False) \
                + self._make_names_helper('zc', True) \
                + self._make_names_helper('zs', False)
        return names

    def _make_names_helper(self, prefix, include0):
        if include0:
            names = [prefix + "(0,0)"]
        else:
            names = []

        names += [prefix + '(0,' + str(n) + ')' for n in range(1, self.ntor + 1)]
        for m in range(1, self.mpol + 1):
            names += [prefix + '(' + str(m) + ',' + str(n) + ')' for n in range(-self.ntor, self.ntor + 1)]
        return names

    def _make_mn(self):
        """
        Make the list of m and n values.
        """
        m1d = np.arange(self.mpol + 1)
        n1d = np.arange(-self.ntor, self.ntor + 1)
        n2d, m2d = np.meshgrid(n1d, m1d)
        m0 = m2d.flatten()[self.ntor:]
        n0 = n2d.flatten()[self.ntor:]
        m = np.concatenate((m0, m0[1:]))
        n = np.concatenate((n0, n0[1:]))
        if not self.stellsym:
            m = np.concatenate((m, m))
            n = np.concatenate((n, n))
        self.m = m
        self.n = n


    def copy(self, **kwargs):
        """
        Return a copy of the ``SurfaceRZFourier`` object, but with the specified
        attributes changed. Keyword arguments accepted:

        - ``ntheta``: number of quadrature points in the theta direction
        - ``nphi``: number of quadrature points in the phi direction
        - ``mpol``: number of poloidal Fourier modes for the surface
        - ``ntor``: number of toroidal Fourier modes for the surface
        - ``nfp``: number of field periods
        - ``stellsym``: whether the surface is stellarator-symmetric
        - ``quadpoints_theta``: theta grid points
        - ``quadpoints_phi``: phi grid points

        """
        otherntheta = self.quadpoints_theta.size
        othernphi = self.quadpoints_phi.size

        ntheta = kwargs.pop("ntheta", otherntheta)
        nphi = kwargs.pop("nphi", othernphi)
        grid_range = kwargs.pop("range", None)
        mpol = kwargs.pop("mpol", self.mpol)
        ntor = kwargs.pop("ntor", self.ntor)
        nfp = kwargs.pop("nfp", self.nfp)
        stellsym = kwargs.pop("stellsym", self.stellsym)
        quadpoints_theta = kwargs.pop("quadpoints_theta", None)
        quadpoints_phi = kwargs.pop("quadpoints_phi", None)

        # recalculate the quadpoints if necessary (grid_range is not stored in the
        # surface object, so assume that if it is given, the gridpoints should be
        # recalculated to the specified size)
        if quadpoints_theta is None and quadpoints_phi is None:
            if ntheta is not otherntheta or nphi is not othernphi or grid_range is not None:
                kwargs["quadpoints_phi"], kwargs["quadpoints_theta"] = Surface.get_quadpoints(
                    ntheta=ntheta, nphi=nphi, nfp=self.nfp, range=grid_range)
            else:
                kwargs["quadpoints_phi"] = self.quadpoints_phi
                kwargs["quadpoints_theta"] = self.quadpoints_theta
        else:
            if quadpoints_theta is None:
                if ntheta is not otherntheta or grid_range is not None:
                    kwargs["quadpoints_theta"] = Surface.get_theta_quadpoints(ntheta)
                else:
                    kwargs["quadpoints_theta"] = self.quadpoints_theta
            else:
                kwargs["quadpoints_theta"] = quadpoints_theta
            if quadpoints_phi is None:
                if nphi is not othernphi or grid_range is not None:
                    kwargs["quadpoints_phi"] = Surface.get_phi_quadpoints(nphi, range=grid_range, nfp=nfp)
                else:
                    kwargs["quadpoints_phi"] = self.quadpoints_phi
            else:
                kwargs["quadpoints_phi"] = quadpoints_phi
        # create new surface in old resolution
        surf = SurfaceRZFourier(mpol=self.mpol, ntor=self.ntor, nfp=nfp, stellsym=stellsym,
                                **kwargs)
        surf.rc[:, :] = self.rc
        surf.zs[:, :] = self.zs
        if not self.stellsym:
            surf.rs[:, :] = self.rs
            surf.zc[:, :] = self.zc
        # set to the requested resolution
        surf.change_resolution(mpol, ntor)
        surf.local_full_x = surf.get_dofs()
        return surf

    def change_resolution(self, mpol, ntor):
        """
        Change the values of `mpol` and `ntor`. Any new Fourier amplitudes
        will have a magnitude of zero.  Any previous nonzero Fourier
        amplitudes that are not within the new range will be
        discarded.
        """
        old_mpol = self.mpol
        old_ntor = self.ntor
        old_rc = self.rc
        old_zs = self.zs
        if not self.stellsym:
            old_rs = self.rs
            old_zc = self.zc
        self.mpol = mpol
        self.ntor = ntor
        self.allocate()

        min_mpol = np.min((mpol, old_mpol))
        min_ntor = np.min((ntor, old_ntor))
        for m in range(min_mpol + 1):
            for n in range(-min_ntor, min_ntor + 1):
                self.rc[m, n + ntor] = old_rc[m, n + old_ntor]
                self.zs[m, n + ntor] = old_zs[m, n + old_ntor]
                if not self.stellsym:
                    self.rs[m, n + ntor] = old_rs[m, n + old_ntor]
                    self.zc[m, n + ntor] = old_zc[m, n + old_ntor]
        self._make_mn()

    def to_RZFourier(self):
        """
        No conversion necessary.
        """
        return self

    def __repr__(self):
        return self.name + f" (nfp={self.nfp}, stellsym={self.stellsym}, " + \
            f"mpol={self.mpol}, ntor={self.ntor})"

    def _validate_mn(self, m, n):
        """
        Check whether `m` and `n` are in the allowed range.
        """
        if m < 0:
            raise IndexError('m must be >= 0')
        if m > self.mpol:
            raise IndexError('m must be <= mpol')
        if n > self.ntor:
            raise IndexError('n must be <= ntor')
        if n < -self.ntor:
            raise IndexError('n must be >= -ntor')

    def get_rc(self, m, n):
        """
        Return a particular `rc` Parameter.
        """
        self._validate_mn(m, n)
        return self.rc[m, n + self.ntor]

    def get_rs(self, m, n):
        """
        Return a particular `rs` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.rs[m, n + self.ntor]

    def get_zc(self, m, n):
        """
        Return a particular `zc` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        return self.zc[m, n + self.ntor]

    def get_zs(self, m, n):
        """
        Return a particular `zs` Parameter.
        """
        self._validate_mn(m, n)
        return self.zs[m, n + self.ntor]

    def set_rc(self, m, n, val):
        """
        Set a particular `rc` Parameter.
        """
        self._validate_mn(m, n)
        self.rc[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def set_rs(self, m, n, val):
        """
        Set a particular `rs` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'rs does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.rs[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def set_zc(self, m, n, val):
        """
        Set a particular `zc` Parameter.
        """
        if self.stellsym:
            return ValueError(
                'zc does not exist for this stellarator-symmetric surface.')
        self._validate_mn(m, n)
        self.zc[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def set_zs(self, m, n, val):
        """
        Set a particular `zs` Parameter.
        """
        self._validate_mn(m, n)
        self.zs[m, n + self.ntor] = val
        self.local_full_x = self.get_dofs()

    def fixed_range(self, mmin, mmax, nmin, nmax, fixed=True):
        """
        Set the 'fixed' property for a range of `m` and `n` values.

        All modes with `m` in the interval [`mmin`, `mmax`] and `n` in the
        interval [`nmin`, `nmax`] will have their fixed property set to
        the value of the `fixed` parameter. Note that `mmax` and `nmax`
        are included (unlike the upper bound in python's range(min,
        max).)
        """
        # TODO: This will be slow because free dof indices are evaluated all
        # TODO: the time in the loop
        fn = self.fix if fixed else self.unfix
        for m in range(mmin, mmax + 1):
            this_nmin = nmin
            if m == 0 and nmin < 0:
                this_nmin = 0
            for n in range(this_nmin, nmax + 1):
                fn(f'rc({m},{n})')
                if m > 0 or n != 0:
                    fn(f'zs({m},{n})')
                if not self.stellsym:
                    fn(f'zc({m},{n})')
                    if m > 0 or n != 0:
                        fn(f'rs({m},{n})')

    def darea(self):
        """
        Short hand for `Surface.darea_by_dcoeff()`
        """
        return self.darea_by_dcoeff()

    def dvolume(self):
        """
        Short hand for `Surface.dvolume_by_dcoeff()`
        """
        return self.dvolume_by_dcoeff()

    def get_nml(self):
        """
        Generates a fortran namelist file containing the RBC/RBS/ZBC/ZBS
        coefficients, in the form used in VMEC and SPEC input
        files. The result will be returned as a string. For saving a
        file, see the ``write_nml()`` function.
        """
        nml = ''
        nml += '&INDATA\n'
        if self.stellsym:
            nml += 'LASYM = .FALSE.\n'
        else:
            nml += 'LASYM = .TRUE.\n'
        nml += f'NFP = {self.nfp}\n'

        for m in range(self.mpol + 1):
            nmin = -self.ntor
            if m == 0:
                nmin = 0
            for n in range(nmin, self.ntor + 1):
                rc = self.get_rc(m, n)
                zs = self.get_zs(m, n)
                if np.abs(rc) > 0 or np.abs(zs) > 0:
                    nml += f"RBC({n:4d},{m:4d}) ={rc:23.15e},    ZBS({n:4d},{m:4d}) ={zs:23.15e}\n"
                if (not self.stellsym):
                    rs = self.get_rs(m, n)
                    zc = self.get_zc(m, n)
                    if np.abs(rs) > 0 or np.abs(zc) > 0:
                        nml += f"RBS({n:4d},{m:4d}) ={rs:23.15e},    ZBC({n:4d},{m:4d}) ={zc:23.15e}\n"
        nml += '/\n'
        return nml

    def write_nml(self, filename: str):
        """
        Writes a fortran namelist file containing the RBC/RBS/ZBC/ZBS
        coefficients, in the form used in VMEC and SPEC input
        files. To just generate the namelist as a string without
        saving a file, see the ``get_nml()`` function.

        Args:
            filename: Name of the file to write.
        """
        with open(filename, 'w') as f:
            f.write(self.get_nml())

    def extend_via_normal(self, distance):
        """
        Extend the surface in the normal direction by a uniform distance.

        Args:
            distance: The distance to extend the surface.
        """
        if len(self.quadpoints_phi) < 2 * self.ntor + 1:
            raise RuntimeError("Number of phi quadrature points should be at least 2 * ntor + 1")
        if len(self.quadpoints_theta) < 2 * self.mpol + 1:
            raise RuntimeError("Number of theta quadrature points should be at least 2 * mpol + 1")

        # Generate points that are a uniform distance from the surface, though
        # at irregular phi values:
        points = (self.gamma() + self.unitnormal() * distance).reshape((-1, 3))

        R = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        phi = np.arctan2(points[:, 1], points[:, 0])
        Z = points[:, 2]
        n_phi = len(self.quadpoints_phi)
        n_theta = len(self.quadpoints_theta)
        theta_1d = np.linspace(0, 2 * np.pi, len(self.quadpoints_theta), endpoint=False)
        theta = (theta_1d[None, :] * np.ones((n_phi, n_theta))).flatten()

        # Evaluate the basis functions at the new (phi, theta) points:
        n_cos_dofs = (2 * self.ntor + 1) * self.mpol + self.ntor + 1
        cos_terms = np.cos(self.m[None, :n_cos_dofs] * theta[:, None] - self.nfp * self.n[None, :n_cos_dofs] * phi[:, None])
        sin_terms = np.sin(self.m[None, 1:n_cos_dofs] * theta[:, None] - self.nfp * self.n[None, 1:n_cos_dofs] * phi[:, None])

        if self.stellsym:
            R_basis_funcs = cos_terms
            Z_basis_funcs = sin_terms
        else:
            R_basis_funcs = np.concatenate((cos_terms, sin_terms), axis=1)
            Z_basis_funcs = R_basis_funcs

        # Fit the mode amplitudes to the new points.
        try:
            R_dofs = np.linalg.lstsq(R_basis_funcs, R, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Try a second time
            R_dofs = np.linalg.lstsq(R_basis_funcs, R, rcond=None)[0]

        try:
            Z_dofs = np.linalg.lstsq(Z_basis_funcs, Z, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Try a second time
            Z_dofs = np.linalg.lstsq(Z_basis_funcs, Z, rcond=None)[0]

        self.x = np.concatenate((R_dofs, Z_dofs))

    def fourier_transform_scalar(self, scalar, mpol=None, ntor=None, normalization=None, **kwargs):
        r"""
        Compute the Fourier components of a scalar on the surface. The scalar
        is evaluated at the quadrature points on the surface. 
        The Fourier uses the conventions of the ``SurfaceRZFourier`` series, 
        with ``npol`` going from ``-ntor`` to ``ntor`` and ``mpol`` from 0 to ``mpol``
        i.e.: 

        .. math::
            f(\theta, \phi) = \sum_{m=0}^{mpol} \sum_{n=-npol}^{npol} A^{mn}_s \sin(m\theta - n N_{fp} \phi)
            + A^{mn}_c \cos(m\theta - n N_{fp} \phi)

        Where the cosine series is only evaluated if the surface is not stellarator
        symmetric (if the scalar does not adhere to the symmetry of the surface, 
        request the cosine series by setting the kwarg ``stellsym=False``)
        By default, the poloidal and toroidal resolution are the same as those
        of the surface, but different quantities can be specified in the kwargs. 

        Args:
            scalar: 2D array of shape ``(numquadpoints_phi, numquadpoints_theta)``.
            mpol: maximum poloidal mode number of the transform, if ``None``,
                the mpol attribute of the surface is used.
            ntor: maximum toroidal mode number of the transform if ``None``, 
                the ntor attribute of the surface is used.
            normalization: (optional) Fourier transform normalization. Can be: 
              ``None``: forward and back transform are not normalized.
              ``float``: forward transform is divided by this number.
            stellsym: (optional) boolean to override the stellsym attribute 
                of the surface if you want to force the calculation of the
                cosine series

        Returns:
            2-element tuple ``(A_mns, A_mnc)``, where ``A_mns`` is a 2D array of shape ``(mpol+1, 2*ntor+1)`` containing the sine
            coefficients, and ``A_mnc`` is a  2D array of shape ``(mpol+1, 2*ntor+1)`` containing the cosine coefficients 
            (these are zero if the surface is stellarator symmetric).
        """
        assert scalar.shape[0] == self.quadpoints_phi.size, "scalar must be evaluated at the quadrature points on the surface.\n the scalar you passed in has shape {}".format(scalar.shape)
        assert scalar.shape[1] == self.quadpoints_theta.size, "scalar must be evaluated at the quadrature points on the surface.\n the scalar you passed in has shape {}".format(scalar.shape)
        stellsym = kwargs.pop('stellsym', self.stellsym)
        if mpol is None:
            try:
                mpol = self.mpol
            except AttributeError:
                raise ValueError("mpol must be specified")
        if ntor is None:
            try:
                ntor = self.ntor
            except AttributeError:
                raise ValueError("ntor must be specified")
        A_mns = np.zeros((int(mpol + 1), int(2 * ntor + 1)))  # sine coefficients
        A_mnc = np.zeros((int(mpol + 1), int(2 * ntor + 1)))  # cosine coefficients
        ntheta_grid = len(self.quadpoints_theta)
        nphi_grid = len(self.quadpoints_phi)

        factor = 2.0 / (ntheta_grid * nphi_grid)

        phi2d, theta2d = np.meshgrid(2 * np.pi * self.quadpoints_phi,
                                     2 * np.pi * self.quadpoints_theta,
                                     indexing="ij")

        for m in range(mpol + 1):
            nmin = -ntor
            if m == 0:
                nmin = 1
            for n in range(nmin, ntor+1):
                angle = m * theta2d - n * self.nfp * phi2d
                sinangle = np.sin(angle)
                factor2 = factor
                # The next 2 lines ensure inverse Fourier transform(Fourier transform) = identity
                if np.mod(ntheta_grid, 2) == 0 and m == (ntheta_grid/2):
                    factor2 = factor2 / 2
                if np.mod(nphi_grid, 2) == 0 and abs(n) == (nphi_grid/2):
                    factor2 = factor2 / 2
                A_mns[m, n + ntor] = np.sum(scalar * sinangle * factor2)
                if not stellsym:
                    cosangle = np.cos(angle)
                    A_mnc[m, n + ntor] = np.sum(scalar * cosangle * factor2)

        if not stellsym:
            A_mnc[0, ntor] = np.sum(scalar) / (ntheta_grid * nphi_grid)
        if normalization is not None:
            if not isinstance(normalization, float):
                raise ValueError("normalization must be a float")
            A_mns = A_mns / normalization
            A_mnc = A_mnc / normalization

        return A_mns, A_mnc

    def inverse_fourier_transform_scalar(self, A_mns, A_mnc, normalization=None, **kwargs):
        r"""
        Compute the inverse Fourier transform of a scalar on the surface, specified by the Fourier coefficients. The quantity must be
        is evaluated at the quadrature points on the surface. The Fourier
        transform is defined as
        :math:`f(\theta, \phi) = \Sum_{m=0}^{mpol} \Sum_{n=-npol}^{npol} A^{mn}_s \sin(m\theta - n*Nfp*\phi) + A^{mn}_c \cos(m\theta - n*Nfp*\phi)`
        Where the cosine series is only evaluated if the surface is not stellarator symmetric.
        *Arguments*:

        - A_mns: 2D array of shape (mpol+1, 2*ntor+1) containing the sine coefficients
        - A_mnc: 2D array of shape (mpol+1, 2*ntor+1) containing the cosine coefficients 
            (these are zero if the surface is stellarator symmetric)

        *Optional keyword arguments*:

        - normalization: Fourier transform normalization. Can be:
            None: forward and back transform are not normalized
            float: inverse transform is multiplied by this number
        - stellsym: boolean to override the stellsym attribute of the surface

        """
        mpol = A_mns.shape[0] - 1
        ntor = int((A_mns.shape[1] - 1) / 2)
        stellsym = kwargs.pop('stellsym', self.stellsym)
        ntheta_grid = len(self.quadpoints_theta)
        nphi_grid = len(self.quadpoints_phi)

        phi2d, theta2d = np.meshgrid(2 * np.pi * self.quadpoints_phi,
                                     2 * np.pi * self.quadpoints_theta,
                                     indexing="ij")

        scalars = np.zeros((nphi_grid, ntheta_grid))
        for m in range(mpol + 1):
            nmin = -ntor
            if m == 0:
                nmin = 1
            for n in range(nmin, ntor+1):
                angle = m * theta2d - n * self.nfp * phi2d
                sinangle = np.sin(angle)
                scalars = scalars + A_mns[m, n + ntor] * sinangle
                if not stellsym:
                    cosangle = np.cos(angle)
                    scalars = scalars + A_mnc[m, n + ntor] * cosangle

        if not stellsym:
            scalars = scalars + A_mnc[0, ntor]
        if normalization is not None:
            if not isinstance(normalization, float):
                raise ValueError("normalization must be a float")
            scalars = scalars * normalization
        return scalars

    def make_rotating_ellipse(self, major_radius, minor_radius, elongation, torsion=0):
        r"""
        Set the surface shape to be a rotating ellipse with the given
        parameters.

        Values of ``elongation`` larger than 1 will result in the elliptical
        cross-section at :math:`\phi=0` being taller than it is wide.
        Values of ``elongation`` less than 1 will result in the elliptical
        cross-section at :math:`\phi=0` being wider than it is tall.

        The sign convention is such that both the rotating elongation and
        positive ``torsion`` will contribute positively to iota according to
        VMEC's sign convention.

        Args:
            major_radius: Average major radius of the surface.
            minor_radius: Average minor radius of the surface.
            elongation: Elongation of the elliptical cross-section.
            torsion: Value to use for the (m,n)=(0,1) mode of RC and -ZS, which
                controls the torsion of the magnetic axis.
        """

        self.local_full_x = np.zeros_like(self.local_full_x)
        self.set_rc(0, 0, major_radius)
        self.set_rc(0, 1, torsion)
        self.set_zs(0, 1, -torsion)

        sqrt_elong = np.sqrt(elongation)
        amplitude = 0.5 * minor_radius * (1 / sqrt_elong - sqrt_elong)
        self.set_rc(1, 1, amplitude)
        self.set_zs(1, 1, -amplitude)

        amplitude = 0.5 * minor_radius * (1 / sqrt_elong + sqrt_elong)
        self.set_rc(1, 0, amplitude)
        self.set_zs(1, 0, amplitude)