# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import contextlib
import keyword
import os.path
import types

import h5py
import matplotlib.pyplot as plt
import numpy as np


class SPECTREout:
    """
    Class for analyzing SPECTRE output, can be created using an output .h5 file or a SPECTRE object.
    Allows for analysis of the geometry and the Beltrami field(s) for this geometry.

    The analysis scripts are in pure Python and based on Numba for speed.
    """

    from ._check_isect import check_intersection_interfaces
    from ._get_field import (
        calc_avg_beta_vol,
        get_cheby_basis_single,
        get_field_contrav,
        get_field_cylindrical,
        get_field_mod,
    )
    from ._get_geometry import (
        calc_flux_surface_average,
        calc_interfaces_area,
        find_which_volume,
        get_coord_transform,
        get_coord_transform_list,
        get_infaces_dAvec,
        get_infaces_norm,
        get_jacobian,
        get_metric,
        get_rz_coords,
        get_st_coords,
    )

    def __init__(self, filename=None, content=None):
        """
        Initialize a SPECTREout object from a .h5 SPECTRE output file.

        Args:
            filename (str, optional): path to the .h5 file. Defaults to None.
            content (h5py.Group, optional): h5py group containing the SPECTRE output.

        Raises:
            RuntimeError: if the input file is not a .h5 file

        Returns:
            None
        """
        _content = None
        if content is None:
            filename = str(filename)
            if filename[-3:] != ".h5":
                raise RuntimeError(f"Input file ({filename}) must be a .h5 file")
            _content = h5py.File(filename, "r")
            self.filename = os.path.abspath(filename)

        elif isinstance(content, h5py.Group):
            _content = content

        if _content is not None:
            for key in _content:
                if isinstance(_content[key], h5py.Group):
                    # recurse into group
                    if key in keyword.kwlist:  # avoid assign python keywords
                        setattr(self, key + "1", SPECTREout(content=_content[key]))
                    else:
                        setattr(self, key, SPECTREout(content=_content[key]))
                elif isinstance(_content[key], h5py.Dataset):  # read dataset
                    if key in keyword.kwlist:  # add underscore avoiding assigning python keywords
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
            self._postprocess()

        return

    def _postprocess(self):
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
            cAte.append(np.atleast_2d(self.vector_potential.Ate)[:, start : start + Lrad[i] + 1])
            cAto.append(np.atleast_2d(self.vector_potential.Ato)[:, start : start + Lrad[i] + 1])
            cAze.append(np.atleast_2d(self.vector_potential.Aze)[:, start : start + Lrad[i] + 1])
            cAzo.append(np.atleast_2d(self.vector_potential.Azo)[:, start : start + Lrad[i] + 1])

            # grid
            if hasattr(self.grid, "Rij"):
                cRij.append(np.atleast_2d(self.grid.Rij)[:, start : start + Lrad[i] + 1])
            if hasattr(self.grid, "Zij"):
                cZij.append(np.atleast_2d(self.grid.Zij)[:, start : start + Lrad[i] + 1])
            if hasattr(self.grid, "sg"):
                csg.append(np.atleast_2d(self.grid.sg)[:, start : start + Lrad[i] + 1])
            if hasattr(self.grid, "BR"):
                cBR.append(np.atleast_2d(self.grid.BR)[:, start : start + Lrad[i] + 1])
            if hasattr(self.grid, "Bp"):
                cBp.append(np.atleast_2d(self.grid.Bp)[:, start : start + Lrad[i] + 1])
            if hasattr(self.grid, "BZ"):
                cBZ.append(np.atleast_2d(self.grid.BZ)[:, start : start + Lrad[i] + 1])

            # move along the merged array dimension
            start = start + Lrad[i] + 1

        # replace original content in data structure
        self.vector_potential.Ate = cAte
        self.vector_potential.Ato = cAto
        self.vector_potential.Aze = cAze
        self.vector_potential.Azo = cAzo

        if hasattr(self.grid, "Rij"):
            self.grid.Rij = cRij
        if hasattr(self.grid, "Zij"):
            self.grid.Zij = cZij
        if hasattr(self.grid, "sg"):
            self.grid.sg = csg
        if hasattr(self.grid, "BR"):
            self.grid.BR = cBR
        if hasattr(self.grid, "Bp"):
            self.grid.Bp = cBp
        if hasattr(self.grid, "BZ"):
            self.grid.BZ = cBZ

        if hasattr(self, "poincare"):
            # remove unsuccessful Poincare trajectories
            self.poincare.R = self.poincare.R[self.poincare.success == 1, :, :]
            self.poincare.Z = self.poincare.Z[self.poincare.success == 1, :, :]
            self.poincare.t = self.poincare.t[self.poincare.success == 1, :, :]
            self.poincare.s = self.poincare.s[self.poincare.success == 1, :, :]

    @classmethod
    def from_spectre(cls, spectre):
        """Initialize a SPECTREout object from a SPECTRE object.

        Done by reading the data from the SPECTRE object and putting it into the same structure as
        if it had been read from a .h5 file. This allows for using the
        same analysis tools on both .h5 files and SPECTRE objects.

        For analysis related to the magnetic field, requires the SPECTRE object to have
        the field information from either a run() or a force_real().

        Args:
            spectre (:class:`~spectre.SPECTRE`): SPECTRE object to be converted to SPECTREout.

        Returns:
            SPECTREout: SPECTREout object with the data from the SPECTRE object.
        """
        from spectre.utils import get_vec_pot

        self = object.__new__(cls)

        lib = spectre.input_list_mod
        glob = spectre.allglobal_mod

        nvol = int(lib.nvol)
        lfreebound = int(lib.lfreebound)
        mvol = nvol + 1 if lfreebound else nvol

        physics = types.SimpleNamespace()
        physics.Lrad = np.array(lib.lrad[:mvol])
        physics.Mpol = int(lib.mpol)
        physics.Igeometry = int(lib.igeometry)
        physics.Nvol = nvol
        physics.Lfreebound = lfreebound
        physics.Mvol = mvol
        physics.Nfp = int(lib.nfp)
        physics.rpol = float(lib.rpol)
        physics.rtor = float(lib.rtor)
        physics.pscale = float(lib.pscale)
        physics.pressure = np.array(lib.pressure[:nvol])
        with contextlib.suppress(AttributeError):
            physics.enforce_stell_sym = bool(lib.enforce_stell_sym)

        input_ = types.SimpleNamespace()
        input_.physics = physics
        self.input = input_

        output = types.SimpleNamespace()
        output.im = np.array(glob.im)
        output.in_ = np.array(glob.in_)
        output.Rbc = np.array(glob.irbc.T)
        output.Zbs = np.array(glob.izbs.T)
        output.Rbs = np.array(glob.irbs.T)
        output.Zbc = np.array(glob.izbc.T)
        self.output = output

        # get_vec_pot returns per-volume lists with shape (mn, lrad+1) each,
        # matching what _postprocess produces from the flat HDF5 arrays
        ate_list, aze_list, ato_list, azo_list = get_vec_pot(spectre)

        vp = types.SimpleNamespace()
        vp.Ate = ate_list
        vp.Aze = aze_list
        vp.Ato = ato_list
        vp.Azo = azo_list
        self.vector_potential = vp

        self.filename = "spectre_" + spectre.name

        return self

    # needed for iterating over the contents of the file
    def __iter__(self):
        return iter(self.__dict__)

    def __next__(self):
        return next(self.__dict__)

    # needed for using SPECTREout with 'with' statement
    def __enter__(self):
        return self

    def __exit__(self, t, v, tb):
        return

    # print a list of items contained in this object
    def inventory(self, prefix=""):
        """Recursively print all attributes, descending into nested :class:`SPECTREout` groups.

        Each item is printed as a ``/``-separated path mirroring the HDF5 hierarchy
        (e.g. ``input/physics/Nvol``).

        Args:
            prefix (str, optional): Path prefix prepended to each item name. Defaults to ``""``.
        """
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
        """Iota plots.

        Args:
            xaxis (str, optional): Choice of x-axis, ``'R'`` (default) or ``'s'``.
            yaxis (str, optional): Choice of y-axis, ``'i'`` (default) or ``'q'``.
            reverse_y_axis (bool, optional): Reverse the y-axis. Defaults to False.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            plot (bool, optional): Whether to display the plot. Defaults to True.
            **kwargs: Additional keyword arguments passed to :func:`matplotlib.pyplot.scatter`.

        Raises:
            ValueError: If ``xaxis`` or ``yaxis`` is not a valid option.

        Returns:
            tuple:
                - **xdata** (*numpy.ndarray*): x-axis values (coordinate ``R`` or flux label ``s``).
                - **ydata** (*numpy.ndarray*): y-axis values (rotational transform
                  :math:`\\iota` or safety factor ``q``).
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

        if plot is False:
            return xdata, ydata

        if ax is None:
            _, ax = plt.subplots()

        ax.plot(xdata, ydata, **kwargs)

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        return xdata, ydata

    def plot_jacobian(
        self,
        zetan=0.0,
        ax=None,
        **kwargs,
    ):
        """Plot the coordinate Jacobian across all volumes for a given toroidal cross-section.

        The Jacobian is shown as a color map on the ``(R, Z)`` plane, with interface
        contours overlaid. The plot title reports the global maximum and minimum values.

        Args:
            zetan (float, optional): Toroidal angle in units of :math:`2\\pi/N_{fp}`. Defaults to 0.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. A new figure is created if None.
                Defaults to None.
            **kwargs: Additional keyword arguments (currently unused).
        """

        nfp = self.input.physics.Nfp
        zeta = zetan * (2 * np.pi / nfp)

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))
        else:
            plt.sca(ax)

        sarr = np.linspace(-0.999, 1, 48)
        tarr = np.linspace(0, 2 * np.pi, 148)
        jacs = []
        for lvol in range(self.input.physics.Nvol):
            jac = self.get_jacobian(lvol, sarr, tarr, zeta)[:, :, 0]
            jacs.append(jac)

            rarr, zarr = self.get_coord_transform(lvol, sarr, tarr, zeta)
            rarr = rarr[0, :, :, 0]
            zarr = zarr[0, :, :, 0]

            ax.plot(rarr[-1], zarr[-1], "k-", lw=1.0)
            im = ax.pcolormesh(rarr, zarr, jac, shading="gouraud")

        plt.colorbar(im, ax=ax)

        ax.axis("equal")
        jac_total = np.array(jacs)
        plt.title(
            f"Jacobian($\\zeta$={zeta:.2f}):  max {np.max(jac_total):.4e}"
            f"   min {np.min(jac_total):.4e}"
        )

    def plot_infaces(
        self,
        surfaces=None,
        num_theta=400,
        zetan=0.0,
        ax=None,
        shift=0.0,
        color="r",
        prange="full",
        plot_axis=False,
        plot_comp_bndry=True,
        **kwargs,
    ):
        """Plot the interfaces and boundary of a SPECTRE equilibrium.

        Args:
            surfaces (array-like, optional): Indices of surfaces to plot. Defaults to all volumes.
            num_theta (int, optional): Number of poloidal points used to trace each surface.
                Defaults to 400.
            zetan (float, optional): Toroidal angle in units of :math:`2\\pi/N_{fp}`. Defaults to 0.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            shift (float, optional): Radial shift applied to plotted coordinates. Defaults to 0.
            color (str, optional): Color of the plotted surfaces. Defaults to ``'r'``.
            prange (str, optional): Poloidal range to plot, one of ``'full'``, ``'upper'``, or
                ``'lower'``. Defaults to ``'full'``.
            plot_axis (bool, optional): Whether to mark the coordinate axis (geometry 3 only).
                Defaults to False.
            plot_comp_bndry (bool, optional): Whether to plot the computational boundary when
                ``Lfreebound`` is set. Defaults to True.
            **kwargs: Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`.

        Raises:
            ValueError: If ``prange`` is not one of ``'full'``, ``'upper'``, or ``'lower'``.
            ValueError: If geometry is not 1, 2, or 3.
        """

        fontsize_labels = 13

        if ax is None:
            fig, ax = plt.subplots()
            newaxis = True
        else:
            fig = plt.gcf()
            newaxis = False

        thetas = np.linspace(0, 2 * np.pi, num_theta)
        figsize = 6.0

        geometry = self.input.physics.Igeometry
        nvol = self.input.physics.Nvol
        nfp = self.input.physics.Nfp

        plot_comp_bndry = plot_comp_bndry and self.input.physics.Lfreebound

        if surfaces is None:
            surfaces = np.arange(nvol)

        if "label" not in kwargs:
            kwargs["label"] = None

        zeta = zetan * (2 * np.pi / nfp)

        if geometry == 1:
            for s in surfaces:
                rarr, _ = self.get_coord_transform(s, 1.0, thetas, zeta)
                r = rarr[0, 0, :, 0]

                ax.plot(thetas, r, color=color, **kwargs)

                if kwargs["label"] is not None:
                    kwargs["label"] = None

            ax.set_xlabel("R [m]", fontsize=fontsize_labels)
            ax.set_ylabel("Z [m]", fontsize=fontsize_labels)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_ylim(0, 2 * np.pi)

            if newaxis:
                fig.set_size_inches(figsize, figsize)

        elif geometry == 2:
            for s in surfaces:
                rarr, _ = self.get_coord_transform(s, 1.0, thetas, zeta)
                r = rarr[0, 0, :, 0]

                x_data = r * np.cos(thetas) + shift
                y_data = r * np.sin(thetas)

                if prange == "full":
                    pass
                elif prange == "upper":
                    mask = y_data > 0.0
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                elif prange == "lower":
                    mask = y_data < 0.0
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                else:
                    raise ValueError("Bad prange input")

                if s == (nvol - 1):
                    ax.plot(x_data, y_data, color="k", **kwargs)
                    continue

                if s == (nvol - 1):
                    ax.plot(x_data, y_data, color="k", **kwargs)
                    continue

                ax.plot(x_data, y_data, color=color, **kwargs)

                if kwargs["label"] is not None:
                    kwargs["label"] = None

            ax.set_xlabel("x [m]", fontsize=fontsize_labels)
            ax.set_ylabel("y [m]", fontsize=fontsize_labels)

            ax.axis("equal")

        elif geometry == 3:
            for s in surfaces:
                rarr, zarr = self.get_coord_transform(s, 1.0, thetas, zeta)
                r = rarr[0, 0, :, 0]
                z = zarr[0, 0, :, 0]

                x_data = r + shift
                y_data = z

                if prange == "full":
                    pass
                elif prange == "upper":
                    mask = y_data > 0.0
                    x_data = x_data[mask][1:]
                    y_data = y_data[mask][1:]
                elif prange == "lower":
                    mask = y_data < 1e-5
                    x_data = x_data[mask][:-1]
                    y_data = y_data[mask][:-1]
                else:
                    raise ValueError("Bad prange input")

                if s == (nvol - 1):
                    ax.plot(x_data, y_data, color=color, **kwargs)
                    continue

                ax.plot(x_data, y_data, color=color, **kwargs)

                if kwargs["label"] is not None:
                    kwargs["label"] = None

            if plot_axis:
                rarr_axis, zarr_axis = self.get_coord_transform(0, -1.0, 0.0, zeta)
                ax.plot(rarr_axis[0, 0, :, 0], zarr_axis[0, 0, :, 0], "+", ms=7, c=color)

            if plot_comp_bndry:
                rarr, zarr = self.get_coord_transform(nvol, 1.0, thetas, zeta)
                r = rarr[0, 0, :, 0]
                z = zarr[0, 0, :, 0]
                x_data = r + shift
                y_data = z

                if prange == "full":
                    pass
                elif prange == "upper":
                    mask = y_data > 0.0
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                elif prange == "lower":
                    mask = y_data < 0.0
                    x_data = x_data[mask]
                    y_data = y_data[mask]
                else:
                    raise ValueError("Bad prange input")

                kwargs["ls"] = "--"
                ax.plot(x_data, y_data, color="g", **kwargs)

            if prange != "full":
                ax.axhline(y=0, xmin=0.02, xmax=0.98, c="k", lw=1)

            ax.set_xlabel("R [m]", fontsize=fontsize_labels)
            ax.set_ylabel("Z [m]", fontsize=fontsize_labels)

            ax.axis("equal")
            if newaxis:
                aspect_ratio = (z.max() - z.min()) / (r.max() - r.min())
                fig.set_size_inches(figsize, figsize * aspect_ratio)

        else:
            raise ValueError(f"Invalid geometry {geometry}")

    def plot_infaces3D(self):
        raise NotImplementedError("3D interface plotting not implemented yet")

    def plot_field_mod(
        self,
        ivol=0,
        sarr=None,
        tarr=None,
        zarr=0.0,
        ax=None,
        colorbar=True,
        **kwargs,
    ):
        """Plot the magnetic field modulus :math:`|\\vec{B}|` in a volume at toroidal cross-section.

        The field is shown as a color map. Axes and aspect ratio are set automatically
        based on the geometry (``Igeometry`` 1: slab, 2: cylindrical, 3: toroidal).

        Args:
            ivol (int, optional): Volume index to evaluate the field in. Defaults to 0.
            sarr (array-like, optional): Radial coordinate samples in :math:`[-1, 1]`.
                Defaults to ``np.linspace(-0.999, 1, 32)``.
            tarr (array-like, optional): Poloidal angle samples in :math:`[0, 2\\pi]`.
                Defaults to ``np.linspace(0, 2\\pi, 42)``.
            zarr (float or array-like, optional): Toroidal angle (radians) at which to evaluate
                the field. Defaults to 0.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. A new figure is created if None.
                Defaults to None.
            colorbar (bool, optional): Whether to add a colorbar labelled :math:`|\\vec{B}|`.
                Defaults to True.
            **kwargs: Additional keyword arguments passed to :func:`matplotlib.pyplot.pcolormesh`.
        """
        if sarr is None:
            sarr = np.linspace(-0.999, 1, 32)
        if tarr is None:
            tarr = np.linspace(0, 2 * np.pi, 42)

        if ax is None:
            _, ax = plt.subplots()

        plt.sca(ax)

        Rfull, Zfull = self.get_coord_transform(lvol=ivol, sarr=sarr, tarr=tarr, zarr=zarr)
        R = Rfull[0, :, :, 0]
        Z = Zfull[0, :, :, 0]

        modB = self.get_field_mod(lvol=ivol, sarr=sarr, tarr=tarr, zarr=zarr)

        Igeometry = self.input.physics.Igeometry

        if Igeometry == 1:
            plotR = tarr[None, :]
            plotZ = R
        if Igeometry == 2:
            plotR = R * np.cos(tarr[None, :])
            plotZ = R * np.sin(tarr[None, :])
        if Igeometry == 3:
            plotR = R
            plotZ = Z

        plot = ax.pcolormesh(plotR[:, :], plotZ[:, :], modB[:, :, 0], **kwargs)  #

        if Igeometry == 1:
            ax.set_xlabel(r"$\theta$")
            ax.set_ylabel(r"$R$")
        if Igeometry == 2:
            ax.set_aspect("equal")
            ax.set_xlabel(r"$X$")
            ax.set_ylabel(r"$Y$")
        if Igeometry == 3:
            # ax.set_aspect("equal")
            ax.set_xlabel(r"$R$")
            ax.set_ylabel(r"$Z$")

        ax.axis("equal")
        plt.tight_layout()

        if colorbar:
            cbar = plt.colorbar(plot, ax=ax)
            cbar.set_label(r"$|\vec B|$", rotation=0)
        return

    def plot_poincare(self, zetan=0, prange="full", ax=None, **kwargs):
        """Plot the Fortran-based Poincaré data for a given toroidal cross-section.

        Args:
            zetan (float, optional): Toroidal angle in units of :math:`2\\pi/N_{fp}`. Defaults to 0.
            prange (str, optional): Poloidal range to plot, one of ``'full'``, ``'upper'``, or
                ``'lower'``. Defaults to ``'full'``.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. Defaults to None.
            **kwargs: Additional keyword arguments passed to :func:`matplotlib.pyplot.scatter`.

        Returns:
            matplotlib.collections.PathCollection: Return value :func:`matplotlib.pyplot.scatter`.
        """

        fontsize_labels = None

        nfp = self.input.physics.Nfp
        Igeometry = self.input.physics.Igeometry

        zeta = zetan * (2 * np.pi / nfp)

        if "poincare" not in self:
            return None

        if Igeometry == 3:
            zeta = zeta - (2 * np.pi / nfp) * int(
                zeta / (2 * np.pi / nfp)
            )  # move zeta into one field period

            num_tor_planes = self.poincare.R.shape[2]
            zetas = np.linspace(0, 2 * np.pi / nfp, num_tor_planes, endpoint=False)
            print(f"Normalized zetas: {zetas / (2 * np.pi / nfp)}")
            toroidalIdx = np.argmin(np.abs(zetas - zeta))
            if abs(zetas[0] + 6.28 / nfp - zeta) < abs(zetas[toroidalIdx] - zeta):
                toroidalIdx = 0

            rr = self.poincare.R[:, :, toroidalIdx]
            zz = self.poincare.Z[:, :, toroidalIdx]
        elif Igeometry == 2:
            num_tor_planes = self.poincare.R.shape[2]
            zetas = np.linspace(0, 2 * np.pi, num_tor_planes, endpoint=False)
            toroidalIdx = np.argmin(np.abs(zetas - zeta))

            rr = self.poincare.R[:, :, toroidalIdx] * np.cos(self.poincare.t[:, :, toroidalIdx])
            zz = self.poincare.R[:, :, toroidalIdx] * np.sin(self.poincare.t[:, :, toroidalIdx])
        elif Igeometry == 1:
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
        if Igeometry == 3:
            row_indices, col_indices = np.where(rr**2 < 1e-16)
            if len(row_indices) > 1 and np.all(row_indices == row_indices[0]):
                rr[row_indices] = None

        zz[rr < 1e-4] = None
        rr[rr < 1e-4] = None

        if prange == "full":
            dots = ax.plot(rr, zz, "o", **kwargs)
        elif prange == "upper":
            dots = ax.plot(rr[zz >= 0], zz[zz >= 0], ".", **kwargs)
        elif prange == "lower":
            dots = ax.plot(rr[zz <= 0], zz[zz <= 0], ".", **kwargs)
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
            plt.xlim([0, 2 * np.pi])

        return dots

    def plot_pressure(
        self, normalize=True, xaxis=None, tarr=None, zarr=None, ax=None, scalar=1.0, **kwargs
    ):
        """Plot the stepped pressure profile.

        Args:
            normalize (bool, optional): If True, pressure is plotted in units of :math:`\\mu_0`.
                Defaults to True.
            xaxis (str, optional): Choice of x-axis, ``'tflux'`` (normalized toroidal flux,
                default) or ``'x'`` (major radius coordinate).
            tarr (array-like, optional): Poloidal angle samples used when ``xaxis='x'``.
                Defaults to ``np.array([3.0])``.
            zarr (array-like, optional): Toroidal angle samples used when ``xaxis='x'``.
                Defaults to ``np.array([0.0])``.
            ax (matplotlib.axes.Axes, optional): Axes to plot on. A new figure is created if None.
                Defaults to None.
            scalar (float, optional): Multiplicative scaling factor applied to the pressure values.
                Defaults to 1.0.
            **kwargs: Additional keyword arguments passed to :func:`matplotlib.pyplot.plot`.
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

        if ax is None:
            _, ax = plt.subplots()

        if xaxis == "tflux":
            tflux = self.output.tflux[: len(pressure)]

            _tflux = np.insert(tflux, 0, 0)
            _pressure = np.append(pressure, 0)
            x_tflux = np.zeros(2 * len(tflux) + 1)
            x_tflux[0::2] = _tflux
            x_tflux[1::2] = tflux
            y_pressure = np.zeros(2 * len(pressure) + 1)
            y_pressure[0::2] = _pressure
            y_pressure[1::2] = pressure

            ax.plot(x_tflux[:-1], y_pressure[:-1] * scalar, **kwargs)

            ax.set_xlabel("Normalized flux")
            ax.set_ylabel("Pressure")

        elif xaxis == "x":
            tarr = np.array([3.0]) if tarr is None else tarr
            zarr = np.array([0.0]) if zarr is None else zarr

            for lvol in range(nvol):
                sarr = np.linspace(-1, 1, 2) if lvol == nvol // 2 else np.array([-1.0, 0.0, 1.0])
                R, _ = self.get_coord_transform(lvol, sarr, tarr, zarr)
                R = R[0, :, 0, 0]
                ax.plot(
                    R,
                    pressure[lvol] * np.ones_like(R) * scalar,
                    "r-",
                    label="SPECTRE" if lvol == 0 else None,
                    **kwargs,
                )

            ax.set_xlabel("x")
            ax.set_ylabel("Pressure")

    def trace_fieldlines(
        self,
        init_pts,
        num_phi_planes=32,
        num_transits=1,
        num_threads=1,
        batch_size=1,
        method="RK45",
        rtol=1.0e-7,
        atol=1.0e-7,
    ):
        """Trace field lines and generate a Poincaré section using the Python-based tracer.

        Constructs a :class:`Field_Tracer` from the equilibrium data, traces all provided
        initial points, and stores the result in ``self._tracer``.

        Args:
            init_pts (list of numpy.ndarray): One array per volume, each of shape
                ``(num_trajectories, 2)``, containing initial :math:`(s, \\theta)` coordinates.
            num_phi_planes (int, optional): Number of toroidal planes to record intersections.
                Defaults to 32.
            num_transits (int, optional): Number of toroidal transits to integrate per fieldline.
                Defaults to 1.
            num_threads (int, optional): Number of threads for parallel integration. Defaults to 1.
            batch_size (int, optional): Number of field lines dispatched per thread at a time.
                Defaults to 1.
            method (str, optional): ODE integrator, one of ``'RK23'``, ``'RK45'``, or ``'DOP853'``.
                Defaults to ``'RK45'``.
            rtol (float, optional): Relative tolerance of the ODE integrator. Defaults to 1e-7.
            atol (float, optional): Absolute tolerance of the ODE integrator. Defaults to 1e-7.
        """

        from spectre.field_tracer import Field_Tracer

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

        tracer = Field_Tracer(
            mvol,
            igeometry,
            nfp,
            mpol,
            im,
            in_,
            Ate,
            Aze,
            Ato,
            Azo,
            Rbs,
            Rbc,
            Zbs,
            Zbc,
            lrad_arr,
            num_threads=num_threads,
            batch_size=batch_size,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        tracer.trace_field(init_pts, num_phi_planes=num_phi_planes, num_transits=num_transits)

        self._tracer = tracer

    def trace_one_fieldline(
        self,
        sth_init,
        ivol,
        num_phi_planes=32,
        num_transits=1,
        method="RK45",
        rtol=1.0e-7,
        atol=1.0e-7,
    ):
        """Trace a single field line from a given starting point.

        On the first call a :class:`Field_Tracer` is constructed from the equilibrium data
        and cached in ``self._tracer_single`` for subsequent calls.

        Args:
            sth_init (array-like): Initial :math:`(s, \\theta)` coordinates of the field line.
            ivol (int): Index of the volume in which to start the trace.
            num_phi_planes (int, optional): Number of toroidal planes where to record intersection.
                Defaults to 32.
            num_transits (int, optional): Number of full toroidal transits to integrate.
                Defaults to 1.
            method (str, optional): ODE integrator, one of ``'RK23'``, ``'RK45'``, or ``'DOP853'``.
                Defaults to ``'RK45'``.
            rtol (float, optional): Relative tolerance of the ODE integrator. Defaults to 1e-7.
            atol (float, optional): Absolute tolerance of the ODE integrator. Defaults to 1e-7.

        Returns:
            tuple:
                - **sth** (*array-like*) – Final :math:`(s, \\theta)` coordinates.
                - **RZ** (*array-like*) – Final :math:`(R, Z)` coordinates.
                - **iota** (*float*) – Estimated rotational transform at the final point.
                - **flag** (*bool*) – True if integration terminated early, False if it completed.
        """

        from spectre.field_tracer import Field_Tracer

        if not hasattr(self, "_tracer_single"):
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

            tracer = Field_Tracer(
                mvol,
                igeometry,
                nfp,
                mpol,
                im,
                in_,
                Ate,
                Aze,
                Ato,
                Azo,
                Rbs,
                Rbc,
                Zbs,
                Zbc,
                lrad_arr,
                method=method,
                rtol=rtol,
                atol=atol,
            )

            self._tracer_single = tracer

        sth, RZ, iota, flag = self._tracer_single.trace_field_single(
            sth_init,
            ivol,
            num_phi_planes=num_phi_planes,
            num_transits=num_transits,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        return sth, RZ, iota, flag

    @property
    def tracer(self):
        if not hasattr(self, "_tracer"):
            raise AttributeError("Field tracing has not been done, run 'trace_fieldlines()' first")
        return self._tracer

    def find_axis(self, max_repeats=0, guess=None, verbose=False):
        """Find the magnetic axis using root-finding on the field-line return map.

        The result is also stored in ``self.r_magaxis`` and ``self.z_magaxis``.

        Args:
            max_repeats (int, optional): Number of additional attempts with random initial guesses
                if the root-finder fails. Defaults to 0.
            guess (array-like, optional): Initial :math:`(R, Z)` estimate for the axis location.
                Defaults to None.
            verbose (bool, optional): Whether to print convergence information. Defaults to False.

        Returns:
            tuple:
                - **rax** (*float*) – :math:`R` coordinate of the magnetic axis.
                - **zax** (*float*) – :math:`Z` coordinate of the magnetic axis.
                - **sax** (*float*) – Radial coordinate :math:`s` of the axis.
                - **thax** (*float*) – Poloidal angle :math:`\\theta` of the axis.
        """
        from spectre.fixed_points import find_axis

        rax, zax, sax, thax = find_axis(self, max_repeats=max_repeats, guess=guess, verbose=verbose)

        self.r_magaxis = rax
        self.z_magaxis = zax

        return rax, zax, sax, thax
