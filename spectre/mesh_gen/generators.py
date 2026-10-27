# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Assemble SPECTRE meshes and interface guesses from mapped geometry.

These helpers act on a live :class:`~spectre.core.core.SPECTRE` object at a given
state (its packed interface vector ``xin``).  The boundary is conformally mapped
inward with the BCM map in :mod:`spectre.mesh_gen.map2disc`; this module performs
the source-agnostic remainder of the pipeline — FFT into Fourier harmonics,
radial differentiation, and packing the result into either SPECTRE's
external-mesh arrays (:func:`gen_mesh_external`) or interface guesses
(:func:`gen_infaces`).
"""

import numpy as np

from spectre.file_io import write_input_parameters_to_toml
from spectre.fourier import spec_fft_fast
from spectre.utils import pack_interfaces

from .map2disc import gen_mesh_calc_rz


def gen_mesh_fft_rz(R_arr, Z_arr, dRdrho_arr, dZdrho_arr, test, th_arr, zeta_arr):
    """FFT R, Z and their radial gradients into even/odd Fourier harmonics.

    Decomposes each real-space ``(rho, zeta, theta)`` array into its even (cos) and
    odd (sin) Fourier harmonics over ``(theta, zeta)`` using ``spec_fft_fast`` at the
    SPECTRE object's ``mpol``/``ntor`` resolution.

    Args:
        R_arr: real-space R array.
        Z_arr: real-space Z array.
        dRdrho_arr: real-space radial gradient of R.
        dZdrho_arr: real-space radial gradient of Z.
        test: a live ``SPECTRE`` object (provides ``mpol``/``ntor``).
        th_arr: poloidal angles.
        zeta_arr: toroidal angles.

    Returns:
        ``(R_even, dRdrho_even, R_odd, dRdrho_odd, Z_even, dZdrho_even, Z_odd,
        dZdrho_odd)`` Fourier-harmonic arrays.
    """
    mpol = test.input_list_mod.mpol
    ntor = test.input_list_mod.ntor

    R_even, R_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, R_arr, mpol, ntor)
    Z_even, Z_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, Z_arr, mpol, ntor)

    dRdrho_even, dRdrho_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, dRdrho_arr, mpol, ntor)
    dZdrho_even, dZdrho_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, dZdrho_arr, mpol, ntor)

    return R_even, dRdrho_even, R_odd, dRdrho_odd, Z_even, dZdrho_even, Z_odd, dZdrho_odd


def gen_mesh_calc_deriv2(dRdrho_even, dRdrho_odd, dZdrho_even, dZdrho_odd, rho_arr):
    """Finite-difference the second radial derivatives of R and Z harmonics.

    Computes ``d2R/drho2`` and ``d2Z/drho2`` (even and odd harmonics) from the first
    radial derivatives by central differencing along the radial axis with
    ``np.gradient``.

    Args:
        dRdrho_even: even first-radial-derivative harmonics of R.
        dRdrho_odd: odd first-radial-derivative harmonics of R.
        dZdrho_even: even first-radial-derivative harmonics of Z.
        dZdrho_odd: odd first-radial-derivative harmonics of Z.
        rho_arr: radial coordinate samples used as the gradient spacing.

    Returns:
        ``(dRdrho2_even, dRdrho2_odd, dZdrho2_even, dZdrho2_odd)``.
    """
    dRdrho2_even = np.gradient(dRdrho_even, rho_arr, axis=0)
    dRdrho2_odd = np.gradient(dRdrho_odd, rho_arr, axis=0)
    dZdrho2_even = np.gradient(dZdrho_even, rho_arr, axis=0)
    dZdrho2_odd = np.gradient(dZdrho_odd, rho_arr, axis=0)

    return dRdrho2_even, dRdrho2_odd, dZdrho2_even, dZdrho2_odd


def gen_mesh_set_remn(
    R_even,
    dRdrho_even,
    dRdrho2_even,
    R_odd,
    dRdrho_odd,
    dRdrho2_odd,
    Z_even,
    dZdrho_even,
    dZdrho2_even,
    Z_odd,
    dZdrho_odd,
    dZdrho2_odd,
    test,
):
    """Store the external-mesh Fourier harmonics into SPECTRE's ``*_ext`` arrays.

    Allocates the ``remn_ext`` / ``romn_ext`` / ``zemn_ext`` / ``zomn_ext`` arrays in
    SPECTRE and fills them with the value, first and second radial derivatives of the
    even/odd R and Z harmonics, laid out as SPECTRE expects.

    Args:
        R_even: even R harmonic (value).
        dRdrho_even: even R first radial derivative.
        dRdrho2_even: even R second radial derivative.
        R_odd: odd R harmonic (value).
        dRdrho_odd: odd R first radial derivative.
        dRdrho2_odd: odd R second radial derivative.
        Z_even: even Z harmonic (value).
        dZdrho_even: even Z first radial derivative.
        dZdrho2_even: even Z second radial derivative.
        Z_odd: odd Z harmonic (value).
        dZdrho_odd: odd Z first radial derivative.
        dZdrho2_odd: odd Z second radial derivative.
        test: a live ``SPECTRE`` object whose external-mesh arrays are populated.

    Returns:
        None. SPECTRE's external-mesh arrays are modified in place.
    """
    test.wrapper_funcs_mod.alloc_remn_ext()

    test.allglobal_mod.remn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([R_even, dRdrho_even, dRdrho2_even]), 0, 2), 0, 1
    )
    test.allglobal_mod.romn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([R_odd, dRdrho_odd, dRdrho2_odd]), 0, 2), 0, 1
    )
    test.allglobal_mod.zemn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([Z_even, dZdrho_even, dZdrho2_even]), 0, 2), 0, 1
    )
    test.allglobal_mod.zomn_ext[:] = np.moveaxis(
        np.moveaxis(np.array([Z_odd, dZdrho_odd, dZdrho2_odd]), 0, 2), 0, 1
    )


def gen_mesh_external(xin, test, bcm_m=6, inface=0, ntheta=101, nzeta=32):
    """Build the external mesh from a SPECTRE interface and store it in SPECTRE.

    Conformally maps interface ``inface`` of ``test`` onto SPECTRE's radial Gaussian
    abscissae, FFTs the result and its first/second radial derivatives, and writes the
    harmonics into SPECTRE's external-mesh arrays via :func:`gen_mesh_set_remn`.

    The radial gradients returned by the map2disc loop are in ``rho in [0, 1]`` while
    SPECTRE's radial coordinate runs ``s in [-1, 1]``; the ``/ 2.0`` factors below
    convert ``d/drho`` to ``d/ds`` for the first and second derivatives.

    Args:
        xin: packed interface state vector.
        test: a live ``SPECTRE`` object; its external-mesh arrays are populated.
        bcm_m: number of BCM Fourier modes
        inface: interface index used as the mapping boundary.
        ntheta: poloidal sample count
        nzeta: toroidal sample count

    Returns:
        None. SPECTRE's external-mesh arrays are modified in place.
    """
    nfp = test.input_list_mod.nfp

    zeta_arr = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
    th_arr = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    th_arr = -th_arr  # must be negative due to some convention in spectre

    rho_spec_abscis = (test.allglobal_mod.gaussianabscissae[:, 0] + 1.0) * 0.5

    R_arr, Z_arr, dRdrho_arr, dZdrho_arr = gen_mesh_calc_rz(
        xin, test, inface, bcm_m, rho_spec_abscis, th_arr, zeta_arr
    )

    # rho in SPECTRE: -1 -> 1, map2disc: 0 -> 1
    dRdrho_arr /= 2.0
    dZdrho_arr /= 2.0

    R_even, dRdrho_even, R_odd, dRdrho_odd, Z_even, dZdrho_even, Z_odd, dZdrho_odd = (
        gen_mesh_fft_rz(R_arr, Z_arr, dRdrho_arr, dZdrho_arr, test, th_arr, zeta_arr)
    )

    dRdrho2_even, dRdrho2_odd, dZdrho2_even, dZdrho2_odd = gen_mesh_calc_deriv2(
        dRdrho_even, dRdrho_odd, dZdrho_even, dZdrho_odd, rho_spec_abscis
    )

    # rho in SPECTRE: -1 -> 1, map2disc: 0 -> 1
    dRdrho2_even /= 2.0
    dRdrho2_odd /= 2.0
    dZdrho2_even /= 2.0
    dZdrho2_odd /= 2.0

    gen_mesh_set_remn(
        R_even,
        dRdrho_even,
        dRdrho2_even,
        R_odd,
        dRdrho_odd,
        dRdrho2_odd,
        Z_even,
        dZdrho_even,
        dZdrho2_even,
        Z_odd,
        dZdrho_odd,
        dZdrho2_odd,
        test,
    )


def gen_infaces(test, xin, tflux=None, set_to_file=False, base_inface=None, fname_out=None):
    """Generate nested internal interface guesses for a SPECTRE equilibrium.

    Conformally maps the ``base_inface`` boundary of ``test`` inward to each requested
    toroidal-flux value and FFTs the result into interface Fourier harmonics.  When
    ``set_to_file`` is set, the new interfaces are packed back into SPECTRE, a final
    diagnostics pass is run, and the resulting parameters are written to TOML via the
    ``SPECTRE.parameters`` property.

    Args:
        test: a live ``SPECTRE`` object whose boundary is mapped.
        xin: packed interface state vector providing the boundary geometry.
        tflux: 1-D array of normalised interface flux values; defaults to the object's
            own ``tflux`` profile (excluding the boundary).
        set_to_file: if ``True``, apply the new interfaces and write a TOML file.
        base_inface: interface index to use as the mapping boundary; defaults to the
            outermost interface (``mvol - 1``).
        fname_out: output TOML path; defaults to ``"{test.name}_init.toml"``.

    Returns:
        ``(R_even, R_odd, Z_even, Z_odd)`` interface Fourier-harmonic arrays.

    Raises:
        RuntimeError: if ``mvol != len(tflux) + 1``.
    """
    mvol = test.allglobal_mod.mvol
    nfp = test.input_list_mod.nfp

    if tflux is None:
        tflux = test.input_list_mod.tflux[: mvol - 1] / test.input_list_mod.tflux[mvol - 1]
    base_inface = mvol - 1 if base_inface is None else base_inface

    num_infaces = len(tflux)

    if mvol != (num_infaces + 1):
        raise RuntimeError("Input equilibria has to have mvol=(num_infaces+1)")

    ntheta = 124
    nzeta = 78
    bcm_m = 6

    # convert tflux to radius assuming tflux~=r^2
    rho_arr = np.sqrt(tflux)

    zeta_arr = np.linspace(0, 2 * np.pi / nfp, nzeta, endpoint=False)
    th_arr = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
    th_arr = -th_arr  # must be negative due to some convention in spectre

    R_arr, Z_arr, _, _ = gen_mesh_calc_rz(xin, test, base_inface, bcm_m, rho_arr, th_arr, zeta_arr)

    R_even, _, R_odd, _, Z_even, _, Z_odd, _ = gen_mesh_fft_rz(
        R_arr, Z_arr, np.empty((nzeta, ntheta)), np.empty((nzeta, ntheta)), test, th_arr, zeta_arr
    )

    if set_to_file:
        xnew = pack_interfaces(R_even.T, R_odd.T, Z_even.T, Z_odd.T, test)

        test.force_real(xnew)

        test.input_list_mod.linitialize = 0

        test.allglobal_mod.skip_write = True
        nppts = test.input_list_mod.nppts
        test.input_list_mod.nppts = 0
        test.lib.simulation_mod.final_diagnostics()
        test.input_list_mod.nppts = nppts

        fname_out = f"{test.name}_init.toml" if fname_out is None else fname_out
        if test.comm.rank == 0:
            # sync fortran memory, then write the validated parameters straight to
            # the requested file via the SPECTRE.parameters property.
            test.wrapper_funcs_mod.prepare_endfile_writing()
            write_input_parameters_to_toml(test.parameters, fname_out)

    return R_even, R_odd, Z_even, Z_odd
