# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Shared ``map2disc`` (BCM) utilities for interpolating internal interfaces.

The :mod:`map2disc` package provides a 2-D boundary-conformal map (``BCM``) that
maps the unit disc onto an arbitrary poloidal cross-section.  By evaluating the
map at radii ``rho = sqrt(tflux)`` it produces geometrically smooth *nested*
interfaces inside a given plasma boundary.

This module holds the single, source-agnostic implementation of that mapping so
that the two callers in this package do not duplicate it:

- :func:`~spectre.converters.mesh_gen.gen_mesh_calc_rz` /
  :func:`~spectre.converters.mesh_gen.gen_infaces` map the boundary stored in a
  live :class:`~spectre.core.core.SPECTRE` object's state.
- :func:`compute_interface_guesses_map2disc` maps a boundary given directly as
  Fourier coefficient dicts (no SPECTRE object required).

Both feed :func:`map_boundary_to_interfaces` a ``make_boundary`` callback — a
callable that, given a toroidal angle ``zeta``, returns the boundary curve
``theta -> (R, Z)`` at that cross-section.

Conventions are the caller's responsibility: the negative-``theta`` sign
convention and any rescaling of the returned radial gradients into SPECTRE's
``s in [-1, 1]`` coordinate are applied by the caller, not here.  This routine
works purely in map2disc's native ``rho in [0, 1]`` coordinate.
"""

import numpy as np


def map_boundary_to_interfaces(make_boundary, rho_arr, th_arr, zeta_arr, bcm_m, with_grad=False):
    """Conformally map a boundary inward to nested interfaces using map2disc BCM.

    For each toroidal cross-section ``zeta`` in ``zeta_arr`` the boundary curve is
    obtained from ``make_boundary(zeta)``, a BCM solve is performed, and the map is
    evaluated on the ``(rho_arr, th_arr)`` grid.  BCM is a purely 2-D map, so the
    full 3-D geometry is assembled one ``zeta`` slice at a time.

    Args:
        make_boundary: ``callable(zeta) -> callable(theta) -> (R, Z)``; given a
            toroidal angle it returns the boundary curve at that cross-section.
            Returning a per-``zeta`` curve lets the caller do any per-slice setup
            once instead of on every BCM evaluation.
        rho_arr: 1-D array of radial positions in map2disc's ``[0, 1]`` coordinate
            (typically ``sqrt(tflux)``); one entry per interface.
        th_arr: 1-D array of poloidal angles at which to evaluate the map.  The
            caller supplies the sign convention (SPECTRE uses negated angles).
        zeta_arr: 1-D array of toroidal angles (one field period is sufficient).
        bcm_m: number of Fourier modes used by the BCM solver.
        with_grad: if ``True`` also return the radial gradients ``dR/drho`` and
            ``dZ/drho`` in map2disc's ``rho`` coordinate.

    Returns:
        ``(R_arr, Z_arr)`` when ``with_grad`` is ``False``, otherwise
        ``(R_arr, Z_arr, dRdrho_arr, dZdrho_arr)``.  Every array has shape
        ``(len(rho_arr), len(zeta_arr), len(th_arr))``.

    Raises:
        ImportError: if ``map2disc`` is not installed.
    """
    try:
        from map2disc import BCM
    except ImportError as exc:
        raise ImportError(
            "map2disc is required for interface_guess='map2disc'. "
            "Install it with: pip install map2disc"
        ) from exc

    rho_arr = np.asarray(rho_arr, dtype=float)
    th_arr = np.asarray(th_arr, dtype=float)
    zeta_arr = np.asarray(zeta_arr, dtype=float)

    shape = (rho_arr.size, zeta_arr.size, th_arr.size)
    R_arr = np.empty(shape)
    Z_arr = np.empty_like(R_arr)
    if with_grad:
        dRdrho_arr = np.empty_like(R_arr)
        dZdrho_arr = np.empty_like(R_arr)
        cos_th_arr = np.cos(th_arr)[None, :]
        sin_th_arr = np.sin(th_arr)[None, :]

    for iz, zeta in enumerate(zeta_arr):
        bcm = BCM(make_boundary(zeta), M=bcm_m)
        bcm.solve()

        x, y = bcm.eval_rt_1d(rho_arr, th_arr)
        R_arr[:, iz, :] = x
        Z_arr[:, iz, :] = y

        if with_grad:
            # eval_grad_rt_1d gives the gradient in (rho, theta) polar coordinates;
            # project onto the radial direction to obtain dR/drho, dZ/drho.
            rgrad = bcm.eval_grad_rt_1d(rho_arr, th_arr)
            dxy_dr = rgrad[:, 0] * cos_th_arr + rgrad[:, 1] * sin_th_arr
            dRdrho_arr[:, iz, :] = dxy_dr[0]
            dZdrho_arr[:, iz, :] = dxy_dr[1]

    if with_grad:
        return R_arr, Z_arr, dRdrho_arr, dZdrho_arr
    return R_arr, Z_arr


def compute_interface_guesses_map2disc(
    rbc, zbs, nfp, tfluxes, mpol, ntor, ntheta=124, nzeta=78, bcm_m=6
):
    """Generate interface geometry guesses using conformal mapping (map2disc).

    Uses the ``map2disc`` BCM solver to conformally map the plasma boundary
    inward to each interface position, producing geometrically smooth nested
    surfaces.  Only the boundary Fourier harmonics are required — no VMEC wout
    or SPECTRE object is needed — so this works from both ``input.*`` and
    ``wout_*.nc`` sources.

    The ``rbc`` / ``zbs`` dicts must be in SPECTRE convention (angle-fix applied,
    i.e. the output of :func:`~spectre.converters.vmec_helpers.vmec_boundary_to_fourier_coeffs`
    with ``angle_fix=True``).  The same sign convention (``th_arr = -th_arr``) as
    :func:`~spectre.converters.mesh_gen.gen_infaces` is used internally so the FFT
    output is directly compatible with ``allrzrz``.

    Args:
        rbc: boundary R cosine harmonics as ``{'(m, n)': value}`` dict
            (SPECTRE convention, angle-fix applied).
        zbs: boundary Z sine harmonics as ``{'(m, n)': value}`` dict
            (SPECTRE convention, angle-fix applied).
        nfp: number of toroidal field periods.
        tfluxes: 1-D array of interface toroidal-flux values in (0, 1].
        mpol: maximum poloidal mode number for the output.
        ntor: maximum toroidal mode number (reduced, without nfp) for the output.
        ntheta: number of poloidal quadrature points (default 124).
        nzeta: number of toroidal quadrature points per field period (default 78).
        bcm_m: number of Fourier modes used by BCM (default 6).

    Returns:
        dict suitable for ``PhysicsParameters.allrzrz``:
        ``{'interface_1': {'rbc': {...}, 'zbs': {...}, 'rbs': {...}, 'zbc': {...}},
        ...}``

    Raises:
        ImportError: if ``map2disc`` is not installed.
    """
    from spectre.fourier import spec_fft_fast

    num_ifaces = len(tfluxes)
    # BCM works in the unit disc with radial coordinate rho = sqrt(tflux).
    rho_arr = np.sqrt(np.asarray(tfluxes, dtype=float))

    # Negative poloidal angle: SPECTRE's internal theta sign convention (same as
    # gen_infaces). The FFT in spec_fft_fast expects this orientation to produce
    # the correct even/odd split.
    th_arr = -np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    # One toroidal field period is sufficient; BCM solves each zeta slice independently.
    zeta_arr = np.linspace(0.0, 2.0 * np.pi / nfp, nzeta, endpoint=False)

    # Parse the '(m, n)' dict keys once; n is the reduced toroidal index (without nfp).
    rbc_modes = [
        (int(k.strip("()").split(",")[0]), int(k.strip("()").split(",")[1]), v)
        for k, v in rbc.items()
    ]
    zbs_modes = [
        (int(k.strip("()").split(",")[0]), int(k.strip("()").split(",")[1]), v)
        for k, v in zbs.items()
    ]

    def make_boundary(zeta):
        # Boundary curve at a fixed zeta: R = sum rbc cos(m*th - n*nfp*zeta), Z likewise.
        def boundary(theta):
            th = np.asarray(theta, dtype=float)
            R = np.zeros_like(th)
            Z = np.zeros_like(th)
            for m, n, v in rbc_modes:
                R += v * np.cos(m * th - n * nfp * zeta)
            for m, n, v in zbs_modes:
                Z += v * np.sin(m * th - n * nfp * zeta)
            return R, Z

        return boundary

    R_arr, Z_arr = map_boundary_to_interfaces(make_boundary, rho_arr, th_arr, zeta_arr, bcm_m)

    # FFT step: decompose R and Z into Fourier harmonics over (theta, zeta).
    # spec_fft_fast expects shape (..., nzeta, ntheta) and returns (..., nmn).
    # R_even/Z_odd are the cos/sin harmonics (RBC/ZBS); R_odd/Z_even are the
    # sin/cos (RBS/ZBC).
    R_even, R_odd, im_, in_ = spec_fft_fast(th_arr, zeta_arr, R_arr, mpol, ntor)
    Z_even, Z_odd, _, _ = spec_fft_fast(th_arr, zeta_arr, Z_arr, mpol, ntor)
    # in_ is the reduced toroidal index (n, not n*nfp) — use it directly in the keys.
    nmn = len(im_)

    # Pack into the allrzrz dict format expected by PhysicsParameters.
    # One entry per interface, each containing the four Fourier coefficient dicts.
    allrzrz = {}
    for ivol in range(num_ifaces):
        allrzrz[f"interface_{ivol + 1}"] = {
            "rbc": {f"({im_[j]}, {in_[j]})": float(R_even[ivol, j]) for j in range(nmn)},
            "zbs": {f"({im_[j]}, {in_[j]})": float(Z_odd[ivol, j]) for j in range(nmn)},
            "rbs": {f"({im_[j]}, {in_[j]})": float(R_odd[ivol, j]) for j in range(nmn)},
            "zbc": {f"({im_[j]}, {in_[j]})": float(Z_even[ivol, j]) for j in range(nmn)},
        }

    return allrzrz
