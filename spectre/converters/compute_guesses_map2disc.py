# SPDX-FileCopyrightText: 2026 Proxima Fusion
# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""``map2disc`` (BCM) interface guesses from boundary Fourier coefficients.

The :mod:`map2disc` package provides a 2-D boundary-conformal map (``BCM``) that
maps the unit disc onto an arbitrary poloidal cross-section.  By evaluating the
map at radii ``rho = sqrt(tflux)`` it produces geometrically smooth *nested*
interfaces inside a given plasma boundary.

This module holds the converter-side entry point
:func:`compute_interface_guesses_map2disc`, which maps a boundary given directly
as Fourier-coefficient dicts (no SPECTRE object required).  The actual conformal
mapping loop lives in :func:`spectre.mesh_gen.map2disc.map_boundary_to_interfaces`
and is shared with the SPECTRE-state mesh generation in
:mod:`spectre.mesh_gen`.
"""

import numpy as np


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
    :func:`~spectre.mesh_gen.generators.gen_infaces` is used internally so the FFT
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
    from spectre.mesh_gen.map2disc import map_boundary_to_interfaces

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
