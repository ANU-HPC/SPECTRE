# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Free-boundary helpers that do not depend on the free-boundary solver.

Virtual-casing evaluation of the plasma-produced normal field, the flux write-back after
``dforce`` has moved ``dtflux(Mvol)``, and point-wise evaluation of the solved vacuum-region
field. Every function takes the :class:`~spectre.core.SPECTRE` instance as its first
argument and is MPI-collective unless stated otherwise.
"""

import numpy as np


def bnorm_from_plasma(spectre, first_free_bound=False):
    """Plasma-induced normal field on the computational boundary, by virtual casing.

    Non-mutating: ``iBns``/``iBnc`` are left alone (:meth:`~spectre.core.SPECTRE.update_bns`
    does the blending). The field in memory must be solved.

    Args:
        spectre (SPECTRE): Instance whose solved field is integrated.
        first_free_bound (bool): Integrate the plasma-side surface current (the bootstrap
            case, when no vacuum solution exists yet) instead of the outer face of the
            plasma. Call-scoped: the Fortran selector is reset afterwards, since a leaked
            ``True`` is fatal for a later ``lfreebsolver=1`` instance.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: ``(efm, ofm)``, the cosine and sine
        harmonics of the normal field, each of length ``mn``.

    Raises:
        FloatingPointError: The virtual casing returned non-finite harmonics (degenerate
            trial interface, failed cubature). Blending those into ``iBns`` would poison
            it for good and defeat every stopping test, so it is an error.
    """
    mn = int(spectre.allglobal_mod.mn)
    efm = np.zeros(mn, dtype=np.float64, order="F")
    ofm = np.zeros(mn, dtype=np.float64, order="F")
    try:
        spectre.allglobal_mod.first_free_bound = first_free_bound
        spectre.wrapper_funcs_mod.get_bnorm_plasmab(mn, efm, ofm)
    finally:
        spectre.allglobal_mod.first_free_bound = False
    stellsym = bool(spectre.input_list_mod.enforce_stell_sym)
    if not (np.all(np.isfinite(ofm[1:])) and (stellsym or np.all(np.isfinite(efm)))):
        raise FloatingPointError("virtual casing returned non-finite Bn harmonics")
    return efm, ofm


def vacuum_field_at(spectre, s, t, z):
    """Evaluate the solved vacuum-region field at points ``(s, theta, zeta)``.

    Args:
        spectre (SPECTRE): Instance whose solved field is evaluated.
        s (array_like): Radial coordinate of each point in the vacuum volume, in [-1, 1].
        t (array_like): Poloidal angle of each point.
        z (array_like): Toroidal angle of each point.

    Returns:
        tuple[numpy.ndarray, ...]: ``gb`` = (sqrt(g) B^s, sqrt(g) B^theta, sqrt(g) B^zeta),
        ``bcyl`` = (B_R, B_phi, B_Z) and ``rz`` = (R, Z), of shape ``(3, npts)``,
        ``(3, npts)`` and ``(2, npts)``.
    """
    s, t, z = (np.asfortranarray(p, dtype=np.float64).ravel() for p in (s, t, z))
    gb = np.zeros((3, s.size), dtype=np.float64, order="F")
    bcyl = np.zeros((3, s.size), dtype=np.float64, order="F")
    rz = np.zeros((2, s.size), dtype=np.float64, order="F")
    spectre.wrapper_funcs_mod.evaluate_vacuum_field(s.size, s, t, z, gb, bcyl, rz)
    return gb, bcyl, rz


def vacuum_field_derivative_at(spectre, s, t, z, ii, irz, issym):
    """Analytic derivative of :func:`vacuum_field_at` with respect to one boundary harmonic.

    Solves the perturbed vacuum problem for the raw plasma-boundary harmonic
    ``(ii, irz, issym)`` (one-based mode index, R/Z, even/odd) at fixed enclosed fluxes
    and evaluates it at fixed ``(s, theta, zeta)``; ``bcyl`` includes the movement of
    the coordinate point, which ``drz`` reports.

    Args:
        spectre (SPECTRE): Instance with a solved field.
        s (array_like): Radial coordinate of each point in the vacuum volume, in [-1, 1].
        t (array_like): Poloidal angle of each point.
        z (array_like): Toroidal angle of each point.
        ii (int): One-based Fourier mode index of the perturbed harmonic.
        irz (int): ``0`` for an R harmonic, ``1`` for a Z harmonic.
        issym (int): ``0`` for the stellarator-symmetric (cos/sin) harmonic, ``1`` otherwise.

    Returns:
        tuple[numpy.ndarray, ...]: ``(d gb, d bcyl, drz)`` shaped like :func:`vacuum_field_at`.
    """
    s, t, z = (np.asfortranarray(p, dtype=np.float64).ravel() for p in (s, t, z))
    gb = np.zeros((3, s.size), dtype=np.float64, order="F")
    bcyl = np.zeros((3, s.size), dtype=np.float64, order="F")
    drz = np.zeros((2, s.size), dtype=np.float64, order="F")
    spectre.wrapper_funcs_mod.solve_evaluate_vacuum_field_deriv(
        s.size, s, t, z, ii, irz, issym, gb, bcyl, drz
    )
    return gb, bcyl, drz


def sync_tflux_pflux(spectre):
    """Rebuild cumulative-normalized ``tflux``/``pflux`` from the per-volume increments.

    Inverse of the diff-and-scale in ``preset_mod.initialize_internal_arrays``:
    ``tflux[v] = sum_{k<=v} dtflux[k] * 2*pi/phiedge`` and likewise for ``pflux``,
    with the Igeometry 2/3 convention that ``pflux[0]`` stays zero. ``tflux`` is
    allocated ``(1:MNvol+1)`` with only the first ``Mvol`` entries real, so the
    volumes are indexed explicitly rather than from the end. Call this whenever
    ``dtflux``/``dpflux`` have moved by code that does not go through a Fortran
    resync. Process-local (no communication).

    Args:
        spectre (SPECTRE): Instance whose ``input_list_mod.tflux``/``pflux`` are rewritten.
    """
    phiedge = float(spectre.input_list_mod.phiedge)
    if phiedge == 0.0:
        return  # no scaling possible; nothing sensible to write

    pi2 = 2.0 * np.pi
    mvol = int(spectre.allglobal_mod.mvol)
    igeometry = int(spectre.input_list_mod.igeometry)

    dtflux = np.asarray(spectre.allglobal_mod.dtflux)
    dpflux = np.asarray(spectre.allglobal_mod.dpflux)

    # Mirror the Fortran resync arithmetic exactly (cumulate the raw increments, then
    # scale the sums by pi2/phiedge) so the write-back is bit-identical to the values
    # the Fortran driver used to produce.
    t = np.empty(mvol)
    p = np.empty(mvol)
    t[0] = dtflux[0]
    p[0] = dpflux[0] if igeometry == 1 else 0.0
    for v in range(1, mvol):
        t[v] = t[v - 1] + dtflux[v]
        p[v] = p[v - 1] + dpflux[v]

    spectre.input_list_mod.tflux[:mvol] = t * pi2 / phiedge
    spectre.input_list_mod.pflux[:mvol] = p * pi2 / phiedge
