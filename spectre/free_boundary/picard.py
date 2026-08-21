# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Picard iteration on the plasma-produced normal field ``iBns`` (``lfreebsolver = 0``).

The public entry point is :meth:`spectre.core.SPECTRE.update_bns`; this module holds the
pieces it is built from: the flux convergence at fixed ``iBns`` and one damped sweep. Every
function takes the :class:`~spectre.core.SPECTRE` instance as its first argument and is
MPI-collective.
"""

import numpy as np

from .free_boundary import bnorm_from_plasma, sync_tflux_pflux

#: ``dforce`` passes allowed to settle ``dtflux(Mvol)`` in a free-boundary sweep (it settles
#: in 2-3; the linking-current constraint is linear in ``dtflux``).
MAX_FLUX_PASSES = 10


def converge_dtflux(spectre, tol_dtflux=1e-10, verbose=False):
    """Repeat ``dforce`` until ``dtflux(Mvol)`` settles, then write back ``tflux``/``pflux``.

    ``dforce`` adjusts the vacuum-volume ``dtflux`` to hold the linking current
    ``curtor``; the constraint is linear in ``dtflux``, so this settles in a few
    passes (capped at :data:`MAX_FLUX_PASSES`).

    Args:
        spectre (SPECTRE): Instance to converge, at its live geometry.
        tol_dtflux (float): Convergence tolerance on ``|delta dtflux(Mvol)|``.
        verbose (bool): Print the converged vacuum-volume ``tflux`` on rank 0.
    """
    x = spectre.xin
    ivac = int(spectre.allglobal_mod.mvol) - 1  # dtflux is allocated (1:Mvol)
    for _ in range(MAX_FLUX_PASSES):
        dt_before = float(spectre.allglobal_mod.dtflux[ivac])
        spectre.dforce(x)
        if abs(float(spectre.allglobal_mod.dtflux[ivac]) - dt_before) < tol_dtflux:
            break
    sync_tflux_pflux(spectre)
    if verbose and spectre.comm.rank == 0:
        tf_vac = float(np.asarray(spectre.input_list_mod.tflux)[ivac])
        print(f"update_bns: vacuum-volume tflux={tf_vac:.6e} after flux convergence")


def refresh_sweep(spectre, damping, first_free_bound=False, tol_dtflux=1e-10, verbose=False):
    """One damped Picard sweep at fixed geometry; returns the pre-blend ``bnserr``.

    Flux-converge (:func:`converge_dtflux`), extrapolate the plasma field onto the
    computational boundary (:func:`bnorm_from_plasma`), measure the fixed-point residual
    ``bnserr = mean|iBns - ofm|`` of the ``iBns`` the field was just solved at, then blend
    ``iBns := damping*iBns + (1-damping)*ofm`` in place. The residual is measured before
    the blend, so it is independent of ``damping``.

    Args:
        spectre (SPECTRE): Instance to sweep, at its live geometry.
        damping (float): Blend factor; ``1.0`` freezes ``iBns``, ``0.0`` replaces it.
        first_free_bound (bool): Passed to :func:`bnorm_from_plasma`.
        tol_dtflux (float): Passed to :func:`converge_dtflux`.
        verbose (bool): Passed to :func:`converge_dtflux`.

    Returns:
        float: The pre-blend ``bnserr``.
    """
    converge_dtflux(spectre, tol_dtflux, verbose)

    # Raises on non-finite harmonics before anything below can touch iBns.
    efm, ofm = bnorm_from_plasma(spectre, first_free_bound)

    mn = int(spectre.allglobal_mod.mn)
    stellsym = bool(spectre.input_list_mod.enforce_stell_sym)
    ibns = spectre.allglobal_mod.ibns
    ibnc = spectre.allglobal_mod.ibnc

    # Fixed-point residual of the iBns the field was just solved at, measured before
    # the blend so it is independent of the damping factor.
    bnserr = 0.0 if mn == 1 else float(np.sum(np.abs(ibns[1:mn] - ofm[1:mn]))) / (mn - 1)
    if not stellsym:
        bnserr += float(np.sum(np.abs(ibnc[:mn] - efm[:mn]))) / mn

    # Damped blend, in place: prepare_endfile_writing and /output/Bns read these
    # arrays. The m = n = 0 sine mode is identically zero and is never blended.
    ibns[1:mn] = damping * ibns[1:mn] + (1.0 - damping) * ofm[1:mn]
    if not stellsym:
        ibnc[:mn] = damping * ibnc[:mn] + (1.0 - damping) * efm[:mn]

    # Nothing on the Fortran side sets bnserr; the output writer reads it from here.
    spectre.allglobal_mod.bnserr = bnserr

    # iBns moved: the Beltrami field in memory no longer matches the free-boundary
    # state, even though the geometry is unchanged.
    spectre._field_fingerprint = None
    spectre._field_solved_at = None
    return bnserr
