# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""``map2disc`` (BCM) conformal mapping of SPECTRE boundaries to nested interfaces.

The :mod:`map2disc` package provides a 2-D boundary-conformal map (``BCM``) that
maps the unit disc onto an arbitrary poloidal cross-section.  By evaluating the
map at radii ``rho = sqrt(tflux)`` it produces geometrically smooth *nested*
interfaces inside a given plasma boundary.

This module holds the BCM-dependent pieces of mesh generation:

- :func:`map_boundary_to_interfaces` — the single, source-agnostic mapping loop.
  It is also reused by
  :func:`spectre.converters.map2disc.compute_interface_guesses_map2disc`, which
  maps a boundary given directly as Fourier-coefficient dicts.
- :func:`get_inface_func` / :func:`gen_mesh_calc_rz` — build the boundary curve
  from a live :class:`~spectre.core.core.SPECTRE` object's packed state and map
  it inward.

Both feed :func:`map_boundary_to_interfaces` a ``make_boundary`` callback — a
callable that, given a toroidal angle ``zeta``, returns the boundary curve
``theta -> (R, Z)`` at that cross-section.

Conventions are the caller's responsibility: the negative-``theta`` sign
convention and any rescaling of the returned radial gradients into SPECTRE's
``s in [-1, 1]`` coordinate are applied by the caller, not here.  This routine
works purely in map2disc's native ``rho in [0, 1]`` coordinate.
"""

import numpy as np

from spectre.geometry import get_coord_transform
from spectre.utils import unpack_interfaces


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


def get_inface_func(xin, test, zeta, inface):
    """Build the boundary curve of one SPECTRE interface at a fixed toroidal angle.

    Unpacks the interface Fourier coefficients from the packed state vector ``xin``
    and returns a callable ``theta -> (R, Z)`` evaluating interface ``inface`` of the
    SPECTRE object ``test`` at the toroidal angle ``zeta``.

    Args:
        xin: packed interface state vector (see :func:`spectre.utils.unpack_interfaces`).
        test: a live ``SPECTRE`` object providing geometry and mode metadata.
        zeta: toroidal angle at which to evaluate the cross-section.
        inface: interface index (``0 <= inface <= mvol - 1``).

    Returns:
        ``callable(theta) -> (R, Z)`` evaluating the interface curve.

    Raises:
        RuntimeError: if ``inface`` is out of range.
    """
    geometry = test.input_list_mod.igeometry
    im = test.allglobal_mod.im
    in_ = test.allglobal_mod.in_
    mvol = test.allglobal_mod.mvol

    if inface > (mvol - 1) or inface < 0:
        raise RuntimeError(f"Invalid {inface=}")

    irbc, izbs, irbs, izbc = unpack_interfaces(xin, test)

    def func(t):
        rarr, zarr = get_coord_transform(
            geometry,
            irbc[:, inface],
            irbc[:, inface + 1],
            izbs[:, inface],
            izbs[:, inface + 1],
            im,
            in_,
            1.0,
            t,
            zeta,
            inface,
        )

        return rarr[0, 0, :, 0], zarr[0, 0, :, 0]

    return func


def gen_mesh_calc_rz(xin, test, inface, bcm_m, rho_arr, th_arr, zeta_arr):
    """Conformally map a SPECTRE interface inward, returning R, Z and radial gradients.

    Thin wrapper around :func:`map_boundary_to_interfaces` that uses interface
    ``inface`` of the SPECTRE object ``test`` as the mapping boundary.  The
    returned gradients are in map2disc's ``rho in [0, 1]`` coordinate; callers
    feeding SPECTRE's ``s in [-1, 1]`` coordinate must rescale them (see
    :func:`spectre.mesh_gen.generators.gen_mesh_external`).

    Args:
        xin: packed interface state vector.
        test: a live ``SPECTRE`` object.
        inface: interface index used as the mapping boundary.
        bcm_m: number of Fourier modes for the BCM solver.
        rho_arr: radial positions in map2disc's ``[0, 1]`` coordinate.
        th_arr: poloidal angles (SPECTRE's negated-theta convention).
        zeta_arr: toroidal angles.

    Returns:
        ``(R_arr, Z_arr, dRdrho_arr, dZdrho_arr)``, each of shape
        ``(len(rho_arr), len(zeta_arr), len(th_arr))``.
    """
    # get_inface_func returns the interface's theta -> (R, Z) curve at a fixed zeta,
    # doing the unpack_interfaces setup once per slice.
    return map_boundary_to_interfaces(
        lambda zeta: get_inface_func(xin, test, zeta, inface),
        rho_arr,
        th_arr,
        zeta_arr,
        bcm_m,
        with_grad=True,
    )
