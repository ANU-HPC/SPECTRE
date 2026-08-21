# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Closed-form ground truth for the direct (lfreebsolver = 1) free-boundary Beltrami solve.

``current_ring.toml`` is the simplest free-boundary configuration that still produces a
non-trivial vacuum field: a single plasma volume shaped as a large-aspect-ratio circular
torus (R0 = 10, a = 0.1) carrying a net toroidal current mu0*I = ``curtor``, inside a
computational boundary twenty times larger, with no external coil field (``vns`` = 0).

Everything the vacuum region sees is then known analytically:

* the plasma is a current ring.  At aspect ratio 100 its exterior field is, to O((a/d)^2),
  the field of a circular filament of current I at R = R0, i.e. the textbook complete
  elliptic integral solution -- see e.g. Simpson et al., NASA/TM-2001-210326, or Jackson
  section 5.5.  This is the poloidal part (B_R, B_Z);
* the poloidal linking current ``curpol`` -- the only current the vacuum region itself
  constrains -- contributes the exact 1/R toroidal field B_phi = curpol / (2 pi R).  The
  plasma's own poloidal currents are confined inside the plasma boundary and, like a
  toroidal solenoid, produce no field outside it.

Nothing here is fitted: both reference fields follow from the two input scalars ``curtor``
and ``curpol``.  A sign, normalisation, or virtual-casing self-consistency defect shows up
at order unity.

The field is checked in two places, which probe different parts of the solve:

* *inside* the vacuum annulus, at scattered (s, theta, zeta), where the answer is the
  Beltrami solution reconstructed from the Chebyshev radial basis;
* *on the computational boundary* (s = +1), which is where the Laplace-virtual-casing
  operator imposes the plasma-induced normal field.  This is the sharpest check of the
  direct solve: the entire free-boundary coupling lives on this surface.

The geometry is axisymmetric but the run is not: ``nfp`` = 1 with ``ntor`` = 2 keeps the
full 3D mode list, the zeta FFTs and the zeta quadrature of the virtual-casing matrix in
play, so the test also pins that the 3D machinery adds nothing to an axisymmetric answer.

The residual error is the finite thickness of the ring, not the solver.  Measured at the
input resolution: 3.3e-3 of the peak poloidal field inside, 1.1e-3 on the computational
boundary.  Doubling the minor radius to a = 0.2 multiplies the boundary error by 4.0 --
the O((a/d)^2) signature of the filament limit -- while raising ``mpol`` to 12 leaves the
poloidal field unchanged to five digits and raising ``vcnt``/``vcnz`` to 192 changes
nothing at all, so neither the Fourier truncation nor the virtual-casing quadrature is
contributing.  ``lrad`` = 24 in the vacuum region is the one resolution that has to be
earned: the annulus spans a factor 20 in minor radius and the field goes like 1/r across
it, so 16 is not converged (5.8e-3) while 32 buys nothing over 24.  The two tolerances
below differ because the boundary sits 20 minor radii from the ring and the innermost
sampled point only 7.6.

The matching Picard test lives at the end of this module: the classic free-boundary
iteration (``lfreebsolver`` = 0, damped Picard on ``iBns`` through ``update_bns``) must
converge to this same closed-form field, pinning both free-boundary paths against ground
truth rather than only against each other as ``test_vc_direct_vs_picard`` does.
"""

import os.path

import numpy as np
import pytest
from scipy.special import ellipe, ellipk

from spectre import SPECTRE, force_real, get_xinit_specwrap, remove_file
from spectre.free_boundary import bnorm_from_plasma, vacuum_field_at

curr_path = os.path.dirname(__file__) + "/"

# interior sample: scattered over the annulus, clear of both radial ends
_N_INSIDE = 48
_S_INSIDE = (-0.3, 0.9)
# computational-boundary sample: a (theta, zeta) grid at s = +1, so the inboard side, the
# outboard side and every zeta plane are all covered
_N_TETA_CB, _N_ZETA_CB = 12, 5


def _loop_field(mu0_i, r_loop, rr, zz):
    """Field of a circular current filament, in cylindrical components.

    The filament lies in the Z = 0 plane at radius ``r_loop`` and carries current I in the
    +phi direction, with ``mu0_i`` = mu0 * I.  ``scipy``'s ``ellipk``/``ellipe`` take the
    parameter m = k^2.

    Args:
        mu0_i: mu0 times the loop current [T m].
        r_loop: major radius of the filament [m].
        rr: cylindrical R of the evaluation points [m].
        zz: cylindrical Z of the evaluation points [m].

    Returns:
        tuple[ndarray, ndarray]: (B_R, B_Z) at the evaluation points [T].
    """
    rr = np.asarray(rr, dtype=float)
    zz = np.asarray(zz, dtype=float)

    far = (r_loop + rr) ** 2 + zz**2  # (a + R)^2 + Z^2
    near = (r_loop - rr) ** 2 + zz**2  # (a - R)^2 + Z^2
    m = 4.0 * r_loop * rr / far
    kk, ee = ellipk(m), ellipe(m)

    pref = mu0_i / (2.0 * np.pi)
    b_r = pref * zz / (rr * np.sqrt(far)) * (-kk + ee * (r_loop**2 + rr**2 + zz**2) / near)
    b_z = pref / np.sqrt(far) * (kk + ee * (r_loop**2 - rr**2 - zz**2) / near)
    return b_r, b_z


@pytest.fixture(scope="module")
def current_ring():
    """Solve the current-ring case once; return the wrapper and the reference scalars."""
    sp = f"{curr_path}current_ring.toml"
    test = SPECTRE(sp, verbose=False)

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)  # direct Beltrami solve at the prescribed geometry
    # the Lconstraint = 3 linking-current constraints are a linear Newton step on the
    # enclosed fluxes; a second pass removes the residual of the initial flux guess
    force_real(xinit, test)
    remove_file(sp + ".end")

    nvol = int(test.input_list_mod.nvol)
    return {
        "test": test,
        "curtor": float(test.input_list_mod.curtor),
        "curpol": float(test.input_list_mod.curpol),
        # the ring's current centroid is the (m, n) = (0, 0) major radius of the plasma
        # boundary, the first entry of the mode list
        "r_loop": float(test.allglobal_mod.irbc[0, nvol]),
    }


def _field_and_reference(case, s_pts, t_pts, z_pts):
    """Evaluate SPECTRE's vacuum field and the closed-form field at the same points.

    Args:
        case: the ``current_ring`` fixture payload.
        s_pts: radial coordinates in the vacuum volume, -1 (plasma boundary) to +1 (CB).
        t_pts: poloidal angles.
        z_pts: toroidal angles.

    Returns:
        tuple[ndarray, ndarray]: SPECTRE's and the reference (B_R, B_phi, B_Z), each
        of shape (3, npts).
    """
    test = case["test"]
    _, bcyl, rz = vacuum_field_at(test, s_pts, t_pts, z_pts)

    b_r, b_z = _loop_field(case["curtor"], case["r_loop"], rz[0], rz[1])
    b_phi = case["curpol"] / (2.0 * np.pi * rz[0])
    return np.asarray(bcyl), np.stack([b_r, b_phi, b_z])


def _assert_matches(bcyl, ref, pol_atol):
    """Compare against the reference: the ring field poloidally, curpol/(2 pi R) toroidally."""
    pol_scale = np.max(np.hypot(ref[0], ref[2]))
    assert pol_scale > 0.0

    assert np.allclose(bcyl[0], ref[0], atol=pol_atol * pol_scale, rtol=0.0)
    assert np.allclose(bcyl[2], ref[2], atol=pol_atol * pol_scale, rtol=0.0)

    # The 1/R toroidal field is exact: the poloidal linking current is its only source, and
    # it comes out at 2e-9 of the scale -- an m-truncation residual that vanishes with mpol
    # (7e-14 at mpol = 12).  The threshold is loose against platform-dependent roundoff;
    # anything physical would land orders of magnitude above it.
    tor_scale = np.max(np.abs(ref[1]))
    assert tor_scale > 0.0
    assert np.allclose(bcyl[1], ref[1], atol=1e-7 * tor_scale, rtol=0.0)


def test_vacuum_field_matches_current_loop(current_ring):
    """Inside the vacuum annulus, the solution is the field of the current ring."""
    rng = np.random.default_rng(7)
    s_pts = rng.uniform(*_S_INSIDE, _N_INSIDE)
    t_pts = rng.uniform(0.0, 2.0 * np.pi, _N_INSIDE)
    z_pts = rng.uniform(0.0, 2.0 * np.pi, _N_INSIDE)

    bcyl, ref = _field_and_reference(current_ring, s_pts, t_pts, z_pts)
    # finite-ring-thickness limited: 3.3e-3 observed, 2x margin
    _assert_matches(bcyl, ref, pol_atol=7e-3)


def test_computational_boundary_field_matches_current_loop(current_ring):
    """On the computational boundary, where the virtual-casing operator sets B.n."""
    teta = np.linspace(0.0, 2.0 * np.pi, _N_TETA_CB, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, _N_ZETA_CB, endpoint=False)
    tt, zz = np.meshgrid(teta, zeta, indexing="ij")
    t_pts, z_pts = tt.ravel(), zz.ravel()
    s_pts = np.ones_like(t_pts)  # s = +1 is exactly the computational boundary

    bcyl, ref = _field_and_reference(current_ring, s_pts, t_pts, z_pts)
    # the CB sits 20 minor radii from the ring, so the filament limit is much cleaner
    # here: 1.1e-3 observed, 2x margin
    _assert_matches(bcyl, ref, pol_atol=2.5e-3)


def test_no_nonaxisymmetric_content(current_ring):
    """The exact solution is axisymmetric; the 3D machinery must not manufacture n /= 0."""
    test = current_ring["test"]
    mn = int(test.allglobal_mod.mn)
    efm, ofm = bnorm_from_plasma(test)

    nonaxi = np.asarray(test.allglobal_mod.in_[:mn]) != 0
    scale = np.max(np.abs(ofm))
    assert scale > 0.0
    # 34 of the 43 modes carry n /= 0; they come out at 5e-17 of the scale, i.e. exactly
    # zero to roundoff, and the stellarator-symmetric partner efm likewise
    assert np.max(np.abs(ofm[nonaxi])) < 1e-12 * scale
    assert np.max(np.abs(efm)) < 1e-12 * scale


# ------------------------------------------------------------------------- Picard companion
# The tests below construct a second SPECTRE instance, which re-initializes the process-
# global Fortran state; they must stay after every test using the direct `current_ring`
# fixture above (the fixture dependency of `current_ring_picard` captures the direct-side
# data before the reset).
#
# Tolerances pinned by a one-knob-at-a-time convergence study on this deck (source grid
# vcnt/vcnz in 64..512, vacuum Lrad 16/32, damping in [0, 0.8]; 2026-08-06). Measured: the converged
# Picard field reproduces the direct solve's oracle errors digit for digit
# (3.294e-3 inside, 1.117e-3 on the boundary — the finite ring thickness, exactly as the
# direct test documents), the converged ofm spectrum agrees with the direct operator to
# 1.9e-12 of its scale at vcnt = vcnz = 256, the damped map contracts at c(lambda) ~
# lambda (undamped eigenvalue ~ 2.5e-3 at this geometry; lambda = 0.5 reaches 1e-8 in
# ~20 sweeps, floor below 1e-11 at sweep 31), and the non-axisymmetric content sits at
# 1e-16 of scale.

_PICARD_BNSERR_TOL = 1e-8  # ~20 sweeps at damping 0.5; three decades above the floor
_PICARD_MAX_SWEEPS = 40  # the study's lambda=0.5 run floors at sweep 31
_PICARD_DAMPING = 0.5  # mid contraction band; every damping in [0, 0.8] converges here
_PICARD_POL_ATOL_INSIDE = 7e-3  # study: 3.294e-3 observed (= direct); 2x margin
_PICARD_POL_ATOL_CB = 2.5e-3  # study: 1.117e-3 observed (= direct); 2x margin
_PICARD_OFM_VS_DIRECT = 1e-6  # study: 1.9e-12 converged; ~1e-8 abs at the test's stop tol


@pytest.fixture(scope="module")
def current_ring_picard(current_ring):
    """Converge the Picard ring from bns = 0; capture the direct-side spectrum first.

    Depending on the direct ``current_ring`` fixture guarantees the direct solve exists
    before this constructor resets the Fortran state, so its spectrum can be captured for
    the cross-path comparison.
    """
    direct = current_ring["test"]
    _, ofm_direct = bnorm_from_plasma(direct)

    sp = f"{curr_path}current_ring_picard.toml"
    test = SPECTRE(sp, verbose=False)
    bnserr = test.update_bns(
        mode="initialize",
        damping=_PICARD_DAMPING,
        tol=_PICARD_BNSERR_TOL,
        max_sweeps=_PICARD_MAX_SWEEPS,
    )
    remove_file(sp + ".end")

    nvol = int(test.input_list_mod.nvol)
    return {
        "test": test,
        "bnserr": bnserr,
        "ofm_direct": ofm_direct,
        "curtor": float(test.input_list_mod.curtor),
        "curpol": float(test.input_list_mod.curpol),
        "r_loop": float(test.allglobal_mod.irbc[0, nvol]),
    }


def test_picard_converges(current_ring_picard):
    """The damped Picard iteration reaches its fixed point from bns = 0."""
    assert current_ring_picard["bnserr"] < _PICARD_BNSERR_TOL


def test_picard_vacuum_field_matches_current_loop(current_ring_picard):
    """Inside the vacuum annulus, the converged Picard solution is the ring field."""
    rng = np.random.default_rng(7)
    s_pts = rng.uniform(*_S_INSIDE, _N_INSIDE)
    t_pts = rng.uniform(0.0, 2.0 * np.pi, _N_INSIDE)
    z_pts = rng.uniform(0.0, 2.0 * np.pi, _N_INSIDE)

    bcyl, ref = _field_and_reference(current_ring_picard, s_pts, t_pts, z_pts)
    _assert_matches(bcyl, ref, pol_atol=_PICARD_POL_ATOL_INSIDE)


def test_picard_computational_boundary_field_matches_current_loop(current_ring_picard):
    """On the computational boundary, where the converged iBns imposes the coupling."""
    teta = np.linspace(0.0, 2.0 * np.pi, _N_TETA_CB, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, _N_ZETA_CB, endpoint=False)
    tt, zz = np.meshgrid(teta, zeta, indexing="ij")
    t_pts, z_pts = tt.ravel(), zz.ravel()
    s_pts = np.ones_like(t_pts)

    bcyl, ref = _field_and_reference(current_ring_picard, s_pts, t_pts, z_pts)
    _assert_matches(bcyl, ref, pol_atol=_PICARD_POL_ATOL_CB)


def test_picard_no_nonaxisymmetric_content(current_ring_picard):
    """The Picard sweeps must not manufacture non-axisymmetric iBns content."""
    test = current_ring_picard["test"]
    mn = int(test.allglobal_mod.mn)

    ibns = np.asarray(test.allglobal_mod.ibns)[:mn]
    nonaxi = np.asarray(test.allglobal_mod.in_[:mn]) != 0
    scale = np.max(np.abs(ibns))
    assert scale > 0.0
    assert np.max(np.abs(ibns[nonaxi])) < 1e-10 * scale


def test_picard_bns_spectrum_matches_direct(current_ring_picard):
    """Both free-boundary paths extrapolate the same plasma spectrum at this geometry."""
    test = current_ring_picard["test"]
    efm, ofm = bnorm_from_plasma(test)

    ofm_direct = current_ring_picard["ofm_direct"]
    scale = np.max(np.abs(ofm_direct))
    assert scale > 0.0
    assert np.allclose(ofm, ofm_direct, atol=_PICARD_OFM_VS_DIRECT * scale, rtol=0.0)
