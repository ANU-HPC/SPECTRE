# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT
#
# Author: Christopher Berg Smiet <chris@bergtopfusion.com>

"""Tests for the flux-corrected interface placement behind fix2free_refresh_infaces.

Does NOT require simsopt — only map2disc (and not even that for _parse_mn_dict).

For an axisymmetric circular-torus boundary R = R0 + a*cos(theta),
Z = a*sin(theta), the vacuum toroidal flux of B_phi = G/R through the centered
disc of minor radius r has the closed form

    Psi(r) = 2*pi*G*(R0 - sqrt(R0**2 - r**2)),

and the conformal map of a circle is the identity scaling (rho = r/a), so the
rho**2 returned by _vacuum_flux_corrected_rho_sq for a target flux fraction f
must satisfy

    rho_sq = (R0**2 - (R0 - f*(R0 - sqrt(R0**2 - a**2)))**2) / a**2.
"""

import numpy as np
import pytest

from spectre.converters.fix2free import (
    _parse_mn_dict,
    _vacuum_flux_corrected_rho_sq,
)


def test_parse_mn_dict():
    """'(m, n)' keys parse into (m, n, value) tuples."""
    assert _parse_mn_dict({"(1, -2)": 0.5}) == [(1, -2, 0.5)]


@pytest.mark.map2disc
@pytest.mark.parametrize("R0, a", [(10.0, 1.0), (2.0, 1.0)])
def test_vacuum_flux_corrected_rho_sq_circular_torus(R0, a):
    """Flux-corrected placements match the analytic Psi(r) of B_phi = G/R.

    The vacuum toroidal field of a coil set carrying poloidal current G is
    B_phi = G/R (Ampere's law around the major axis).  Its flux through the
    centered disc of minor radius r in the poloidal plane of a circular torus
    of major radius R0 is

        Psi(r) = G * ∫∫_disc (1/R) dR dZ
               = G * ∫_{-r}^{r} 2*sqrt(r**2 - x**2) / (R0 + x) dx
               = 2*pi*G*(R0 - sqrt(R0**2 - r**2)),

    a standard contour/residue integral.  Two limits make the formula easy to
    sanity-check: for r << R0 it reduces to the uniform-field result
    Psi ≈ pi*r**2 * (G/R0), and it grows *slower* than r**2 because the outer
    part of the disc (R > R0) carries weaker field — exactly the effect the
    flux-corrected placement is meant to capture.

    A circular cross-section is also the one case where the conformal disc
    coordinate is trivial: the BCM map of a circle is the identity scaling, so
    the conformal radius is rho = r/a with no shape distortion.  Inverting
    Psi(rho*a) = f * Psi(a) for a target flux fraction f therefore gives the
    closed form

        rho_sq(f) = (R0**2 - (R0 - f*(R0 - sqrt(R0**2 - a**2)))**2) / a**2,

    which _vacuum_flux_corrected_rho_sq must reproduce.  Its numerics
    (Green's-theorem line integral ∮ ln(R) dZ on a discrete curve, profile
    interpolation over n_rho samples) converge well inside rtol=1e-3 at
    n_rho=200: the observed error is ~5e-5 at both aspect ratios tested here,
    R0/a = 10 (tokamak-like, nearly uniform field) and R0/a = 2 (tight torus,
    strong 1/R variation across the disc).
    """
    pytest.importorskip("map2disc")

    comp_rbc = {"(0, 0)": R0, "(1, 0)": a}
    comp_zbs = {"(1, 0)": a}
    fracs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # Axisymmetric boundary: all zeta cuts are identical, one BCM solve suffices.
    rho_sq = _vacuum_flux_corrected_rho_sq(
        comp_rbc, comp_zbs, nfp=1, target_fracs=fracs, n_rho=200, nzeta=1
    )

    psi_boundary = R0 - np.sqrt(R0**2 - a**2)
    expected = (R0**2 - (R0 - fracs * psi_boundary) ** 2) / a**2
    np.testing.assert_allclose(rho_sq, expected, rtol=1e-3)

    # Because Psi(rho) grows slower than rho**2 (the outer shells sit at larger
    # R where B_phi = G/R is weaker), reaching a given flux fraction requires a
    # *larger* radius than the uniform-field guess rho = sqrt(f): every
    # interface moves outward, toward the comp boundary.  This is the feature's
    # raison d'etre — without it the vacuum region comes out too large.
    assert np.all(rho_sq > fracs)
    # The flux profile is strictly monotone in rho, so inverting it must keep
    # the placements ordered: interfaces stay nested, no crossings.
    assert np.all(np.diff(rho_sq) > 0)
