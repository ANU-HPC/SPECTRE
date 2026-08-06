# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Tests for fixed-point finding and tracking functions."""

import os.path
from pathlib import Path

import numpy as np
import pytest
import tomlkit

from spectre import SPECTRE, SPECTREout, force_real, get_xinit_specwrap
from spectre.fixed_points import FixedPointClassifier, FixedPointTracker, root_fixed_point
from spectre.fixed_points.classification import classify_point

curr_path = os.path.dirname(__file__) + "/"

# qa_7vol_test.toml is a shared wrapper fixture (used by ~11 tests); reference it in place rather
# than duplicating it. test_freeb_w7x.toml lives beside this file (via curr_path).
QA_TOML = Path(__file__).resolve().parents[1] / "wrapper" / "qa_7vol_test.toml"


@pytest.fixture(scope="module")
def qa():
    """QA 7-volume vacuum test case."""
    return str(QA_TOML), 12.546318091772426, 0.0


@pytest.fixture(scope="module")
def w7x():
    """Free-boundary W7-X test case."""
    return f"{curr_path}test_freeb_w7x.toml", 5.908497385241743, 0.0


@pytest.fixture(scope="module")
def qa_specout():
    """Solved SPECTREout for the QA 7-volume test case."""
    obj = SPECTRE(str(QA_TOML))
    force_real(get_xinit_specwrap(obj), obj)
    return obj.create_spectreout()


@pytest.mark.parametrize("fixture", ["qa", "w7x"])
def test_find_axis(fixture, request):
    """Magnetic axis R, Z are recovered within 1e-7 of the reference values."""
    fixture = request.getfixturevalue(fixture)
    fname, r_expected, z_expected = fixture

    out_obj = SPECTRE(fname)
    xin = get_xinit_specwrap(out_obj)
    force_real(xin, out_obj)
    analysis_obj = out_obj.create_spectreout()

    r, z, s, theta = analysis_obj.find_axis(guess=[-0.9, 0.0], max_repeats=0, verbose=True)

    tol = 1e-7
    assert abs(r - r_expected) < tol
    assert abs(z - z_expected) < tol


def test_fp_tracker_axis_only_mode(qa_specout):
    """FixedPointTracker with m=None must track axis without raising."""
    tk = FixedPointTracker(m=None, volume=None, n_steps=1)
    report = tk.step(qa_specout)

    # Axis must have been located.
    assert not np.isnan(tk.axis_rz[0, 0]), "axis R should not be NaN in axis-only mode"
    # no_fp should be True because no island tracking was requested.
    assert report.no_fp is True
    assert report.reason == "axis-only mode"


def test_axis_is_a_fixed_point_over_five_field_periods(qa_specout):
    """The located axis returns to itself after five field periods.

    The magnetic axis is a period-1 fixed point of the field-period return map, so tracing it
    for several field periods must close back onto the start point. This pins that the axis found
    by :meth:`find_axis` is genuinely a fixed point, not a near miss.
    """
    _, _, s_ax, th_ax = qa_specout.find_axis(guess=[-0.9, 0.0], max_repeats=0)

    r0, z0 = qa_specout.get_rz_coords(0, s_ax, th_ax, 0.0)
    r0, z0 = float(np.ravel(r0)[0]), float(np.ravel(z0)[0])

    # num_transits = q + 1 records the last puncture at q field periods (q = 5 here).
    _, RZ, _, success = qa_specout.trace_one_fieldline(
        np.array([s_ax, th_ax]), 0, num_phi_planes=1, num_transits=6
    )
    assert success, "field line left the volume before closing"

    drift = np.hypot(RZ[0][-1] - r0, RZ[1][-1] - z0)
    assert drift < 1e-6, f"axis drifted {drift:.2e} m after 5 field periods (should be ~0)"


def test_axis_classifies_as_o_point(qa_specout):
    """The magnetic axis is elliptic: disc < 0 and Greene residue in (0, 1)."""
    _, _, s_ax, th_ax = qa_specout.find_axis(guess=[-0.9, 0.0], max_repeats=0)

    # The axis is a period-1 fixed point.
    clf = FixedPointClassifier(qa_specout, vol=0, m=1)
    g = clf.greene(s_ax, th_ax)

    assert g is not None, "tangent-map integration should succeed at the axis"
    assert abs(g["det"] - 1.0) < 1e-6, (
        f"return map is area-preserving; |det-1|={abs(g['det'] - 1):.1e}"
    )
    assert g["disc"] < 0.0 and g["kind"] == "O", f"axis should be elliptic, got {g}"
    assert 0.0 < g["residue"] < 1.0, f"elliptic residue must be in (0, 1), got {g['residue']:.3e}"


def test_tangent_map_matches_finite_differences(qa_specout):
    """Analytic return-map Jacobian J agrees with central finite differences at the axis.

    The variational integration returns ``J = d(s, theta)_end / d(s, theta)_start`` over one
    field period. Evaluated at the axis (a bounded fixed point, so neighbouring field lines stay
    in the volume) it must match a central-difference Jacobian of the same forward map, and must
    predict the map's response to a specific step: ``f(x + dx) = f(x) + J.dx`` to second order.

    The point is fixed (the found axis), not random, so the test is deterministic.
    """
    _, _, s_ax, th_ax = qa_specout.find_axis(guess=[-0.9, 0.0], max_repeats=0)

    # Warm and grab the single-field-line tracer (for the analytic tangent map J), then drive
    # everything at tight tolerances so the finite differences are not integrator-limited.
    qa_specout.trace_one_fieldline(np.array([-0.9, 0.0]), 0, num_phi_planes=1, num_transits=2)
    tracer = qa_specout._tracer_single
    tg = dict(method="DOP853", rtol=1e-12, atol=1e-12)

    def forward(s, th):
        # Plain forward field-line map over one field period (num_transits = q+1 with q = 1); the
        # last puncture (s, theta) is the map image. No tangent computed here.
        sth, _, _, ok = qa_specout.trace_one_fieldline(
            np.array([s, th]), 0, num_phi_planes=1, num_transits=2, **tg
        )
        assert ok, "forward trace left the volume"
        return np.array([sth[0][-1], sth[1][-1]])

    _, J, ok = tracer.trace_tangent_single(np.array([s_ax, th_ax]), 0, num_transits=1, **tg)
    assert ok

    h = 1e-6
    j_fd = np.column_stack(
        [
            (forward(s_ax + h, th_ax) - forward(s_ax - h, th_ax)) / (2 * h),
            (forward(s_ax, th_ax + h) - forward(s_ax, th_ax - h)) / (2 * h),
        ]
    )
    assert np.allclose(J, j_fd, rtol=1e-4, atol=1e-5), (
        f"Jacobian not good linearization of the field line map: analytic J=\n{J}\n"
        f"finite-diff J=\n{j_fd}"
    )

    # f(x + dx) = f(x) + J.dx, to second order in |dx|.
    dx = np.array([2e-6, -3e-6])
    predicted = forward(s_ax, th_ax) + J @ dx
    actual = forward(s_ax + dx[0], th_ax + dx[1])
    assert np.linalg.norm(actual - predicted) < 1e-9, (
        f"Jacobian not good linearization of the field line map: "
        f"jac pred {predicted}, actual map {actual}"
    )


def test_tracker_constructs_in_all_modes():
    """The tracker accepts each documented calling mode without error.

    Construction is cheap and needs no equilibrium; it pins that the axis-only, guess-seeded and
    self-discovery islands modes all build (island stepping itself is covered by the NCSX island
    tests below).
    """
    # Axis-only.
    FixedPointTracker(m=None, volume=None, n_steps=1)
    # Island, seeded by an explicit (R, Z) guess.
    FixedPointTracker(m=5, volume=4, n_steps=1, guess=[1.47, 0.0], n=3)
    # Island, self-discovery.
    FixedPointTracker(m=5, volume=4, n_steps=1, discover=True, n=3)
    # Problem-dependent tuning overrides are accepted.
    FixedPointTracker(m=5, volume=4, n_steps=1, discover=True, n=3,
                      root_ftol=1e-7, disc_n_s=151, geom_ceiling=0.5)  # fmt: skip


# --------------------------------------------------------------------------------------------
# NCSX island tests.
#
# Free-boundary NCSX (a converged h5 provided by the reviewer, with the real-space grid cache
# stripped -> 0.36 MB) is loaded directly with no re-solve, since free-boundary solving is not
# available on this branch. It carries a 6/11 island chain in volume 2 with both O- and X-points,
# so it exercises discovery + the O/X classifier on a real low-resolution island in a
# free-boundary (vacuum-coupled) equilibrium.
# --------------------------------------------------------------------------------------------

NCSX_FREEB_H5 = Path(__file__).resolve().parents[1] / "test_files" / "freeb_ncsx.h5"
NCSX_FREEB_VOL = 2
NCSX_FREEB_P, NCSX_FREEB_Q = 6, 11  # iota = 6/11


@pytest.fixture(scope="module")
def ncsx_freeb_specout():
    """Converged free-boundary NCSX equilibrium, loaded from the committed h5 (no re-solve)."""
    return SPECTREout(str(NCSX_FREEB_H5))


def test_freeb_island_chain_distinguishes_o_and_x_points(ncsx_freeb_specout):
    """The 6/11 chain in volume 2 is found and its O- and X-points are correctly distinguished.

    Discovery enumerates the chain members; the analytic-tangent-map classifier must return both
    elliptic O-points (Greene discriminant < 0, residue in (0, 1)) and hyperbolic X-points
    (discriminant > 0), each with an area-preserving return map (``|det - 1|`` ~ 0). This pins the
    O/X distinction on a real free-boundary island.
    """
    out = ncsx_freeb_specout
    tk = FixedPointTracker(
        m=NCSX_FREEB_Q,
        volume=NCSX_FREEB_VOL,
        n_steps=1,
        discover=True,
        n=NCSX_FREEB_P,
    )
    theta_grid = np.linspace(0.0, 2.0 * np.pi, tk.disc_n_theta, endpoint=False)
    members, _, stats = tk.find_fp(
        out, NCSX_FREEB_VOL, -0.99, 0.99, theta_grid,
        return_all=True, require_n=NCSX_FREEB_P,
    )  # fmt: skip
    assert members, f"6/11 chain should be found in volume {NCSX_FREEB_VOL}, got stats={stats}"

    clf = FixedPointClassifier(out, NCSX_FREEB_VOL, NCSX_FREEB_Q)
    by_kind: dict = {}
    for m in members:
        c = classify_point(clf, (m["s"], m["theta"]))
        assert c is not None, "each discovered chain member should classify"
        assert abs(c["det"] - 1.0) < 1e-6, (
            f"return map must be area-preserving, |det-1|={abs(c['det'] - 1):.1e}"
        )
        by_kind.setdefault(c["kind"], []).append(c)

    assert "o" in by_kind and "x" in by_kind, (
        f"a 6/11 chain has both O- and X-points; classifier found only {sorted(by_kind)}"
    )
    o, x = by_kind["o"][0], by_kind["x"][0]
    assert o["disc"] < 0.0 < x["disc"], (
        f"O-point must have Greene discriminant < 0 and X-point > 0; "
        f"O disc={o['disc']:.2e}, X disc={x['disc']:.2e}"
    )
    assert 0.0 < o["residue"] < 1.0, f"O-point residue must be in (0, 1), got {o['residue']:.3e}"


def test_scan_discovery_agrees_with_primed_finder(ncsx_freeb_specout):
    """Grid-scan discovery and a primed root-find converge to the same island O-point.

    The full grid scan discovers the 3/5 chain in volume 3; a root-find primed from a seed offset
    in theta from a discovered member must converge back to the same ``(R, Z)``. This confirms the
    scan lands on a genuine fixed point, agreeing with the primed finder, not on a grid artifact.
    """
    out = ncsx_freeb_specout
    vol, p, q = 3, 3, 5  # the period-5 chain in the free-boundary case (faster than 6/11)
    tk = FixedPointTracker(m=q, volume=vol, n_steps=1, discover=True, n=p)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, tk.disc_n_theta, endpoint=False)
    members, _, stats = tk.find_fp(out, vol, -0.99, 0.99, theta_grid, return_all=True, require_n=p)
    assert members, f"3/5 chain should be found in volume {vol}, got stats={stats}"

    m = members[0]  # a discovered chain member (grid scan + polish)
    r_scan, z_scan = out.get_rz_coords(vol, m["s"], m["theta"], 0.0)
    r_scan, z_scan = float(np.ravel(r_scan)[0]), float(np.ravel(z_scan)[0])

    # Prime a root-find from a seed offset in theta. The chain spacing is 2*pi/5 ~ 1.26 rad, so a
    # fixed 0.05 rad offset stays well inside the same island (deterministic, no random seed).
    # maxfev=100 matches the budget find_fp gives its internal polish, so it reaches ftol.
    primed = root_fixed_point(
        out, vol, q + 1, (m["s"], m["theta"] + 0.05),
        method="DOP853", rtol=1e-9, atol=1e-9, maxfev=100,
    )  # fmt: skip
    assert primed.converged, "primed root-find should converge from the offset seed"
    assert np.hypot(primed.R - r_scan, primed.Z - z_scan) < 1e-3, (
        f"scan O-point ({r_scan:.4f}, {z_scan:.4f}) and primed "
        f"({primed.R:.4f}, {primed.Z:.4f}) should agree"
    )


def test_find_fixed_point_discovers_the_chain(ncsx_freeb_specout):
    """``SPECTREout.find_fixed_point`` with no guess discovers the 3/5 chain in volume 3.

    The no-guess path mirrors the in-loop default (no seed -> discover by scanning): it must
    return at least one converged member of the period-5 chain, the members must be distinct
    points, and the result must be cached on the object.
    """
    out = ncsx_freeb_specout
    fps = out.find_fixed_point(3, 5, return_all=True)
    assert fps, "the 3/5 chain should be discovered with no guess"
    for fp in fps:
        assert fp.converged, "return_all must only contain converged points"
        assert fp.residual < 1e-6, f"converged residual must pass ftol, got {fp.residual:.1e}"
    rz = np.array([[fp.R, fp.Z] for fp in fps])
    if len(rz) > 1:
        d = np.hypot(rz[:, None, 0] - rz[None, :, 0], rz[:, None, 1] - rz[None, :, 1])
        assert d[~np.eye(len(rz), dtype=bool)].min() > 1e-3, "members must be deduplicated"
    assert out.fixed_points[(3, 5)] is fps, "the result should be cached on the object"


def test_find_fixed_point_guess_reconverges_to_the_same_member(ncsx_freeb_specout):
    """An ``(R, Z)`` guess offset from a discovered member re-converges to the same point.

    A fixed 5 mm offset in Z stays well inside the island (deterministic, no random seed).
    The offset seed needs the tracker-grade polish budget (the default ``maxfev=30`` is a
    fail-fast setting for scan minima), matching the primed-finder test above.
    """
    out = ncsx_freeb_specout
    best = out.find_fixed_point(3, 5)
    assert best.converged, "discovery should converge on the 3/5 chain"
    fp = out.find_fixed_point(
        3, 5, guess_rz=(best.R, best.Z + 0.005),
        method="DOP853", rtol=1e-9, atol=1e-9, maxfev=100,
    )  # fmt: skip
    assert fp.converged, "the guess path should converge from the offset (R, Z) seed"
    assert np.hypot(fp.R - best.R, fp.Z - best.Z) < 1e-3, (
        f"guess path ({fp.R:.4f}, {fp.Z:.4f}) should land on the discovered member "
        f"({best.R:.4f}, {best.Z:.4f})"
    )


def test_find_fixed_point_axis_agrees_with_find_axis(qa_specout):
    """``find_fixed_point(0, 1)`` seeded near the axis agrees with ``find_axis`` to 1e-6.

    The magnetic axis is the period-1 fixed point of the return map, so the general finder and
    the dedicated axis finder must locate the same point. The seed sits 1 mm off the axis: the
    exact axis is the degenerate s -> -1 point of volume 0, where the (R, Z) -> (s, theta)
    inversion is ill-conditioned.
    """
    rax, zax, _, _ = qa_specout.find_axis(guess=[-0.9, 0.0])
    fp = qa_specout.find_fixed_point(0, 1, guess_rz=(rax + 1e-3, zax))
    assert fp.converged, "the near-axis seed should converge as a period-1 fixed point"
    assert abs(fp.R - rax) < 1e-6 and abs(fp.Z - zax) < 1e-6, (
        f"find_fixed_point axis ({fp.R:.6f}, {fp.Z:.6f}) should match "
        f"find_axis ({rax:.6f}, {zax:.6f})"
    )


# --------------------------------------------------------------------------------------------
# Fixed-boundary NCSX (generated from wout_li383 via simsvmec2spectre): a 3/5 island chain that
# sits near the volume-3/volume-4 interface. Moving that interface (tflux[3]) walks the resonant
# surface between the two volumes, so the volume-pinned tracker records the island being lost in
# the vacated volume (NaN rows, no_fp) and appearing in the neighbour. Solved with force_real
# (no optimization), so the whole thing stays cheap.
# --------------------------------------------------------------------------------------------

NCSX_FIXB_TOML = Path(__file__).resolve().parents[1] / "test_files" / "ncsx_fixb.toml"
FIXB_P, FIXB_Q = 3, 5  # iota = 3/5
TFLUX3_ISLAND_IN_VOL4 = 0.58  # the 3/5 surface sits in volume 4 (volume 3 empty)
TFLUX3_ISLAND_IN_VOL3 = 0.68  # the 3/5 surface sits in volume 3 (volume 4 empty)


def _solve_ncsx_fixb(tmp_path, tflux3):
    """Solve the fixb NCSX case with the vol-3/4 interface at ``tflux3``; return the SPECTREout."""
    doc = tomlkit.parse(NCSX_FIXB_TOML.read_text())
    tflux = list(doc["physics"]["tflux"])
    tflux[3] = tflux3
    doc["physics"]["tflux"] = tflux
    case = tmp_path / f"ncsx_fixb_t3_{tflux3:.2f}.toml"
    case.write_text(tomlkit.dumps(doc))
    obj = SPECTRE(str(case))
    force_real(get_xinit_specwrap(obj), obj)
    return obj.create_spectreout()


def test_island_migrates_between_volumes_and_is_refound(tmp_path):
    """As tflux walks the 3/5 surface across the vol-3/4 interface, the island is lost in the
    vacated volume (NaN, no_fp) and re-found in the neighbour.

    Two equilibria are solved: one with the 3/5 surface in volume 4, one with it in volume 3. A
    volume-4 tracker finds the island then loses it (its per-step ``island_rz`` row becomes NaN and
    ``no_fp`` is set); a volume-3 tracker finds nothing then re-discovers it. This exercises the
    no_fp / NaN bookkeeping and the volume-pinned tracker's behaviour when a chain migrates across
    an interface.
    """
    seq = [
        _solve_ncsx_fixb(tmp_path, TFLUX3_ISLAND_IN_VOL4),  # island in volume 4
        _solve_ncsx_fixb(tmp_path, TFLUX3_ISLAND_IN_VOL3),  # island in volume 3
    ]

    # Tracker pinned to volume 4: island present, then lost when it migrates out.
    tk4 = FixedPointTracker(m=FIXB_Q, volume=4, n_steps=2, discover=True, n=FIXB_P)
    for specout in seq:
        tk4.step(specout)
    assert not tk4.no_fp[0], "the 3/5 island should be found in volume 4 in the first equilibrium"
    assert tk4.no_fp[1], "the island should be lost from volume 4 once it migrates to volume 3"
    assert np.isnan(tk4.island_rz[1, 0]), "the lost step must leave a NaN island position"
    assert tk4.kind[1] == "", "no O/X kind should be recorded for the lost step"

    # Tracker pinned to volume 3: nothing, then the island appears (NaN -> found).
    tk3 = FixedPointTracker(m=FIXB_Q, volume=3, n_steps=2, discover=True, n=FIXB_P)
    for specout in seq:
        tk3.step(specout)
    assert tk3.no_fp[0], "volume 3 has no island in the first equilibrium"
    assert np.isnan(tk3.island_rz[0, 0]), "the empty step must leave a NaN island position"
    assert not tk3.no_fp[1], "the island should be re-found in volume 3 after it migrates"
    assert abs(abs(tk3.iota[1]) - FIXB_P / FIXB_Q) < 5e-3, (
        f"the re-found chain should have iota = 3/5, got {tk3.iota[1]:.4f}"
    )
