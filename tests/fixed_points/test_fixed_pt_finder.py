# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT
"""Tests for fixed-point finding and tracking functions."""

import os.path
from pathlib import Path

import numpy as np
import pytest

from spectre import SPECTRE, SPECTREout, force_real, get_xinit_specwrap
from spectre.file_io import read_input_parameters_from_toml
from spectre.fixed_points import FixedPointClassifier, FixedPointTracker, root_fixed_point
from spectre.fixed_points.classification import classify_point

curr_path = os.path.dirname(__file__) + "/"

# qa_3vol_light.toml is a shared wrapper fixture (used by ~11 tests); reference it in place rather
# than duplicating it. test_freeb_w7x.toml lives beside this file (via curr_path).
QA_TOML = Path(__file__).resolve().parents[1] / "wrapper" / "qa_3vol_light.toml"


@pytest.fixture(scope="module")
def qa():
    """QA 3-volume vacuum test case."""
    return str(QA_TOML), 12.546318091772426, 0.0


@pytest.fixture(scope="module")
def w7x():
    """Free-boundary W7-X test case."""
    return f"{curr_path}test_freeb_w7x.toml", 5.908497385241743, 0.0


@pytest.fixture(scope="module")
def qa_specout():
    """Solved SPECTREout for the QA 3-volume test case."""
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


def test_axis_is_an_elliptic_period_one_fixed_point(qa_specout):
    """The magnetic axis is an elliptic period-1 fixed point, and both finders agree on it.

    Three properties of the one axis, asserted off a single :meth:`find_axis` call:

    1. **It is a fixed point** -- tracing it for five field periods closes back onto the start
       point, so what ``find_axis`` returned is genuinely periodic, not a near miss.
    2. **It is elliptic** -- the analytic tangent map gives ``disc < 0`` and a Greene residue in
       (0, 1), on an area-preserving return map (``|det-1| ~ 0``).
    3. **The general finder agrees** -- the axis is the period-1 fixed point of the return map,
       so ``find_fixed_point(0, 1)`` must locate the same point to 1e-6. Its seed sits 1 mm off
       the axis: the exact axis is the degenerate ``s -> -1`` point of volume 0, where the
       ``(R, Z) -> (s, theta)`` inversion is ill-conditioned.
    """
    rax, zax, s_ax, th_ax = qa_specout.find_axis(guess=[-0.9, 0.0], max_repeats=0)

    # 1. Closes on itself. num_transits = q + 1 records the last puncture at q field periods.
    r0, z0 = qa_specout.get_rz_coords(0, s_ax, th_ax, 0.0)
    r0, z0 = float(np.ravel(r0)[0]), float(np.ravel(z0)[0])
    _, RZ, _, success = qa_specout.trace_one_fieldline(
        np.array([s_ax, th_ax]), 0, num_phi_planes=1, num_transits=6
    )
    assert success, "field line left the volume before closing"
    drift = np.hypot(RZ[0][-1] - r0, RZ[1][-1] - z0)
    assert drift < 1e-6, f"axis drifted {drift:.2e} m after 5 field periods (should be ~0)"

    # 2. Elliptic.
    clf = FixedPointClassifier(qa_specout, vol=0, m=1)
    g = clf.greene(s_ax, th_ax)
    assert g is not None, "tangent-map integration should succeed at the axis"
    assert abs(g["det"] - 1.0) < 1e-6, (
        f"return map is area-preserving; |det-1|={abs(g['det'] - 1):.1e}"
    )
    assert g["disc"] < 0.0 and g["kind"] == "O", f"axis should be elliptic, got {g}"
    assert 0.0 < g["residue"] < 1.0, f"elliptic residue must be in (0, 1), got {g['residue']:.3e}"

    # 3. The general finder lands on the same point.
    fp = qa_specout.find_fixed_point(0, 1, guess_rz=(rax + 1e-3, zax))
    assert fp.converged, "the near-axis seed should converge as a period-1 fixed point"
    assert abs(fp.R - rax) < 1e-6 and abs(fp.Z - zax) < 1e-6, (
        f"find_fixed_point axis ({fp.R:.6f}, {fp.Z:.6f}) should match "
        f"find_axis ({rax:.6f}, {zax:.6f})"
    )


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


@pytest.mark.parametrize(
    ("s_found", "residual", "solver_success", "expected", "why"),
    [
        (-0.485, 1.7e-15, False, True, "machine-precision point the solver calls a failure"),
        (0.5, 1.7e-15, True, True, "ordinary converged point"),
        (0.5, 1e-2, True, False, "residual above ftol"),
        (1.5, 1e-15, True, False, "left the volume"),
        (-0.9995, 1e-15, True, True, "axis root outside the 0.999 margin but inside the volume"),
    ],
)
def test_root_fixed_point_acceptance_rule(
    qa_specout, monkeypatch, s_found, residual, solver_success, expected, why
):
    """``converged`` is exactly "|s| < 1 and residual below ftol" -- nothing else vetoes.

    ``hybr`` reports failure whenever its step-size test trips, which is what happens when the
    guess is *already* at machine precision: a member of the NCSX 6/11 chain converges to a
    residual of 1.7e-15 with ``solver_success`` False, and it is a genuine fixed point. Vetoing
    on that flag would silently discard the best-converged points, so it is diagnostic only.

    ``S_CLAMP`` does not veto either. The acceptance bound is the volume, ``|s| < 1``; the
    tighter 0.999 is a margin, reported as ``in_volume`` and asserted separately below. A
    magnetic axis converges at ``|s| > 0.999`` in a large fraction of equilibria, and gating on
    the margin rejected those as "not converged" while their residual was at machine precision.

    Driven by faking :func:`scipy.optimize.root` rather than an equilibrium, so each case pins
    one branch of the rule deterministically and for free.
    """
    from spectre.fixed_points import primitives as prim

    class _Sol:
        def __init__(self):
            self.x = np.array([s_found, 0.1])
            self.fun = np.array([residual, 0.0])
            self.success = solver_success
            self.nfev = 7

    monkeypatch.setattr(prim, "root", lambda *a, **k: _Sol())
    res = prim.root_fixed_point(qa_specout, 1, 2, (0.5, 0.1))

    assert res.converged is expected, f"{why}: converged should be {expected}"
    assert res.solver_success is solver_success, "the solver flag must still be reported"
    assert res.in_volume is (abs(s_found) < prim.S_CLAMP)


def test_refine_returns_none_for_a_non_converged_solve(qa_specout, monkeypatch):
    """``refine``/``classify_point`` propagate the verdict: a failed solve is never classified.

    ``refine`` owns no acceptance rule of its own -- it forwards ``root_fixed_point``'s -- so an
    unconverged result must come back as ``None`` rather than receiving a scientifically
    meaningless Greene classification.
    """
    from spectre.fixed_points import FixedPointResult
    from spectre.fixed_points import classification as cls_mod

    clf = FixedPointClassifier(qa_specout, 1, 2)
    failed = FixedPointResult(
        R=12.5, Z=0.0, s=0.5, theta=0.1, residual=1e-2, converged=False, in_volume=True
    )
    monkeypatch.setattr(cls_mod, "root_fixed_point", lambda *a, **k: failed)

    assert clf.refine((0.5, 0.1)) is None, "refine must reject a non-converged solve"
    assert classify_point(clf, (0.5, 0.1)) is None, (
        "a point whose refinement failed must not be classified"
    )


def test_tracker_constructs_in_all_modes():
    """The tracker accepts each documented calling mode without error.

    Construction is cheap and needs no equilibrium; it pins that the axis-only, guess-seeded and
    self-discovery island modes all build. Island stepping itself is covered by the NCSX island
    tests in this file, and the TOML-level configuration rules by
    ``tests/file_io/test_minimization_parameters.py``.
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


def test_find_fixed_point_discovers_the_chain_and_reconverges_to_a_member(ncsx_freeb_specout):
    """Discovery finds the 3/5 chain, and both seeded paths return to the same member.

    Three properties of the same chain (volume 3 of the free-boundary case), asserted off a
    single discovery scan because each seeded path only needs a member to start from:

    1. **Discovery** -- ``find_fixed_point`` with no guess mirrors the in-loop default and must
       return converged, deduplicated members, recording the result on the object.
    2. **Primed primitive** -- ``root_fixed_point`` seeded 0.05 rad off a discovered member in
       theta must come back to it. The chain spacing is 2*pi/5 ~ 1.26 rad, so the offset stays
       well inside the same island (deterministic, no random seed). This is what shows the scan
       landed on a genuine fixed point rather than a grid artifact.
    3. **Public guess path** -- ``find_fixed_point(guess_rz=...)`` seeded 5 mm off in Z must do
       the same. Both seeded calls need the tracker-grade polish budget; the ``maxfev=30``
       default is a fail-fast setting for scan minima.
    """
    out = ncsx_freeb_specout
    vol, q = 3, 5  # the period-5 chain in the free-boundary case (faster than 6/11)

    # 1. Discovery.
    fps = out.find_fixed_point(vol, q, return_all=True)
    assert fps, "the 3/5 chain should be discovered with no guess"
    for fp in fps:
        assert fp.converged, "return_all must only contain converged points"
        assert fp.residual < 1e-6, f"converged residual must pass ftol, got {fp.residual:.1e}"
    rz = np.array([[fp.R, fp.Z] for fp in fps])
    if len(rz) > 1:
        d = np.hypot(rz[:, None, 0] - rz[None, :, 0], rz[:, None, 1] - rz[None, :, 1])
        assert d[~np.eye(len(rz), dtype=bool)].min() > 1e-3, "members must be deduplicated"
    assert out.fixed_points[(vol, q)] is fps, "the result should be recorded on the object"

    best = fps[0]

    # 2. Primed primitive, seeded off in theta.
    primed = root_fixed_point(
        out, vol, q + 1, (best.s, best.theta + 0.05),
        method="DOP853", rtol=1e-9, atol=1e-9, maxfev=100,
    )  # fmt: skip
    assert primed.converged, "primed root-find should converge from the offset seed"
    assert np.hypot(primed.R - best.R, primed.Z - best.Z) < 1e-3, (
        f"discovered member ({best.R:.4f}, {best.Z:.4f}) and primed "
        f"({primed.R:.4f}, {primed.Z:.4f}) should agree"
    )

    # 3. Public guess path, seeded off in Z.
    fp = out.find_fixed_point(
        vol, q, guess_rz=(best.R, best.Z + 0.005),
        method="DOP853", rtol=1e-9, atol=1e-9, maxfev=100,
    )  # fmt: skip
    assert fp.converged, "the guess path should converge from the offset (R, Z) seed"
    assert np.hypot(fp.R - best.R, fp.Z - best.Z) < 1e-3, (
        f"guess path ({fp.R:.4f}, {fp.Z:.4f}) should land on the discovered member "
        f"({best.R:.4f}, {best.Z:.4f})"
    )


# --------------------------------------------------------------------------------------------
# Fixed-boundary NCSX (generated from wout_li383 via simsvmec2spectre): a 3/5 island chain that
# sits near the volume-3/volume-4 interface. Moving that interface (tflux[3]) walks the resonant
# surface between the two volumes, so the volume-pinned tracker records the island being lost in
# the vacated volume (NaN rows, no_fp) and appearing in the neighbour. Nothing here is a
# converged equilibrium: each case is a single unconverged Beltrami field evaluated at the
# prescribed interfaces (force_real, no minimization), which is all the tracker consumes and
# keeps the whole thing cheap.
# --------------------------------------------------------------------------------------------

NCSX_FIXB_TOML = Path(__file__).resolve().parents[1] / "test_files" / "ncsx_fixb.toml"
FIXB_P, FIXB_Q = 3, 5  # iota = 3/5
TFLUX3_ISLAND_IN_VOL4 = 0.58  # the 3/5 surface sits in volume 4 (volume 3 empty)
TFLUX3_ISLAND_IN_VOL3 = 0.68  # the 3/5 surface sits in volume 3 (volume 4 empty)


def _ncsx_field_with_interface_at(tflux3):
    """Compute the unconverged NCSX field with the vol-3/4 interface at ``tflux3``.

    Only the interface positions change; ``force_real`` evaluates the Beltrami field once at
    those interfaces. This is not an equilibrium solve -- it is the cheapest way to produce a
    field in which the 3/5 surface sits in a chosen volume.
    """
    params = read_input_parameters_from_toml(str(NCSX_FIXB_TOML))
    tflux = list(params.physics.tflux)
    tflux[3] = tflux3
    params = params.model_copy(
        update={"physics": params.physics.model_copy(update={"tflux": tflux})}
    )
    obj = SPECTRE.from_input_params(params, name=f"ncsx_fixb_t3_{tflux3:.2f}")
    force_real(get_xinit_specwrap(obj), obj)
    return SPECTREout.from_spectre(obj)


def test_island_migrates_between_volumes_and_is_refound():
    """A step that moves the island across the vol-3/4 interface: it is lost in the vacated
    volume (NaN, no_fp) and re-found in the neighbour.

    Two unconverged fields are computed, differing only in tflux[3]: one with the 3/5 surface in
    volume 4, one with it in volume 3. Feeding them to the tracker in sequence simulates an
    optimisation step that moves the island artificially; the tracker is what is under test. A
    volume-4 tracker finds the island then loses it (its per-step ``island_rz`` row becomes NaN and
    ``no_fp`` is set); a volume-3 tracker finds nothing then re-discovers it. This exercises the
    no_fp / NaN bookkeeping and the volume-pinned tracker's behaviour when a chain migrates across
    an interface.
    """
    seq = [
        _ncsx_field_with_interface_at(TFLUX3_ISLAND_IN_VOL4),  # island in volume 4
        _ncsx_field_with_interface_at(TFLUX3_ISLAND_IN_VOL3),  # island in volume 3
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
