# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from fractions import Fraction

import numpy as np
from scipy.optimize import root


def _return_map(sth, obj, vol=0, tracer_num_transits=2):
    """Return-map residual: (R,Z) at phi=0 after tracer_num_transits circuits, minus the start (R,Z)."""
    s, th = sth

    large = 1.0e2

    if s > 1.0 or s < -1:
        return np.array([large, large])

    sth, RZ, iota, flag = obj.trace_one_fieldline(
        np.array([s, th]),
        vol,
        num_phi_planes=1,
        num_transits=tracer_num_transits,
    )

    if flag:
        return np.array([large, large])

    r_start, z_start = obj.get_rz_coords(vol, s, th, 0.0)

    return np.array([RZ[0][-1] - r_start, RZ[1][-1] - z_start], dtype=float)


def classify_residue(
    obj,
    vol,
    r0,
    z0,
    tracer_num_transits,
    *,
    h_list=(1e-3, 5e-4, 1e-4, 5e-5),
    det_tol=0.02,
    agree_tol=5e-3,
    parabolic_tol=1e-4,
):
    """Classify a fixed point as O / X / parabolic via Greene's residue R = (2 - tr)/4.

    Builds the symplectic (R,Z) tangent map by central FD of the return map over the h_list
    ladder; the step with |det - 1| smallest classifies. 0 < R < 1 is elliptic (O), R < 0 or
    R > 1 is hyperbolic (X), R near 0 or 1 is parabolic. agree_tol is unused (kept for API
    compatibility).
    """
    _ = agree_tol
    rows_by_h = {}
    rows = []

    for h in h_list:
        if h in rows_by_h:
            continue

        # Symplectic tangent map J = d(RZ_image)/d(RZ_start) by central FD in (R,Z).
        cols = []
        for dR, dZ in ((h, 0.0), (-h, 0.0), (0.0, h), (0.0, -h)):
            s, th = obj.get_st_coords(vol, r0 + dR, z0 + dZ, 0.0)
            disp = _return_map([s, th], obj, vol, tracer_num_transits)
            rs, zs = obj.get_rz_coords(vol, s, th, 0.0)
            cols.append(np.array([rs + disp[0], zs + disp[1]]))
        j = np.column_stack([(cols[0] - cols[1]) / (2.0 * h), (cols[2] - cols[3]) / (2.0 * h)])
        tr, det = np.trace(j), np.linalg.det(j)

        # Greene residue R = (2 − tr)/4 → O / X / parabolic.
        residue = (2.0 - tr) / 4.0
        if not np.isfinite(residue):
            raise RuntimeError(f"non-finite Greene residue: {residue}")
        if abs(residue) <= parabolic_tol or abs(residue - 1.0) <= parabolic_tol:
            kind_h, refl_h = "parabolic", False
        elif 0.0 < residue < 1.0:
            kind_h, refl_h = "o", False
        elif residue < 0.0:
            kind_h, refl_h = "x", False
        else:
            kind_h, refl_h = "x", True

        det_clean = abs(det - 1.0) < det_tol
        row = {
            "h": h,
            "trace": tr,
            "det": det,
            "residue": residue,
            "det_clean": det_clean,
            "kind_h": kind_h,
            "reflection_hyperbolic": refl_h,
            "det_err": abs(det - 1.0),
        }
        rows_by_h[h] = row
        rows.append(row)

    finite_rows = [row for row in rows if np.isfinite(row["det"])]
    if not finite_rows:
        raise RuntimeError("classify_residue: no finite FD steps")

    best = min(finite_rows, key=lambda row: row["det_err"])
    n_det_clean = sum(1 for row in rows if row["det_clean"])
    det_clean_kinds = {row["kind_h"] for row in rows if row["det_clean"]}
    kind = best["kind_h"]
    if n_det_clean == 0:
        stable = False
    else:
        stable = det_clean_kinds == {kind}

    return {
        "kind": kind,
        "residue": float(best["residue"]),
        "trace": float(best["trace"]),
        "det": float(best["det"]),
        "reflection_hyperbolic": best["reflection_hyperbolic"],
        "stable": stable,
        "h_used": [best["h"]],
        "rows": rows,
        "meta": {
            "n_used": 1,
            "n_det_clean": n_det_clean,
            "h_best": best["h"],
            "det_err_best": float(best["det_err"]),
            "single_reason": None,
            "undersampled": False,
        },
    }


def _random_guess(small=1e-7):
    r_rand = np.random.rand(2)
    return [
        (r_rand[0] * (1 - 2 * small) + small) * 2.0 - 1.0,
        r_rand[1] * 2.0 * np.pi,
    ]


def find_fixed_point(
    obj,
    vol=0,
    tracer_num_transits=2,
    guess=None,
    ftol=1e-6,
    xtol=1e-8,
    verbose=False,
):
    """Root-find (s, theta) with ||return map|| < ftol; returns (R, Z, s, theta)."""
    x0 = [-0.9, 0.0] if guess is None else list(guess)
    fp = root(
        _return_map,
        x0=x0,
        args=(obj, vol, tracer_num_transits),
        tol=ftol,
        options=dict(xtol=xtol),
    )
    fun_norm = np.linalg.norm(fp.fun)
    if fun_norm >= ftol:
        raise RuntimeError(
            f"find_fixed_point failed for tracer_num_transits={tracer_num_transits}: "
            f"||fun||={fun_norm:.3e} >= ftol={ftol:.1e}"
        )

    sax, thax = float(fp.x[0]), float(fp.x[1])
    rax, zax = obj.get_rz_coords(vol, sax, thax, 0.0)

    if verbose:
        print(
            f"find_fixed_point vol={vol} tracer_num_transits={tracer_num_transits}: "
            f"||fun||={fun_norm:.3e}\n"
            f"fp {fp}\nrax {rax:5.6f}  zax {zax:5.6f}  s {sax:5.6f}  th {thax:5.6f}"
        )
    return rax, zax, sax, thax


def find_island(
    obj,
    vol,
    iota,
    max_repeats=0,
    guess=None,
    verbose=False,
    point_type=None,
):
    """Find an island-chain fixed point at iota = p/q (Fraction in lowest terms); returns (R, Z, s, theta).

    Uses q+1 transits at phi=0; pass guess as (s, theta) and use vol != 0 (vol 0 is the axis).
    With point_type=None returns the first converged root. With point_type 'o' or 'x', only that
    kind is accepted (classified by Greene's residue); on mismatch the guess is steered by
    theta*+-pi/mpol, then randomly, for up to max_repeats retries.
    """
    if not isinstance(iota, Fraction):
        raise TypeError("iota must be fractions.Fraction, e.g. Fraction(12, 13)")
    if point_type is not None and point_type not in ("o", "x"):
        raise ValueError(f"point_type must be None, 'o', or 'x'; got {point_type!r}")

    mpol = iota.denominator
    # mpol+1 transits put the endpoint one step past orbit closure, so the residual is non-zero
    # except at the true fixed point (a well-conditioned root-finding target).
    tracer_num_transits = mpol + 1

    if point_type is None:
        return find_fixed_point(
            obj,
            vol=vol,
            tracer_num_transits=tracer_num_transits,
            guess=guess,
            verbose=verbose,
        )

    user_guess = [-0.9, 0.0] if guess is None else list(guess)
    steer_idx = 0
    rejected_guess = None
    last_clf = None

    for it in range(max_repeats + 1):
        # Next (s, θ): user guess → θ*±π/mpol around the rejected root → random.
        if steer_idx == 0:
            guess_try = list(user_guess)
        elif rejected_guess is not None and mpol > 0 and steer_idx in (1, 2):
            s_star, th_star = rejected_guess
            dth = (np.pi / mpol) if steer_idx == 1 else (-np.pi / mpol)
            guess_try = [s_star, (th_star + dth) % (2.0 * np.pi)]
        else:
            guess_try = _random_guess()
        try:
            rax, zax, sax, thax = find_fixed_point(
                obj,
                vol=vol,
                tracer_num_transits=tracer_num_transits,
                guess=guess_try,
                verbose=False,
            )
        except RuntimeError as exc:
            if it == max_repeats:
                raise RuntimeError(
                    f"find_island failed vol={vol} iota={iota}: "
                    f"no convergence after {max_repeats + 1} tries; last error: {exc}"
                ) from exc
            steer_idx = min(steer_idx + 1, 3)
            continue

        last_clf = classify_residue(obj, vol, rax, zax, tracer_num_transits)
        kind = last_clf["kind"]
        if kind == point_type:
            if verbose:
                want = "O-point" if point_type == "o" else "X-point"
                print(
                    f"find_island: found {kind}; want {want} | "
                    f"green_residue={last_clf['residue']:.4f} det={last_clf['det']:.4f} "
                    f"trace={last_clf['trace']:.4f}"
                )
            return rax, zax, sax, thax

        rejected_guess = (sax, thax)
        if verbose:
            want = "O-point" if point_type == "o" else "X-point"
            print(
                f"find_island: wrong type — found {kind} "
                f"(green_residue={last_clf['residue']:+.4f}, det={last_clf['det']:.4f}, "
                f"trace={last_clf['trace']:+.4f}), want {want} — "
                f"steered retry (steer_idx {min(steer_idx + 1, 3)})"
            )
        if it == max_repeats:
            raise RuntimeError(
                f"find_island failed vol={vol} iota={iota}: converged to "
                f"{kind} (green_residue={last_clf['residue']:.4f}, "
                f"det={last_clf['det']:.4f}) but "
                f"point_type={point_type!r} after {max_repeats + 1} tries"
            )
        steer_idx = min(steer_idx + 1, 3)

    raise RuntimeError(f"find_island failed vol={vol} iota={iota}")


def find_axis(obj, max_repeats=0, guess=None, plot=False, verbose=False):
    """Find the magnetic axis in volume 0 (``tracer_num_transits=2`` return map)."""
    user_guess = [-0.9, 0.0] if guess is None else list(guess)

    for it in range(max_repeats + 1):
        guess_try = user_guess if it == 0 else _random_guess()
        try:
            rax, zax, sax, thax = find_fixed_point(
                obj,
                vol=0,
                tracer_num_transits=2,
                guess=guess_try,
                verbose=verbose,
            )
            break
        except RuntimeError:
            if it == max_repeats:
                raise
    else:
        raise RuntimeError("find_axis failed")

    if plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        obj.plot_infaces(plot_axis=True, ax=ax)
        obj.plot_poincare(ax=ax)
        ax.plot(rax, zax, "bX", ms=2)

    return rax, zax, sax, thax
