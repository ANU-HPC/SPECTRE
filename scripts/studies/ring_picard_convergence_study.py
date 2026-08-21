#!/usr/bin/env -S python -u

# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""One-knob-at-a-time convergence study for the Picard current-ring validation.

Pins the tolerances of the ``lvcvacuum = 0`` current-ring test (the Picard companion to
``test_vc_current_ring.py``) with evidence rather than a single passing run. Every case
starts from ``tests/wrapper/current_ring_picard.toml``, changes exactly one knob, converges
``iBns`` with damped Picard sweeps, and measures

* the pre-blend fixed-point residual ``bnserr`` per sweep (contraction rate and floor);
* the vacuum field against the closed-form current-loop oracle, sampled inside the vacuum
  annulus and on the computational boundary (same samples as the direct test);
* the plasma-induced normal-field spectrum ``ofm`` against the direct (lvcvacuum = 1) run
  at the identical geometry;
* the non-axisymmetric leakage (the exact solution is axisymmetric).

Knob ladders (baseline in brackets):

* damping        0.0, 0.3, [0.5], 0.666, 0.8   -- contraction band of the damped map
* sweeps         bnserr history at the baseline -- when does the residual floor
* vcnt = vcnz    64, 128, [256], 512            -- virtual-casing source resolution
* lrad (vacuum)  16, [24], 32                   -- radial Chebyshev basis of the annulus

Run from the repository root (all ranks measure, rank 0 reports):

    mpirun -n 1 .venv/bin/python scripts/studies/ring_picard_convergence_study.py

Results append to ``runs/ring_picard_study/results.csv``; the console table carries the
same numbers. Tolerances for the test are then chosen with a 2x margin over the observed
plateau, mirroring the direct test's practice.
"""

import csv
import shutil
import sys
from pathlib import Path

import numpy as np
import tomlkit

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "wrapper"))

from test_vc_current_ring import _loop_field  # noqa: E402  (the analytic oracle)

from spectre import SPECTRE, force_real, get_xinit_specwrap  # noqa: E402

BASE_DECK = REPO / "tests" / "wrapper" / "current_ring_picard.toml"
DIRECT_DECK = REPO / "tests" / "wrapper" / "current_ring.toml"
OUTDIR = REPO / "runs" / "ring_picard_study"

# The samples of the direct test, reused verbatim so the errors are comparable.
N_INSIDE = 48
S_INSIDE = (-0.3, 0.9)
N_TETA_CB, N_ZETA_CB = 12, 5

MAX_SWEEPS = 40
TOL_SWEEP = 1e-11  # run essentially to the floor; the study reports where the floor is


def _sample_points():
    """Return the interior scatter and the computational-boundary grid of the direct test."""
    rng = np.random.default_rng(7)
    s_in = rng.uniform(*S_INSIDE, N_INSIDE)
    t_in = rng.uniform(0.0, 2.0 * np.pi, N_INSIDE)
    z_in = rng.uniform(0.0, 2.0 * np.pi, N_INSIDE)

    teta = np.linspace(0.0, 2.0 * np.pi, N_TETA_CB, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, N_ZETA_CB, endpoint=False)
    tt, zz = np.meshgrid(teta, zeta, indexing="ij")
    s_cb = np.ones(tt.size)
    return (s_in, t_in, z_in), (s_cb, tt.ravel(), zz.ravel())


def _field_error(test, curtor, curpol, r_loop, pts):
    """Max poloidal-field error against the loop oracle, normalized to the peak field."""
    _, bcyl, rz = test.vacuum_field_at(*pts)

    b_r, b_z = _loop_field(curtor, r_loop, rz[0], rz[1])
    b_phi = curpol / (2.0 * np.pi * rz[0])
    pol_scale = np.max(np.hypot(b_r, b_z))
    err_pol = max(np.max(np.abs(bcyl[0] - b_r)), np.max(np.abs(bcyl[2] - b_z))) / pol_scale
    err_tor = np.max(np.abs(bcyl[1] - b_phi)) / np.max(np.abs(b_phi))
    return err_pol, err_tor


def _plasma_ofm(test):
    """Non-mutating plasma-induced normal-field harmonics (outer face)."""
    return test.bnorm_from_plasma()


def _case_deck(workdir, name, **overrides):
    """Write a one-knob variant of the base Picard deck; returns its path."""
    doc = tomlkit.parse(BASE_DECK.read_text())
    for dotted, value in overrides.items():
        group, key = dotted.split(".")
        doc[group][key] = value
    path = workdir / f"{name}.toml"
    path.write_text(tomlkit.dumps(doc))
    return str(path)


def _run_picard_case(deck, damping):
    """Bootstrap + damped sweeps to the floor; returns the instance and diagnostics."""
    test = SPECTRE(deck, verbose=False)

    # Pure plasma-side bootstrap (no converge tail), then explicit sweeps so the whole
    # bnserr history is recorded.
    history = [test.update_bns(mode="initialize", damping=damping, max_sweeps=0)]
    for _ in range(MAX_SWEEPS):
        bnserr = test.update_bns(mode="refresh", damping=damping)
        history.append(bnserr)
        if bnserr < TOL_SWEEP:
            break
        if len(history) >= 3 and history[-1] > history[-2] > history[-3]:
            break  # past the floor: quadrature noise, not contraction

    curtor = float(test.input_list_mod.curtor)
    curpol = float(test.input_list_mod.curpol)
    nvol = int(test.input_list_mod.nvol)
    r_loop = float(test.allglobal_mod.irbc[0, nvol])
    return test, history, curtor, curpol, r_loop


def main():
    """Run the ladder of one-knob cases and report the evidence table."""
    rank0 = True
    try:
        from mpi4py import MPI

        rank0 = MPI.COMM_WORLD.Get_rank() == 0
    except ImportError:
        pass

    OUTDIR.mkdir(parents=True, exist_ok=True)
    workdir = OUTDIR / "decks"
    workdir.mkdir(exist_ok=True)
    pts_in, pts_cb = _sample_points()

    # ------------------------------------------------------------------ direct reference
    shutil.copy(DIRECT_DECK, workdir / "direct.toml")
    direct = SPECTRE(str(workdir / "direct.toml"), verbose=False)
    xinit = get_xinit_specwrap(direct)
    force_real(xinit, direct)
    force_real(xinit, direct)  # second pass settles the linear flux Newton
    _, ofm_direct = _plasma_ofm(direct)
    d_curtor = float(direct.input_list_mod.curtor)
    d_curpol = float(direct.input_list_mod.curpol)
    d_rloop = float(direct.allglobal_mod.irbc[0, int(direct.input_list_mod.nvol)])
    d_in = _field_error(direct, d_curtor, d_curpol, d_rloop, pts_in)
    d_cb = _field_error(direct, d_curtor, d_curpol, d_rloop, pts_cb)

    rows = [
        dict(
            case="direct-reference",
            damping=np.nan,
            sweeps=0,
            bnserr=np.nan,
            err_pol_inside=d_in[0],
            err_pol_cb=d_cb[0],
            err_tor=max(d_in[1], d_cb[1]),
            dofm_vs_direct=0.0,
            nonaxi=np.nan,
        )
    ]

    # -------------------------------------------------------------------- knob ladders
    cases = [("baseline", {}, 0.5)]
    cases += [(f"damping-{lam}", {}, lam) for lam in (0.0, 0.3, 0.666, 0.8)]
    cases += [(f"vc-{n}", {"global.vcnt": n, "global.vcnz": n}, 0.5) for n in (64, 128, 512)]
    cases += [(f"lradvac-{lr}", {"physics.lrad": [8, lr]}, 0.5) for lr in (16, 32)]

    ofm_scale = np.max(np.abs(ofm_direct))
    for name, overrides, damping in cases:
        deck = _case_deck(workdir, name, **overrides)
        test, history, curtor, curpol, r_loop = _run_picard_case(deck, damping)

        efm, ofm = _plasma_ofm(test)
        nonaxi_mask = np.asarray(test.allglobal_mod.in_[: ofm.size]) != 0
        err_in = _field_error(test, curtor, curpol, r_loop, pts_in)
        err_cb = _field_error(test, curtor, curpol, r_loop, pts_cb)

        rows.append(
            dict(
                case=name,
                damping=damping,
                sweeps=len(history) - 1,
                bnserr=history[-1],
                err_pol_inside=err_in[0],
                err_pol_cb=err_cb[0],
                err_tor=max(err_in[1], err_cb[1]),
                dofm_vs_direct=float(np.max(np.abs(ofm - ofm_direct))) / ofm_scale,
                nonaxi=float(np.max(np.abs(ofm[nonaxi_mask]))) / ofm_scale,
            )
        )
        if rank0:
            hist = "  ".join(f"{b:.2e}" for b in history)
            print(f"\n[{name}] bnserr per sweep: {hist}")

    if rank0:
        header = list(rows[0].keys())
        with open(OUTDIR / "results.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)

        print("\n" + "  ".join(f"{h:>16}" for h in header))
        for row in rows:
            print(
                "  ".join(
                    f"{row[h]:>16}" if isinstance(row[h], str) else f"{row[h]:>16.3e}"
                    for h in header
                )
            )
        print(f"\nresults written to {OUTDIR / 'results.csv'}")


if __name__ == "__main__":
    main()
