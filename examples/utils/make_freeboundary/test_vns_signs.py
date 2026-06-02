"""Diagnostic script: compare initial forces for vns sign/index variants.

Tests whether a sign flip or n→-n or m→-m index flip in the vns Fourier
coefficients (or phiedge/curpol sign) reduces the initial force below the
baseline generated directly by fix2free.

Usage (run from the repo root):
    python examples/utils/make_freeboundary/test_vns_signs.py

Requires: mpi4py, simsopt, spectre installed in dev mode.
The W7-X VMEC input file must be present in the same directory as this script.
"""

from pathlib import Path

import numpy as np
from mpi4py import MPI

from spectre import SPECTRE
from spectre.converters import simsvmec2spectre
from spectre.converters.fix2free import (
    fix2free,
    params_to_simsopt_boundary,
)
from spectre.file_io.input_parameters import mmpol, mntor
from spectre.force_targets import force_real

HERE = Path(__file__).parent
comm = MPI.COMM_WORLD


# ---------------------------------------------------------------------------
# Variant transformations
# ---------------------------------------------------------------------------


def flip_vns_n(phys):
    """Swap vns rows n → -n: arr[mntor+n, :] ↔ arr[mntor-n, :] for n>0.

    Tests whether the FFT n-indexing in spec_fft_fast is reversed.
    After the swap, Fortran mode (m, n) picks up Vns(-n, m) = original (m, -n) coeff.
    """
    v = phys.vns.copy()
    for n in range(1, phys.ntor + 1):
        v[mntor + n, :], v[mntor - n, :] = v[mntor - n, :].copy(), v[mntor + n, :].copy()
    return v


def flip_vns_m(phys):
    """Swap vns columns m → -m: arr[:, mmpol+m] ↔ arr[:, mmpol-m] for m>0.

    For stellarator-symmetric vns (only m≥0 columns populated), this moves
    the m>0 content into the m<0 columns.  The Fortran antisymmetric combination
    iVns(m,n) = (Vns(n,m) - Vns(-n,-m)) * jj then reads the m>0 content from
    the -m position, giving iVns(m,n) = -Vns(n,-m) = -original_vns(m,-n), which
    is equivalent to a simultaneous n-flip and sign change for m>0 modes.
    Tests whether there is an m-index reversal in the converter or FFT.
    """
    v = phys.vns.copy()
    for m in range(1, phys.mpol + 1):
        v[:, mmpol + m], v[:, mmpol - m] = v[:, mmpol - m].copy(), v[:, mmpol + m].copy()
    return v


def flip_vns_sign(phys):
    """Negate all vns coefficients.

    Tests whether the overall sign convention (vns = -oddb vs vns = +oddb) is wrong.
    """
    return -phys.vns


def flip_vns_theta(phys):
    """Apply the theta-flip: vns_new(m, n) = -vns_old(m, -n) for m>0.

    When the boundary is parameterised with the opposite theta direction
    (Lchangeangle=true), sine modes transform as
      sin(mθ - nNfpζ) → -sin(mθ - (-n)Nfpζ),
    i.e., both a sign flip and an n-index flip for each m>0 mode.
    m=0 modes are unchanged (no theta dependence).
    """
    v = phys.vns.copy()
    for m in range(1, phys.mpol + 1):
        col = phys.vns[:, mmpol + m].copy()
        for n in range(-phys.ntor, phys.ntor + 1):
            v[mntor + n, mmpol + m] = -col[mntor - n]
    return v


def make_params(base_params, **updates):
    """Return a new InputParameters with physics fields updated."""
    phys = base_params.physics.model_copy(update=updates)
    return base_params.model_copy(update={"physics": phys})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def print_dominant_modes(vns_arr, ntor, mpol, n_modes=10):
    """Print the n_modes largest-amplitude vns modes."""
    modes = []
    for m in range(0, mpol + 1):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n < 0:
                continue
            val = float(vns_arr[mntor + n, mmpol + m])
            if val != 0.0:
                modes.append((abs(val), m, n, val))
    modes.sort(reverse=True)
    print(f"  Top {n_modes} vns modes by amplitude:")
    for amp, m, n, val in modes[:n_modes]:
        print(f"    vns({m:3d}, {n:3d}) = {val:+.6e}")


def measure_force(params, label):
    """Create a SPECTRE object and return RMS and max initial force."""
    s = SPECTRE.from_input_params(params, name=label, verbose=False, comm=comm)
    xin = s.xin
    f = force_real(xin, s)
    rms = float(np.sqrt(np.mean(f**2)))
    mx = float(np.max(np.abs(f)))
    return rms, mx


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    if comm.rank == 0:
        from simsopt.configs.zoo import get_data
        from simsopt.mhd import Vmec

        print("Building free-boundary params from VMEC + W7-X coils...")
        _, _, _, _, bs = get_data("w7x")

        vmec_file = HERE / "input.W7-X_standard_configuration"
        vmec = Vmec(str(vmec_file))
        fixed_params = simsvmec2spectre(vmec, nvol=1, mpol=11, ntor=11)

        plasma_boundary = params_to_simsopt_boundary(fixed_params, which="plasma")
        comp_boundary = plasma_boundary.copy()
        comp_boundary.extend_via_normal(-0.15)

        params = fix2free(fixed_params, bs, comp_boundary)
        phys = params.physics

        print(f"  phiedge = {phys.phiedge:.6f}")
        print(f"  curpol  = {phys.curpol:.6f}")
        print(f"  mpol={phys.mpol}, ntor={phys.ntor}, nfp={phys.nfp}")
        print()
        print("Dominant vns modes (baseline from fix2free):")
        print_dominant_modes(phys.vns, phys.ntor, phys.mpol)
        print()
    else:
        params = None

    params = comm.bcast(params, root=0)
    phys = params.physics

    nfp = phys.nfp
    two_pi = 2.0 * np.pi

    vns_theta = flip_vns_theta(phys)
    variants = [
        # --- sign / index variants (from previous run) ---
        ("original", params),
        ("vns_n_flip", make_params(params, vns=flip_vns_n(phys))),
        ("vns_m_flip", make_params(params, vns=flip_vns_m(phys))),
        ("vns_sign_flip", make_params(params, vns=flip_vns_sign(phys))),
        ("vns_theta_flip", make_params(params, vns=vns_theta)),
        ("vns_theta_flip_phiedge", make_params(params, vns=vns_theta, phiedge=-phys.phiedge)),
        ("vns_theta_flip_curpol", make_params(params, vns=vns_theta, curpol=-phys.curpol)),
        (
            "vns_theta_flip_phiedge_curpol",
            make_params(params, vns=vns_theta, phiedge=-phys.phiedge, curpol=-phys.curpol),
        ),
        ("phiedge_flip", make_params(params, phiedge=-phys.phiedge)),
        ("curpol_flip", make_params(params, curpol=-phys.curpol)),
        ("phiedge_curpol_flip", make_params(params, phiedge=-phys.phiedge, curpol=-phys.curpol)),
        (
            "vns_n_flip_phiedge_flip",
            make_params(params, vns=flip_vns_n(phys), phiedge=-phys.phiedge),
        ),
        ("vns_n_flip_curpol_flip", make_params(params, vns=flip_vns_n(phys), curpol=-phys.curpol)),
        (
            "vns_sign_phiedge_curpol",
            make_params(
                params, vns=flip_vns_sign(phys), phiedge=-phys.phiedge, curpol=-phys.curpol
            ),
        ),
        # --- magnitude / normalisation variants ---
        # Each test checks whether Python missed a factor in converting B·simsopt.normal()/(2π)²
        # to the SPECTRE Nxyz = dr/dθ × dr/dζ convention used in the virtual-casing integral.
        ("vns_x2pi", make_params(params, vns=phys.vns * two_pi)),
        ("vns_div2pi", make_params(params, vns=phys.vns / two_pi)),
        ("vns_x2pi2", make_params(params, vns=phys.vns * two_pi**2)),
        ("vns_div2pi2", make_params(params, vns=phys.vns / two_pi**2)),
        ("vns_xnfp", make_params(params, vns=phys.vns * nfp)),
        ("vns_divnfp", make_params(params, vns=phys.vns / nfp)),
        # virtualcasingfactor = -1/(4π): check matching scale for iVns
        ("vns_x4pi", make_params(params, vns=phys.vns * 4.0 * np.pi)),
        ("vns_div4pi", make_params(params, vns=phys.vns / (4.0 * np.pi))),
        # Nfp combined with 2π (if ζ is full-torus vs field-period angle)
        ("vns_x2pi_divnfp", make_params(params, vns=phys.vns * two_pi / nfp)),
        ("vns_xnfp_div2pi", make_params(params, vns=phys.vns * nfp / two_pi)),
    ]

    results = []
    for label, p in variants:
        if comm.rank == 0:
            print(f"Running: {label} ...", flush=True)
        rms, mx = measure_force(p, label)
        results.append((label, rms, mx))
        if comm.rank == 0:
            print(f"  RMS force = {rms:.4e}   max force = {mx:.4e}")

    if comm.rank == 0:
        print()
        print("=" * 65)
        print(f"{'Variant':<40} {'RMS force':>12} {'Max force':>12}")
        print("=" * 65)
        for label, rms, mx in sorted(results, key=lambda x: x[1]):
            print(f"{label:<40} {rms:>12.4e} {mx:>12.4e}")
        print("=" * 65)
        best = min(results, key=lambda x: x[1])
        print(f"\nLowest RMS force: '{best[0]}' at {best[1]:.4e}")


if __name__ == "__main__":
    main()
