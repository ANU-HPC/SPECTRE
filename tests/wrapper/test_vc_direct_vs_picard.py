# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Physics validation of the direct (lvcvacuum = 1) free-boundary solve.

At fixed interface geometry, the fixed point of the classic Picard iteration (update iBns by
virtual casing, re-solve, blend) and the direct Laplace-virtual-casing solve are solutions of the
same self-consistency equation, discretized independently (mainline ``casinggrid`` resolution
cascade vs the fixed dVC source grid). The converged plasma-induced normal-field harmonics on the
computational boundary and the vacuum vector potential must therefore agree up to quadrature error.
"""

import os.path

import numpy as np

from spectre import (
    SPECTRE,
    force_real,
    get_vec_pot_flat,
    get_xinit_specwrap,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


def _plasma_bnorm(test):
    return test.bnorm_from_plasma()


def test_direct_matches_converged_picard():
    # ------------------------------------------------------------------ Picard reference (run A)
    sp_picard = f"{curr_path}test_freeb_w7x.toml"
    test = SPECTRE(sp_picard, verbose=False)

    xinit = get_xinit_specwrap(test)
    force_real(xinit, test)

    # Fixture preparation, not an assertion: stopping at a pre-blend bnserr of 1e-6 leaves
    # the Picard reference within ~1e-6 of its fixed point -- two orders below the 3e-4
    # comparison band asserted below -- while saving the further full solves that 1e-9
    # would cost. The final blend then moves iBns by at most (1-gbnbld)*1e-6, so the field
    # in memory stays inside the same margin.
    bnserr = test.update_bns(mode="converge", tol=1e-6, max_sweeps=60)
    assert bnserr < 1e-6, f"Picard iteration did not converge: bnserr = {bnserr:.3e}"

    _, ofm_picard = _plasma_bnorm(test)
    ate_picard, aze_picard, _, _ = get_vec_pot_flat(test)
    remove_file(sp_picard + ".end")

    # -------------------------------------------------------------------- direct solve (run B)
    sp_direct = f"{curr_path}test_freeb_w7x_vc.toml"
    test = SPECTRE(sp_direct, verbose=False)

    force_real(xinit, test)

    _, ofm_direct = _plasma_bnorm(test)
    ate_direct, aze_direct, _, _ = get_vec_pot_flat(test)
    remove_file(sp_direct + ".end")

    # ------------------------------------------------------------------------------ comparison
    scale = np.max(np.abs(ofm_picard))
    assert scale > 0.0
    assert np.allclose(ofm_direct, ofm_picard, atol=3e-4 * scale, rtol=0.0)

    # the whole field solution (all volumes) must agree, not just the boundary harmonics
    a_scale = max(np.max(np.abs(ate_picard)), np.max(np.abs(aze_picard)))
    assert np.allclose(ate_direct, ate_picard, atol=3e-4 * a_scale, rtol=0.0)
    assert np.allclose(aze_direct, aze_picard, atol=3e-4 * a_scale, rtol=0.0)
