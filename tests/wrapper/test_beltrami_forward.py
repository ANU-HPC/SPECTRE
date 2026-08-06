# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""End-to-end check that the matrix-free forward field solve matches the dense one.

With ``Lbeltramisolver`` enabled, ``solve_beltrami_system`` replaces the dense LU factor and
solve with a matrix-free preconditioned GMRES (op_bind_iter_sparse + op_solve_iter), while
the (mu, dpflux) constraint root-find and everything downstream is unchanged. A full
``force_real`` evaluation must therefore return the same real-space force as the default
dense solver, to the GMRES tolerance. ``Lbeltramisolver=2`` forces the iterative path on
every eligible volume so small test cases exercise it; ``Lbeltramisolver=1`` (THRESHOLD)
selects GMRES only in volumes with ``NAdof >= nadofgmres``. The free-boundary case runs
the vacuum volume — its RHS assembly, flux response and current constraint — through the
iterative path as well.
"""

import os.path

import numpy as np
import pytest
from helpers import spectre_freeb_w7x_reduced, spectre_with_lconstraint

from spectre import force_real, get_xinit_specwrap, remove_file

curr_path = os.path.dirname(__file__) + "/"


def _run(test, lbeltramisolver):
    """Run one forward solve and return the force plus constraint/backend observables."""
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False
    test.input_list_mod.lbeltramisolver = lbeltramisolver
    if lbeltramisolver:
        test.input_list_mod.nitergmres = 3000
        test.input_list_mod.ngmresrestart = 60
        test.input_list_mod.epsgmres = 1e-13
    xin = get_xinit_specwrap(test)
    f = np.asarray(force_real(xin, test))
    mvol = int(test.allglobal_mod.mvol)
    return {
        "force": f,
        "mu": np.array(test.input_list_mod.mu[:mvol], copy=True),
        "dpflux": np.array(test.allglobal_mod.dpflux[:mvol], copy=True),
        # (toroidal, poloidal) currents of the last volume, the vacuum volume when free-boundary
        "last_vol_currents": np.array(test.allglobal_mod.ditgpdxtp[:, 1, mvol - 1], copy=True),
        "gmres_its": np.array(test.allglobal_mod.beltrami_gmres_iterations[:mvol], copy=True),
        "gmres_status": np.array(test.allglobal_mod.beltrami_gmres_status[:mvol], copy=True),
    }


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_iterative_forward_matches_dense(fname):
    f_dense = _run(spectre_with_lconstraint(fname, 1), 0)["force"]
    f_iter = _run(spectre_with_lconstraint(fname, 1), 2)["force"]
    remove_file(fname + ".end")

    rel = np.linalg.norm(f_iter - f_dense) / max(np.linalg.norm(f_dense), 1e-300)
    assert rel < 1e-8, (
        f"{os.path.basename(fname)}: matrix-free forward solve force differs from the "
        f"dense solve by rel {rel:.3e}"
    )


def test_iterative_forward_freeboundary_matches_dense():
    """Free boundary: the vacuum volume runs iteratively and reproduces the dense solve.

    The vacuum volume has its own RHS assembly (strict vacuum, mu = 0), flux response,
    and (curtor, curpol) current constraint. ``Lbeltramisolver=2`` must run it — and the
    interior plasma volume — through GMRES (the coordinate-singularity volume stays
    dense), and forces, mu, dpflux and the vacuum currents must match the dense path.
    """
    dense = _run(spectre_freeb_w7x_reduced(1), 0)
    iterative = _run(spectre_freeb_w7x_reduced(1), 2)

    np.testing.assert_array_equal(dense["gmres_its"], 0)
    assert iterative["gmres_its"][1] > 0, "interior plasma volume did not run GMRES"
    assert iterative["gmres_its"][2] > 0, "vacuum volume did not run GMRES"
    np.testing.assert_array_equal(iterative["gmres_status"], 0)

    rel = np.linalg.norm(iterative["force"] - dense["force"]) / max(
        np.linalg.norm(dense["force"]), 1e-300
    )
    assert rel < 1e-8, f"free-boundary iterative force differs from dense by rel {rel:.3e}"
    np.testing.assert_allclose(iterative["mu"], dense["mu"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(iterative["dpflux"], dense["dpflux"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        iterative["last_vol_currents"], dense["last_vol_currents"], rtol=1e-8, atol=1e-10
    )


@pytest.mark.parametrize(
    "nadofgmres, expect_iterative",
    [
        (90, []),  # cutoff just above the largest volume: GMRES must never run
        (89, [3]),  # NAdof exactly at the cutoff is selected (>= comparison)
        (57, [2, 3]),  # volume 1 (NAdof 60) is above the cutoff but stays dense: singularity guard
    ],
)
def test_threshold_backend_splits_volumes_by_nadof(nadofgmres, expect_iterative):
    """THRESHOLD (``Lbeltramisolver=1``) runs GMRES exactly where ``NAdof >= nadofgmres``."""
    fname = f"{curr_path}cyl_manyvol_test.toml"
    test = spectre_with_lconstraint(fname, 1)
    test.input_list_mod.lfindzero = 0
    test.input_list_mod.lhmatrix = False
    test.input_list_mod.lhevalues = False
    test.input_list_mod.lbeltramisolver = 1
    test.input_list_mod.nadofgmres = nadofgmres
    xin = get_xinit_specwrap(test)
    force_real(xin, test)

    mvol = int(test.allglobal_mod.mvol)
    nadof = list(test.allglobal_mod.nadof[:mvol])
    assert nadof == [60, 57, 89], "fixture resolution changed; update the cutoffs above"

    its = np.asarray(test.allglobal_mod.beltrami_gmres_iterations[:mvol])
    ran_gmres = [vol for vol in range(1, mvol + 1) if its[vol - 1] > 0]
    assert ran_gmres == expect_iterative, (
        f"nadofgmres={nadofgmres}: GMRES ran in volumes {ran_gmres}, expected {expect_iterative}"
    )
    np.testing.assert_array_equal(np.asarray(test.allglobal_mod.beltrami_gmres_status[:mvol]), 0)
    remove_file(fname + ".end")
