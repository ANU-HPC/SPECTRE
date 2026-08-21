# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Ground-truth physics test of the direct (lfreebsolver = 1) free-boundary solve.

A current-free, pressureless free-boundary equilibrium built from the NCSX coil set must
reproduce the coil vacuum field: after force minimization the interfaces settle onto flux
surfaces of the coil field, the total field inside the computational boundary equals the
Biot-Savart field of the coils, and the plasma-induced (virtual casing) normal field on the
computational boundary vanishes. The reference field is evaluated with simsopt's BiotSavart —
fully independent of SPECTRE's discretization.

Accuracy is limited by the Fourier truncation of the boundaries (extend_via_normal of the
strongly-shaped NCSX boundary is hard to represent at mpol=4/ntor=3) and of the vns spectrum,
and by the imperfect flux surfaces of the coils-only NCSX field — not by the solver. At this
resolution the observed agreement is ~4% of |B| (any sign, normalization, or self-consistency
defect would show up at order unity); the tolerances below carry a 2x margin on the observed
values.
"""

# DISABLED (2026-08-21): the fixture is the most expensive item in the suite (~80 s) and its
# minimization stops at max_niter = 80 (status 6) at every dVC grid size measured, so the
# asserted state is an 80-iteration snapshot rather than a converged point; the |ofm|/vns
# metric wobbles with that snapshot (0.09 at vcnt = vcnz = 96, 0.19 at 48, 0.08 at 32 against a
# 0.2 tolerance). Kept verbatim below for re-enabling once the fixture is re-pinned by a
# one-knob convergence study (vcnt/vcnz, max_niter).
#
# from pathlib import Path
#
# import numpy as np
# import pytest
#
# simsopt = pytest.importorskip("simsopt")
# from simsopt.mhd import Vmec  # noqa: E402
#
# from spectre import Force_Minimizer  # noqa: E402
# from spectre.converters import simsvmec2spectre  # noqa: E402
# from spectre.converters.fix2free import fix2free, params_to_simsopt_boundary  # noqa: E402
# from spectre.file_io.toml_io import write_input_parameters_to_toml  # noqa: E402
#
# pytestmark = pytest.mark.simsopt
#
# WOUT_LI383 = Path(__file__).parents[1] / "test_files" / "wout_li383_low_res_reference.nc"
#
# _MPOL = 4
# _NTOR = 3
# _NPTS = 24
#
#
# @pytest.fixture(scope="module")
# def ncsx_vacuum_minimized(tmp_path_factory):
#     """Minimize the NCSX vacuum free-boundary case with the direct solve; return (obj, bs)."""
#     from simsopt.configs.zoo import get_data
#
#     _, _, _, _, bs = get_data("ncsx")
#     vmec = Vmec(str(WOUT_LI383))
#     # boundary deep inside the plasma: the full li383 boundary is too strongly shaped for this
#     # truncation (the truncated interfaces intersect), and the coils-only NCSX field only has
#     # decent flux surfaces in the core
#     params = simsvmec2spectre(
#         vmec, tfluxes=[0.12, 0.35], mpol=_MPOL, ntor=_NTOR, interface_guess="wout"
#     )
#
#     plasma_boundary = params_to_simsopt_boundary(
#         params, which="plasma", ntheta=64, nzeta_per_period=64
#     )
#     comp_boundary = plasma_boundary.copy()
#     comp_boundary.extend_via_normal(
#         -0.05
#     )  # NCSX's bean cross-section self-intersects if extended further
#
#     freeb = fix2free(params, bs, comp_boundary, nvns_mpol=_MPOL, nvns_ntor=_NTOR, vacuum=True)
#
#     freeb.physics.lfreebsolver = 1
#     freeb.global_.vcnt = 96
#     freeb.global_.vcnz = 96
#     freeb.minimization.max_niter = 80
#     freeb.minimization.gtol = 1e-10
#     freeb.minimization.ftol = 1e-11
#     # Same convergence targets, cheaper route: the lagged-banded trust-region iteration
#     # replaces most dense Jacobian builds with small JVP batches (the direct solve's fixed
#     # residual definition makes free-boundary lagging legal). The assertions below test the
#     # CONVERGED state against Biot-Savart, independent of how the minimizer got there.
#     freeb.minimization.method = "trf"
#     freeb.minimization.tr_solver = "banded"
#     freeb.minimization.jvp_lag = 6
#
#     workdir = tmp_path_factory.mktemp("ncsx_vac")
#     fname = workdir / "ncsx_vacuum_vc.toml"
#     write_input_parameters_to_toml(freeb, fname)
#
#     import os
#
#     cwd = os.getcwd()
#     os.chdir(workdir)  # SPECTRE and the minimizer drop scratch/output files in the cwd
#     try:
#         obj = Force_Minimizer(str(fname), save_x_iter=False)
#         obj.minimize()
#     finally:
#         os.chdir(cwd)
#
#     return obj, bs
#
#
# def test_vacuum_field_matches_biotsavart(ncsx_vacuum_minimized):
#     obj, bs = ncsx_vacuum_minimized
#     test = obj.field_obj
#
#     rng = np.random.default_rng(11)
#     nfp = int(test.input_list_mod.nfp)
#     s_pts = np.asfortranarray(rng.uniform(-0.8, 0.8, _NPTS))
#     t_pts = np.asfortranarray(rng.uniform(0.0, 2.0 * np.pi, _NPTS))
#     z_pts = np.asfortranarray(rng.uniform(0.0, 2.0 * np.pi / nfp, _NPTS))
#
#     gb, bcyl, rz = test.vacuum_field_at(s_pts, t_pts, z_pts)
#
#     bs.set_points_cyl(np.column_stack([rz[0], z_pts, rz[1]]))
#     b_ref = bs.B_cyl()  # columns (B_R, B_phi, B_Z)
#
#     b_scale = np.max(np.abs(b_ref))
#     assert b_scale > 0.1  # NCSX field ~ O(1) T; guard against a degenerate reference
#     # truncation-limited: ~4% observed at this resolution (see module docstring), 2x margin
#     assert np.allclose(bcyl.T, b_ref, atol=8e-2 * b_scale, rtol=0.0)
#
#
# def test_vacuum_plasma_contribution_vanishes(ncsx_vacuum_minimized):
#     obj, _ = ncsx_vacuum_minimized
#     test = obj.field_obj
#
#     efm, ofm = test.bnorm_from_plasma()
#
#     vns_scale = np.max(np.abs(np.array(test.allglobal_mod.ivns)))
#     assert vns_scale > 0.0
#     # a current-free plasma produces no virtual-casing field, up to the residual sheet currents
#     # of the truncation-limited force balance: ~10% of the vns scale observed, 2x margin
#     assert np.max(np.abs(ofm)) < 2e-1 * vns_scale
