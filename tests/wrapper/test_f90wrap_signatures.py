"""Guard the f90wrap-generated Fortran signatures that the Python layer calls.

Every routine below is called from ``spectre/`` or ``scripts/`` through the
f90wrap bindings. When a Fortran signature changes, the callers otherwise break
only at run time — typically as a ``TypeError`` deep inside a minimizer
callback. This test pins the exact wrapper parameter lists so a signature
change fails immediately, by name, with a reminder to migrate the callers.
"""

import inspect
import os.path

import numpy as np
import pytest
import spectre4py_wrap

from spectre import SPECTRE

curr_path = os.path.dirname(__file__) + "/"

# "module.routine" -> exact wrapper parameter names, in order. On mismatch:
# migrate the Python callers (grep for the routine in spectre/ and scripts/),
# then update the entry here in the same commit.
EXPECTED_SIGNATURES = {
    "allglobal_mod.set_mpi_comm": ["comm"],
    "allglobal_mod.check_inputs": [],
    "allglobal_mod.broadcast_inputs": [],
    "field_mod.solve_field": ["position", "recompute_axis"],
    "field_mod.solve_field_deriv": ["position", "mpol_rz", "ntor_rz", "recompute_axis"],
    "field_mod.calc_b2mag_jump": ["dbbtotal", "nt_dbb", "nz_dbb"],
    "field_mod.calc_b2mag_jump_deriv": ["force_real_jac", "nt_dbb", "nz_dbb"],
    "forces_mod.dforce": [
        "ngdof",
        "position",
        "force_ft_trunc",
        "lcomputederivatives",
        "lcomputeaxis",
    ],
    "geometry_mod.compute_coordinates": [
        "lvol",
        "lss",
        "lcurvature",
        "nt",
        "nz",
        "mn",
        "lcoordinatesingularity",
        "rij",
        "zij",
        "sg",
        "inv_jac",
        "guvij",
    ],
    "hessian_mod.compute_hessian": ["position", "ohessian"],
    "io_mod.mute_output": ["mute"],
    "input_list_mod.initialize_inputs": [],
    "packing_mod.packxi": [
        "ngdof",
        "position",
        "mvol",
        "mn",
        "nt",
        "nz",
        "irbc",
        "izbs",
        "irbs",
        "izbc",
        "packorunpack",
        "lcomputederivatives",
        "lcomputeaxis",
    ],
    "preset_mod.initialize_internal_arrays": [],
    "simulation_mod.spectre": [],
    "simulation_mod.final_diagnostics": [],
    "sphdf5_mod.init_outfile": [],
    "sphdf5_mod.mirror_input_to_outfile": [],
    "sphdf5_mod.write_grid": [],
    "sphdf5_mod.hdfint": [],
    "sphdf5_mod.finish_outfile": [],
    "wrapper_funcs_mod.alloc_allrzrz": ["nvol", "mntor", "mmpol"],
    "wrapper_funcs_mod.alloc_for_hessian": ["ntz_loc"],
    "wrapper_funcs_mod.alloc_remn_ext": [],
    "wrapper_funcs_mod.angconst_dfp200": [
        "ang_constr_jac_full",
        "add_psifactor",
        "add_sweightfactor",
    ],
    "wrapper_funcs_mod.beltrami_matvec_selftest": ["lvol", "nvec", "testmu"],
    "wrapper_funcs_mod.beltrami_solve_selftest": [
        "lvol",
        "testmu",
        "restart",
        "maxiter",
        "gmres_tol",
        "use_sparse",
    ],
    "wrapper_funcs_mod.calc_beltrami_error": [
        "lvol",
        "error_type",
        "lquad",
        "jerroravg",
        "jerrormax",
        "err_grid",
    ],
    "wrapper_funcs_mod.dealloc_hessian": [],
    "wrapper_funcs_mod.diag_integrals_selftest": ["lvol"],
    "wrapper_funcs_mod.get_angle_constraint": ["iil_total", "dll_total"],
    "wrapper_funcs_mod.get_mesh": ["vvol", "rij_grid", "zij_grid", "ind_i", "ind_j"],
    "wrapper_funcs_mod.get_vec_pot": ["sumlrad", "allate", "allaze", "allato", "allazo"],
    "wrapper_funcs_mod.gmres_selftest": ["k", "restart", "maxiter"],
    "wrapper_funcs_mod.prepare_endfile_writing": [],
    "wrapper_funcs_mod.samemn_precond_compare": ["lvol", "testmu"],
    "wrapper_funcs_mod.update_bnorm": [],
}


@pytest.mark.parametrize("dotted", sorted(EXPECTED_SIGNATURES))
def test_wrapper_signature(dotted):
    """The wrapper for each Python-called Fortran routine has the pinned signature."""
    mod_name, routine = dotted.split(".")
    func = getattr(getattr(spectre4py_wrap, mod_name), routine)
    assert list(inspect.signature(func).parameters) == EXPECTED_SIGNATURES[dotted], (
        f"{dotted} changed signature; migrate its callers in spectre/ and "
        f"scripts/, then update EXPECTED_SIGNATURES"
    )


def test_compute_coordinates_all_volumes():
    """compute_coordinates fills a finite, sign-consistent Jacobian in every volume.

    Called directly on a freshly instantiated 3D case (no solve), exactly like
    the minimizer-callback path in ``utils.get_fsa_wrapper``. Volume 1 with
    Igeometry=3 exercises the coordinate-singular Zernike branch, the outer
    volumes the regular branch.
    """
    test = SPECTRE(f"{curr_path}qa_7vol_test.toml", verbose=False)

    nt = int(test.field_mod.nt_alloc)
    nz = int(test.field_mod.nz_alloc)
    mn = int(test.allglobal_mod.mn)
    mvol = int(test.allglobal_mod.mvol)
    ntz = nt * nz

    rij = np.zeros((ntz, 4, 4), dtype=np.float64, order="F")
    zij = np.zeros((ntz, 4, 4), dtype=np.float64, order="F")
    sg = np.zeros((ntz, 4), dtype=np.float64, order="F")
    inv_jac = np.zeros((ntz,), dtype=np.float64, order="F")
    guvij = np.zeros((ntz, 4, 4, 5), dtype=np.float64, order="F")

    for vol in range(1, mvol + 1):
        # lregion rule: only volume 1 is coordinate-singular, except in slab geometry
        lcoordinatesingularity = bool(vol == 1 and test.input_list_mod.igeometry != 1)
        test.geometry_mod.compute_coordinates(
            vol,
            0.5,
            1,
            nt,
            nz,
            mn,
            lcoordinatesingularity,
            rij,
            zij,
            sg,
            inv_jac,
            guvij,
        )
        jac = sg[:, 0]
        assert np.all(np.isfinite(jac)), f"non-finite Jacobian in volume {vol}"
        assert (jac > 0).all() or (jac < 0).all(), f"Jacobian changes sign in volume {vol}"
