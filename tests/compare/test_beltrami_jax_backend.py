import os.path

import numpy as np
import pytest

pytest.importorskip("beltrami_jax")

from spectre import SPECTRE, force_real, get_vec_pot_flat, get_xinit_specwrap, update_bnorm

curr_path = os.path.dirname(__file__) + "/"


def _injection_relative_error(test):
    solution = getattr(test, "_beltrami_jax_solution", None)
    assert solution is not None
    expected = solution.vector_potential
    actual = get_vec_pot_flat(test)
    return max(
        np.linalg.norm(np.asarray(got) - np.asarray(ref))
        / max(np.linalg.norm(np.asarray(ref)), 1.0)
        for got, ref in zip(
            actual,
            (expected.ate, expected.aze, expected.ato, expected.azo),
            strict=True,
        )
    )


def test_beltrami_jax_free_boundary_force_seam_matches_fortran():
    input_file = curr_path + "G3V8L3Free/input.toml"
    fortran_obj = SPECTRE(input_file, verbose=False)
    jax_obj = SPECTRE(input_file, verbose=False)
    xin = get_xinit_specwrap(fortran_obj)

    force_fortran = force_real(xin, fortran_obj, beltrami_backend="fortran")
    force_jax = force_real(
        xin,
        jax_obj,
        beltrami_backend="jax",
        solve_local_constraints=True,
        jax_backend_options={"verbose": False},
    )

    for it in range(int(fortran_obj.input_list_mod.mfreeits)):
        update_bnorm(fortran_obj, first_free_bound=(it == 0), print_info=False)
        update_bnorm(jax_obj, first_free_bound=(it == 0), print_info=False)
        force_fortran = force_real(xin, fortran_obj, beltrami_backend="fortran")
        force_jax = force_real(
            xin,
            jax_obj,
            beltrami_backend="jax",
            solve_local_constraints=True,
            jax_backend_options={"verbose": False},
        )

    relative_error = np.linalg.norm(force_jax - force_fortran) / max(
        np.linalg.norm(force_fortran), 1.0
    )
    solution = jax_obj._beltrami_jax_solution
    assert relative_error < 1.0e-11
    assert _injection_relative_error(jax_obj) == 0.0
    assert float(solution.global_constraint.residual_norm) < 1.0e-12
    assert float(solution.max_relative_residual_norm) < 1.0e-10
