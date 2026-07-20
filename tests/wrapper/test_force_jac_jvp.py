# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Golden test of the force-Jacobian JVP against the dense Jacobian build.

``force_real_jvp`` computes J @ d with one general-direction linearised solve per volume
(Fortran ``solve_field_deriv_dir``); the dense build computes one column per interface
harmonic. Both are exact linearisations of the same discretisation, so they must agree to
rounding for unit directions (single dense columns) and for random directions spanning
every interface, R and Z, cosine and sine, including the constraint (dmupfdx) coupling.
"""

import os.path

import numpy as np
import pytest

from spectre import (
    SPECTRE,
    force_real,
    force_real_jac,
    force_real_jvp,
    get_xinit_specwrap,
    remove_file,
)
from spectre.force_targets import _process_input_nt_nz

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize("lcons", [0, 1, 3])
@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
def test_force_real_jvp_matches_dense_jacobian(fname, lcons):
    test = SPECTRE(fname, verbose=False)
    test.input_list_mod.lcons = lcons
    nt, nz = _process_input_nt_nz(None, None, test)

    xinit = np.asarray(get_xinit_specwrap(test))
    ngdof = int(test.allglobal_mod.ngdof)

    force_real(xinit, test, nt=nt, nz=nz, filter_stellsym=False)
    J = force_real_jac(xinit, test, nt=nt, nz=nz, filter_stellsym=False).reshape(-1, ngdof + 1)

    # unit directions (single dense columns) and a random direction across all interfaces
    rng = np.random.default_rng(11)
    dirs = np.zeros((ngdof + 1, 3), order="F")
    dirs[1, 0] = 1.0
    dirs[min(5, ngdof), 1] = 1.0
    dirs[1:, 2] = rng.standard_normal(ngdof)

    jvp = force_real_jvp(xinit, dirs, test, nt=nt, nz=nz, filter_stellsym=False)
    jvp = jvp.reshape(-1, dirs.shape[1])

    ref = J @ dirs
    for k in range(dirs.shape[1]):
        scale = max(np.max(np.abs(ref[:, k])), 1e-300)
        rel = np.max(np.abs(jvp[:, k] - ref[:, k])) / scale
        assert rel < 1e-10, (
            f"{os.path.basename(fname)} lcons={lcons}: JVP direction {k} deviates from the "
            f"dense Jacobian by rel {rel:.3e}"
        )

    remove_file(fname + ".end")
