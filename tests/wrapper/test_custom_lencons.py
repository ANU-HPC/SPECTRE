# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint

from spectre import (
    approx_derivative,
    get_theta_chord_length,
    get_theta_chord_length_jac,
    get_xinit_specwrap,
    input_dict,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def tor():

    return input_dict(
        dict(
            dofs=np.array([1, 5, 37]),
            method="3-point",
            abs_step=np.ones(3) * 1e-9,
            sp_fname=f"{curr_path}qa_7vol_test.toml",
        )
    )


@pytest.mark.parametrize("fixture", ["tor"])
def test_custom_lenconc(fixture, request):

    fixture = request.getfixturevalue(fixture)

    dofs = fixture.dofs
    method = fixture.method
    abs_step = fixture.abs_step
    sp_fname = fixture.sp_fname

    test = spectre_with_lconstraint(sp_fname, 0)

    thetas = np.linspace(0, 2 * np.pi, 32)
    zetas = np.array([0.0, 0.8])
    ngdof = test.allglobal_mod.ngdof

    xin = get_xinit_specwrap(test)

    dlength_dxin = get_theta_chord_length_jac(xin, test, thetas=thetas, zetas=zetas)

    abs_step = 1e-6

    def func(x):
        return get_theta_chord_length(x, test, thetas=thetas, zetas=zetas).reshape(-1)

    jac_numeric = approx_derivative(func, xin, method=method, dofs=dofs, abs_step=abs_step)

    jac_analytic = dlength_dxin.reshape(-1, ngdof)

    maxelem = max(np.max(np.abs(jac_numeric)), np.max(np.abs(jac_analytic)))

    assert np.allclose(jac_numeric, jac_analytic[:, dofs], atol=1e-8 * maxelem, rtol=1e-8)
