# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import os.path

from spectre import (
    SPECTRE,
    input_dict,
    force_real,
    force_real_jac,
    approx_derivative,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def tor():

    return input_dict(
        dict(
            dofs=np.array([1, 5, 37]),
            method="3-point",
            abs_step=np.ones(3) * 1e-8,
            sp_fname=f"{curr_path}qa_7vol_test.toml",
        )
    )


@pytest.mark.parametrize("lcons", [0, 3])
@pytest.mark.parametrize("fixture", ["tor"])
def test_force_filtering(fixture, lcons, request):

    fixture = request.getfixturevalue(fixture)

    sp_fname = fixture.sp_fname

    test = SPECTRE(sp_fname, verbose=False)
    test.input_list_mod.lcons = lcons

    xinit = test.xin

    force_full = force_real(xinit, test, filter_stellsym=False)

    force_filtered = force_real(xinit, test, filter_stellsym=True)

    assert np.allclose(np.sum(force_full**2), np.sum(force_filtered**2) * 2)

    remove_file(sp_fname + ".end")


@pytest.mark.parametrize("lcons", [0, 3])
@pytest.mark.parametrize("fixture", ["tor"])
def test_force_jac_filtering(fixture, lcons, request):

    filter_stellsym = True

    fixture = request.getfixturevalue(fixture)

    dofs = fixture.dofs
    method = fixture.method
    abs_step = fixture.abs_step
    sp_fname = fixture.sp_fname

    test = SPECTRE(sp_fname, verbose=False)
    test.input_list_mod.lcons = lcons

    xinit = test.xin

    force_real(xinit, test, filter_stellsym=filter_stellsym)

    freal_jac_analytic_full = force_real_jac(xinit, test, filter_stellsym=filter_stellsym)

    freal_jac_numeric = approx_derivative(
        lambda x: force_real(x, test, filter_stellsym=filter_stellsym).flatten(order="C"),
        xinit,
        method=method,
        dofs=dofs + 0,
        abs_step=abs_step,
    )

    freal_jac_analytic = freal_jac_analytic_full[:, :, dofs]

    freal_jac_analytic = freal_jac_analytic.reshape(-1, len(dofs))

    maxelem = max(np.max(np.abs(freal_jac_numeric)), np.max(np.abs(freal_jac_analytic)))
    assert np.allclose(freal_jac_numeric, freal_jac_analytic, atol=5e-6 * maxelem, rtol=5e-6)

    remove_file(sp_fname + ".end")
