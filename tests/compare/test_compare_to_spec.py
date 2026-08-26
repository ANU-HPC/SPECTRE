# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest
from h5py import File

from spectre import (
    SPECTRE,
    force_modes_jac_speclike,
    force_modes_speclike,
    get_fmn_force_mask,
    get_vec_pot_flat,
    get_xinit_specwrap,
    input_dict,
)
from spectre.stability import calc_hessian

curr_path = os.path.dirname(__file__) + "/"


@pytest.fixture(scope="module")
def cylinder():
    return input_dict(dict(loc=f"{curr_path}G2V32L1Fi/", atol=1e-10, rtol=1e-10))


@pytest.fixture(scope="module")
def rot_ellipse():
    return input_dict(dict(loc=f"{curr_path}G3V3L3Fi/", atol=1e-10, rtol=1e-10))


@pytest.fixture(scope="module")
def rot_ellipse_lcons2():
    return input_dict(dict(loc=f"{curr_path}G3V3L2Fi_stability/", atol=1e-6, rtol=1e-6))


@pytest.fixture(scope="module")
def freeb_tokamak():
    return input_dict(dict(loc=f"{curr_path}G3V8L3Free/", atol=1e-10, rtol=1e-10))


@pytest.mark.parametrize(
    "fixture", ["cylinder", "rot_ellipse", "rot_ellipse_lcons2", "freeb_tokamak"]
)
def test_compare_with_spec(fixture, request):

    fixture = request.getfixturevalue(fixture)

    test = SPECTRE(fixture.loc + "input.toml", verbose=False)

    xin = get_xinit_specwrap(test)

    mask = get_fmn_force_mask(test)

    with File(fixture.loc + "reference.h5", "r") as f:
        fmn_spec = f["output/force_final"][()]
        fmn_grad_spec = f["output/force_final_grad"][()]
        ate_spec = f["vector_potential/Ate"][()]
        azo_spec = f["vector_potential/Azo"][()]
        ato_spec = f["vector_potential/Ato"][()]
        aze_spec = f["vector_potential/Aze"][()]

    # test force modes

    fmn_custom = force_modes_speclike(xin, test)

    if test.input_list_mod.lfreebound:
        # The stored reference was generated with mfreeits = 2, gBnbld = 0.0: exactly two
        # undamped sweeps, the first integrating the plasma-side surface current. That is
        # a property of the reference file, not of the input deck, so it is hardcoded
        # here: "initialize" performs the plasma-side replacement sweep, and
        # max_sweeps=1 grants the converge tail exactly one further undamped sweep.
        test.update_bns(mode="initialize", damping=0.0, max_sweeps=1)

        fmn_custom = force_modes_speclike(xin, test)

    maxelem = max(np.max(np.abs(fmn_custom)), np.max(np.abs(fmn_spec)))
    assert np.allclose(
        fmn_custom, fmn_spec[1:][mask], atol=fixture.atol * maxelem, rtol=fixture.rtol
    )

    # test vector potential

    ate, aze, ato, azo = get_vec_pot_flat(test)

    assert np.allclose(ate, ate_spec.T, atol=fixture.atol * np.max(np.abs(ate)), rtol=fixture.rtol)
    assert np.allclose(aze, aze_spec.T, atol=fixture.atol * np.max(np.abs(aze)), rtol=fixture.rtol)
    assert np.allclose(ato, ato_spec.T, atol=fixture.atol * np.max(np.abs(ato)), rtol=fixture.rtol)
    assert np.allclose(azo, azo_spec.T, atol=fixture.atol * np.max(np.abs(azo)), rtol=fixture.rtol)

    # test force modes jacobian

    fmn_jac_custom = force_modes_jac_speclike(xin, test)

    maxelem = max(np.max(np.abs(fmn_jac_custom)), np.max(np.abs(fmn_grad_spec[:, mask])))

    assert np.allclose(
        fmn_jac_custom, fmn_grad_spec[:, mask], atol=fixture.atol * maxelem, rtol=fixture.rtol
    )

    # if lconstraint is 2 and lhmatrix is true, also compare the true hessian, which relates to lineary stability
    if test.input_list_mod.lconstraint == 2 and test.input_list_mod.lhmatrix:
        with File(fixture.loc + "reference.h5", "r") as f:
            hess_spec = f["stability/ohessian"][()]
        hess_custom = calc_hessian(xin, test)

        assert np.allclose(hess_spec, hess_custom, atol=1e-10, rtol=1e-8)
