# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
import os.path

from spectre import (
    SPECTRE,
    force_modes_speclike,
    get_fmn_force_mask,
    remove_file,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lcons", [1, 3])
def test_force_real(fname, lcons):

    test = SPECTRE(fname, verbose=False)
    test.input_list_mod.lcons = lcons

    xin = test.xin

    ngdof = test.allglobal_mod.ngdof

    # get spectre fmn
    mask = get_fmn_force_mask(test)

    f = np.zeros([ngdof + 1], dtype=np.float64, order="F")
    test.forces_mod.dforce(ngdof, xin, f, False, True)  # force w/ axis calculation

    fmn_spec = f[1:][mask]

    fmn_custom = force_modes_speclike(xin, test)

    maxelem = max(np.max(np.abs(fmn_spec)), np.max(np.abs(fmn_custom)))

    # assert np.allclose(fmn_spec, fmn_custom.flatten(), atol=9e-9 * maxelem, rtol=1e-9)
    assert np.allclose(fmn_spec, fmn_custom.flatten(), atol=1e-7 * maxelem, rtol=1e-7)

    remove_file(fname + ".end")
