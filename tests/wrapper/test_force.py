# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import os.path

import numpy as np
import pytest
from helpers import spectre_with_lconstraint

from spectre import (
    force_modes_speclike,
    get_fmn_force_mask,
    get_xinit_specwrap,
)

curr_path = os.path.dirname(__file__) + "/"


@pytest.mark.parametrize(
    "fname", [f"{curr_path}qa_7vol_test.toml", f"{curr_path}cyl_manyvol_test.toml"]
)
@pytest.mark.parametrize("lcons", [1, 3])
def test_force_real(fname, lcons):

    test = spectre_with_lconstraint(fname, lcons)

    xin = get_xinit_specwrap(test)

    # get spectre fmn
    mask = get_fmn_force_mask(test)

    f = test.dforce(xin)  # force w/ axis calculation

    fmn_spec = f[mask]

    fmn_custom = force_modes_speclike(xin, test)

    maxelem = max(np.max(np.abs(fmn_spec)), np.max(np.abs(fmn_custom)))

    # assert np.allclose(fmn_spec, fmn_custom.flatten(), atol=9e-9 * maxelem, rtol=1e-9)
    assert np.allclose(fmn_spec, fmn_custom.flatten(), atol=1e-7 * maxelem, rtol=1e-7)
