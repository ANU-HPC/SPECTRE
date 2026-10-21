# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import numpy as np

import spectre.force_targets as force_targets


def test_force_real_jac_has_constant_objective_shape_on_overlap(monkeypatch):
    """An overlap sentinel must have a Jacobian with the objective shape."""

    test = SimpleNamespace(
        input_list_mod=SimpleNamespace(
            mpol=5,
            ntor=4,
            ndiscrete=2,
            nfp=5,
            enforce_stell_sym=True,
        ),
        allglobal_mod=SimpleNamespace(mvol=4, ngdof=17),
    )
    monkeypatch.setattr(force_targets, "check_intersection_wrapper", lambda *args, **kwargs: True)

    jac = force_targets.force_real_jac(np.zeros(18), test, nt=40, nz=32)

    assert jac.shape == (3, 1280, 18)
    assert np.all(jac == 0.0)
