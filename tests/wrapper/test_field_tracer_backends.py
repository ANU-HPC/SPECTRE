# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

import importlib.util
import os.path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, get_xinit_specwrap

curr_path = os.path.dirname(__file__) + "/"

HAS_CYRK = importlib.util.find_spec("CyRK") is not None

# ivol 0 contains the coordinate singularity (bfield_flow_sing right-hand
# side); the outermost volume exercises the regular bfield_flow_nonsing one.
VOLUME_CASES = ["singular", "nonsingular"]


@pytest.fixture(scope="module")
def qa_out():
    obj = SPECTRE(f"{curr_path}qa_7vol_test.toml")
    xin = get_xinit_specwrap(obj)
    force_real(xin, obj)
    return obj.create_spectreout()


def _trace(qa_out, which, rk_backend):
    ivol = 0 if which == "singular" else qa_out.input.physics.Mvol - 1
    return qa_out.trace_one_fieldline(
        np.array([0.5, 0.3]),
        ivol,
        num_phi_planes=8,
        num_transits=2,
        rtol=1e-9,
        atol=1e-9,
        rk_backend=rk_backend,
    )


@pytest.mark.parametrize("which", VOLUME_CASES)
def test_tracer_scipy_sing_and_nonsing(qa_out, which):
    sth, RZ, iota, flag = _trace(qa_out, which, "scipy")
    assert not flag
    assert np.all(np.isfinite(sth))
    assert np.all(np.isfinite(RZ))
    assert np.isfinite(iota)


@pytest.mark.skipif(not HAS_CYRK, reason="CyRK not installed")
@pytest.mark.parametrize("which", VOLUME_CASES)
def test_tracer_backends_agree(qa_out, which):
    sth_s, RZ_s, iota_s, flag_s = _trace(qa_out, which, "scipy")
    sth_c, RZ_c, iota_c, flag_c = _trace(qa_out, which, "cyrk")
    assert not flag_s
    assert not flag_c
    assert np.allclose(sth_s, sth_c, atol=1e-6)
    assert np.allclose(RZ_s, RZ_c, atol=1e-6)
    assert abs(iota_s - iota_c) < 1e-6
