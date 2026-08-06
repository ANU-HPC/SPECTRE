# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for global-radial field-trace inspection."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from spectre.field_tracer import Field_Tracer  # noqa: E402


@pytest.fixture
def tracer():
    """Return a two-volume tracer with synthetic completed trace data."""
    obj = Field_Tracer(2, 3, 1, tflux=[0.25, 1.0])
    obj.sth_total = [
        np.array([[[-1.0]], [[0.0]], [[1.0]]]),
        np.array([[[-1.0]], [[0.0]], [[1.0]]]),
    ]
    obj.RZ_total = [np.zeros((3, 2, 1)), np.zeros((3, 2, 1))]
    obj.iotas = [np.array([-0.2, -0.3, -0.4]), np.array([-0.5, -0.6, -0.7])]
    obj.bad_traj_count = [[], [(1, 0)]]
    obj.phi_eval = np.array([0.0])
    obj.num_phi_planes = 1
    obj.num_transits = 1
    obj.num_pts_traj = 1
    obj.zetan_start = 0.0
    return obj


def test_s_to_tflux(tracer):
    """Local volume coordinates map continuously onto global normalized flux."""
    np.testing.assert_allclose(tracer.s_to_tflux(0, [-1.0, 0.0, 1.0]), [0.0, 0.125, 0.25])
    np.testing.assert_allclose(tracer.s_to_tflux(1, [-1.0, 0.0, 1.0]), [0.25, 0.625, 1.0])


def test_s_to_tflux_validates_inputs():
    """Missing flux data and invalid coordinates raise clear errors."""
    empty = Field_Tracer(1, 3, 1)
    with pytest.raises(RuntimeError, match="tflux"):
        empty.s_to_tflux(0, 0.0)

    traced = Field_Tracer(1, 3, 1, tflux=[1.0])
    with pytest.raises(ValueError, match="ivol"):
        traced.s_to_tflux(1, 0.0)
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        traced.s_to_tflux(0, 2.0)


def test_plot_iota_radial(tracer):
    """The composite profile uses global rho and absolute iota by default."""
    ax = tracer.plot_iota_radial()
    np.testing.assert_allclose(ax.lines[0].get_xdata(), np.sqrt([0.0, 0.125, 0.25]))
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [0.2, 0.3, 0.4])
    assert ax.get_xlim()[0] == pytest.approx(0.0)
    assert ax.lines[-1].get_color() == "lightgrey"
    plt.close(ax.figure)


def test_tflux_h5_round_trip(tracer, tmp_path):
    """Saved field traces retain the flux mapping needed by the radial plot."""
    filename = tmp_path / "trace.h5"
    tracer.save_h5(str(filename))
    loaded = Field_Tracer.init_from_h5(str(filename))
    np.testing.assert_array_equal(loaded.tflux, tracer.tflux)
