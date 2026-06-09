# SPDX-FileCopyrightText: 2026 Proxima Fusion
#
# SPDX-License-Identifier: MIT

"""Tests for the ``spectre_obj.inspect`` plotting namespace (``spectre.core.inspect``)."""

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: no display needed for the plot smoke tests
import matplotlib.pyplot as plt  # noqa: E402

from spectre import SPECTRE, remove_file  # noqa: E402

TEST_FILES = Path(__file__).parent.parent / "test_files"

# Free-boundary, toroidal (igeometry=3) rotating-ellipse case, light resolution.
FNAME = str(TEST_FILES / "rot_ell_light.toml")


@pytest.fixture
def test():
    """A fresh SPECTRE instance, with figures and output files cleaned up afterwards."""
    obj = SPECTRE(FNAME, verbose=False)
    yield obj
    plt.close("all")
    remove_file(FNAME + ".end")


@pytest.mark.parametrize("method", ["plot_infaces", "plot_jacobian", "plot_grid"])
def test_plot_methods_are_bound_and_delegate(test, method):
    """The xin-based plot helpers live on the ``inspect`` namespace and run headless."""
    assert callable(getattr(test.inspect, method))
    getattr(test.inspect, method)(test.xin)
    assert plt.get_fignums()  # something was actually drawn


def test_plot_methods_default_xin(test):
    """The xin argument defaults to the instance's current ``xin``."""
    test.inspect.plot_infaces()
    assert plt.get_fignums()


def test_plot_true_mesh_returns_mesh(test):
    """plot_true_mesh is on the namespace and returns the (R, Z) mesh arrays."""
    rij, zij = test.inspect.plot_true_mesh(vvol=0)
    assert rij.shape == zij.shape
    assert rij.ndim == 3
