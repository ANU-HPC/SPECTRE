"""Check optimizer controls against analytic, ill-scaled least-squares problems."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from spectre import Force_Minimizer
from spectre.file_io.input_parameters import MinimizationParameters
from spectre.force_minimizer import optimizer_default
from spectre.least_squares.least_squares import least_squares


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_input_scaling_solves_anisotropic_problem(method):
    params = MinimizationParameters(method=method, x_scale="jac", max_niter=30, xtol=1e-12)
    obj = SimpleNamespace(min_params=params, iteration_start=0, callback=lambda *a: None)
    # A unit step in residual space requires very different physical displacements.
    diagonal = np.array([1e-6, 1e6])
    target = np.array([1e6, 1e-6])
    result = optimizer_default(
        obj,
        np.zeros(2),
        lambda x, obj: diagonal * (x - target),
        lambda x, obj: np.diag(diagonal),
    )
    np.testing.assert_allclose(result.x, target, rtol=1e-8)
    assert result.success


@pytest.mark.parametrize("method", ["trf", "dogbox"])
def test_iteration_cap_is_not_success(method):
    result = least_squares(
        lambda x: x - 10,
        np.zeros(1),
        jac=lambda x: np.eye(1),
        method=method,
        max_niter=0,
        cb=lambda *a: None,
    )
    assert result.status == 6
    assert not result.success
    assert np.linalg.norm(result.fun) > 1


@pytest.mark.parametrize("method,module", [("trf", "trf"), ("dogbox", "dogbox")])
def test_requested_stop_is_not_success(method, module, monkeypatch):
    algorithm = import_module(f"spectre.least_squares.{module}")
    monkeypatch.setitem(algorithm.state, "flag", False)
    result = least_squares(
        lambda x: x - 10,
        np.zeros(1),
        jac=lambda x: np.eye(1),
        method=method,
        cb=lambda *a: None,
    )
    assert result.status == 5
    assert not result.success


def test_fourier_interface_jacobian_matches_directional_difference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = Path(__file__).with_name("qa_3vol_light.toml")
    obj = Force_Minimizer(str(source), save_x_iter=False)
    x = obj.x_init.copy()
    direction = np.random.default_rng(185850).normal(size=x.size)
    direction /= np.linalg.norm(direction)
    predicted = obj.jac_func(x, obj) @ direction
    step = 1e-6
    difference = (
        obj.obj_func(x + step * direction, obj) - obj.obj_func(x - step * direction, obj)
    ) / (2 * step)
    np.testing.assert_allclose(predicted, difference, rtol=3e-5, atol=3e-5)
