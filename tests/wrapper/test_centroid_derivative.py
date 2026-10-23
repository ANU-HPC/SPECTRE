"""Differentiate a shaped public centroid through its native Fourier projection."""

from pathlib import Path

import numpy as np
import pytest

from spectre import SPECTRE, force_real, force_real_jac, get_xinit_specwrap
from spectre.file_io.toml_io import (
    input_parameters_from_dict,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)


@pytest.mark.parametrize("constraint", [1, 3])
def test_projected_centroid_coefficients(tmp_path, monkeypatch, constraint):
    p = read_input_parameters_from_toml(Path(__file__).with_name("qa_3vol_light.toml")).model_dump()
    p["physics"].update(enforce_stell_sym=False, lconstraint=constraint)
    p["numeric"]["lrzaxis"] = 1
    p["minimization"]["filter_stellsym"] = False
    filename = tmp_path / "centroid.toml"
    write_input_parameters_to_toml(input_parameters_from_dict(p), filename)
    monkeypatch.chdir(tmp_path)
    obj = SPECTRE(str(filename), verbose=False)
    x = get_xinit_specwrap(obj)
    direction = np.random.default_rng(9021).normal(size=x.size)
    direction[0] = 0.0
    direction /= np.linalg.norm(direction)
    force_real(x, obj)
    force_real_jac(x, obj)
    derivatives = {
        key: getattr(obj.allglobal_mod, key).copy() for key in ["dradr", "dradz", "dzadr", "dzadz"]
    }
    step = 1e-6
    sides = []
    for sign in [1, -1]:
        force = force_real(x + sign * step * direction, obj).ravel()
        geometry = {
            key: getattr(obj.allglobal_mod, key).copy() for key in ["irbc", "irbs", "izbc", "izbs"]
        }
        sides.append((force, geometry))
    geometry = {key: (sides[0][1][key] - sides[1][1][key]) / (2 * step) for key in sides[0][1]}
    columns = [geometry[key][:, 1] for key in ["irbc", "irbs", "izbc", "izbs"]]
    for output, target, cross in [
        ("irbc", "dradr", "dradz"),
        ("irbs", "dradr", "dradz"),
        ("izbc", "dzadr", "dzadz"),
        ("izbs", "dzadr", "dzadz"),
    ]:
        parity = int(output in ["irbs", "izbs"])
        predicted = sum(
            derivatives[name][:, parity, kind, :] @ column
            for name, kind, column in [
                (target, 0, columns[0]),
                (target, 1, columns[1]),
                (cross, 0, columns[2]),
                (cross, 1, columns[3]),
            ]
        )
        np.testing.assert_allclose(predicted, geometry[output][:, 0], rtol=2e-6, atol=2e-8)
