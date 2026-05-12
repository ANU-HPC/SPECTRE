# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:  # Python 3.11+
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[assignment]

from spectre.file_io.input_parameters import InputParameters, PhysicsParameters
from spectre.file_io.toml_io import (
    input_parameters_from_dict,
    input_parameters_to_dict,
    read_input_parameters_from_toml,
    write_input_parameters_to_toml,
)


@pytest.fixture()
def sample_parameters() -> InputParameters:
    physics = PhysicsParameters(
        nvol=3,
        mpol=1,
        ntor=1,
        tflux=[0.1, 0.2, 0.3, 0.4],
        helicity=[0.5, 0.6, 0.7],
        iota=[0.0, 0.25, 0.5, 0.75],
        rbc={
            "(0, 0)": -0.2,
            "(0, 1)": -0.3,
            "(1, -1)": 0.1,
            "(1, 0)": 0.2,
            "(1, 1)": 0.3,
        },
    )
    return InputParameters(physics=physics)


def test_to_dict_uses_global_section(sample_parameters: InputParameters):
    data = input_parameters_to_dict(sample_parameters)

    assert "global" in data
    assert "global_" not in data
    assert data["global"]["vcasingeps"] == pytest.approx(1.0e-12)


def test_round_trip_through_toml(tmp_path: Path, sample_parameters: InputParameters):
    output_path = tmp_path / "params.toml"

    write_input_parameters_to_toml(sample_parameters, output_path)
    assert output_path.exists()

    content = output_path.read_text()
    assert "[global]" in content
    parsed = tomllib.loads(content)
    assert parsed["global"]["vcasingeps"] == pytest.approx(1.0e-12)

    rehydrated = read_input_parameters_from_toml(output_path)

    original_dump = sample_parameters.model_dump(mode="json", exclude={"physics": {"allrzrz"}})
    new_dump = rehydrated.model_dump(mode="json", exclude={"physics": {"allrzrz"}})
    assert original_dump == new_dump

    # Ensure Fourier coefficients survive the round-trip
    original_rbc = sample_parameters.physics.rbc
    new_rbc = rehydrated.physics.rbc
    np.testing.assert_allclose(original_rbc, new_rbc)

    # Keys emitted through serialization should never contain negative n values
    serialized = input_parameters_to_dict(sample_parameters)
    rbc_keys = serialized["physics"]["rbc"].keys()
    for key in rbc_keys:
        m_raw, n_raw = (part.strip() for part in key.strip("()").split(","))
        assert int(m_raw) >= 0


def test_dict_round_trip(sample_parameters: InputParameters):
    data = input_parameters_to_dict(sample_parameters)
    rebuilt = input_parameters_from_dict(data)

    assert rebuilt.model_dump(
        mode="json", exclude={"physics": {"allrzrz"}}
    ) == sample_parameters.model_dump(mode="json", exclude={"physics": {"allrzrz"}})


# def test_reading_fixture_returns_model():
#    path = Path("input_files/test_cases/G3V01L0Fr.001.toml")
#
#    params = read_input_parameters_from_toml(path)
#
#    assert isinstance(params, InputParameters)
#    assert params.physics.nvol >= 1
#    assert params.physics.rbc.shape[0] >= params.physics.ntor + 1
