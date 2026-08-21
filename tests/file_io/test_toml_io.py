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

from pydantic import ValidationError

from spectre.file_io.input_parameters import BeltramiSolver, InputParameters, PhysicsParameters
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


@pytest.mark.parametrize("ngmresrestart", [0, -1])
def test_zero_gmres_restart_is_rejected(ngmresrestart: int):
    """A restart length below one leaves GMRES without a single inner iteration."""
    with pytest.raises(ValidationError):
        PhysicsParameters(ngmresrestart=ngmresrestart)


@pytest.mark.parametrize("lbeltramisolver", [-1, 3, 42])
def test_unknown_beltrami_backend_is_rejected(lbeltramisolver: int):
    """Backend selection handles 0, 1 and 2 only; anything else must not reach Fortran."""
    with pytest.raises(ValidationError):
        PhysicsParameters(lbeltramisolver=lbeltramisolver)


def test_beltrami_backend_modes_are_named_and_serialize_as_integers(tmp_path: Path):
    params = InputParameters(physics=PhysicsParameters(lbeltramisolver=BeltramiSolver.ITERATIVE))

    assert params.physics.lbeltramisolver == 2
    assert input_parameters_to_dict(params)["physics"]["lbeltramisolver"] == 2

    output_path = tmp_path / "params.toml"
    write_input_parameters_to_toml(params, output_path)
    reloaded = read_input_parameters_from_toml(output_path)
    assert reloaded.physics.lbeltramisolver is BeltramiSolver.ITERATIVE


def test_lfreebsolver_defaults_off():
    """The direct free-boundary solve is opt-in: the physics default is the Picard scheme."""
    assert PhysicsParameters().lfreebsolver == 0
    assert InputParameters().physics.lfreebsolver == 0


@pytest.mark.parametrize("lfreebsolver", [-1, 2, 42])
def test_invalid_lfreebsolver_is_rejected(lfreebsolver: int):
    with pytest.raises(ValidationError):
        PhysicsParameters(lfreebsolver=lfreebsolver)


def test_lfreebsolver_serializes_under_physics(tmp_path: Path):
    """lfreebsolver lives in [physics]; vcnt/vcnz stay in [global]."""
    params = InputParameters(
        physics=PhysicsParameters(lfreebsolver=1, lfreebound=1, igeometry=3, mpol=2, ntor=2)
    )

    data = input_parameters_to_dict(params)
    assert data["physics"]["lfreebsolver"] == 1
    assert "lfreebsolver" not in data["global"]
    assert data["global"]["vcnt"] == 256
    assert data["global"]["vcnz"] == 256

    output_path = tmp_path / "params.toml"
    write_input_parameters_to_toml(params, output_path)
    parsed = tomllib.loads(output_path.read_text())
    assert parsed["physics"]["lfreebsolver"] == 1
    assert "lfreebsolver" not in parsed["global"]

    reloaded = read_input_parameters_from_toml(output_path)
    assert reloaded.physics.lfreebsolver == 1
    assert reloaded.global_.vcnt == 256


def test_lfreebsolver_requires_free_boundary_toroidal_geometry():
    with pytest.raises(ValidationError, match="lfreebound"):
        InputParameters(physics=PhysicsParameters(lfreebsolver=1, lfreebound=0))
    with pytest.raises(ValidationError, match="igeometry"):
        InputParameters(physics=PhysicsParameters(lfreebsolver=1, lfreebound=1, igeometry=1))


def test_lfreebsolver_source_grid_resolution_is_validated():
    """vcnt/vcnz (in [global]) must resolve the boundary harmonics of the direct solve."""
    physics = PhysicsParameters(lfreebsolver=1, lfreebound=1, igeometry=3, mpol=80, ntor=2)
    with pytest.raises(ValidationError, match="vcnt"):
        InputParameters(physics=physics)
    direct_mpol40 = {"lfreebsolver": 1, "lfreebound": 1, "igeometry": 3, "mpol": 40, "ntor": 2}
    with pytest.raises(ValidationError, match="vcnt"):
        input_parameters_from_dict({"physics": direct_mpol40, "global": {"vcnt": 128, "vcnz": 128}})

    ok = input_parameters_from_dict(
        {"physics": direct_mpol40, "global": {"vcnt": 192, "vcnz": 128}}
    )
    assert ok.global_.vcnt == 192


def test_misplaced_global_lfreebsolver_is_rejected():
    """lfreebsolver selects the free-boundary physics: a [global] placement must fail loudly."""
    with pytest.raises(ValidationError, match=r"\[physics\]"):
        input_parameters_from_dict({"global": {"lfreebsolver": 1}})
    with pytest.raises(ValidationError, match=r"\[physics\]"):
        InputParameters.model_validate({"global_": {"lfreebsolver": 0}})


def test_removed_picard_inputs_are_rejected_with_replacements():
    """The removed Fortran-Picard inputs fail loudly, naming their Python replacements.

    The models ignore unknown keys, so without the dedicated rejection a legacy deck
    carrying them would silently run with different semantics than it asked for.
    """
    with pytest.raises(ValidationError, match="update_bns"):
        input_parameters_from_dict({"global": {"mfreeits": 2}})
    with pytest.raises(ValidationError, match="update_bns"):
        InputParameters.model_validate({"global_": {"mfreeits": 0}})
    with pytest.raises(ValidationError, match="initialize_bns_from_current"):
        input_parameters_from_dict({"numeric": {"lautoinitbn": 1}})
    # The replacement keeps the legacy default (LautoinitBn = 1): a deck without the key
    # still bootstraps iBns from the plasma.
    assert input_parameters_from_dict({}).minimization.initialize_bns_from_current is True
    with pytest.raises(ValidationError, match="no remaining effect"):
        input_parameters_from_dict({"numeric": {"lzerovac": 0}})


def test_removed_cteta_czeta_are_rejected_naming_vcnt_vcnz():
    """The merged source-grid keys fail loudly, pointing at vcnt/vcnz.

    One uniform source grid now serves both the Picard virtual-casing integral and the
    Laplace-virtual-casing matrix; a deck still carrying the old keys must not silently run
    the direct solve on the (different) vcnt/vcnz defaults.
    """
    with pytest.raises(ValidationError, match="vcnt"):
        input_parameters_from_dict({"global": {"cteta": 96}})
    with pytest.raises(ValidationError, match="vcnz"):
        input_parameters_from_dict({"global": {"czeta": 64}})
    with pytest.raises(ValidationError, match="vcnt"):
        InputParameters.model_validate({"global_": {"cteta": 128, "czeta": 128}})


def test_removed_lvcvacuum_is_rejected_naming_lfreebsolver():
    """The old switch name fails loudly, pointing at physics.lfreebsolver."""
    with pytest.raises(ValidationError, match="lfreebsolver"):
        input_parameters_from_dict({"physics": {"lvcvacuum": 1}})
    with pytest.raises(ValidationError, match="lfreebsolver"):
        InputParameters.model_validate({"physics": {"lvcvacuum": 0}})


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
