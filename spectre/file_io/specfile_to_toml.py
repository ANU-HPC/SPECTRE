# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Convert SPEC namelist files (``.sp``) into TOML input files.

This module exposes :func:`specfile_to_toml`, a convenience wrapper that reads an
existing SPEC namelist, validates the content against the high-level
:class:`~spectre.file_io.input_parameters.InputParameters` model, and writes an
adjacent TOML representation that fits the Python tooling in this repository.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from .input_parameters import arrays_fourier

import f90nml
import numpy as np

from .input_parameters import (
    DiagnosticsParameters,
    GlobalParameters,
    InputParameters,
    NumericParameters,
    PhysicsParameters,
    ScreenParameters,
)
from .toml_io import write_input_parameters_to_toml

_PathLike = str | Path

_RENAMED_FIELDS = {
    "istellsym": "enforce_stell_sym",
}

_removed_fields = ["lextrap", "lbeltrami", "lmatsolver", "lperturbed", "wpp00aa", ]

_SECTION_TO_FIELD = {
    "physicslist": "physics",
    "numericlist": "numeric",
    "globallist": "global_",
    "diagnosticslist": "diagnostics",
    "screenlist": "screen",
}

_SECTION_FIELDS: dict[str, set[str]] = {
    "physicslist": set(PhysicsParameters.model_fields.keys()),
    "numericlist": set(NumericParameters.model_fields.keys()),
    "globallist": set(GlobalParameters.model_fields.keys()),
    "diagnosticslist": set(DiagnosticsParameters.model_fields.keys()),
    "screenlist": set(ScreenParameters.model_fields.keys()),
}


def _read_guess(path: Path) -> np.NDArray[np.float_]:
    """Read the interface guess at the end of the .sp file."""
    if not isinstance(path, Path):
        path = Path(path)
    lines = path.read_text().splitlines()
    last_end_idx = -1
    for idx, line in enumerate(lines):
        if line.strip().upper() in {"&END", "/"}:
            last_end_idx = idx

    if last_end_idx == -1 or last_end_idx == len(lines) - 1:
        return None

    dump_data = []
    for line in lines[last_end_idx + 1 :]:
        tokens = line.strip().split()
        if tokens:
            dump_data.append([float(x) for x in tokens])

    return None if not dump_data else np.array(dump_data, dtype=float)


def specfile_to_toml(
    source: _PathLike,
    destination: _PathLike | None = None,
) -> tuple[Path, InputParameters] | list[tuple[Path, InputParameters]]:
    """Translate a SPEC ``.sp`` namelist into a TOML input file.

    Parameters
    ----------
    source:
        Path to either a single ``.sp`` file or a directory containing such files.
    destination:
        Optional target file or directory. When omitted, the TOML file will be
        placed next to the source namelist. When *source* is a directory, *destination*
        should also refer to a directory.

    Raises
    ------
    FileNotFoundError
        If *source* does not exist.
    ValueError
        When *source* is a directory but *destination* resolves to a file.
    """

    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(f"No such file or directory: {src_path}")

    dest_path = _determine_destination(src_path, Path(destination) if destination is not None else None)
    params = _parameters_from_spec(source)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    write_input_parameters_to_toml(params, dest_path)
    return


def _parameter_dict_from_spec(path: Path) -> InputParameters:
    """read the .sp file using f90nml and put it into a dictionary 
    of dictionaries to give to the pydantic model InputParameters.
    
    The fourier arrays need to be transposed.  
    """

    nml = f90nml.read(path)
    payload: dict[str, Any] = {}

    for section, field_name in _SECTION_TO_FIELD.items():
        section_data = nml.get(section)
        if section_data is None:
            continue
        payload[field_name] = _normalise_section(section, dict(section_data))

    # fix physics parameters 
    if "physics" in payload:
        # Note, the arrays in the fortran namelist are read by f90nml in row-major order. in the file n is is the first index specified, but in python the first index is m (0 to mpol). We do not need to transpose but beware!:
        for array_name in arrays_fourier:
            if array_name in payload["physics"]:
                data_from_namelist = payload["physics"][array_name]
                payload["physics"][array_name] = np.array(data_from_namelist)  # we DO NOT transpose here so Fortran row odering makes the first index m like we like it
        if "enforce_stell_sym" in payload["physics"]:
            payload["physics"]["enforce_stell_sym"] = bool(payload["physics"]["enforce_stell_sym"])

    # Check if interface guess is present at the end of the file (f90nml does not read it)
    guess_array = _read_guess(path)
    if guess_array is not None:
        payload["physics"]["allrzrz"] = guess_array

    return payload


def _parameters_from_spec(path: Path) -> InputParameters:
    """Read a SPEC namelist file and convert it into an InputParameters model."""

    param_dict = _parameter_dict_from_spec(Path(path))
    return InputParameters.model_validate(param_dict) 


def _normalise_section(section: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Filter and normalise a namelist section for Pydantic validation."""

    allowed = _SECTION_FIELDS[section]
    clean: dict[str, Any] = {}
    for key, value in raw.items():
        canonical = key.lower()
        if canonical in _RENAMED_FIELDS:
            canonical = _RENAMED_FIELDS[canonical]

        if canonical not in allowed:
            if canonical in _removed_fields:
                print(f"{key} from SPEC is not used in SPECTRE and will be ignored.")
                continue
            print(f"Ignoring unknown key '{key}' in section '{section}'")
            continue
        clean[canonical] = _normalise_value(value)
    return clean


def _normalise_value(value: Any) -> Any:
    """Convert f90nml containers into plain Python primitives."""

    if value is None:
        return 0.0

    if isinstance(value, (list, tuple)):
        return [_normalise_value(item) for item in value]
    return value


def _determine_destination(source: Path, destination: Path | None) -> Path:
    """Resolve the TOML destination for *source* with an optional override."""

    if destination is None:
        return source.with_suffix(".toml")

    if destination.suffix:
        return destination

    return destination / (source.stem + ".toml")
