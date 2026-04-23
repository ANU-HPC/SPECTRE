# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Utilities for reading and writing SPEC input parameters in TOML format."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import tomlkit

from .input_parameters import InputParameters

PathLike = os.PathLike[str] | str

__all__ = [
    "input_parameters_to_dict",
    "input_parameters_from_dict",
    "write_input_parameters_to_toml",
    "read_input_parameters_from_toml",
]


def input_parameters_to_dict(params: InputParameters) -> dict[str, Any]:
    """Convert an :class:`InputParameters` instance into a TOML-friendly mapping.

    The returned dictionary contains only plain Python types (``dict``, ``list``,
    ``int``/``float``/``bool``/``str``). Numpy containers are converted to lists to
    ensure compatibility with :mod:`tomli_w`.
    """

    payload = params.model_dump(mode="json", exclude_none=True)

    # Rename the `global_` section to `global` for nicer TOML tables.
    global_section = payload.pop("global_", None)
    if global_section is not None:
        payload["global"] = global_section

    return payload


def input_parameters_from_dict(data: Mapping[str, Any]) -> InputParameters:
    """Create an :class:`InputParameters` instance from a mapping."""

    payload: MutableMapping[str, Any] = deepcopy(dict(data))

    if "global" in payload and "global_" not in payload:
        payload["global_"] = payload.pop("global")

    return InputParameters.model_validate(payload)


def write_input_parameters_to_toml(
    params: InputParameters, destination: PathLike
) -> None:
    """Serialise *params* to TOML and write them to *destination*.

    Parameters
    ----------
    params:
        The :class:`InputParameters` instance to write.
    destination:
        Filesystem path that will be opened in binary write mode.
    """

    data = input_parameters_to_dict(params)
    path = Path(destination)
    with path.open("w") as handle:
        tomlkit.dump(data, handle)


def read_input_parameters_from_toml(source: PathLike) -> InputParameters:
    """Read *source* and return an :class:`InputParameters` instance."""

    path = Path(source)
    with path.open("r") as handle:
        data = tomlkit.load(handle).unwrap()

    return input_parameters_from_dict(data)


