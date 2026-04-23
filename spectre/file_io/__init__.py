# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""TOML-based input/output helpers for SPEC."""

from __future__ import annotations

from .input_parameters import InputParameters
from .specfile_to_toml import specfile_to_toml
from .toml_io import (
	input_parameters_from_dict,
	input_parameters_to_dict,
	read_input_parameters_from_toml,
	write_input_parameters_to_toml,
)

__all__ = [
	"InputParameters",
	"input_parameters_from_dict",
	"input_parameters_to_dict",
	"read_input_parameters_from_toml",
	"write_input_parameters_to_toml",
	"specfile_to_toml",
]

