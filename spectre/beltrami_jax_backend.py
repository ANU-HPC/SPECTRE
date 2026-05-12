# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

"""Optional SPECTRE-to-beltrami_jax adapter.

This module is deliberately small: SPECTRE owns initialization, geometry state,
force evaluation, HDF5 output, and MPI orchestration. ``beltrami_jax`` owns the
Beltrami TOML/interface-geometry solve and returns SPECTRE-shaped
``Ate/Aze/Ato/Azo`` coefficient arrays.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from spectre.file_io import write_input_parameters_to_toml
from spectre.utils import set_vec_pot_flat


def _load_beltrami_jax():
    try:
        import beltrami_jax
    except ImportError as exc:
        raise ImportError(
            "The optional beltrami_jax backend is not installed. Install it in the "
            "same environment as SPECTRE, for example with "
            "`python -m pip install /path/to/beltrami_jax`."
        ) from exc
    return beltrami_jax


def solve_input_file_with_beltrami_jax(
    input_file: str | Path,
    *,
    volumes: tuple[int, ...] | None = None,
    solve_local_constraints: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Solve SPECTRE TOML/interface geometry through ``beltrami_jax``.

    Parameters are forwarded to
    ``beltrami_jax.solve_spectre_toml``. The returned object is the
    ``beltrami_jax.SpectreMultiVolumeSolve`` result, including the flattened
    vector potential, per-volume solves, residuals, and diagnostics.
    """

    beltrami_jax = _load_beltrami_jax()
    return beltrami_jax.solve_spectre_toml(
        input_file,
        volumes=volumes,
        solve_local_constraints=solve_local_constraints,
        verbose=verbose,
        **kwargs,
    )


def _current_boundary_normal_field(test, beltrami_jax):
    """Return SPECTRE's live free-boundary normal-field state if present."""

    lfreebound = bool(int(getattr(test.input_list_mod, "lfreebound", 0)))
    if not lfreebound:
        return None

    mn = int(test.allglobal_mod.mn)
    field = beltrami_jax.SpectreBoundaryNormalField(
        ivns=np.asarray(test.allglobal_mod.ivns[:mn], dtype=np.float64),
        ibns=np.asarray(test.allglobal_mod.ibns[:mn], dtype=np.float64),
        ivnc=np.asarray(test.allglobal_mod.ivnc[:mn], dtype=np.float64),
        ibnc=np.asarray(test.allglobal_mod.ibnc[:mn], dtype=np.float64),
    )
    field.validate(mn)
    return field


def _require_serial_backend(test) -> None:
    size = test.comm.Get_size()
    if size != 1:
        raise RuntimeError(
            "The experimental beltrami_jax backend currently supports serial SPECTRE runs only. "
            f"Got MPI size {size}."
        )


def solve_current_state_with_beltrami_jax(
    test,
    *,
    volumes: tuple[int, ...] | None = None,
    solve_local_constraints: bool = False,
    include_normal_field_state: bool = True,
    verbose: bool = False,
    **kwargs,
):
    """Solve the current in-memory SPECTRE state through ``beltrami_jax``.

    This path is used by Python force workflows where the interface vector has
    already been packed into SPECTRE memory. It serializes the current validated
    SPECTRE parameters to a temporary TOML file and hands that file to
    ``beltrami_jax``.
    """

    _require_serial_backend(test)
    beltrami_jax = _load_beltrami_jax()
    if include_normal_field_state and "normal_field" not in kwargs:
        normal_field = _current_boundary_normal_field(test, beltrami_jax)
        if normal_field is not None:
            kwargs["normal_field"] = normal_field
    with TemporaryDirectory(prefix="spectre-beltrami-jax-") as tmpdir:
        input_file = Path(tmpdir) / "current_state.toml"
        write_input_parameters_to_toml(test.parameters, input_file)
        return solve_input_file_with_beltrami_jax(
            input_file,
            volumes=volumes,
            solve_local_constraints=solve_local_constraints,
            verbose=verbose,
            **kwargs,
        )


def apply_beltrami_jax_solution(
    test,
    input_file: str | Path | None = None,
    *,
    volumes: tuple[int, ...] | None = None,
    solve_local_constraints: bool = False,
    include_normal_field_state: bool = True,
    update_fortran: bool = True,
    use_current_state: bool = False,
    verbose: bool = False,
    **kwargs,
):
    """Solve with ``beltrami_jax`` and optionally inject ``Ate/Aze/Ato/Azo``.

    ``update_fortran=True`` requires a full-volume solve because SPECTRE stores
    one contiguous vector-potential block for all packed volumes. Partial-volume
    solves are still useful for validation but are returned without mutating
    SPECTRE memory.
    """

    _require_serial_backend(test)
    if use_current_state:
        result = solve_current_state_with_beltrami_jax(
            test,
            volumes=volumes,
            solve_local_constraints=solve_local_constraints,
            include_normal_field_state=include_normal_field_state,
            verbose=verbose,
            **kwargs,
        )
    else:
        if input_file is None:
            raise ValueError("input_file is required unless use_current_state=True")
        result = solve_input_file_with_beltrami_jax(
            input_file,
            volumes=volumes,
            solve_local_constraints=solve_local_constraints,
            verbose=verbose,
            **kwargs,
        )
    if not update_fortran:
        return result
    if volumes is not None:
        raise ValueError(
            "update_fortran=True requires volumes=None so the full vector-potential block "
            "is available"
        )

    vector_potential = result.vector_potential
    set_vec_pot_flat(
        test,
        np.asarray(vector_potential.ate, dtype=np.float64, order="F"),
        np.asarray(vector_potential.aze, dtype=np.float64, order="F"),
        np.asarray(vector_potential.ato, dtype=np.float64, order="F"),
        np.asarray(vector_potential.azo, dtype=np.float64, order="F"),
    )
    test._beltrami_jax_solution = result
    return result
