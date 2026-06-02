# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is SPECTRE

SPECTRE (Stepped Pressure Equilibrium Code — The Revised Edition) is a 3D MHD equilibrium solver. It computes general 3D MHD equilibria in the Multi-Region relaxed MHD (MRxMHD) framework, dividing plasma into stepped-pressure subdomains with ideal interfaces where Beltrami fields are computed. The code supports fixed-boundary and free-boundary configurations (e.g., QA stellarators, W7-X).

The codebase is a **hybrid Fortran/Python** project: ~35 Fortran 90 modules form the physics/numerics core, exposed to Python via f90wrap/f2py wrappers. The Python layer handles input parsing, optimization driving, post-processing, and user-facing API.

## Commands

```bash
# Install in development mode
pip install -e .

# Run all tests
pytest

# Run tests, skipping over crashes (segfaults from Fortran)
pytest -n1

# Lint and format Python
ruff check --fix
ruff format

# Build Fortran manually (if needed)
cmake -S . -B build/
cmake --build build/ -j4

# Build documentation
pip install -e .[docs]
cd docs && make html

# Run all pre-commit hooks
pre-commit run --all-files
```

## Architecture

### Fortran → Python interface

The build pipeline: Fortran `.F90` sources → cpp preprocessing → f90wrap generates `f90wrap_*.f90` wrappers → f2py compiles to `_spectre4py_wrap.so` + `spectre4py_wrap.py`. When modifying or debugging the Fortran-Python boundary, look at `spectre/CMakeLists.txt` (f90wrap/f2py pipeline) and `fortran_src/wrapper_funcs_mod.F90` (functions explicitly exposed to Python).

### Python package layout (`spectre/`)

- `core/core.py` — `SPECTRE` class: the main Fortran wrapper. Reads a TOML input, initializes Fortran state, owns the MPI communicator.
- `force_minimizer.py` — `Force_Minimizer`: drives the optimization loop, calling Fortran force evaluations via the `SPECTRE` object.
- `force_targets.py` / `angle_targets.py` — objective functions passed to the minimizer.
- `file_io/input_parameters.py` — Pydantic models for all TOML input sections (`[physics]`, `[numeric]`, `[global_]`, `[diagnostics]`, `[minimization]`). This is the authoritative schema for inputs.
- `file_io/toml_io.py` — TOML serialization/deserialization.
- `postprocess/` — `SPECTREout` class for reading `.h5` output files and computing derived quantities.
- `surfaces/` — RZ-Fourier, Henneberg, and generic surface geometry classes.
- `least_squares/` — Optimization backends (dogbox, trust-region-reflective).
- `converters/` — VMEC → SPECTRE input conversion (`simsvmec2spectre()`).
- `field_tracer.py` — Field line integration and Poincaré mapping (uses CyRK).

The public API is assembled in `spectre/__init__.py`.

### Fortran modules (`fortran_src/`)

Key modules: `constants_mod`, `allglobal_mod` (global state + MPI), `input_parameters_mod`, `forces_mod`, `beltrami_solver_mod`, `geometry_mod`, `field_mod`, `hessian_mod`, `sphdf5_mod` (HDF5 I/O), `poincare_mod`, `wrapper_funcs_mod`.

### Typical workflow

```python
s = SPECTRE("input.toml", name="myrun", verbose=True, comm=MPI.COMM_WORLD)
minimizer = Force_Minimizer(s, ...)
result = minimizer.solve()
out = SPECTREout("myrun.h5")
```

## Coding conventions

### Python
- Line length: 100 characters
- Formatter/linter: Ruff; Google-style docstrings
- Python 3.10+
- Double quotes

### Fortran
- Extension: `.F90` (capital F, free-form with preprocessing)
- One module per file; `private` by default with explicit `public` lists
- Indentation: 2 spaces (no tabs)
- Line length: 132 characters max
- Lower-case keywords; modern operators (`==` not `.eq.`); spaced ends (`end if`, `end do`)
- No commented-out blank lines
