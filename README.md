# SPECTRE

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://gitlab.com/spectre-eq/spectre/-/blob/main/LICENSE) [![Documentation](https://readthedocs.org/projects/spectre/badge/?version=latest)](https://spectre.readthedocs.io/en/latest/index.html) [![Pipeline status](https://gitlab.com/spectre-eq/spectre/badges/main/pipeline.svg)](https://gitlab.com/%{project_path}/-/commits/%{default_branch})

**Stepped Pressure Equilibrium Code — The Revised Edition**

SPECTRE computes general 3D MHD equilibria with generalized magnetic topology, in the framework of Multi-Region relaxed MHD
(MRxMHD). The plasma is subdivided into stepped-pressure subdomains separated by ideal interfaces, in each of which a Beltrami
field is computed. SPECTRE builds on the earlier SPEC code with significant algorithmic and software improvements.

**Full documentation:** [spectre.readthedocs.io](https://spectre.readthedocs.io/en/latest/index.html)

---

## Features

- 3D MHD equilibrium calculations for arbitrary stepped-pressure plasma configurations
- Fixed-boundary and free-boundary scenarios (e.g. QA stellarators, W7-X)
- Force minimization to find MRxMHD equilibria
- Field line tracing and Poincaré analysis
- Postprocessing tools for geometry, field, and stability diagnostics
- TOML-based human-readable input files

## Installation

```bash
pip install -e .
```

SPECTRE itself is MIT-licensed, but a default build uses the fastest backends
regardless of their license: FFTW (GPL) for the FFT and CyRK (CC-BY-NC-SA,
NonCommercial) for field-line tracing. For commercial use, build with

```bash
SPECTRE_COMMERCIAL_USE=1 pip install -e .
pip uninstall CyRK
```

which switches to permissively-licensed backends (pocketfft, scipy) instead.

For platform-specific instructions and the FFT-backend / licensing details, see
the [installation guide](https://spectre.readthedocs.io/en/latest/index.html) in the docs.

## Quick start

The [Get Started guide](https://spectre.readthedocs.io/en/latest/index.html) walks through input file preparation, running a
Beltrami field calculation, and postprocessing results.

A self-contained example is available in `examples/qa_vac_vol7/` — a force minimization calculation for a 7-volume QA stellarator
vacuum equilibrium.

## Testing

```bash
pytest
```

This runs the full test suite in `tests/` and verifies that the compiled Fortran back-end is working correctly.

## Contributing

See the [contributing guide](https://spectre.readthedocs.io/en/latest/index.html) in the docs for the Git workflow, coding
conventions, and pre-commit hook setup.

## Citation

Coming soon.

## Authors

Erol Balkovic, Joaquim Loizu, Emmanuel Lanti, Chris Smiet
