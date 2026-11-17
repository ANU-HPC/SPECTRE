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

SPECTRE is MIT-licensed and a default build is fully permissive: the FFT uses
pocketfft (BSD-3-Clause) and field-line tracing uses CyRK (Apache-2.0,
installed from GitHub because the PyPI releases up to 0.17.1 still carry a
stale NonCommercial license string in their metadata; see
`thirdparty_licenses/`). The faster GPL-licensed FFTW can be selected
explicitly:

```bash
pip install -e . -C cmake.define.SPECTRE_FFT_BACKEND=fftw
```

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

Any work that uses SPECTRE, or results obtained with it, needs to cite the following paper:

> E. Balkovic, J. Loizu, E. Lanti, C. Smiet, C. Lazzati, R. Ramasamy, A. Goodman, J. Geiger and J. P. Graves,
> "SPECTRE: a robust solver for 3D equilibria with arbitrary topology",
> *Plasma Phys. Control. Fusion* **68**, 115021 (2026). [doi:10.1088/1361-6587/aebef8](https://doi.org/10.1088/1361-6587/aebef8)

```bibtex
@article{balkovic2026spectre,
  title   = {{SPECTRE}: a robust solver for {3D} equilibria with arbitrary topology},
  author  = {Balkovic, E. and Loizu, J. and Lanti, E. and Smiet, C. and Lazzati, C. and Ramasamy, R. and
             Goodman, A. and Geiger, J. and Graves, J. P.},
  journal = {Plasma Physics and Controlled Fusion},
  volume  = {68},
  number  = {11},
  pages   = {115021},
  year    = {2026},
  doi     = {10.1088/1361-6587/aebef8}
}
```

## Authors

Erol Balkovic, Joaquim Loizu, Emmanuel Lanti, Chris Smiet
