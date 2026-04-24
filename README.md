# SPECTRE

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

For platform-specific instructions, see the [installation guide](https://spectre.readthedocs.io/en/latest/index.html) in the docs.

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
