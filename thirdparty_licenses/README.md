# Third-party licenses

Licenses of third-party software that SPECTRE depends on and that is
redistributed alongside SPECTRE in binary distributions (e.g. the Docker
images built from `tools/run_container/`). The SPECTRE wheel itself does not
bundle these packages, so these texts are not shipped inside the wheel; any
image or bundle that installs the dependencies must carry them.

| File | Project | License | Provenance |
|------|---------|---------|------------|
| `CyRK.LICENSE.md` | [CyRK](https://github.com/jrenaud90/CyRK) — Runge-Kutta ODE integrator used by the field-line tracer | Apache-2.0 | `LICENSE.md` at commit `64bc40e86d12c161c6f1c3e04075b12d187a7f58` (v0.17.1 + metadata fixes), verbatim |

Notes:

- CyRK changed its license from CC-BY-NC-SA-4.0 to Apache-2.0 in v0.16.1
  (2025-12-10). The PyPI artifacts up to and including 0.17.1 still carry a
  stale `CC-BY-NC-SA-4.0` SPDX string in their metadata, so SPECTRE installs
  CyRK directly from the GitHub repository at a pinned commit where both the
  license file and the package metadata state Apache-2.0 (see
  `pyproject.toml`).
- CyRK has no NOTICE file, so per Apache-2.0 §4 only the license text itself
  must accompany redistributions.
