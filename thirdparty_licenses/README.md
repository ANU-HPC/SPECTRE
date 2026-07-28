# Third-party licenses

Licenses of third-party software whose terms SPECTRE must carry. These fall
into three categories, and they travel differently:

- **Source-derived** — third-party code adapted into SPECTRE's own sources.
  These terms ship *inside the wheel*, because the code itself does, and they
  apply to every redistribution of SPECTRE regardless of form.
- **Redistributed dependencies** — packages installed alongside SPECTRE in
  binary distributions (e.g. the Docker images built from
  `tools/run_container/`). The SPECTRE wheel does not bundle these packages,
  but keeps their license texts available for downstream bundles; any image or
  bundle that installs the dependencies must carry them.
- **Compiled-in dependencies** — third-party code linked into SPECTRE's
  extension modules. These terms ship inside the wheel and every other binary
  redistribution that contains the compiled code.

| File | Project | License | Category | Provenance |
|------|---------|---------|----------|------------|
| `scipy.LICENSE.txt` | [SciPy](https://scipy.org) — least-squares optimizers and numerical differentiation adapted into `spectre/least_squares/` and `spectre/numdiff_scipy.py` | BSD-3-Clause | source-derived | scipy 1.18.0 `LICENSE.txt`, SciPy's own BSD-3-Clause section (verbatim) |
| `pocketfft.LICENSE.txt` | [pocketfft](https://github.com/mreineck/pocketfft) — default FFT implementation compiled into SPECTRE's Fortran extension | BSD-3-Clause | compiled-in dependency | license header in `pocketfft_hdronly.h` at commit `5f27d5a8f51c5c25030cb22abf434decc9faf0ff` (verbatim) |
| `CyRK.LICENSE.md` | [CyRK](https://github.com/jrenaud90/CyRK) — Runge-Kutta ODE integrator used by the field-line tracer | Apache-2.0 | redistributed dependency | `LICENSE.md` at commit `64bc40e86d12c161c6f1c3e04075b12d187a7f58` (v0.17.1 + metadata fixes), verbatim |

Not to be confused with `LICENSES/`, which is the REUSE pool: one file per SPDX
identifier used in a `SPDX-License-Identifier:` header, holding the plain
license text with `<year> <owner>` placeholders. `LICENSES/BSD-3-Clause.txt`
therefore resolves the identifier for *all* BSD-3-Clause material here (SciPy
and pocketfft alike); the notices naming actual copyright holders are the files
in this directory and the per-file headers.

Notes:

- SciPy's BSD-3-Clause applies because SPECTRE adapts SciPy source rather than
  merely importing it. The following files are derived works and carry SciPy's
  copyright in addition to SPECTRE's own:
  `spectre/least_squares/least_squares.py`, `spectre/least_squares/trf.py`,
  `spectre/least_squares/dogbox.py`, `spectre/least_squares/common.py`, and
  `spectre/numdiff_scipy.py`.
  Clause 1 requires the copyright notice, conditions, and disclaimer to be
  retained in redistributions **of source code**, so this text must remain in
  the repository and in the wheel — not only in binary bundles.
- Only SciPy's own BSD-3-Clause section is reproduced here. SciPy's full
  `LICENSE.txt` additionally covers libraries its *binary wheels* bundle
  (OpenBLAS, LAPACK, the GCC runtime, libquadmath); SPECTRE redistributes none
  of those, so their terms are out of scope.
- PocketFFT is fetched from its pinned commit at configure time, but its code
  is compiled into the default backend. Its BSD-3-Clause terms therefore
  accompany the wheel and runtime container through `pocketfft.LICENSE.txt`.
- CyRK changed its license from CC-BY-NC-SA-4.0 to Apache-2.0 in v0.16.1
  (2025-12-10). The PyPI artifacts up to and including 0.17.1 still carry a
  stale `CC-BY-NC-SA-4.0` SPDX string in their metadata, so SPECTRE installs
  CyRK directly from the GitHub repository at a pinned commit where both the
  license file and the package metadata state Apache-2.0 (see
  `pyproject.toml`).
- CyRK has no NOTICE file, so per Apache-2.0 §4 only the license text itself
  must accompany redistributions.
