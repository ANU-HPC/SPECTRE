<!--
SPDX-FileCopyrightText: 2026 Proxima Fusion

SPDX-License-Identifier: MIT
-->

# SPEC comparison references

Each case directory holds three files:

- `input_spec.sp` — the exact legacy namelist the reference was generated from;
- `reference.h5` — the reference data extracted from that run: the datasets
  `output/force_final`, `output/force_final_grad` and the four vector-potential
  arrays `vector_potential/{Ate, Aze, Ato, Azo}`;
- `input.toml` — the same case in SPECTRE's TOML format, compared against the
  reference by `test_compare_to_spec.py`.

## How the references were generated

By a legacy SPEC/SPECTRE `xspec` binary, which this repository no longer builds
(the Fortran driver was excised in 3fbe929). The recipe, formerly automated by
`tests/compare/setup_case.py` (removed together with the binary; recover it with
`git show 3fbe929:tests/compare/setup_case.py` if needed):

1. Take the case's `.sp` namelist and set `rac(1) = -1.0`, `nppts = 0` and
   `lcheck = 7` (the legacy diagnostic mode that writes `force_final`,
   `force_final_grad` and the vector potentials to the output file). This
   adjusted namelist is the checked-in `input_spec.sp`.
2. Run `xspec input_spec.sp` from a checkout that still builds it (any commit
   before 3fbe929, or upstream SPEC).
3. Extract the six datasets above from `input_spec.sp.h5` into `reference.h5`:
   `./extract_h5.py input_spec.sp.h5 -o reference.h5` (this step needs no legacy
   binary, only its output file).
4. Derive the modern input: replace `istellsym` by `enforce_stell_sym` in the
   namelist and convert to `input.toml`
   (`spectre.file_io.specfile_to_toml`); drop inputs the current model rejects
   (`mfreeits`, `lautoinitbn`, `lzerovac`, ...).
5. Register the case as a fixture in `test_compare_to_spec.py`
   (`loc`, `atol`, `rtol`) and add it to the parametrize list.

## Case-specific provenance

- `G3V8L3Free` (the only free-boundary case) was generated with
  `mfreeits = 2, gBnbld = 0.0`: exactly two undamped virtual-casing sweeps, the
  first integrating the plasma-side surface current. Those inputs no longer
  exist; `test_compare_to_spec.py` reproduces the sequence explicitly with
  `update_bns(mode="initialize", damping=0.0, max_sweeps=1)`. This is a
  property of the stored reference, not of the input deck — regenerate the
  reference before changing it.
