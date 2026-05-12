# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from spectre.file_io import read_input_parameters_from_toml, specfile_to_toml
from spectre.file_io.specfile_to_toml import _parameters_from_spec
from spectre.core.core import SPECTRE

SPEC_FILES = sorted(Path("tests").rglob("*.sp"))


@pytest.mark.parametrize("specfile", SPEC_FILES, ids=lambda path: str(path))
def test_specfile_to_toml_single_file(tmp_path: Path, specfile: Path) -> None:
    destination_dir = tmp_path / "outputs"
    destination_dir.mkdir()

    specfile_to_toml(specfile, destination_dir)

    output_path = destination_dir / f"{specfile.stem}.toml"
    params = _parameters_from_spec(specfile)

    assert output_path.exists()
    assert output_path.suffix == ".toml"

    round_tripped = read_input_parameters_from_toml(output_path)
    # 
    assert round_tripped.model_dump(exclude={'physics': {'allrzrz'}}) == params.model_dump(exclude={'physics': {'allrzrz'}})


@pytest.mark.parametrize("specfile", SPEC_FILES, ids=lambda path: str(path))
def test_toml_and_deprecated_readers_produce_identical_models(tmp_path: Path, specfile: Path) -> None:
    toml_path = tmp_path / f"{specfile.stem}.toml"
    specfile_to_toml(specfile, toml_path)

    with pytest.warns(DeprecationWarning):
        deprecated_instance = SPECTRE.from_deprecated_input_file(str(specfile), verbose=False)
    toml_instance = SPECTRE(str(toml_path), verbose=False)

    deprecated_params = deprecated_instance.last_set_params
    toml_params = toml_instance.last_set_params

    assert deprecated_params.model_dump(exclude={"physics": {"allrzrz"}}) == toml_params.model_dump(exclude={"physics": {"allrzrz"}})

    orig_allrzrz = deprecated_params.physics.allrzrz
    new_allrzrz = toml_params.physics.allrzrz
    if orig_allrzrz is None:
        assert new_allrzrz is None
    else:
        # Only compare modes that are kept through serialization:
        # - modes within mpol/ntor limits
        # The order in allrzrz does not matter, so we compare by (m, n) key
        mpol = toml_params.physics.mpol
        ntor = toml_params.physics.ntor
        orig_mmrzrz = deprecated_params.physics.mmrzrz
        orig_nnrzrz = deprecated_params.physics.nnrzrz
        new_mmrzrz = toml_params.physics.mmrzrz
        new_nnrzrz = toml_params.physics.nnrzrz
        
        # Build a mapping from (m, n) -> coefficients for the new data
        # Only keep entries with non-zero data (to handle duplicate (m,n) entries)
        new_mode_data = {}
        for mode_idx in range(new_allrzrz.shape[2]):
            m = new_mmrzrz[mode_idx]
            n = new_nnrzrz[mode_idx]
            if m <= mpol and abs(n) <= ntor:
                coeffs = new_allrzrz[:, :, mode_idx]
                # Only store if it has non-zero data, or if we haven't seen this mode yet
                if (m, n) not in new_mode_data or np.any(coeffs != 0):
                    new_mode_data[(m, n)] = coeffs
        
        # Compare original modes (within limits) against new data by (m, n) key
        for mode_idx in range(orig_allrzrz.shape[2]):
            m = orig_mmrzrz[mode_idx]
            n = orig_nnrzrz[mode_idx]
            if m <= mpol and abs(n) <= ntor:
                orig_coeffs = orig_allrzrz[:, :, mode_idx]
                # Skip if all coefficients are zero (no data to compare)
                if not np.any(orig_coeffs != 0):
                    continue
                assert (m, n) in new_mode_data, f"Mode (m={m}, n={n}) missing in round-tripped data"
                np.testing.assert_allclose(
                    orig_coeffs, 
                    new_mode_data[(m, n)],
                    err_msg=f"Mismatch at mode (m={m}, n={n})"
                )


