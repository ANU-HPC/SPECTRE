#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import os

DATASETS_TO_EXTRACT = [
    "output/force_final",
    "output/force_final_grad",
    "vector_potential/Ate",
    "vector_potential/Azo",
    "vector_potential/Ato",
    "vector_potential/Aze",
]


def extract_to_new_hdf5(input_path, output_path):
    """Extract specified datasets from one HDF5 file into a new one."""
    from h5py import File

    with File(input_path, "r") as src, File(output_path, "w") as dst:
        for ds_path in DATASETS_TO_EXTRACT:
            if ds_path in src:
                data = src[ds_path][()]

                group_path = os.path.dirname(ds_path)
                if group_path:
                    dst.require_group(group_path)
                dst.create_dataset(ds_path, data=data)
            else:
                raise RuntimeError(f"Dataset '{ds_path}' not found")


if __name__ == "__main__":
    parser = ArgumentParser(description="Setup a SPECTRE comparison case")
    parser.add_argument("input_file", type=str, help="Path to the .sp input file")
    parser.add_argument("root_dir", type=str, help="Directory to store the case")
    parser.add_argument(
        "run_cmd",
        type=str,
        default="xspec",
        help="Command running a legacy SPEC/SPECTRE `xspec` binary to generate the "
        "reference (this repository no longer builds one; use a reference checkout)",
    )
    args = parser.parse_args()
    args.root_dir += "/"

    from subprocess import run
    import f90nml

    os.system(f"mkdir -p {args.root_dir}")

    ## Step 1: Take .sp input file and create an input file for NSPEC
    nml = f90nml.read(f"{args.input_file}")

    nml["physicslist"]["rac"][0] = -1.0
    nml["diagnosticslist"]["nppts"] = 0
    nml["diagnosticslist"]["lcheck"] = 7

    xspec_sp_name = f"{args.root_dir}input_spec.sp"
    nml.write(xspec_sp_name, force=True)

    ## Step 2: Run old SPECTRE to generate the output .h5 file
    p = run(f"{args.run_cmd} {xspec_sp_name}", shell=True, capture_output=False, text=True)

    ## Step 3: Extract relevant data from the .h5 file into a new file
    extract_to_new_hdf5(f"{xspec_sp_name}.h5", f"{args.root_dir}reference.h5")

    # os.system(f"rm {xspec_sp_name}.h5 {xspec_sp_name}.end")

    # Need to adjust the file for the new code. Other variables might need to be removed here...
    nml = f90nml.read(f"{args.root_dir}input_spec.sp")
    del nml["physicslist"]["istellsym"]
    nml["physicslist"]["enforce_stell_sym"] = True
    nml.write(f"{args.root_dir}input.sp", force=True)

    ## Note: final step is to add an entry to compare.py, e.g.:
    # @pytest.fixture(scope="module")
    # def G3V08L3Fi():
    #     return input_dict(
    #         dict(
    #             loc=f"{curr_path}G3V08L3Fi/",
    #             atol=1e-8,
    #             rtol=1e-8
    #         )
    #     )
