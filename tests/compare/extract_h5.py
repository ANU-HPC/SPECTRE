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
    # "stability/ohessian" # the true hessian for stability
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
    parser.add_argument("input_file", type=str, help="Path to the .sp.h5 input file")
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        default="reference.h5",
        help="Path to the .sp.h5 output file",
    )
    args = parser.parse_args()

    extract_to_new_hdf5(args.input_file, args.output_file)
