#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

## usage: ./get_fsa_force.py test.sp

import sys

from spectre import SPECTRE, get_freal_fsa_metric

files = sys.argv[1:]

for f in files:
    test = SPECTRE(f, verbose=False)

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.nppts = 0
    test.input_list_mod.lcheck = 0

    test.force_real(
        nt=test.min_params.nt_dbb,
        nz=test.min_params.nz_dbb,
        filter_stellsym=test.min_params.filter_stellsym,
    )

    if test.comm.Get_rank() == 0:
        print(f"{f:^20}   <f> = {get_freal_fsa_metric(test):.9e}")

    del test
