#!/usr/bin/env python

## usage: ./get_fsa_force.py test.sp

from spectre import (
    SPECTRE,
    force_real,
    get_freal_fsa_metric,
    get_xinit_specwrap,
)
import sys

files = sys.argv[1:]

for f in files:

    test = SPECTRE.from_input_file(f, verbose=False)

    test.input_list_mod.lfindzero = 0
    test.input_list_mod.nppts = 0
    test.input_list_mod.lcheck = 0

    xin = get_xinit_specwrap(test)

    force_real(xin, test)

    if test.comm.Get_rank() == 0:
        print(f"{f:^20}   <f> = {get_freal_fsa_metric(test):.12e}")

    del test
