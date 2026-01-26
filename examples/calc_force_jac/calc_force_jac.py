#!/usr/bin/env python
import sys
import time

from spectre import (
    SPECTRE,
    get_xinit_specwrap,
    force_real,
    force_real_jac,
)

test = SPECTRE.from_input_file(sys.argv[1], verbose=False)

test.read()
test.input_list_mod.lfindzero = 0
test.input_list_mod.lhmatrix = False
test.input_list_mod.lhevalues = False

xinit = get_xinit_specwrap(test)

force_real(xinit, test)

t0 = time.time()

fjac = force_real_jac(xinit, test)

t1 = time.time()

print(f"Total time {t1-t0:.3f} s")

print('fjac \n', fjac[:3, 2:4])

