#!/usr/bin/env python
import sys
from spectre import SPECout

fnames = sys.argv[1:]

for fname in fnames:

    o = SPECout(fname)

    print(f'{fname}')
    o.calc_avg_beta_vol(plot=False)
    print('\n')