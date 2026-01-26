#!/usr/bin/env python
import sys
from spectre import SPECTREout

fnames = sys.argv[1:]

for fname in fnames:

    o = SPECTREout(fname)

    print(f'{fname}')
    o.calc_avg_beta_vol(plot=False)
    print('\n')