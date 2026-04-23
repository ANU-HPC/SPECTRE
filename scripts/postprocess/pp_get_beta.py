#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import sys
from spectre import SPECTREout

fnames = sys.argv[1:]

for fname in fnames:
    o = SPECTREout(fname)

    print(f"{fname}")
    o.calc_avg_beta_vol(plot=False)
    print("\n")
