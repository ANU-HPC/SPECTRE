#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

# Run with `mpirun -n 7 python run_vmec.py input.xyz`

# Runs vmec through simsopt

import sys
from simsopt.mhd import Vmec

vobj = Vmec(sys.argv[1], verbose=True)

vobj.run()
