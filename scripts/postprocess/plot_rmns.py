#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
import sys
import matplotlib.pyplot as plt

from spectre import SPECTRE, get_xinit_specwrap, infaces_unflatten, plot_fmn

fnames = sys.argv[1:]
vols = [0,1,2,5]

for fname in fnames:
	if fname[-5:] != '.toml':
		raise ValueError(f"Invalid input: '{fname}'")

	test = SPECTRE.from_input_file(fname, verbose=True)

	xin = get_xinit_specwrap(test)

	r, z = infaces_unflatten(xin, test)

	if vols is None:
		vols = np.arange(test.inputlist.nvol)

	fig, ax = plt.subplots(1, len(vols),figsize=(3.6*len(vols), 4.3))

	for i in range(len(vols)):
		plottant = 0.5*np.log10(r[vols[i]]**2)
		# plottant = (r[vols[i]])
		plot_fmn(plottant, test.allglobal.im, test.allglobal.in_/2, ax=ax[i], mpol_box=4, ntor_box=4)
		ax[i].set_title(f"inface={vols[i]}")
		if i>0:
			ax[i].set_ylabel('')

	fig.suptitle(fname)

	plt.tight_layout()

plt.show()
