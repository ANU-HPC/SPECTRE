#!/usr/bin/env python

# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from os import system

from spectre import SPECTRE, SPECTREout

zetan = 0.1875

choose_case = 0 # 0: fixed boundary, 1: free boundary

if choose_case == 0:
    # Test case 1: A fixed-b. stellarator with 7 volumes
    fname = "fixb_7vol.toml"
    num_transits = 1000
    num_trajectories = 20
    out_obj = SPECTREout("fixb_7vol.h5")

elif choose_case == 1:
    # Test case 2: A free-b. stellarator with 1 plasma and 1 vacuum volume 
    # some field lines may leave the plasma
    fname = "freeb_2vol.toml"
    num_transits = 300
    num_trajectories = 10
    out_obj = SPECTREout('freeb_2vol.h5')

## Generate custom poincare and save to the usual .h5 file
test = SPECTRE.from_input_file(input_file=fname)

test.input_list_mod.nppts = 0  # Disable old poincare tracing
test.run(save_output=False)

num_phi_planes = test.allglobal_mod.nz
init_pts = [
    np.array(
        [
            [(-1.0 + 1e-14) + itrj * (2.0 - 2e-14) / num_trajectories, 0.0]
            for itrj in range(0 if j > 0 else 1, num_trajectories + 1)
        ]
    )
    for j in range(test.allglobal_mod.mvol)
]

out_wrapper = test.create_spectreout()

out_wrapper.trace_fieldlines(
    init_pts,
    num_phi_planes=num_phi_planes,
    num_transits=num_transits,
    method="RK45",
    rtol=1.0e-7,
    atol=1.0e-7,
)

if MPI.COMM_WORLD.Get_rank() == 0:
    system(f"rm {test.name}_pncare.h5")
    out_wrapper.tracer.save_h5(test.name + "_pncare.h5")

# Plot results, and compare to old poincare
if MPI.COMM_WORLD.Get_rank() == 0:
    from spectre import Field_Tracer

    tracer = Field_Tracer.init_from_h5(test.name + "_pncare.h5")

    fig, ax = plt.subplots(1, 1, figsize=(9, 9))
    out_obj.plot_infaces(ax=ax, color="k", zetan=zetan, prange="full")
    out_obj.plot_poincare(ax=ax, c="blue", zetan=zetan, ms=1.4, prange="full")
    tracer.plot(ax=ax, c="red", zetan=zetan, ms=1.2, prange="full")
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    out_obj.plot_iota(
        ax=ax,
        xaxis="s",
    )
    tracer.plot_iota(ax=ax, xaxis="s", marker="x", ls="--")
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    out_obj.plot_iota(
        ax=ax,
        xaxis="R",
    )
    tracer.plot_iota(ax=ax, xaxis="R", marker="x", ls="--")
    plt.tight_layout()

    plt.show()
