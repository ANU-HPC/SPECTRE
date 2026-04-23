#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from spectre import SPECTRE, SPECTREout

zetan = 0.1875

# Test case 1: A fixed-b. stellarator with 7 volumes
fname = "fixb_7vol_wrp.sp"
num_transits = 1000
num_trajectories = 20
out_obj = SPECTREout('fixb_7vol.sp.h5')

# # Test case 2: A free-b. stellarator with 1 plasma and 1 vacuum volume (some points leave the plasma)
# fname = "freeb_2vol_wrp.sp"
# num_transits = 300
# num_trajectories = 10
# out_obj = SPECTREout('freeb_2vol.sp.h5')

## Generate custom poincare and save to the usual .h5 file
test = SPECTRE.from_input_file(input_file=fname)

test.input_list_mod.nppts = 0 # Disable old poincare tracing
test.run(save_output=True)

num_phi_planes=test.allglobal_mod.nz
init_pts = [
    np.array(
        [[(-1.+1e-14) + itrj * (2.0-2e-14)/num_trajectories, 0.] for itrj in range(0 if j>0 else 1, num_trajectories+1)]
    ) for j in range(test.allglobal_mod.mvol)
]

test.trace_fieldlines(init_pts, num_phi_planes=num_phi_planes, num_transits=num_transits, method='RK45', rtol=1.0e-7, atol=1.0e-7)

if MPI.COMM_WORLD.Get_rank() == 0:
    test.tracer.save_h5(test.input_file+".sp.h5")

# Plot results, and compare to old poincare
if MPI.COMM_WORLD.Get_rank() == 0:

    from spectre import Field_Tracer

    tracer = Field_Tracer.init_from_h5(test.input_file+".sp.h5")

    fig, ax = plt.subplots(1, 1, figsize=(9,9))
    out_obj.plot_infaces(ax=ax, color='k', zetan=zetan, prange='full')
    out_obj.plot_poincare(ax=ax, c='blue', zetan=zetan, ms=1.4, prange='full')
    tracer.plot(ax=ax, c='red', zetan=zetan, ms=1.2, prange='full')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    out_obj.plot_iota(ax=ax, xaxis='s', )
    tracer.plot_iota(ax=ax, xaxis='s', marker='x', ls='--')
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    out_obj.plot_iota(ax=ax, xaxis='R', )
    tracer.plot_iota(ax=ax, xaxis='R', marker='x', ls='--')
    plt.tight_layout()

    plt.show()
