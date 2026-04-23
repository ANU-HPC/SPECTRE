# SPDX-FileCopyrightText: 2026 École Polytechnique Fédérale de Lausanne - Swiss Plasma Center
#
# SPDX-License-Identifier: MIT

import numpy as np
from scipy.optimize import root

# TODO: Add funcitonality to find island fixed points and add tests for that as well.
# Can be done by changing ntor

def _return_map(sth, obj, vol=0, ntor=1):
    """Trace a field line starting from (s, th) for ntor turns in volume vol, and return the difference in (r, z) between the start and end points."""
    
    s, th = sth

    large = 1.0e2

    if s > 1. or s < -1:
        return np.array([large, large])

    sth, RZ, iota, flag = obj.trace_one_fieldline(np.array([s, th]), vol, num_phi_planes=1, num_transits=1+ntor, )

    if flag:
        return np.array([large, large])
    
    r_start, z_start = obj.get_rz_coords(vol, s, th, 0.0)

    return np.array([RZ[0][-1] - r_start, RZ[1][-1] - z_start], dtype=float)


def find_axis(obj, max_repeats=0, guess=None, plot=False, verbose=False, name=""):
    """Finds the magnetic axis using root-finding and _return_map based on custom field line tracing. If intial guess doesn't work, retries max_repeats times with random guesses."""
    
    small = 1e-7

    vol = 0
    guess = [-0.9, 0.0] if guess is None else guess
    
    for it in range(max_repeats+1):
        if it > 0:
            guess = np.random.rand(2)*(1-2*small) + small
            guess[0] = guess[0]*2.0 - 1.0
            guess[1] = guess[1] * 2*np.pi
        fp = root(_return_map, x0=guess, args=(obj,), tol=1e-10, options=dict(xtol=1e-10) )

        if fp.success:
            break
    
    sax, thax = fp.x
    rax, zax = obj.get_rz_coords(vol, sax, thax, 0.)

    if verbose:
        print(f"find_axis for {name}: {it} iterations\nfp {fp}\nrax {rax:5.6f}  zax {zax:5.6f}  s {sax:5.6f}  th {thax:5.6f}")
    
    if plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        obj.plot_infaces(plot_axis=True, ax=ax)
        obj.plot_poincare(ax=ax)
        ax.plot(rax, zax, 'bX', ms=2)

    return rax, zax, sax, thax