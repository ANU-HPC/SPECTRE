#######################
Equilibrium calculation
#######################

An equilibrium in MRxMHD simultaneously satisfies the Beltrami equation and the
interface force balance condition:

.. math::
    \nabla \times \mathbf{B} = \mu \mathbf{B} \\
    \mathbf{F}_l = [p + B^2/2\mu_0]_l

An equilibrium is found by minimizing the Force in an iterative loop which
involves solving the Beltrami equation and adjusting interface geometry.

In a free-boundary calculation (``lfreebound = 1``) the outermost interface — the plasma
boundary — is itself part of the unknowns; the formulation and the available vacuum solvers
are described in :doc:`/numerics/free_boundary`.

The minimization is performed using the script ``scripts/calc_spectre_eq.py``
which takes a TOML file as input. The output will be an HDF5 file containing the
optimal interface geometry, along with the resulting Beltrami field, and other
postprocessing information. The output can be analyzed as described in the
:doc:`postprocessing </get_started/postprocessing>` section. During force
minimization, the output is piped to a text file, where each iteration of the
minimization is logged, along with extra information.

.. toctree::
   :maxdepth: 1
   :caption: Example calculations

   qa_vac
   w7x_freeb_vac
