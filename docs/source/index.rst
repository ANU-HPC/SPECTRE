.. SPECTRE documentation master file, created by
   sphinx-quickstart on Tue Sep 16 10:18:03 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
#####################
SPECTRE documentation
#####################

SPECTRE (Stepped Pressure Equilibrium Code The Revised Edition) is a tool for
calculating general 3D MHD equilibrium solutions with generalized magnetic
topology. This is done in the framework of the Multi-Region relaxed MHD (MRxMHD)
model which solves plasma subdivided into stepped-pressure subdomains,
separated by the ideal interfaces. The code is built upon a previous code
`SPEC <https://github.com/PrincetonUniversity/SPEC>`_, but includes significant
improvements.

.. toctree::
   :maxdepth: 2
   :caption: Get started

   get_started/installation
   get_started/generating_input
   get_started/calculating_field
   get_started/postprocessing
   get_started/citing

.. toctree::
   :maxdepth: 2
   :caption: User guide

   user_guide/equilibrium_calculation/index
   user_guide/analysis/index
   user_guide/differentiation_jax


.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
