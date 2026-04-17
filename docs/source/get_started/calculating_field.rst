##############
Calculating Beltrami field
##############

For a given set of interfaces one can calculate the Beltrami field :math:`\nabla \times \mathbf{B} = \mu \mathbf{B}`, with a set of constraints specified using the ``Lconstraint`` flag. The interfaces used for field calculation are determined through the ``Linitialize`` flag.

The field for a configuration can be found by running the script ``scripts/calc_spectre_field.py`` with a ``.toml`` file. The output will be in the form of a ``.h5`` file containing the field on a grid, also including the poincare trace if it's requested in the input file. The output field can be analysed, as described in the section :doc:`postprocessing`

Alternatively, one can create a SPECTRE object and calculate the interface force :math:`\mathbf{F}_l = [p + B^2/2\mu_0]_l` which will trigger a field calculation, as the following code shows:

.. code-block:: python

   from spectre import Spectre, force_real, get_xinit_specwrap
   
   obj = SPECTRE("file.toml")

   interfaces = get_xinit_specwrap(obj)

   force = force_real(interfaces, obj)
   