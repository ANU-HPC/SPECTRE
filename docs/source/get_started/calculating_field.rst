##############################
Calculating the Beltrami field
##############################

For a given set of interfaces one can calculate the Beltrami field
:math:`\nabla \times \mathbf{B} = \mu \mathbf{B}`, with a set of constraints
specified using the ``lconstraint`` flag. The interfaces used for field
calculation are determined through the ``linitialize`` flag.

The field for a configuration can be found by running the script
``scripts/calc_spectre_field.py`` with a TOML file. The output will be in the
form of a HDF5 file containing the field on a grid, also including the poincare
trace if it is requested in the input file. The output field can be analysed,
as described in the section :doc:`/get_started/postprocessing`

Alternatively, one can create a ``SPECTRE`` object and calculate the interface
force :math:`\mathbf{F}_l = [p + B^2/2\mu_0]_l` which will trigger a field
calculation, as the following code shows:

.. code-block:: python

   from spectre import Spectre, force_real, get_xinit_specwrap

   obj = SPECTRE("file.toml")

   interfaces = get_xinit_specwrap(obj)

   force = force_real(interfaces, obj)

Optional beltrami_jax backend
=============================

SPECTRE also exposes an experimental Python-side replacement seam for the local
Beltrami solve. If ``beltrami_jax`` is installed in the same environment, the
same TOML/interface geometry can be solved with JAX and the resulting
``Ate/Aze/Ato/Azo`` coefficients can be injected back into SPECTRE memory:

.. code-block:: python

   from spectre import SPECTRE, get_vec_pot_flat

   obj = SPECTRE.from_input_file("file.toml")
   result = obj.solve_beltrami_jax(solve_local_constraints=False)

   ate, aze, ato, azo = get_vec_pot_flat(obj)
   print(result.max_relative_residual_norm)

For Python force workflows, the initial force check can also bypass SPECTRE's
Fortran Beltrami solve and inject the JAX coefficients before evaluating the
existing SPECTRE force diagnostic:

.. code-block:: python

   from spectre import SPECTRE, force_real, get_xinit_specwrap

   obj = SPECTRE.from_input_file("file.toml")
   x = get_xinit_specwrap(obj)
   f = force_real(x, obj, beltrami_backend="jax")

The command-line runner exposes the same experimental switch:

.. code-block:: bash

   calc_spectre_field.py input.toml --beltrami-backend jax

This path is intentionally narrow. SPECTRE still owns input handling, geometry
state, free-boundary normal-field updates, force evaluation, and output, while
``beltrami_jax`` owns the TOML/interface-geometry Beltrami solve and the
supported branch constraint updates. The hook is designed so validation can
compare the injected vector-potential coefficients against the current Fortran
backend before making it the default. The current implementation is serial-only
and does not yet replace SPECTRE's force Jacobian.
