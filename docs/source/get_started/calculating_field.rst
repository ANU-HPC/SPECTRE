##############################
Calculating the Beltrami field
##############################

For a given set of interfaces one can calculate the Beltrami field
:math:`\nabla \times \mathbf{B} = \mu \mathbf{B}`, with a set of constraints
specified using the ``lconstraint`` flag. The interfaces used for field
calculation are determined through the ``linitialize`` flag.

The field for a configuration can be found by running the script
``scripts/run/calc_spectre_field.py`` with a TOML file. The output will be in the
form of a HDF5 file containing the field on a grid, also including the poincare
trace if it is requested in the input file. The output field can be analysed,
as described in the section :doc:`/get_started/postprocessing`

Alternatively, one can create a ``SPECTRE`` object and calculate the interface
force :math:`\mathbf{F}_l = [p + B^2/2\mu_0]_l` which will trigger a field
calculation, as the following code shows:

.. code-block:: python

   from spectre import SPECTRE

   obj = SPECTRE("file.toml")
   force = obj.force_real()

The current packed interface degrees of freedom are available as ``obj.xin``.
Assigning to this property updates the live interfaces. Force and angle helpers
are also bound to the object, and default to ``obj.xin``. Pass an explicit
interface vector when evaluating a different state, for example
``obj.force_real(other_xin)``.

Call ``obj.compute_fields()`` when only the in-memory Beltrami field is needed.
``obj.field_is_current`` reports whether that field corresponds to the live
interfaces.

The ``obj.inspect`` facade provides non-mutating numerical and visualization
helpers. For example:

.. code-block:: python

   obj.inspect.print_errors()
   obj.inspect.plot_infaces()
   obj.inspect.plot_grid(interactive=True)
   obj.inspect.plot_worst_jacobian()

Beltrami-error and field-coloured 3D plots require a current field. Call
``obj.compute_fields()`` or ``obj.force_real()`` first. Three-dimensional
interface rendering additionally requires the optional ``pyvista`` package.
See :doc:`inspecting` for the complete inspection workflow and example plots.
