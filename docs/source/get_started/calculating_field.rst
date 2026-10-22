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

With SI magnetic fields and pressure stored as ``mu0*p``, ``force_real``
returns the jump in ``B**2/2 + mu0*p`` in tesla squared. Divide by ``mu0``
to obtain a pressure imbalance in pascals. A residual normalized by a reference
field squared requires a separate, explicit division by that reference value.

For interface relaxation with ``Force_Minimizer``, ``[minimization]`` accepts
``x_scale = "jac"`` to scale the trust region by inverse Jacobian column norms.
The default is ``x_scale = 1.0``. This can improve conditioning when interface
degrees of freedom have different sensitivities. The optimizer's ``ftol``,
``xtol`` and ``gtol`` test progress or stationarity, so its success flag alone
does not establish an acceptable force residual. Iteration limits and requested
interruptions return ``success = False`` while retaining the accepted state.

The Fourier interface Jacobian selects the active packed coordinates directly,
avoiding a dense multiplication by an identity matrix. Dense field solves also
use BLAS for LU factorization and back substitution, so forcing BLAS to one
thread can limit throughput even when OpenMP uses several threads. Measure
both field and Jacobian timings when selecting their thread counts.
