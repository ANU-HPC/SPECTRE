#############################
Inspecting a live calculation
#############################

Every :class:`spectre.SPECTRE` instance exposes an ``inspect`` facade for
examining the geometry, field, and force data currently held in memory. The
inspection methods do not change the live interfaces or numerical solution.

Start with a configured instance:

.. code-block:: python

   from spectre import SPECTRE

   equilibrium = SPECTRE("input.toml", verbose=False)

Geometry inspection
===================

Geometry plots are available immediately and default to the current
``equilibrium.xin``:

.. code-block:: python

   equilibrium.inspect.plot_infaces()
   equilibrium.inspect.plot_grid()
   equilibrium.inspect.plot_jacobian()

Pass ``interactive=True`` to ``plot_infaces``, ``plot_grid``, or
``plot_jacobian`` to add a slider for the normalized toroidal angle ``zetan``.
An explicit packed interface vector can be supplied as the first argument to
inspect another geometry without installing it as the live state.

Finding the worst Jacobian
--------------------------

``plot_worst_jacobian`` scans toroidal planes over one field period and plots
the plane containing the least healthy coordinate Jacobian. The central volume
is excluded from the search by default because its Jacobian vanishes at the
coordinate-axis singularity. It remains visible in the plot for context.

.. code-block:: python

   equilibrium.inspect.plot_worst_jacobian()

.. figure:: /_static/inspect/worst_jacobian.png
   :alt: Cross-section selected by plot_worst_jacobian, with the worst cell circled.
   :align: center
   :width: 85%

   The selected toroidal cross-section. The title reports the minimum
   Jacobian, volume, and normalized toroidal angle. The green circle marks the
   worst cell. Negative Jacobians, when present, are shown in magenta.

Use ``vvol`` to restrict the volumes in the search, or ``nzeta`` to control the
number of toroidal planes sampled:

.. code-block:: python

   equilibrium.inspect.plot_worst_jacobian(vvol=[1, 2], nzeta=32)

Field and force inspection
==========================

Field diagnostics require a Beltrami field that matches the current
interfaces. Compute it directly when no force residual is needed:

.. code-block:: python

   equilibrium.compute_fields()
   assert equilibrium.field_is_current
   equilibrium.inspect.beltrami_errors()
   equilibrium.inspect.plot_beltrami_error(vvol="all")

Alternatively, ``equilibrium.force_real()`` computes both the field and the
force residual. This enables the combined error summary:

.. code-block:: python

   force = equilibrium.force_real()
   equilibrium.inspect.print_errors()

Finding the worst Beltrami error
--------------------------------

``plot_worst_beltrami`` scans every native toroidal plane in the selected
volumes. With the default ``component=None``, cells are ranked by the largest
component of :math:`\lvert\nabla\times\mathbf{B}-\mu\mathbf{B}\rvert`, and the
responsible component is plotted.

.. code-block:: python

   equilibrium.compute_fields()
   equilibrium.inspect.plot_worst_beltrami()

.. figure:: /_static/inspect/worst_beltrami.png
   :alt: Cross-section selected by plot_worst_beltrami, with the worst cell circled.
   :align: center
   :width: 85%

   The selected component and toroidal cross-section. The logarithmic colour
   scale shows the spatial error distribution, and the green circle marks the
   cell that determined the selection.

A specific component can be selected by index or by ``"s"``, ``"t"``, or
``"z"``. Small values can be clipped from the colour range with ``min_val``:

.. code-block:: python

   equilibrium.inspect.plot_worst_beltrami(component="s", min_val=1.0e-8)

Three-dimensional interfaces
============================

With the optional ``pyvista`` dependency installed, the live field can also be
used to colour three-dimensional interface surfaces:

.. code-block:: python

   equilibrium.compute_fields()
   plotter = equilibrium.inspect.plot_infaces_3d(color_by="force")

Supported quantities include ``"force"``, ``"pressure_jump"``,
``"b2_inner"``, ``"b2_outer"``, ``"current"``, and ``"angle"``.
