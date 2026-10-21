####################################
Automatic differentiation with JAX
####################################

SPECTRE computes the analytic Jacobian of the interface force with respect to
the interface geometry. The optional module ``spectre.jax_wrapper`` exposes the
force and the equilibrium solve as `JAX <https://docs.jax.dev>`_ functions whose
derivatives use this Jacobian. SPECTRE can then be used inside codes that rely on
JAX automatic differentiation (``jax.grad``, ``jax.jvp``, ``jax.jacfwd``, ...)
without finite differences and without rewriting SPECTRE in JAX.

Installation
============

JAX is an optional dependency. Install it together with SPECTRE with

.. code-block:: bash

   pip install -e ".[jax]"

The wrapper works in double precision only. Enable 64-bit floats in JAX before
creating the wrapper:

.. code-block:: python

   import jax

   jax.config.update("jax_enable_x64", True)

Which function to use
=====================

The wrapper class :class:`spectre.jax_wrapper.SpectreJax` offers two entry
points:

``SpectreJax.force(x)``
   The interface force :math:`\mathbf{F}(x) = [B^2/2]` as a function of the
   packed interface geometry ``x``. Use it when the geometry itself is the
   variable, for example to include the force balance as a penalty term in a
   larger JAX objective.

``SpectreJax.equilibrium(param_force)``
   Returns a function ``p -> x*(p)`` that computes the equilibrium geometry for
   the parameters ``p`` by force minimization. Use it when the equilibrium is an
   intermediate result and you need derivatives with respect to the parameters,
   for example the sensitivity of the interface shapes to the pressure profile.

The parameters enter through ``param_force``, a JAX function that returns an
additive force term. ``SpectreJax.pressure_force`` provides the kinetic pressure
jump across each interface, so passing it differentiates the equilibrium with
respect to the volume pressures.

Example
=======

The example uses the cylindrical test case shipped with the tests and is run
from the repository root.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from spectre import SPECTRE, get_xinit_specwrap
   from spectre.jax_wrapper import SpectreJax

   test = SPECTRE("tests/wrapper/cyl_manyvol_test.toml", verbose=False)
   sj = SpectreJax(test, solver_kwargs=dict(verbose=0))

   # 1. Gradient of the force residual with respect to the interface geometry.
   x0 = jnp.asarray(get_xinit_specwrap(test))
   residual_grad = jax.grad(lambda x: 0.5 * jnp.sum(sj.force(x) ** 2))(x0)

   # 2. Equilibrium as a differentiable function of the volume pressures.
   x_star = sj.equilibrium(sj.pressure_force)
   p0 = jnp.array([1.0, 0.9, 0.8])  # one pressure per volume, scaled by pscale

   dx_dp = jax.jacfwd(x_star)(p0)  # shape (ngdof + 1, mvol)

   # 3. Any JAX objective evaluated on the equilibrium, here the sum of squares
   #    of the interface Fourier coefficients, can be differentiated and jitted.
   @jax.jit
   def objective(p):
       return jnp.sum(x_star(p) ** 2)

   value, grad = jax.value_and_grad(objective)(p0)

Each evaluation of ``x_star`` runs a full force minimization, warm-started from
the previous solution, so gradients cost about one equilibrium solve plus one
force Jacobian.

How it works
============

``force`` evaluates the Fortran code through ``jax.pure_callback`` and defines
its derivative with ``jax.custom_jvp`` as the product of the force Jacobian
with the tangent. Because that product is linear, JAX derives reverse mode
automatically.

``equilibrium`` solves :math:`\mathbf{F}(x) + g(p) = 0` in the least-squares
sense, where :math:`g` is ``param_force``. Its derivative follows from the
implicit function theorem,

.. math::
   \frac{\partial x^*}{\partial p} = -J^{+} \frac{\partial g}{\partial p},

where :math:`J^{+}` is the pseudo-inverse of the force Jacobian at
:math:`x^*`. The pseudo-inverse removes the component along the gauge null
space of the Jacobian; its cutoff is set with the ``rcond`` argument of
``SpectreJax``.

Limitations
===========

* **The derivative is exact only for a converged equilibrium.** If the force
  residual does not vanish at the solution, the derivative neglects a
  second-order term and is approximate. The error does not decrease in
  proportion to the residual. For the cylindrical test case (residual about
  1e-13) the gradient agrees with finite differences of re-solved equilibria to
  1e-8. For the QA test case ``tests/wrapper/qa_7vol_test.toml`` with
  ``pscale = 0.01`` the error is 13%, 19% and 4.5% at
  ``(mpol, ntor) = (3, 1), (5, 2), (7, 3)``, while the RMS residual falls from
  7.8e-2 to 1.9e-3. Check the residual at the solution before relying on the
  gradients.
* **Only additive parameters are supported.** The force Jacobian contains
  derivatives with respect to the interior interfaces only, so parameters inside
  the field solve, such as the plasma boundary, cannot be differentiated yet.
* **One wrapper per SPECTRE object, single-threaded.** SPECTRE keeps global
  Fortran state, so do not evaluate one wrapper from several threads, and do
  not share a SPECTRE object between wrappers.
