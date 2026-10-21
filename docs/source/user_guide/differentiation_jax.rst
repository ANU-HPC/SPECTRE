####################################
Automatic differentiation with JAX
####################################

SPECTRE computes the analytic Jacobian of the interface force with respect to
the interface geometry. The optional module ``spectre.jax_wrapper`` exposes the
force and the equilibrium solve as `JAX <https://docs.jax.dev>`_ functions whose
derivatives use this Jacobian. SPECTRE can then be used inside codes that rely on
JAX automatic differentiation (``jax.grad``, ``jax.jvp``, ``jax.jacfwd``, ...)
without finite differences and without rewriting SPECTRE in JAX.

Setup
=====

The wrapper requires JAX, which is not installed with SPECTRE. It works in
double precision only. Enable 64-bit floats in JAX before creating the wrapper:

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

Example: reconstructing the pressure profile
============================================

Suppose the interface shapes of an equilibrium are known, for example from
measurements or from a reference calculation, but the pressure in each volume
is not. Reconstructing the pressures means finding the ``p`` whose equilibrium
``x*(p)`` matches the known geometry ``x_target``. That is an optimization
problem, and gradient-based optimizers need the derivative of the mismatch with
respect to ``p``, which passes through the equilibrium solve.

With ``equilibrium`` the mismatch is an ordinary JAX function, so its gradient
comes from ``jax.grad`` and can be handed to any optimizer:

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from scipy.optimize import minimize

   jax.config.update("jax_enable_x64", True)

   from spectre import SPECTRE
   from spectre.jax_wrapper import SpectreJax

   sj = SpectreJax(SPECTRE("file.toml", verbose=False))
   x_star = sj.equilibrium(sj.pressure_force)

   # The known geometry. Here it is generated from a pressure profile that the
   # optimizer is not told about; in practice it would come from elsewhere.
   x_target = x_star(jnp.array([1.0, 0.7, 0.4]))

   @jax.jit
   def mismatch(p):
       return jnp.sum((x_star(p) - x_target) ** 2)

   def mismatch_and_grad(p):
       value, grad = jax.value_and_grad(mismatch)(jnp.asarray(p))
       return float(value), jax.device_get(grad)

   # The mismatch is small in absolute terms, so tighten the default tolerances.
   p_guess = jnp.array([1.0, 0.9, 0.8])
   result = minimize(
       mismatch_and_grad, p_guess, jac=True, method="L-BFGS-B",
       options=dict(ftol=1e-20, gtol=1e-14),
   )

The optimizer recovers the pressure *jumps* between neighbouring volumes. The
overall pressure level stays undetermined, because only the jumps enter the
force balance.

Every evaluation of ``x_star`` runs a full force minimization, warm-started from
the previous solution, so each gradient costs about one equilibrium solve plus
one force Jacobian.

When the geometry itself is the variable, ``SpectreJax.force`` can be used in
the same way, e.g. ``jax.grad(lambda x: jnp.sum(sj.force(x) ** 2))(x0)``.

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
  second-order term and is approximate, and the error does not decrease in
  proportion to the residual. Check the residual at the solution before relying
  on the gradients.
* **Only additive parameters are supported.** The force Jacobian contains
  derivatives with respect to the interior interfaces only, so parameters inside
  the field solve, such as the plasma boundary, cannot be differentiated yet.
* **One wrapper per SPECTRE object, single-threaded.** SPECTRE keeps global
  Fortran state, so do not evaluate one wrapper from several threads, and do
  not share a SPECTRE object between wrappers.
