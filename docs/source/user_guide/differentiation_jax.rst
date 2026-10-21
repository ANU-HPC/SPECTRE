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

Example: sensitivity of the Shafranov shift
===========================================

Raising the plasma pressure in a tokamak pushes the nested flux surfaces
outward, towards the low-field side. This Shafranov shift grows with pressure
and eventually limits how much pressure a configuration can hold. How strongly
the shift reacts to a change in pressure is exactly the kind of question a
design study or an optimizer asks: it is the derivative of a geometric quantity
of the equilibrium with respect to a parameter of the force balance.

Without derivatives one would solve neighbouring equilibria and difference
them. With ``equilibrium`` the shift is an ordinary JAX function of the
pressure, and its sensitivity comes from ``jax.jacfwd`` at the cost of one
extra linear solve with the force Jacobian.

The example scales a fixed pressure profile by its value ``p1`` in the core
volume and measures the shift of each interface by its midplane centre,
:math:`(R_\text{outboard} + R_\text{inboard})/2`. For a stellarator-symmetric
interface this is the sum of the cosine coefficients :math:`R_{m0}` with even
:math:`m`. The midplane centre depends only on the surface, not on how the
poloidal angle is parametrized along it.

.. code-block:: python

   import jax
   import jax.numpy as jnp

   jax.config.update("jax_enable_x64", True)

   from spectre import SPECTRE
   from spectre.jax_wrapper import SpectreJax

   test = SPECTRE("tokamak.toml", verbose=False)
   sj = SpectreJax(test)
   x_star = sj.equilibrium(sj.pressure_force)

   profile = jnp.array([1.0, 0.5, 0.0])  # pressure in each volume, per unit p1
   mn = test.allglobal_mod.mn
   even_m = jnp.asarray(test.allglobal_mod.im[:mn] % 2 == 0)

   def midplane_centre(p1):
       rbc, _ = sj.interface_modes(x_star(p1 * profile))
       return jnp.sum(jnp.where(even_m, rbc, 0.0), axis=1)  # one value per interface

   p1 = 1e-3
   centre = midplane_centre(p1)
   sensitivity = jax.jacfwd(midplane_centre)(p1)  # d(centre)/d(p1) for each interface

The boundary is fixed, so ``sensitivity`` is also the derivative of the shift
relative to the boundary. It is positive, the shift grows outward with
pressure, and it is largest for the innermost interface. Evaluating it along a
range of ``p1`` shows how the response stiffens or softens as the pressure
rises. Because ``midplane_centre`` is a JAX function, it can be composed with
other JAX code, e.g. a penalty on the shift inside a larger optimization, and
differentiated with ``jax.grad`` in the same way.

When the geometry itself is the variable rather than a parameter,
``SpectreJax.force`` can be differentiated directly, e.g.
``jax.grad(lambda x: jnp.sum(sj.force(x) ** 2))(x0)``.

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
* **The equilibrium must be well posed.** If a deformation of the interfaces
  meets almost no restoring force, e.g. the Shafranov shift in a tokamak with a
  very weak poloidal field, the force Jacobian is nearly singular, the forward
  solve converges poorly and the derivatives are not reliable. A sensitivity
  that changes with the ``rcond`` argument of ``SpectreJax`` is a warning sign.
* **Only additive parameters are supported.** The force Jacobian contains
  derivatives with respect to the interior interfaces only, so parameters inside
  the field solve, such as the plasma boundary, cannot be differentiated yet.
* **One wrapper per SPECTRE object, single-threaded.** SPECTRE keeps global
  Fortran state, so do not evaluate one wrapper from several threads, and do
  not share a SPECTRE object between wrappers.
