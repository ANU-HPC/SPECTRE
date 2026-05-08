##################
Coding Conventions
##################

A code is generally written once but read many times. Consistent style
improves readability and makes collaboration easier. This page describes the
conventions to follow for Python and Fortran code in SPECTRE.

.. contents:: On this page
   :local:
   :depth: 1

Python
======

Style
-----

Python code in SPECTRE follows `PEP 8 <https://peps.python.org/pep-0008/>`_.
The specific settings are configured in ``pyproject.toml``:

- Line length: **100 characters**
- Indentation: **4 spaces**
- String quotes: **double quotes**

Linting and formatting with Ruff
---------------------------------

`Ruff <https://docs.astral.sh/ruff/>`_ is used for both linting and
formatting. Install the development dependencies to get it:

.. code-block:: bash

   pip install -e ".[dev]"

Run the formatter to auto-fix style issues:

.. code-block:: bash

   ruff format .

Run the linter to check for code quality issues:

.. code-block:: bash

   ruff check .

Both checks are enforced by the CI pipeline on every merge request. Make sure
your code passes both before pushing.

Docstrings
----------

Docstrings follow the `Google style
<https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
convention, as enforced by Ruff's ``pydocstyle`` plugin. Every public
function, method, and class should have a docstring. A minimal example:

.. code-block:: python

   def compute_flux(field: np.ndarray, area: float) -> np.ndarray:
       """Compute the flux of a field through a surface.

       Args:
           field: Array of field values at each point on the surface.
           area: Total surface area.

       Returns:
           Array of flux values, one per field component.

       Raises:
           ValueError: If ``field`` is empty.
       """
       ...

Use the ``Args``, ``Returns``, ``Raises``, and ``Note`` sections as needed.
Omit sections that do not apply.

Fortran
=======

Files and general rules
------------------------

- All Fortran files use the ``.F90`` (capital F) extension.
- The file name must match the program unit it contains. For example, module
  ``my_mod`` must be in file ``my_mod.F90``.
- Only one module per file.
- Every function or subroutine must be contained in a module.
- Line length is limited to **132 characters**.
- Use lowercase Fortran intrinsics.
- Indentation uses **2 spaces**. Never use the tab character.
- Do not put commented blank lines; use an empty line instead.
- Avoid trailing whitespace.

Module structure
----------------

All modules should follow this structure:

.. code-block:: fortran

   module my_mod
     ! First the Fortran intrinsic modules (in alphabetical order)
     use intrinsic_mod_a
     use intrinsic_mod_b

     ! Then third-party and local modules (in alphabetical order)
     use mod_a
     use mod_b

     ! Always use a single implicit none at the beginning
     implicit none (type, external)

     ! Everything should be private by default
     private
     public :: ! Add your list of public functions/subroutines

   contains
   end module my_mod

Style rules
-----------

- Do not use semicolons (``;``) for multiple statements per line.
- Always use the double colon after a type declaration:

  .. code-block:: fortran

     integer :: a
     real(dp) :: x, y

- Use modern comparison operators: ``==``, ``/=``, ``>``, ``<``, ``<=``,
  ``>=`` instead of ``.eq.``, ``.neq.``, ``.gt.``, ``.lt.``, ``.leq.``,
  ``.geq.``.
- Use spaced ``end`` keywords: ``end if``, ``end do``, ``end subroutine``,
  not ``endif``, ``enddo``, *etc*.
- Avoid non-standard Fortran extensions.
- Do not use Fortran keywords (*e.g.* ``data``) as variable names.
- Put a space before and after operators (``=``, ``.and.``, ``<=``, *etc*.)
  and punctuation (``.``, ``,``).
- Add spaces between mathematical operators to improve readability. Grouping
  related terms is allowed:

  .. code-block:: fortran

     a = 2*b + c   ! Acceptable — multiplication groups b and 2
     a = 2 * b + c ! Also acceptable

Comments
--------

Use ``!`` for inline and block comments. Prefer comments that explain *why*
the code does something rather than restating *what* it does. Place block
comments on a separate line directly above the code they describe:

.. code-block:: fortran

   ! Normalize the vector before computing the dot product to avoid
   ! floating-point overflow for large field values.
   norm = sqrt(sum(v**2))
   v = v / norm

Note that many editors can automate the enforcement of these conventions (*e.g.*
automatic trailing whitespace removal, indentation guides).
