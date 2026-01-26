# Contributing to the SPECTRE code

## Coding guidelines

## Coding style

The following rules are mostly there to increase the uniformity of the code as
well as its robustness. Remember that a code is generally written once, but read
many times. It is therefore important for the general quality of the code to
enforce a few cosmetic rules that will increase readabilty.

### Files and general comments
  - All the Fortran files use the `.F90` (capital F) extension
  - File name is the same as the program unit it contains. For example, module
  `my_mod` should be in file `my_mod.F90`.
  - Only one module per file.
  - Every function or subroutine must be contained into a module
  - Line length is limited to 132 characters.
  - Use lower case Fortran intrinsics.

  - Indentation is done using **2 spaces**. Do not use the tab character.
  - Do not put commented blank lines, just an empty line.

### Fortran style
  - All the modules should have the following structure
  ```f90
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
  ```
  - Do not use semi colons (;) for indenting or to have multiple statements per line.
  - Always use the double colon after a type declaration, _e.g._ `integer :: a`
  - Use modern comparison operators, _i.e._ `==`, `/=`, `>`, `<`, `<=`, and `>=`
  instead of `.eq.`, `.neq.`, `.gt.`, `.lt.`, `.geq.`, and `.leq.`.
  - Use spaced ends, _e.g._ `end if` instead of `endif`.
  - Avoid non-standard Fortran
  - Fortran keywords, _e.g._ data, must not be used as variable names
  - Put a space before and after an operator (=, .and., <=, etc.), a punctuation sign ('.', ',', etc.)
  - Add spaces between mathematical operators to lighten the code. Note that you
    can group some terms. For example, use `a = 2 * b + c` instead of `a = 2*b+c`
  - Avoid trailing whitespaces

Note that various editors allow you to automatize those coding conventions.