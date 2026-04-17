################
Pre-commit Hooks
################

`pre-commit <https://pre-commit.com/>`_ is a framework for running automated
checks before each git commit. SPECTRE ships a ``.pre-commit-config.yaml``
that catches common mistakes before they reach the repository.

Installation
============

Install the ``pre-commit`` tool and activate the hooks for your local clone:

.. code-block:: bash

   pip install pre-commit
   pre-commit install

After running ``pre-commit install``, the hooks run automatically every time
you execute ``git commit``.

Configured hooks
================

The following hooks are enabled:

``check-toml``
   Validates all TOML files (*e.g.* ``pyproject.toml``) for syntax errors.

``detect-private-key``
   Scans staged files for private key patterns and blocks the commit if any
   are found, preventing accidental exposure of credentials.

``end-of-file-fixer``
   Ensures every file ends with exactly one newline character.

``no-commit-to-branch``
   Prevents direct commits to the ``main`` branch. All changes to ``main``
   must go through a merge request (see :doc:`git_workflow`).

``trailing-whitespace``
   Strips trailing whitespace from all lines.

Running hooks manually
======================

You can run all hooks against every file at any time without making a commit:

.. code-block:: bash

   pre-commit run --all-files

This is useful to check the full codebase after pulling in upstream changes or
before opening a merge request.

Bypassing hooks
===============

If you need to commit without running the hooks (*e.g.* to save a work-in-progress
that does not yet pass all checks), you can skip them with:

.. code-block:: bash

   git commit --no-verify

.. warning::

   Use this sparingly. The CI pipeline will still enforce all checks on your
   merge request, so any issues will need to be resolved before the MR can be
   merged.
