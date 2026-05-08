############
Git Workflow
############

SPECTRE uses a `GitLab <https://gitlab.com/spectre-eq/spectre>`_ merge
request-based workflow. All contributions go through a branch and merge request
cycle — direct commits to ``main`` are not allowed.

Overview
========

The typical contribution cycle is:

1. `Open or pick up a GitLab issue`_
2. `Create a branch`_
3. `Make commits`_
4. `Open a Merge Request`_
5. `Pass code review and CI`_
6. `Merge`_

.. _Open or pick up a GitLab issue:

Open or pick up a GitLab issue
===============================

Every non-trivial contribution should be tied to a GitLab issue. Issues are
used to track bugs, feature requests, and planned work. Before starting, check
whether an issue already exists for the work you intend to do. If not, open
one and describe the problem or feature clearly.

.. _Create a branch:

Create a branch
===============

If you are working from a GitLab issue, the recommended approach is to create
a **draft merge request directly from the issue page**. GitLab will
automatically create a branch tied to that issue and open a draft MR. The
draft MR acts as a discussion forum for the work-in-progress: you can post
questions, share early results, and receive feedback before the implementation
is complete. Mark the MR as ready for review only when the work is done.

Alternatively, create a branch manually from an up-to-date ``main``:

.. code-block:: bash

   git switch main
   git pull origin main
   git switch -c my-descriptive-branch-name

Use a short, descriptive branch name that conveys what the branch does (e.g.
``fix-coordinate-transform`` or ``add-vmec-output``). There is no strict naming
convention, but clarity is preferred.

.. note::

   The ``no-commit-to-branch`` pre-commit hook prevents accidental direct
   commits to ``main``. See :doc:`pre_commit` for details.

.. _Make commits:

Make commits
============

Keep commits small and focused. Each commit should represent one logical
change. Write commit messages in the imperative mood and summarize *what* the
commit does and *why*.

A commit message must start with a short summary line, followed by a blank
line, and then a more detailed explanation if needed:

.. code-block:: text

   Fix off-by-one error in surface indexing

   The loop upper bound was inclusive, causing the last surface to be
   processed twice. Changed to exclusive to match the array layout.

.. tip::

   If you cannot summarize a commit in a single line, the commit is probably
   doing too many things at once. Consider splitting it into smaller,
   more focused commits.

.. _Open a Merge Request:

Open a Merge Request
====================

Push your branch to GitLab and open a Merge Request (MR) targeting ``main``:

.. code-block:: bash

   git push origin my-descriptive-branch-name

Then go to the GitLab project page and click **New merge request**. In the MR
description:

- Summarize what the MR does and why.
- Reference the related issue using the ``Closes #<issue-number>`` keyword so
  GitLab closes it automatically on merge.
- List any important decisions or trade-offs made during implementation.

.. _Pass code review and CI:

Pass code review and CI
========================

Before a MR can be merged:

- **CI must pass.** The pipeline runs automated checks including Ruff linting
  and formatting of the Python code. Ensure your changes do not introduce any
  linting errors (see :doc:`coding_convention`).
- **At least one reviewer must approve.** Assign a reviewer in GitLab. Address
  all review comments before requesting re-review.

.. _Merge:

Merge
=====

Once the MR is approved and CI is green, the branch can be merged into
``main``. The branch is deleted automatically after merge.
