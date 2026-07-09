############
Git Workflow
############

SPECTRE uses a `GitLab <https://gitlab.com/spectre-eq/spectre>`_ merge
request-based workflow. Contributions go through a personal **fork** of the
project and a merge request targeting the upstream ``main`` branch. Contributors
do not have write access to the canonical repository, so you cannot push
branches to it or commit to ``main`` directly — all work happens in your fork.

Overview
========

The typical contribution cycle is:

1. `Open or pick up a GitLab issue`_
2. `Fork the repository`_
3. `Clone your fork`_
4. `Create a branch`_
5. `Make commits`_
6. `Keep your fork in sync`_
7. `Open a Merge Request`_
8. `Pass code review and CI`_
9. `Merge`_

.. _Open or pick up a GitLab issue:

Open or pick up a GitLab issue
===============================

Every non-trivial contribution should be tied to a GitLab issue. Issues are
used to track bugs, feature requests, and planned work. Before starting, check
whether an issue already exists for the work you intend to do. If not, open
one and describe the problem or feature clearly. Leave a comment on the issue
or assign it to yourself to let others know you are picking it up, so effort is
not duplicated.

.. _Fork the repository:

Fork the repository
===================

Since you cannot push to the canonical repository, start by creating your own
copy. On the `GitLab project page <https://gitlab.com/spectre-eq/spectre>`_,
click **Fork** and create the fork under your own namespace. This gives you a
personal copy of SPECTRE that you have full write access to.

.. note::

   For the workflow to work smoothly, keep your fork **public**. With a public
   fork, GitLab automatically pre-selects the upstream ``spectre-eq/spectre``
   ``main`` branch as the target when you open a merge request. If you make your
   fork **private**, you will need to manually select the target repository and
   branch each time you open a merge request.

.. _Clone your fork:

Clone your fork
==============

Clone your fork locally and add the canonical repository as a second remote so
you can pull in upstream changes:

.. code-block:: bash

   git clone git@gitlab.com:<your-username>/spectre.git
   cd spectre
   git remote add upstream https://gitlab.com/spectre-eq/spectre.git

By convention, ``origin`` now points to **your fork** (where you push your work)
and ``upstream`` points to the **canonical SPECTRE repository** (where you pull
the latest changes from and where merge requests are opened).

.. _Create a branch:

Create a branch
==============

Create a branch from an up-to-date ``main``. Sync your local ``main`` from
``upstream`` first so your branch starts from the latest state:

.. code-block:: bash

   git switch main
   git pull upstream main
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

.. _Keep your fork in sync:

Keep your fork in sync
=====================

For long-running branches, the upstream ``main`` may move ahead while you work.
Periodically fetch the latest changes and rebase your branch on top of them to
keep it up to date and avoid a messy merge later:

.. code-block:: bash

   git fetch upstream
   git rebase upstream/main

Resolve any conflicts as they arise, then continue the rebase.

.. _Open a Merge Request:

Open a Merge Request
====================

Push your branch to **your fork** (the ``origin`` remote):

.. code-block:: bash

   git push origin my-descriptive-branch-name

GitLab prints a link in the terminal output that takes you directly to the
**New merge request** page. The merge request should target the ``main`` branch
of the canonical ``spectre-eq/spectre`` repository.

When you open the merge request, GitLab automatically applies the project's
`default template
<https://gitlab.com/spectre-eq/spectre/-/blob/main/.gitlab/merge_request_templates/Default.md>`_
(``.gitlab/merge_request_templates/Default.md``). Fill in each sections that are
relevant to your merge request and delete the others:

- **What does this MR do and why** : summarize the change and its motivation.
- **How to set up and validate locally** : numbered steps a reviewer can follow.
- **MR acceptance checklist** : tick the items that apply.
- **AI usage declaration** : declare any AI assistance used. See :doc:`ai_policy`
  for the project's policy on AI-assisted contributions.

Also reference the related issue using the ``Closes #<issue-number>`` keyword so
GitLab closes it automatically on merge, and note any important decisions or
trade-offs made during implementation.

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

Once the MR is approved and CI is green, a maintainer merges it into ``main``.
You can then safely delete the branch in your fork.
