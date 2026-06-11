# Contributing to the SPECTRE code

Any contribution to the SPECTRE code is welcome.

Before starting, we invite you to read our documentation on
[contributions](https://spectre.readthedocs.io/en/latest/contributing/index.html),
which covers the Git workflow, coding conventions, pre-commit hooks, and AI policy
about contributions. Please follow these guidelines so that your contribution can be
reviewed and accepted quickly.

## Getting started

1. Install the development version:

   ```bash
   pip install -e .[dev]
   pre-commit install
   ```

2. Create a branch (direct commits to `main` are not allowed) and open a Merge
   Request against `main` on
   [GitLab](https://gitlab.com/spectre-eq/spectre). Reference the related issue
   with `Closes #<issue-number>` so it is closed automatically on merge.

See the [contribution docs](https://spectre.readthedocs.io/en/latest/contributing/index.html)
for the full workflow.
