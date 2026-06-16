# AGENTS.md

Instructions for AI coding assistants and agents operating in the SPECTRE
repository. They apply to any automated tool (Copilot, Cursor, Codex, Claude,
ChatGPT, …) used while preparing a contribution.

**Read this first:** SPECTRE has a policy on AI-assisted contributions. See
[AI policy](docs/source/contributing/ai_policy.rst). The points below restate
the parts that constrain how you, an AI tool, may be used. They are not
optional.

## Non-negotiable rules

- **A human is responsible for every line.** You assist; you are not a
  contributor and cannot take responsibility for the result. Everything you
  produce must be read, understood, and validated by the human before it is
  submitted.
- **No autonomous contributions.** Do not open, author, or push merge requests
  on your own. Do not commit on the human's behalf without an explicit,
  in-the-moment instruction. SPECTRE does not accept MRs authored by agents,
  bots, or pipelines.
- **No fully AI-written contributions.** Help with parts of a change; do not
  generate an entire contribution to be submitted as-is.
- **Never commit directly to `main`.** Work on a branch and open a merge request
  against `main`. A pre-commit hook blocks commits to `main` — do not try to
  bypass it.
- **Never skip or weaken checks.** Do not use `--no-verify`, do not disable
  pre-commit hooks, and do not silence linters to make code pass.
- **Copyright.** Do not reproduce code verbatim from training data or external
  sources under an incompatible license. Flag to the human anything that may
  carry licensing constraints.
- **Disclosure.** Any AI assistance must be declared in the merge request. The
  default MR template at `.gitlab/merge_request_templates/Default.md` contains
  an "AI usage declaration" checklist — it must be filled in by the human.

## Make your output reviewable

The easier your output is to review, the more useful you are. Prefer small,
focused changes. Explain non-obvious choices. Do not introduce dependencies,
abstractions, or files that were not asked for.

## Coding conventions (so generated code fits)

Full details: [coding conventions](docs/source/contributing/coding_convention.rst).
Conventions and build details for this repo are also in [CLAUDE.md](CLAUDE.md).

**Python**
- Lint and format with Ruff; isort import ordering.
- Line length 100, 4-space indent, double quotes.
- Google-style docstrings on every public function, method, and class.

**Fortran (`.F90`)**
- 2-space indent (never tabs), 132-character line limit.
- One module per file, named `<file>_mod.F90`; every procedure inside a module.
- `implicit none (type, external)`; `private` by default with an explicit
  `public` list.
- Modern comparison operators (`==`, `/=`, `>=`, …), spaced `end if` / `end do`,
  double colon on every declaration, lowercase intrinsics.

## Before proposing a contribution

- Code is formatted and passes `ruff check .` / `ruff format .` (Python) and the
  Fortran conventions above.
- Tests are added or updated and pass (`pytest`).
- Documentation is updated if behavior changed.
- The human has read and understood every changed line and has filled in the AI
  usage declaration in the merge request.
