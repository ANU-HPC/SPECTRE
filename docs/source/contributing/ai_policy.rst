######################################
Policy about AI-assisted contributions
######################################

Artificial-intelligence (AI) coding assistants and large language models are
now part of many developers' workflows. While SPECTRE acknowledges the added
value of such tools, we think that they must be used in a very defined environment.
This page describes what is acceptable, what is not, and the responsibilities
that come with using these tools when contributing to SPECTRE.

The guiding principle is simple: **a human contributor is fully and personally
responsible for every line they submit, regardless of how it was produced.** AI
can assist you, but it cannot be a contributor, and it cannot take
responsibility for the code that ends up in SPECTRE.

SPECTRE is a scientific project and as such, one needs to understand how the
results are produced to make sure of their quality.

Declaring AI use
================

**Any use of AI in a contribution must be declared in the merge request.**
Transparency lets reviewers focus their attention appropriately and keeps the
project's provenance clean. Disclosing AI use is never held against you. It is
expected and welcome.

The default template for merge requests adds a checklist for AI-assistance
disclosure. Fill it appropriately when opening a merge request that involved AI
assistance in any form (code, documentation, commit messages, or translation).

.. code-block:: markdown

   ## AI usage declaration

   - [ ] AI tools were used in this contribution
   - **Tool(s) used:** <*e.g.* GitHub Copilot, ChatGPT, Claude>
   - **What it was used for:** <*e.g.* autocompletion, drafting a docstring,
     translating a comment>
   - [ ] I have read and understood every line I am submitting
   - [ ] I confirm the contribution does not infringe any copyright or license

If no AI was used, you do not need to check any box, or you can remove this
section entirely.

If a reviewer suspects undeclared AI involvement, they may ask you about it.
Please do not take such a question negatively. It is part of keeping the
process honest, and an open conversation is all that is needed to resolve it.

Your responsibility as a contributor
=====================================

You — not the AI, and not the maintainers — are solely responsible for the code
and content you push. Before submitting, you must be able to stand behind it as
if you had written it by hand. Concretely, this means:

- **Read and understand every single line.** Do not submit code you cannot
  explain, justify, or maintain. If you do not understand what a generated
  snippet does, it does not belong in your merge request.
- **Make sure it does not infringe any copyright.** AI models can reproduce
  code from their training data verbatim, including code under incompatible
  licenses. It is your responsibility to ensure that what you submit is
  compatible with SPECTRE's license and does not copy protected material.
- **Verify correctness.** Generated code can be subtly wrong, untested against
  SPECTRE's conventions, or based on outdated APIs. Test it and confirm it
  behaves as intended.

What we do not accept
=====================

- **Contributions written entirely by AI.** A merge request whose code is, in
  substance, the raw output of a language model is not an acceptable
  contribution, even if it works.
- **Automatic or agentic contributions.** We do not accept merge requests
  opened or authored by autonomous agents, bots, or pipelines that generate and
  submit changes without a human author who has reviewed and understood them.

In short, AI may help *you* write a contribution, but it may not *be* the
contributor.

What is acceptable
==================

Used as an assistant under your direct supervision, AI tools are welcome. Some
examples of acceptable, limited use:

- **IDE autocompletion**: completing a variable name, a function call, or a
  few lines you were already going to write.
- **Boilerplate**: generating repetitive, mechanical code (*e.g.* an argument
  parser stub or a test skeleton) that you then review and adapt.
- **Drafting docstrings or comments** that you check for accuracy.
- **Explaining unfamiliar code or errors**, or suggesting how to approach a
  problem, while you write the actual solution.
- **Refactoring suggestions** on code you already wrote and understand.

The common thread: the contribution remains *yours*. You drive it, you
understand it, and you would have been able to produce it without the tool. The
AI only made it faster.

Documentation
=============

Using AI to help with documentation: improving phrasing, fixing grammar,
restructuring a paragraph, or drafting an explanation is acceptable, **as long
as you have proofread and validated the content.** Documentation must be
technically correct and must reflect how SPECTRE actually behaves. Do not paste
generated prose into the docs without reading it carefully. A confident but
wrong sentence in the documentation is worse than no sentence at all.

Translation and non-native English speakers
===========================================

We are an international project and we know that not everyone is a native
English speaker. **Language should never be a barrier to contributing.** Using
AI to translate or polish your English in issues, merge request descriptions,
comments, or documentation is explicitly encouraged.

When you do, try to keep your own wording and voice rather than letting the tool
rewrite your message into something generic. Your intent and meaning matter more
than perfect prose, and a clear human message in imperfect English is always
welcome.

If in doubt
===========

If you are unsure whether a particular use of AI fits within this policy, just
ask. Open a discussion or mention it in your merge request. We would much
rather have an open question than an undisclosed assumption. This policy will
evolve as these tools and our experience with them change.
