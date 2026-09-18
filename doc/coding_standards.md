# Coding standards

## Comment language: SE100 — Simple English

Every comment and docstring in this codebase's Python source must follow **SE100 — Simple
English**:

> Use short, clear sentences and common vocabulary. Avoid idioms, jargon, and unnecessarily
> complex grammatical structures.

This is a language-style rule, not a content rule. A comment can still explain a subtle
technical reason, a past bug, or a design trade-off — it must do that in plain words.

### The rule, in practice

- **Short sentences.** One idea per sentence. Split a long sentence into two or three short
  ones instead of joining them with commas and "which"/"whereas"/"given that".
- **Common words.** Use the plain word, not the fancy one. Say "use" instead of "utilize",
  "show" instead of "surface", "so" instead of "hence" or "thus", "if" instead of "in the
  event that".
- **No idioms.** Avoid phrases like "footgun", "escape hatch", "under the hood", "at the end
  of the day". Say what the thing actually does instead.
- **No unexplained jargon.** A technical term this codebase needs (e.g. "idempotent",
  "checkpointer", "cosine distance") is fine — but a rare or invented shorthand is not. If a
  term needs a technical word, keep the word and explain it in the next sentence if it is not
  obvious from context.
- **Simple grammar.** Avoid nested parentheticals, stacked em-dashes, and long passive-voice
  chains. Prefer active voice: "This function checks X" instead of "X is checked by this
  function".
- **Keep the "why".** SE100 changes HOW a comment is written, never WHAT it says. Never delete
  a fact, a caveat, or a reason to make a comment shorter — split it into more short sentences
  instead.

### Before / after

Before (dense, long sentences, jargon):

```python
# `tenant` filters here because two tenants can legitimately upload a file with the exact
# same name and reach the exact same version number independently — without this filter,
# tenant B's first-ever extraction could be skipped because tenant A already did "the same"
# (source_component, version).
```

After (SE100):

```python
# We filter by tenant here. Two different tenants can upload a file with the same name.
# They can reach the same version number by chance. Without this filter, we could skip
# tenant B's first extraction. That would happen because tenant A already did the "same"
# (source_component, version) pair.
```

Before (idiom, one long sentence):

```python
# reasoning_effort="none" is required alongside it for a reasoning-locked model (e.g.
# gpt-5.6-sol/terra): those models otherwise reject any temperature other than 1 outright.
```

After (SE100):

```python
# Some models lock their reasoning mode (for example gpt-5.6-sol/terra). These models
# reject any temperature value other than 1. Passing reasoning_effort="none" turns this
# check off, so temperature=0 is accepted.
```

### Scope

SE100 applies to:

- `#` comments and `"""docstrings"""` in every `.py` file in this repository.

SE100 does **not** apply to:

- `prompts/**/*.jinja` and `prompts/**/*.md` — these are LLM prompt text, not code comments.
  Their wording is part of the application's behavior, not documentation about it. Changing
  their language changes what the LLM does, so they follow the review process for prompt
  changes instead (see `TESTING_REPORT.md`), not this style rule.
- Commit messages, PR descriptions, and other Markdown documentation (`README.md`, `doc/*.md`)
  — SE100 is a comment-language rule for source code specifically.

### Why

Some of this codebase's comments carry real, load-bearing detail: a past bug, a subtle
constraint, a reason a simpler approach does not work. Dense, long-sentence prose makes that
detail slower to read and easier to skim past. Short, plain sentences are faster to read
correctly the first time, for a human and for an LLM assistant reading the code later.
