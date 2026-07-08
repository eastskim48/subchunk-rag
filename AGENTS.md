# AGENTS.md

## Purpose
Help implement research code for RAG context compression.

## Follow these goals
- Help implement experiment code safely.
- Preserve reproducibility.
- Prefer minimal, local changes over large refactors.
- Do not silently change evaluation logic.
- Explain every change in detail.

## Read these files first
1. `docs/project_context.md`
2. `docs/research_goal.md`
3. `docs/glossary.md`
4. `docs/codebase_guide.md`

Do not read all of `docs/handoff.md` by default. Read only the latest relevant section when the user asks to continue
from current project state, and read older sections only when the user asks about past experiment history or when a
specific old decision/result is needed.

## Scope rules
- Work mainly in `src/` and `test/`.

## Coding rules
- Make the smallest change that solves the task.
- Reuse existing code style and naming.
- Python code must be formatted with Black using the repo `pyproject.toml`.
- Do not refactor broadly unless necessary.
- Do not change experiment assumptions unless explicitly requested.
- Keep method code and baseline code easy to compare.
- Add comments only when they clarify non-obvious logic.

## Evaluation rules
- Treat evaluation code as sensitive.
- Do not silently change:
  - dataset handling
  - retrieval flow
  - prompt format
  - metrics
  - scoring
  - output format
- If evaluation changes, update `docs/eval_protocol.md`.
- If evaluation changes, state:
  - what changed
  - why it changed
  - whether old and new results are directly comparable

## Important constraints
- Reproducibility is critical.
- Keep random seed handling explicit.
- Avoid hardcoding dataset-specific paths.

## Handoff rules
- `docs/handoff.md` contains the latest project state across sessions.
- Always update `docs/handoff.md` at the end of each meaningful work session.
- To conserve context, prefer `tail`, `rg`, or a targeted date/heading range instead of loading the entire handoff file.
- Treat `docs/handoff.md` as the source of truth for:
  - current experiment status
  - decisions already made
  - known issues
  - next recommended steps

## Session update format
When updating `docs/handoff.md`, include:
- date
- objective of the session
- what was changed
- what was verified
- open issues
- next steps


## Do not
- Do not store session-specific notes in `AGENTS.md`.
- Do not overwrite prior handoff notes without preserving useful history.
