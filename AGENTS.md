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
- Follow the user's requested change literally and limit edits to that exact scope.
- Do not extend a requested change into adjacent cleanup, dead-code removal, refactoring, renaming, or behavior changes
  without explicit user approval.
- If a potentially better or cleaner approach exceeds the requested scope, explain it as a proposal and wait for
  approval before implementing it.
- Preserve code that the user asked to comment out; do not delete it unless the user explicitly asks for deletion.

## Communication rules
- On first use, expand every acronym and abbreviation and explain every unfamiliar technical or project-specific term
  in plain language.
- Do not assume that a term is familiar merely because it is common in systems or machine-learning research.
- After defining a term once in the current conversation, the abbreviated form may be used.
- Clearly distinguish verified facts, measured results, code-derived conclusions, and estimates.
- Do not present an inference or assumption as a verified fact.
- When the user asks to research or look up information, report verified facts without subjective judgment,
  recommendation, or preference unless the user explicitly requests interpretation or a recommendation. If the
  available evidence is insufficient, state that the point is unknown or unverified.
- Explicitly label every custom dataset subset, corpus construction, preprocessing rule, or evaluation protocol as
  custom when first mentioning it. Never present a custom configuration as an official dataset or benchmark setting.
- Use engineering and logical language for technical explanations. State the concrete component, field, function,
  input, transformation, invariant, and observed output involved in a claim.
- Avoid anthropomorphic or metaphorical descriptions of code and data, such as saying that metadata "claims,"
  "thinks," "wants," or "knows" something. Describe the exact stored value and the operation that consumes it.
- Explain causal chains explicitly: identify the incorrect representation or operation, the affected downstream
  component, and the resulting observable behavior.

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
- When constructing an evaluation dataset or implementing an evaluation
  metric, apply only the exact rules explicitly specified by the user. If the
  user explicitly requests an official protocol, reproduce that protocol
  without local additions. Never introduce a filter, exclusion, repair,
  fallback, matching heuristic, evidence scope, denominator change,
  aggregation rule, or other policy based on the agent's inspection of the
  data or subjective judgment.
- Observing an apparent data problem does not authorize changing the data or
  metric. Preserve and report only the concrete observed fact. Do not invent,
  propose, recommend, compare, or seek approval for any treatment unless the
  user first explicitly asks for treatment options or instructs a treatment.
- Never infer permission for an evaluation-policy change from broad requests
  such as "clean", "correct", "gold", "official", or "robust". If the exact
  treatment is not already defined by the user's explicit instructions, do
  not formulate one. Stop after reporting the unresolved concrete fact.
- Do not proactively design, consider, or suggest an evaluation-data or metric
  policy that the user did not request. User approval of an agent-originated
  policy is not an acceptable substitute because the agent must not originate
  that policy.
- Documentation, a new run name, or later user approval do not legitimize an
  evaluation policy that the user did not originate and explicitly instruct.
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
