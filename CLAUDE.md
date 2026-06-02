# CLAUDE.md — repo-wide instructions for Claude / any agent

Vietnamese Legal Knowledge-Graph QA system (offline build → Neo4j → runtime arms
→ `eval_core` experiments). Full orientation: [`docs/architecture.md`](docs/architecture.md).
The experiment workflow + the inviolable rules live in the **required-reading
skill** [`.claude/skills/legal-kg-logic-extraction/SKILL.md`](.claude/skills/legal-kg-logic-extraction/SKILL.md)
— read it before touching anything under `offline/`, `runtime/`, `eval_core/`,
`experiments/`, `prompts/`, or `schema/`.

## Non-negotiable — experiments: the agent MEASURES, the researcher CONCLUDES

For **every** experiment (anything under `experiments/`, any `eval_core` run) the
agent's job is: **run it → produce the deterministic `eval_core` metrics → report
a factual run status → link `metrics/` + `report/`. Nothing more.**

**Do NOT** write — in any file, commit message, chat reply, memory, or spawned
task — a **nhận xét / interpretation / verdict / conclusion** about an
experiment's *results*. No "X did/didn't regress", "arm Y won/lost", "metric is
healthy / non-degenerate / pre-existing / a benchmark artifact", no root-cause
claims, no "this means …". A **Result summary** holds **only** run status (n
records, pass/fail, `validate` OK) + the **verbatim measured numbers** + links.

Reading the numbers, comparing, diagnosing causes, and drawing any conclusion are
the **researcher's (user's) sole authority** — this protects the integrity and
authorship of the thesis. A pre-registered success criterion, if any, is the
researcher's to set and to apply; the agent never authors a decision rule or
declares an outcome. If asked "what does this mean?", surface the relevant numbers
and let the user judge — do not editorialize.

This binds **every session and every agent** on this repo. The other inviolable
experiment rules (no result prediction; honest, unbiased measurement; result
isolation per folder) are Rules 1–7 in the skill above.
