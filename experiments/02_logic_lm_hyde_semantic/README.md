# 02 — Logic-LM × HyDE-semantic hypothesis

> Plan: [`docs/plans/logic_lm_hyde_semantic.md`](../../docs/plans/logic_lm_hyde_semantic.md).
> This README is the source of truth once the experiment runs.

## What

Does feeding the `dense_hyde_semantic` **hypothesis** (a generated hypothetical
legal passage) into Logic-LM's **Prolog rule-generation** step — as a 3rd input
used *only* for conceptual framing (which predicates / conditions / Vietnamese
legal terms to model) — improve the QA arm, versus an otherwise-identical arm
that does not see the hypothesis?

Three direct-answer reference arms (no logic-LM) are scored alongside so the
logic-LM arms can be read in context: `qa_hyde_semantic` (the same
`dense_hyde_semantic` retrieval answered directly — isolates the logic-LM
layer), `graphrag` (Neo4j vector RAG), and `llm_only` (no retrieval).

## Why

`dense_hyde_semantic` already generates a hypothesis to compute its retrieval
embedding, then discards it. Logic-LM's rule generator currently sees only
`training_question` + `retrieved_chunks`. The hypothesis is a free byproduct
that encodes the concept frame of the question; the question is whether it helps
the rule generator pick better predicates/structure without leaking facts or
citations (those must still come only from `retrieved_chunks` / the question).

## Setup

- **Family:** qa. All arms `mode: run` (no qa baseline in this repo to inherit).
- **Logic-LM arms (answer via a generated + executed Prolog program):**
  - `logic_lm_hyde_semantic` (**treatment**) — `dense_hyde_semantic` retrieval;
    the hypothesis is injected into rule-gen via the canonical
    `prompts/runtime/logic_lm/rule_gen_hyde_semantic.md` (hypothesis = guidance
    only; thresholds + citations still grounded in `retrieved_chunks`).
  - `logic_lm_hyde_semantic_nohyp` (**control**) — the *same* retriever (same
    dense search, same HyDE cache) but the hypothesis is **not** injected;
    rule-gen uses the default `rule_gen.md`. Isolates exactly one variable.
- **Direct-answer reference arms (no logic-LM; answer straight from the generator):**
  - `qa_hyde_semantic` — the *same* `dense_hyde_semantic` retrieval as the
    logic-LM arms, but the chunks go straight to the GraphRAG generator (no
    Prolog). Isolates the contribution of the logic-LM layer on identical
    retrieval.
  - `graphrag` — Neo4j vector retrieval → generator.
  - `llm_only` — no retrieval; pure LLM.
- **Dataset / N:** `data/eval/questions_200.json`, full (N = 200).
- **Prompt overrides:** none (prompt selection is baked into the pipeline classes).
- **Model:** generator default `gpt-4o-mini` (override per-run if needed).
- **Metrics:** the standard academic set (citation recall/precision/F1, display
  rate, latency, BERTScore, Prolog reliability rates) **plus** answer-vs-gold
  text overlap — **ROUGE-1, ROUGE-2, ROUGE-L, BLEU** (added to `eval_core`;
  scored on the same prose candidate as BERTScore, fail-soft).

## Success criterion & cost (pre-registered — no result prediction)

**Decision rule (objective bar, decided before the run):** after
`metrics/academic_metrics.json` exists, **adopt** treatment over control **only
if** citation **F1 does not drop** AND (citation **recall increases** OR
**prolog_success increases**) AND **unable_to_conclude does not increase**.
Otherwise keep control. (No prediction of the magnitude or direction of any
metric — this is the anti-post-hoc decision rule, per Rule 5.)

**Cost forecast (the only quantity forecast up front):** per treatment question
≈ 1 HyDE call (cached) + 1–3 rule-gen + 1 IRAC render ≈ 2–5 chat completions;
control shares the HyDE cache ≈ 2–4 calls. 2 arms × 200 q ⇒ ≲ ~1,800
`gpt-4o-mini` chat completions (≲ ~700 output tokens/call) ⇒ a few USD;
wall-clock ~tens of minutes. Re-running identical inputs ⇒ ~$0 (HyDE cache).

## How to run

```powershell
# Tier-1 + Tier-2 (inference + metrics + report) — all outputs land in this folder
python -m eval_core all experiments/02_logic_lm_hyde_semantic
# Validate before comparing / copying to experiments_repo/
python -m experiment_contract validate experiments/02_logic_lm_hyde_semantic
```

> Smoke before full: set `dataset.n` to a small integer (e.g. 8) for a pilot
> pass, eyeball records (treatment has a non-empty `hypothesis` field, control
> empty), then restore `n: null`.

## Result summary

_Filled in after the full run._ Will link `metrics/academic_metrics.json` and
`report/academic_report.md` and state the decision-rule outcome (adopt / keep
control) with the margins it rests on.
