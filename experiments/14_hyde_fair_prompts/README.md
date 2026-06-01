# exp 14 — Fair-prompt re-test (grounded + semantic HyDE)

> **Status: executed (pilot-50, 2026-06-01).** Created from the 2026-06-01
> code audit, which found the grounded/semantic HyDE prompts were not on equal
> footing with HyDE1. This experiment removes that confound and re-tests.
> **Result below** — split finding: the confound was real for the semantic arm
> (gap to HyDE1 halved) but not for the grounded arm.

## What this fixes (the confound)

The audit compared the dense retrieval arms at the code level and found the
plumbing is symmetric (same `dense_k=100`, same index, same `_dense_search`
path, same mean-pool+normalize, **same shared BGE-M3 instance**). The only
asymmetry was in the **prompts**:

| | HyDE1 (`hyde_generate.md`) | grounded / semantic (canonical) |
|---|---|---|
| Static ~20-term BHXH vocab list | **present** (lines 19-27) | **removed** |
| Grounding block | none (zero-shot) | seed text / concept frame |
| Self-contradiction | none | grounded: "use context vocab" **+** "don't copy verbatim" |

Because the concept frame is thin (exp 13: mean **1.47** concepts/q), the
challengers often got *less* vocabulary scaffolding than HyDE1 — they lost the
static list and gained little from the frame. That is exactly the
"over-diffusion" exp 13 blamed on the method. **exp 10 (GRACE-R) independently
showed removing HyDE1's vocab templating costs −0.012 R@12** → the list does
measurable work, so the earlier head-to-head was confounded, not single-variable.

## The fix — parity, equalised UP

The two challenger prompts under
[`prompts_override/runtime/`](prompts_override/runtime/) now reproduce HyDE1's
skeleton **byte-for-byte** and add *only* a grounding block:

- **`YÊU CẦU NỘI DUNG` (incl. the full vocab list), `CẤM TUYỆT ĐỐI` (1–5),
  `CÁCH VIẾT`, both `VÍ DỤ` blocks** — identical to `prompts/runtime/hyde_generate.md`.
- **Added, and the only delta:** one role sentence + one `CÁCH SỬ DỤNG
  CONTEXT/KHUNG` section + the user-message context block.
- **Confound B resolved:** the grounded prompt now allows phrase-level reuse of
  context terms and forbids only *long verbatim* copying (no more contradiction).

HyDE1 is **unchanged** — it is the frozen exp 08 baseline and already *is* the
shared skeleton. `scripts/exp14_run.py` sets `LEGAL_KG_PROMPTS_DIR` to this
folder's `prompts_override/`, so `load_prompt` serves the parity prompts for
the two challengers and **falls back to the canonical `hyde_generate.md`** for
HyDE1 (per-file fallback, `src/prompts.py`). The runner asserts each generator
resolved to the expected file. Canonical prompts + frozen exp 09/13 records are
not touched (project rule: don't mutate canonical prompts committed baselines
depend on).

Why "equalise up" (keep the list, add it to challengers) rather than "equalise
down" (strip it from all three): stripping would require re-running the frozen
exp 08 baseline and answers a *different* question (is templated HyDE inflating
results?) that exp 10 already measured. Keeping the shared skeleton and varying
only the grounding is the minimal intervention that isolates the variable.

## Pre-commitment predictions (stated BEFORE running)

On the in_corpus stratum, pilot-50 (n≈38), vs the **frozen** exp 13 / exp 09
numbers on the same pilot. Falsifiable scorecard:

| # | Prediction | Threshold | Confidence |
|---|---|---|:-:|
| P1 | semantic-fair R@12 ≥ exp 13 semantic R@12 (0.4248) | Δ ≥ 0 | 70% |
| P2 | semantic-fair narrows the gap to HyDE1 by ≥ 1/3 | (HyDE1−fair)/(HyDE1−exp13) ≤ 0.67 | 45% |
| P3 | grounded-fair R@12 ≥ exp 09 HyDE2 R@12 (same pilot) | Δ ≥ 0 | 60% |
| P4 | Neither challenger crosses HyDE1 (S1 still fails) | both Δ vs HyDE1 < 0 | 60% |
| P5 | R-Precision of both challengers ≥ their exp 09/13 value | Δ ≥ 0 | 55% |

**What would change the conclusion:**
- If a challenger now **passes S1** (R@12 ≥ HyDE1 − 0.01) → a meaningful share
  of the earlier "loss" was the dropped-vocab confound, not the grounding idea.
  → promote the parity prompt to canonical (graduate) + run full-200 to confirm.
- If R@12 is **unchanged** (Δ < 0.005 both) → the confound was negligible here;
  the earlier negative stands, and the bottleneck is the thin concept frame /
  domain-noisy seed (a data lever, not a prompt lever).
- Either way: **pair a bootstrap CI** before any firm claim — n≈38 estimates
  swing ~0.05 from sampling alone (HyDE1's own R@12 moved 0.4736→0.5207 between
  full-200 and pilot-50).

This is a fairness fix, **not** an attempt to make the challengers win. A null
result here is a perfectly valid outcome and should be reported as such.

## Result — pilot-50 (2026-06-01)

4 arms on exp 08's pilot-50 (in_corpus **n=38**), 0 failures, cost **~$0.031**.
Sanity: HyDE1 + raw dense are **byte-identical to exp 13** (R@12 0.5207 / 0.4154)
→ HyDE1 stayed canonical; only the challenger PROMPT changed. Frame provenance
identical (concept_match 0.84, 1.47 concepts/q).

### Before → after parity (in_corpus, n=38, R@12)

| arm | before (confounded) | after (parity) | Δ | gap vs HyDE1 (0.5207) |
|---|---:|---:|---:|---:|
| semantic | 0.4248 (exp 13) | **0.4731** | **+0.048** | −0.096 → **−0.048** (halved) |
| grounded (HyDE2) | 0.4812 (exp 09) | 0.4330 | −0.048 | −0.040 → −0.088 (widened) |

R-Precision: semantic 0.1529 → 0.1046 (−0.048); grounded 0.1178 → 0.1128 (−0.005).

### Split finding (honest)

- **Semantic: Confound A was REAL.** Restoring HyDE1's vocab scaffold lifted R@12
  +0.048 and **closed ~half the gap** to HyDE1 — so ~half of semantic's exp-13
  "loss" was the dropped vocab list, not the grounding idea. **S2 flipped
  FAIL→PASS** (now clearly beats raw dense). BUT R-Precision dropped by the same
  magnitude — the scaffold trades rank-sharpness for top-12 coverage (mirror of
  exp 10 GRACE-R). The remaining −0.048 is within ~1 sampling SE at n=38 →
  semantic-fair and HyDE1 are **not clearly distinguishable** on R@12 (needs a
  bootstrap CI / full-200 to confirm).
- **Grounded: Confound A was NOT the issue.** Parity made HyDE2 *worse* on R@12
  (−0.048); the original prompt was fine there. R@all stays depressed (0.617 <
  dense 0.722) — the domain-noisy clause seed (exp 09's diagnosed root cause)
  dominates, and the static list competes with the clause-text grounding.

### Pre-registered predictions — scorecard

| # | prediction | verdict |
|---|---|:-:|
| P1 | semantic-fair R@12 ≥ exp 13 (0.4248) | ✅ 0.4731 |
| P2 | semantic narrows gap to HyDE1 by ≥ 1/3 | ✅ ~50% |
| P3 | grounded-fair R@12 ≥ exp 09 (0.4812) | ❌ 0.4330 (worse) |
| P4 | neither crosses HyDE1 (S1 fails both) | ✅ |
| P5 | R-Precision of both ≥ before | ❌ semantic R-Prec fell |

3/5 hold; the two misses (P3, P5) are genuine surprises, not engineered away.

### Conclusion

The exp-13 verdict *"semantic loses by −0.096, unambiguous"* was **partly a
prompt-confound artifact**: under a fair prompt the gap halves to −0.048, within
sampling noise at n=38. The audit's core claim — the head-to-head was not
single-variable and the "unambiguous" language outran the evidence — is
**vindicated for the semantic arm**. For grounded, the audit's *other* claim
holds: its deficit is the seed, not the prompt. HyDE1 remains the top R@12
retriever, but for semantic that is now a **fair** conclusion, not a confounded
one. Next levers are data-side (richer multi-law concept frame; seed filtering
for HyDE2) + a bootstrap CI + full-200 before any firm claim. Report:
[`report/retrieval_report.md`](report/retrieval_report.md).

## How to run

```powershell
# 4 arms on exp 08's stratified pilot-50 (only the two challenger arms hit
# OpenAI on cache-miss; HyDE1 reuses artifacts/hyde/ → $0). ~$0.05 cold.
python -m scripts.exp14_run --pilot-50
python -m scripts.exp14_metrics            # in_corpus headline + S1/S2 per challenger

# Full-200 only if a pilot challenger passes S1:
python -m scripts.exp14_run
python -m scripts.exp14_metrics --full
```

The runner prints which prompt file each generator resolved to and **asserts**
the two challengers use the override while HyDE1 stays canonical — so a
mis-set `LEGAL_KG_PROMPTS_DIR` fails fast instead of silently testing the wrong
prompt.

## What this does NOT change

- No edit to canonical `prompts/`, `runtime/`, `eval_core`, or the frozen
  exp 08/09/13 records.
- The **concept-frame builder** (`runtime/retrievers/semantic_context.py`) and
  the **HyDE2 seed pass** (`hyde2_seed_k=5`) are unchanged — only the prompt
  text differs. So a remaining gap isolates the grounding *signal*, not the
  prompt scaffold.
- The thin-frame / L41-only-ontology coverage limit (exp 13 plan §10) is a
  separate data lever, out of scope here.

## Files

- Parity prompts: [`prompts_override/runtime/hyde_generate_grounded.md`](prompts_override/runtime/hyde_generate_grounded.md),
  [`prompts_override/runtime/hyde_generate_semantic.md`](prompts_override/runtime/hyde_generate_semantic.md)
- Runner: [`../../scripts/exp14_run.py`](../../scripts/exp14_run.py)
- Metrics: [`../../scripts/exp14_metrics.py`](../../scripts/exp14_metrics.py)
- Audit precedent: [exp 13](../13_hyde_semantic/README.md) (semantic, confounded),
  [exp 10](../10_hyde_gracer/README.md) (quantified the vocab-templating effect),
  [exp 09](../09_hyde2_grounded/README.md) (grounded, confounded + contradiction).
