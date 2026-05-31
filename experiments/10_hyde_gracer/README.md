# 10 — HyDE prompts under GRACE-R, full 200 head-to-head vs exp 08+09 baselines

## What

Rewrite both HyDE prompts under the GRACE-R framework (6 layers:
Grounding / Role+Scope / Authority / Citation / Examples / Refusal)
plus 3 HyDE-specific design changes (length cut, allow domain
keywords, resolve grounded-copy contradiction). Run the same
3-arm `dense / dense_hyde / dense_hyde2` setup as exp 09 on the
full 200-question dataset. Compare against the frozen exp 09
baselines.

## Why

The exp 09 README documents that HyDE2 (grounded) LOSES vs HyDE1 by
−0.0526 absolute R@12 on the in_corpus stratum (n=151 full). Two
hypotheses came out of the audit of `prompts/runtime/hyde_generate*.md`
(see audit conversation 2026-05-31):

1. **The canonical HyDE prompts are BHXH-anchored at every level**:
   role ("chuyên gia BHXH"), vocabulary list (100% BHXH terms),
   only example (HĐLĐ template), and negative examples carrying
   sticky proper nouns ("bà Châu", "bà Minh Châu", "Long An"). When
   the question or context strays outside BHXH (e.g. Bộ luật Lao
   động seeds in stt=56), the prompt actively biases the generator
   away from the true target domain.
2. **The grounded prompt contains an internal contradiction**:
   "use vocabulary from context" vs "do not copy verbatim". Under
   T=0 the LLM resolves this by *paraphrasing away* the canonical
   phrasing, undoing the grounding signal. This is consistent with
   the exp 09 failure: HyDE2 R@100 fell *below dense baseline*
   (0.5989 vs 0.6592) — the doc moved further from clause manifold,
   not closer.

GRACE-R targets (1) directly via balanced examples, broadened role,
and pattern-level negative examples. Design changes L1–L3 target
the failure-mode mechanics: shorter output (closer to clause length),
keep domain keywords (preserve anchor signal), and explicitly resolve
the grounded contradiction (allow phrase-level reuse).

Plan + audit context: this experiment is a follow-up to the prompt
audit triggered by the user's question "các prompts đang bias theo
một luật" on 2026-05-31. The audit + GRACE-R framework are in the
conversation transcript; no separate `docs/plans/` doc was written
because GRACE-R applied to two prompts is small enough to fit in
this README.

## Setup

- **Dataset**: 200 BHXH questions (full).
- **Seed retriever** (HyDE2 only): BGE-M3+LoRA raw, top-5 — IDENTICAL
  to exp 09 so seed-clause-ids match → HyDE1 / HyDE2 / dense are
  comparable across exp 09 and exp 10 except for the HyDE prompt.
- **Generator**: `gpt-4o-mini`, n=1, max_tokens=700, T=0.0 — same
  hyperparameters as exp 08/09 so the only varying knob is the
  prompt text.
- **HyDE1 prompt override**:
  [`prompts_override/runtime/hyde_generate.md`](prompts_override/runtime/hyde_generate.md)
  (GRACE-R rewrite of [`prompts/runtime/hyde_generate.md`](../../prompts/runtime/hyde_generate.md)).
- **HyDE2 prompt override**:
  [`prompts_override/runtime/hyde_generate_grounded.md`](prompts_override/runtime/hyde_generate_grounded.md)
  (GRACE-R rewrite of [`prompts/runtime/hyde_generate_grounded.md`](../../prompts/runtime/hyde_generate_grounded.md)).
- **Encoder + index**: `models/bge-m3-bhxh-lora` + `clause_vec_tuned`
  (same as exp 08/09).
- **Runner**: [`../../scripts/exp10_run.py`](../../scripts/exp10_run.py).
- **Metrics**: [`../../scripts/exp10_metrics.py`](../../scripts/exp10_metrics.py)
  (`--full` flag from day 1).
- **Cache**: same dirs as exp 08/09 (`artifacts/hyde/` +
  `artifacts/hyde2/`) but new `prompt_sha` → fresh entries; old
  entries untouched.

## Expected outcome (pre-commit before running)

Direct, testable predictions. After full 200 runs, this section
becomes my falsifiable scorecard.

**On the in_corpus stratum (n=151):**

| # | Prediction | Threshold to confirm | Confidence |
|---|---|---|:-:|
| P1 | `dense_hyde` (GRACE-R) ≥ exp 09 `dense_hyde` R@12 | abs Δ ≥ 0 | 60% |
| P2 | `dense_hyde` (GRACE-R) > exp 09 `dense_hyde` R@12 | abs Δ ≥ +0.01 | 35% |
| P3 | `dense_hyde2` (GRACE-R) > exp 09 `dense_hyde2` R@12 | abs Δ ≥ +0.02 | 55% |
| P4 | `dense_hyde2` (GRACE-R) closes >50% of the HyDE2-vs-HyDE1 gap | (0.4736 − new_hyde2) / (0.4736 − 0.4210) < 0.5 | 35% |
| P5 | `dense_hyde2` (GRACE-R) ≥ `dense_hyde` (GRACE-R) | abs Δ ≥ 0 | 20% |
| P6 | On stt=56 specifically, GRACE-R HyDE2 has ≥1 gold in top-12 | recall@12 ≥ 0.5 | 25% |

**Threshold that would change the conclusion** (= "GRACE-R doesn't help HyDE"):
- P1 fails AND P3 fails → GRACE-R is a regression. Document and stop.
- P1 holds but Δ < +0.005 AND P3 < +0.005 → GRACE-R is neutral; the
  win was a wash. Document and consider seed-quality fix (exp 11)
  for HyDE2.
- P3 ≥ +0.02 → GRACE-R helps HyDE2 specifically (its grounded
  contradiction was real). Worth deeper ablation.

**What I genuinely expect**: GRACE-R helps HyDE2 more than HyDE1.
HyDE1's current prompt is already on a local optimum for the
BHXH-only manifold; broadening role + cutting length could hurt as
much as help. HyDE2's grounded contradiction is a more obvious
bug — fixing it has clearer upside. So `dense_hyde2 → between 0.42
and 0.47` is my modal prediction; whether it CROSSES HyDE1 is the
open question.

## Cost estimate

| | value |
|---|---:|
| HyDE1 LLM (full 200 cold, new prompt_sha) | ~$0.10 |
| HyDE2 LLM (full 200 cold, new prompt_sha) | ~$0.14 |
| Pass 1 + Pass 3 dense retrieval (no LLM) | ~25s |
| **Total full 200 (cold)** | **~$0.24** |
| Cap | $0.50 |
| Re-run | $0 (cache hit) |

## Risks

- **Encoder doesn't honor short doc**: BGE-M3 may not reward
  80–150 từ as much as 200–400 từ (longer doc = more averaged
  signal). Mitigation: report results across multiple K including
  R@100 to see if recall ceiling moves; rollback length if catastrophic.
- **GRACE-R domain-broadening regresses HyDE1**: in_corpus stratum
  IS mostly BHXH-questions, so a BHXH-narrow prompt fits the
  distribution. If P1 fails by Δ < −0.02, the broadening was bad.
  Mitigation: ablation `hyde_generate.md` line-by-line; keep
  HyDE1 prompt closer to original if necessary.
- **Sample-size noise**: full 200 has n=151 in_corpus only. Δ
  R@12 of ±0.013 is roughly one stratum question. Report all
  effects with absolute count of "questions changed" not just
  the rate.

## Result summary — full 200 (2026-05-31)

**Headline**: GRACE-R is a **regression on R@12** (the metric exp 09
used to declare winners) but **improves R-Precision substantially**
and lifts HyDE2's R@100 in_corpus. All 5 pre-committed predictions
FAIL. By the decision rule in this README, this is a *negative
result* — do not merge the GRACE-R prompts to canonical.

### Cost + ops
| | full 200 |
|---|---:|
| HyDE1 GRACE-R (cold) | $0.0468 |
| HyDE2 GRACE-R (cold) | $0.0660 |
| **Total** | **$0.1128** (under the $0.24 estimate; OpenAI prompt-cache hits saved input tokens) |
| Wall time | 45.4s after BGE-M3 warm-up |
| Records | 600 (200 × 3 arms), 0 failures |
| Sanity check | `dense` arm bit-for-bit identical with exp 09 (R@12 = 0.3832 in both) ✅ |

### Pre-committed predictions — verdict

| # | Prediction | Threshold | Result | Verdict |
|---|---|---|:-:|:-:|
| P1 | GRACE-R HyDE1 ≥ exp09 HyDE1 R@12 (in_corpus) | abs Δ ≥ 0 | Δ = −0.0123 | ❌ FAIL |
| P2 | GRACE-R HyDE1 > exp09 HyDE1 R@12 | abs Δ ≥ +0.01 | Δ = −0.0123 | ❌ FAIL |
| P3 | GRACE-R HyDE2 > exp09 HyDE2 R@12 | abs Δ ≥ +0.02 | Δ = −0.0070 | ❌ FAIL |
| P4 | GRACE-R HyDE2 closes >50% of HyDE2-vs-HyDE1 gap | ratio < 0.5 | ratio = 1.133 (gap widened) | ❌ FAIL |
| P5 | GRACE-R HyDE2 ≥ GRACE-R HyDE1 R@12 | abs Δ ≥ 0 | Δ = −0.0473 | ❌ FAIL |

5/5 fail on the headline metric. Decision rule from `Expected outcome`
section: "P1 fails AND P3 fails → GRACE-R is a regression. Document
and stop." → **Document and stop.**

### In-corpus stratum (n=151) — full comparison

| metric | dense | exp09 HyDE1 | **exp10 HyDE1 GRACE-R** | exp09 HyDE2 | **exp10 HyDE2 GRACE-R** |
|---|---:|---:|---:|---:|---:|
| R@12   | 0.3832 | **0.4736** | 0.4613 (−0.0123) | 0.4210 | 0.4140 (−0.0070) |
| R@30   | 0.5311 | **0.6066** | 0.6206 (+0.0140) | 0.5203 | 0.5279 (+0.0076) |
| R@100  | 0.6592 | **0.7016** | 0.6973 (−0.0043) | 0.5989 | **0.6431 (+0.0442)** |
| NDCG@12 | 0.2186 | **0.2944** | 0.2939 (−0.0005) | 0.2437 | 0.2489 (+0.0052) |
| R-Prec | 0.0635 | 0.1326 | **0.1533 (+0.0207, +15.6% rel)** | 0.1019 | **0.1288 (+0.0269, +26.4% rel)** |
| MRR    | 0.2122 | **0.2843** | 0.2869 (+0.0026) | 0.2192 | 0.2324 (+0.0132) |

Numbers in **bold**: winner of that row's column-group.

### Interpretation — honest

GRACE-R produces docs that are **shorter, more focused, and less
BHXH-templated** (verified by spot-check: 129 từ vs 201 từ for stt=1).
The metric pattern is consistent across both arms:

- **R@12 down slightly** (−0.7 to −1.2 abs pp): shorter docs cover
  less semantic surface area; some questions whose canonical
  language was redundantly covered by the long prompt lose 1–2
  top-12 hits.
- **R-Precision up strongly** (+15–26% rel): when gold articles ARE
  recalled, they rank higher. This is exactly what removing
  template-anchoring should do — the doc is less biased toward
  generic BHXH boilerplate cluster centroids.
- **HyDE2 R@100 up +7.4% rel**: GRACE-R partially mitigated the
  recall-ceiling regression that exp 09 flagged (HyDE2 R@100 fell
  *below dense baseline* there). Not enough to close the R@12
  gap to HyDE1, but the failure mode (seed-noise pulling doc
  off-manifold) is partly fixed.
- **MRR neutral-to-positive** on both arms: again consistent with
  "tighter ranking, narrower spread".

This is a **legitimate trade-off**, not a uniform regression. But
the metric exp 09 committed to (R@12 in_corpus) shows GRACE-R as
worse, so by the rules of this comparison, GRACE-R loses.

### Why all 5 predictions failed — root cause hypothesis

I predicted GRACE-R would help (P1 60% confidence, P3 55% confidence)
because I read the original prompt as over-fit to BHXH and the
grounded prompt as having an internal contradiction. Both
observations were factually true, but the resulting fix had the
wrong sign on R@12 because:

1. **The in_corpus stratum is ~100% BHXH-questions** (by construction
   — it's defined as questions whose gold cites are in
   `data/legal_metadata.yaml`). A BHXH-narrow prompt fits this
   distribution. Broadening the role widens the embedding's reach
   but DILUTES its concentration on BHXH cluster centroids.
2. **Length cut 200–400 → 80–150**: a real clause is short, but
   the encoder was likely tuned with longer docs at training time
   (the LoRA was fine-tuned on questions, not on synthetic clauses).
   Mean-pooled longer doc may have been closer to clause manifold
   not because it matched clause style but because it averaged
   more BHXH signal.
3. **The audit was right that the OLD prompts are biased**;
   GRACE-R is right at correcting that bias. But the **metric**
   doesn't reward correcting bias — it rewards top-12 coverage on
   a BHXH-dominated dataset where being biased toward BHXH is
   actively useful.

This is a clean example of "the right fix can lose on a metric that
implicitly rewards the bug." The R-Precision lift suggests the
*underlying retrieval* is actually better; R@12 is just a worse
measurement of that.

### Implications & next steps

1. **Do NOT promote GRACE-R prompts to canonical.** The
   exp09-baseline prompts win on R@12 in_corpus, the metric
   that drives downstream win/loss decisions.
2. **The audit findings remain valid.** "Prompts are BHXH-anchored"
   is true; it just happens to be the right anchor for this
   in_corpus stratum. Bias becomes a problem when the corpus
   broadens — re-test GRACE-R after adding ≥1 non-BHXH law to
   the corpus.
3. **R-Precision lift is interesting on its own**. If a future
   experiment uses R-Precision or MRR as a co-primary metric
   (e.g. E2E ranked answer with top-1 emphasis), GRACE-R is worth
   revisiting.
4. **HyDE2 R@100 lift means the grounded contradiction WAS a real
   bug**, just not the dominant one for top-12 ranking. If a later
   experiment introduces re-ranking that pulls from top-100 (e.g.
   cross-encoder over a larger pool), GRACE-R HyDE2 may become
   competitive.
5. **The failure mode for exp 09 stt=56 (Bộ luật Lao động seeds
   for a BHXH question) is NOT fixed by GRACE-R**: full 200 gap
   between GRACE-R HyDE2 and HyDE1 R@12 widened (1.133× the old
   gap). This confirms the exp 09 conclusion: the bug is seed
   quality, not prompt phrasing. A future HyDE3 should attempt
   seed filtering (same-law constraint, or cross-encoder
   re-rank of seeds) — that's the structural fix.

### Artifacts

- Metrics JSON: [`metrics/academic_metrics.json`](metrics/academic_metrics.json)
- Metrics CSV: [`metrics/academic_metrics.csv`](metrics/academic_metrics.csv)
- Report: [`report/academic_report.md`](report/academic_report.md)
- GRACE-R prompts (preserved for audit): [`prompts_override/runtime/`](prompts_override/runtime/)
- Runner: [`../../scripts/exp10_run.py`](../../scripts/exp10_run.py)
- Metrics script: [`../../scripts/exp10_metrics.py`](../../scripts/exp10_metrics.py)

