# exp 13 — Semantic-grounded HyDE (concept-anchored, no dense seed)

Plan: [`docs/plans/exp13_hyde_semantic.md`](../../docs/plans/exp13_hyde_semantic.md).

## What

HyDE whose hypothetical doc is grounded on a **BHXH concept frame** —
`runtime/retrievers/semantic_context.build_semantic_context` maps the
question to concepts + KG entities (Subject/Benefit/Obligation…) drawn
from `data/ontology/ontology_kg_full.json` (multi-law) **without a dense
clause seed**. The frame → `OpenAISemanticHydeGenerator` → BGE-M3(+LoRA)
embed → dense top-K over `clause_vec_tuned`.

| arm | grounding | source |
|---|---|---|
| `dense` | none (raw question) | tuned dense |
| `dense_hyde` | none (HyDE1 zero-shot) | the bar |
| `dense_hyde_semantic` | **concept frame, no dense seed** | NEW |

## Why

[exp 09](../09_hyde2_grounded/README.md) grounded HyDE on **clause text
from a top-5 dense seed** and *lost* to HyDE1: the raw-question seed
landed in the wrong domain cluster (stt=56) and grounding amplified it.
exp 13 removes the dense seed and anchors on canonical BHXH concepts
(robust to informal phrasing) → it cannot inherit the seed's domain
bias. This is "exp 09 done right".

## Bar & metrics (plan §0/§7/§8)

- Headline stratum: **in_corpus** (gold ⊆ the 4 indexed laws).
- Metrics: recall@{5,10,12}, **precision@1**, **R-Precision**, NDCG@12, MRR.
  `precision@2` is *not* a target (cardinality ceiling 0.7575).
- **The bar is HyDE1**, tuned in_corpus: **R@12 0.4736 / R-Prec 0.1326**.

The originally-requested `recall@5 > 0.60` / `precision@2 > 0.80` were
**retired** (2026-06-01 feasibility analysis): precision@2 is
mathematically capped at 0.7575 (48.5% of questions have a single gold
article) and 38% of all gold mentions are in laws not in the index.
They survive as a labelled **north-star** (plan §9), reachable only via
the full rerank pipeline and/or corpus expansion.

## Pre-commitment predictions (plan §8, stated BEFORE the run)

- **S1 (the gate exp 09 failed):** semantic R@12 ≥ HyDE1 R@12 − 0.01.
- **S2:** semantic R@12 ≥ dense R@12 + 0.03.
- **Headline Δ:** semantic R@12 − HyDE1 R@12 ∈ [−0.01, +0.04]; **> +0.05 → audit**.
- `concept_match_rate` ≥ 0.85; `fallback` ≤ 0.15.
- Lift (if any) concentrates on l41/mixed; `no_l41`-ish in-corpus ≈ 0.

**Win = S1 ∧ S2 ∧ (headline Δ ≥ +0.02).**

## How to run

```powershell
python -m scripts.exp13_run --pilot-50      # 3 arms; only HyDE arms hit OpenAI (cached → ~$0)
python -m scripts.exp13_metrics             # in_corpus headline + S1/S2 + provenance
```

## Result — pilot-50 (2026-06-01)

**Negative on the headline: `dense_hyde_semantic` does NOT beat HyDE1.**
concept_match 86% (43/50); semantic HyDE cost $0.014; 0 failures.

In-corpus (n=38):

| metric | dense | **dense_hyde (HyDE1)** | dense_hyde_semantic |
|---|---:|---:|---:|
| recall@5 | 0.2456 | **0.3716** | 0.3289 |
| recall@12 | 0.4154 | **0.5207** | 0.4248 |
| recall@20 | 0.4944 | 0.5294 | **0.5873** |
| recall@all | 0.7224 | 0.7005 | 0.7014 |
| R-Precision | 0.0746 | 0.1479 | **0.1529** |
| precision@1 | 0.1053 | **0.1316** | 0.1053 |
| NDCG@12 | 0.2318 | **0.3134** | 0.2673 |

Pre-registered verdict (plan §8): **S1 FAIL** (sem R@12 − HyDE1 = **−0.096**,
the gate exp 09 also failed), S2 FAIL (+0.009 vs raw, < +0.03). Headline Δ
**−0.096** — *worse* than the pre-registered band [−0.01, +0.04]; my
prediction was too optimistic.

**But a cleaner negative than exp 09, and an actionable signal:**

1. **No recall-ceiling collapse.** exp 09 (clause-seed grounding) drove
   R@100 *below* baseline; exp 13's recall@all (0.701) ≈ HyDE1 (0.701) ≈
   dense (0.722). The concept anchor did **not** inject the domain noise
   that sank exp 09 — that §8 prediction held.
2. **Gold is in the pool, just mis-ranked at the top.** Semantic *wins*
   recall@20 (**0.587** vs HyDE1 0.529, +0.058) and ties/edges R-Precision
   (0.153 vs 0.148). It loses only at the sharp top (R@5/@12) — the
   concept frame produces a more *diffuse* hypothesis than HyDE1's
   free-form zero-shot doc.

**Conclusion.** Zero-shot **HyDE1 remains the best retriever**; concept-frame
grounding doesn't beat it at top-rank. The mechanism is over-diffusion, not
exp 09's domain-noise collapse. Because the gold sits in top-20 (R@20 win),
the natural next lever is a **cross-encoder reranker** (full V5 pipeline) on
the semantic candidate set — the north-star path (plan §9), now evidence-backed.
A full-200 would confirm, but the pilot signal (S1 −0.096) is unambiguous.
