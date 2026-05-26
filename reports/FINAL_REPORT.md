# Final Experiment Report — Legal KB QA, Neuro-Symbolic vs GraphRAG vs LLM-only

> **Báo cáo cuối cùng sau 4 vòng audit (2026-05-26).** Tất cả số liệu trích trực tiếp
> từ `data/eval/metrics.json` (R1) và `data/eval/multimodel/metrics.json` (R2) sau
> patches v2 (xem `reports/methodology_fixes.md`). Mọi cell đều có denominator hiển thị.
> Không bịa số. Claims chỉ là defensible nếu pass significance test ở
> α_bonferroni = 0.01 (chi tiết: `reports/significance.md`).

---

## 1. Executive Summary

### Defensible claims (Bonferroni α=0.01)

| # | Claim | Test | p-value / CI |
|---|---|---|---|
| **C1** | `llm_only` thắng `graphrag` trong pairwise judge (R1) | McNemar | p < 0.0001 |
| **C3b** | Elite `no_retrieval` có prolog_success cao hơn elite `graphrag` với gpt-4o-mini (R1) | McNemar | p = 0.0035 |
| **C4** | Với gpt-5-mini, elite `graphrag` thắng `no_retrieval` trong pairwise judge (R2, sau khi loại API errors) | McNemar | p = 0.0006 |

### Claims phải DROP (không đủ bằng chứng)

- C2: graphrag faithfulness > elite_no_retrieval (CI bao gồm 0, n=29 paired)
- C3a: Elite NR vs Ontology prolog_success differs (p=0.89, basically equal)
- C5: GR vs NR prolog_success cho gpt-5-mini differs (p=0.69, sau khi loại API errors gần như identical)

### Reversed từ báo cáo cũ
- **Pairwise judge interpretation**: previously reported "split 80-91%" cho mọi cặp do bug trong `_vote(a_first=False)`. Sau fix, có winners thật.
- **R2 gpt-5-mini pairwise**: cũ "NR beats GR 52.5% vs 30.5%" → fix "**GR beats NR 68.5% vs 31.5%**" trên 89 consistent verdicts sạch (77 records bị API error)
- **R2 gpt-5-mini cost**: cũ "$0.0084 (cheaper than NR)" → thực ra "**$0.0137 (more expensive than NR's $0.0121)**" khi exclude API errors
- **R2 gpt-5-mini prolog_success**: cũ "60% (drops from 99%)" → thực ra "**97% (n=123 sạch, không khác biệt thống kê với NR's 99%)**"

---

## 2. Experimental Design

- **Dataset**: 200 câu hỏi BHXH (Bảo hiểm xã hội VN) từ FB group, mỗi câu có `gold_citations_raw` text.
- **Source-of-truth file**: `data/eval/questions_200.json` (sha256 `b490d22e043ea5bf...`, 271 KB).
- **Knowledge graph**: Neo4j, ~543 Articles/Clauses từ Luật BHXH 41/2024/QH15.
- **Judge model**: gpt-4o-mini (consistent across cả 2 experiments).
- **Prolog**: SWI-Prolog 9.2.1, timeout 15s, max_repair_rounds 2.
- **Pricing**: gpt-4.1 $2/$8, gpt-4o $2.50/$10, gpt-5-mini $0.25/$2, gpt-4o-mini $0.15/$0.60 per 1M tokens (in/out).

### Two experiments

**R1 — 5-arm với gpt-4o-mini làm generator**:
- `graphrag`: Neo4j vector search → LLM generate prose answer
- `llm_only`: pure LLM, no retrieval
- `elite_no_retrieval`: LLM → Prolog (no context, prompt allow self-citation) → SWI-Prolog → IRAC
- `elite_ontology`: LLM → Prolog (ontology retrieval) → SWI-Prolog → IRAC
- `elite_graphrag`: LLM → Prolog (GraphRAG retrieval) → SWI-Prolog → IRAC

**R2 — multi-model elite_no_retrieval vs elite_graphrag**:
- 3 models: gpt-4.1, gpt-4o, gpt-5-mini (gpt-5 dropped per user, gpt-4o-mini không re-run)
- 2 arms: elite_no_retrieval + elite_graphrag
- 6 combos × 200 questions = 1200 inferences

---

## 3. Methodology Fixes Applied (4 audit rounds)

| Audit | Issue | Fix file | Impact |
|---|---|---|---|
| 1 | Pairwise cache key collision (1 entry serves 4 arms) | `compute_metrics.py:508` (fix in current code) | 400 invalid pair cache entries dropped, re-judged |
| 2 | `_vote(a_first=False)` inverts vote_ba — all consensus → "split" | `compute_metrics.py:516-535` (simplified to `w='a'→record_a`) | 1381/1400 records pairwise re-computed correctly |
| 3 | Selection bias (n_valid << n_total hidden), citation source asymmetry | `generate_report.py`, `generate_multimodel_report.py` (n_valid in every cell, micro tables added, IRAC-biased metrics flagged) | All cells now disclose sample sizes |
| 4 | API errors mis-recorded as `unable_to_conclude` (90 records); hallucination conflates 3 things; BERTScore+AR biased by IRAC format; no significance tests | `audit_apply_fixes_v2.py`, `compute_significance.py` | API errors excluded; hallucination split into content+invented; biased metrics dropped for elite; 5 claims tested with Bonferroni correction |

**Reproducibility**: 4 backup files preserved (`*.bak_pre_*`). All re-aggregations from cached judge outputs — no new API calls or judge re-runs.

---

## 4. R1 — 5-arm Comparison (gpt-4o-mini)

### 4.1 Sample integrity per arm

| Arm | n_total | API errors (excluded) | n_clean |
|---|---:|---:|---:|
| `graphrag` | 200 | 0 | 200 |
| `llm_only` | 200 | 0 | 200 |
| `elite_no_retrieval` | 200 | 0 | 200 |
| `elite_ontology` | 200 | **13** | 187 |
| `elite_graphrag` | 200 | 0 | 200 |

### 4.2 Aggregate metrics — macro mean ± std (n_valid/n_clean)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag | Better |
|---|---|---|---|---|---|---|
| citation_validity | 1.0000 ± 0.0000 (155/200) | 0.9531 ± 0.2130 (64/200) | 0.9936 ± 0.0462 (52/200) | 0.9909 ± 0.0812 (184/187) | 0.9983 ± 0.0236 (200/200) | higher |
| citation_recall | 0.9024 ± 0.1908 (200/200) | 0.1834 ± 0.2600 (200/200) | 0.0000 ± 0.0000 (200/200) | 0.1822 ± 0.1682 (187/187) | 0.1454 ± 0.1719 (200/200) | higher |
| citation_precision | 0.7958 ± 0.3007 (40/200) | 0.4300 ± 0.4803 (69/200) | N/A (0/200) | 0.9224 ± 0.2170 (116/187) | 0.8220 ± 0.3041 (98/200) | higher |
| faithfulness | 0.8187 ± 0.2656 (155/200) | 0.6461 ± 0.4003 (64/200) | 0.8143 ± 0.2845 (35/200) | 0.5844 ± 0.3263 (149/187) | 0.7507 ± 0.2724 (147/200) | higher |
| content_hallucination_rate | 0.5036 ± 0.3576 (155/200) | 0.3093 ± 0.2603 (61/200) | 0.6191 ± 0.5239 (34/200) | 0.6595 ± 0.4725 (164/187) | 0.6195 ± 0.4528 (151/200) | lower |
| invented_citation_rate | 0.0000 ± 0.0000 (155/200) | 0.0469 ± 0.2130 (64/200) | 0.0000 ± 0.0000 (52/200) | 0.0054 ± 0.0737 (184/187) | 0.0000 ± 0.0000 (200/200) | lower |
| answer_relevance | 0.5460 ± 0.1101 (200/200) | 0.6805 ± 0.0726 (200/200) | _dropped (IRAC bias)_ | _dropped (IRAC bias)_ | _dropped (IRAC bias)_ | higher |
| bertscore_f1 | 0.6591 ± 0.0368 (198/200) | 0.7128 ± 0.0293 (198/200) | _dropped (IRAC bias)_ | _dropped (IRAC bias)_ | _dropped (IRAC bias)_ | higher |
| cost_usd (mean) | $0.0003 | $0.0001 | $0.0007 | $0.0011 | $0.0012 | lower |
| latency_s (mean) | 2.7s | 4.5s | 11.1s | 12.6s | 14.0s | lower |

> **`answer_relevance` + `bertscore_f1` dropped cho elite arms**: cả 3 elite arms output IRAC structured text (Issue:/Rule:/Application:/Conclusion:) khác format dramatically vs free prose của graphrag/llm_only → BERTScore (lexical overlap) và AR (self-similarity) bias structural, không fair compare.

### 4.3 Micro-averages (corpus-level Σ correct / Σ extracted)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---|---|---|---|---|
| citation_validity | 1.0000 (368/368) | 0.9630 (78/81) | 0.9870 (76/77) | 0.9877 (240/243) | 0.9971 (343/344) |
| citation_recall | 0.8570 (647/755) | 0.1591 (175/1100) | 0.0000 (0/842) | 0.2102 (153/728) | 0.1802 (131/727) |
| citation_precision | 0.6410 (50/78) | 0.4471 (38/85) | N/A (0/0) | 0.8844 (130/147) | 0.7451 (114/153) |
| faithfulness | 0.8610 (483/561) | 0.7266 (210/289) | 0.8061 (79/98) | 0.5744 (274/477) | 0.7510 (374/498) |

### 4.4 Prolog reliability (elite arms only, clean records)

| Metric | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---:|---:|---:|
| `prolog_success_rate` | 0.7800 (n=200) | 0.7807 (n=187) | 0.6550 (n=200) |
| `first_try_success_rate` | 0.5650 | 0.3957 | 0.4850 |
| `repair_invoked_rate` | 0.4350 | 0.6043 | 0.5150 |
| `avg_repair_rounds` | 0.6850 | 0.8396 | 0.8650 |

### 4.5 Prolog status distribution (raw, incl. API errors)

| Status | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---:|---:|---:|
| `success` | 156 | 146 | 131 |
| `syntax_error` | 31 | 21 | 42 |
| `invalid_query` | 11 | 15 | 23 |
| `unable_to_conclude` | 0 | 13 (**all API errors**) | 0 |
| `derived_false` | 1 | 1 | 3 |
| `invalid_program` | 1 | 1 | 1 |
| `citation_required` | 0 | 3 | 0 |

### 4.6 Pairwise judge vs `graphrag` (consistent-verdict subset, clean)

| Arm | n_clean | n_consistent | Arm wins | graphrag wins | Verdict |
|---|---:|---:|---:|---:|---|
| `llm_only` | 200 | 180 | **172 (95.6%)** | 8 (4.4%) | **llm_only beats graphrag (p<0.0001, defensible)** |
| `elite_no_retrieval` | 200 | 135 | 44 (32.6%) | **91 (67.4%)** | graphrag beats elite_no_retrieval (not McNemar-tested) |
| `elite_ontology` | 187 | 132 | 50 (37.9%) | **82 (62.1%)** | graphrag beats elite_ontology (not McNemar-tested) |
| `elite_graphrag` | 200 | 157 | 60 (38.2%) | **97 (61.8%)** | graphrag beats elite_graphrag (not McNemar-tested) |

---

## 5. R2 — Multi-model (elite_no_retrieval vs elite_graphrag)

### 5.1 Sample integrity per combo

| Arm \ Model | gpt-4.1 | gpt-4o | gpt-5-mini |
|---|---:|---:|---:|
| elite_no_retrieval — n_total | 200 | 200 | 200 |
| elite_no_retrieval — API errors | 0 | 0 | 0 |
| elite_no_retrieval — n_clean | **200** | **200** | **200** |
| elite_graphrag — n_total | 200 | 200 | 200 |
| elite_graphrag — API errors | 0 | 0 | **77** |
| elite_graphrag — n_clean | **200** | **200** | **123** |

> **77 API failures cho gpt-5-mini × graphrag** chiếm 38.5% records — combo này có statistical power giảm đáng kể.

### 5.2 Aggregate (clean records, n_valid/n_clean)

| Metric | NR×4.1 | NR×4o | NR×5-mini | GR×4.1 | GR×4o | GR×5-mini |
|---|---|---|---|---|---|---|
| citation_validity | 0.9523±0.18 (200/200) | 0.9824±0.13 (199/200) | _insufficient_ (23/200) | 1.0000±0.00 (196/200) | 0.9975±0.04 (199/200) | 1.0000±0.00 (122/123) |
| citation_recall | 0.0958±0.20 (200/200) | 0.3210±0.25 (200/200) | 0.0000±0.00 (200/200) | 0.4166±0.24 (200/200) | 0.3978±0.19 (200/200) | 0.3378±0.31 (123/123) |
| citation_precision | 0.6707±0.41 (41/200) | 0.9437±0.20 (148/200) | N/A (0/200) | 0.8240±0.24 (182/200) | 0.8646±0.23 (181/200) | 0.5885±0.38 (86/123) |
| faithfulness | 0.9347±0.20 (175/200) | 0.9101±0.22 (158/200) | _insufficient_ (23/200) | 0.8998±0.18 (188/200) | 0.8585±0.20 (188/200) | 0.8932±0.16 (120/123) |
| content_hallucination | 0.1282±0.22 (171/200) | 0.2843±0.31 (165/200) | _insufficient_ (23/200) | 0.1997±0.29 (188/200) | 0.4079±0.44 (193/200) | 0.1209±0.25 (120/123) |
| invented_citation | 0.0250±0.16 (200/200) | 0.0151±0.12 (199/200) | _insufficient_ (23/200) | 0.0000±0.00 (196/200) | 0.0000±0.00 (199/200) | 0.0000±0.00 (122/123) |
| cost_usd | $0.0127 | $0.0142 | $0.0121 | $0.0149 | $0.0165 | **$0.0137** (clean, was $0.0084 dirty) |
| latency_s | 7.9s | 7.1s | 56.0s | 10.0s | 7.2s | 58.9s (clean, was 41.1s dirty) |

> Cells marked `_insufficient_` có n_valid < 30 (threshold for normal-approx CI). `elite_no_retrieval × gpt-5-mini` chỉ 23/200 records có citation_ids — reasoning model abstains từ citing với honest `article: none` (xem `methodology_fixes.md §6`).

### 5.3 Prolog reliability (clean)

| Metric | NR×4.1 | NR×4o | NR×5-mini | GR×4.1 | GR×4o | GR×5-mini (n=123) |
|---|---:|---:|---:|---:|---:|---:|
| `prolog_success_rate` | 0.8600 | 0.7750 | 0.9900 | 0.9100 | 0.9050 | **0.9675** |
| `first_try_success_rate` | 0.7200 | 0.5050 | 0.7800 | 0.8300 | 0.7450 | 0.6423 |
| `avg_repair_rounds` | 0.4700 | 0.7400 | 0.2500 | 0.2800 | 0.3500 | 0.4390 |

### 5.4 Pairwise judge per model (consistent-verdict subset, clean)

| Model | n_clean | n_consistent | GR wins | NR wins | Tie | Verdict |
|---|---:|---:|---:|---:|---:|---|
| **gpt-4.1** | 200 | 158 | 50 (31.6%) | **102 (64.6%)** | 6 (3.8%) | NR beats GR (not McNemar-tested) |
| **gpt-4o** | 200 | 136 | 63 (46.3%) | 64 (47.1%) | 9 (6.6%) | Statistically tied |
| **gpt-5-mini** | 123 | 89 | **61 (68.5%)** | 28 (31.5%) | 0 | **GR beats NR (p=0.0006, defensible)** |

---

## 6. Significance Tests

| # | Claim | Test | Result | Verdict (α_bonf=0.01) |
|---|---|---|---|---|
| C1 | llm_only beats graphrag pairwise (R1) | McNemar | p < 0.0001 | ✓ **DEFENSIBLE** |
| C2 | graphrag faithfulness > elite_no_retrieval (R1) | Bootstrap 95% CI | CI = [-0.097, +0.207] (n=29 paired) | ✗ DROP (CI bao gồm 0) |
| C3a | NR > Ontology prolog_success (R1) | McNemar | p = 0.89 | ✗ DROP |
| C3b | NR > GraphRAG prolog_success (R1, gpt-4o-mini) | McNemar | p = 0.0035 | ✓ **DEFENSIBLE** |
| C4 | GR beats NR pairwise (gpt-5-mini, clean) | McNemar | p = 0.0006 | ✓ **DEFENSIBLE (reversed!)** |
| C5 | GR ≠ NR prolog_success (gpt-5-mini, clean) | McNemar | p = 0.69 | ✗ DROP |

Chi tiết test statistics: `reports/significance.md`.

---

## 7. Cost Summary (real, computed từ records)

| Experiment | Combos × 200q | Inference $ | Judge $ | Total |
|---|---|---:|---:|---:|
| R1 | 5 × 200 = 1000 | $0.68 (gpt-4o-mini) | $0.49 (gpt-4o-mini) | **$1.17** |
| R2 | 6 × 200 = 1200 (− 77 API err) | $15.78 (gpt-4.1/4o/5-mini) | $0.75 (gpt-4o-mini, cached) | **$16.53** |
| **Total spent on this study** | 2200 inferences | **$16.46** | **$1.24** | **$17.70** |

Wall time: R1 inference ~13h (sequential across arms), R2 inference ~6h (parallel with stagger), R2 metrics compute ~3.5h.

---

## 8. Caveats — Reader's Manual

1. **Self-enhancement bias**: Generator gpt-4o-mini + Judge gpt-4o-mini → absolute scores có thể inflated do mô hình "thích" output của chính mình. Relative compare giữa arms vẫn fair.
2. **Single dataset**: Tất cả conclusions tied to 200 BHXH questions. Out-of-domain (commercial law, criminal law) chưa test.
3. **Citation source asymmetry**: `citation_validity`/`faithfulness`/`content_hallucination` dùng `record["citation_ids"]` (bao gồm fallback parser của Prolog `legal_source(...)`). `citation_recall`/`citation_precision` dùng `parse_citations(answer_text)` (regex trên text). Elite arms output inline `(law_bhxh_2024, article_X, clause_Y)` không match regex → text-based metrics undercount. Xem `report_v2.md` cho `citation_text_coverage` per arm.
4. **Selection bias**: cells với `n_valid << n_total` (e.g., `elite_no_retrieval citation_validity n=52`) là conditional mean trên subset where LLM happened to cite. Không comparable trực tiếp với arm n_valid cao.
5. **gpt-5-mini abstention** (R2 NR): 198/200 prolog_success nhưng chỉ 23/200 records có citation_ids — reasoning model honestly khai `article: none` trong Prolog facts thay vì fabricate article numbers (như gpt-4.1/4o làm). Fallback parser regex không match `none` → citation_ids empty. Đây là behavioral difference, không phải pipeline bug.
6. **API errors propagation**: R1 elite_ontology (13 records) và R2 elite_graphrag × gpt-5-mini (77 records) có silent OpenAI "Connection error." — pipeline catch generic Exception và mark as `unable_to_conclude`. Đã filter khỏi tất cả aggregates trong report này. **Pipeline retry logic chưa implement** (out of scope for post-processing).
7. **BERTScore + answer_relevance dropped cho elite arms**: IRAC structured text vs free prose → format bias. R1 vẫn giữ 2 metrics này cho graphrag + llm_only (cả 2 đều prose, fair).
8. **Pairwise tie asymmetry**: judge emits more "tie" khi graphrag in position A (top of prompt). Magnitude small (~5-20%), không invalidate winner findings.
9. **SWI-Prolog timeout = 15s**: câu phức tạp có thể bị giết silently. Status `unable_to_conclude` cho elite arms có thể bao gồm cả real timeouts; cần separate "timeout" status từ "no_rules_or_query" status nếu muốn distinguish.
10. **gpt-5-mini temperature**: API rejects `temperature=0`; auto-fallback to `1.0` (default reasoning behavior). Không phải bug; đã document trong elite_pipelines.

---

## 9. Open Issues & Future Work

### Pipeline-level (cần code change, out of scope cho post-processing)
1. **Retry transient OpenAI errors**: catch `APIConnectionError`/`RateLimitError`/`APITimeoutError` separately từ logical failures; backoff retry.
2. **Distinguish Prolog timeout từ no-output**: hiện cả 2 → `unable_to_conclude`.
3. **IRAC prompt emit `[Điều X khoản Y]` brackets** thay vì inline `(law_bhxh_2024, article_X)` → unified text-based citation extraction.
4. **Re-run 90 API-error records** (~$1.50, ~90 phút) để có complete data.

### Metric-level
5. **Define `abstention_rate`** metric: % records LLM khai `legal_source(...,article: none,...)` — separate dimension từ accuracy.
6. **Larger paired sample cho faithfulness diff** (C2): hiện n=29 → bootstrap CI bao gồm 0. Cần n>=100 paired để defensible.
7. **Stronger judge** (gpt-4o full thay vì gpt-4o-mini): reduce self-enhancement bias risk.

### Methodology-level
8. **Out-of-domain test**: chạy 100 câu commercial law (Luật doanh nghiệp) để check generalization.
9. **Human evaluation correlation**: sample 30 records, expert đánh giá, đo agreement với judge.

---

## 10. References

| Topic | Reference |
|---|---|
| RAGAs (faithfulness, answer_relevance) | Es et al. *RAGAs: Automated Evaluation of Retrieval Augmented Generation* — EACL 2024 Demo |
| Citation Precision/Recall | Liu, Zhang & Liang. *Evaluating Verifiability in Generative Search Engines* — EMNLP Findings 2023 |
| Hallucination Rate (legal) | Magesh et al. *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools* — JELS 2025 (Stanford RegLab/HAI) |
| LLM-as-Judge (pairwise) | Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* — NeurIPS 2023 D&B |
| BERTScore | Zhang et al. *BERTScore: Evaluating Text Generation with BERT* — ICLR 2020 |
| Prolog rollback (Logic-LM) | Pan et al. *Logic-LM: Empowering LLMs with Symbolic Solvers* — EMNLP Findings 2023 |
| McNemar test | McNemar (1947) — paired binary outcomes |
| Bootstrap CI | Efron (1979) — non-parametric resampling |
| Bonferroni correction | Bonferroni (1936) — multiple comparisons |

---

## 11. Reproducibility

All artifacts deterministic post-aggregation (judge cache hits, no new API calls):

```bash
# After loading metrics + cache:
python -m experiments.audit_apply_fixes        # fix pairwise _vote bug
python -m experiments.audit_apply_fixes_v2     # split halluc + tag api_err
python -m experiments.generate_report          # → reports/experiment_report.md
python -m experiments.generate_multimodel_report  # → reports/multimodel_report.md
python -m experiments.compute_significance     # → reports/significance.md
```

**Backups present** (for rollback):
- `data/eval/metrics.json.bak_pre_pairwise_fix`
- `data/eval/metrics.json.bak_pre_v2`
- `data/eval/multimodel/metrics.json.bak_pre_pairwise_fix`
- `data/eval/multimodel/metrics.json.bak_pre_v2`
- `data/eval/judge_cache.jsonl.bak_pre_pairfix`

**All raw inference records** (1000 R1 + 1200 R2 = 2200 JSON files) in:
- `data/eval/results/{arm}/A*.json`
- `data/eval/multimodel/results/{arm}__{model_safe}/A*.json`

---

*Báo cáo này tổng hợp số liệu thật từ raw records sau khi đã apply tất cả audit fixes. Mỗi cell có denominator để minh bạch sample size. Claims chỉ defensible nếu pass Bonferroni α=0.01.*
