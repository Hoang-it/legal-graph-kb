# Experiment Report — 5-arm comparison (GraphRAG vs LLM-only vs Elite × 3)

**Dataset**: 200 câu BHXH (FB group). Arms compared: `graphrag, llm_only, elite_no_retrieval, elite_ontology, elite_graphrag`. Số sample / arm: graphrag=200, llm_only=200, elite_no_retrieval=200, elite_ontology=200, elite_graphrag=200

**Models**: generator + judge đều `gpt-4o-mini` (self-bias risk — chỉ affect *absolute* scores, *relative* fair).

**Arms**:

- `graphrag`: vector search Neo4j + LLM generate answer text

- `llm_only`: chỉ LLM, no retrieval

- `elite_no_retrieval`: LLM → Prolog (no context, prompt relaxed) → SWI-Prolog → IRAC

- `elite_ontology`: LLM → Prolog (ontology retrieval) → SWI-Prolog → IRAC

- `elite_graphrag`: LLM → Prolog (GraphRAG retrieval) → SWI-Prolog → IRAC


## Metrics & paper refs (peer-reviewed, không arXiv)

| Metric | Paper | Venue |
|---|---|---|
| Faithfulness, Answer Relevance | Es et al. *RAGAs: Automated Evaluation of Retrieval Augmented Generation* | [EACL 2024 Demo](https://aclanthology.org/2024.eacl-demo.16/) |
| Citation Precision/Recall | Liu, Zhang & Liang. *Evaluating Verifiability in Generative Search Engines* | [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.467/) |
| Hallucination Rate (legal) | Magesh et al. *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools* | [JELS 2025, Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12413) (Stanford RegLab/HAI) |
| LLM-as-Judge (pairwise) | Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [NeurIPS 2023 D&B](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) |
| BERTScore | Zhang et al. *BERTScore: Evaluating Text Generation with BERT* | [ICLR 2020 (OpenReview)](https://openreview.net/forum?id=SkeHuCVFDr) |
| **Prolog rollback rate** (Logic-LM family) | Pan et al. *Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning* | [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.248/) |


## Aggregate results (macro mean ± std, n_valid/total)

> Cells với `n_valid < 30` → 'insufficient' (sample size không đủ tin cậy).
> Khi n_valid khác n_total → metric chỉ đo trên subset of records có valid value (selection bias warning).
> **API errors** (records với prompt+completion tokens = 0): elite_ontology=13. Đã exclude khỏi mọi aggregate dưới.

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag | Direction |
|---|---|---|---|---|---|---|
| **citation_validity** | 1.0000 ± 0.0000 (n=155/200) | 0.9531 ± 0.2130 (n=64/200) | 0.9936 ± 0.0462 (n=52/200) | 0.9909 ± 0.0812 (n=184/187) | 0.9983 ± 0.0236 (n=200/200) | higher better |
| **citation_recall** | 0.9024 ± 0.1908 (n=200/200) | 0.1834 ± 0.2600 (n=200/200) | 0.0000 ± 0.0000 (n=200/200) | 0.1822 ± 0.1682 (n=187/187) | 0.1454 ± 0.1719 (n=200/200) | higher better |
| **citation_precision** | 0.7958 ± 0.3007 (n=40/200) | 0.4300 ± 0.4803 (n=69/200) | insufficient (n=0/200) | 0.9224 ± 0.2170 (n=116/187) | 0.8220 ± 0.3041 (n=98/200) | higher better |
| **faithfulness** | 0.8187 ± 0.2656 (n=155/200) | 0.6461 ± 0.4003 (n=64/200) | 0.8143 ± 0.2845 (n=35/200) | 0.5844 ± 0.3263 (n=149/187) | 0.7507 ± 0.2724 (n=147/200) | higher better |
| **content_hallucination_rate** | 0.5036 ± 0.3576 (n=155/200) | 0.3093 ± 0.2603 (n=61/200) | 0.6191 ± 0.5239 (n=34/200) | 0.6595 ± 0.4725 (n=164/187) | 0.6195 ± 0.4528 (n=151/200) | lower better |
| **invented_citation_rate** | 0.0000 ± 0.0000 (n=155/200) | 0.0469 ± 0.2130 (n=64/200) | 0.0000 ± 0.0000 (n=52/200) | 0.0054 ± 0.0737 (n=184/187) | 0.0000 ± 0.0000 (n=200/200) | lower better |
| **answer_relevance** | 0.5460 ± 0.1101 (n=200/200) | 0.6805 ± 0.0726 (n=200/200) | dropped (IRAC bias) | dropped (IRAC bias) | dropped (IRAC bias) | higher better |
| **bertscore_f1** | 0.6591 ± 0.0368 (n=198/200) | 0.7128 ± 0.0293 (n=198/200) | dropped (IRAC bias) | dropped (IRAC bias) | dropped (IRAC bias) | higher better |
| **cost_usd** | 0.0003 ± 0.0000 (n=200/200) | 0.0001 ± 0.0000 (n=200/200) | 0.0007 ± 0.0003 (n=200/200) | 0.0011 ± 0.0004 (n=187/187) | 0.0012 ± 0.0005 (n=200/200) | lower better |
| **latency_s** | 2.7420 ± 1.6928 (n=200/200) | 4.5487 ± 1.5621 (n=200/200) | 11.0912 ± 7.3898 (n=200/200) | 12.5972 ± 8.8846 (n=187/187) | 14.0109 ± 6.8819 (n=200/200) | lower better |

> **'dropped (IRAC bias)'** = elite arms output structured IRAC text (Issue/Rule/Application/Conclusion headers), không thể fair compare với free prose của graphrag/llm_only bằng BERTScore (lexical overlap) hay answer_relevance (self-similarity với generated questions).

### Citation metrics — macro vs micro

> **Macro** = mean of per-record rates (current table).
> **Micro** = corpus-level Σ correct / Σ extracted (less sensitive to records with few citations).

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---|---|---|---|---|
| **citation_validity** | 1.0000 (Σ=368/368) | 0.9630 (Σ=78/81) | 0.9870 (Σ=76/77) | 0.9877 (Σ=240/243) | 0.9971 (Σ=343/344) |
| **citation_recall** | 0.8570 (Σ=647/755) | 0.1591 (Σ=175/1100) | 0.0000 (Σ=0/842) | 0.2029 (Σ=153/754) | 0.1802 (Σ=131/727) |
| **citation_precision** | 0.6410 (Σ=50/78) | 0.4471 (Σ=38/85) | N/A | 0.8844 (Σ=130/147) | 0.7451 (Σ=114/153) |
| **faithfulness** | 0.8610 (Σ=483/561) | 0.7266 (Σ=210/289) | 0.8061 (Σ=79/98) | 0.5744 (Σ=274/477) | 0.7510 (Σ=374/498) |

## Prolog reliability (Logic-LM metrics — chỉ áp dụng cho elite arms)

> Đo độ tin cậy của symbolic solver loop. Pan et al. EMNLP'23 báo cáo các metric tương tự để compare LLM-as-reasoner vs LLM+symbolic.
> **API-error records excluded** từ mọi tỉ lệ dưới (tránh ô nhiễm với infrastructure failures).

| Metric | elite_no_retrieval | elite_ontology | elite_graphrag | Direction |
|---|---|---|---|---|
| **prolog_success_rate** | 0.7800 (n=200) | 0.7807 (n=187) | 0.6550 (n=200) | higher better |
| **first_try_success_rate** | 0.5650 (n=200) | 0.3957 (n=187) | 0.4850 (n=200) | higher better |
| **repair_invoked_rate** | 0.4350 (n=200) | 0.6043 (n=187) | 0.5150 (n=200) | lower better |
| **avg_repair_rounds** | 0.6850 (n=200) | 0.8396 (n=187) | 0.8650 (n=200) | lower better |

### API error rate (excluded from above)

| Arm | API errors | % of 200 |
|---|---:|---:|
| elite_no_retrieval | 0 | 0.0% |
| elite_ontology | 13 | 6.5% |
| elite_graphrag | 0 | 0.0% |

### Prolog status distribution (raw, including API errors)

> Note: `unable_to_conclude` count cho elite arms có thể bao gồm API errors. Số real Prolog failures = (unable_to_conclude − api_errors).

| Status | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---:|---:|---:|
| `citation_required` | 0 | 3 | 0 |
| `derived_false` | 1 | 1 | 3 |
| `invalid_program` | 1 | 1 | 1 |
| `invalid_query` | 11 | 15 | 23 |
| `success` | 156 | 146 | 131 |
| `syntax_error` | 31 | 21 | 42 |
| `unable_to_conclude` | 0 | 13 | 0 |

## Pairwise judge vs `graphrag` (LLM-as-Judge, position swap)

### `llm_only` vs `graphrag` (n=200)

**Consensus breakdown:**

| Consensus | Count | % of all (n=200) |
|---|---:|---:|
| llm_only | 172 | 86.0% |
| split | 20 | 10.0% |
| graphrag | 8 | 4.0% |

**On consistent-verdict subset** (n_consistent = 180 = 200 − 20 split):

- **llm_only**: 172/180 = 95.6% wins
- **graphrag**: 8/180 = 4.4% wins

**Position-swap detail (raw votes per direction):**

| Vote | A=graphrag B=llm_only | A=llm_only B=graphrag |
|---|---:|---:|
| graphrag | 16 | 18 |
| llm_only | 181 | 182 |
| tie | 3 | 0 |

### `elite_no_retrieval` vs `graphrag` (n=200)

**Consensus breakdown:**

| Consensus | Count | % of all (n=200) |
|---|---:|---:|
| graphrag | 91 | 45.5% |
| split | 65 | 32.5% |
| elite_no_retrieval | 44 | 22.0% |

**On consistent-verdict subset** (n_consistent = 135 = 200 − 65 split):

- **elite_no_retrieval**: 44/135 = 32.6% wins
- **graphrag**: 91/135 = 67.4% wins

**Position-swap detail (raw votes per direction):**

| Vote | A=graphrag B=elite_no_retrieval | A=elite_no_retrieval B=graphrag |
|---|---:|---:|
| elite_no_retrieval | 73 | 47 |
| graphrag | 91 | 153 |
| tie | 36 | 0 |

### `elite_ontology` vs `graphrag` (n=200)

**Consensus breakdown:**

| Consensus | Count | % of all (n=200) |
|---|---:|---:|
| graphrag | 93 | 46.5% |
| split | 57 | 28.5% |
| elite_ontology | 50 | 25.0% |

**On consistent-verdict subset** (n_consistent = 143 = 200 − 57 split):

- **elite_ontology**: 50/143 = 35.0% wins
- **graphrag**: 93/143 = 65.0% wins

**Position-swap detail (raw votes per direction):**

| Vote | A=graphrag B=elite_ontology | A=elite_ontology B=graphrag |
|---|---:|---:|
| elite_ontology | 88 | 50 |
| graphrag | 94 | 149 |
| tie | 18 | 1 |

### `elite_graphrag` vs `graphrag` (n=200)

**Consensus breakdown:**

| Consensus | Count | % of all (n=200) |
|---|---:|---:|
| graphrag | 97 | 48.5% |
| elite_graphrag | 60 | 30.0% |
| split | 43 | 21.5% |

**On consistent-verdict subset** (n_consistent = 157 = 200 − 43 split):

- **elite_graphrag**: 60/157 = 38.2% wins
- **graphrag**: 97/157 = 61.8% wins

**Position-swap detail (raw votes per direction):**

| Vote | A=graphrag B=elite_graphrag | A=elite_graphrag B=graphrag |
|---|---:|---:|
| elite_graphrag | 94 | 60 |
| graphrag | 97 | 137 |
| tie | 9 | 3 |

## Breakdown theo luật version (từ gold_citations_raw)

### `new_2024` (148 câu)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---|---|---|---|---|
| citation_validity | 1.0000 | 1.0000 | 0.9910 | 0.9952 | 0.9977 |
| citation_recall | 0.9049 | 0.1701 | 0.0000 | 0.1749 | 0.1370 |
| citation_precision | 0.8214 | 0.4722 | N/A | 0.9138 | 0.8685 |
| faithfulness | 0.8206 | 0.6639 | 0.8299 | 0.6026 | 0.7498 |
| content_hallucination_rate | 0.5232 | 0.2980 | 0.6508 | 0.6346 | 0.6009 |
| invented_citation_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| answer_relevance | 0.5445 | 0.6813 | 0.6240 | 0.5896 | 0.5540 |
| bertscore_f1 | 0.6613 | 0.7107 | 0.6530 | 0.6446 | 0.6334 |
| cost_usd | 0.0003 | 0.0001 | 0.0007 | 0.0011 | 0.0012 |
| latency_s | 2.5609 | 4.5543 | 11.2614 | 12.3814 | 14.4946 |

### `old_2014` (8 câu)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---|---|---|---|---|
| citation_validity | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| citation_recall | 0.7933 | 0.4545 | 0.0000 | 0.0813 | 0.1232 |
| citation_precision | 0.8750 | 0.3333 | N/A | 1.0000 | 1.0000 |
| faithfulness | 0.8759 | 0.9000 | 0.7083 | 0.6417 | 0.6800 |
| content_hallucination_rate | 0.3299 | 0.3189 | 0.8000 | 0.8200 | 0.8167 |
| invented_citation_rate | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| answer_relevance | 0.6330 | 0.6616 | 0.6442 | 0.4677 | 0.4854 |
| bertscore_f1 | 0.6829 | 0.7249 | 0.6659 | 0.6114 | 0.6357 |
| cost_usd | 0.0004 | 0.0002 | 0.0006 | 0.0010 | 0.0013 |
| latency_s | 4.8125 | 4.6442 | 11.9242 | 12.7922 | 15.7705 |

### `unknown` (44 câu)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---|---|---|---|---|
| citation_validity | 1.0000 | 0.7692 | 1.0000 | 0.9744 | 1.0000 |
| citation_recall | 0.9137 | 0.1792 | 0.0000 | 0.1712 | 0.1780 |
| citation_precision | 0.6667 | 0.3333 | N/A | 0.9444 | 0.6731 |
| faithfulness | 0.7999 | 0.4852 | 0.8214 | 0.5157 | 0.7624 |
| content_hallucination_rate | 0.4722 | 0.3566 | 0.4229 | 0.7275 | 0.6496 |
| invented_citation_rate | 0.0000 | 0.2308 | 0.0000 | 0.0256 | 0.0000 |
| answer_relevance | 0.5350 | 0.6812 | 0.6209 | 0.6070 | 0.6482 |
| bertscore_f1 | 0.6466 | 0.7181 | 0.6405 | 0.6428 | 0.6599 |
| cost_usd | 0.0003 | 0.0001 | 0.0007 | 0.0009 | 0.0010 |
| latency_s | 2.9743 | 4.5121 | 10.3673 | 11.0978 | 12.0642 |

## Discussion (auto-generated)

### Winner per metric

> Arms với `n_valid < 30` cho metric đó bị loại khỏi competition (insufficient sample).

| Metric | Winner | Value | n_valid |
|---|---|---|---:|
| citation_validity | **graphrag** | 1.0000 | 155 |
| citation_recall | **graphrag** | 0.9024 | 200 |
| citation_precision | **elite_ontology** | 0.9224 | 116 |
| faithfulness | **graphrag** | 0.8187 | 155 |
| content_hallucination_rate | **llm_only** | 0.3093 | 61 |
| invented_citation_rate | **graphrag** | 0.0000 | 155 |
| answer_relevance | **llm_only** | 0.6805 | 200 |
| bertscore_f1 | **llm_only** | 0.7128 | 198 |
| cost_usd | **llm_only** | 0.0001 | 200 |
| latency_s | **graphrag** | 2.7420 | 200 |
| prolog_success_rate | **elite_no_retrieval** | 0.7800 | 200 |
| first_try_success_rate | **elite_no_retrieval** | 0.5650 | 200 |
| repair_invoked_rate | **elite_no_retrieval** | 0.4350 | 200 |
| avg_repair_rounds | **elite_no_retrieval** | 0.6850 | 200 |

### Pairwise winner per arm (vs `graphrag` baseline)

> Strong-consensus wins (both directions agree). Numbers từ corrected `_vote` logic (2026-05-26 fix).

| Arm | Wins vs graphrag | graphrag wins | Split | Tie | Verdict |
|---|---:|---:|---:|---:|---|
| llm_only | 172 (86.0%) | 8 (4.0%) | 20 (10.0%) | 0 | **llm_only beats graphrag** |
| elite_no_retrieval | 44 (22.0%) | 91 (45.5%) | 65 (32.5%) | 0 | **graphrag beats elite_no_retrieval** |
| elite_ontology | 50 (25.0%) | 93 (46.5%) | 57 (28.5%) | 0 | **graphrag beats elite_ontology** |
| elite_graphrag | 60 (30.0%) | 97 (48.5%) | 43 (21.5%) | 0 | **graphrag beats elite_graphrag** |

**Elite no-retrieval ablation**: prolog_success_rate = 78%. Càng thấp càng chứng minh elite CẦN retrieval. Câu nào success nhờ LLM tự sinh được valid Prolog từ training data.

**Ontology vs GraphRAG retrieval for symbolic reasoning**: elite_ontology success=73%, elite_graphrag success=66%. `elite_ontology` retrieval cho ra Prolog program hợp lệ thường xuyên hơn.


## Caveats / Limitations

1. **Self-enhancement bias** (Zheng 2023): judge = generator = `gpt-4o-mini` → bias đều cả 5 arm. Relative compare OK, absolute có thể inflated.
2. **Elite no-retrieval prompt được relax** cho phép LLM tự cite — citation_validity của arm này dùng để cảnh báo (không equivalent với D/E).
3. **Citation source asymmetry** (audit 2026-05-26): `citation_validity` + `faithfulness` + `hallucination` dùng `record['citation_ids']` (bao gồm fallback parser của Prolog `legal_source(...)` facts). `citation_recall` + `citation_precision` chỉ dùng `parse_citations(answer_text)` — regex trên text. Khi IRAC text không có `[Điều X]` brackets (elite arms), text-based metrics có thể undercount vs ID-based metrics. Xem `reports/report_v2.md` cho `citation_text_coverage` per arm.
4. **Selection bias**: cells với `n_valid < 30` được mark 'insufficient'. Cells với `n_valid << n_total` (e.g., elite_no_retrieval citation_validity n=52) là conditional mean trên subset, KHÔNG comparable trực tiếp với arm có `n_valid` cao. Macro + micro tables được show để giảm bias.
5. **Pairwise judge** (FIXED 2026-05-26): trước đây `_vote(w, a_first=False)` invert vote_ba → tất cả consensus đều thành 'split'. Đã fix — bảng dưới phản ánh đúng. Position bias trong realityー moderate, không 'mạnh' như báo cáo cũ.
6. **Prolog rollback** đo trên max=2 repair rounds (default elite). Cap thấp → chưa thấy điểm hội tụ thật của LLM-with-feedback.
7. **SWI-Prolog timeout=15s** — câu phức tạp có thể bị giết silently → count vào prolog_success=False (status có thể là 'unable_to_conclude').
