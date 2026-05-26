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


## Aggregate results (mean ± std)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag | Direction |
|---|---|---|---|---|---|---|
| **citation_validity** | 1.0000 ± 0.0000 | 0.9531 ± 0.2130 | 0.9936 ± 0.0462 | 0.9909 ± 0.0812 | 0.9983 ± 0.0236 | higher better |
| **citation_recall** | 0.9024 ± 0.1908 | 0.1834 ± 0.2600 | 0.0000 ± 0.0000 | 0.1703 ± 0.1687 | 0.1454 ± 0.1719 | higher better |
| **citation_precision** | 0.7958 ± 0.3007 | 0.4300 ± 0.4803 | N/A | 0.9224 ± 0.2170 | 0.8220 ± 0.3041 | higher better |
| **faithfulness** | 0.8187 ± 0.2656 | 0.6461 ± 0.4003 | 0.8143 ± 0.2845 | 0.5844 ± 0.3263 | 0.7507 ± 0.2724 | higher better |
| **answer_relevance** | 0.5460 ± 0.1101 | 0.6805 ± 0.0726 | 0.6241 ± 0.1701 | 0.5885 ± 0.1665 | 0.5719 ± 0.1713 | higher better |
| **hallucination_rate** | 0.5036 ± 0.3576 | 0.3417 ± 0.2936 | 0.4240 ± 0.5191 | 0.5935 ± 0.4902 | 0.4694 ± 0.4742 | lower better |
| **bertscore_f1** | 0.6591 ± 0.0368 | 0.7128 ± 0.0293 | 0.6508 ± 0.0565 | 0.6429 ± 0.0558 | 0.6391 ± 0.0650 | higher better |
| **cost_usd** | 0.0003 ± 0.0000 | 0.0001 ± 0.0000 | 0.0007 ± 0.0003 | 0.0011 ± 0.0005 | 0.0012 ± 0.0005 | lower better |
| **latency_s** | 2.7420 ± 1.6928 | 4.5487 ± 1.5621 | 11.0912 ± 7.3898 | 12.1155 ± 8.8173 | 14.0109 ± 6.8819 | lower better |

## Prolog reliability (Logic-LM metrics — chỉ áp dụng cho elite arms)

> Đo độ tin cậy của symbolic solver loop. Pan et al. EMNLP'23 báo cáo các metric tương tự để compare LLM-as-reasoner vs LLM+symbolic.

| Metric | elite_no_retrieval | elite_ontology | elite_graphrag | Direction |
|---|---|---|---|---|
| **prolog_success_rate** | 0.7800 | 0.7300 | 0.6550 | higher better |
| **first_try_success_rate** | 0.5650 | 0.3700 | 0.4850 | higher better |
| **repair_invoked_rate** | 0.4350 | 0.6300 | 0.5150 | lower better |
| **avg_repair_rounds** | 0.6850 | 0.9150 | 0.8650 | lower better |

### Prolog status distribution

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

| Consensus | Count | % |
|---|---:|---:|
| split | 183 | 91.5% |
| llm_only | 9 | 4.5% |
| graphrag | 8 | 4.0% |

**Position swap detail:**

| Vote | A=graphrag B=llm_only | A=llm_only B=graphrag |
|---|---:|---:|
| graphrag | 16 | 182 |
| llm_only | 181 | 18 |
| tie | 3 | 0 |

### `elite_no_retrieval` vs `graphrag` (n=200)

| Consensus | Count | % |
|---|---:|---:|
| split | 171 | 85.5% |
| elite_no_retrieval | 29 | 14.5% |

**Position swap detail:**

| Vote | A=graphrag B=elite_no_retrieval | A=elite_no_retrieval B=graphrag |
|---|---:|---:|
| elite_no_retrieval | 73 | 153 |
| graphrag | 91 | 47 |
| tie | 36 | 0 |

### `elite_ontology` vs `graphrag` (n=200)

| Consensus | Count | % |
|---|---:|---:|
| split | 162 | 81.0% |
| elite_ontology | 38 | 19.0% |

**Position swap detail:**

| Vote | A=graphrag B=elite_ontology | A=elite_ontology B=graphrag |
|---|---:|---:|
| elite_ontology | 88 | 149 |
| graphrag | 94 | 50 |
| tie | 18 | 1 |

### `elite_graphrag` vs `graphrag` (n=200)

| Consensus | Count | % |
|---|---:|---:|
| split | 169 | 84.5% |
| elite_graphrag | 31 | 15.5% |

**Position swap detail:**

| Vote | A=graphrag B=elite_graphrag | A=elite_graphrag B=graphrag |
|---|---:|---:|
| elite_graphrag | 94 | 137 |
| graphrag | 97 | 60 |
| tie | 9 | 3 |

## Breakdown theo luật version (từ gold_citations_raw)

### `new_2024` (148 câu)

| Metric | graphrag | llm_only | elite_no_retrieval | elite_ontology | elite_graphrag |
|---|---|---|---|---|---|
| citation_validity | 1.0000 | 1.0000 | 0.9910 | 0.9952 | 0.9977 |
| citation_recall | 0.9049 | 0.1701 | 0.0000 | 0.1749 | 0.1370 |
| citation_precision | 0.8214 | 0.4722 | N/A | 0.9138 | 0.8685 |
| faithfulness | 0.8206 | 0.6639 | 0.8299 | 0.6026 | 0.7498 |
| answer_relevance | 0.5445 | 0.6813 | 0.6240 | 0.5896 | 0.5540 |
| hallucination_rate | 0.5232 | 0.2980 | 0.3964 | 0.5710 | 0.4367 |
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
| answer_relevance | 0.6330 | 0.6616 | 0.6442 | 0.4677 | 0.4854 |
| hallucination_rate | 0.3299 | 0.3189 | 0.8000 | 0.6833 | 0.4083 |
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
| answer_relevance | 0.5350 | 0.6812 | 0.6209 | 0.6070 | 0.6482 |
| hallucination_rate | 0.4722 | 0.5051 | 0.3383 | 0.6598 | 0.5905 |
| bertscore_f1 | 0.6466 | 0.7181 | 0.6405 | 0.6428 | 0.6599 |
| cost_usd | 0.0003 | 0.0001 | 0.0007 | 0.0009 | 0.0010 |
| latency_s | 2.9743 | 4.5121 | 10.3673 | 11.0978 | 12.0642 |

## Discussion (auto-generated)

### Winner per metric

| Metric | Winner | Value |
|---|---|---|
| citation_validity | **graphrag** | 1.0000 |
| citation_recall | **graphrag** | 0.9024 |
| citation_precision | **elite_ontology** | 0.9224 |
| faithfulness | **graphrag** | 0.8187 |
| answer_relevance | **llm_only** | 0.6805 |
| hallucination_rate | **llm_only** | 0.3417 |
| bertscore_f1 | **llm_only** | 0.7128 |
| cost_usd | **llm_only** | 0.0001 |
| latency_s | **graphrag** | 2.7420 |
| prolog_success_rate | **elite_no_retrieval** | 0.7800 |
| first_try_success_rate | **elite_no_retrieval** | 0.5650 |
| repair_invoked_rate | **elite_no_retrieval** | 0.4350 |
| avg_repair_rounds | **elite_no_retrieval** | 0.6850 |

**Elite no-retrieval ablation**: prolog_success_rate = 78%. Càng thấp càng chứng minh elite CẦN retrieval. Câu nào success nhờ LLM tự sinh được valid Prolog từ training data.

**Ontology vs GraphRAG retrieval for symbolic reasoning**: elite_ontology success=73%, elite_graphrag success=66%. `elite_ontology` retrieval cho ra Prolog program hợp lệ thường xuyên hơn.


## Caveats / Limitations

1. **Self-enhancement bias** (Zheng 2023): judge = generator = `gpt-4o-mini` → bias đều cả 5 arm. Relative compare OK, absolute có thể inflated.
2. **Elite no-retrieval prompt được relax** cho phép LLM tự cite — citation_validity của arm này dùng để cảnh báo (không equivalent với D/E).
3. **Citation extraction** từ IRAC text (elite arms) dùng cả bracketed `[Điều X khoản Y]` và inline `Điều X, khoản Y` patterns; có thể miss vài citation natural language.
4. **Pairwise judge** vs `graphrag` baseline có position bias mạnh (đã thấy trong eval 2-arm trước). Chỉ tin strong-consensus rows.
5. **Prolog rollback** đo trên max=2 repair rounds (default elite). Cap thấp → chưa thấy điểm hội tụ thật của LLM-with-feedback.
6. **SWI-Prolog timeout=15s** — câu phức tạp có thể bị giết silently → count vào prolog_success=False (status có thể là 'unable_to_conclude').
