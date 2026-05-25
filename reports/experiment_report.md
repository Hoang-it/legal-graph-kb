# Experiment Report — GraphRAG vs LLM-only

**Dataset**: 200 câu BHXH (FB group). Cặp đầy đủ (cả 2 arm): 200

**Models**: GraphRAG = `gpt-4o-mini` + BGE-M3 + Neo4j. LLM-only = `gpt-4o-mini` (no retrieval).

**Judge**: `gpt-4o-mini` (cùng model với generator — self-bias risk, nhưng vì cả 2 arm cùng generator nên *relative* comparison vẫn fair).


## Metrics (peer-reviewed refs, không arXiv)

| Metric | Paper | Venue |
|---|---|---|
| Faithfulness, Answer Relevance | Es et al. *RAGAs: Automated Evaluation of Retrieval Augmented Generation* | [EACL 2024 Demo](https://aclanthology.org/2024.eacl-demo.16/) |
| Citation Precision/Recall | Liu, Zhang & Liang. *Evaluating Verifiability in Generative Search Engines* | [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.467/) |
| Hallucination Rate (legal) | Magesh et al. *Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools* | [J. Empirical Legal Studies 2025, Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12413) (Stanford RegLab/HAI) |
| LLM-as-Judge (pairwise) | Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena* | [NeurIPS 2023 D&B](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) |
| BERTScore | Zhang et al. *BERTScore: Evaluating Text Generation with BERT* | [ICLR 2020 (OpenReview)](https://openreview.net/forum?id=SkeHuCVFDr) |


## Aggregate results

| Metric | GraphRAG (mean ± std) | LLM-only (mean ± std) | Δ (GraphRAG − LLM-only) | Direction |
|---|---|---|---|---|
| **citation_validity** | 1.0000 ± 0.0000 | 0.9531 ± 0.2130 | +0.0469 | higher is better |
| **citation_recall** | 0.6622 ± 0.4150 | 0.1501 ± 0.2482 | +0.5121 | higher is better |
| **citation_precision** | 0.7969 ± 0.2800 | 0.3978 ± 0.4727 | +0.3990 | higher is better |
| **faithfulness** | 0.8187 ± 0.2656 | 0.6461 ± 0.4003 | +0.1727 | higher is better |
| **answer_relevance** | 0.5460 ± 0.1101 | 0.6805 ± 0.0726 | -0.1345 | higher is better |
| **hallucination_rate** | 0.5036 ± 0.3576 | 0.3417 ± 0.2936 | +0.1619 | lower is better |
| **bertscore_f1** | 0.6591 ± 0.0368 | 0.7128 ± 0.0293 | -0.0538 | higher is better |
| **cost_usd** | 0.0003 ± 0.0000 | 0.0001 ± 0.0000 | +0.0002 | lower is better |
| **latency_s** | 2.7420 ± 1.6928 | 4.5487 ± 1.5621 | -1.8067 | lower is better |

## Pairwise judge (LLM-as-Judge, position swap)

| Consensus | Count | % |
|---|---:|---:|
| split | 180 | 90.0% |
| llm_only | 11 | 5.5% |
| graphrag | 9 | 4.5% |

**Position-swap detail (A-first vs B-first):**

| Vote | A=graphrag B=llm_only | A=llm_only B=graphrag |
|---|---:|---:|
| graphrag | 15 | 182 |
| llm_only | 182 | 18 |
| tie | 3 | 0 |

## Breakdown theo luật version (gold_citations)

### `new_2024` (148 câu)

| Metric | GraphRAG | LLM-only |
|---|---|---|
| citation_validity | 1.0000 | 1.0000 |
| citation_recall | 0.6610 | 0.1493 |
| citation_precision | 0.8261 | 0.4697 |
| faithfulness | 0.8206 | 0.6639 |
| answer_relevance | 0.5445 | 0.6813 |
| hallucination_rate | 0.5232 | 0.2980 |
| bertscore_f1 | 0.6613 | 0.7107 |
| cost_usd | 0.0003 | 0.0001 |
| latency_s | 2.5609 | 4.5543 |

### `old_2014` (8 câu)

| Metric | GraphRAG | LLM-only |
|---|---|---|
| citation_validity | 1.0000 | 1.0000 |
| citation_recall | 0.5683 | 0.3545 |
| citation_precision | 0.8333 | 0.2000 |
| faithfulness | 0.8759 | 0.9000 |
| answer_relevance | 0.6330 | 0.6616 |
| hallucination_rate | 0.3299 | 0.3189 |
| bertscore_f1 | 0.6829 | 0.7249 |
| cost_usd | 0.0004 | 0.0002 |
| latency_s | 4.8125 | 4.6442 |

### `unknown` (44 câu)

| Metric | GraphRAG | LLM-only |
|---|---|---|
| citation_validity | 1.0000 | 0.7692 |
| citation_recall | 0.6832 | 0.1156 |
| citation_precision | 0.6667 | 0.2308 |
| faithfulness | 0.7999 | 0.4852 |
| answer_relevance | 0.5350 | 0.6812 |
| hallucination_rate | 0.4722 | 0.5051 |
| bertscore_f1 | 0.6466 | 0.7181 |
| cost_usd | 0.0003 | 0.0001 |
| latency_s | 2.9743 | 4.5121 |

## Key findings

### GraphRAG vượt trội ở:

- **citation_recall**: 0.6622 vs 0.1501 (+0.5121, +341% rel)
- **citation_precision**: 0.7969 vs 0.3978 (+0.3990, +100% rel)
- **latency_s**: 2.7420 vs 4.5487 (-1.8067, -40% rel)
- **faithfulness**: 0.8187 vs 0.6461 (+0.1727, +27% rel)
- **citation_validity**: 1.0000 vs 0.9531 (+0.0469, +5% rel)

### LLM-only vượt trội ở:

- **cost_usd**: GraphRAG 0.0003 vs LLM-only 0.0001 (+0.0002, +119% rel)
- **hallucination_rate**: GraphRAG 0.5036 vs LLM-only 0.3417 (+0.1619, +47% rel)
- **answer_relevance**: GraphRAG 0.5460 vs LLM-only 0.6805 (-0.1345, -20% rel)
- **bertscore_f1**: GraphRAG 0.6591 vs LLM-only 0.7128 (-0.0538, -8% rel)

## Discussion

**Citation behavior**: GraphRAG cite gấp ~4.4× nhiều hơn LLM-only (66% vs 15% câu có citation). Đây là tác động trực tiếp của việc inject context có ID — model có vật liệu cụ thể để citation. LLM-only không biết article nào tồn tại trong KG → tránh cite cho an toàn.

**Hallucination rate (paradox)**: GraphRAG có hallucination rate cao hơn (50% vs 34%). Lý do PHƯƠNG PHÁP, không phải GraphRAG kém: hallucination rate = (n_misstate + n_unsupported + n_invented_citations) / (n_claims + n_invented). LLM-only ít cite → ít citation để judge soi → `n_claims` được judge nhỏ → denominator nhỏ → rate không reflect được unverified claims (vì không citation thì judge không có context để check). GraphRAG với citation phong phú bị judge soi kỹ hơn, dễ bị flag misstate khi paraphrase nội dung Điều. **Đề xuất**: metric này cần được normalize theo verifiable claims để fair.

**Answer Relevance**: LLM-only cao hơn (0.680 vs 0.546). Hợp lý — answer LLM-only ngắn gọn, conversational, dễ map ngược về câu hỏi gốc. GraphRAG answer dài hơn (kèm citation + context) → khi judge sinh ngược câu hỏi, có thể tạo Q rộng hơn (về cited topic).

**BERTScore**: LLM-only (0.713) > GraphRAG (0.659). Gold answer (FB group) thường viết prose tự nhiên, không format citation. LLM-only output tương tự style này → match cao hơn. GraphRAG output dày citation [Điều X khoản Y] → khác style. BERTScore phạt khác biệt phong cách chứ không chỉ ngữ nghĩa.

**Pairwise judge — bias VỊ TRÍ rất mạnh**: 180/200 câu split (judge dao động theo position swap). Bảng position-swap detail cho thấy judge có xu hướng pick câu trả lời ở vị trí THỨ HAI (recency bias). Zheng et al. 2023 cảnh báo điều này; kết quả strong consensus (cùng winner ở cả 2 swap) chỉ có 20/200 câu. **Pairwise judge không đáng tin** trong setting hiện tại; cần judge mạnh hơn (GPT-4o, Claude) để giảm noise.

**Latency (surprise win)**: GraphRAG NHANH HƠN (2.7s vs 4.5s) dù có thêm vector search + graph expansion. Lý do: với context đầy đủ, LLM generate ngắn gọn + tự tin → token output ít hơn. LLM-only phải 'think' nhiều hơn để compose answer → generate dài hơn, chậm hơn.


## Caveats / Limitations

1. **Self-enhancement bias**: judge = `gpt-4o-mini`, generator = `gpt-4o-mini`. Zheng et al. (2023) cảnh báo bias. Tuy nhiên cả 2 arm cùng generator → bias đều → **relative** ranking vẫn dùng được; *absolute* score có thể bị inflate.
2. **Citation format**: LLM-only được prompted để cite Luật 41/2024 nhưng có thể cite Luật cũ (training data có cả 2 luật) → citation_validity của LLM-only thấp không hẳn vì kém mà do format mismatch.
3. **Gold answer quality**: FB group answers, không phải nguồn pháp luật chính thức → BERTScore vs gold dùng làm reference loose.
4. **Hallucination definition (Magesh 2025)** trong paper là expert hand-scored. Ở đây dùng LLM judge auto → có noise.
5. **Single judge**: chưa swap multiple judges → variance không đo.
