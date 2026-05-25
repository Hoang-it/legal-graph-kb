# Experiments — GraphRAG vs LLM-only

So sánh **GraphRAG** (Neo4j + BGE-M3 + GPT-4o-mini) với **LLM thuần** (chỉ GPT-4o-mini, không context) trên 200 câu hỏi BHXH thu thập từ FB group.

## Dataset

`data/eval/questions_200.json` — 200 câu đầu từ sheet `Tiền xử lý` của file
`_Câu hỏi BHXH trên group facebook.xlsx`. Cấu trúc mỗi câu:
- `stt`, `question`, `group`
- `gold_answer` (198/200 có)
- `gold_citations_raw` (196/200 có) — đa số chỉ Luật BHXH 2024 (kế thừa Luật 58/2014 ở vài điểm)

**Lưu ý version**: KG build từ Luật **41/2024/QH15** mới. ~151/200 câu refer Luật 2024 (khớp KG), 2 câu refer Luật 2014 cũ, còn lại không rõ. Sẽ tag mỗi câu + breakdown trong report.

## Pipeline

```
inference (run_inference.py)
   ├── arm A: GraphRAG  → src.rag_query.RagPipeline.ask()
   └── arm B: LLM-only  → experiments.llm_only.LlmOnlyPipeline.ask()
                          (cùng SYSTEM prompt yêu cầu citation, KHÔNG có context)

→ data/eval/results/{graphrag,llm_only}/A{stt}.json

compute_metrics (compute_metrics.py)
   ├── Deterministic: citation_precision, citation_recall, latency, cost
   ├── Judge (GPT-4o-mini): faithfulness, answer_relevance, hallucination_rate, pairwise_winner
   └── Semantic: BERTScore vs gold_answer

→ data/eval/metrics.json + metrics.csv

generate_report (generate_report.py)
   → reports/experiment_report.md (so sánh, breakdown, citation paper)
```

## Metrics + Paper refs (peer-reviewed, không arXiv)

| Metric | Paper | Venue |
|---|---|---|
| **Faithfulness** | Es et al. "RAGAs: Automated Evaluation of Retrieval Augmented Generation" | [EACL 2024 Demo](https://aclanthology.org/2024.eacl-demo.16/) |
| **Answer Relevance** | Es et al. RAGAS (cùng paper) | [EACL 2024 Demo](https://aclanthology.org/2024.eacl-demo.16/) |
| **Citation Precision/Recall** | Liu, Zhang & Liang. "Evaluating Verifiability in Generative Search Engines" | [EMNLP Findings 2023](https://aclanthology.org/2023.findings-emnlp.467/) |
| **Hallucination Rate (legal)** | Magesh et al. "Hallucination-Free? Assessing the Reliability of Leading AI Legal Research Tools" | [Journal of Empirical Legal Studies 2025, Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jels.12413) (Stanford RegLab/HAI) |
| **LLM-as-Judge (pairwise)** | Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" | [NeurIPS 2023 Datasets & Benchmarks](https://papers.nips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html) |
| **BERTScore** | Zhang et al. "BERTScore: Evaluating Text Generation with BERT" | [ICLR 2020 (OpenReview)](https://openreview.net/forum?id=SkeHuCVFDr) |
| Cost / Latency | (objective, không cần paper) | — |

## Giới hạn (caveats — sẽ ghi rõ trong report)

1. **Self-enhancement bias**: judge = GPT-4o-mini, generator cũng = GPT-4o-mini → có thể bias. Tuy nhiên cả 2 arm dùng cùng generator nên bias affect đều → **relative ranking vẫn fair**. Zheng et al. NeurIPS'23 cảnh báo nhưng ghi rõ "controlled comparison" vẫn dùng được. Để strict-fair, có thể chạy lại với judge khác provider.
2. **Position bias**: pairwise judge sẽ swap A↔B mỗi câu và lấy trung bình.
3. **Law version mismatch**: 49/200 câu có thể về luật cũ → GraphRAG sẽ trả lời "không có thông tin" (đúng theo design). LLM-only có training data Luật cũ → có thể trả lời nhưng không verify được. Sẽ report breakdown.
4. **Ground truth quality**: gold_answer từ FB group, không phải nguồn pháp chính thức → dùng làm reference loose, không strict.

## Chạy

```powershell
# 1. Inference cả 2 arms (cost ~$0.80, ~30 phút với 200 câu × 2)
python -m experiments.run_inference --n 200

# 2. Compute metrics (cost ~$3.20, ~15 phút)
python -m experiments.compute_metrics

# 3. Generate report
python -m experiments.generate_report
```

Pilot trước với `--n 10` để xác nhận pipeline.

## Tính trung thực

- Không sửa bất kỳ file nào trong `src/`, `schema/`. Chỉ thêm code trong `experiments/`.
- Mọi LLM call thật → API.
- Mọi metric tính từ output thật, không hardcode.
- Per-question result lưu đầy đủ (question, answer, citations, judge raw response) → reproducible/auditable.
