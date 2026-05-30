# Experiment 06 — Retrieval-only A/B (dense vs full_rerank)

Dataset: 200 BHXH questions. Metric granularity: article.

## Overall macro (all 200 questions with non-empty gold)

### Citation recall
| arm | n | @5 | @10 | @12 | @20 | @30 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 200 | 0.2169 | 0.2902 | 0.3166 | 0.3825 | 0.4258 | 0.4312 |
| full_rerank | 200 | 0.2488 | 0.3556 | 0.3568 | 0.3568 | 0.3568 | 0.3568 |

### Citation precision
| arm | n | @5 | @10 | @12 | @20 | @30 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 200 | 0.0620 | 0.0425 | 0.0400 | 0.0299 | 0.0254 | 0.0250 |
| full_rerank | 200 | 0.0740 | 0.0638 | 0.0627 | 0.0627 | 0.0627 | 0.0627 |

### Citation F1
| arm | n | @5 | @10 | @12 | @20 | @30 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 200 | 0.0921 | 0.0716 | 0.0685 | 0.0541 | 0.0472 | 0.0465 |
| full_rerank | 200 | 0.1078 | 0.1034 | 0.1021 | 0.1021 | 0.1021 | 0.1021 |

### Average retrieved-set size at K
| arm | n | @5 | @10 | @12 | @20 | @30 | @all |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense | 200 | 5 | 10 | 12 | 19.9500 | 27.8100 | 29.9000 |
| full_rerank | 200 | 4.8300 | 8.7600 | 9.1500 | 9.1500 | 9.1500 | 9.1500 |

### Rank-aware metrics (recommended when |gold| << K)

| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |
|---|---:|---:|---:|---:|---:|
| dense | 200 | 0.0677 | 0.1848 | 0.1796 | 0.2188 |
| full_rerank | 200 | 0.1022 | 0.2141 | 0.2225 | 0.2231 |

- **R-Precision** = precision at K=|gold| per question. Since K=|gold| here, R-Precision = recall = F1 — a single fair number when |gold| is small.
- **MRR** = mean reciprocal rank of the *first* gold article retrieved. Captures "how fast does the right answer appear at the top?"
- **NDCG@10** = binary-relevance NDCG truncated at 10, normalised by ideal DCG (= 1 if all |gold| are in top-10 in order).
- **NDCG@all** = NDCG over the full retrieved list (caps at retrieved size).

### Latency

| arm | avg elapsed (s) |
|---|---:|
| dense | 0.1570 |
| full_rerank | 1.2020 |

## Stratified by gold corpus type

### in_corpus

| arm | n | R@5 | R@10 | R@12 | R@20 | R@30 | R@all | P@5 | P@10 | P@12 | P@20 | P@30 | P@all | F1@5 | F1@10 | F1@12 | F1@20 | F1@30 | F1@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 151 | 0.2577 | 0.3549 | 0.3832 | 0.4682 | 0.5190 | 0.5262 | 0.0728 | 0.0517 | 0.0486 | 0.0365 | 0.0308 | 0.0304 | 0.1089 | 0.0872 | 0.0832 | 0.0662 | 0.0572 | 0.0565 |
| full_rerank | 151 | 0.3078 | 0.4450 | 0.4467 | 0.4467 | 0.4467 | 0.4467 | 0.0901 | 0.0784 | 0.0772 | 0.0772 | 0.0772 | 0.0772 | 0.1322 | 0.1277 | 0.1261 | 0.1261 | 0.1261 | 0.1261 |

_Rank-aware (in_corpus)_

| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |
|---|---:|---:|---:|---:|---:|
| dense | 151 | 0.0635 | 0.2104 | 0.2097 | 0.2572 |
| full_rerank | 151 | 0.1290 | 0.2649 | 0.2782 | 0.2789 |

### mixed

| arm | n | R@5 | R@10 | R@12 | R@20 | R@30 | R@all | P@5 | P@10 | P@12 | P@20 | P@30 | P@all | F1@5 | F1@10 | F1@12 | F1@20 | F1@30 | F1@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 5 | 0.1917 | 0.1917 | 0.1917 | 0.2583 | 0.4583 | 0.4583 | 0.1200 | 0.0600 | 0.0500 | 0.0405 | 0.0448 | 0.0422 | 0.1379 | 0.0863 | 0.0752 | 0.0681 | 0.0804 | 0.0760 |
| full_rerank | 5 | 0.2583 | 0.2833 | 0.2833 | 0.2833 | 0.2833 | 0.2833 | 0.1600 | 0.1133 | 0.1067 | 0.1067 | 0.1067 | 0.1067 | 0.1879 | 0.1560 | 0.1515 | 0.1515 | 0.1515 | 0.1515 |

_Rank-aware (mixed)_

| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |
|---|---:|---:|---:|---:|---:|
| dense | 5 | 0.0917 | 0.3229 | 0.1747 | 0.2511 |
| full_rerank | 5 | 0.1917 | 0.3667 | 0.2299 | 0.2299 |

### ooc

| arm | n | R@5 | R@10 | R@12 | R@20 | R@30 | R@all | P@5 | P@10 | P@12 | P@20 | P@30 | P@all | F1@5 | F1@10 | F1@12 | F1@20 | F1@30 | F1@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_rerank | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

_Rank-aware (ooc)_

| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |
|---|---:|---:|---:|---:|---:|
| dense | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| full_rerank | 8 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### unparseable

| arm | n | R@5 | R@10 | R@12 | R@20 | R@30 | R@all | P@5 | P@10 | P@12 | P@20 | P@30 | P@all | F1@5 | F1@10 | F1@12 | F1@20 | F1@30 | F1@all |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 36 | 0.0972 | 0.0972 | 0.1250 | 0.1250 | 0.1250 | 0.1250 | 0.0222 | 0.0111 | 0.0116 | 0.0069 | 0.0057 | 0.0057 | 0.0357 | 0.0198 | 0.0211 | 0.0131 | 0.0109 | 0.0108 |
| full_rerank | 36 | 0.0556 | 0.0694 | 0.0694 | 0.0694 | 0.0694 | 0.0694 | 0.0111 | 0.0098 | 0.0098 | 0.0098 | 0.0098 | 0.0098 | 0.0185 | 0.0171 | 0.0171 | 0.0171 | 0.0171 | 0.0171 |

_Rank-aware (unparseable)_

| arm | n | R-Precision | MRR | NDCG@10 | NDCG@all |
|---|---:|---:|---:|---:|---:|
| dense | 36 | 0.0972 | 0.0995 | 0.0941 | 0.1016 |
| full_rerank | 36 | 0.0000 | 0.0278 | 0.0375 | 0.0375 |

## Notes

- Recall denominator = |gold|; questions with empty gold are skipped.
- Precision denominator = |retrieved@K|; if a question's retrieved set is empty, precision = 0.
- F1 = harmonic mean of per-question P/R; macro across questions.
- Arm `dense` retrieves up to `dense_k=50` BGE-M3 LoRA hits (article-deduped).
- Arm `full_rerank` retrieves up to 12 final articles (rerank2_top_k=12).
  Reported recall@K for K > 12 caps at the natural pool size.
