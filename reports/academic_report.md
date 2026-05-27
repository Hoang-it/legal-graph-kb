# Academic Metrics Report

- metric_version: `academic_v1`
- results_root: `data\eval\results`
- gold source: `gold_citations_raw`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag | 200 | 0.1292 | 0.1050 | 0.1081 | 0.0000 | 0.6660 | 2.7420 |
| llm_only | 200 | 0.0056 | 0.0100 | 0.0061 | 0.1771 | 0.7141 | 4.5487 |
| elite_no_retrieval | 200 | 0.0144 | 0.0233 | 0.0160 | 0.0000 | 0.5321 | 11.0912 |
| elite_ontology | 200 | 0.0712 | 0.0567 | 0.0578 | 0.5471 | 0.4941 | 12.1155 |
| elite_graphrag | 200 | 0.0965 | 0.0867 | 0.0852 | 0.4990 | 0.4902 | 14.2871 |
| elite_graphrag_logic | 100 | 0.0821 | 0.0900 | 0.0810 | 0.6145 | 0.4391 | 27.8011 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| graphrag | 0.0875 (Σ=35/400) | 0.1032 (Σ=35/339) | 0.0000 (Σ=0/368) |
| llm_only | 0.0075 (Σ=3/400) | 0.0385 (Σ=3/78) | 0.1975 (Σ=16/81) |
| elite_no_retrieval | 0.0200 (Σ=8/400) | 0.1039 (Σ=8/77) | 0.0000 (Σ=0/77) |
| elite_ontology | 0.0400 (Σ=16/400) | 0.0727 (Σ=16/220) | 0.4691 (Σ=114/243) |
| elite_graphrag | 0.0700 (Σ=28/400) | 0.0912 (Σ=28/307) | 0.3932 (Σ=127/323) |
| elite_graphrag_logic | 0.0597 (Σ=12/201) | 0.0839 (Σ=12/143) | 0.4733 (Σ=71/150) |

## Prolog Metrics

| Arm | n_elite | first_try_solution | repair_invoked | repair_success |
|---|---:|---:|---:|---:|
| elite_no_retrieval | 200 | 0.5650 | 0.4350 | 0.4943 (Σ=43/87) |
| elite_ontology | 200 | 0.3700 | 0.6300 | 0.5714 (Σ=72/126) |
| elite_graphrag | 200 | 0.5400 | 0.4600 | 0.3804 (Σ=35/92) |
| elite_graphrag_logic | 100 | 0.3700 | 0.6300 | 0.3968 (Σ=25/63) |

## BERTScore Status

```json
{
  "status": "ok",
  "model_type": "bert-base-multilingual-cased",
  "lang": "vi",
  "device": "cuda",
  "rescale_with_baseline": false
}
```

## Error Counts

| Arm | pred_citation_parse_errors | records_with_no_pred_citations |
|---|---:|---:|
| graphrag | 0 | 45 |
| llm_only | 0 | 136 |
| elite_no_retrieval | 0 | 148 |
| elite_ontology | 0 | 16 |
| elite_graphrag | 0 | 0 |
| elite_graphrag_logic | 0 | 1 |
