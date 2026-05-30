# Academic Metrics Report

- metric_version: `academic_v1`
- n_input_records: `460`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag_v5_m2 | 30 | 0.2236 | 0.2111 | 0.1994 | 1.0000 | 0.6343 | 4.1914 |
| graphrag_v5 | 30 | 0.2361 | 0.2133 | 0.2093 | 1.0000 | 0.6319 | 39.1799 |
| graphrag | 200 | 0.1120 | 0.0820 | 0.0848 | 0.0000 | 0.6682 | 4.3942 |
| llm_only | 200 | 0.0067 | 0.0100 | 0.0075 | 1.0000 | 0.7139 | 4.6997 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| graphrag_v5_m2 | 0.2069 (sum=12/58) | 0.1935 (sum=12/62) | 1.0000 (sum=72/72) |
| graphrag_v5 | 0.2069 (sum=12/58) | 0.2553 (sum=12/47) | 1.0000 (sum=55/55) |
| graphrag | 0.0746 (sum=30/402) | 0.0777 (sum=30/386) | 0.0000 (sum=0/409) |
| llm_only | 0.0050 (sum=2/402) | 0.0690 (sum=2/29) | 1.0000 (sum=29/29) |

## Prolog Metrics

| Arm | n_prolog | first_try_solution | repair_invoked | repair_success |
|---|---:|---:|---:|---:|

## BERTScore Status

```json
{
  "graphrag_v5_m2": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "graphrag_v5": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "graphrag": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "llm_only": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  }
}
```

## Error Counts

| Arm | pred_citation_parse_errors | records_with_no_pred_citations |
|---|---:|---:|
| graphrag_v5_m2 | 0 | 2 |
| graphrag_v5 | 0 | 5 |
| graphrag | 0 | 29 |
| llm_only | 0 | 173 |
