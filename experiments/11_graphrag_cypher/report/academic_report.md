# Academic Metrics Report

- metric_version: `academic_v2`
- n_input_records: `250`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag_cypher | 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7097 | 8.5808 |
| graphrag | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6682 | 4.3942 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| graphrag_cypher | 0.0000 (sum=0/96) | 0.0000 (sum=0/65) | 0.0000 (sum=0/65) |
| graphrag | 0.0000 (sum=0/402) | 0.0000 (sum=0/409) | 0.0000 (sum=0/409) |

## Prolog Metrics

| Arm | n_prolog | first_try_solution | repair_invoked | repair_success |
|---|---:|---:|---:|---:|

## BERTScore Status

```json
{
  "graphrag_cypher": {
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
  }
}
```

## Error Counts

| Arm | pred_citation_parse_errors | records_with_no_pred_citations |
|---|---:|---:|
| graphrag_cypher | 0 | 6 |
| graphrag | 0 | 29 |
