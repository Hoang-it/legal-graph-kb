# Academic Metrics Report

- metric_version: `academic_v1`
- n_input_records: `4`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0.5000 | 0.5000 | 0.5000 | 0.8333 | N/A | 2.5000 |

## Citation Micro Metrics

| recall | precision | display_rate |
|---:|---:|---:|
| 0.5000 (sum=2/4) | 0.5000 (sum=2/4) | 0.7500 (sum=3/4) |

## Prolog Metrics

| n_prolog | first_try_solution | repair_invoked | repair_success |
|---:|---:|---:|---:|
| 2 | 0.5000 | 0.5000 | 0.0000 (sum=0/1) |

## BERTScore Status

```json
{
  "status": "no_records_with_gold_answer"
}
```

## Error Counts

| pred_citation_parse_errors | records_with_no_pred_citations |
|---:|---:|
| 1 | 1 |
