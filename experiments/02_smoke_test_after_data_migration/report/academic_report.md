# Academic Metrics Report

- metric_version: `academic_v1`
- n_input_records: `25`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6731 | 7.6460 |
| llm_only | 5 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7156 | 7.1286 |
| logic_lm_no_retrieval | 5 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4054 | 69.5294 |
| logic_lm_ontology | 5 | 0.0000 | 0.0000 | 0.0000 | 0.8000 | 0.5485 | 15.2202 |
| logic_lm_graphrag | 5 | 0.1000 | 0.2000 | 0.1333 | 1.0000 | 0.6914 | 18.0334 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| graphrag | 0.0000 (sum=0/10) | 0.0000 (sum=0/12) | 0.0000 (sum=0/12) |
| llm_only | 0.0000 (sum=0/10) | 0.0000 (sum=0/4) | 1.0000 (sum=4/4) |
| logic_lm_no_retrieval | 0.0000 (sum=0/10) | 0.0000 (sum=0/3) | 0.3333 (sum=1/3) |
| logic_lm_ontology | 0.0000 (sum=0/10) | 0.0000 (sum=0/5) | 0.6667 (sum=4/6) |
| logic_lm_graphrag | 0.1000 (sum=1/10) | 0.1250 (sum=1/8) | 1.0000 (sum=9/9) |

## Prolog Metrics

| Arm | n_prolog | first_try_solution | repair_invoked | repair_success |
|---|---:|---:|---:|---:|
| logic_lm_no_retrieval | 5 | 0.6000 | 0.4000 | 0.0000 (sum=0/2) |
| logic_lm_ontology | 5 | 0.8000 | 0.2000 | 0.0000 (sum=0/1) |
| logic_lm_graphrag | 5 | 1.0000 | 0.0000 | N/A (sum=0/0) |

## BERTScore Status

```json
{
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
  },
  "logic_lm_no_retrieval": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_ontology": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_graphrag": {
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
| graphrag | 0 | 0 |
| llm_only | 0 | 2 |
| logic_lm_no_retrieval | 0 | 3 |
| logic_lm_ontology | 0 | 0 |
| logic_lm_graphrag | 0 | 0 |
