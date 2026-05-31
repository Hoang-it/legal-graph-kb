# Academic Metrics Report

- metric_version: `academic_v2`
- n_input_records: `2200`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6682 | 4.3942 |
| llm_only | 200 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7139 | 4.6997 |
| logic_lm_no_retrieval | 200 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5493 | 12.9868 |
| logic_lm_ontology | 200 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.5478 | 13.3554 |
| logic_lm_graphrag | 200 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.4904 | 15.3858 |
| logic_lm_graphrag__gpt-4_1 | 200 | 0.0338 | 0.0212 | 0.0234 | 0.0000 | 0.6204 | 9.9546 |
| logic_lm_graphrag__gpt-4o | 200 | 0.0260 | 0.0208 | 0.0212 | 0.0000 | 0.6274 | 7.1562 |
| logic_lm_graphrag__gpt-5-mini | 200 | 0.0315 | 0.0189 | 0.0183 | 0.0041 | 0.4013 | 41.1039 |
| logic_lm_no_retrieval__gpt-4_1 | 200 | 0.0008 | 0.0025 | 0.0013 | 0.0000 | 0.5933 | 7.9387 |
| logic_lm_no_retrieval__gpt-4o | 200 | 0.0033 | 0.0100 | 0.0048 | 0.0000 | 0.5415 | 7.1271 |
| logic_lm_no_retrieval__gpt-5-mini | 200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6592 | 56.0409 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| graphrag | 0.0000 (sum=0/402) | 0.0000 (sum=0/409) | 0.0000 (sum=0/409) |
| llm_only | 0.0000 (sum=0/402) | 0.0000 (sum=0/29) | 1.0000 (sum=29/29) |
| logic_lm_no_retrieval | 0.0000 (sum=0/402) | 0.0000 (sum=0/18) | 1.0000 (sum=18/18) |
| logic_lm_ontology | 0.0000 (sum=0/402) | 0.0000 (sum=0/43) | 1.0000 (sum=43/43) |
| logic_lm_graphrag | 0.0000 (sum=0/402) | 0.0000 (sum=0/46) | 1.0000 (sum=46/46) |
| logic_lm_graphrag__gpt-4_1 | 0.0299 (sum=12/402) | 0.0280 (sum=12/429) | 0.0000 (sum=0/429) |
| logic_lm_graphrag__gpt-4o | 0.0274 (sum=11/402) | 0.0312 (sum=11/352) | 0.0000 (sum=0/352) |
| logic_lm_graphrag__gpt-5-mini | 0.0299 (sum=12/402) | 0.0303 (sum=12/396) | 0.0025 (sum=1/396) |
| logic_lm_no_retrieval__gpt-4_1 | 0.0025 (sum=1/402) | 0.0030 (sum=1/337) | 0.0000 (sum=0/337) |
| logic_lm_no_retrieval__gpt-4o | 0.0050 (sum=2/402) | 0.0074 (sum=2/269) | 0.0000 (sum=0/269) |
| logic_lm_no_retrieval__gpt-5-mini | 0.0000 (sum=0/402) | 0.0000 (sum=0/32) | 0.0000 (sum=0/32) |

## Prolog Metrics

| Arm | n_prolog | first_try_solution | repair_invoked | repair_success |
|---|---:|---:|---:|---:|
| logic_lm_no_retrieval | 200 | 0.5650 | 0.4350 | 0.5402 (sum=47/87) |
| logic_lm_ontology | 200 | 0.4250 | 0.5750 | 0.6609 (sum=76/115) |
| logic_lm_graphrag | 200 | 0.5500 | 0.4500 | 0.3667 (sum=33/90) |
| logic_lm_graphrag__gpt-4_1 | 200 | 0.8300 | 0.1700 | 0.4706 (sum=16/34) |
| logic_lm_graphrag__gpt-4o | 200 | 0.7450 | 0.2550 | 0.6275 (sum=32/51) |
| logic_lm_graphrag__gpt-5-mini | 200 | 0.3950 | 0.6050 | 0.3306 (sum=40/121) |
| logic_lm_no_retrieval__gpt-4_1 | 200 | 0.7200 | 0.2800 | 0.5000 (sum=28/56) |
| logic_lm_no_retrieval__gpt-4o | 200 | 0.5050 | 0.4950 | 0.5455 (sum=54/99) |
| logic_lm_no_retrieval__gpt-5-mini | 200 | 0.7800 | 0.2200 | 0.9545 (sum=42/44) |

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
  },
  "logic_lm_graphrag__gpt-4_1": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_graphrag__gpt-4o": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_graphrag__gpt-5-mini": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_no_retrieval__gpt-4_1": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_no_retrieval__gpt-4o": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_no_retrieval__gpt-5-mini": {
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
| graphrag | 0 | 29 |
| llm_only | 0 | 173 |
| logic_lm_no_retrieval | 0 | 184 |
| logic_lm_ontology | 0 | 158 |
| logic_lm_graphrag | 0 | 159 |
| logic_lm_graphrag__gpt-4_1 | 0 | 4 |
| logic_lm_graphrag__gpt-4o | 0 | 1 |
| logic_lm_graphrag__gpt-5-mini | 0 | 78 |
| logic_lm_no_retrieval__gpt-4_1 | 0 | 0 |
| logic_lm_no_retrieval__gpt-4o | 0 | 1 |
| logic_lm_no_retrieval__gpt-5-mini | 0 | 177 |
