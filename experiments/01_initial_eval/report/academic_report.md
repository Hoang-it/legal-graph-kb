# Academic Metrics Report

- metric_version: `academic_v1`
- n_input_records: `2200`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| graphrag | 200 | 0.1120 | 0.0820 | 0.0848 | 0.0000 | 0.6682 | 4.3942 |
| llm_only | 200 | 0.0067 | 0.0100 | 0.0075 | 1.0000 | 0.7139 | 4.6997 |
| logic_lm_no_retrieval | 200 | 0.0023 | 0.0100 | 0.0036 | 1.0000 | 0.5493 | 12.9868 |
| logic_lm_ontology | 200 | 0.0073 | 0.0150 | 0.0086 | 1.0000 | 0.5478 | 13.3554 |
| logic_lm_graphrag | 200 | 0.0175 | 0.0200 | 0.0183 | 1.0000 | 0.4904 | 15.3858 |
| logic_lm_graphrag__gpt-4_1 | 200 | 0.1565 | 0.1112 | 0.1193 | 0.0000 | 0.6204 | 9.9546 |
| logic_lm_graphrag__gpt-4o | 200 | 0.1407 | 0.1070 | 0.1130 | 0.0000 | 0.6274 | 7.1562 |
| logic_lm_graphrag__gpt-5-mini | 200 | 0.0785 | 0.0535 | 0.0554 | 0.0041 | 0.4013 | 41.1039 |
| logic_lm_no_retrieval__gpt-4_1 | 200 | 0.0433 | 0.0475 | 0.0413 | 0.0000 | 0.5933 | 7.9387 |
| logic_lm_no_retrieval__gpt-4o | 200 | 0.0400 | 0.0500 | 0.0419 | 0.0000 | 0.5415 | 7.1271 |
| logic_lm_no_retrieval__gpt-5-mini | 200 | 0.0081 | 0.0175 | 0.0095 | 0.0000 | 0.6592 | 56.0409 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| graphrag | 0.0748 (sum=30/401) | 0.0777 (sum=30/386) | 0.0000 (sum=0/409) |
| llm_only | 0.0050 (sum=2/401) | 0.0690 (sum=2/29) | 1.0000 (sum=29/29) |
| logic_lm_no_retrieval | 0.0050 (sum=2/401) | 0.1111 (sum=2/18) | 1.0000 (sum=18/18) |
| logic_lm_ontology | 0.0075 (sum=3/401) | 0.0714 (sum=3/42) | 1.0000 (sum=43/43) |
| logic_lm_graphrag | 0.0100 (sum=4/401) | 0.0870 (sum=4/46) | 1.0000 (sum=46/46) |
| logic_lm_graphrag__gpt-4_1 | 0.1147 (sum=46/401) | 0.1150 (sum=46/400) | 0.0000 (sum=0/429) |
| logic_lm_graphrag__gpt-4o | 0.1047 (sum=42/401) | 0.1232 (sum=42/341) | 0.0000 (sum=0/352) |
| logic_lm_graphrag__gpt-5-mini | 0.0648 (sum=26/401) | 0.0751 (sum=26/346) | 0.0025 (sum=1/396) |
| logic_lm_no_retrieval__gpt-4_1 | 0.0399 (sum=16/401) | 0.0542 (sum=16/295) | 0.0000 (sum=0/337) |
| logic_lm_no_retrieval__gpt-4o | 0.0324 (sum=13/401) | 0.0533 (sum=13/244) | 0.0000 (sum=0/269) |
| logic_lm_no_retrieval__gpt-5-mini | 0.0100 (sum=4/401) | 0.1290 (sum=4/31) | 0.0000 (sum=0/32) |

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
