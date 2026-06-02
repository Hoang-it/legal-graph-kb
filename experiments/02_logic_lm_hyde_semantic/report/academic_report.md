# Academic Metrics Report

- metric_version: `academic_v2`
- n_input_records: `250`
- gold source: `record.gold_articles`
- judge metrics: not included

## Headline Macro Metrics

| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |
|---|---:|---:|---:|---:|---:|---:|---:|
| logic_lm_hyde_semantic | 50 | 0.0000 | 0.0000 | 0.0000 | 0.8400 | 0.6075 | 27.7607 |
| logic_lm_hyde_semantic_nohyp | 50 | 0.0000 | 0.0000 | 0.0000 | 0.8800 | 0.6226 | 14.9039 |
| qa_hyde_semantic | 50 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.6688 | 4.4144 |
| graphrag | 50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6642 | 3.8878 |
| llm_only | 50 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.7193 | 4.2859 |

## Text-overlap Macro Metrics (ROUGE / BLEU)

| Arm | n | rouge1 | rouge2 | rougeL | bleu |
|---|---:|---:|---:|---:|---:|
| logic_lm_hyde_semantic | 50 | 0.4515 | 0.1867 | 0.2540 | 0.0418 |
| logic_lm_hyde_semantic_nohyp | 50 | 0.4750 | 0.1994 | 0.2662 | 0.0417 |
| qa_hyde_semantic | 50 | 0.4496 | 0.1796 | 0.2652 | 0.0425 |
| graphrag | 50 | 0.4989 | 0.1807 | 0.2772 | 0.0374 |
| llm_only | 50 | 0.5727 | 0.2897 | 0.3249 | 0.0729 |

## Citation Micro Metrics

| Arm | recall | precision | display_rate |
|---|---:|---:|---:|
| logic_lm_hyde_semantic | 0.0000 (sum=0/104) | 0.0000 (sum=0/67) | 0.7612 (sum=51/67) |
| logic_lm_hyde_semantic_nohyp | 0.0000 (sum=0/104) | 0.0000 (sum=0/64) | 0.8125 (sum=52/64) |
| qa_hyde_semantic | 0.0000 (sum=0/104) | 0.0000 (sum=0/156) | 1.0000 (sum=156/156) |
| graphrag | 0.0000 (sum=0/104) | 0.0000 (sum=0/111) | 0.0000 (sum=0/111) |
| llm_only | 0.0000 (sum=0/104) | 0.0000 (sum=0/15) | 1.0000 (sum=15/15) |

## Prolog Metrics

| Arm | n_prolog | first_try_solution | repair_invoked | repair_success |
|---|---:|---:|---:|---:|
| logic_lm_hyde_semantic | 50 | 0.8200 | 0.1800 | 0.1111 (sum=1/9) |
| logic_lm_hyde_semantic_nohyp | 50 | 0.7800 | 0.2200 | 0.4545 (sum=5/11) |

## BERTScore Status

```json
{
  "logic_lm_hyde_semantic": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "logic_lm_hyde_semantic_nohyp": {
    "status": "ok",
    "model_type": "bert-base-multilingual-cased",
    "lang": "vi",
    "device": "cuda",
    "rescale_with_baseline": false
  },
  "qa_hyde_semantic": {
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

## Text-overlap Status (ROUGE / BLEU)

```json
{
  "logic_lm_hyde_semantic": {
    "status": "ok",
    "rouge": {
      "types": [
        "rouge1",
        "rouge2",
        "rougeL"
      ],
      "score": "fmeasure",
      "use_stemmer": false
    },
    "bleu": {
      "impl": "sacrebleu.sentence_bleu",
      "tokenize": "13a",
      "scale": "[0,1] = sacrebleu/100"
    }
  },
  "logic_lm_hyde_semantic_nohyp": {
    "status": "ok",
    "rouge": {
      "types": [
        "rouge1",
        "rouge2",
        "rougeL"
      ],
      "score": "fmeasure",
      "use_stemmer": false
    },
    "bleu": {
      "impl": "sacrebleu.sentence_bleu",
      "tokenize": "13a",
      "scale": "[0,1] = sacrebleu/100"
    }
  },
  "qa_hyde_semantic": {
    "status": "ok",
    "rouge": {
      "types": [
        "rouge1",
        "rouge2",
        "rougeL"
      ],
      "score": "fmeasure",
      "use_stemmer": false
    },
    "bleu": {
      "impl": "sacrebleu.sentence_bleu",
      "tokenize": "13a",
      "scale": "[0,1] = sacrebleu/100"
    }
  },
  "graphrag": {
    "status": "ok",
    "rouge": {
      "types": [
        "rouge1",
        "rouge2",
        "rougeL"
      ],
      "score": "fmeasure",
      "use_stemmer": false
    },
    "bleu": {
      "impl": "sacrebleu.sentence_bleu",
      "tokenize": "13a",
      "scale": "[0,1] = sacrebleu/100"
    }
  },
  "llm_only": {
    "status": "ok",
    "rouge": {
      "types": [
        "rouge1",
        "rouge2",
        "rougeL"
      ],
      "score": "fmeasure",
      "use_stemmer": false
    },
    "bleu": {
      "impl": "sacrebleu.sentence_bleu",
      "tokenize": "13a",
      "scale": "[0,1] = sacrebleu/100"
    }
  }
}
```

## Error Counts

| Arm | pred_citation_parse_errors | records_with_no_pred_citations |
|---|---:|---:|
| logic_lm_hyde_semantic | 0 | 0 |
| logic_lm_hyde_semantic_nohyp | 0 | 0 |
| qa_hyde_semantic | 0 | 0 |
| graphrag | 0 | 11 |
| llm_only | 0 | 36 |
