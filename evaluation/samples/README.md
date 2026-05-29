# Academic Metrics Dummy Sample

Dummy dataset de chay thu core `evaluation.compute_academic_metrics`.

`records.json` la mot list records. Core evaluation khong biet folder, group,
`arm`, hoac `stt`.

```powershell
python -m evaluation.compute_academic_metrics `
  --records evaluation/samples/records.json `
  --output-dir metrics/sample_academic
```

Sample nay co y khong chua `gold_answer`, de BERTScore khong load model ngoai.
Citation/prolog metrics van duoc tinh day du.
