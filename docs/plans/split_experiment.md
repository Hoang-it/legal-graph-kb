# Plan — Tách `experiments/` thành repo độc lập, tự tổng hợp + report (leaderboard)

## Context

Hiện `experiments/` chỉ là dữ liệu kết quả; mọi logic tính/đọc metrics nằm rải ở
`eval_core/`, `scripts/exp*_*.py`, `src/`. Có **2 họ experiment** với 2 pipeline và
2 shape metrics khác nhau (cùng tên file `academic_metrics.json`):

| Họ | Exp | Metrics shape | Headline |
|---|---|---|---|
| **QA / academic** | 01–04 | `aggregates.<arm>.macro` | citation R/P/F1, display_rate, bertscore_f1, latency, 3 Prolog rate |
| **Retrieval-only** | 06–14 | `overall_macro[arm]` + `stratified[arm][stratum]`, `Ks` | recall@k, precision@k, r_precision, mrr, ndcg@k |

(05 chưa có metrics → bỏ qua + cảnh báo.)

**Mục tiêu (đã chốt với user):** dựng một thư mục **standalone** (sau này lift ra thành
git repo riêng) có khả năng **đọc metrics đã sinh sẵn của nhiều experiment → tổng hợp →
xuất LEADERBOARD theo 1 metric**, family-aware, bao **cả 2 họ**. **KHÔNG** chạy lại
inference (không cần Neo4j/OpenAI/Prolog/GPU). Cho phép trùng lặp code vì sẽ tách repo.
Dạng output đã chốt: **leaderboard theo metric** (các dạng side-by-side/delta/trend
KHÔNG làm ở bản này — combined CSV để mở đường mở rộng sau).

## Target layout

Thư mục mới ở root repo (tên gợi ý, đổi tự do khi tách): `experiments_repo/`

```
experiments_repo/                 # future standalone repo
├── README.md                     # NEW  — repo này là gì, chạy leaderboard thế nào
├── pyproject.toml                # NEW  — deps tối thiểu (pyyaml); entry `expkit`
├── .gitignore                    # NEW
├── experiments/                  # COPY nguyên từ ./experiments (01..14 + _template + __init__ + results/metrics/report)
├── eval_core/                    # COPY phần metrics-side (BỎ inference.py, multimodel.py, rerender.py)
├── scripts/                      # COPY chỉ exp*_metrics.py + exp*_funnel.py (+ __init__)  ← cho recompute retrieval (optional)
├── src/                          # COPY citations.py, ids.py, legal_metadata.py, (+ schema.py nếu cần), __init__
├── data/                         # COPY legal_sources.yaml, legal_metadata.yaml, eval/questions_200.json
└── expkit/                       # NEW — bộ tổng hợp cross-experiment (phần lõi user cần)
    ├── __init__.py
    ├── loaders.py                # quét experiments/, đọc academic_metrics.json + config.yaml
    ├── families.py               # nhận diện họ theo KEY của JSON + adapter trích metric → rows
    ├── leaderboard.py            # rank rows theo metric đã chọn
    ├── report.py                 # ghi leaderboard .md + combined .csv
    ├── cli.py                    # `python -m expkit leaderboard ...`
    └── __main__.py
```

**Vì sao giữ tên package `eval_core/`, `src/`, `scripts/` y nguyên:** code copy import nhau
qua `from eval_core import paths`, `from src.citations import ...`, `from scripts.exp09_metrics import ...`.
Giữ nguyên tên package ở root repo mới → **không phải sửa import nào**. Trùng tên với repo gốc
là chấp nhận được (repo tách riêng).

## Bundle manifest (copy gì, vì sao)

- **`experiments/` (toàn bộ, ~56MB gồm results/):** nguồn dữ liệu. Leaderboard chỉ đọc
  `metrics/*.json`; results/ giữ lại để mở đường recompute offline (Phase 2) và để repo trọn vẹn.
- **`eval_core/`:** copy `metrics.py, report.py, runners.py, gold.py, experiment.py, paths.py,
  arms.py, text_normalize.py, judge.py, __init__.py`. **Bỏ** `inference.py, multimodel.py, rerender.py`
  (phụ thuộc `runtime/` = inference). `cli.py` → trim còn lệnh `metrics` (bỏ run/multimodel/all),
  hoặc bỏ luôn vì entry chính là `expkit`.
- **`scripts/`:** chỉ copy `exp06_metrics, exp07_metrics, exp08_metrics, exp09_metrics,
  exp10_metrics, exp11_metrics, exp12_metrics, exp13_metrics, exp14_metrics` + các `exp*_funnel.py`
  + `__init__.py`. **Bỏ** mọi `exp*_run.py` (cần runtime) và các script audit/seal/install.
  (Đã xác nhận: `_metrics` chỉ import sibling `_metrics` + `eval_core.gold` + `src.legal_metadata`,
  KHÔNG import `_run`.)
- **`src/`:** `citations.py` (gold/registry), `ids.py`, `legal_metadata.py`, `__init__.py`.
  Thêm `schema.py` chỉ nếu `legal_metadata`/`citations` cần (kiểm tra import lúc làm). **Bỏ**
  `retrieval/`, `bge_m3_loader.py`, `prompts.py` (chỉ phục vụ inference/embedding).
- **`data/`:** `legal_sources.yaml` (registry — gold validate), `legal_metadata.yaml`
  (legal_metadata loader), `eval/questions_200.json` (gold). **Bỏ** `graph/processed` (40MB),
  `ontology/` (10MB), raw/interim (chỉ cần khi chạy lại).

## New code — `expkit` (phần lõi user yêu cầu)

### `loaders.py`
- `discover_experiments(root: Path) -> list[Path]`: liệt kê `experiments/[0-9]*_*` có
  `metrics/academic_metrics.json`. Bỏ qua (warn) folder thiếu metrics (vd 05).
- `load_experiment(path) -> ExperimentMetrics`: đọc `metrics/academic_metrics.json` (json)
  + `config.yaml` (yaml: `name`, `description`, `date`, `parent`). Trả dataclass gồm
  `slug` (tên folder), `name`, `date`, `family`, `raw_metrics`.

### `families.py` — nhận diện theo NỘI DUNG, không theo tên file
- `detect_family(raw: dict) -> "qa"|"retrieval"`: có `aggregates` → qa; có
  `overall_macro`/`stratified`/`Ks` → retrieval; else `unknown` (skip + warn).
- `qa_rows(exp) -> list[Row]`: mỗi arm trong `aggregates`, lấy từ `macro`:
  `citation_recall, citation_precision, citation_f1, citation_display_rate, bertscore_f1, latency_s`
  + `prolog.*rate`, `n_records`. → Row(exp_slug, exp_name, arm, family="qa", metrics={...}).
- `retrieval_rows(exp, stratum) -> list[Row]`: nguồn = `stratified[arm][stratum]` nếu có,
  else `overall_macro[arm]`. Lấy mọi key dạng `recall@k`, `precision@k`, `ndcg@k`, `r_precision`,
  `mrr`, `n`. → Row(exp_slug, exp_name, arm, family="retrieval", stratum, metrics={...}).
- Adapter **chịu lỗi thiếu key**: experiment cũ (06/07) có thể thiếu vài metric → giá trị `None`,
  leaderboard bỏ qua dòng không có metric đang xếp hạng (kèm ghi chú).

### `leaderboard.py`
- `build_leaderboard(rows, metric, descending=True) -> list[RankedRow]`: lọc rows có `metric`,
  sort theo giá trị, gắn `rank` 1..N. Cùng metric ⇒ cùng thang, so sánh xuyên experiment.
- Default metric: QA = `citation_f1`; Retrieval = `recall@12` @ stratum `in_corpus`
  (đúng headline pre-registered của dòng HyDE). Cho phép override qua CLI.

### `report.py`
- `write_leaderboard_md(ranked, family, metric, out)`: bảng Markdown
  `| rank | experiment | arm | <metric> | <cột phụ...> | n |`, kèm header (metric, stratum,
  ngày build, số experiment quét, danh sách bị skip).
- `write_combined_csv(rows, out)`: 1 hàng/(experiment, arm[, stratum]), tất cả metric thành cột
  — nền cho side-by-side/delta/trend về sau.
- Output vào `experiments_repo/reports/leaderboard_<family>.md` + `combined_<family>.csv`.

### `cli.py` / `__main__.py`
```
python -m expkit leaderboard [--root experiments] [--family qa|retrieval|both]
                             [--metric NAME] [--stratum in_corpus]
                             [--exp 08_hyde_retrieval 13_hyde_semantic ... | --all]
                             [--out reports]
```
- `--all` (mặc định) quét toàn bộ; hoặc liệt kê slug để tổng hợp "1 hoặc nhiều experiment".
- `--family both` (mặc định): sinh cả 2 leaderboard (mỗi họ 1 file, vì metric khác thang).
- In bảng ra stdout + ghi file.

## Phasing (giao tăng dần, feature lõi xong trước)

1. **Phase 1 — Leaderboard từ metrics có sẵn (phần user thực sự cần).** Tạo `experiments_repo/`,
   copy `experiments/` + `data/` (3 file) ; viết `expkit/*` ; deps = `pyyaml` (đọc config) + json.
   Chạy được leaderboard 2 họ ngay, **không** cần `eval_core`/`scripts`/`src`.
2. **Phase 2 — Self-containment / recompute offline (optional).** Copy `eval_core` (metrics-side),
   `scripts/exp*_metrics+funnel`, `src` để repo có thể tự dựng lại `metrics/*.json` từ `results/`
   đã commit (không gọi dịch vụ ngoài). `expkit` thêm cờ `--recompute` gọi
   `eval_core.runners.compute_metrics_for_experiment` (QA) / `scripts.exp*_metrics` (retrieval).
3. **Phase 3 — Đóng gói repo.** `pyproject.toml` (entry `expkit = expkit.cli:main`, deps:
   `pyyaml`; extra `recompute = [...]`), `README.md` (mục đích + cách chạy + bảng coverage 2 họ),
   `.gitignore`. Sẵn sàng `git init` khi user tách ra.

## Verification

```powershell
# Phase 1 — leaderboard từ metrics có sẵn
python -m expkit leaderboard --all --family qa --metric citation_f1
python -m expkit leaderboard --all --family retrieval --metric recall@12 --stratum in_corpus
```
- QA: top phải khớp arm có `citation_f1` cao nhất khi đối chiếu thủ công
  `experiments/*/metrics/academic_metrics.json`.
- Retrieval: top khớp `recall@12` cao nhất ở stratum `in_corpus` (vd exp08/13/14).
- 05 bị skip kèm warning; 06/07 thiếu metric mới → dòng bị bỏ qua có ghi chú, không crash.
- Mở `reports/leaderboard_qa.md`, `reports/leaderboard_retrieval.md`, `combined_*.csv` kiểm tra cột.
- (Phase 2) `python -m expkit leaderboard --exp 13_hyde_semantic --recompute` ⇒ regenerate
  `metrics/academic_metrics.json` từ `results/` rồi xếp hạng; số khớp file cũ.
- Thêm `experiments_repo/tests/test_leaderboard.py`: 1 fixture mỗi họ (JSON nhỏ) → kiểm tra
  `detect_family`, thứ tự rank, xử lý thiếu key. Chạy `pytest experiments_repo/tests`.

## Ngoài phạm vi (chốt)
- Không copy `runtime/`, `prompts/`, `data/graph/processed`, `data/ontology`, `exp*_run.py`.
- Không chạy inference; không gọi Neo4j/OpenAI/Prolog/GPU.
- Không sửa repo gốc (chỉ tạo thư mục mới, copy). Bản gốc `experiments/` giữ nguyên cho tới khi
  user tự quyết định xoá sau khi tách.
- Side-by-side / delta-vs-baseline / trend: chưa làm (user chỉ chọn leaderboard); combined CSV
  đã để sẵn nền dữ liệu.
```