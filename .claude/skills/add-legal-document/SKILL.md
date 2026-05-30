---
name: add-legal-document
description: Add a new Vietnamese legal document (Luật / Bộ luật / Nghị định / Quyết định / Thông tư / Công văn / Hiệp định / Pháp lệnh) to the legal KG, wiring it end-to-end through the offline pipeline (B1→B6) into Neo4j. Trigger phrases include "thêm luật mới", "add luật", "thêm văn bản pháp luật", "load Nghị định/Thông tư/Quyết định", "ingest văn bản vào graph", "load văn bản vào KG", "thêm vào Neo4j", or when the user mentions a .docx in data/graph/raw/ that needs registration. Covers both registries (legal_metadata.yaml + legal_sources.yaml), all conditional flags (allow_no_chapter, llm_skip_articles), and verification at each phase. Use whenever a new legal document needs to be made retrievable by GraphRAG / Logic-LM arms.
---

# Skill — Thêm văn bản pháp luật mới vào KG

Tự động hoá quy trình load 1 văn bản pháp luật Việt Nam vào Knowledge Graph hiện hữu. Bám đúng pipeline B1→B6 trong `offline/` + 2 registry YAML + schema Neo4j hiện tại. Không bịa, không hardcode, không tự ý đụng code.

---

## 1. When to invoke

Kích hoạt skill này khi user yêu cầu:
- "thêm 1 luật/văn bản mới", "add luật", "ingest Nghị định/Quyết định/Thông tư"
- "load `<filename>.docx` vào graph"
- "thêm ND143_2018", "thêm QD366_BHXH" (canonical slug)
- "load nốt các file trong `data/graph/raw/` chưa đăng ký"

Khi user chỉ cấp filename hoặc canonical slug mà thiếu metadata, **dùng `AskUserQuestion`** để lấy đủ thông tin bắt buộc (xem §3) trước khi bắt đầu — KHÔNG đoán.

---

## 2. Inviolable rules

| # | Rule | Lý do |
|---|---|---|
| 1 | **Không** sửa code `offline/`, `src/`, `schema/` để hỗ trợ riêng văn bản này. Mọi behavior riêng biệt phải data-driven qua YAML. | Hardcode-creep phá multi-law support đã refactor. |
| 2 | Đăng ký vào **CẢ HAI** YAML: `data/legal_metadata.yaml` (pipeline) **và** `data/legal_sources.yaml` (citation). | Một registry không kích hoạt → citation/retrieval không nhận diện. |
| 3 | Slug canonical phải **trùng** giữa hai YAML. | `parse_id` + `citations._INTERNAL_ID_RE` đều dùng prefix làm khoá. |
| 4 | Validate `expected.{chapters, sections, articles}` đúng số thực tế **trước** khi commit YAML. | B1 fail-hard nếu sai. |
| 5 | Không thêm file vào `data/eval/`, `data/ontology/`, hay bất kỳ thư mục input nào ngoài `data/graph/raw/`. | Vi phạm Rule-1 của `legal-kg-logic-extraction` skill. |
| 6 | **Pause trước B3** (LLM extract): có cost OpenAI; hỏi user xác nhận. | Tránh chi phí ngoài ý muốn. |
| 7 | **Pause trước B6** (load Neo4j) nếu user không có sẵn `.env` Neo4j: yêu cầu xác nhận URI. | Tránh load nhầm vào DB sai. |
| 8 | **Không** chạy `--reset` hay `git reset --hard` để "fix" lỗi. Debug root cause + sửa YAML. | Mất dữ liệu/work khác. |
| 9 | Mọi commit phải có message "feat(kg): add <KEY> — <canonical_title>" và **chỉ** include 2 YAML + file `.docx` (không commit interim/processed). | Giữ history sạch. |

---

## 3. Information to gather (bắt buộc trước khi bắt đầu)

Nếu user không cấp đủ, dùng `AskUserQuestion` để lấy:

| Field | Mô tả | Ví dụ | Bắt buộc? |
|---|---|---|---|
| `source_file` | Path tới `.docx`/`.doc` trong `data/graph/raw/` | `data/graph/raw/143_2018_ND-CP_346012.docx` | ✅ |
| `KEY` (slug canonical) | ID trong YAML — phải uppercase, [A-Z][A-Z0-9_]* | `ND143_2018`, `QD366_BHXH`, `TT18_2022_BYT`, `L41_2024` | ✅ |
| `full_id` | Mã văn bản gốc | `143/2018/NĐ-CP`, `366/QĐ-BHXH`, `41/2024/QH15` | ✅ |
| `title` | Tiêu đề ngắn | `Nghị định quy định Luật BHXH bắt buộc đối với NLĐ là công dân nước ngoài` | ✅ |
| `canonical_title` | Tiêu đề chuẩn dùng cho citation | `Nghị định 143/2018/NĐ-CP` | ✅ |
| `type` | `law` / `code` / `decree` / `decision` / `circular` / `official_letter` / `agreement` / `ordinance` | `decree` | ✅ |
| `hierarchy_level` | `luật`, `bộ luật`, `nghị định`, `quyết định`, `thông tư`, … | `nghị định` | ✅ |
| `priority` | int, 100=luật, 80=NĐ, 70=QĐ, 60=TT (heuristic — hỏi user nếu không chắc) | `80` | ✅ |
| `issuer` | Cơ quan ban hành | `Chính phủ`, `Quốc hội`, `Bảo hiểm xã hội Việt Nam` | ✅ |
| `issued_date` | YYYY-MM-DD | `2018-10-15` | ✅ |
| `effective_date` | YYYY-MM-DD | `2018-12-01` | ✅ |
| `repealed_date` | YYYY-MM-DD hoặc rỗng | (rỗng nếu còn hiệu lực) | ⚪ |
| `aliases` | Danh sách string user có thể dùng để gọi tên | `["Nghị định 143/2018/NĐ-CP", "143/2018/NĐ-CP"]` | ✅ |
| `prolog_law_ids` | Tên atom Prolog (cho logic-LM, có thể bỏ trống) | `["law_nd_143_2018"]` | ⚪ |
| `repeals` | Danh sách KEY luật cũ bị thay thế | `["ND_X_OLD"]` | ⚪ |
| `allow_no_chapter` | `true` nếu document không có "Chương" (NĐ/QĐ/TT thường) | `true` cho NĐ/QĐ | ⚪ |
| `llm_skip_articles` | Danh sách số Điều B3 phải skip (vd Điều chuyển tiếp) | `[]` mặc định | ⚪ |
| `llm_skip_reason` | Lý do skip (text tự do) | `""` mặc định | ⚪ |

**Heuristic cho `KEY` slug khi user không cấp**:
- Luật QH `XX/YYYY/QH<n>` → `L<XX>_<YYYY>` (vd `L41_2024`)
- Bộ luật QH → tương tự `L<XX>_<YYYY>`
- Nghị định `XX/YYYY/NĐ-CP` → `ND<XX>_<YYYY>` (vd `ND143_2018`)
- Quyết định `XX/QĐ-<authority>` → `QD<XX>_<authority>` (vd `QD366_BHXH`)
- Thông tư `XX/YYYY/TT-<authority>` → `TT<XX>_<YYYY>_<authority>` (vd `TT18_2022_BYT`)
- Công văn `XX/<authority>-<sub>` → `CV<XX>_<authority>_<sub>` (vd `CV2068_BYT_BH`)

Cross-check với `data/legal_sources.yaml` xem KEY đã tồn tại chưa. Nếu rồi, dùng đúng KEY đó (đừng tạo trùng).

---

## 4. Workflow — execute phases sequentially

### Phase A — Inspect document (NO write yet)

```powershell
# Xem 50 paragraph đầu để xác định cấu trúc Chương/Mục/Điều
python -c "from docx import Document; d=Document('data/graph/raw/<file>.docx'); [print(p.text) for p in d.paragraphs[:80] if p.text.strip()]"
```

**Mục tiêu**:
- Document có dòng "Chương I/II/..." không? → quyết định `allow_no_chapter`.
- Đếm chính xác `expected.chapters`, `expected.sections`, `expected.articles`.
- Đảm bảo Điều numbering liên tiếp 1..N (B1 sẽ assert).

**Đếm nhanh**:
```powershell
python -c @'
from docx import Document
import re
d = Document('data/graph/raw/<file>.docx')
texts = [p.text.strip() for p in d.paragraphs if p.text.strip()]
n_ch = sum(1 for t in texts if re.match(r'^Chương\s+[IVXLCDM]+', t, re.IGNORECASE))
n_sec = sum(1 for t in texts if re.match(r'^Mục\s+\d+', t, re.IGNORECASE))
arts = [int(m.group(1)) for t in texts if (m := re.match(r'^Điều\s+(\d+)\.', t))]
print(f"chapters={n_ch} sections={n_sec} articles={len(arts)} (max={max(arts) if arts else 0})")
print(f"liên tiếp 1..N: {arts == list(range(1, len(arts)+1))}")
'@
```

Báo cáo kết quả cho user — nếu user đã cấp `expected` mà số đếm khác → **dừng**, hỏi user.

### Phase B — Thêm entry vào 2 YAML

**B.1) `data/legal_metadata.yaml`** — thêm vào block `laws:` (đặt theo thứ tự alphabetical hoặc theo type):

```yaml
laws:
  ...existing entries...
  <KEY>:
    id: <KEY>
    code: <KEY>
    full_id: "<full_id>"
    title: "<title>"
    canonical_title: "<canonical_title>"
    type: <type>
    hierarchy_level: "<hierarchy_level>"
    priority: <priority>
    issuer: "<issuer>"
    issued_date: <issued_date>
    effective_date: <effective_date>
    repealed_date:                       # bỏ trống nếu còn hiệu lực
    source_file: "data/graph/raw/<file>.docx"
    expected:
      chapters: <n>                      # số thực tế từ Phase A
      sections: <n>
      articles: <n>
    allow_no_chapter: <true|false>       # bỏ field nếu false (mặc định)
    llm_skip_articles: []                # bỏ field nếu không skip
    aliases:
      - "<alias 1>"
      - "<alias 2>"
    prolog_law_ids:
      - <atom 1>                         # tuỳ chọn

load_order:
  - L45_2019
  - L58_2014
  - L41_2024
  - <KEY>                                # ← THÊM VÀO ĐÂY (thường append cuối)
```

⚠️ **Vị trí trong `load_order`** ảnh hưởng thứ tự B4 build graph. Append cuối là an toàn nhất; chỉ đặt khác nếu user có `repeals` chain.

**B.2) `data/legal_sources.yaml`** — thêm vào block `sources:`:

```yaml
sources:
  ...existing entries...
  <KEY>:
    canonical_title: "<canonical_title>"
    type: <type>
    aliases:
      - "<alias 1>"
      - "<alias 2>"
    prolog_law_ids:
      - <atom 1>                         # tuỳ chọn — chỉ thêm nếu logic-LM arms cần
```

**Quan trọng**: `<KEY>`, `canonical_title`, `aliases`, `prolog_law_ids` phải **giống** giữa 2 YAML. Nếu khác → citation parser không match.

Sau khi sửa: chạy `python -c "from src.legal_metadata import metadata_for; m = metadata_for('<KEY>'); print(m)"` để verify YAML parse OK.

### Phase C — B1 parse_docx

```powershell
python -m offline.parse_docx --law <KEY>
```

**Expected output**:
- `Đọc data/graph/raw/<file>.docx cho <KEY>...`
- `OK chapters=N sections=N articles=N clauses=N points=N`
- `Saved: data/graph/interim/structured_law_<KEY>.json`

**Failure modes**:
| Triệu chứng | Nguyên nhân | Fix |
|---|---|---|
| `Article number không liên tiếp 1..N` | Document có Điều bổ sung (5a, 64a) hoặc nhảy số | Báo user — out of scope hiện tại của parser |
| `Chương <X> thiếu title` | Title rỗng vì `_finalize_title` chưa được flush | Kiểm tra paragraph "Chương I" có dòng kế tiếp là title không |
| `số Chương sai: mong đợi X, thực tế Y` | YAML `expected.chapters` sai | Sửa YAML, không sửa code |
| `Điều X text rỗng` | Điều không có nội dung dưới header | Inspect document, có thể document hỏng |
| Mọi Điều bị bỏ qua, chapters=0 | Document không có "Chương" nhưng `allow_no_chapter: false` | Set `allow_no_chapter: true` trong YAML |

### Phase D — B2 rule_extract

```powershell
python -m offline.rule_extract --input data/graph/interim/structured_law_<KEY>.json --law <KEY>
```

**Expected**: 4 file output `{internal_refs,external_refs,definitions,amendments}_<KEY>.json`.

**Lưu ý**:
- Internal refs trong NĐ/QĐ thường rất ít vì các file dưới luật chủ yếu cite Luật mẹ (external).
- Nếu external_refs đếm = 0 → kiểm tra regex `RE_EXT_TYPED` ở [rule_extract.py:37-44](offline/rule_extract.py#L37) có cover doc_type của file này không.

### Phase E — B3 LLM extract (⚠️ COST — pause for confirmation)

**Tính cost ước trước**:
- `articles` từ Phase A × ~$0.005 (gpt-4o-mini ~2.5K input + 1K output trung bình/Điều)
- Vd 50 Điều ≈ $0.25; 141 Điều ≈ $0.70

**Hỏi user**: "B3 sẽ gọi OpenAI ~N Articles, ước cost $X. Tiếp tục không?" qua `AskUserQuestion`.

**Nếu user OK**:
```powershell
python -m offline.llm_extract --law <KEY>
```
Output: `data/graph/interim/llm_extractions/<KEY>/A{n}.json` (1 file/Article, idempotent).

**Nếu user skip**: B4 vẫn build graph được — chỉ structural + reference edges, không có semantic entities/edges từ LLM. Ghi nhận trong commit message.

### Phase F — B4 merge_normalize

```powershell
python -m offline.merge_normalize
```

**Expected**: `data/graph/processed/merged_graph.json` + `extraction_summary.md`.

**Failure modes**:
- `Phải có N Law node, có M` → YAML `load_order` chưa thêm `<KEY>` hoặc thiếu file interim.
- `<Label>/<id> thiếu mentioned_in` → bug B3, kiểm tra `llm_extractions/<KEY>/`.
- `Edge T có source_clause không tồn tại: <sc>` → bug B2 sinh ref tới Clause không có thực — kiểm tra `internal_refs_<KEY>.json`.

### Phase G — B5 embed

```powershell
python -m offline.embed --force
```

**Expected**: `data/graph/processed/embeddings.parquet` — Article/Clause/Point của `<KEY>` đã được encode chung với các luật cũ.

**Sanity**:
- Tổng unit = `articles + clauses + points` của tất cả luật trong KG.
- Spot-check 1 vector của `<KEY>.A1`: `python -c "import pandas as pd; df = pd.read_parquet('data/graph/processed/embeddings.parquet'); print(df[df['id']=='<KEY>.A1'])"`.

### Phase H — B6 load_neo4j

⚠️ Pause: confirm `.env` Neo4j URI/credentials đúng với DB target. Nếu user không có sẵn → dừng, hỏi.

```powershell
python -m offline.load_neo4j --apply-schema
```

**KHÔNG dùng `--reset`** — sẽ xoá toàn bộ KG. MERGE-by-id đã idempotent.

**Expected output**:
- `=== APPLY SCHEMA ===` OK + already-exists.
- `=== LOAD NODES ===` — số node của `<KEY>` thêm vào counts.
- `=== LOAD EDGES ===`.
- `=== LOAD EMBEDDINGS ===`.
- `=== SANITY QUERIES ===`: Counts, vector search A64, provenance roundtrip, reachability.

### Phase I — Verify in Neo4j

```cypher
// 1. Đếm node của luật mới
MATCH (n {law_code: '<KEY>'}) RETURN labels(n)[0] AS label, count(n) AS c ORDER BY c DESC;

// 2. Reachable từ Law
MATCH (l:Law {id: '<KEY>'})-[:HAS_CHAPTER]->(ch:Chapter)-[:HAS_ARTICLE]->(a:Article)-[:HAS_CLAUSE]->(c:Clause)
RETURN count(DISTINCT c) AS clauses;

// 3. Vector search smoke test (1 article của KEY mới)
MATCH (a:Article {id: '<KEY>.A1'})
CALL db.index.vector.queryNodes('article_vec', 5, a.embedding) YIELD node, score
RETURN node.id, node.title, score ORDER BY score DESC;

// 4. External ref đã resolve sang KG nếu có cross-law
MATCH ()-[r:REFERS_TO]->(a:Article {law_code: '<KEY>'}) RETURN count(r);
```

Báo cáo kết quả cho user. Nếu Phase I OK → done.

---

## 5. Commit hygiene

Sau khi tất cả phase OK:

```powershell
git status
# Expected staged:
#  - data/legal_metadata.yaml
#  - data/legal_sources.yaml
#  - data/graph/raw/<file>.docx  (untracked, cần add)
# KHÔNG commit:
#  - data/graph/interim/*  (gitignored)
#  - data/graph/processed/* (vẫn cần check theo policy repo)

git add data/legal_metadata.yaml data/legal_sources.yaml data/graph/raw/<file>.docx
git commit -m "feat(kg): add <KEY> — <canonical_title>

- Register in legal_metadata.yaml (load_order updated)
- Register in legal_sources.yaml (citation registry)
- B1-B6 verified: <n> articles, <n> clauses loaded into Neo4j
- allow_no_chapter=<bool>, llm_skip_articles=<list>

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

KHÔNG push trừ khi user yêu cầu rõ.

---

## 6. Code references (single source of truth)

Khi cần verify hành vi pipeline:

| Phase | File | Key entry point |
|---|---|---|
| B1 | [offline/parse_docx.py](offline/parse_docx.py) | `parse_docx(path, law_code, metadata)`, `validate_against_metadata` |
| B1 (no-chapter mode) | [offline/parse_docx.py](offline/parse_docx.py) — `_ensure_chapter` | Synth Chương I khi `meta.allow_no_chapter=True` |
| B2 | [offline/rule_extract.py](offline/rule_extract.py) | `extract_all(structured, law)`, `_verify_provenance` |
| B3 | [offline/llm_extract.py](offline/llm_extract.py) | `run(meta, ...)`, `structured_path_for`, `out_dir_for`, `meta.llm_skip_articles` |
| B4 | [offline/merge_normalize.py](offline/merge_normalize.py) | `load_all`, `_llm_files_for(law_id)`, `validate(g, law_inputs)` |
| B5 | [offline/embed.py](offline/embed.py) | `collect_units`, `encode_all` |
| B6 | [offline/load_neo4j.py](offline/load_neo4j.py) | `apply_schema`, `load_nodes`, `load_edges`, `load_embeddings`, `sanity` |
| Schema | [schema/schema.cypher](schema/schema.cypher) | Constraints + vector indexes (idempotent) |
| ID convention | [src/ids.py](src/ids.py) | `parse_id` permissive (mọi `[A-Z][A-Z0-9_]*` prefix), `law_id` canonical pass-through |
| Metadata loader | [src/legal_metadata.py](src/legal_metadata.py) | `metadata_for`, `LawMetadata` dataclass |
| Citation registry | [src/citations.py](src/citations.py) | `load_registry`, `parse_internal_citation_id`, `format_citation` |

---

## 7. Files NOT to touch

| File | Lý do |
|---|---|
| Bất kỳ file nào trong `offline/`, `src/`, `runtime/`, `eval_core/`, `schema/` | Behavior phải data-driven qua YAML. Sửa code = vi phạm. |
| `tests/*.py` | Không weaken test để pass — sửa root cause. |
| `experiments/*/results/` | Frozen results, không đụng. |
| `data/eval/questions_*.json` | Eval input, không phải KG input. |
| `data/ontology/` | Logic-LM ontology, không liên quan adding-law. |

Nếu phát hiện cần sửa code (bug B1 chưa cover edge case của document mới): **dừng**, báo user, gợi ý PR riêng — không tự sửa trong cùng luồng add-law.

---

## 8. Reusable commands cheat sheet

```powershell
# Inspect document structure
python -c "from docx import Document; d=Document('<path>'); [print(p.text) for p in d.paragraphs[:80] if p.text.strip()]"

# Count chapters/sections/articles
python -c "from docx import Document; import re; d=Document('<path>'); texts=[p.text.strip() for p in d.paragraphs if p.text.strip()]; ch=sum(1 for t in texts if re.match(r'^Chương\s+[IVXLCDM]+', t, re.I)); sec=sum(1 for t in texts if re.match(r'^Mục\s+\d+', t, re.I)); arts=[int(m.group(1)) for t in texts if (m:=re.match(r'^Điều\s+(\d+)\.', t))]; print(f'ch={ch} sec={sec} art={len(arts)} max={max(arts) if arts else 0}')"

# Verify YAML parses
python -c "from src.legal_metadata import metadata_for; print(metadata_for('<KEY>'))"

# Pipeline (per-law)
python -m offline.parse_docx --law <KEY>
python -m offline.rule_extract --input data/graph/interim/structured_law_<KEY>.json --law <KEY>
python -m offline.llm_extract --law <KEY>            # optional, costs $
python -m offline.merge_normalize
python -m offline.embed --force
python -m offline.load_neo4j --apply-schema

# Or batch all laws in load_order
python -m offline.parse_docx --all
python -m offline.rule_extract --all
python -m offline.llm_extract --all                  # optional, costs $
python -m offline.merge_normalize
python -m offline.embed --force
python -m offline.load_neo4j --apply-schema
```

---

## 9. Quick mental model — when in doubt

```
Document.docx
   ↓
[legal_metadata.yaml + legal_sources.yaml]  ← user-driven config (data-driven)
   ↓
B1 parse_docx        — Chương → Điều → Khoản → Điểm tree (Pydantic-validated)
   ↓
B2 rule_extract      — internal/external refs, definitions, amendments (regex, byte-verified)
   ↓
B3 llm_extract       — semantic entities/edges (OpenAI, per-Article, provenance-validated)
   ↓
B4 merge_normalize   — dedup, validate, gom 1 file merged_graph.json
   ↓
B5 embed             — BGE-M3 1024-d cho Article/Clause/Point
   ↓
B6 load_neo4j        — MERGE-by-id idempotent, load vectors, sanity
```

Mỗi phase fail-hard, idempotent (re-run an toàn). Nếu Phase X fail, sửa root cause + re-run Phase X — không skip.

---

## 10. Done criteria

Skill này hoàn thành nhiệm vụ khi:

- [x] 2 YAML đã có entry `<KEY>` với mọi field bắt buộc đúng.
- [x] `load_order` đã có `<KEY>`.
- [x] B1 pass với `expected.{chapters,sections,articles}` khớp.
- [x] B2 4 file interim của `<KEY>` đã sinh.
- [x] B3 đã chạy (hoặc user explicit skip).
- [x] B4 `merge_normalize.py` validate OK, không lỗi.
- [x] B5 `embeddings.parquet` chứa node của `<KEY>`.
- [x] B6 sanity queries pass trong Neo4j (counts + vector search + reachability).
- [x] Cypher query Phase I trả số node > 0 cho luật mới.
- [x] Git commit message rõ ràng, không leak file gitignored.

Báo cáo cuối cho user: bảng tổng kết (số Article/Clause/Point đã load, có external refs hay không, có semantic edges từ B3 không) + 1 ví dụ Cypher query để user tự verify.
