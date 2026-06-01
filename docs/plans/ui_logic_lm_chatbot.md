# Plan + Handoff — Streamlit UI: chatbot BHXH (Logic-LM + HyDE-semantic)

> Tài liệu bàn giao để **bất kỳ session nào tiếp tục được**. Mockup xem được ở
> [`ui_logic_lm_chatbot_mockup.html`](ui_logic_lm_chatbot_mockup.html) (mở bằng trình duyệt).
> Plan của backend arm: [`logic_lm_hyde_semantic.md`](logic_lm_hyde_semantic.md).

## Mục tiêu

Chatbot web: người dùng nhập câu hỏi BHXH → hệ thống sinh **câu trả lời + citation**;
với mỗi câu trả lời, **trực quan hóa vì sao** ra kết quả đó theo đúng chuỗi:
**câu hỏi → hypothesis → chương trình Prolog → kết luận**. Tech = **Streamlit**, gọi
**pipeline thật** (arm `logic_lm_hyde_semantic`).

---

## Trạng thái hiện tại (2026-06-01)

### ✅ Backend arm `logic_lm_hyde_semantic` — ĐÃ implement + smoke-verified (CHƯA chạy eval)
Smoke 1 câu qua cả 2 arm (treatment + control), gọi thật Neo4j + BGE-M3 + OpenAI +
SWI-Prolog: pipeline chạy end-to-end, Prolog solve, citation verify được trên Neo4j,
treatment bơm hypothesis vào rule-gen còn control thì không. Eval đầy đủ (B5) CHƯA chạy.
Files đã tạo/sửa (theo `docs/plans/logic_lm_hyde_semantic.md`):
- **Mới**
  - `prompts/runtime/logic_lm/rule_gen_hyde_semantic.md` — rule-gen prompt + input `hypothesis`.
  - `runtime/retrievers/dense_hyde_semantic_logic_adapter.py` — `DenseHydeSemanticAsLogicLMRetriever`
    (build concept-frame → HyDE → dense search → chunks; expose `last_hypothesis`,
    `last_semantic_context`, `verify_citations`).
  - `experiments/02_logic_lm_hyde_semantic/{config.yaml,README.md,.gitignore}`.
- **Sửa (additive / no-op cho 3 arm logic-lm cũ)**
  - `src/retrieval/pipeline.py` — `+ dense_hyde_semantic_rows(question, frame_text, context_key_ids, top_k) -> (rows, docs)`.
  - `runtime/logic_lm_pipelines.py` — `_TokenTrackingLLMClient(hypothesis=...)` chèn hypothesis vào
    rule-gen user-message; base `_rule_gen_hypothesis()` hook; `LogicLMAnswer.hypothesis`;
    2 class mới `LogicLMHydeSemanticPipeline` (treatment) + `LogicLMHydeSemanticNoHypPipeline` (control).
  - `eval_core/arms.py` — `+ "logic_lm_hyde_semantic"`, `+ "logic_lm_hyde_semantic_nohyp"` vào `ALL_ARMS`.
  - `eval_core/inference.py` — 2 runner + `ARM_RUNNERS` + `"hypothesis"` vào record.

### ✅ UI Streamlit — ĐÃ build + verified (serves HTTP 200, render data thật)
- `ui/app.py` — chat + sidebar (arm treatment/control, top_k, toggle reasoning) +
  expander 4 stage. **Share 1 retriever** cho cả 2 arm (`@st.cache_resource` →
  BGE-M3 load đúng 1 lần, vừa GPU 4GB; cô lập đúng biến hypothesis).
- `.streamlit/config.toml` — dark theme khớp mockup. `scripts/ui.ps1` — launcher Windows (UTF-8).
- `experiments/02_logic_lm_hyde_semantic/{config.yaml,README.md,.gitignore}` — folder eval (B5),
  config valid; eval CHƯA chạy (contract validate báo thiếu `metrics/` — đúng chủ đích).

### ➡️ Việc tiếp theo cho session sau
1. (Tách riêng, tốn tiền) chạy eval arm 02: `python -m eval_core all experiments/02_logic_lm_hyde_semantic`
   → `experiment_contract validate` → copy sang `experiments_repo/` → `expkit leaderboard`.
2. (Tùy chọn) pilot `n:8` trước khi chạy full 200 câu.

---

## Data contract — UI tiêu thụ gì

`pipe = LogicLMHydeSemanticPipeline()` rồi `ans = pipe.ask(question)` trả về
`LogicLMAnswer` (`runtime/logic_lm_pipelines.py`):

| Field | Dùng cho stage |
|---|---|
| `ans.plain_answer` / `ans.answer` (IRAC text) | Bong bóng trả lời + Stage 4 |
| `ans.citations[]` (display) / `ans.citation_ids[]` | Citation chips |
| `ans.hypothesis` | **Stage 2** — đoạn HyDE-semantic |
| `ans.prolog_program`, `ans.prolog_trace` | **Stage 3** — code + trace |
| `ans.prolog_status`, `ans.prolog_success`, `ans.n_repair_rounds` | **Stage 3** — badges |
| `ans.irac_sections` `{issue,rule,application,conclusion}` | **Stage 4** |
| `ans.elapsed_s`, `ans.prompt_tokens`, `ans.completion_tokens` | Footer/metadata |

Đọc thêm **ngay sau `ask()`** từ `pipe.retriever` (`DenseHydeSemanticAsLogicLMRetriever`):

| Thuộc tính | Dùng cho |
|---|---|
| `pipe.retriever.last_semantic_context` → `SemanticContext` (`concept_ids`, `kg_entity_ids`, `frame_text`, `laws`, `n_concepts`, `n_kg_entities`) | **Stage 2** — concept frame (chips) |
| `pipe.retriever.last_hypothesis` (== `ans.hypothesis`) | Stage 2 |
| `pipe.retriever.verify_citations(ans.citation_ids)` → `{id: bool}` | Dấu ✓ trên citation |

> ⚠️ `last_*` bị **ghi đè mỗi lần `retrieve()`**. Đọc ngay sau `ask()` và **lưu vào
> `st.session_state.history`** — đừng đọc lại ở lần rerun sau (đã bị câu mới ghi đè).

---

## Tech + bố cục file
- `ui/app.py` — Streamlit app. Chạy: `streamlit run ui/app.py` (thêm `scripts/ui.ps1` cho Windows).
- `@st.cache_resource` nạp pipeline **1 lần** (nặng: Neo4j + BGE-M3 + OpenAI + SWI-Prolog);
  cache theo `arm` để đổi treatment/control.
- `st.session_state.history` = list các turn đã render (question + ans + semantic + verified).
- Đầu file: `load_dotenv` + pop `OPENAI_BASE_URL` nếu rỗng (gotcha #2) + `sys.path` repo root.

## Layout (ASCII — chi tiết xem mockup HTML)
```
┌───────────┬──────────────────────────────────────────────┐
│ SIDEBAR   │  Legal KG — Hỏi đáp BHXH (Logic-LM + HyDE)     │
│ • Arm:    │  ┌── chat ───────────────────────────────────┐ │
│   ▸treat  │  │ 🧑 <câu hỏi>                               │ │
│   ▸ctrl   │  │ 🤖 <plain_answer>                          │ │
│ • top_k 8 │  │    [Điều X kh.Y ✓] [Điều Z ✓]   (citation) │ │
│ • ☑reason │  │    ▼ 🔍 Vì sao có câu trả lời này          │ │
│ • model   │  │       1 Câu hỏi → 2 Hypothesis →           │ │
│           │  │       3 Prolog (code+trace+badge) →        │ │
│           │  │       4 Kết luận (IRAC + plain + cite)      │ │
│           │  └───────────────────────────────────────────┘ │
│           │  [ st.chat_input: "Nhập câu hỏi BHXH..." ]      │
└───────────┴──────────────────────────────────────────────┘
```
4 stage trong expander "🔍 Vì sao có câu trả lời này":
1. **Câu hỏi** — text gốc.
2. **Hypothesis** — chips concept (`frame`: concept_ids + kg_entity_ids), rồi đoạn
   `hypothesis` (blockquote/`st.info`). Caption: "Khung khái niệm BHXH khớp → đoạn văn giả định".
3. **Prolog** — `st.code(prolog_program, language="prolog")`; dưới đó `prolog_trace`;
   badges: `success ✓/✗`, `status=<...>`, `repair=<n>`.
4. **Kết luận** — 4 ô IRAC (issue/rule/application/conclusion) + `plain_answer` + citation chips ✓.

## Spec Streamlit — skeleton (next session ráp gần như copy được)
```python
# ui/app.py
import os, sys
from pathlib import Path
from dotenv import load_dotenv

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO)) if str(_REPO) not in sys.path else None
load_dotenv(_REPO / ".env")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

import streamlit as st
from runtime.logic_lm_pipelines import (
    LogicLMHydeSemanticPipeline, LogicLMHydeSemanticNoHypPipeline,
)

st.set_page_config(page_title="Legal KG — BHXH Q&A", layout="wide")

@st.cache_resource(show_spinner="Nạp pipeline (Neo4j + BGE-M3 + Prolog)...")
def get_pipeline(arm: str):
    cls = LogicLMHydeSemanticPipeline if arm == "treatment" else LogicLMHydeSemanticNoHypPipeline
    return cls()

arm = st.sidebar.radio("Arm", ["treatment", "control"],
                       format_func=lambda a: "Có hypothesis" if a == "treatment" else "Không (control)")
top_k = st.sidebar.slider("top_k", 4, 20, 8)
show_reasoning = st.sidebar.toggle("Hiện reasoning", True)

st.session_state.setdefault("history", [])

def render_turn(t):
    with st.chat_message("user"): st.write(t["question"])
    with st.chat_message("assistant"):
        st.markdown(t["ans"].plain_answer or t["ans"].answer or "_(không kết luận)_")
        if t["ans"].citations:
            st.markdown(" ".join(
                f"`{c}{' ✓' if t['verified'].get(cid) else ''}`"
                for c, cid in zip(t["ans"].citations, t["ans"].citation_ids)))
        if show_reasoning:
            with st.expander("🔍 Vì sao có câu trả lời này"):
                a, ctx = t["ans"], t["semantic"]
                st.markdown("**1 · Câu hỏi**"); st.write(t["question"])
                st.markdown("**2 · Hypothesis**")
                if ctx and (ctx.concept_ids or ctx.kg_entity_ids):
                    st.caption("Khung khái niệm khớp: " + ", ".join(ctx.concept_ids[:8]))
                st.info(a.hypothesis or "_(không có concept khớp → fallback chung)_")
                st.markdown("**3 · Prolog**")
                badge = f"{'✓' if a.prolog_success else '✗'} {a.prolog_status} · repair={a.n_repair_rounds}"
                st.caption(badge)
                st.code(a.prolog_program or "(rỗng)", language="prolog")
                if a.prolog_trace: st.text(a.prolog_trace)
                st.markdown("**4 · Kết luận (IRAC)**")
                for k in ("issue", "rule", "application", "conclusion"):
                    if a.irac_sections.get(k):
                        st.markdown(f"- **{k.capitalize()}:** {a.irac_sections[k]}")

for t in st.session_state.history:
    render_turn(t)

if q := st.chat_input("Nhập câu hỏi BHXH..."):
    pipe = get_pipeline(arm); pipe.top_k = top_k
    with st.spinner("Đang suy luận (HyDE → Prolog → kết luận)..."):
        try:
            ans = pipe.ask(q)
            ctx = getattr(pipe.retriever, "last_semantic_context", None)
            try: verified = pipe.retriever.verify_citations(ans.citation_ids)
            except Exception: verified = {}
        except Exception as e:
            st.error(f"Pipeline lỗi: {type(e).__name__}: {e}"); st.stop()
    st.session_state.history.append(
        {"question": q, "ans": ans, "semantic": ctx, "verified": verified})
    st.rerun()
```

## Verify (chạy được sau khi env sẵn sàng)
```powershell
# 1) Smoke 1 câu (in ra prolog_success / citations / hypothesis≠rỗng)
python -c "from runtime.logic_lm_pipelines import LogicLMHydeSemanticPipeline as P; a=P().ask('Lao động nữ sinh con được nghỉ thai sản bao lâu?'); print(a.prolog_success, a.citation_ids); print(a.hypothesis[:200])"
# 2) UI
streamlit run ui/app.py
```

## Điểm cần quyết khi build
- **Latency** vài giây/câu (HyDE + rule-gen + repair + IRAC) → luôn để `st.spinner`.
- **Lỗi hạ tầng** (Neo4j down / API lỗi) → `st.error`, không để app crash; nhớ gotcha #3
  (API error có thể biến thành `unable_to_conclude`, soi `prompt_tokens==0`).
- **Đổi arm** treatment/control → 2 pipeline cache riêng (mỗi cái 1 Neo4j driver).
- **Đa luật**: `document` của chunk lấy theo `pipe._law_display(law_id)` (đã xử lý trong adapter).
- Không hiển thị số liệu eval ở UI (Rule 6 — UI là demo cơ chế, không phải nơi báo cáo kết quả).
