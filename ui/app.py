"""Streamlit chatbot — BHXH Q&A trên arm `logic_lm_hyde_semantic`.

Gọi pipeline THẬT (`runtime.logic_lm_pipelines`): dense_hyde_semantic retrieval →
Logic-LM sinh Prolog (SWI-Prolog) → IRAC + plain answer. Mỗi câu trả lời được
trực quan hoá theo chuỗi suy luận: câu hỏi → hypothesis → chương trình Prolog →
kết luận. Không hiển thị số liệu eval (Rule 6 — UI demo cơ chế, không báo cáo kết quả).

Chạy:  streamlit run ui/app.py   (hoặc scripts/ui.ps1 trên Windows)

Cần sẵn sàng: Neo4j + BGE-M3 (GPU/CPU) + OPENAI_API_KEY + SWI-Prolog (swipl trên PATH).
"""
from __future__ import annotations

import html
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# --- repo root + env (gotcha #2: OPENAI_BASE_URL rỗng → SDK dùng "" làm URL) ---
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
load_dotenv(_REPO / ".env")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import streamlit as st

from runtime.logic_lm_pipelines import (
    LogicLMHydeSemanticNoHypPipeline,
    LogicLMHydeSemanticPipeline,
)
from runtime.retrievers.dense_hyde_semantic_logic_adapter import (
    DenseHydeSemanticAsLogicLMRetriever,
)

st.set_page_config(
    page_title="Legal KG — Hỏi đáp BHXH",
    page_icon="⚖️",
    layout="wide",
)

# Custom styling — bám palette của mockup (chip / badge / khung hypothesis).
st.markdown(
    """
    <style>
      .cite-chip{display:inline-block;background:#27313d;border:1px solid #2d3845;
                 border-radius:14px;padding:3px 11px;font-size:12.5px;margin:3px 6px 3px 0}
      .cite-ok{color:#3fb950;font-weight:700}
      .cite-no{color:#d29922;font-weight:700}
      .concept-chip{display:inline-block;background:#27313d;border:1px solid #2d3845;
                    border-radius:13px;padding:3px 10px;font-size:12px;color:#cfe0ff;margin:2px 5px 2px 0}
      .stage-badge{display:inline-block;font-size:12px;padding:3px 9px;border-radius:6px;
                   border:1px solid #2d3845;margin:0 6px 0 0}
      .b-ok{background:#0f2a17;color:#3fb950;border-color:#1d4a2b}
      .b-no{background:#2a1717;color:#f85149;border-color:#4a1d1d}
      .b-muted{background:#27313d;color:#9aa7b4}
      .hypo-box{background:#10202f;border-left:3px solid #4f9cf9;border-radius:0 8px 8px 0;
                padding:11px 14px;font-size:13.5px;line-height:1.55;color:#d7e6f5;margin:6px 0}
      .hypo-box.empty{border-left-color:#33405a;background:#161d27;color:#9aa7b4;font-style:italic}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Resource loading — nặng (Neo4j + BGE-M3 + SWI-Prolog), cache 1 lần.
# Cả 2 arm chia sẻ MỘT retriever (cùng dense_hyde_semantic retrieval + cache) →
# chỉ load BGE-M3 một lần, và cô lập đúng một biến: có/không bơm hypothesis vào
# rule-gen. Khác biệt arm nằm hoàn toàn ở lớp Logic-LM, không ở retrieval.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Nạp retriever (Neo4j + BGE-M3)…")
def get_retriever() -> DenseHydeSemanticAsLogicLMRetriever:
    return DenseHydeSemanticAsLogicLMRetriever()


@st.cache_resource(show_spinner="Khởi tạo pipeline Logic-LM…")
def get_pipeline(arm: str):
    retriever = get_retriever()
    cls = (
        LogicLMHydeSemanticPipeline
        if arm == "treatment"
        else LogicLMHydeSemanticNoHypPipeline
    )
    return cls(retriever=retriever)


# ---------------------------------------------------------------------------
# Sidebar — cài đặt
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    arm = st.radio(
        "Arm sinh câu trả lời",
        ["treatment", "control"],
        format_func=lambda a: (
            "Có hypothesis (treatment)" if a == "treatment" else "Không hypothesis (control)"
        ),
        help=(
            "treatment: bơm đoạn hypothesis (HyDE-semantic) vào bước sinh Prolog. "
            "control: cùng retrieval nhưng KHÔNG bơm hypothesis. Retrieval & citation "
            "giống hệt nhau ở cả hai."
        ),
    )
    top_k = st.slider("top_k (số khoản retrieve)", 4, 20, 8)
    show_reasoning = st.toggle("Hiện reasoning", value=True)
    st.caption(f"Model: `{os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')}`")
    st.divider()
    st.caption(
        "Pipeline: **dense_hyde_semantic → Logic-LM (SWI-Prolog) → IRAC**.\n\n"
        "Cần Neo4j + BGE-M3 + OpenAI + SWI-Prolog."
    )
    if st.button("🗑️ Xoá hội thoại", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚖️ Legal KG — Hỏi đáp BHXH")
st.caption("Logic-LM + HyDE-semantic · trực quan hoá: câu hỏi → hypothesis → Prolog → kết luận")

st.session_state.setdefault("history", [])


# ---------------------------------------------------------------------------
# Render một turn (đọc dữ liệu đã lưu trong history — KHÔNG đọc lại retriever.last_*)
# ---------------------------------------------------------------------------


def _citation_chips(ans, verified: dict) -> str:
    chips = []
    for disp, cid in zip(ans.citations, ans.citation_ids):
        ok = verified.get(cid)
        mark = (
            ' <span class="cite-ok">✓</span>'
            if ok
            else (' <span class="cite-no">?</span>' if verified else "")
        )
        chips.append(f'<span class="cite-chip">{html.escape(str(disp))}{mark}</span>')
    return "".join(chips)


def render_turn(turn: dict) -> None:
    ans = turn["ans"]
    ctx = turn["semantic"]
    verified = turn["verified"]

    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(ans.plain_answer or ans.answer or "_(không đưa ra kết luận)_")
        if ans.citations:
            st.markdown(_citation_chips(ans, verified), unsafe_allow_html=True)

        if not show_reasoning:
            return

        with st.expander("🔍 Vì sao có câu trả lời này", expanded=False):
            # 1 · Câu hỏi
            st.markdown("**1 · Câu hỏi**")
            st.markdown(
                f'<div class="hypo-box empty">{html.escape(turn["question"])}</div>',
                unsafe_allow_html=True,
            )

            # 2 · Hypothesis (khung khái niệm + đoạn văn giả định)
            st.markdown("**2 · Hypothesis**")
            st.caption(
                "Khung khái niệm BHXH khớp từ câu hỏi → đoạn “văn bản luật giả định” "
                "(định hướng sinh Prolog; không chứa số Điều/Khoản)."
            )
            if ctx is not None and (ctx.concept_ids or ctx.kg_entity_ids):
                names = list(ctx.concept_ids) + [
                    e.split(":", 1)[-1].replace("-", " ") for e in ctx.kg_entity_ids
                ]
                st.markdown(
                    "".join(
                        f'<span class="concept-chip">{html.escape(n)}</span>'
                        for n in names[:10]
                    ),
                    unsafe_allow_html=True,
                )
            if ans.hypothesis:
                st.markdown(
                    f'<div class="hypo-box">{html.escape(ans.hypothesis)}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="hypo-box empty">(arm control — không bơm hypothesis '
                    "vào bước sinh Prolog)</div>",
                    unsafe_allow_html=True,
                )

            # 3 · Chương trình Prolog
            st.markdown("**3 · Chương trình Prolog**")
            ok_cls = "b-ok" if ans.prolog_success else "b-no"
            ok_txt = "prolog_success ✓" if ans.prolog_success else "prolog_success ✗"
            st.markdown(
                f'<span class="stage-badge {ok_cls}">{ok_txt}</span>'
                f'<span class="stage-badge b-muted">status = {html.escape(ans.prolog_status or "—")}</span>'
                f'<span class="stage-badge b-muted">repair_rounds = {ans.n_repair_rounds}</span>',
                unsafe_allow_html=True,
            )
            st.code(ans.prolog_program or "(không sinh được chương trình)", language="prolog")
            if ans.prolog_trace:
                st.caption("Trace thực thi (SWI-Prolog):")
                st.code(ans.prolog_trace, language="prolog")
            elif ans.prolog_error:
                st.caption("Lỗi solver:")
                st.code(ans.prolog_error[:800], language="text")

            # 4 · Kết luận (IRAC)
            st.markdown("**4 · Kết luận (IRAC)**")
            irac = ans.irac_sections or {}
            labels = {
                "issue": "Issue",
                "rule": "Rule",
                "application": "Application",
                "conclusion": "Conclusion",
            }
            if any(irac.get(k) for k in labels):
                for k, lab in labels.items():
                    if irac.get(k):
                        st.markdown(f"- **{lab}:** {irac[k]}")
            else:
                st.markdown("_(không tách được mục IRAC)_")
            st.caption(
                f"⏱ {ans.elapsed_s:.1f}s · tokens {ans.prompt_tokens}+{ans.completion_tokens} "
                f"· n_repair {ans.n_repair_rounds} · arm `{turn['arm']}`"
            )


for turn in st.session_state.history:
    render_turn(turn)


# ---------------------------------------------------------------------------
# Input + suy luận
# ---------------------------------------------------------------------------

if q := st.chat_input("Nhập câu hỏi BHXH…"):
    try:
        pipe = get_pipeline(arm)
    except Exception as e:  # hạ tầng (Neo4j/model) hỏng → báo lỗi, không crash app
        st.error(f"Không khởi tạo được pipeline: {type(e).__name__}: {e}")
        st.stop()

    pipe.top_k = top_k
    with st.spinner("Đang suy luận (HyDE → Prolog → kết luận)…"):
        try:
            ans = pipe.ask(q)
            # Đọc NGAY sau ask() rồi lưu — last_* bị ghi đè ở câu hỏi kế tiếp.
            ctx = getattr(pipe.retriever, "last_semantic_context", None)
            try:
                verified = pipe.retriever.verify_citations(ans.citation_ids)
            except Exception:
                verified = {}
        except Exception as e:
            st.error(f"Pipeline lỗi: {type(e).__name__}: {e}")
            st.stop()

    if ans.error:
        st.error(f"Pipeline trả về lỗi: {ans.error}")

    st.session_state.history.append(
        {"question": q, "ans": ans, "semantic": ctx, "verified": verified, "arm": arm}
    )
    st.rerun()
