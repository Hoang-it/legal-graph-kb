"""Streamlit chatbot — BHXH Q&A trên arm `logic_lm_hyde_semantic`.

Gọi pipeline THẬT (`runtime.logic_lm_pipelines`): dense_hyde_semantic retrieval →
Logic-LM sinh Prolog (SWI-Prolog) → IRAC + plain answer. Hệ thống luôn dùng
hypothesis (HyDE-semantic). Mỗi câu trả lời được trực quan hoá theo chuỗi suy
luận live (Hiểu → Giả thuyết → Tìm kiếm → Suy luận → Kết luận). Mục "Suy luận"
hiển thị dạng dễ đọc (I-R-A của IRAC) hoặc raw Prolog. Mô hình + retriever được
nạp sẵn lúc khởi động. Không hiển thị số liệu eval (Rule 6 — UI demo cơ chế).

Chạy:  streamlit run ui/app.py   (hoặc scripts/ui.ps1 trên Windows)

Cần sẵn sàng: Neo4j + BGE-M3 (GPU/CPU) + OPENAI_API_KEY + SWI-Prolog (swipl trên PATH).
"""
from __future__ import annotations

import html
import os
import sys
import uuid
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

from runtime.logic_lm_pipelines import LogicLMHydeSemanticPipeline
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
# Cấu hình model + nhãn các bước reasoning
# ---------------------------------------------------------------------------

# Một model GPT duy nhất, dùng chung cho cả 3 lệnh LLM của arm (sinh giả thuyết
# HyDE + sinh Prolog + render IRAC).
DEFAULT_MODEL = (os.environ.get("OPENAI_MODEL") or "gpt-4o-mini").strip()
MODEL_OPTIONS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1", "gpt-5-mini"]
if DEFAULT_MODEL not in MODEL_OPTIONS:
    MODEL_OPTIONS.insert(0, DEFAULT_MODEL)

# Khoá bước (pipeline phát ra) → nhãn hiển thị tiếng Việt. Thứ tự đúng theo
# thực thi: hiểu → giả thuyết → tìm kiếm → suy luận → kết luận.
STEP_LABELS = {
    "understand": "Hiểu câu hỏi — khớp khái niệm BHXH (ontology)",
    "hypothesis": "Hình thành giả thuyết (HyDE)",
    "search": "Tìm kiếm điều/khoản luật (dense BGE-M3 + Neo4j)",
    "reason": "Tổng hợp & suy luận (sinh Prolog + thực thi)",
    "conclude": "Kết luận (IRAC)",
}

# ---------------------------------------------------------------------------
# Resource loading — nặng (Neo4j + BGE-M3 + SWI-Prolog), cache 1 lần.
# Retriever (BGE-M3) độc lập với model GPT → cache một lần; model GPT được áp
# vào HyDE generator ngay trước mỗi câu hỏi (set_llm_model) nên đổi model KHÔNG
# phải nạp lại BGE-M3. Pipeline (sinh Prolog + IRAC) nhẹ, cache theo model.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Nạp retriever (Neo4j + BGE-M3)…")
def get_retriever() -> DenseHydeSemanticAsLogicLMRetriever:
    return DenseHydeSemanticAsLogicLMRetriever()


@st.cache_resource(show_spinner=False)
def get_pipeline(model: str) -> LogicLMHydeSemanticPipeline:
    # Nhẹ: chỉ bọc retriever đã cache + nạp prompt rule-gen. Retriever (BGE-M3)
    # tự hiện spinner riêng khi nạp lần đầu.
    return LogicLMHydeSemanticPipeline(retriever=get_retriever(), model=model)


# ---------------------------------------------------------------------------
# Hội thoại — đa hội thoại, lưu trong session (reset khi tắt app)
# ---------------------------------------------------------------------------


def _new_conversation(force: bool = False) -> str:
    convs = st.session_state.conversations
    active = st.session_state.get("active_id")
    # Tránh tạo hàng loạt hội thoại rỗng: nếu đang ở một hội thoại chưa có lượt
    # nào thì dùng lại nó.
    if not force and active in convs and not convs[active]["turns"]:
        return active
    cid = uuid.uuid4().hex[:8]
    convs[cid] = {"title": "Hội thoại mới", "turns": []}
    st.session_state.active_id = cid
    return cid


if "conversations" not in st.session_state:
    st.session_state.conversations = {}
    _new_conversation(force=True)

# Cài đặt — lưu trong session, chỉnh trong dialog riêng (nút ⚙️).
st.session_state.setdefault("model", DEFAULT_MODEL)
st.session_state.setdefault("top_k", 8)
st.session_state.setdefault("show_reasoning", True)


@st.dialog("⚙️ Cài đặt")
def _settings_dialog() -> None:
    m = st.selectbox(
        "Model GPT",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(st.session_state.model)
        if st.session_state.model in MODEL_OPTIONS
        else 0,
        help="Dùng chung cho cả sinh giả thuyết (HyDE), sinh Prolog và render IRAC.",
    )
    tk = st.slider("top_k (số khoản retrieve)", 4, 20, st.session_state.top_k)
    sr = st.toggle("Hiện reasoning chi tiết", value=st.session_state.show_reasoning)
    if st.button("Lưu", type="primary", use_container_width=True):
        st.session_state.model = m
        st.session_state.top_k = tk
        st.session_state.show_reasoning = sr
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar — Hội thoại mới + nút Cài đặt (góc trái trên) + lịch sử
# ---------------------------------------------------------------------------

with st.sidebar:
    c_new, c_set = st.columns([4, 1])
    with c_new:
        if st.button("➕ Hội thoại mới", use_container_width=True, type="primary"):
            _new_conversation()
            st.rerun()
    with c_set:
        if st.button("⚙️", use_container_width=True, help="Cài đặt"):
            _settings_dialog()

    st.markdown("##### 💬 Lịch sử hội thoại")
    for cid, conv in reversed(list(st.session_state.conversations.items())):
        is_active = cid == st.session_state.active_id
        label = ("● " if is_active else "○ ") + (conv["title"] or "Hội thoại mới")
        if st.button(label, key=f"conv_{cid}", use_container_width=True):
            st.session_state.active_id = cid
            st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("⚖️ Legal KG — Hỏi đáp BHXH")
st.caption("Logic-LM + HyDE-semantic · câu hỏi → giả thuyết → Prolog → kết luận")

# ---------------------------------------------------------------------------
# Pre-warm lúc khởi động — nạp BGE-M3 + kết nối Neo4j + dựng pipeline NGAY khi
# mở app (spinner của cache_resource chỉ hiện lần đầu). Nhờ vậy câu hỏi đầu tiên
# không phải chờ nạp model — người dùng hỏi là trả lời. Các lần rerun sau: cache
# hit, tức thì.
# ---------------------------------------------------------------------------

try:
    _retriever = get_retriever()
    _retriever.set_llm_model(st.session_state.model)
    get_pipeline(st.session_state.model)
    _warm_ok = True
except Exception as e:  # hạ tầng (Neo4j/model) hỏng → báo lỗi rõ, không crash
    _warm_ok = False
    st.error(f"Không nạp được mô hình/Neo4j: {type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# Render một turn đã lưu trong history
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


def _render_reasoning_section(ans, key: str) -> None:
    """Mục 'Suy luận' — mặc định 'Dễ đọc' (I-R-A của IRAC: Issue / Rule /
    Application, dễ nắm bắt). Chuyển 'Raw Prolog' để xem chương trình + các
    status (prolog_success / status / repair_rounds chỉ hiện ở chế độ này)."""
    irac = ans.irac_sections or {}
    view = st.radio(
        "Hiển thị suy luận",
        ["Dễ đọc", "Raw Prolog"],
        horizontal=True,
        key=f"pv_{key}",
        label_visibility="collapsed",
    )
    if view == "Dễ đọc":
        ira = {
            "issue": "Vấn đề (Issue)",
            "rule": "Căn cứ pháp lý (Rule)",
            "application": "Áp dụng (Application)",
        }
        if any(irac.get(k) for k in ira):
            for k, lab in ira.items():
                if irac.get(k):
                    st.markdown(f"- **{lab}:** {irac[k]}")
        else:
            st.caption("(chưa suy luận ra kết quả — xem Raw Prolog để biết chi tiết)")
    else:
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


def _user_facing_answer(ans) -> str:
    """Câu trả lời cho người đọc — không lộ chi tiết Prolog. Khi suy luận thất
    bại (không có plain_answer/IRAC) trả về thông báo thân thiện thay vì sentinel
    nội bộ ("[Pipeline không trả về kết luận. prolog_status=…]"); chi tiết kỹ
    thuật vẫn xem được ở mục Suy luận → Raw Prolog."""
    if ans.plain_answer:
        return ans.plain_answer
    if ans.prolog_success and ans.answer:
        return ans.answer
    return (
        "Hệ thống chưa suy luận ra kết luận chắc chắn cho câu hỏi này. Bạn có thể "
        "thử diễn đạt lại câu hỏi, hoặc mở mục **3 · Suy luận → Raw Prolog** để xem "
        "chi tiết kỹ thuật."
    )


def render_turn(turn: dict) -> None:
    ans = turn["ans"]
    ctx = turn["semantic"]
    verified = turn["verified"]
    key = turn["id"]

    with st.chat_message("user"):
        st.write(turn["question"])

    with st.chat_message("assistant"):
        st.markdown(_user_facing_answer(ans))
        if ans.citations:
            st.markdown(_citation_chips(ans, verified), unsafe_allow_html=True)
        if ans.error:
            st.error(f"Pipeline trả về lỗi: {ans.error}")

        if not st.session_state.show_reasoning:
            return

        with st.expander("🔍 Vì sao có câu trả lời này", expanded=False):
            # 1 · Câu hỏi
            st.markdown("**1 · Câu hỏi**")
            st.markdown(
                f'<div class="hypo-box empty">{html.escape(turn["question"])}</div>',
                unsafe_allow_html=True,
            )

            # 2 · Hypothesis (khung khái niệm + đoạn văn giả định)
            st.markdown("**2 · Giả thuyết (HyDE-semantic)**")
            st.caption(
                "Khung khái niệm BHXH khớp từ câu hỏi (so khớp chuỗi với ontology) → "
                "đoạn “văn bản luật giả định” định hướng sinh Prolog (không chứa số Điều/Khoản)."
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
                    '<div class="hypo-box empty">(không có đoạn giả thuyết)</div>',
                    unsafe_allow_html=True,
                )

            # 3 · Suy luận (Dễ đọc: I-R-A của IRAC / Raw Prolog)
            st.markdown("**3 · Suy luận**")
            _render_reasoning_section(ans, key)

            # 4 · Kết luận — chỉ phần Conclusion cuối của IRAC (nói thẳng kết luận)
            st.markdown("**4 · Kết luận**")
            irac = ans.irac_sections or {}
            conclusion = (irac.get("conclusion") or "").strip() or _user_facing_answer(ans)
            st.markdown(conclusion)
            st.caption(
                f"⏱ {ans.elapsed_s:.1f}s · tokens {ans.prompt_tokens}+{ans.completion_tokens} "
                f"· model `{turn['model']}`"
            )


active = st.session_state.conversations[st.session_state.active_id]
for turn in active["turns"]:
    render_turn(turn)


# ---------------------------------------------------------------------------
# Input — câu hỏi hiện TRƯỚC, rồi reasoning live, rồi kết quả (sau rerun)
# ---------------------------------------------------------------------------

if q := st.chat_input("Nhập câu hỏi BHXH…"):
    model = st.session_state.model        # cài đặt nằm trong dialog ⚙️ → đọc từ session
    top_k = st.session_state.top_k
    try:
        retriever = get_retriever()
        retriever.set_llm_model(model)        # 1 model dùng chung cho HyDE + Prolog + IRAC
        pipe = get_pipeline(model)
        pipe.top_k = top_k
    except Exception as e:  # hạ tầng (Neo4j/model) hỏng → báo lỗi, không crash app
        st.error(f"Không khởi tạo được pipeline: {type(e).__name__}: {e}")
        st.stop()

    with st.chat_message("user"):          # #1 — câu hỏi hiện ngay
        st.write(q)

    ans = ctx = None
    verified: dict = {}
    with st.chat_message("assistant"):
        with st.status("Reasoning…", expanded=True) as status:  # #7, #8 — live từng bước
            def on_step(key: str) -> None:
                st.write("• " + STEP_LABELS.get(key, key))
                status.update(label="Reasoning… — " + STEP_LABELS.get(key, key))

            try:
                ans = pipe.ask(q, on_step=on_step)
                # Đọc NGAY sau ask() rồi lưu — last_* bị ghi đè ở câu hỏi kế tiếp.
                ctx = getattr(pipe.retriever, "last_semantic_context", None)
                try:
                    verified = pipe.retriever.verify_citations(ans.citation_ids)
                except Exception:
                    verified = {}
            except Exception as e:
                status.update(label="Lỗi khi suy luận", state="error")
                st.error(f"Pipeline lỗi: {type(e).__name__}: {e}")
                st.stop()
            status.update(label="Hoàn tất reasoning", state="complete", expanded=False)

    # Lưu turn vào hội thoại đang mở rồi rerun → kết quả render qua render_turn.
    conv = st.session_state.conversations[st.session_state.active_id]
    conv["turns"].append(
        {
            "id": uuid.uuid4().hex[:8],
            "question": q,
            "ans": ans,
            "semantic": ctx,
            "verified": verified,
            "model": model,
        }
    )
    if conv["title"] in ("", "Hội thoại mới"):
        conv["title"] = (q[:38] + "…") if len(q) > 38 else q
    st.rerun()
