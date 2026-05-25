"""Tính 6 metrics cho 2 arm + lưu metrics.json + metrics.csv.

Metrics (mỗi metric có ref tới paper peer-reviewed):
- citation_validity         : % citation_id tồn tại trong Neo4j (deterministic)
- citation_recall (Liu 2023): % câu có ≥1 citation nearby
- citation_precision (Liu 2023): citation thực sự support claim gần đó (judge)
- faithfulness (Es 2024 RAGAS) : % claim được support bởi text của cited articles
- answer_relevance (Es 2024)   : cosine sim giữa Q gốc và Q sinh ngược từ answer
- hallucination_rate (Magesh 2025): % response có misstate/invent (judge)
- pairwise_winner (Zheng 2023) : judge so sánh A vs B (position swap)
- bertscore_f1 (Zhang 2020)    : semantic sim với gold_answer
- cost_usd / latency_s         : objective
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")
JUDGE_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cùng model với generator

GRAPHRAG_DIR = Path("data/eval/results/graphrag")
LLM_ONLY_DIR = Path("data/eval/results/llm_only")
METRICS_OUT = Path("data/eval/metrics.json")
METRICS_CSV = Path("data/eval/metrics.csv")
JUDGE_CACHE = Path("data/eval/judge_cache.jsonl")  # cache raw judge responses

# GPT-4o-mini pricing (USD per 1M tokens, late-2024)
COST_INPUT_PER_M = 0.15
COST_CACHED_INPUT_PER_M = 0.075
COST_OUTPUT_PER_M = 0.60


_CIT_PAT = re.compile(r"\[Điều\s+(\d+)(?:\s+khoản\s+(\d+))?(?:\s+điểm\s+([a-zđ]))?\]")


def parse_citations(answer: str) -> list[dict]:
    out = []
    for m in _CIT_PAT.finditer(answer):
        art, cl, pt = m.group(1), m.group(2), m.group(3)
        cid = f"L41_2024.A{art}"
        if cl:
            cid += f".K{cl}"
            if pt:
                cid += f".{pt}"
        out.append({"str": m.group(0), "id": cid, "pos": m.start()})
    return out


def split_sentences_vi(text: str) -> list[tuple[str, int, int]]:
    """Split câu tiếng Việt simple. Returns [(sentence, start_pos, end_pos)]."""
    sents: list[tuple[str, int, int]] = []
    text_clean = re.sub(r"\s+", " ", text)
    pos = 0
    for s in re.split(r"(?<=[.!?])\s+", text_clean.strip()):
        s = s.strip()
        if len(s) < 12:
            pos += len(s) + 1
            continue
        sents.append((s, pos, pos + len(s)))
        pos += len(s) + 1
    return sents


# ---------------------------------------------------------------------------
# Neo4j helper
# ---------------------------------------------------------------------------


class _Neo:
    def __init__(self):
        from neo4j import GraphDatabase

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PWD))

    def get_texts(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        with self.driver.session(database=NEO4J_DB) as s:
            rows = s.run(
                """
                UNWIND $ids AS id
                OPTIONAL MATCH (n) WHERE n.id = id AND (n:Article OR n:Clause OR n:Point)
                RETURN id, n.text AS text
            """,
                ids=list(dict.fromkeys(ids)),
            ).data()
        return {r["id"]: r["text"] for r in rows if r["text"]}

    def close(self):
        self.driver.close()


# ---------------------------------------------------------------------------
# DETERMINISTIC METRICS
# ---------------------------------------------------------------------------


def m_citation_validity(record: dict, neo: _Neo) -> dict:
    """% citation IDs tồn tại trong KG."""
    cits = list(dict.fromkeys(record.get("citation_ids") or []))
    if not cits:
        return {"n_citations": 0, "n_valid": 0, "validity_rate": None}
    texts = neo.get_texts(cits)
    n_valid = sum(1 for c in cits if c in texts)
    return {
        "n_citations": len(cits),
        "n_valid": n_valid,
        "validity_rate": round(n_valid / len(cits), 4),
    }


def m_citation_recall(record: dict) -> dict:
    """Liu 2023: % câu có ≥1 citation trong câu hoặc câu liền sau."""
    text = record.get("answer", "")
    cits = parse_citations(text)
    sents = split_sentences_vi(text)
    if not sents:
        return {"n_sentences": 0, "n_with_cite": 0, "recall": None}
    cit_positions = [c["pos"] for c in cits]
    # Một câu có citation nếu trong câu hoặc trong 100 chars sau câu có citation
    n_with = 0
    for s, st, en in sents:
        for p in cit_positions:
            if st - 50 <= p <= en + 100:
                n_with += 1
                break
    return {
        "n_sentences": len(sents),
        "n_with_cite": n_with,
        "recall": round(n_with / len(sents), 4),
    }


def m_cost(record: dict, arm: str, neo_ctx_tokens_est: int = 0) -> dict:
    """Cost USD dựa trên token usage. GraphRAG record không có usage → ước tính."""
    pin = record.get("prompt_tokens")
    pout = record.get("completion_tokens")
    if pin is None or pout is None:
        # GraphRAG: ước tính từ answer length + context (~5000 chars ≈ 1300 tokens)
        answer = record.get("answer", "")
        # Estimate: input ~ context (1300) + system prompt (300) + question (50)
        pin = 1650
        pout = len(answer) // 3  # rough VI chars to tokens
    cost = (pin * COST_INPUT_PER_M + pout * COST_OUTPUT_PER_M) / 1e6
    return {
        "prompt_tokens": pin,
        "completion_tokens": pout,
        "cost_usd": round(cost, 6),
        "estimated": record.get("prompt_tokens") is None,
    }


def m_latency(record: dict) -> dict:
    return {"latency_s": record.get("elapsed_s")}


# ---------------------------------------------------------------------------
# JUDGE METRICS (cached to avoid re-calling)
# ---------------------------------------------------------------------------


class JudgeCache:
    def __init__(self, path: Path):
        self.path = path
        self.cache: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    obj = json.loads(line)
                    self.cache[obj["key"]] = obj["result"]
        self._fh = None

    def get(self, key: str) -> dict | None:
        return self.cache.get(key)

    def put(self, key: str, result: dict) -> None:
        self.cache[key] = result
        if self._fh is None:
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps({"key": key, "result": result}, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh:
            self._fh.close()


def _judge_call(
    client,
    system: str,
    user: str,
    judge_cache: JudgeCache | None = None,
    cache_key: str | None = None,
) -> dict:
    if judge_cache and cache_key:
        cached = judge_cache.get(cache_key)
        if cached:
            return cached
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"_error": "json_decode_failed", "_raw": content}
    result = {
        "data": data,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        },
    }
    if judge_cache and cache_key:
        judge_cache.put(cache_key, result)
    return result


def m_faithfulness(record: dict, neo: _Neo, client, jc: JudgeCache) -> dict:
    """Es 2024 RAGAS: split answer thành claims, judge mỗi claim có support
    bởi text của cited articles không."""
    text = record.get("answer", "")
    cits = list(dict.fromkeys(record.get("citation_ids") or []))
    if not text or not cits:
        return {
            "n_claims": 0,
            "n_supported": 0,
            "faithfulness": None,
            "_skip_reason": "no_text_or_no_citations",
        }

    ctx_map = neo.get_texts(cits)
    if not ctx_map:
        return {
            "n_claims": 0,
            "n_supported": 0,
            "faithfulness": 0.0,
            "_skip_reason": "all_citations_invalid",
        }

    context_block = "\n\n".join(f"[{cid}]:\n{txt[:600]}" for cid, txt in ctx_map.items())

    sys_p = (
        "Bạn là chuyên gia luật BHXH. Đánh giá FAITHFULNESS của câu trả lời theo định nghĩa "
        "RAGAS (Es et al., EACL 2024): với mỗi 'claim' (khẳng định pháp lý) trong câu trả lời, "
        "đánh dấu SUPPORTED nếu có thể suy luận trực tiếp từ CONTEXT, UNSUPPORTED nếu không."
    )
    user_p = (
        f"CONTEXT (text gốc của các Điều/Khoản được citation trong câu trả lời):\n"
        f"{context_block}\n\n"
        f"CÂU TRẢ LỜI:\n{text}\n\n"
        f'Trả về JSON: {{"claims": [{{"text": "...", "supported": true/false}}, ...]}}'
    )

    key = f"faith_{record['arm']}_{record['stt']}"
    r = _judge_call(client, sys_p, user_p, jc, key)
    claims = r["data"].get("claims", []) if isinstance(r["data"], dict) else []
    n = len(claims)
    n_sup = sum(1 for c in claims if c.get("supported"))
    return {
        "n_claims": n,
        "n_supported": n_sup,
        "faithfulness": round(n_sup / n, 4) if n else None,
        "_judge_usage": r["usage"],
    }


def m_answer_relevance(record: dict, client, jc: JudgeCache, embed_model) -> dict:
    """Es 2024 RAGAS: LLM sinh ngược câu hỏi từ answer, đo cosine sim với Q gốc."""
    text = record.get("answer", "")
    question = record.get("question", "")
    if not text or not question:
        return {"answer_relevance": None, "_skip": "missing"}

    sys_p = (
        "Bạn nhận được một câu trả lời. Sinh 3 câu hỏi tiếng Việt mà câu trả lời này "
        "trả lời được. Mỗi câu hỏi ngắn gọn, đầy đủ ý."
    )
    user_p = f'CÂU TRẢ LỜI:\n{text}\n\nTrả về JSON: {{"questions": ["q1", "q2", "q3"]}}'
    key = f"relv_{record['arm']}_{record['stt']}"
    r = _judge_call(client, sys_p, user_p, jc, key)
    qs = r["data"].get("questions", []) if isinstance(r["data"], dict) else []
    if not qs:
        return {"answer_relevance": None, "_skip": "no_questions_generated"}

    # Embed
    embs = embed_model.encode([question] + qs, normalize_embeddings=True, show_progress_bar=False)
    q_emb = embs[0]
    gen_embs = embs[1:]
    sims = [float(np.dot(q_emb, g)) for g in gen_embs]
    return {
        "answer_relevance": round(float(np.mean(sims)), 4),
        "n_generated": len(qs),
        "sims": [round(s, 4) for s in sims],
        "_judge_usage": r["usage"],
    }


def m_citation_precision(record: dict, neo: _Neo, client, jc: JudgeCache) -> dict:
    """Liu 2023: với mỗi citation trong answer, judge xem claim ngay TRƯỚC citation
    có được support bởi text của cited article không. Precision = supported / total."""
    text = record.get("answer", "")
    cits = parse_citations(text)
    if not cits:
        return {"n_citations": 0, "n_supported": 0, "precision": None}

    pairs = []
    for c in cits:
        before = text[: c["pos"]]
        # tìm sentence boundary gần nhất
        last_pun = max(
            before.rfind("."), before.rfind("?"), before.rfind("!"), before.rfind("\n"), 0
        )
        nearby = text[last_pun + 1 : c["pos"]].strip()
        # bỏ bracket markdown nếu có
        nearby = re.sub(r"\[Điều\s+[^\]]*\]", "", nearby).strip()
        if len(nearby) >= 8:
            pairs.append({"id": c["id"], "claim": nearby})

    if not pairs:
        return {
            "n_citations": 0,
            "n_supported": 0,
            "precision": None,
            "_skip_reason": "no_claim_text_before_citations",
        }

    ids = list(dict.fromkeys(p["id"] for p in pairs))
    ctx_map = neo.get_texts(ids)

    items = []
    for i, p in enumerate(pairs):
        ctx = ctx_map.get(p["id"], "<KHÔNG TỒN TẠI TRONG KG>")
        items.append(
            f"  ({i+1}) Citation: [{p['id']}]\n"
            f"      Cited content: {ctx[:500] if ctx != '<KHÔNG TỒN TẠI TRONG KG>' else ctx}\n"
            f"      Claim ngay trước citation: {p['claim'][:300]}"
        )

    sys_p = (
        "Bạn là chuyên gia luật BHXH. Đánh giá CITATION PRECISION theo Liu et al. "
        "(EMNLP Findings 2023): với mỗi cặp (claim, citation), xem claim có "
        "được support bởi cited content không. Citation tới điều không tồn tại = "
        "unsupported."
    )
    user_p = (
        "Danh sách các cặp (claim, citation):\n\n"
        + "\n\n".join(items)
        + '\n\nTrả về JSON: {"items": [{"i": 1, "supported": true/false}, ...]}'
    )

    key = f"citprec_{record['arm']}_{record['stt']}"
    r = _judge_call(client, sys_p, user_p, jc, key)
    items_data = r["data"].get("items", []) if isinstance(r["data"], dict) else []
    n = len(items_data)
    n_sup = sum(1 for it in items_data if it.get("supported"))
    return {
        "n_citations": len(pairs),
        "n_supported": n_sup,
        "precision": round(n_sup / n, 4) if n else None,
        "_judge_usage": r["usage"],
    }


def m_hallucination(record: dict, neo: _Neo, client, jc: JudgeCache) -> dict:
    """Magesh 2025 (Stanford HAI legal): đếm 3 loại hallucination:
       1. misstates_law: nội dung sai so với text gốc
       2. invented_citation: cite tới Điều/Khoản không tồn tại trong KG
       3. unsupported_claim: claim không có evidence

    Output rate = (# hallucinations) / (max(1, # claims))
    """
    text = record.get("answer", "")
    cits = list(dict.fromkeys(record.get("citation_ids") or []))
    if not text:
        return {"hallucination_rate": None}

    valid_texts = neo.get_texts(cits)
    n_invented = sum(1 for c in cits if c not in valid_texts)

    if not cits or not valid_texts:
        # Không có citation thật → không thể judge misstate; mark
        return {
            "n_invented_citations": n_invented,
            "n_total_citations": len(cits),
            "hallucination_rate": 1.0 if n_invented > 0 else None,
            "_skip_reason": "no_valid_citations",
        }

    context_block = "\n\n".join(f"[{cid}]:\n{txt[:600]}" for cid, txt in valid_texts.items())
    sys_p = (
        "Bạn là chuyên gia luật BHXH. Đánh giá HALLUCINATION theo Magesh et al. "
        "(JELS 2025, Stanford RegLab): xác định các claim trong câu trả lời có "
        "(a) misstate (sai nội dung so với text gốc), hoặc (b) unsupported (không "
        "có evidence trong context). Bỏ qua các câu chỉ là intro/outro."
    )
    user_p = (
        f"CONTEXT:\n{context_block}\n\n"
        f"CÂU TRẢ LỜI:\n{text}\n\n"
        f'Trả về JSON: {{"claims": [{{"text": "...", "misstates": true/false, '
        f'"unsupported": true/false}}, ...]}}'
    )
    key = f"halu_{record['arm']}_{record['stt']}"
    r = _judge_call(client, sys_p, user_p, jc, key)
    claims = r["data"].get("claims", []) if isinstance(r["data"], dict) else []
    n_claims = len(claims)
    n_misstate = sum(1 for c in claims if c.get("misstates"))
    n_unsup = sum(1 for c in claims if c.get("unsupported"))
    n_halu_total = n_misstate + n_unsup + n_invented
    denom = max(1, n_claims + n_invented)
    return {
        "n_claims": n_claims,
        "n_misstate": n_misstate,
        "n_unsupported": n_unsup,
        "n_invented_citations": n_invented,
        "hallucination_rate": round(n_halu_total / denom, 4),
        "_judge_usage": r["usage"],
    }


def m_pairwise(record_a: dict, record_b: dict, client, jc: JudgeCache) -> dict:
    """Zheng 2023 LLM-as-Judge pairwise: judge chọn answer tốt hơn cho cùng câu hỏi.
    Position-swap để giảm bias."""
    question = record_a["question"]
    ans_a = record_a["answer"]
    ans_b = record_b["answer"]

    sys_p = (
        "Bạn là chuyên gia luật BHXH. So sánh 2 câu trả lời cho cùng câu hỏi. "
        "Tiêu chí: (1) chính xác về pháp lý, (2) trả lời đúng câu hỏi, (3) citation rõ "
        "ràng và hợp lý. Chọn câu tốt hơn hoặc 'tie'."
    )

    def _ask(a_first: bool, swap_id: str):
        a_label, b_label = ("A", "B") if a_first else ("B", "A")
        first, second = (ans_a, ans_b) if a_first else (ans_b, ans_a)
        user_p = (
            f"CÂU HỎI: {question}\n\n"
            f"TRẢ LỜI {a_label}:\n{first}\n\n"
            f"TRẢ LỜI {b_label}:\n{second}\n\n"
            f'Trả về JSON: {{"winner": "A" / "B" / "tie", "reason": "..."}}'
        )
        key = f"pair_{record_a['stt']}_{swap_id}"
        return _judge_call(client, sys_p, user_p, jc, key)

    r1 = _ask(a_first=True, swap_id="ab")
    r2 = _ask(a_first=False, swap_id="ba")
    w1 = (r1["data"].get("winner") or "").lower() if isinstance(r1["data"], dict) else ""
    w2 = (r2["data"].get("winner") or "").lower() if isinstance(r2["data"], dict) else ""

    # Convert to "graphrag wins" / "llm_only wins" / "tie"
    # arm A của record là record_a["arm"]
    def _vote(w: str, a_first: bool) -> str:
        if w == "tie":
            return "tie"
        # Khi a_first=True: A = record_a, B = record_b
        # Khi a_first=False: A = record_b, B = record_a (đã swap)
        winner_label = w  # "a" or "b"
        if (winner_label == "a" and a_first) or (winner_label == "b" and not a_first):
            return record_a["arm"]
        else:
            return record_b["arm"]

    vote1 = _vote(w1, True)
    vote2 = _vote(w2, False)
    return {
        "vote_ab": vote1,
        "vote_ba": vote2,
        "consensus": vote1 if vote1 == vote2 else "split",
        "raw": {"ab": r1["data"], "ba": r2["data"]},
        "_judge_usage": {
            "prompt_tokens": r1["usage"]["prompt_tokens"] + r2["usage"]["prompt_tokens"],
            "completion_tokens": r1["usage"]["completion_tokens"]
            + r2["usage"]["completion_tokens"],
        },
    }


# ---------------------------------------------------------------------------
# BERTScore (Zhang ICLR 2020)
# ---------------------------------------------------------------------------


def compute_bertscore_all(records_with_gold: list[dict]) -> dict[tuple[str, int], dict]:
    """Tính BERTScore F1 vs gold_answer cho list các record có gold."""
    from bert_score import score as bertscore

    cands = [r["answer"] for r in records_with_gold]
    refs = [r["gold_answer"] for r in records_with_gold]
    print(f"Computing BERTScore for {len(cands)} pairs (multilingual model)...")
    P, R, F1 = bertscore(
        cands,
        refs,
        model_type="bert-base-multilingual-cased",
        lang="vi",
        verbose=False,
        device="cuda" if os.getenv("EMBED_DEVICE", "cuda") == "cuda" else "cpu",
    )
    out = {}
    for i, r in enumerate(records_with_gold):
        out[(r["arm"], r["stt"])] = {
            "bertscore_p": round(float(P[i]), 4),
            "bertscore_r": round(float(R[i]), 4),
            "bertscore_f1": round(float(F1[i]), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_records(arm_dir: Path, arm_name: str) -> list[dict]:
    out = []
    for fp in sorted(arm_dir.glob("A*.json")):
        if fp.name.endswith(".error.json"):
            continue
        with fp.open(encoding="utf-8") as f:
            r = json.load(f)
        r["arm"] = arm_name
        out.append(r)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0, help="Chỉ tính N câu đầu (debug)")
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Bỏ qua các metric cần LLM judge (faithfulness, hallucination, pairwise).",
    )
    p.add_argument(
        "--skip-bertscore", action="store_true", help="Bỏ qua BERTScore (tiết kiệm setup)."
    )
    args = p.parse_args()

    print("Loading records...")
    recs_graph = load_records(GRAPHRAG_DIR, "graphrag")
    recs_llm = load_records(LLM_ONLY_DIR, "llm_only")
    print(f"  graphrag: {len(recs_graph)} records")
    print(f"  llm_only: {len(recs_llm)} records")
    if args.limit:
        recs_graph = recs_graph[: args.limit]
        recs_llm = recs_llm[: args.limit]
        print(f"  (limit {args.limit})")

    # Pair theo stt
    by_stt: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in recs_graph:
        by_stt[r["stt"]]["graphrag"] = r
    for r in recs_llm:
        by_stt[r["stt"]]["llm_only"] = r
    paired_stts = sorted(s for s, v in by_stt.items() if "graphrag" in v and "llm_only" in v)
    print(f"  Cặp (graphrag+llm_only) đầy đủ: {len(paired_stts)}")

    neo = _Neo()
    jc = JudgeCache(JUDGE_CACHE)
    print(f"Judge cache: {len(jc.cache)} entries")

    client = None
    embed_model = None
    if not args.skip_judge:
        from openai import OpenAI

        client = OpenAI()
    # Embed model cho answer_relevance
    if not args.skip_judge:
        print("Loading BGE-M3 for answer_relevance...")
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(
            os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            device=os.getenv("EMBED_DEVICE", "cuda"),
        )

    # ---- Compute per-record metrics ----
    all_metrics: dict[str, list[dict]] = {"graphrag": [], "llm_only": []}
    t_start = time.time()

    for i, stt in enumerate(paired_stts, 1):
        pair = by_stt[stt]
        per = {}
        for arm in ("graphrag", "llm_only"):
            rec = pair[arm]
            m: dict[str, Any] = {
                "stt": stt,
                "arm": arm,
                "citation_validity": m_citation_validity(rec, neo),
                "citation_recall": m_citation_recall(rec),
                "cost": m_cost(rec, arm),
                "latency": m_latency(rec),
            }
            if not args.skip_judge:
                m["faithfulness"] = m_faithfulness(rec, neo, client, jc)
                m["citation_precision"] = m_citation_precision(rec, neo, client, jc)
                m["answer_relevance"] = m_answer_relevance(rec, client, jc, embed_model)
                m["hallucination"] = m_hallucination(rec, neo, client, jc)
            all_metrics[arm].append(m)
            per[arm] = m

        # Pairwise judge
        if not args.skip_judge:
            pw = m_pairwise(pair["graphrag"], pair["llm_only"], client, jc)
            all_metrics["graphrag"][-1]["pairwise"] = pw
            all_metrics["llm_only"][-1]["pairwise"] = pw

        if i % 5 == 0 or i == len(paired_stts):
            elapsed = time.time() - t_start
            print(f"  [{i:>3}/{len(paired_stts)}] {elapsed:.0f}s elapsed", flush=True)

    # ---- BERTScore batch ----
    if not args.skip_bertscore:
        try:
            bs_recs = []
            for arm, recs in [("graphrag", recs_graph), ("llm_only", recs_llm)]:
                for r in recs:
                    if r["stt"] in paired_stts and r.get("gold_answer"):
                        bs_recs.append(
                            {
                                "arm": arm,
                                "stt": r["stt"],
                                "answer": r["answer"],
                                "gold_answer": r["gold_answer"],
                            }
                        )
            bs_results = compute_bertscore_all(bs_recs)
            # Merge into per-record
            for arm in ("graphrag", "llm_only"):
                for m in all_metrics[arm]:
                    bs = bs_results.get((arm, m["stt"]))
                    if bs:
                        m["bertscore"] = bs
        except ImportError as e:
            print(f"  ✗ BERTScore skip — `pip install bert-score` ({e})", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ BERTScore failed: {e}", file=sys.stderr)

    # ---- Save ----
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUT.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved per-record: {METRICS_OUT} ({METRICS_OUT.stat().st_size / 1024:.1f} KB)")

    # Quick summary
    _print_summary(all_metrics)

    neo.close()
    jc.close()
    return 0


def _print_summary(all_metrics):
    print("\n=== SUMMARY (mean values) ===")
    for arm in ("graphrag", "llm_only"):
        recs = all_metrics[arm]
        if not recs:
            continue

        def _avg(key_chain):
            vals = []
            for r in recs:
                v = r
                for k in key_chain:
                    if v is None:
                        break
                    v = v.get(k) if isinstance(v, dict) else None
                if v is not None:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else None

        print(f"\n[{arm}] n={len(recs)}")
        print(f"  citation_validity      : {_avg(['citation_validity', 'validity_rate'])}")
        print(f"  citation_recall        : {_avg(['citation_recall', 'recall'])}")
        print(f"  citation_precision     : {_avg(['citation_precision', 'precision'])}")
        print(f"  faithfulness           : {_avg(['faithfulness', 'faithfulness'])}")
        print(f"  answer_relevance       : {_avg(['answer_relevance', 'answer_relevance'])}")
        print(f"  hallucination_rate     : {_avg(['hallucination', 'hallucination_rate'])}")
        print(f"  bertscore_f1           : {_avg(['bertscore', 'bertscore_f1'])}")
        print(f"  cost_usd (mean)        : {_avg(['cost', 'cost_usd'])}")
        print(f"  latency_s (mean)       : {_avg(['latency', 'latency_s'])}")

    # Pairwise winner tally
    pw_consensus = [r["pairwise"]["consensus"] for r in all_metrics["graphrag"] if "pairwise" in r]
    if pw_consensus:
        from collections import Counter

        c = Counter(pw_consensus)
        print(f"\nPairwise consensus (n={len(pw_consensus)}):")
        for k, v in c.most_common():
            print(f"  {k:<12} {v}")


if __name__ == "__main__":
    sys.exit(main())
