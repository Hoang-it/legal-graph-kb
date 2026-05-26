"""Tính metrics cho multimodel experiment.

Mỗi (arm, model) là 1 combo. Reuse toàn bộ metric functions từ
`compute_metrics.py` (faithfulness, hallucination, citation_*, answer_relevance,
bertscore, cost, latency, prolog rollback).

Pairwise judge: per-model, so sánh elite_graphrag vs elite_no_retrieval —
trả lời câu hỏi "Within mỗi model, retrieval có giúp elite không?"

Cost/Token-aware: mỗi combo dùng pricing model-specific
(gpt-4o-mini judge giữ nguyên).

Output:
    data/eval/multimodel/metrics.json   — per-record by combo key
    data/eval/multimodel/judge_cache.jsonl — separate cache file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

# Reuse metric functions + helpers
from experiments.compute_metrics import (  # noqa: E402
    JudgeCache,
    _Neo,
    ELITE_ARMS,
    m_answer_relevance,
    m_citation_precision,
    m_citation_recall,
    m_citation_validity,
    m_faithfulness,
    m_hallucination,
    m_latency,
    m_pairwise,
    m_prolog_rollback,
    compute_bertscore_all,
)

RESULTS_ROOT = Path("data/eval/multimodel/results")
METRICS_OUT = Path("data/eval/multimodel/metrics.json")
METRICS_CSV = Path("data/eval/multimodel/metrics.csv")
JUDGE_CACHE = Path("data/eval/multimodel/judge_cache.jsonl")

# Pricing per 1M tokens (USD). Input / Output.
MODEL_PRICING = {
    "gpt-4.1":     (2.00, 8.00),
    "gpt-4o":      (2.50, 10.00),
    "gpt-5":       (1.25, 10.00),
    "gpt-5-mini":  (0.25, 2.00),
    # Fallback for judges (gpt-4o-mini)
    "gpt-4o-mini": (0.15, 0.60),
}


# ---------------------------------------------------------------------------
# Combo discovery
# ---------------------------------------------------------------------------

def combo_key(arm: str, model: str) -> str:
    """Synthetic key dùng cho cả filename combo + metric key + judge cache."""
    model_safe = model.replace(".", "_")
    return f"{arm}__{model_safe}"


def parse_combo(combo: str) -> tuple[str, str]:
    """Inverse: 'elite_no_retrieval__gpt-4_1' → ('elite_no_retrieval', 'gpt-4.1')."""
    if "__" not in combo:
        raise ValueError(f"Combo dir name không có '__': {combo}")
    arm, model_safe = combo.rsplit("__", 1)
    # Reverse: chỉ chuyển '_' sau 'gpt-' về '.' (gpt-4_1 → gpt-4.1, giữ gpt-5-mini)
    # Heuristic: nếu model có dạng gpt-N_M thì replace '_' → '.'
    import re
    model = re.sub(r"(gpt-\d+)_(\d+)", r"\1.\2", model_safe)
    return arm, model


def discover_combos(root: Path) -> list[tuple[str, str, str]]:
    """Trả về list (combo_key, arm, model)."""
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if not any(d.glob("A*.json")):
            continue
        arm, model = parse_combo(d.name)
        out.append((d.name, arm, model))
    return out


def load_records(combo_dir: Path, combo_key_str: str, arm: str, model: str) -> list[dict]:
    out = []
    for fp in sorted(combo_dir.glob("A*.json")):
        if fp.name.endswith(".error.json"):
            continue
        with fp.open(encoding="utf-8") as f:
            r = json.load(f)
        # Override arm = combo_key để cache key (faith/halu/etc) phân biệt được
        # giữa các combos (gpt-4o vs gpt-5 same question different answer).
        r["base_arm"] = arm
        r["model"] = model
        r["arm"] = combo_key_str
        out.append(r)
    return out


def _prolog_rollback_combo(record: dict) -> dict:
    """Wrapper m_prolog_rollback: check base_arm (combo key không thuộc ELITE_ARMS)."""
    base = record.get("base_arm")
    if base not in ELITE_ARMS:
        return {
            "prolog_success": None,
            "n_repair_rounds": None,
            "first_try_success": None,
            "repair_invoked": None,
            "prolog_status": None,
        }
    ps = bool(record.get("prolog_success", False))
    nr = int(record.get("n_repair_rounds", 0) or 0)
    return {
        "prolog_success": ps,
        "n_repair_rounds": nr,
        "first_try_success": ps and nr == 0,
        "repair_invoked": nr >= 1,
        "prolog_status": record.get("prolog_status") or "",
    }


# ---------------------------------------------------------------------------
# Model-aware cost
# ---------------------------------------------------------------------------

def m_cost_for_model(record: dict, model: str) -> dict:
    pin = record.get("prompt_tokens") or 0
    pout = record.get("completion_tokens") or 0
    in_p, out_p = MODEL_PRICING.get(model, (1.0, 4.0))
    cost = (pin * in_p + pout * out_p) / 1e6
    return {
        "prompt_tokens": pin,
        "completion_tokens": pout,
        "cost_usd": round(cost, 6),
        "model": model,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--skip-judge", action="store_true")
    p.add_argument("--skip-bertscore", action="store_true")
    args = p.parse_args()

    if not RESULTS_ROOT.exists():
        print(f"FAIL: {RESULTS_ROOT} không tồn tại", file=sys.stderr)
        return 1

    combos = discover_combos(RESULTS_ROOT)
    if not combos:
        print(f"FAIL: không tìm thấy combo nào trong {RESULTS_ROOT}",
              file=sys.stderr)
        return 1

    print(f"Discovered {len(combos)} combos:")
    recs_by_combo: dict[str, list[dict]] = {}
    for combo_k, arm, model in combos:
        recs = load_records(RESULTS_ROOT / combo_k, combo_k, arm, model)
        recs_by_combo[combo_k] = recs
        print(f"  {combo_k:<45} {len(recs)} records  (arm={arm} model={model})")

    if args.limit:
        for k in recs_by_combo:
            recs_by_combo[k] = recs_by_combo[k][: args.limit]
        print(f"  (limit {args.limit})")

    # Pair theo stt: chỉ giữ stt có TẤT CẢ combos
    by_stt: dict[int, dict[str, dict]] = defaultdict(dict)
    for combo_k, recs in recs_by_combo.items():
        for r in recs:
            by_stt[r["stt"]][combo_k] = r
    paired_stts = sorted(
        s for s, v in by_stt.items() if all(k in v for k in recs_by_combo)
    )
    print(f"  Cặp đầy đủ (mọi combo có): {len(paired_stts)}")

    neo = _Neo()
    jc = JudgeCache(JUDGE_CACHE)
    print(f"Judge cache: {len(jc.cache)} entries  (path={JUDGE_CACHE})")

    client = None
    embed_model = None
    if not args.skip_judge:
        from openai import OpenAI
        client = OpenAI()
        print("Loading BGE-M3 for answer_relevance...")
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(
            os.getenv("EMBED_MODEL", "BAAI/bge-m3"),
            device=os.getenv("EMBED_DEVICE", "cuda"),
        )

    # ---- Per-record metrics ----
    all_metrics: dict[str, list[dict]] = {k: [] for k in recs_by_combo}
    t_start = time.time()

    # Pairwise: per model, so sánh elite_graphrag vs elite_no_retrieval
    # Build mapping model → (no_retrieval_combo, graphrag_combo)
    model_to_combos: dict[str, dict[str, str]] = defaultdict(dict)
    for combo_k, arm, model in combos:
        model_to_combos[model][arm] = combo_k

    for i, stt in enumerate(paired_stts, 1):
        pair = by_stt[stt]
        for combo_k, arm, model in combos:
            rec = pair[combo_k]
            m: dict[str, Any] = {
                "stt": stt,
                "arm": arm,
                "model": model,
                "combo": combo_k,
                "citation_validity": m_citation_validity(rec, neo),
                "citation_recall": m_citation_recall(rec),
                "cost": m_cost_for_model(rec, model),
                "latency": m_latency(rec),
                "prolog_rollback": _prolog_rollback_combo(rec),
            }
            if not args.skip_judge:
                m["faithfulness"] = m_faithfulness(rec, neo, client, jc)
                m["citation_precision"] = m_citation_precision(rec, neo, client, jc)
                m["answer_relevance"] = m_answer_relevance(rec, client, jc, embed_model)
                m["hallucination"] = m_hallucination(rec, neo, client, jc)
            all_metrics[combo_k].append(m)

        # Pairwise per model: elite_graphrag vs elite_no_retrieval
        if not args.skip_judge:
            for model, arm_to_combo in model_to_combos.items():
                if "elite_no_retrieval" in arm_to_combo and "elite_graphrag" in arm_to_combo:
                    nr_combo = arm_to_combo["elite_no_retrieval"]
                    gr_combo = arm_to_combo["elite_graphrag"]
                    # baseline=elite_no_retrieval, compare=elite_graphrag
                    pw = m_pairwise(pair[nr_combo], pair[gr_combo], client, jc)
                    # Lưu vào record của elite_graphrag combo
                    all_metrics[gr_combo][-1]["pairwise_vs_no_retrieval"] = pw

        if i % 5 == 0 or i == len(paired_stts):
            elapsed = time.time() - t_start
            n_combos = len(recs_by_combo)
            print(f"  [{i:>3}/{len(paired_stts)}] "
                  f"{elapsed:.0f}s elapsed "
                  f"(~{elapsed * len(paired_stts) / max(i, 1) - elapsed:.0f}s remaining)",
                  flush=True)

    # ---- BERTScore batch ----
    if not args.skip_bertscore:
        try:
            bs_recs = []
            for combo_k, recs in recs_by_combo.items():
                for r in recs:
                    if (r["stt"] in paired_stts
                        and r.get("gold_answer") and r.get("answer")):
                        bs_recs.append({
                            "arm": combo_k,
                            "stt": r["stt"],
                            "answer": r["answer"],
                            "gold_answer": r["gold_answer"],
                        })
            if bs_recs:
                print(f"\nBERTScore over {len(bs_recs)} records...")
                bs_results = compute_bertscore_all(bs_recs)
                for combo_k in recs_by_combo:
                    for m in all_metrics[combo_k]:
                        bs = bs_results.get((combo_k, m["stt"]))
                        if bs:
                            m["bertscore"] = bs
        except ImportError as e:
            print(f"  ! BERTScore skip: {e}", file=sys.stderr)
        except Exception as e:
            print(f"  ! BERTScore failed: {e}", file=sys.stderr)

    # ---- Save ----
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    with METRICS_OUT.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nSaved per-record: {METRICS_OUT} "
          f"({METRICS_OUT.stat().st_size / 1024:.1f} KB)")

    _print_summary(all_metrics, combos)

    neo.close()
    jc.close()
    return 0


def _print_summary(all_metrics: dict, combos: list[tuple[str, str, str]]):
    print("\n=== SUMMARY (mean values) ===")
    for combo_k, arm, model in combos:
        recs = all_metrics.get(combo_k, [])
        if not recs:
            continue

        def _avg(chain, _recs=recs):
            vals = []
            for r in _recs:
                v = r
                for k in chain:
                    if v is None:
                        break
                    v = v.get(k) if isinstance(v, dict) else None
                if v is not None:
                    vals.append(v)
            return sum(vals) / len(vals) if vals else None

        def _bool_rate(chain, _recs=recs):
            vals = []
            for r in _recs:
                v = r
                for k in chain:
                    if v is None:
                        break
                    v = v.get(k) if isinstance(v, dict) else None
                if v is not None:
                    vals.append(1.0 if v else 0.0)
            return sum(vals) / len(vals) if vals else None

        print(f"\n[{combo_k}] n={len(recs)}  arm={arm} model={model}")
        print(f"  citation_validity      : {_avg(['citation_validity', 'validity_rate'])}")
        print(f"  citation_recall        : {_avg(['citation_recall', 'recall'])}")
        print(f"  citation_precision     : {_avg(['citation_precision', 'precision'])}")
        print(f"  faithfulness           : {_avg(['faithfulness', 'faithfulness'])}")
        print(f"  answer_relevance       : {_avg(['answer_relevance', 'answer_relevance'])}")
        print(f"  hallucination_rate     : {_avg(['hallucination', 'hallucination_rate'])}")
        print(f"  bertscore_f1           : {_avg(['bertscore', 'bertscore_f1'])}")
        print(f"  cost_usd (mean)        : {_avg(['cost', 'cost_usd'])}")
        print(f"  latency_s (mean)       : {_avg(['latency', 'latency_s'])}")
        if arm in ELITE_ARMS:
            print(f"  prolog_success_rate    : {_bool_rate(['prolog_rollback', 'prolog_success'])}")
            print(f"  repair_invoked_rate    : {_bool_rate(['prolog_rollback', 'repair_invoked'])}")
            print(f"  avg_repair_rounds      : {_avg(['prolog_rollback', 'n_repair_rounds'])}")
            print(f"  first_try_success_rate : {_bool_rate(['prolog_rollback', 'first_try_success'])}")

    # Pairwise consensus per model
    from collections import Counter
    print("\n=== PAIRWISE per-model: elite_graphrag vs elite_no_retrieval ===")
    for combo_k, arm, model in combos:
        if arm != "elite_graphrag":
            continue
        recs = all_metrics.get(combo_k, [])
        pws = [r["pairwise_vs_no_retrieval"] for r in recs
               if "pairwise_vs_no_retrieval" in r]
        if not pws:
            continue
        consensus = Counter(p["consensus"] for p in pws)
        n = len(pws)
        print(f"\n  Model {model} (n={n}):")
        for label, cnt in consensus.most_common():
            print(f"    {label:<28} {cnt:>3} ({cnt / n * 100:.1f}%)")


if __name__ == "__main__":
    sys.exit(main())
