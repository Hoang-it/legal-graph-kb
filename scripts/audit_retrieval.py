"""Stage A post-hoc retrieval audit — Plan v5 Phase 7.

Reads stored experiment records for arms ``graphrag`` (Sprint 0),
``graphrag_v5`` (Sprint 1) and ``graphrag_v5_m2`` (Sprint 2) on the same
30-probe (stt 1..30), extracts the article ids of every retrieved hit, and
computes ``retrieval_recall@K`` for K in {5, 10, 12, all}.

Output:
- Overall macro recall@K per arm.
- Stratified macro recall@K per category (in_corpus, mixed, ooc, unparseable).
- E2E vs retrieval gap = bottleneck signal (LLM-citing vs retrieval ceiling).
- Per-record audit JSON for inspection.

Zero API calls; reads existing JSON record files only.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.legal_metadata import load_law_metadata

# ---------------------------------------------------------------------------
# Configuration — same 30-probe everyone ran on
# ---------------------------------------------------------------------------

PROBE_STT = list(range(1, 31))
KS = (5, 10, 12)

ARMS = [
    # (label, results_root, arm_subdir, hit_extractor_key)
    ("graphrag (S0)", "experiments/01_initial_eval/results", "graphrag", "graphrag"),
    ("graphrag_v5 (S1)", "experiments/03_v5_sprint1_vanilla/results", "graphrag_v5", "v5"),
    ("graphrag_v5_m2 (S2)", "experiments/04_v5_sprint2_m2/results", "graphrag_v5_m2", "v5"),
]

# Source of truth for in-corpus categorization
_RE_CODE = re.compile(r"\d+/\d{4}/(?:QH\d+|N[ĐD]-CP|NQ-CP|TT-[A-Z]+|CP|TTg)")


# ---------------------------------------------------------------------------
# Hit extraction per arm flavour
# ---------------------------------------------------------------------------

def _articles_from_clause_id(cid: str) -> str | None:
    """`L41_2024.A64.K1` → `L41_2024.A64`."""
    if not cid:
        return None
    # Strip everything after the article token (.K.., .D..)
    m = re.match(r"^([A-Z0-9_]+\.A\d+[a-z]?)", cid)
    return m.group(1) if m else None


def extract_articles(record: dict, flavour: str) -> list[str]:
    """Return unique retrieved article ids in rank order.

    - flavour 'graphrag': from ``vector_hits[*].clause_id``.
    - flavour 'v5': from ``hits[*]`` — uses ``article_id`` for seeds,
      ``target_id`` for neighbours.
    """
    seen: dict[str, None] = {}
    if flavour == "graphrag":
        for h in record.get("vector_hits", []) or []:
            aid = _articles_from_clause_id(h.get("clause_id") or "")
            if aid:
                seen.setdefault(aid, None)
    elif flavour == "v5":
        for h in record.get("hits", []) or []:
            if h.get("kind") == "seed":
                aid = h.get("article_id") or _articles_from_clause_id(h.get("clause_id") or "")
            elif h.get("kind") == "neighbor":
                aid = h.get("target_id")
            else:
                aid = h.get("article_id") or h.get("target_id")
            if aid:
                seen.setdefault(aid, None)
    return list(seen.keys())


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], gold: set[str], k: int | None) -> float | None:
    if not gold:
        return None
    pool = set(retrieved if k is None else retrieved[:k])
    return len(gold & pool) / len(gold)


# ---------------------------------------------------------------------------
# Categorization (gold source-type)
# ---------------------------------------------------------------------------

def categorize(raw: str | list, in_corpus_codes: set[str]) -> str:
    if not raw:
        return "unparseable"
    if isinstance(raw, list):
        raw = "\n".join(str(x) for x in raw)
    hits = _RE_CODE.findall(raw)
    if not hits:
        return "unparseable"
    in_kg = sum(1 for h in hits if h in in_corpus_codes)
    if in_kg == len(hits):
        return "in_corpus"
    if in_kg == 0:
        return "ooc"
    return "mixed"


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def load_gold_articles() -> dict[int, list[str]]:
    """Prefer Sprint 2 normalized gold (most recent, includes QD366 fix)."""
    p = Path("experiments/04_v5_sprint2_m2/metrics/gold_citations_normalized.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data.get("records", {}).items()}


def main() -> int:
    questions = json.loads(Path("data/eval/questions_200.json").read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}
    gold_map = load_gold_articles()

    # ----- compute per-arm per-stt -----
    arm_results: dict[str, list[dict]] = defaultdict(list)
    for label, root, arm, flavour in ARMS:
        for stt in PROBE_STT:
            p = Path(root) / arm / f"A{stt}.json"
            if not p.exists():
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            retrieved = extract_articles(rec, flavour)
            gold = set(gold_map.get(stt) or [])
            row = {
                "stt": stt,
                "arm": label,
                "n_retrieved": len(retrieved),
                "n_gold": len(gold),
                "retrieved_top_k_ids": retrieved,
                "gold_articles": sorted(gold),
                "category": categorize(
                    q_by_stt[stt].get("gold_citations_raw"), in_corpus_codes
                ),
                "e2e_citation_ids": list(rec.get("citation_ids") or []),
            }
            for k in KS:
                row[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
            row["recall@all"] = recall_at_k(retrieved, gold, None)
            # E2E recall: article-level intersection with gold
            e2e_articles = {
                _articles_from_clause_id(c) for c in row["e2e_citation_ids"]
            }
            e2e_articles.discard(None)
            row["e2e_recall"] = recall_at_k(list(e2e_articles), gold, None)
            arm_results[label].append(row)

    # ----- aggregate macro -----
    def _macro(values: list) -> float | None:
        vs = [v for v in values if v is not None]
        return round(mean(vs), 4) if vs else None

    print("=" * 78)
    print("Stage A — Retrieval-only audit (30-probe, post-hoc)")
    print("=" * 78)
    print(f'{"arm":<22}{"n":>4}{"@5":>10}{"@10":>10}{"@12":>10}{"@all":>10}{"E2E":>10}{"gap":>10}')
    overall_macros = {}
    for label, _r, _a, _f in ARMS:
        rows = arm_results.get(label) or []
        m5 = _macro([r["recall@5"] for r in rows])
        m10 = _macro([r["recall@10"] for r in rows])
        m12 = _macro([r["recall@12"] for r in rows])
        m_all = _macro([r["recall@all"] for r in rows])
        m_e2e = _macro([r["e2e_recall"] for r in rows])
        gap = None if (m12 is None or m_e2e is None) else round(m12 - m_e2e, 4)
        overall_macros[label] = {"m5": m5, "m10": m10, "m12": m12, "m_all": m_all, "e2e": m_e2e, "gap": gap}
        print(
            f"{label:<22}{len(rows):>4}{(m5 if m5 is not None else '-'):>10}"
            f"{(m10 if m10 is not None else '-'):>10}"
            f"{(m12 if m12 is not None else '-'):>10}"
            f"{(m_all if m_all is not None else '-'):>10}"
            f"{(m_e2e if m_e2e is not None else '-'):>10}"
            f"{(gap if gap is not None else '-'):>10}"
        )

    # ----- stratified -----
    print()
    print("=== Stratified by gold corpus type ===")
    cats = ["in_corpus", "mixed", "ooc", "unparseable"]
    print(f'{"category":<14}{"arm":<22}{"n":>4}{"@5":>10}{"@10":>10}{"@12":>10}{"E2E":>10}')
    for cat in cats:
        for label, _r, _a, _f in ARMS:
            rows = [r for r in (arm_results.get(label) or []) if r["category"] == cat]
            if not rows:
                continue
            m5 = _macro([r["recall@5"] for r in rows])
            m10 = _macro([r["recall@10"] for r in rows])
            m12 = _macro([r["recall@12"] for r in rows])
            e2e = _macro([r["e2e_recall"] for r in rows])
            print(
                f"{cat:<14}{label:<22}{len(rows):>4}"
                f"{(m5 if m5 is not None else '-'):>10}"
                f"{(m10 if m10 is not None else '-'):>10}"
                f"{(m12 if m12 is not None else '-'):>10}"
                f"{(e2e if e2e is not None else '-'):>10}"
            )
        print()

    # ----- per-record dump -----
    out_dir = Path("experiments/05_v5_retrieval_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "probe_stt": PROBE_STT,
        "Ks": list(KS),
        "arms": [label for label, *_ in ARMS],
        "overall_macros": overall_macros,
        "per_record": arm_results,
    }
    out = out_dir / "stage_a_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Per-record details written: {out}")

    # ----- bottleneck verdict -----
    print()
    print("=== Bottleneck verdict (gap = retrieval@12 - E2E) ===")
    for label, m in overall_macros.items():
        if m["gap"] is None:
            continue
        if m["gap"] >= 0.15:
            verdict = "LLM citing is the cap (large gap) — Sprint 3 focus = generator"
        elif m["gap"] <= 0.05:
            verdict = "Retrieval is the cap (small gap) — Sprint 3 focus = retrieval"
        else:
            verdict = "Mixed — retrieval + generator both contribute"
        print(f"  {label:<22} gap={m['gap']:+.4f}  → {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
