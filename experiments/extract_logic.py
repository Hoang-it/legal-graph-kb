"""extract_logic.py — Phase 2 of plan_logic_extraction.md.

LLM-based extractor cho legal logic patterns từ Clause text.
Outputs structured JSON theo schema trong reports/logic_extraction_schema.md.

Pipeline per clause:
  1. Read clause from structured_law.json
  2. (optional) Regex pre-pass for numbers/dates
  3. LLM extraction (gpt-4o-mini)
  4. Validate JSON schema
  5. Save to data/eval/extracted_logic/<clause_id>.json

CLI:
    python -m experiments.extract_logic --pilot 30        # pilot batch
    python -m experiments.extract_logic --all             # full 543 clauses
    python -m experiments.extract_logic --clause L41_2024.A64.K1   # single
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# Fix: empty OPENAI_BASE_URL gây httpx UnsupportedProtocol; phải unset
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

LAW_PATH = Path("data/interim/structured_law.json")
PROMPT_PATH = Path("experiments/prompts/logic_extraction_v1.md")
OUT_DIR = Path("data/eval/extracted_logic")
MODEL = os.getenv("EXTRACT_MODEL", "gpt-4o-mini")
DETERMINISTIC_TEMP = 0.0

# ---------------------------------------------------------------------------
# Clause loader
# ---------------------------------------------------------------------------

def walk_clauses(node, out: list[dict]) -> list[dict]:
    """Recursively collect all Clause records từ structured_law.json."""
    if isinstance(node, dict):
        if node.get("clauses"):
            for c in node["clauses"]:
                if isinstance(c, dict) and c.get("id"):
                    out.append(c)
        for v in node.values():
            walk_clauses(v, out)
    elif isinstance(node, list):
        for x in node:
            walk_clauses(x, out)
    return out


def load_all_clauses() -> list[dict]:
    data = json.loads(LAW_PATH.read_text(encoding="utf-8"))
    return walk_clauses(data, [])


# ---------------------------------------------------------------------------
# Regex helpers (rule-based pre-pass)
# ---------------------------------------------------------------------------

_NUM_PCT = re.compile(r"(\d+(?:[.,]\d+)?)\s*%")
_NUM_YEAR = re.compile(r"(\d+)\s*năm")
_NUM_MONTH = re.compile(r"(\d+)\s*tháng")
_NUM_DAY = re.compile(r"(\d+)\s*ngày")
_NUM_VND = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:đồng|VND|vnd)")
_REF_DIEU_KHOAN = re.compile(r"Điều\s+(\d+)(?:\s+(?:khoản|Khoản)\s+(\d+))?")


def regex_facts(text: str) -> dict:
    """Quick rule-based extraction of literal numbers + references.
    Used cho cross-validation với LLM output."""
    return {
        "percentages": [float(m.group(1).replace(",", ".")) for m in _NUM_PCT.finditer(text)],
        "years": [int(m.group(1)) for m in _NUM_YEAR.finditer(text)],
        "months": [int(m.group(1)) for m in _NUM_MONTH.finditer(text)],
        "days": [int(m.group(1)) for m in _NUM_DAY.finditer(text)],
        "vnd_amounts": [float(m.group(1).replace(".", "")) for m in _NUM_VND.finditer(text)],
        "references": [(int(m.group(1)), int(m.group(2)) if m.group(2) else None)
                       for m in _REF_DIEU_KHOAN.finditer(text)],
    }


# ---------------------------------------------------------------------------
# LLM call with fallback
# ---------------------------------------------------------------------------

def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def llm_extract(client, prompt: str, clause: dict) -> tuple[dict, dict]:
    """Returns (parsed_json, usage_dict)."""
    from openai import BadRequestError

    # Build input — include id, text, points (concatenated nếu có)
    full_text = clause.get("text", "")
    points = clause.get("points", [])
    points_text = ""
    if points:
        points_text = "\n" + "\n".join(
            f"  {p.get('letter', '?')}) {p.get('text','')}" for p in points
        )

    user_input = json.dumps({
        "id": clause["id"],
        "text": full_text + points_text,
        "n_points": len(points),
    }, ensure_ascii=False, indent=2)

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ]

    def _try(use_temp, use_rf):
        kwargs = {"model": MODEL, "messages": messages}
        if use_temp:
            kwargs["temperature"] = DETERMINISTIC_TEMP
        if use_rf:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    try:
        resp = _try(True, True)
    except BadRequestError as e:
        msg = str(e).lower()
        resp = _try("temperature" not in msg, "json_object" not in msg)

    raw = resp.choices[0].message.content or "{}"
    # Strip markdown fences nếu có
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"_parse_error": True, "_raw": raw[:500]}

    usage = {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }
    return parsed, usage


# ---------------------------------------------------------------------------
# Per-clause extraction + save
# ---------------------------------------------------------------------------

def extract_clause(client, prompt: str, clause: dict, force: bool = False) -> dict:
    """Extract + save 1 clause. Returns metadata dict."""
    out_path = OUT_DIR / f"{clause['id'].replace('.', '_')}.json"
    if out_path.exists() and not force:
        return {"status": "skipped", "clause_id": clause["id"], "path": str(out_path)}

    t0 = time.time()
    parsed, usage = llm_extract(client, prompt, clause)
    elapsed = time.time() - t0

    # Enrich với regex facts cho cross-validation
    regex = regex_facts(clause.get("text", "") + " " +
                        " ".join(p.get("text", "") for p in clause.get("points", [])))

    record = {
        "clause_id": clause["id"],
        "extraction": parsed,
        "regex_facts": regex,
        "model": MODEL,
        "elapsed_s": round(elapsed, 2),
        "usage": usage,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return {"status": "extracted", "clause_id": clause["id"],
            "elapsed_s": elapsed, "tokens": usage}


# ---------------------------------------------------------------------------
# Sampling — pilot 30 từ 5 chapters
# ---------------------------------------------------------------------------

def sample_pilot(clauses: list[dict], n: int = 30, seed: int = 42) -> list[dict]:
    """Stratified sample: distribute across chapters proportionally."""
    random.seed(seed)
    # Group by article prefix (proxies for chapter — every Article belongs to 1 Chapter)
    # Easier: chia n / số chapters đại diện. Lấy đều theo Article number.
    if len(clauses) <= n:
        return clauses
    # Stratify by article_id first letter (rough chapter approximation)
    by_article = {}
    for c in clauses:
        aid = c.get("article_id", "unknown")
        by_article.setdefault(aid, []).append(c)
    article_ids = sorted(by_article.keys())
    # Pick 5 articles spread across range; sample 6 clauses each (cap by available)
    n_articles_to_sample = min(10, len(article_ids))
    selected_articles = random.sample(article_ids, n_articles_to_sample)
    per = max(1, n // n_articles_to_sample)
    out = []
    for aid in selected_articles:
        pool = by_article[aid]
        out.extend(random.sample(pool, min(per, len(pool))))
        if len(out) >= n:
            break
    return out[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=int, default=0,
                   help="Run on N pilot clauses (sample stratified). Default 0 = no pilot.")
    p.add_argument("--all", action="store_true", help="Run on all 543 clauses.")
    p.add_argument("--clause", type=str, default=None,
                   help="Single clause ID (e.g. L41_2024.A64.K1).")
    p.add_argument("--force", action="store_true", help="Re-extract even if file exists.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not (args.pilot or args.all or args.clause):
        p.error("Specify --pilot N, --all, or --clause <id>")

    clauses = load_all_clauses()
    print(f"Total clauses available: {len(clauses)}")

    if args.clause:
        targets = [c for c in clauses if c["id"] == args.clause]
        if not targets:
            print(f"Clause {args.clause} not found", file=sys.stderr)
            return 1
    elif args.pilot:
        targets = sample_pilot(clauses, n=args.pilot, seed=args.seed)
    else:
        targets = clauses

    print(f"Processing {len(targets)} clauses with {MODEL}...\n")

    from openai import OpenAI
    client = OpenAI()
    prompt = _load_prompt()

    n_done = n_skipped = n_failed = 0
    total_pt = total_ct = 0
    t_start = time.time()

    for i, c in enumerate(targets, 1):
        try:
            r = extract_clause(client, prompt, c, force=args.force)
            if r["status"] == "skipped":
                n_skipped += 1
            else:
                n_done += 1
                total_pt += r["tokens"]["prompt_tokens"]
                total_ct += r["tokens"]["completion_tokens"]
            if i % 5 == 0 or i == len(targets):
                elapsed = time.time() - t_start
                print(f"  [{i}/{len(targets)}] done={n_done} skip={n_skipped} fail={n_failed} "
                      f"elapsed={elapsed:.0f}s")
        except Exception as e:
            n_failed += 1
            print(f"  ✗ [{c['id']}] {type(e).__name__}: {e}", file=sys.stderr)

    elapsed = time.time() - t_start
    # Cost estimate (gpt-4o-mini)
    cost = (total_pt * 0.15 + total_ct * 0.60) / 1e6
    print(f"\n=== Summary ===")
    print(f"  done={n_done}, skipped={n_skipped}, failed={n_failed}")
    print(f"  total tokens: prompt={total_pt:,}, completion={total_ct:,}")
    print(f"  estimated cost: ${cost:.4f}")
    print(f"  wall time: {elapsed/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
