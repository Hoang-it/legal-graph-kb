"""rerender_plain_answer.py — Backfill plain_answer field cho elite records cũ.

Re-render IRAC using new prompt (experiments/prompts/irac_with_plain.md) which
outputs JSON with BOTH `irac` (preserves analysis) AND `plain_answer` (prose form
suitable cho fair compare với prose arms).

**1 LLM call per record** — input = existing record's (question, IRAC text,
citations) as trace; output = JSON với plain_answer extracted.

Skips:
- Records with empty IRAC (prolog failed, no render happened originally)
- Records already có plain_answer non-empty (idempotent)

Usage:
    python -m experiments.rerender_plain_answer --combos all --pilot 10
    python -m experiments.rerender_plain_answer --combos all  # full

Cost (gpt-4o-mini backfill model, không phải original inference model):
    ~$0.002 per record × ~1700 elite records ≈ $3.4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

# Map of combos → result directory
R1_COMBOS = {
    "elite_no_retrieval": Path("data/eval/results/elite_no_retrieval"),
    "elite_ontology":     Path("data/eval/results/elite_ontology"),
    "elite_graphrag":     Path("data/eval/results/elite_graphrag"),
}
R2_COMBOS = {
    "elite_no_retrieval__gpt-4_1":    Path("data/eval/multimodel/results/elite_no_retrieval__gpt-4_1"),
    "elite_no_retrieval__gpt-4o":     Path("data/eval/multimodel/results/elite_no_retrieval__gpt-4o"),
    "elite_no_retrieval__gpt-5-mini": Path("data/eval/multimodel/results/elite_no_retrieval__gpt-5-mini"),
    "elite_graphrag__gpt-4_1":        Path("data/eval/multimodel/results/elite_graphrag__gpt-4_1"),
    "elite_graphrag__gpt-4o":         Path("data/eval/multimodel/results/elite_graphrag__gpt-4o"),
    "elite_graphrag__gpt-5-mini":     Path("data/eval/multimodel/results/elite_graphrag__gpt-5-mini"),
}
ALL_COMBOS = {**R1_COMBOS, **R2_COMBOS}

PROMPT_PATH = Path("experiments/prompts/irac_with_plain.md")
BACKFILL_MODEL = os.getenv("BACKFILL_MODEL", "gpt-4o-mini")


def _load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _build_trace_from_record(rec: dict) -> dict:
    """Reconstruct minimal trace payload từ existing record fields.

    Original render dùng full envelope + result. Backfill chỉ có IRAC + citations
    + trace text — đủ để LLM produce plain_answer dựa trên IRAC analysis sẵn có.
    """
    return {
        "normalized_question": rec.get("question", ""),
        "legal_issue": rec.get("question", ""),
        "domain_context": "Vietnam Social Insurance Law (BHXH)",
        "selected_function": {
            "name": "render_from_existing_irac",
            "description": "Backfill plain_answer from already-rendered IRAC",
        },
        "slot_bindings": {},
        "verify_facts": [],
        "citations": [
            {
                "index": i,
                "document": "Luật BHXH 2024 (41/2024/QH15)",
                "article": None,
                "clause": None,
                "raw_text": cit,
            }
            for i, cit in enumerate(rec.get("citations", []))
        ],
        "execution_result": [],
        "prolog_trace": rec.get("prolog_trace", ""),
        "generated_program": {},
        # Pass existing IRAC text để LLM có thể base plain_answer trên đó
        "existing_irac": rec.get("answer", ""),
    }


def _parse_response(text: str) -> tuple[str, str]:
    """Parse JSON response → (irac, plain_answer). Tolerant of markdown fences."""
    if not text:
        return "", ""
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```\s*$", "", clean)
    try:
        parsed = json.loads(clean)
        if isinstance(parsed, dict):
            return (parsed.get("irac") or "").strip(), (parsed.get("plain_answer") or "").strip()
    except json.JSONDecodeError:
        pass
    # Regex fallback
    m_plain = re.search(r'"plain_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)
    m_irac = re.search(r'"irac"\s*:\s*"((?:[^"\\]|\\.)*)"', clean, re.DOTALL)
    irac = m_irac.group(1).encode().decode("unicode_escape") if m_irac else text
    plain = m_plain.group(1).encode().decode("unicode_escape") if m_plain else ""
    return irac.strip(), plain.strip()


def _call_llm(client, prompt: str, trace: dict, model: str) -> tuple[str, dict]:
    """One API call → return (raw_text, usage_dict). With reasoning-model fallback."""
    from openai import BadRequestError

    user_msg = json.dumps(trace, ensure_ascii=False, indent=2, default=str)
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_msg},
    ]

    def _try(use_temp: bool, use_rf: bool):
        kwargs = {"model": model, "messages": messages}
        if use_temp:
            kwargs["temperature"] = 0.0
        if use_rf:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    # Try with both, fallback if rejected
    try:
        resp = _try(use_temp=True, use_rf=True)
    except BadRequestError as e:
        msg = str(e).lower()
        try:
            resp = _try(
                use_temp="temperature" not in msg,
                use_rf=not ("response_format" in msg or "json_object" in msg),
            )
        except BadRequestError:
            resp = _try(use_temp=False, use_rf=False)
    text = resp.choices[0].message.content or ""
    return text, {
        "prompt_tokens": resp.usage.prompt_tokens,
        "completion_tokens": resp.usage.completion_tokens,
    }


def process_combo(combo: str, results_dir: Path, prompt: str, model: str,
                  pilot: int | None, force: bool, client, verbose: bool) -> dict:
    files = sorted(results_dir.glob("A*.json"))
    files = [f for f in files if not f.name.endswith(".error.json")]
    if pilot:
        files = files[:pilot]

    n_skipped = n_done = n_failed = 0
    total_pt = total_ct = 0
    t0 = time.time()

    for i, fp in enumerate(files, 1):
        try:
            rec = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            n_failed += 1
            continue

        # Skip if no IRAC was rendered (Prolog failed) or already has plain_answer
        irac_text = rec.get("answer", "") or ""
        if not irac_text or irac_text.startswith("[Pipeline không trả về"):
            n_skipped += 1
            continue
        if rec.get("plain_answer") and not force:
            n_skipped += 1
            continue
        if rec.get("api_error"):
            n_skipped += 1
            continue

        trace = _build_trace_from_record(rec)
        try:
            text, usage = _call_llm(client, prompt, trace, model)
            new_irac, plain = _parse_response(text)
            if not plain:
                # Couldn't extract — count as failed but save raw response for inspection
                rec["plain_answer"] = ""
                rec["plain_answer_raw_response"] = text[:1000]
                n_failed += 1
            else:
                rec["plain_answer"] = plain
                # Update IRAC nếu LLM produced fresh version (cleaner)
                if new_irac and len(new_irac) > 50:
                    rec["answer_v2_irac"] = new_irac  # don't overwrite original; store separate
                n_done += 1
            rec["plain_answer_tokens"] = usage
            total_pt += usage["prompt_tokens"]
            total_ct += usage["completion_tokens"]
            fp.write_text(json.dumps(rec, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            if verbose or i % 25 == 0:
                elapsed = time.time() - t0
                print(f"  [{combo:<32} {i:>3}/{len(files)}] done={n_done} "
                      f"skip={n_skipped} fail={n_failed} "
                      f"(elapsed {elapsed:.0f}s)", flush=True)
        except Exception as e:
            n_failed += 1
            if verbose:
                print(f"  ! [{combo} stt={rec.get('stt')}] {type(e).__name__}: {e}",
                      file=sys.stderr)

    elapsed = time.time() - t0
    return {
        "combo": combo,
        "n_total": len(files),
        "n_done": n_done,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "elapsed_s": round(elapsed, 1),
        "total_prompt_tokens": total_pt,
        "total_completion_tokens": total_ct,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill plain_answer cho elite records.")
    p.add_argument("--combos", type=str, default="all",
                   help=f"Comma-separated combo names hoặc 'all'. Available: {list(ALL_COMBOS)}")
    p.add_argument("--pilot", type=int, default=0, help="Chỉ chạy N records đầu (debug)")
    p.add_argument("--force", action="store_true", help="Re-do dù plain_answer đã có")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--model", type=str, default=BACKFILL_MODEL,
                   help=f"Model dùng backfill (default {BACKFILL_MODEL}). Có thể override với env BACKFILL_MODEL.")
    args = p.parse_args()

    if args.combos == "all":
        combos = list(ALL_COMBOS)
    else:
        combos = [c.strip() for c in args.combos.split(",") if c.strip()]
    invalid = [c for c in combos if c not in ALL_COMBOS]
    if invalid:
        raise SystemExit(f"Unknown combo(s): {invalid}")

    prompt = _load_prompt()
    from openai import OpenAI
    client = OpenAI()

    print(f"Backfill model: {args.model}")
    print(f"Combos: {combos}")
    print(f"Pilot: {args.pilot or 'full'}\n")

    summaries = []
    for c in combos:
        print(f"=== {c} ===")
        s = process_combo(c, ALL_COMBOS[c], prompt, args.model,
                          args.pilot or None, args.force, client, args.verbose)
        summaries.append(s)
        print(f"  Summary: {s}\n")

    print("=== TOTAL ===")
    tot_done = sum(s["n_done"] for s in summaries)
    tot_skip = sum(s["n_skipped"] for s in summaries)
    tot_fail = sum(s["n_failed"] for s in summaries)
    tot_pt = sum(s["total_prompt_tokens"] for s in summaries)
    tot_ct = sum(s["total_completion_tokens"] for s in summaries)
    # Cost estimate for gpt-4o-mini
    cost = (tot_pt * 0.15 + tot_ct * 0.60) / 1e6 if args.model == "gpt-4o-mini" else None
    print(f"  done={tot_done} skipped={tot_skip} failed={tot_fail}")
    print(f"  tokens: prompt={tot_pt:,}, completion={tot_ct:,}")
    if cost is not None:
        print(f"  estimated cost (gpt-4o-mini): ${cost:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
