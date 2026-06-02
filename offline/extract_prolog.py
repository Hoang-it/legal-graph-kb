"""Extract direct Prolog rules from structured law clauses using OpenAI."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from offline.validate_prolog import validate_record
from src.legal_metadata import load_order
from src.prolog_utils import parse_json_object

load_dotenv()
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

MODEL = os.getenv("PROLOG_EXTRACT_MODEL", "gpt-4o")
PROMPT_PATH = Path("prompts/offline/prolog_extraction_v1.md")
OUT_ROOT = Path("data/eval/extracted_prolog")


def walk_clauses(structured: dict) -> list[dict[str, Any]]:
    out = []
    for ch in structured.get("chapters", []):
        for art in ch.get("articles", []):
            for clause in art.get("clauses", []):
                item = dict(clause)
                item["article_title"] = art.get("title")
                item["article_number"] = art.get("number")
                item["law_code"] = structured.get("law", {}).get("id")
                out.append(item)
    return out


def clause_payload(clause: dict[str, Any]) -> dict[str, Any]:
    points = clause.get("points") or []
    text = clause.get("text") or ""
    if points:
        text += "\n" + "\n".join(f"{p.get('letter')}) {p.get('text')}" for p in points)
    return {
        "clause_id": clause["id"],
        "law_code": clause["law_code"],
        "article_number": clause.get("article_number"),
        "article_title": clause.get("article_title"),
        "clause_number": clause.get("number"),
        "text": text,
    }


def load_clauses(law_code: str) -> list[dict[str, Any]]:
    path = Path(f"data/graph/interim/structured_law_{law_code}.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    return walk_clauses(data)


def output_path(clause_id: str) -> Path:
    law_code = clause_id.split(".")[0]
    return OUT_ROOT / law_code / f"{clause_id.replace('.', '_')}.json"


def _usage(resp) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
    }


def call_openai(client, prompt: str, payload: dict[str, Any], repair: dict[str, Any] | None = None):
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
    if repair:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous JSON failed SWI-Prolog validation. Repair it once.\n"
                    "Common fixes: every unquoted atom must start with a lowercase letter; "
                    "replace atoms such as `5_years` with `years_5`; "
                    "list cons has one tail only, so use `[source, Trace1, Trace2]` "
                    "instead of `[source | Trace1, Trace2]`; ensure commas and periods are valid.\n"
                    f"Validation error:\n{repair.get('diagnostic')}\n\n"
                    f"Previous JSON:\n{json.dumps(repair.get('previous'), ensure_ascii=False)}"
                ),
            }
        )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return parse_json_object(resp.choices[0].message.content or "{}"), _usage(resp)


def extract_clause(client, prompt: str, clause: dict[str, Any], force: bool = False) -> dict[str, Any]:
    out_path = output_path(clause["id"])
    if out_path.exists() and not force:
        return {"status": "skipped", "clause_id": clause["id"], "path": str(out_path)}

    payload = clause_payload(clause)
    t0 = time.time()
    extraction, usage = call_openai(client, prompt, payload)
    record = {
        "status": "extracted",
        "clause_id": clause["id"],
        "law_code": clause["law_code"],
        "model": MODEL,
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_payload": payload,
        "extraction": extraction,
        "usage": usage,
    }
    validation = validate_record(record)
    record["validation"] = validation

    if not validation.get("ok"):
        repaired, repair_usage = call_openai(
            client,
            prompt,
            payload,
            repair={"diagnostic": validation.get("diagnostic"), "previous": extraction},
        )
        repaired_record = {**record, "extraction": repaired}
        repair_validation = validate_record(repaired_record)
        record["repair"] = {
            "attempted": True,
            "usage": repair_usage,
            "validation": repair_validation,
        }
        record["usage"] = {
            "prompt_tokens": usage["prompt_tokens"] + repair_usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"] + repair_usage["completion_tokens"],
        }
        if repair_validation.get("ok"):
            record["extraction"] = repaired
            record["validation"] = repair_validation

    record["status"] = "validated" if record["validation"].get("ok") else "invalid"
    record["elapsed_s"] = round(time.time() - t0, 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": record["status"],
        "clause_id": clause["id"],
        "elapsed_s": record["elapsed_s"],
        "path": str(out_path),
    }


def select_clauses(clauses: list[dict[str, Any]], args) -> list[dict[str, Any]]:
    if args.clause:
        return [c for c in clauses if c["id"] == args.clause]
    if args.pilot:
        rng = random.Random(args.seed)
        sample = list(clauses)
        rng.shuffle(sample)
        return sample[: args.pilot]
    if args.offset:
        clauses = clauses[args.offset :]
    if args.limit:
        return clauses[: args.limit]
    return clauses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--law", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--pilot", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--clause", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("FAIL: OPENAI_API_KEY is required for real Prolog extraction")

    from openai import OpenAI

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    client = OpenAI()
    laws = load_order() if args.all else [args.law or "L41_2024"]

    total = {"validated": 0, "invalid": 0, "skipped": 0}
    for law_code in laws:
        clauses = select_clauses(load_clauses(law_code), args)
        print(f"{law_code}: extracting {len(clauses)} clauses with {MODEL}")
        for i, clause in enumerate(clauses, 1):
            result = extract_clause(client, prompt, clause, force=args.force)
            total[result["status"]] = total.get(result["status"], 0) + 1
            if i % 10 == 0 or result["status"] == "invalid":
                print(f"  {i}/{len(clauses)} {result['clause_id']} {result['status']}")
    print(json.dumps(total, ensure_ascii=False, indent=2))
    return 0 if not total.get("invalid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
