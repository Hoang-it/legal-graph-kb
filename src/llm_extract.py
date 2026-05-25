"""B3 — LLM extraction (OpenAI GPT-4o-mini).

Đọc data/interim/structured_law.json, gọi OpenAI per-Article để trích thực
thể và quan hệ ngữ nghĩa. Output: data/interim/llm_extractions/A{n}.json
(1 file/Article, idempotent — skip nếu đã có).

NGUYÊN TẮC PROVENANCE (không bịa):
1. Prompt cấm bịa, ép mỗi quan hệ có `source_clause` + `source_text`.
2. Sau khi LLM trả về, validate cứng:
   - `source_clause` phải là Clause.id / Point.id thuộc đúng Article này.
   - `source_text` phải là substring nguyên văn của text(source_clause).
   - Edge nào không pass → loại bỏ + log (KHÔNG đưa vào output).
3. Entity nào có `mentioned_in` chứa Clause không thuộc Article này → loại.

Concurrency: asyncio.Semaphore (env OPENAI_CONCURRENCY, mặc định 5).
Retry: tenacity với exponential backoff trên RateLimitError / APIError.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import APIError, AsyncOpenAI, RateLimitError
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.schema import LLMArticleExtraction

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
CONCURRENCY = int(os.getenv("OPENAI_CONCURRENCY", "5"))
BASE_URL = os.getenv("OPENAI_BASE_URL") or None

PROMPTS_DIR = Path("prompts")
STRUCTURED_PATH = Path("data/interim/structured_law.json")
OUT_DIR = Path("data/interim/llm_extractions")


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def _load_system_prompt() -> str:
    """Đọc phần SYSTEM từ prompts/extract_v1.md."""
    md = (PROMPTS_DIR / "extract_v1.md").read_text(encoding="utf-8")
    # Phần SYSTEM kết thúc khi gặp "# USER"
    start = md.find("# SYSTEM")
    end = md.find("# USER")
    if start == -1 or end == -1:
        raise RuntimeError("prompts/extract_v1.md thiếu header # SYSTEM / # USER")
    return md[start + len("# SYSTEM") : end].strip()


SYSTEM_PROMPT = _load_system_prompt()


# ---------------------------------------------------------------------------
# Build user prompt cho 1 Article
# ---------------------------------------------------------------------------


def build_user_prompt(art: dict, chapter: dict, section: dict | None) -> str:
    lines = [
        f"ARTICLE_HEADER: Điều {art['number']}. {art['title']}",
        f"CHAPTER: Chương {chapter['roman']} — {chapter['title']}",
    ]
    if section:
        lines.append(f"SECTION: Mục {section['number']} — {section['title']}")

    lines.append("")
    lines.append("# CLAUSES")
    for cl in art["clauses"]:
        lines.append(f"[{cl['id']}] {cl['text']}")
        for pt in cl["points"]:
            lines.append(f"  [{pt['id']}] ({pt['letter']}) {pt['text']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OpenAI call
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# JSON Schema strict (cho OpenAI response_format)
# ---------------------------------------------------------------------------
# OpenAI strict mode: tất cả field phải required, additionalProperties=false,
# không default. Ta sinh schema bằng tay để kiểm soát hoàn toàn.

_SEMANTIC_BASE_PROPS = {
    "id": {"type": "string"},
    "mentioned_in": {"type": "array", "items": {"type": "string"}, "minItems": 1},
}


def _make_entity_schema(extra_props: dict) -> dict:
    props = {**_SEMANTIC_BASE_PROPS, **extra_props}
    return {
        "type": "object",
        "properties": props,
        "required": list(props.keys()),
        "additionalProperties": False,
    }


EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "article_id": {"type": "string"},
        "concepts": {
            "type": "array",
            "items": _make_entity_schema(
                {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                    "defined_in": {"type": "string"},
                }
            ),
        },
        "subjects": {
            "type": "array",
            "items": _make_entity_schema(
                {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["individual", "organization", "role", "group", "other"],
                    },
                }
            ),
        },
        "organizations": {
            "type": "array",
            "items": _make_entity_schema(
                {
                    "name": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": [
                            "state_agency",
                            "ministry",
                            "social_insurance_agency",
                            "employer",
                            "fund_manager",
                            "other",
                        ],
                    },
                }
            ),
        },
        "roles": {
            "type": "array",
            "items": _make_entity_schema({"name": {"type": "string"}}),
        },
        "benefits": {
            "type": "array",
            "items": _make_entity_schema(
                {
                    "name": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "huu_tri",
                            "om_dau",
                            "thai_san",
                            "tu_tuat",
                            "tnld_bnn",
                            "tro_cap_huu_tri_xa_hoi",
                            "bhxh_tu_nguyen",
                            "bhht_bo_sung",
                            "khac",
                        ],
                    },
                }
            ),
        },
        "conditions": {
            "type": "array",
            "items": _make_entity_schema({"description": {"type": "string"}}),
        },
        "obligations": {
            "type": "array",
            "items": _make_entity_schema({"description": {"type": "string"}}),
        },
        "rights": {
            "type": "array",
            "items": _make_entity_schema({"description": {"type": "string"}}),
        },
        "prohibited_acts": {
            "type": "array",
            "items": _make_entity_schema({"description": {"type": "string"}}),
        },
        "funds": {
            "type": "array",
            "items": _make_entity_schema({"name": {"type": "string"}}),
        },
        "semantic_edges": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "DEFINES",
                            "ENTITLED_TO",
                            "HAS_OBLIGATION",
                            "HAS_RIGHT",
                            "APPLIES_TO",
                            "REQUIRES",
                            "PAID_FROM",
                            "MANAGES",
                            "RESPONSIBLE_FOR",
                            "PROHIBITED_BY",
                        ],
                    },
                    "src": {"type": "string"},
                    "dst": {"type": "string"},
                    "source_clause": {"type": "string"},
                    "source_text": {"type": "string", "maxLength": 300},
                },
                "required": ["type", "src", "dst", "source_clause", "source_text"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "article_id",
        "concepts",
        "subjects",
        "organizations",
        "roles",
        "benefits",
        "conditions",
        "obligations",
        "rights",
        "prohibited_acts",
        "funds",
        "semantic_edges",
    ],
    "additionalProperties": False,
}


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
async def call_openai(client: AsyncOpenAI, user_prompt: str) -> dict:
    """Gọi OpenAI với JSON Schema strict mode → ép cấu trúc output."""
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "LLMArticleExtraction",
                "strict": True,
                "schema": EXTRACTION_JSON_SCHEMA,
            },
        },
        temperature=0,
        seed=42,
    )
    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI trả về content rỗng")
    return {
        "raw_json": content,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "cached_tokens": getattr(resp.usage.prompt_tokens_details, "cached_tokens", 0)
            if resp.usage.prompt_tokens_details
            else 0,
        },
    }


# ---------------------------------------------------------------------------
# Validation — quy tắc PROVENANCE
# ---------------------------------------------------------------------------


def _build_article_unit_index(art: dict) -> dict[str, str]:
    """{clause_id|point_id: text} cho đúng Article này (KHÔNG bao gồm Article khác).

    Edge nào trỏ source_clause ra ngoài Article hiện tại sẽ bị loại bỏ.
    """
    idx: dict[str, str] = {}
    for cl in art["clauses"]:
        idx[cl["id"]] = cl["text"]
        for pt in cl["points"]:
            idx[pt["id"]] = pt["text"]
    return idx


def validate_extraction(raw: dict, art: dict) -> tuple[LLMArticleExtraction, dict]:
    """Parse + validate. Trả về (extraction_đã_lọc, drop_stats).

    Lọc bỏ:
    - Edge có source_clause không thuộc Article này.
    - Edge có source_text không phải substring của text source_clause.
    - Entity có mentioned_in chứa ID không thuộc Article này.
    """
    art_id = art["id"]
    raw["article_id"] = art_id  # ép article_id đúng (LLM có thể quên)
    unit_index = _build_article_unit_index(art)

    # ---- Bước 1: Pydantic validate cấu trúc ----
    try:
        ext = LLMArticleExtraction.model_validate(raw)
    except ValidationError as e:
        # Cố gắng vớt vát: chỉ giữ edge hợp lệ trước khi validate lại
        edges = raw.get("semantic_edges", [])
        kept_edges = []
        for ed in edges:
            # Kiểm tra nhanh source_clause
            sc = ed.get("source_clause", "")
            if sc not in unit_index:
                continue
            st = ed.get("source_text", "")
            if not st or st not in unit_index[sc]:
                continue
            if len(st) > 300:
                ed["source_text"] = st[:300]
            kept_edges.append(ed)
        raw["semantic_edges"] = kept_edges
        # Lọc mentioned_in của entities về only IDs thuộc Article này
        for key in [
            "concepts",
            "subjects",
            "organizations",
            "roles",
            "benefits",
            "conditions",
            "obligations",
            "rights",
            "prohibited_acts",
            "funds",
        ]:
            for ent in raw.get(key, []):
                mi = [m for m in ent.get("mentioned_in", []) if m in unit_index]
                # mentioned_in không được rỗng theo schema; nếu rỗng → mark None
                ent["mentioned_in"] = (
                    mi if mi else [art["clauses"][0]["id"]] if art["clauses"] else []
                )
            raw[key] = [e for e in raw.get(key, []) if e.get("mentioned_in")]
        try:
            ext = LLMArticleExtraction.model_validate(raw)
        except ValidationError as e2:
            raise RuntimeError(f"Vẫn không validate được sau khi lọc: {e2}") from e

    # ---- Bước 2: lọc edge theo provenance ----
    n_edges_before = len(ext.semantic_edges)
    drop_invalid_source = 0
    drop_text_mismatch = 0
    kept_edges = []
    for ed in ext.semantic_edges:
        if ed.source_clause not in unit_index:
            drop_invalid_source += 1
            continue
        if ed.source_text not in unit_index[ed.source_clause]:
            drop_text_mismatch += 1
            continue
        kept_edges.append(ed)
    ext.semantic_edges = kept_edges

    # ---- Bước 3: lọc entities có mentioned_in ngoài Article ----
    drop_entity_outside = 0
    for attr in [
        "concepts",
        "subjects",
        "organizations",
        "roles",
        "benefits",
        "conditions",
        "obligations",
        "rights",
        "prohibited_acts",
        "funds",
    ]:
        entities = getattr(ext, attr)
        filtered = []
        for ent in entities:
            mi_in_article = [m for m in ent.mentioned_in if m in unit_index]
            if not mi_in_article:
                drop_entity_outside += 1
                continue
            ent.mentioned_in = mi_in_article
            filtered.append(ent)
        setattr(ext, attr, filtered)

    return ext, {
        "edges_total": n_edges_before,
        "edges_kept": len(kept_edges),
        "drop_invalid_source": drop_invalid_source,
        "drop_text_mismatch": drop_text_mismatch,
        "drop_entity_outside_article": drop_entity_outside,
    }


# ---------------------------------------------------------------------------
# Per-article extraction (idempotent)
# ---------------------------------------------------------------------------


async def extract_article(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    art: dict,
    chapter: dict,
    section: dict | None,
    skip_existing: bool = True,
) -> dict | None:
    """Trả về dict {article_id, ok, extraction, drop_stats, usage, elapsed_s, error?}
    hoặc None nếu skip."""
    art_id = art["id"]
    art_n = art["number"]
    out_path = OUT_DIR / f"A{art_n}.json"

    if skip_existing and out_path.exists():
        return None

    if not art["clauses"]:
        # Không có clause → không thể anchor edge. Lưu file trống để skip lần sau.
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "article_id": art_id,
                    "skipped_reason": "no_clauses (lead_text only)",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"article_id": art_id, "ok": True, "skipped": True}

    if art_n == 141:
        # Điều 141 chuyển tiếp — phức tạp, skip ở phase này theo prompt rule
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "article_id": art_id,
                    "skipped_reason": "transitional clauses (Điều 141) — handled separately",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return {"article_id": art_id, "ok": True, "skipped": True}

    user_prompt = build_user_prompt(art, chapter, section)

    async with sem:
        t0 = time.monotonic()
        try:
            result = await call_openai(client, user_prompt)
            raw_data = json.loads(result["raw_json"])
            ext, drop_stats = validate_extraction(raw_data, art)
        except Exception as e:
            elapsed = time.monotonic() - t0
            return {
                "article_id": art_id,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(elapsed, 2),
            }
        elapsed = time.monotonic() - t0

    out_data = {
        "article_id": art_id,
        "extraction": ext.model_dump(),
        "drop_stats": drop_stats,
        "usage": result["usage"],
        "elapsed_s": round(elapsed, 2),
        "model": MODEL,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "article_id": art_id,
        "ok": True,
        "n_edges": len(ext.semantic_edges),
        "drop_stats": drop_stats,
        "usage": result["usage"],
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(
    articles_to_process: list[int] | None = None,
    skip_existing: bool = True,
) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("FAIL: thiếu OPENAI_API_KEY trong .env", file=sys.stderr)
        return 1

    if not STRUCTURED_PATH.exists():
        print(f"FAIL: không có {STRUCTURED_PATH}. Chạy B1 trước.", file=sys.stderr)
        return 1

    with STRUCTURED_PATH.open(encoding="utf-8") as f:
        structured = json.load(f)

    # Build (article, chapter, section) tuples
    sec_map: dict[str, dict] = {}
    for ch in structured["chapters"]:
        for sec in ch["sections"]:
            sec_map[sec["id"]] = sec

    tasks_input: list[tuple[dict, dict, dict | None]] = []
    for ch in structured["chapters"]:
        for art in ch["articles"]:
            if articles_to_process and art["number"] not in articles_to_process:
                continue
            sec = sec_map.get(art.get("section_id") or "")
            tasks_input.append((art, ch, sec))

    print(
        f"Chuẩn bị extract {len(tasks_input)} Article(s) "
        f"với model={MODEL}, concurrency={CONCURRENCY}, skip_existing={skip_existing}"
    )

    client = AsyncOpenAI(base_url=BASE_URL) if BASE_URL else AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)

    t0 = time.monotonic()
    coros = [
        extract_article(client, sem, art, ch, sec, skip_existing=skip_existing)
        for art, ch, sec in tasks_input
    ]

    results: list[dict] = []
    # SIM113: enumerate không dùng được vì as_completed trả Future, không phải iterable enumerable theo thứ tự
    for n_done, coro in enumerate(asyncio.as_completed(coros), start=1):
        r = await coro
        if r is not None:
            results.append(r)
        if n_done % 10 == 0 or n_done == len(coros):
            elapsed = time.monotonic() - t0
            print(f"  [{n_done:>3}/{len(coros)}] {elapsed:6.1f}s elapsed")

    elapsed = time.monotonic() - t0
    print(f"\nXong trong {elapsed:.1f}s")

    # Stats
    ok = [r for r in results if r.get("ok") and not r.get("skipped")]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if not r.get("ok")]

    print("\n=== STATS ===")
    print(f"  Thành công    : {len(ok)}")
    print(f"  Đã skip       : {len(skipped)}")
    print(f"  Thất bại      : {len(failed)}")

    if ok:
        total_edges = sum(r["n_edges"] for r in ok)
        total_dropped = sum(
            r["drop_stats"]["drop_invalid_source"] + r["drop_stats"]["drop_text_mismatch"]
            for r in ok
        )
        total_input_tokens = sum(r["usage"]["prompt_tokens"] for r in ok)
        total_cached = sum(r["usage"].get("cached_tokens", 0) for r in ok)
        total_output_tokens = sum(r["usage"]["completion_tokens"] for r in ok)
        # Pricing GPT-4o-mini: input $0.15/M, cached $0.075/M, output $0.60/M
        cost = (
            (total_input_tokens - total_cached) * 0.15 / 1e6
            + total_cached * 0.075 / 1e6
            + total_output_tokens * 0.60 / 1e6
        )
        print(f"  Tổng edges    : {total_edges} (loại {total_dropped} vì provenance sai)")
        print(f"  Input tokens  : {total_input_tokens:,} ({total_cached:,} cached)")
        print(f"  Output tokens : {total_output_tokens:,}")
        print(f"  Chi phí ước   : ${cost:.4f}")

    if failed:
        print("\n  ✗ Article thất bại:")
        for r in failed[:10]:
            print(f"    {r['article_id']}: {r['error']}")

    return 0 if not failed else 2


def main():
    parser = argparse.ArgumentParser(description="B3 — LLM extraction (OpenAI)")
    parser.add_argument(
        "--articles",
        type=str,
        default="",
        help="CSV danh sách số Article cần xử lý (vd '1,64,140'). Để trống = tất cả.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bỏ qua file cache, gọi LLM lại cho mọi Article.",
    )
    args = parser.parse_args()
    to_process = (
        [int(x.strip()) for x in args.articles.split(",") if x.strip()] if args.articles else None
    )
    return asyncio.run(run(to_process, skip_existing=not args.force))


if __name__ == "__main__":
    sys.exit(main())
