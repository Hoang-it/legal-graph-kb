"""Phase 1 of v5 Sprint 2 — synthetic Q/clause training data for BGE-M3 LoRA fine-tune.

Pipeline for each clause in the KG (1585 total):

1. **Query generation** — ``gpt-4o-mini`` reads clause text and emits 2 natural
   queries (direct + with-context). Prompt: ``prompts/offline/synthetic_query_gen.md``.
2. **Candidate harvest** — Cypher fetches up to 8 sibling clauses (same Article
   first, then same Chapter as fallback). The seed clause itself becomes the
   primary positive.
3. **Distance filter** — vanilla BGE-M3 cosine. Candidates with sim < 0.3
   (uninformative trivial negatives) or > 0.85 (likely false negatives — too
   similar to the positive) are dropped.
4. **LLM verifier** — ``gpt-4o-mini`` labels each surviving candidate as
   YES / PARTIAL / NO. Prompt: ``prompts/offline/synthetic_pair_verifier.md``.
   YES → additional positive (multi-positive). NO → hard negative.
   PARTIAL → dropped (false-negative protection).
5. **Easy-negative augmentation** — 3 random clauses from distant chapters
   appended to ``neg`` for in-batch negative diversity.
6. Row written to ``data/finetune-bge/qa_pairs_v1.jsonl``.

Idempotent — resumes from where it left off using the
``data/finetune-bge/processed_clause_ids.txt`` marker file.

Eval-leak invariants (enforced at module top):
- This script reads only ``data/graph/processed/merged_graph.json``.
- It does NOT read ``data/eval/questions_*.json``.
- The leak protection is enforced by file path; modifying this script to read
  eval files is a Rule-2 violation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Eval-leak invariant: this script reads only `data/graph/processed/merged_graph.json`.
# It MUST NOT open any file under `data/eval/`. Adding such an open() is a
# Rule-2 (skill) violation — review every diff that touches GRAPH_PATH / OUT_DIR.

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.prompts import load_prompt

load_dotenv()
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)

OPENAI_MODEL = os.getenv("SYNTHETIC_MODEL", "gpt-4o-mini")
CONCURRENCY = int(os.getenv("SYNTHETIC_CONCURRENCY", "5"))

GRAPH_PATH = Path("data/graph/processed/merged_graph.json")
OUT_DIR = Path("data/finetune-bge")
OUT_JSONL = OUT_DIR / "qa_pairs_v1.jsonl"
PROCESSED_MARKER = OUT_DIR / "processed_clause_ids.txt"
STATS_PATH = OUT_DIR / "stats.json"
GEN_META_PATH = OUT_DIR / "generation_metadata.json"

QUERY_GEN_PROMPT_REL = "offline/synthetic_query_gen.md"
VERIFIER_PROMPT_REL = "offline/synthetic_pair_verifier.md"

# Distance filter thresholds (Layer 3 of false-negative defense)
SIM_MIN = 0.30
SIM_MAX = 0.85

# Candidate budget per query (user decision: 8 not 15)
N_CANDIDATES_PER_QUERY = 8
N_EASY_NEGATIVES = 3


# ---------------------------------------------------------------------------
# Prompt parsing — same SYSTEM/USER convention as offline/llm_extract.py
# ---------------------------------------------------------------------------

def _split_prompt(md: str) -> tuple[str, str]:
    s_start = md.find("# SYSTEM")
    u_start = md.find("# USER")
    if s_start == -1 or u_start == -1:
        raise RuntimeError("Prompt must contain # SYSTEM and # USER sections")
    return md[s_start + len("# SYSTEM"): u_start].strip(), md[u_start + len("# USER"):].strip()


QUERY_GEN_SYSTEM, QUERY_GEN_USER_TMPL = _split_prompt(load_prompt(QUERY_GEN_PROMPT_REL))
VERIFIER_SYSTEM, VERIFIER_USER_TMPL = _split_prompt(load_prompt(VERIFIER_PROMPT_REL))


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Clause:
    id: str
    text: str
    article_id: str
    article_number: int
    chapter_id: str
    law_id: str


@dataclass
class RunStats:
    n_clauses_total: int = 0
    n_clauses_processed: int = 0
    n_clauses_skipped_empty: int = 0
    n_queries_emitted: int = 0
    n_candidates_total: int = 0
    n_dropped_by_distance: int = 0
    n_dropped_partial: int = 0
    n_kept_yes: int = 0
    n_kept_no: int = 0
    n_rows_written: int = 0
    n_multi_positive_rows: int = 0
    sum_pos_per_row: int = 0
    sum_neg_per_row: int = 0
    api_calls_query_gen: int = 0
    api_calls_verifier: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        if self.n_rows_written:
            d["mean_pos_per_row"] = round(self.sum_pos_per_row / self.n_rows_written, 3)
            d["mean_neg_per_row"] = round(self.sum_neg_per_row / self.n_rows_written, 3)
            d["pct_multi_positive"] = round(
                100 * self.n_multi_positive_rows / self.n_rows_written, 1
            )
        return d


# ---------------------------------------------------------------------------
# KG loading + candidate harvesting
# ---------------------------------------------------------------------------


def load_clauses() -> list[Clause]:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"{GRAPH_PATH} not found. Run offline B1-B4 first.")
    g = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    article_by_id = {a["id"]: a for a in g["nodes"].get("Article", [])}
    out: list[Clause] = []
    for cl in g["nodes"].get("Clause", []):
        art = article_by_id.get(cl.get("article_id") or "")
        if art is None:
            continue
        text = (cl.get("text") or "").strip()
        if not text:
            continue
        out.append(Clause(
            id=cl["id"],
            text=text,
            article_id=art["id"],
            article_number=int(art.get("number") or 0),
            chapter_id=str(art.get("chapter_id") or ""),
            law_id=str(cl.get("law_code") or art.get("law_code") or "").upper(),
        ))
    return out


def candidates_by_proximity(target: Clause, all_clauses: list[Clause]) -> list[Clause]:
    """Same Article first, then same Chapter, capped at N_CANDIDATES_PER_QUERY.

    No LLM call here — pure graph proximity using already-loaded KG.
    """
    same_article = [c for c in all_clauses if c.article_id == target.article_id and c.id != target.id]
    same_chapter = [
        c for c in all_clauses
        if c.chapter_id == target.chapter_id and c.article_id != target.article_id
    ]
    out: list[Clause] = []
    for c in same_article:
        if len(out) >= N_CANDIDATES_PER_QUERY:
            break
        out.append(c)
    for c in same_chapter:
        if len(out) >= N_CANDIDATES_PER_QUERY:
            break
        out.append(c)
    return out


def pick_easy_negatives(
    target: Clause,
    all_clauses: list[Clause],
    rng: random.Random,
    n: int = N_EASY_NEGATIVES,
) -> list[Clause]:
    """Random clauses from different chapter as guaranteed easy-neg diversity."""
    pool = [c for c in all_clauses if c.chapter_id != target.chapter_id]
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# OpenAI calls with retry
# ---------------------------------------------------------------------------


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
async def gen_queries(client: AsyncOpenAI, clause_text: str) -> list[str]:
    user_msg = QUERY_GEN_USER_TMPL.format(clause_text=clause_text)
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": QUERY_GEN_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.7,  # diversity-friendly for query generation
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    q1 = (parsed.get("q1") or "").strip()
    q2 = (parsed.get("q2") or "").strip()
    return [q for q in (q1, q2) if q]


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
    reraise=True,
)
async def verify_pair(
    client: AsyncOpenAI, query: str, clause_text: str,
) -> tuple[str, str]:
    """Return (label ∈ {YES, PARTIAL, NO}, reason)."""
    user_msg = VERIFIER_USER_TMPL.format(query=query, clause_text=clause_text)
    resp = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    parsed = json.loads(resp.choices[0].message.content or "{}")
    label = str(parsed.get("label") or "").strip().upper()
    reason = str(parsed.get("reason") or "")
    if label not in {"YES", "PARTIAL", "NO"}:
        label = "PARTIAL"  # conservative: refuse to use uncertain rows
    return label, reason


# ---------------------------------------------------------------------------
# Pipeline per clause
# ---------------------------------------------------------------------------


async def process_clause(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    seed: Clause,
    all_clauses: list[Clause],
    embed_model,
    rng: random.Random,
    stats: RunStats,
) -> list[dict[str, Any]]:
    async with sem:
        try:
            queries = await gen_queries(client, seed.text)
            stats.api_calls_query_gen += 1
        except Exception as e:
            stats.errors.append(f"gen_queries failed for {seed.id}: {type(e).__name__}: {e}")
            return []

    if not queries:
        return []
    stats.n_queries_emitted += len(queries)

    candidates = candidates_by_proximity(seed, all_clauses)
    stats.n_candidates_total += len(candidates) * len(queries)

    rows: list[dict[str, Any]] = []
    for q in queries:
        # Distance filter (Layer 3) — encode in batch
        if candidates:
            q_emb = embed_model.encode([q], normalize_embeddings=True, show_progress_bar=False)[0]
            cand_embs = embed_model.encode(
                [c.text for c in candidates],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            sims = (cand_embs @ q_emb).tolist()
        else:
            sims = []

        kept_after_distance: list[tuple[Clause, float]] = []
        for c, s in zip(candidates, sims):
            if s < SIM_MIN or s > SIM_MAX:
                stats.n_dropped_by_distance += 1
                continue
            kept_after_distance.append((c, float(s)))

        # LLM verifier (Layer 2) on each kept candidate, in parallel
        async with sem:
            verify_tasks = [
                verify_pair(client, q, c.text) for c, _ in kept_after_distance
            ]
            verify_results = await asyncio.gather(*verify_tasks, return_exceptions=True)
        stats.api_calls_verifier += len(verify_tasks)

        positives_extra: list[tuple[Clause, str, float]] = []
        negatives: list[tuple[Clause, str, float]] = []
        for (c, sim), res in zip(kept_after_distance, verify_results):
            if isinstance(res, Exception):
                stats.errors.append(f"verifier failed (q={q[:40]!r}, c={c.id}): {res}")
                continue
            label, reason = res
            if label == "YES":
                positives_extra.append((c, reason, sim))
                stats.n_kept_yes += 1
            elif label == "NO":
                negatives.append((c, reason, sim))
                stats.n_kept_no += 1
            else:  # PARTIAL
                stats.n_dropped_partial += 1

        # Easy-neg augmentation
        easy_negs = pick_easy_negatives(seed, all_clauses, rng)

        # Row assembly (multi-positive supported)
        row = {
            "query": q,
            "pos": [seed.text] + [c.text for c, _r, _s in positives_extra],
            "neg": [c.text for c, _r, _s in negatives] + [c.text for c in easy_negs],
            "_meta": {
                "seed_clause_id": seed.id,
                "seed_law_id": seed.law_id,
                "seed_article": seed.article_number,
                "verified_pos_clause_ids": [c.id for c, _r, _s in positives_extra],
                "verified_neg_clause_ids": [c.id for c, _r, _s in negatives],
                "easy_neg_clause_ids": [c.id for c in easy_negs],
                "verifier_pos_reasons": [r for _c, r, _s in positives_extra],
                "verifier_neg_reasons": [r for _c, r, _s in negatives],
                "distance_filter_thresholds": [SIM_MIN, SIM_MAX],
            },
        }
        rows.append(row)
        stats.n_rows_written += 1
        stats.sum_pos_per_row += len(row["pos"])
        stats.sum_neg_per_row += len(row["neg"])
        if len(row["pos"]) > 1:
            stats.n_multi_positive_rows += 1

    return rows


# ---------------------------------------------------------------------------
# Resume + persistence
# ---------------------------------------------------------------------------


def load_processed_set() -> set[str]:
    if not PROCESSED_MARKER.exists():
        return set()
    return {
        line.strip()
        for line in PROCESSED_MARKER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def mark_processed(clause_id: str) -> None:
    with PROCESSED_MARKER.open("a", encoding="utf-8") as f:
        f.write(clause_id + "\n")


def append_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with OUT_JSONL.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def write_stats(stats: RunStats) -> None:
    STATS_PATH.write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_metadata(n_clauses_total: int) -> None:
    import hashlib
    h = hashlib.sha256(GRAPH_PATH.read_bytes()).hexdigest()
    meta = {
        "script_version": "v5-sprint2-phase1-1",
        "openai_model": OPENAI_MODEL,
        "graph_sha256": h,
        "graph_path": str(GRAPH_PATH),
        "n_clauses_in_kg": n_clauses_total,
        "distance_filter": {"min": SIM_MIN, "max": SIM_MAX},
        "n_candidates_per_query": N_CANDIDATES_PER_QUERY,
        "n_easy_negatives": N_EASY_NEGATIVES,
        "seed": 42,
    }
    GEN_META_PATH.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(limit: int | None) -> int:
    if not os.getenv("OPENAI_API_KEY"):
        print("FAIL: missing OPENAI_API_KEY", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    clauses = load_clauses()
    print(f"Loaded {len(clauses)} clauses from KG")
    write_metadata(len(clauses))

    processed = load_processed_set()
    if processed:
        print(f"Resuming: {len(processed)} clauses already processed.")

    queue = [c for c in clauses if c.id not in processed]
    if limit:
        queue = queue[:limit]
    print(f"To process: {len(queue)}")

    # Lazy-load embed model (vanilla BGE-M3 for distance filter)
    from src.bge_m3_loader import load_bge_m3

    embed_model = load_bge_m3(adapter_path=None)

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)
    rng = random.Random(42)
    stats = RunStats(n_clauses_total=len(clauses))

    t0 = time.monotonic()
    BATCH = 25  # flush stats every N clauses
    for i in range(0, len(queue), BATCH):
        chunk = queue[i : i + BATCH]
        results = await asyncio.gather(
            *(
                process_clause(client, sem, c, clauses, embed_model, rng, stats)
                for c in chunk
            ),
            return_exceptions=True,
        )
        for c, res in zip(chunk, results):
            if isinstance(res, Exception):
                stats.errors.append(f"process_clause({c.id}) crashed: {res}")
                continue
            append_rows(res)
            mark_processed(c.id)
            stats.n_clauses_processed += 1
        stats.elapsed_s = round(time.monotonic() - t0, 1)
        write_stats(stats)
        done = i + len(chunk)
        print(
            f"  [{done:>4}/{len(queue)}] rows={stats.n_rows_written} "
            f"yes={stats.n_kept_yes} no={stats.n_kept_no} partial-drop={stats.n_dropped_partial} "
            f"({stats.elapsed_s:.0f}s elapsed)",
            flush=True,
        )

    write_stats(stats)
    print("\n=== DONE ===")
    print(json.dumps(stats.to_dict(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only first N un-processed clauses (smoke test). Default = all.",
    )
    args = p.parse_args()
    return asyncio.run(run(args.limit))


if __name__ == "__main__":
    sys.exit(main())
