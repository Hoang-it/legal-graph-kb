"""Deterministic metric engine for the **retrieval** experiment family.

This is the single, generic, config-driven home for retrieval metrics —
the counterpart of :mod:`eval_core.metrics` (which owns the ``qa`` family).
Both families now recompute through ``python -m eval_core metrics <exp>``.

A retrieval experiment declares, in its ``config.yaml``::

    family: retrieval
    recompute: eval_core
    retrieval:
      arms: [dense, dense_hyde, dense_hyde2]   # REQUIRED — arms to score
      ks: [12, 20, 30, 50, 70, 100, all]       # optional — K cutoffs (`all`/null = full list)
      record_field: retrieval_only.final_article_ids   # optional — where the ranked ids live
      latency_field: retrieval_only.elapsed_s          # optional — per-record latency
      pilot_subset: experiments/<NN>/pilot_50_stt.json # optional — score only these stt
      registry: data/legal_sources.yaml                # optional — citation authority registry

For K ∈ ``ks`` (``all`` = the full retrieved list) it computes, per arm,
macro-averaged over questions with non-empty gold:

- recall@K, precision@K, F1@K, NDCG@K (binary relevance, log2 discount)

plus K-independent rank-aware metrics:

- R-Precision (precision at K=|gold|), MRR (first gold article).

Everything is stratified by gold-corpus type (``in_corpus`` is the
pre-registered headline stratum / ``mixed`` / ``ooc`` / ``unparseable``).

Outputs (all inside the experiment folder — Rule 1):
- ``metrics/gold_citations_normalized.json``  (re-derived, self-contained)
- ``metrics/academic_metrics.json``           (the Tier-2 contract artifact)
- ``metrics/academic_metrics.csv``
- ``report/academic_report.md``

Pure stdlib + the shared ``eval_core.gold`` / ``src.legal_metadata`` helpers;
fully offline and byte-exact (no Neo4j / embeddings / OpenAI). The metric
*definitions* below are the contract — do not edit them to flatter a report
(see the skill's Rule 2); fix the system, not the metric.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from eval_core.gold import validate_gold_citations
from src.legal_metadata import load_law_metadata

_REPO = Path(__file__).resolve().parents[1]

STRATA = ("in_corpus", "mixed", "ooc", "unparseable")

# Vietnamese legal-document code, e.g. 41/2024/QH15, 158/2025/NĐ-CP, 12/2025/TT-BNV.
_RE_CODE = re.compile(r"\d+/\d{4}/(?:QH\d+|N[ĐD]-CP|NQ-CP|TT-[A-Z]+|CP|TTg)")

# Default K cutoffs when the config omits `retrieval.ks`. None = the full list.
_DEFAULT_KS: tuple[int | None, ...] = (12, 20, 30, 50, 70, 100, None)


# --------------------------------------------------------------------------- #
# Metric primitives — THE definitions. Ported verbatim from the retrieval
# experiments so existing numbers reproduce exactly. Single source of truth.
# --------------------------------------------------------------------------- #
def categorize(raw: Any, in_corpus_codes: set[str]) -> str:
    """Stratum of a question by its gold citations vs the in-corpus law codes."""
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


def _retrieved_at_k(retrieved: list[str], k: int | None) -> list[str]:
    return retrieved if k is None else retrieved[:k]


def recall(retrieved_k: list[str], gold: set[str]) -> float | None:
    if not gold:
        return None
    return len(gold & set(retrieved_k)) / len(gold)


def precision(retrieved_k: list[str], gold: set[str]) -> float | None:
    if not retrieved_k:
        return 0.0 if gold else None
    return len(gold & set(retrieved_k)) / len(retrieved_k)


def f1_score(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def r_precision(retrieved: list[str], gold: set[str]) -> float | None:
    """Precision at K=|gold|. With |gold|<<K_max this reads precision/recall
    as a single number — at K=|gold| the two coincide, so the value is the
    fraction of gold the retriever placed in the top-|gold| positions."""
    if not gold:
        return None
    k = len(gold)
    return len(gold & set(retrieved[:k])) / k


def reciprocal_rank(retrieved: list[str], gold: set[str]) -> float | None:
    """1 / rank of the first retrieved article in gold (1-indexed); 0 if none."""
    if not gold:
        return None
    for i, aid in enumerate(retrieved, start=1):
        if aid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float | None:
    """Binary-relevance NDCG@K. Gain = 1 if retrieved[i] in gold else 0,
    discount = 1/log2(i+1), normalised by the ideal DCG."""
    if not gold:
        return None
    pool = retrieved[:k]
    dcg = 0.0
    for i, aid in enumerate(pool, start=1):
        if aid in gold:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def _macro(values) -> float | None:
    vs = [v for v in values if v is not None]
    return round(mean(vs), 4) if vs else None


# --------------------------------------------------------------------------- #
# Config parsing
# --------------------------------------------------------------------------- #
def _parse_ks(raw: Any) -> tuple[int | None, ...]:
    """Parse the `ks` config list. `all`/`null`/`None` -> the full-list sentinel."""
    if not raw:
        return _DEFAULT_KS
    out: list[int | None] = []
    for k in raw:
        if k is None or (isinstance(k, str) and k.strip().lower() == "all"):
            out.append(None)
        else:
            out.append(int(k))
    return tuple(out)


def _dig(rec: dict, dotted: str) -> Any:
    """Walk ``a.b.c`` through nested dicts, tolerating missing keys."""
    cur: Any = rec
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _resolve(path_str: str, exp_dir: Path) -> Path:
    """Resolve a config path: absolute as-is; relative tried under the repo
    root first, then under the experiment folder."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    repo_rel = _REPO / p
    if repo_rel.exists():
        return repo_rel
    return exp_dir / p


# --------------------------------------------------------------------------- #
# Loading + scoring
# --------------------------------------------------------------------------- #
def _load_gold(questions_path: Path, registry_path: Path, metrics_dir: Path) -> dict[int, list[str]]:
    ok, summary = validate_gold_citations(
        questions_path=questions_path,
        registry_path=registry_path,
        out_dir=metrics_dir,
    )
    if not ok:
        raise ValueError(f"gold validation failed; see {summary['errors_path']}")
    data = json.loads(Path(summary["normalized_path"]).read_text(encoding="utf-8"))
    return {int(k): v.get("gold_articles") or [] for k, v in data["records"].items()}


def _load_arm_records(
    arm_dir: Path, stt_subset: set[int] | None = None
) -> dict[int, dict]:
    if not arm_dir.is_dir():
        raise FileNotFoundError(arm_dir)
    out: dict[int, dict] = {}
    for p in sorted(arm_dir.glob("A*.json")):
        if p.name.endswith(".error.json"):
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        stt = int(rec["stt"])
        if stt_subset is not None and stt not in stt_subset:
            continue
        out[stt] = rec
    return out


def score_arm(
    arm: str,
    records: dict[int, dict],
    gold_map: dict[int, list[str]],
    questions: dict[int, dict],
    in_corpus_codes: set[str],
    ks: tuple[int | None, ...],
    record_field: str,
    latency_field: str,
) -> list[dict]:
    rows: list[dict] = []
    for stt, rec in records.items():
        retrieved = list(_dig(rec, record_field) or [])
        gold = set(gold_map.get(stt) or [])
        row: dict = {
            "stt": stt,
            "arm": arm,
            "n_gold": len(gold),
            "n_retrieved_all": len(retrieved),
            "category": categorize(questions[stt].get("gold_citations_raw"), in_corpus_codes),
            "elapsed_s": _dig(rec, latency_field),
        }
        for k in ks:
            r_k = _retrieved_at_k(retrieved, k)
            r = recall(r_k, gold)
            p = precision(r_k, gold)
            f1 = f1_score(p, r)
            n = ndcg_at_k(retrieved, gold, k if k is not None else max(len(retrieved), 1))
            kk = "all" if k is None else str(k)
            row[f"recall@{kk}"] = r
            row[f"precision@{kk}"] = p
            row[f"f1@{kk}"] = f1
            row[f"ndcg@{kk}"] = n
            row[f"n_retrieved@{kk}"] = len(r_k)
        row["r_precision"] = r_precision(retrieved, gold)
        row["mrr"] = reciprocal_rank(retrieved, gold)
        rows.append(row)
    return rows


def aggregate_macro(rows: list[dict], ks: tuple[int | None, ...]) -> dict:
    summary: dict = {"n": len(rows)}
    for k in ks:
        kk = "all" if k is None else str(k)
        summary[f"recall@{kk}"] = _macro([r[f"recall@{kk}"] for r in rows])
        summary[f"precision@{kk}"] = _macro([r[f"precision@{kk}"] for r in rows])
        summary[f"f1@{kk}"] = _macro([r[f"f1@{kk}"] for r in rows])
        summary[f"ndcg@{kk}"] = _macro([r[f"ndcg@{kk}"] for r in rows])
        summary[f"avg_n_retrieved@{kk}"] = round(
            mean([r[f"n_retrieved@{kk}"] for r in rows]), 2
        ) if rows else None
    summary["r_precision"] = _macro([r["r_precision"] for r in rows])
    summary["mrr"] = _macro([r["mrr"] for r in rows])
    latencies = [r["elapsed_s"] for r in rows if r["elapsed_s"] is not None]
    summary["avg_elapsed_s"] = round(mean(latencies), 3) if latencies else None
    return summary


def aggregate_stratified(rows: list[dict], ks: tuple[int | None, ...]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cat in STRATA:
        sub = [r for r in rows if r["category"] == cat]
        out[cat] = aggregate_macro(sub, ks) if sub else {"n": 0}
    return out


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_csv(metrics_dir: Path, per_arm_rows: dict[str, list[dict]], ks: tuple[int | None, ...]) -> Path:
    out = metrics_dir / "academic_metrics.csv"
    field_ks = [("all" if k is None else str(k)) for k in ks]
    fieldnames = ["arm", "stt", "category", "n_gold"]
    for kk in field_ks:
        fieldnames += [
            f"n_retrieved@{kk}",
            f"recall@{kk}",
            f"precision@{kk}",
            f"f1@{kk}",
            f"ndcg@{kk}",
        ]
    fieldnames += ["r_precision", "mrr", "elapsed_s"]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for arm, rows in per_arm_rows.items():
            for r in rows:
                row_out = {"arm": arm, "stt": r["stt"], "category": r["category"],
                           "n_gold": r["n_gold"]}
                for kk in field_ks:
                    row_out[f"n_retrieved@{kk}"] = r[f"n_retrieved@{kk}"]
                    row_out[f"recall@{kk}"] = r[f"recall@{kk}"]
                    row_out[f"precision@{kk}"] = r[f"precision@{kk}"]
                    row_out[f"f1@{kk}"] = r[f"f1@{kk}"]
                    row_out[f"ndcg@{kk}"] = r[f"ndcg@{kk}"]
                row_out["r_precision"] = r["r_precision"]
                row_out["mrr"] = r["mrr"]
                row_out["elapsed_s"] = r["elapsed_s"]
                writer.writerow(row_out)
    return out


def _write_json(
    metrics_dir: Path,
    slug: str,
    arms: list[str],
    ks: tuple[int | None, ...],
    per_arm_summary: dict,
    per_arm_strat: dict,
) -> Path:
    out = metrics_dir / "academic_metrics.json"
    payload = {
        "experiment": slug,
        "family": "retrieval",
        "arms": list(arms),
        "Ks": [("all" if k is None else k) for k in ks],
        "overall_macro": per_arm_summary,
        "stratified": per_arm_strat,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def _write_report(
    report_dir: Path,
    title: str,
    description: str,
    arms: list[str],
    ks: tuple[int | None, ...],
    per_arm_summary: dict,
    per_arm_strat: dict,
    n_scored: int,
) -> Path:
    out = report_dir / "academic_report.md"
    report_dir.mkdir(parents=True, exist_ok=True)
    field_ks = [("all" if k is None else str(k)) for k in ks]

    lines: list[str] = []
    lines.append(f"# {title} — retrieval metrics")
    lines.append("")
    if description:
        lines.append(description.strip())
        lines.append("")
    lines.append(f"Dataset: {n_scored} questions with non-empty gold. Metric granularity: article.")
    lines.append(f"Arms: {', '.join(f'`{a}`' for a in arms)}.")
    lines.append("")
    lines.append(f"## Overall macro (n={n_scored})")
    lines.append("")

    def _table_metric(name: str, summary: dict) -> list[str]:
        rows = []
        rows.append("| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |")
        rows.append("|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|")
        for arm in arms:
            s = summary[arm]
            vals = " | ".join(_fmt(s.get(f"{name}@{kk}")) for kk in field_ks)
            rows.append(f"| {arm} | {s.get('n', 0)} | {vals} |")
        return rows

    lines.append("### Recall@K"); lines += _table_metric("recall", per_arm_summary); lines.append("")
    lines.append("### Precision@K"); lines += _table_metric("precision", per_arm_summary); lines.append("")
    lines.append("### F1@K"); lines += _table_metric("f1", per_arm_summary); lines.append("")
    lines.append("### NDCG@K (binary relevance)"); lines += _table_metric("ndcg", per_arm_summary); lines.append("")
    lines.append("### Average retrieved-set size at K"); lines += _table_metric("avg_n_retrieved", per_arm_summary); lines.append("")

    lines.append("### Rank-aware (K-independent)")
    lines.append("")
    lines.append("| arm | n | R-Precision | MRR |")
    lines.append("|---|---:|---:|---:|")
    for arm in arms:
        s = per_arm_summary[arm]
        lines.append(f"| {arm} | {s.get('n', 0)} | {_fmt(s.get('r_precision'))} | {_fmt(s.get('mrr'))} |")
    lines.append("")

    lines.append("### Latency")
    lines.append("")
    lines.append("| arm | avg elapsed (s) |")
    lines.append("|---|---:|")
    for arm in arms:
        lines.append(f"| {arm} | {_fmt(per_arm_summary[arm].get('avg_elapsed_s'))} |")
    lines.append("")

    lines.append("## Stratified by gold corpus type")
    lines.append("")
    for cat in STRATA:
        lines.append(f"### {cat}")
        lines.append("")
        n_present = max(per_arm_strat[arm][cat].get("n", 0) for arm in arms)
        if not n_present:
            lines.append("_(no questions in this stratum)_")
            lines.append("")
            continue
        for metric_name in ("recall", "precision", "f1", "ndcg"):
            lines.append(f"_{metric_name.capitalize()}@K — {cat}_")
            lines.append("")
            lines.append("| arm | n | " + " | ".join(f"@{kk}" for kk in field_ks) + " |")
            lines.append("|---|---:|" + "|".join(["---:"] * len(field_ks)) + "|")
            for arm in arms:
                s = per_arm_strat[arm][cat]
                if not s.get("n"):
                    continue
                vals = " | ".join(_fmt(s[f"{metric_name}@{kk}"]) for kk in field_ks)
                lines.append(f"| {arm} | {s['n']} | {vals} |")
            lines.append("")
        lines.append(f"_Rank-aware — {cat}_")
        lines.append("")
        lines.append("| arm | n | R-Precision | MRR |")
        lines.append("|---|---:|---:|---:|")
        for arm in arms:
            s = per_arm_strat[arm][cat]
            if not s.get("n"):
                continue
            lines.append(
                f"| {arm} | {s['n']} | {_fmt(s.get('r_precision'))} | {_fmt(s.get('mrr'))} |"
            )
        lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Retrieval-only diagnostic at article granularity.")
    lines.append("- `in_corpus` is the pre-registered headline stratum.")
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# Entry point — called by eval_core.cli.cmd_metrics when family == retrieval.
# --------------------------------------------------------------------------- #
def compute_retrieval_metrics(
    experiment,
    arms_filter: list[str] | None = None,
    force_full: bool = False,
) -> dict[str, Any]:
    """Compute + write retrieval metrics for ``experiment``.

    ``experiment`` is an :class:`eval_core.experiment.Experiment`. Returns a
    dict with ``metrics_out`` / ``csv_out`` / ``report_out`` / ``records`` so
    the CLI can report identically to the qa path.
    """
    cfg = experiment.config or {}
    rcfg = cfg.get("retrieval") or {}

    arms = list(rcfg.get("arms") or [])
    if not arms:
        raise ValueError(
            f"{experiment.name}: config.yaml needs a non-empty `retrieval.arms` list "
            "for the retrieval metric engine."
        )
    if arms_filter:
        arms = [a for a in arms if a in set(arms_filter)]

    ks = _parse_ks(rcfg.get("ks"))
    record_field = rcfg.get("record_field") or "retrieval_only.final_article_ids"
    latency_field = rcfg.get("latency_field") or "retrieval_only.elapsed_s"

    metrics_dir = experiment.metrics_dir
    report_dir = experiment.report_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    questions_path = _resolve(str(experiment.dataset.questions), experiment.path)
    registry_path = _resolve(str(rcfg.get("registry") or "data/legal_sources.yaml"), experiment.path)

    # Optional pilot subset (score only these stt).
    stt_subset: set[int] | None = None
    pilot = rcfg.get("pilot_subset")
    if pilot and not force_full:
        pilot_path = _resolve(str(pilot), experiment.path)
        if pilot_path.exists():
            payload = json.loads(pilot_path.read_text(encoding="utf-8"))
            stt_subset = {int(s) for s in payload.get("stt_list") or []}

    gold_map = _load_gold(questions_path, registry_path, metrics_dir)
    questions = json.loads(questions_path.read_text(encoding="utf-8"))
    q_by_stt = {q["stt"]: q for q in questions}
    in_corpus_codes = {m.full_id for m in load_law_metadata().values()}

    per_arm_rows: dict[str, list[dict]] = {}
    per_arm_summary: dict[str, dict] = {}
    per_arm_strat: dict[str, dict] = {}
    for arm in arms:
        try:
            recs = _load_arm_records(experiment.arm_results_dir(arm), stt_subset=stt_subset)
        except FileNotFoundError:
            recs = {}
        if not recs:
            per_arm_rows[arm] = []
            per_arm_summary[arm] = aggregate_macro([], ks)
            per_arm_strat[arm] = aggregate_stratified([], ks)
            continue
        rows = score_arm(
            arm, recs, gold_map, q_by_stt, in_corpus_codes, ks, record_field, latency_field
        )
        per_arm_rows[arm] = rows
        per_arm_summary[arm] = aggregate_macro(rows, ks)
        per_arm_strat[arm] = aggregate_stratified(rows, ks)

    csv_out = _write_csv(metrics_dir, per_arm_rows, ks)
    metrics_out = _write_json(
        metrics_dir, experiment.path.name, arms, ks, per_arm_summary, per_arm_strat
    )
    n_scored = max((s.get("n") or 0) for s in per_arm_summary.values()) if per_arm_summary else 0
    report_out = _write_report(
        report_dir, experiment.name, experiment.description, arms, ks,
        per_arm_summary, per_arm_strat, n_scored,
    )

    return {
        "metrics_out": metrics_out,
        "csv_out": csv_out,
        "report_out": report_out,
        "records": per_arm_rows,
        "n_scored": n_scored,
    }
