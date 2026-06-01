"""Deterministic academic metrics for legal QA evaluation.

The core evaluator receives an already-loaded list of records. It does not
decide where records live on disk, how records are grouped, or how question
files are joined to result files. Each record must already include
``gold_items`` (strict tuple, the primary E2E metric basis); ``gold_articles``
is accepted for backward compat and used as a fallback.

**Citation matching policy: STRICT TUPLE-EQUAL** (v5 plan §5, locked
2026-05-31). A predicted citation matches a gold citation iff the full
4-tuple ``(law_id, article, clause, point)`` is identical — no component
may differ, be missing, or be over-specified. A wrong khoản may not exist
in the law; a missing khoản leaves the reader unable to locate the rule.

This policy applies to the primary E2E metric only — retrieval-only
diagnostic scripts (``scripts/exp{06,07,08}_metrics.py``) intentionally
stay article-deduped, since they probe "did dense surface the right Điều"
before the LLM commits to a specific khoản.

Report / CSV writing lives in :mod:`eval_core.report` so the metric engine
stays pure-computational.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from src.citations import (
    DEFAULT_REGISTRY_PATH,
    CitationRef,
    displayed_matches_pipeline,
    load_registry,
    parse_displayed_citations,
    parse_internal_citation_id,
)

DEFAULT_OUTPUT_DIR = Path("metrics")
METRICS_OUT = DEFAULT_OUTPUT_DIR / "academic_metrics.json"
CSV_OUT = DEFAULT_OUTPUT_DIR / "academic_metrics.csv"
REPORT_OUT = DEFAULT_OUTPUT_DIR / "academic_report.md"
# v2 (2026-05-31): switched citation matching from article-only to strict
# tuple-equal on (law_id, article, clause, point). See module docstring.
METRIC_VERSION = "academic_v2"


def _safe_div(num: int | float, denom: int | float) -> float | None:
    if denom == 0:
        return None
    return round(float(num) / float(denom), 4)


def _safe_mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 4) if vals else None


def _safe_std(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.stdev(vals), 4) if len(vals) > 1 else None


def _clean_answer_for_semantic(record: dict[str, Any]) -> tuple[str, str]:
    plain = str(record.get("plain_answer") or "").strip()
    if plain:
        return plain, "plain_answer"
    text = str(record.get("answer") or "")
    text = re.sub(r"\[[^\]]+\]", "", text)
    text = re.sub(r"(?im)^\s*(citation|citations|nguồn)\s*:.*$", "", text)
    text = re.sub(r"(?im)^\s*(issue|rule|application|conclusion)\s*:\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, "cleaned_answer"


def compute_citation_metrics(
    record: dict[str, Any],
    gold_items: set[str],
    registry,
    gold_articles: set[str] | None = None,
) -> dict[str, Any]:
    """Compute strict-tuple citation recall / precision / F1.

    ``gold_items`` — set of full-tuple gold ids (e.g. ``"L58_2014.A2"``
    when gold is article-only, ``"L58_2014.A2.K1"`` when gold has khoản).
    ``gold_articles`` — optional article-deduped set, kept on the output
    for debugging and for the citation-display rate (which still asks
    "did the LLM mention the right article at all"). When omitted, it
    is derived by stripping each item's khoản / điểm suffix.

    Matching policy is strict: ``pred_items ∩ gold_items``. The arm
    cannot match a gold item by over-specifying (e.g. arm cite
    ``Điều 2 khoản 1`` against gold ``Điều 2`` is a MISS — the arm
    may have hallucinated a non-existent khoản) nor by under-specifying.
    See module docstring for the full rationale.
    """
    raw_ids = [str(x).strip() for x in (record.get("citation_ids") or []) if str(x).strip()]
    pred_articles: dict[str, CitationRef] = {}
    pred_items: dict[str, CitationRef] = {}
    pred_parse_errors: list[str] = []

    for citation_id in raw_ids:
        try:
            ref = parse_internal_citation_id(citation_id, registry)
        except ValueError:
            pred_parse_errors.append(citation_id)
            continue
        pred_articles[ref.article_id] = CitationRef(source=ref.source, article=ref.article)
        pred_items[ref.item_id] = ref

    # Derive article-only views for backward-compat fields. The strict
    # primary comparison uses items only.
    if gold_articles is None:
        gold_articles = {item_id.split(".K", 1)[0] for item_id in gold_items}
    correct_items = sorted(set(pred_items) & set(gold_items))
    correct_articles = sorted(set(pred_articles) & set(gold_articles))

    recall = _safe_div(len(correct_items), len(gold_items))
    pred_precision_denom = len(pred_items) + len(pred_parse_errors)
    precision = _safe_div(len(correct_items), pred_precision_denom)
    if pred_precision_denom == 0 and gold_items:
        precision = 0.0
    if recall is None and gold_items:
        recall = 0.0
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if gold_items else None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    display = compute_citation_display(record, list(pred_items.values()), len(pred_parse_errors), registry)
    return {
        "gold_items": sorted(gold_items),
        "gold_articles": sorted(gold_articles),
        "pred_articles": sorted(pred_articles),
        "pred_items": sorted(pred_items),
        "pred_parse_errors": pred_parse_errors,
        # `correct_articles` is kept for debugging / backward-compat with
        # external readers; the primary correctness count is now
        # `correct_items`. `recall_num` / `precision_num` reflect the
        # STRICT TUPLE counts as of academic_v2.
        "correct_articles": correct_articles,
        "n_correct_articles": len(correct_articles),
        "correct_items": correct_items,
        "n_correct_items": len(correct_items),
        "recall_num": len(correct_items),
        "recall_denom": len(gold_items),
        "precision_num": len(correct_items),
        "precision_denom": pred_precision_denom,
        "citation_recall": recall,
        "citation_precision": precision,
        "citation_f1": f1,
        "citation_display": display,
    }


def compute_citation_display(
    record: dict[str, Any],
    pipeline_items: list[CitationRef],
    unparseable_count: int,
    registry,
) -> dict[str, Any]:
    answer = str(record.get("answer") or "")
    displayed = parse_displayed_citations(answer, registry)
    matched: set[str] = set()
    displayed_ids = sorted({d.item_id for d in displayed})
    for item in pipeline_items:
        if any(displayed_matches_pipeline(item, d) for d in displayed):
            matched.add(item.item_id)
    denom = len({p.item_id for p in pipeline_items}) + unparseable_count
    value = _safe_div(len(matched), denom) if denom else None
    return {
        "displayed_items": displayed_ids,
        "matched_pipeline_items": sorted(matched),
        "display_num": len(matched),
        "display_denom": denom,
        "citation_display_rate": value,
    }


def compute_prolog_fields(record: dict[str, Any]) -> dict[str, Any]:
    if "prolog_success" not in record and "n_repair_rounds" not in record:
        return {
            "prolog_first_try_solution": None,
            "repair_invoked": None,
            "repair_success": None,
        }
    success = bool(record.get("prolog_success", False))
    n_repair = int(record.get("n_repair_rounds", 0) or 0)
    return {
        "prolog_first_try_solution": success and n_repair == 0,
        "repair_invoked": n_repair >= 1,
        "repair_success": (success if n_repair >= 1 else None),
    }


def compute_bertscore(records: list[dict[str, Any]]) -> tuple[dict[str, dict], dict]:
    bs_records = []
    for r in records:
        candidate, source = _clean_answer_for_semantic(r)
        if not r.get("gold_answer"):
            continue
        bs_records.append(
            {
                "key": str(r["_metric_key"]),
                "candidate": candidate,
                "reference": str(r["gold_answer"]),
                "candidate_source": source,
            }
        )
    if not bs_records:
        return {}, {"status": "no_records_with_gold_answer"}

    try:
        from bert_score import score as bertscore
    except ImportError as exc:
        return {}, {"status": "bertscore_unavailable", "error": str(exc)}

    try:
        cands = [r["candidate"] for r in bs_records]
        refs = [r["reference"] for r in bs_records]
        device = "cuda" if os.getenv("EMBED_DEVICE", "cuda") == "cuda" else "cpu"
        p, r, f1 = bertscore(
            cands,
            refs,
            model_type="bert-base-multilingual-cased",
            lang="vi",
            verbose=False,
            device=device,
            rescale_with_baseline=False,
        )
    except Exception as exc:  # fail-soft by design
        return {}, {"status": "bertscore_failed", "error": f"{type(exc).__name__}: {exc}"}

    out = {}
    for i, rec in enumerate(bs_records):
        out[rec["key"]] = {
            "bertscore_p": round(float(p[i]), 4),
            "bertscore_r": round(float(r[i]), 4),
            "bertscore_f1": round(float(f1[i]), 4),
            "candidate_source": rec["candidate_source"],
            "status": "ok",
        }
    return out, {
        "status": "ok",
        "model_type": "bert-base-multilingual-cased",
        "lang": "vi",
        "device": device,
        "rescale_with_baseline": False,
    }


def compute_text_overlap(records: list[dict[str, Any]]) -> tuple[dict[str, dict], dict]:
    """ROUGE-1 / ROUGE-2 / ROUGE-L (F-measure) + BLEU of the answer vs gold.

    Lexical-overlap counterpart to :func:`compute_bertscore` (which is
    semantic). It scores the SAME candidate BERTScore uses — ``plain_answer``
    when present, otherwise the citation/IRAC-stripped ``answer`` (see
    :func:`_clean_answer_for_semantic`) — so the numbers are comparable across
    prose arms (llm_only / graphrag / qa_hyde_semantic) and the logic-LM arms.

    Fail-soft, exactly like BERTScore: a missing dependency or a runtime error
    skips the metric (status recorded in metadata) rather than failing the run.
    Citation metrics remain the hard contract; text overlap is auxiliary.

    BLEU uses sacrebleu sentence BLEU, normalised to ``[0, 1]`` (sacrebleu/100)
    so it shares the 0–1 scale of ROUGE/BERTScore. ROUGE uses no stemmer
    (English Porter stemming is meaningless for Vietnamese).
    """
    to_records = []
    for r in records:
        candidate, source = _clean_answer_for_semantic(r)
        if not r.get("gold_answer"):
            continue
        to_records.append(
            {
                "key": str(r["_metric_key"]),
                "candidate": candidate,
                "reference": str(r["gold_answer"]),
                "candidate_source": source,
            }
        )
    if not to_records:
        return {}, {"status": "no_records_with_gold_answer"}

    try:
        import sacrebleu
        from rouge_score import rouge_scorer
    except ImportError as exc:
        return {}, {"status": "text_overlap_unavailable", "error": str(exc)}

    try:
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
        out: dict[str, dict] = {}
        for rec in to_records:
            cand, ref = rec["candidate"], rec["reference"]
            rs = scorer.score(ref, cand)  # rouge_score API: score(target, prediction)
            bleu = sacrebleu.sentence_bleu(cand, [ref], tokenize="13a").score / 100.0
            out[rec["key"]] = {
                "rouge1": round(float(rs["rouge1"].fmeasure), 4),
                "rouge2": round(float(rs["rouge2"].fmeasure), 4),
                "rougeL": round(float(rs["rougeL"].fmeasure), 4),
                "bleu": round(float(bleu), 4),
                "candidate_source": rec["candidate_source"],
                "status": "ok",
            }
    except Exception as exc:  # fail-soft by design
        return {}, {"status": "text_overlap_failed", "error": f"{type(exc).__name__}: {exc}"}

    return out, {
        "status": "ok",
        "rouge": {"types": ["rouge1", "rouge2", "rougeL"], "score": "fmeasure", "use_stemmer": False},
        "bleu": {"impl": "sacrebleu.sentence_bleu", "tokenize": "13a", "scale": "[0,1] = sacrebleu/100"},
    }


def _aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def vals(path: tuple[str, ...]) -> list[float | None]:
        out = []
        for rec in records:
            v: Any = rec
            for key in path:
                v = v.get(key) if isinstance(v, dict) else None
            out.append(v if isinstance(v, int | float) else None)
        return out

    cit = [r["citation"] for r in records]
    recall_num = sum(int(c["recall_num"]) for c in cit)
    recall_denom = sum(int(c["recall_denom"]) for c in cit)
    precision_num = sum(int(c["precision_num"]) for c in cit)
    precision_denom = sum(int(c["precision_denom"]) for c in cit)
    display_num = sum(int(c["citation_display"]["display_num"]) for c in cit)
    display_denom = sum(int(c["citation_display"]["display_denom"]) for c in cit)

    prolog_records = [r for r in records if r["prolog"]["prolog_first_try_solution"] is not None]
    repaired = [r for r in prolog_records if r["prolog"]["repair_invoked"]]
    return {
        "n_records": len(records),
        "macro": {
            "citation_recall": _safe_mean(vals(("citation", "citation_recall"))),
            "citation_precision": _safe_mean(vals(("citation", "citation_precision"))),
            "citation_f1": _safe_mean(vals(("citation", "citation_f1"))),
            "citation_display_rate": _safe_mean(
                vals(("citation", "citation_display", "citation_display_rate"))
            ),
            "bertscore_p": _safe_mean(vals(("bertscore", "bertscore_p"))),
            "bertscore_r": _safe_mean(vals(("bertscore", "bertscore_r"))),
            "bertscore_f1": _safe_mean(vals(("bertscore", "bertscore_f1"))),
            "rouge1": _safe_mean(vals(("text_overlap", "rouge1"))),
            "rouge2": _safe_mean(vals(("text_overlap", "rouge2"))),
            "rougeL": _safe_mean(vals(("text_overlap", "rougeL"))),
            "bleu": _safe_mean(vals(("text_overlap", "bleu"))),
            "latency_s": _safe_mean(vals(("latency", "latency_s"))),
        },
        "std": {
            "citation_recall": _safe_std(vals(("citation", "citation_recall"))),
            "citation_precision": _safe_std(vals(("citation", "citation_precision"))),
            "citation_f1": _safe_std(vals(("citation", "citation_f1"))),
            "citation_display_rate": _safe_std(
                vals(("citation", "citation_display", "citation_display_rate"))
            ),
            "bertscore_f1": _safe_std(vals(("bertscore", "bertscore_f1"))),
            "rougeL": _safe_std(vals(("text_overlap", "rougeL"))),
            "bleu": _safe_std(vals(("text_overlap", "bleu"))),
            "latency_s": _safe_std(vals(("latency", "latency_s"))),
        },
        "micro": {
            "citation_recall": _safe_div(recall_num, recall_denom),
            "citation_precision": _safe_div(precision_num, precision_denom),
            "citation_display_rate": _safe_div(display_num, display_denom),
            "recall_num": recall_num,
            "recall_denom": recall_denom,
            "precision_num": precision_num,
            "precision_denom": precision_denom,
            "display_num": display_num,
            "display_denom": display_denom,
        },
        "prolog": _aggregate_prolog(prolog_records, repaired),
        "error_counts": {
            "pred_citation_parse_errors": sum(
                len(r["citation"]["pred_parse_errors"]) for r in records
            ),
            "records_with_no_pred_citations": sum(
                1 for r in records if r["citation"]["precision_denom"] == 0
            ),
        },
    }


def _aggregate_prolog(prolog_records: list[dict[str, Any]], repaired: list[dict[str, Any]]) -> dict[str, Any]:
    denom = len(prolog_records)
    if denom == 0:
        return {
            "n_prolog_records": 0,
            "prolog_first_try_solution_rate": None,
            "repair_invoked_rate": None,
            "repair_success_rate": None,
        }
    first = sum(1 for r in prolog_records if r["prolog"]["prolog_first_try_solution"])
    repair = sum(1 for r in prolog_records if r["prolog"]["repair_invoked"])
    repair_success = sum(1 for r in repaired if r["prolog"]["repair_success"])
    return {
        "n_prolog_records": denom,
        "prolog_first_try_solution_rate": _safe_div(first, denom),
        "repair_invoked_rate": _safe_div(repair, denom),
        "repair_success_rate": _safe_div(repair_success, len(repaired)) if repaired else None,
        "repair_success_num": repair_success,
        "repair_success_denom": len(repaired),
    }


def _coerce_gold_items(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    """Return ``(gold_items, gold_articles)`` from a metric record.

    Prefers the explicit ``gold_items`` field (strict-tuple, written by
    the post-2026-05-31 :mod:`eval_core.gold`). Falls back to
    ``gold_articles`` for older records — in that case items = articles
    since the legacy normalizer never extracted khoản/điểm.

    Raises if neither field is present.
    """
    raw_items = record.get("gold_items")
    raw_articles = record.get("gold_articles")
    if raw_items is None and raw_articles is None:
        raise RuntimeError(
            "Academic metric records must include `gold_items` (or legacy "
            "`gold_articles`); load and validate gold citations before "
            "calling evaluation."
        )
    items_set = {str(x).strip() for x in (raw_items or raw_articles or []) if str(x).strip()}
    articles_set = {str(x).strip() for x in (raw_articles or []) if str(x).strip()}
    if not articles_set:
        # Derive article-level view from items by stripping khoản/điểm.
        articles_set = {item_id.split(".K", 1)[0] for item_id in items_set}
    return items_set, articles_set


def _prepare_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        raise ValueError("No metric records provided")

    prepared: list[dict[str, Any]] = []
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            raise RuntimeError(f"Metric record at index {index} must be an object")
        rec = dict(raw)
        rec["_metric_index"] = index
        rec["_metric_key"] = str(index)
        prepared.append(rec)
    return prepared


def compute_academic_metrics(
    records: list[dict[str, Any]],
    registry=None,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = registry or load_registry(registry_path)
    prepared_records = _prepare_records(records)
    bs_results, bs_meta = compute_bertscore(prepared_records)
    to_results, to_meta = compute_text_overlap(prepared_records)

    result_records: list[dict[str, Any]] = []
    for rec in prepared_records:
        gold_items, gold_articles = _coerce_gold_items(rec)
        metric_rec = {
            "record_index": int(rec["_metric_index"]),
            "_record_path": rec.get("_record_path", ""),
            "citation": compute_citation_metrics(
                rec, gold_items, registry, gold_articles=gold_articles
            ),
            "latency": {"latency_s": rec.get("elapsed_s")},
            "prolog": compute_prolog_fields(rec),
        }
        if rec.get("record_id") is not None:
            metric_rec["record_id"] = str(rec["record_id"])
        bs = bs_results.get(str(rec["_metric_key"]))
        if bs:
            metric_rec["bertscore"] = bs
        to = to_results.get(str(rec["_metric_key"]))
        if to:
            metric_rec["text_overlap"] = to
        result_records.append(metric_rec)

    aggregate = _aggregate_records(result_records)
    result = {
        "metric_version": METRIC_VERSION,
        "n_input_records": len(prepared_records),
        "registry_path": str(registry_path),
        "gold_source": "record.gold_items (falls back to gold_articles)",
        "metadata": metadata or {},
        "bertscore_metadata": bs_meta,
        "text_overlap_metadata": to_meta,
        "records": result_records,
        "aggregate": aggregate,
    }
    return result


def _load_records_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Records JSON must be a list of metric records")
    if not all(isinstance(record, dict) for record in data):
        raise ValueError("Records JSON entries must be objects")
    return data


def main() -> int:
    # Late import: report writers live in eval_core.report, which imports
    # from this module — keep the import inside main() to avoid a cycle.
    from eval_core.report import compute_and_write_academic_metrics

    p = argparse.ArgumentParser(description="Compute deterministic academic metrics.")
    p.add_argument(
        "--records",
        type=Path,
        required=True,
        help="JSON list of already-loaded metric records.",
    )
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for all generated metric artifacts. Defaults to "
            f"{DEFAULT_OUTPUT_DIR}."
        ),
    )
    p.add_argument("--metrics-out", type=Path, default=None)
    p.add_argument("--csv-out", type=Path, default=None)
    p.add_argument("--report-out", type=Path, default=None)
    args = p.parse_args()

    try:
        records = _load_records_json(args.records)
        result = compute_and_write_academic_metrics(
            records=records,
            registry_path=args.registry,
            output_dir=args.output_dir,
            metrics_out=args.metrics_out,
            csv_out=args.csv_out,
            report_out=args.report_out,
            metadata={"records_path": str(args.records)},
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        f"OK: wrote {result['metrics_out']}, {result['csv_out']}, {result['report_out']} "
        f"for {len(result['records'])} records."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
