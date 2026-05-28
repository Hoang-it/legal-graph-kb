"""Deterministic academic metrics for legal QA evaluation.

This script intentionally excludes all LLM-judge metrics. It validates
`gold_citations_raw` first and fails hard if dataset gold citations are not
usable.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

from experiments.arms import MAIN_EXPERIMENT_ARMS, parse_metrics_arms
from experiments.validate_gold_citations import (
    DEFAULT_OUT_DIR as DEFAULT_ACADEMIC_DIR,
)
from experiments.validate_gold_citations import (
    DEFAULT_QUESTIONS,
    NORMALIZED_OUT,
    validate_gold_citations,
)
from src.citations import (
    DEFAULT_REGISTRY_PATH,
    CitationRef,
    displayed_matches_pipeline,
    load_registry,
    parse_displayed_citations,
    parse_internal_citation_id,
)

DEFAULT_RESULTS_ROOT = Path("data/eval/results")
METRICS_OUT = Path("data/eval/academic_metrics.json")
CSV_OUT = Path("data/eval/academic_metrics.csv")
REPORT_OUT = Path("reports/academic_report.md")
LOGIC_LM_ARMS = {"logic_lm_no_retrieval", "logic_lm_ontology", "logic_lm_graphrag"}
METRIC_VERSION = "academic_v1"


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


def _load_gold_map(path: Path) -> dict[int, set[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, set[str]] = {}
    for stt, rec in (data.get("records") or {}).items():
        out[int(stt)] = set(rec.get("gold_articles") or [])
    return out


def _load_records(
    results_root: Path,
    arms: list[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    if arms:
        arm_dirs = [results_root / arm for arm in arms]
        missing = [str(path) for path in arm_dirs if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing result arm directories: {missing}")
    else:
        arm_dirs = sorted(p for p in results_root.iterdir() if p.is_dir())
    for arm_dir in arm_dirs:
        records = []
        for path in sorted(arm_dir.glob("A*.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON record: {path}") from exc
            rec.setdefault("arm", arm_dir.name)
            rec["_record_path"] = str(path)
            records.append(rec)
        if records:
            out[arm_dir.name] = sorted(records, key=lambda r: int(r.get("stt", 0)))
    if not out:
        raise ValueError(f"No result records found under {results_root}")
    return out


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
    gold_articles: set[str],
    registry,
) -> dict[str, Any]:
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

    correct_articles = sorted(set(pred_articles) & set(gold_articles))
    recall = _safe_div(len(correct_articles), len(gold_articles))
    pred_precision_denom = len(pred_articles) + len(pred_parse_errors)
    precision = _safe_div(len(correct_articles), pred_precision_denom)
    if pred_precision_denom == 0 and gold_articles:
        precision = 0.0
    if recall is None and gold_articles:
        recall = 0.0
    if precision is None or recall is None or precision + recall == 0:
        f1 = 0.0 if gold_articles else None
    else:
        f1 = round(2 * precision * recall / (precision + recall), 4)

    display = compute_citation_display(record, list(pred_items.values()), len(pred_parse_errors), registry)
    return {
        "gold_articles": sorted(gold_articles),
        "pred_articles": sorted(pred_articles),
        "pred_items": sorted(pred_items),
        "pred_parse_errors": pred_parse_errors,
        "n_correct_articles": len(correct_articles),
        "correct_articles": correct_articles,
        "recall_num": len(correct_articles),
        "recall_denom": len(gold_articles),
        "precision_num": len(correct_articles),
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


def _is_elite_record(record: dict[str, Any]) -> bool:
    arm = str(record.get("arm") or "")
    base_arm = str(record.get("base_arm") or "")
    return arm in LOGIC_LM_ARMS or base_arm in LOGIC_LM_ARMS


def compute_prolog_fields(record: dict[str, Any]) -> dict[str, Any]:
    if not _is_elite_record(record):
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


def compute_bertscore(records: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], dict], dict]:
    bs_records = []
    for r in records:
        candidate, source = _clean_answer_for_semantic(r)
        if not r.get("gold_answer"):
            continue
        bs_records.append(
            {
                "key": (str(r["arm"]), int(r["stt"])),
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


def _aggregate_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
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
            "n_elite_records": 0,
            "prolog_first_try_solution_rate": None,
            "repair_invoked_rate": None,
            "repair_success_rate": None,
        }
    first = sum(1 for r in prolog_records if r["prolog"]["prolog_first_try_solution"])
    repair = sum(1 for r in prolog_records if r["prolog"]["repair_invoked"])
    repair_success = sum(1 for r in repaired if r["prolog"]["repair_success"])
    return {
        "n_elite_records": denom,
        "prolog_first_try_solution_rate": _safe_div(first, denom),
        "repair_invoked_rate": _safe_div(repair, denom),
        "repair_success_rate": _safe_div(repair_success, len(repaired)) if repaired else None,
        "repair_success_num": repair_success,
        "repair_success_denom": len(repaired),
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "stt",
        "citation_recall",
        "citation_precision",
        "citation_f1",
        "citation_display_rate",
        "bertscore_p",
        "bertscore_r",
        "bertscore_f1",
        "latency_s",
        "prolog_first_try_solution",
        "repair_invoked",
        "repair_success",
        "pred_citation_parse_errors",
        "record_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "arm": r["arm"],
                    "stt": r["stt"],
                    "citation_recall": r["citation"]["citation_recall"],
                    "citation_precision": r["citation"]["citation_precision"],
                    "citation_f1": r["citation"]["citation_f1"],
                    "citation_display_rate": r["citation"]["citation_display"][
                        "citation_display_rate"
                    ],
                    "bertscore_p": r.get("bertscore", {}).get("bertscore_p"),
                    "bertscore_r": r.get("bertscore", {}).get("bertscore_r"),
                    "bertscore_f1": r.get("bertscore", {}).get("bertscore_f1"),
                    "latency_s": r["latency"]["latency_s"],
                    "prolog_first_try_solution": r["prolog"]["prolog_first_try_solution"],
                    "repair_invoked": r["prolog"]["repair_invoked"],
                    "repair_success": r["prolog"]["repair_success"],
                    "pred_citation_parse_errors": "|".join(
                        r["citation"]["pred_parse_errors"]
                    ),
                    "record_path": r.get("_record_path", ""),
                }
            )


def _fmt(v: Any) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_report(result: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Academic Metrics Report",
        "",
        f"- metric_version: `{result['metric_version']}`",
        f"- results_root: `{result['results_root']}`",
        "- gold source: `gold_citations_raw`",
        "- judge metrics: not included",
        "",
        "## Headline Macro Metrics",
        "",
        "| Arm | n | citation_recall | citation_precision | citation_f1 | citation_display_rate | bertscore_f1 | latency_s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, agg in result["aggregates"].items():
        macro = agg["macro"]
        lines.append(
            "| "
            + " | ".join(
                [
                    arm,
                    str(agg["n_records"]),
                    _fmt(macro["citation_recall"]),
                    _fmt(macro["citation_precision"]),
                    _fmt(macro["citation_f1"]),
                    _fmt(macro["citation_display_rate"]),
                    _fmt(macro["bertscore_f1"]),
                    _fmt(macro["latency_s"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Citation Micro Metrics",
            "",
            "| Arm | recall | precision | display_rate |",
            "|---|---:|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        micro = agg["micro"]
        lines.append(
            f"| {arm} | {_fmt(micro['citation_recall'])} "
            f"(Σ={micro['recall_num']}/{micro['recall_denom']}) | "
            f"{_fmt(micro['citation_precision'])} "
            f"(Σ={micro['precision_num']}/{micro['precision_denom']}) | "
            f"{_fmt(micro['citation_display_rate'])} "
            f"(Σ={micro['display_num']}/{micro['display_denom']}) |"
        )
    lines.extend(
        [
            "",
            "## Prolog Metrics",
            "",
            "| Arm | n_elite | first_try_solution | repair_invoked | repair_success |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        pr = agg["prolog"]
        if pr["n_elite_records"] == 0:
            continue
        lines.append(
            f"| {arm} | {pr['n_elite_records']} | "
            f"{_fmt(pr['prolog_first_try_solution_rate'])} | "
            f"{_fmt(pr['repair_invoked_rate'])} | "
            f"{_fmt(pr['repair_success_rate'])} "
            f"(Σ={pr['repair_success_num']}/{pr['repair_success_denom']}) |"
        )
    lines.extend(
        [
            "",
            "## BERTScore Status",
            "",
            "```json",
            json.dumps(result["bertscore_metadata"], ensure_ascii=False, indent=2),
            "```",
            "",
            "## Error Counts",
            "",
            "| Arm | pred_citation_parse_errors | records_with_no_pred_citations |",
            "|---|---:|---:|",
        ]
    )
    for arm, agg in result["aggregates"].items():
        e = agg["error_counts"]
        lines.append(
            f"| {arm} | {e['pred_citation_parse_errors']} | "
            f"{e['records_with_no_pred_citations']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_academic_metrics(
    results_root: Path = DEFAULT_RESULTS_ROOT,
    questions_path: Path = DEFAULT_QUESTIONS,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    academic_dir: Path = DEFAULT_ACADEMIC_DIR,
    metrics_out: Path = METRICS_OUT,
    csv_out: Path = CSV_OUT,
    report_out: Path = REPORT_OUT,
    arms: list[str] | None = None,
) -> dict[str, Any]:
    ok, summary = validate_gold_citations(questions_path, registry_path, academic_dir)
    if not ok:
        raise RuntimeError(
            "gold citation validation failed; fix dataset before computing metrics. "
            f"See {summary['errors_path']}"
        )
    registry = load_registry(registry_path)
    gold_map = _load_gold_map(academic_dir / NORMALIZED_OUT)
    records_by_arm = _load_records(results_root, arms=arms)
    flat_raw = [rec for records in records_by_arm.values() for rec in records]
    bs_results, bs_meta = compute_bertscore(flat_raw)

    result_records: dict[str, list[dict[str, Any]]] = {}
    flat_rows: list[dict[str, Any]] = []
    for arm, records in records_by_arm.items():
        out_records = []
        for rec in records:
            stt = int(rec["stt"])
            if stt not in gold_map:
                raise RuntimeError(f"No validated gold citations for stt={stt}")
            metric_rec = {
                "arm": arm,
                "stt": stt,
                "_record_path": rec.get("_record_path", ""),
                "citation": compute_citation_metrics(rec, gold_map[stt], registry),
                "latency": {"latency_s": rec.get("elapsed_s")},
                "prolog": compute_prolog_fields(rec),
            }
            bs = bs_results.get((arm, stt))
            if bs:
                metric_rec["bertscore"] = bs
            out_records.append(metric_rec)
            flat_rows.append(metric_rec)
        result_records[arm] = out_records

    aggregates = {arm: _aggregate_arm(recs) for arm, recs in result_records.items()}
    result = {
        "metric_version": METRIC_VERSION,
        "results_root": str(results_root),
        "questions_path": str(questions_path),
        "registry_path": str(registry_path),
        "gold_artifact": str(academic_dir / NORMALIZED_OUT),
        "arms_filter": arms,
        "bertscore_metadata": bs_meta,
        "records": result_records,
        "aggregates": aggregates,
    }
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(flat_rows, csv_out)
    _write_report(result, report_out)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Compute deterministic academic metrics.")
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    p.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    p.add_argument("--academic-dir", type=Path, default=DEFAULT_ACADEMIC_DIR)
    p.add_argument("--metrics-out", type=Path, default=METRICS_OUT)
    p.add_argument("--csv-out", type=Path, default=CSV_OUT)
    p.add_argument("--report-out", type=Path, default=REPORT_OUT)
    p.add_argument(
        "--arms",
        type=str,
        default="main",
        help=(
            "Arm selection: 'main' (default), 'all' to discover every result dir, "
            "or comma-separated arms. Main = "
            + ",".join(MAIN_EXPERIMENT_ARMS)
        ),
    )
    args = p.parse_args()
    try:
        arms = parse_metrics_arms(args.arms)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    try:
        result = compute_academic_metrics(
            results_root=args.results_root,
            questions_path=args.questions,
            registry_path=args.registry,
            academic_dir=args.academic_dir,
            metrics_out=args.metrics_out,
            csv_out=args.csv_out,
            report_out=args.report_out,
            arms=arms,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        f"OK: wrote {args.metrics_out}, {args.csv_out}, {args.report_out} "
        f"for {len(result['records'])} arms."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
