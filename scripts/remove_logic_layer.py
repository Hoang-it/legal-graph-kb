"""Backup the live Neo4j KG, then remove ONLY the logic/Prolog enrichment layer.

Lớp logic (do `offline/extract_prolog.py -> validate_prolog.py -> load_prolog.py`
nạp) gồm các node:

    LegalRule, LegalCondition, NumericalThreshold, LegalTerm,
    ProcedureStep, LegalEntity, CanonicalPredicate

cùng các cạnh kề chúng (EXTRACTED_FROM / INVOLVES_ENTITY / DEFINES_PREDICATE /
USES_PREDICATE / SAME_CONCEPT_AS / EXCEPTION_OF / NEXT_STEP + REQUIRES nội bộ
rule->condition + DEFINES tới LegalTerm).

Đây là thao tác **data-only**: KHÔNG đụng code logic-extraction (vẫn để dormant) và
KHÔNG drop constraint/index -> có thể nạp lại lớp này sau.

BACKUP CÓ THỂ RESTORE (đã kiểm chứng):
  * `graph_full.(cypher|json)` — snapshot toàn KG (APOC stream; fallback driver-JSON).
  * `logic_layer.json` — dump driver của ĐÚNG subgraph logic (node theo `id` +
    cạnh kề theo `id` hai đầu). Đây là artifact restore được **không cần APOC**,
    idempotent (MERGE theo `id`). Lệnh `--apply` mặc định **tự chứng minh restore**
    bằng round-trip thật: backup -> xoá -> restore lại từ `logic_layer.json` ->
    đối chiếu khớp 100% (id + nhãn + cạnh) -> rồi mới xoá lần cuối. Có thể restore
    độc lập bất cứ lúc nào: `--restore <backup_dir>`.

AN TOÀN:
  * Mặc định (không cờ) = `--verify`: chỉ đọc.
  * Xoá **theo LABEL** bằng DETACH DELETE, batched — giữ nguyên cạnh REQUIRES/
    DEFINES giữa các node *semantic* (script tự đối chiếu count trước/sau).
  * `--apply` luôn backup trước (trừ khi `--use-backup DIR` hợp lệ). Nếu proof
    restore THẤT BẠI, script DỪNG và để DB ở trạng thái đã-restore (an toàn).

CÁCH DÙNG:
    python scripts/remove_logic_layer.py                    # verify (read-only)
    python scripts/remove_logic_layer.py --backup           # chỉ backup
    python scripts/remove_logic_layer.py --apply            # backup + prove-restore + xoá
    python scripts/remove_logic_layer.py --apply --no-prove # backup + xoá (bỏ proof)
    python scripts/remove_logic_layer.py --restore backups/neo4j/<ts>   # nạp lại lớp logic
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

_REPO = Path(__file__).resolve().parents[1]
load_dotenv(_REPO / ".env")

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

URI = os.getenv("NEO4J_URI")
USER = os.getenv("NEO4J_USER")
PWD = os.getenv("NEO4J_PASSWORD")
DB = os.getenv("NEO4J_DATABASE", "neo4j")

# Lớp logic/Prolog — đồng bộ với offline/load_prolog.py:LOGIC_LABELS.
LOGIC_LABELS = [
    "LegalRule",
    "LegalCondition",
    "NumericalThreshold",
    "LegalTerm",
    "ProcedureStep",
    "LegalEntity",
    "CanonicalPredicate",
]

# Chỉ chấp nhận rel-type khớp pattern này khi restore (chặn injection qua type string).
_TYPE_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

BACKUP_ROOT = _REPO / "backups" / "neo4j"

# Artifacts copy kèm vào thư mục backup (ontology + KG processed) để snapshot trọn vẹn.
COPY_FILES = [
    "data/ontology/ontology_kg_full.json",
    "data/ontology/ontology_2024.json",
    "data/ontology/corpus_2024.jsonl",
    "data/graph/processed/merged_graph.json",
    "data/graph/processed/embeddings.parquet",
    "data/predicate_aliases.yaml",
    "schema/schema.cypher",
]
COPY_DIRS = ["data/eval/extracted_prolog"]  # chỉ có nếu đã sinh local


def _logic_or(var: str) -> str:
    """`n:LegalRule OR n:LegalCondition OR ...` cho mệnh đề WHERE."""
    return " OR ".join(f"{var}:{lab}" for lab in LOGIC_LABELS)


# ---------------------------------------------------------------------------
# Read-only probes
# ---------------------------------------------------------------------------


def probe_apoc(session) -> str | None:
    try:
        rec = session.run("RETURN apoc.version() AS v").single()
        return rec["v"] if rec else None
    except Exception:
        return None


def count_logic(session) -> dict:
    per = {}
    for lab in LOGIC_LABELS:
        per[lab] = session.run(f"MATCH (n:{lab}) RETURN count(n) AS c").single()["c"]
    per["TOTAL"] = sum(v for k, v in per.items() if k != "TOTAL")
    return per


def count_shared_edges(session) -> dict:
    """Baseline cạnh DÙNG CHUNG giữa semantic & logic: chỉ đếm cạnh mà CẢ HAI đầu
    mút KHÔNG phải node logic -> đây là phần semantic phải được bảo toàn."""
    out = {}
    for rel in ("REQUIRES", "DEFINES"):
        q = (
            f"MATCH (a)-[r:{rel}]->(b) "
            f"WHERE NOT ({_logic_or('a')}) AND NOT ({_logic_or('b')}) "
            f"RETURN count(r) AS c"
        )
        out[rel] = session.run(q).single()["c"]
    return out


# ---------------------------------------------------------------------------
# Logic subgraph dump / restore (driver-based, restore được không cần APOC)
# ---------------------------------------------------------------------------


def dump_logic_subgraph(session) -> dict:
    """Dump ĐÚNG subgraph logic: node logic (id+labels+props) + mọi cạnh kề (theo
    id hai đầu). Đầu mút non-logic (Clause/Article) KHÔNG dump (chúng vẫn còn sau
    khi xoá) -> restore match lại bằng id."""
    nodes = []
    for rec in session.run(
        f"MATCH (n) WHERE {_logic_or('n')} "
        "RETURN n.id AS id, labels(n) AS labels, properties(n) AS props"
    ):
        nodes.append(
            {"id": rec["id"], "labels": list(rec["labels"]), "properties": dict(rec["props"])}
        )
    rels = []
    for rec in session.run(
        f"MATCH (a)-[r]->(b) WHERE ({_logic_or('a')}) OR ({_logic_or('b')}) "
        "RETURN a.id AS src, b.id AS dst, type(r) AS type, properties(r) AS props"
    ):
        rels.append(
            {
                "src": rec["src"],
                "dst": rec["dst"],
                "type": rec["type"],
                "properties": dict(rec["props"]),
            }
        )
    return {"nodes": nodes, "relationships": rels}


def fingerprint(dump: dict) -> dict:
    """Dấu vân tay cấu trúc của subgraph logic để so khớp baseline vs restored."""
    node_ids = sorted(n["id"] for n in dump["nodes"] if n.get("id"))
    per_label: dict[str, int] = {}
    for n in dump["nodes"]:
        for lab in n["labels"]:
            if lab in LOGIC_LABELS:
                per_label[lab] = per_label.get(lab, 0) + 1
    edges = sorted(f'{e["src"]}|{e["type"]}|{e["dst"]}' for e in dump["relationships"])
    return {
        "n_nodes": len(node_ids),
        "node_ids": node_ids,
        "per_label": per_label,
        "n_edges": len(edges),
        "edges": edges,
    }


def restore_logic_subgraph(session, dump: dict) -> tuple[int, int]:
    """Nạp lại subgraph logic bằng MERGE theo `id` (idempotent). Node trước, cạnh
    sau (để hai đầu mút tồn tại). Trả về (n_nodes, n_rels)."""
    # --- nodes: gom theo label logic chính ---
    by_label: dict[str, list[dict]] = {}
    for n in dump["nodes"]:
        lab = next((l for l in n["labels"] if l in LOGIC_LABELS), None)
        if lab is None:
            continue
        by_label.setdefault(lab, []).append(
            {"id": n["id"], "props": {k: v for k, v in n["properties"].items() if v is not None}}
        )
    n_nodes = 0
    for lab, batch in by_label.items():
        session.run(
            f"UNWIND $batch AS n MERGE (x:{lab} {{id: n.id}}) SET x += n.props",
            batch=batch,
        )
        n_nodes += len(batch)

    # --- relationships: gom theo type (đã validate pattern) ---
    by_type: dict[str, list[dict]] = {}
    for e in dump["relationships"]:
        t = e["type"]
        if not _TYPE_RE.match(t):
            raise ValueError(f"Rel-type không hợp lệ trong dump: {t!r}")
        by_type.setdefault(t, []).append(e)
    n_rels = 0
    for t, batch in by_type.items():
        # Chuẩn hoá về {src, dst, props} để Cypher tham chiếu `e.props` đồng nhất
        # với node-restore (dump lưu key `properties`).
        norm = [
            {"src": e["src"], "dst": e["dst"], "props": dict(e.get("properties") or {})}
            for e in batch
        ]
        with_sc = [e for e in norm if e["props"].get("source_clause") is not None]
        without_sc = [e for e in norm if e["props"].get("source_clause") is None]
        if with_sc:
            session.run(
                "UNWIND $batch AS e MATCH (a {id: e.src}) MATCH (b {id: e.dst}) "
                f"MERGE (a)-[r:{t} {{source_clause: e.props.source_clause}}]->(b) SET r += e.props",
                batch=with_sc,
            )
            n_rels += len(with_sc)
        if without_sc:
            session.run(
                "UNWIND $batch AS e MATCH (a {id: e.src}) MATCH (b {id: e.dst}) "
                f"MERGE (a)-[r:{t}]->(b) SET r += e.props",
                batch=without_sc,
            )
            n_rels += len(without_sc)
    return n_nodes, n_rels


# ---------------------------------------------------------------------------
# Full-graph backup (APOC stream / driver fallback)
# ---------------------------------------------------------------------------


def _stream_apoc(session, query: str, out_path: Path) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in session.run(query):
            stmt = rec.get("cypherStatements")
            if stmt:
                f.write(stmt)
                if not stmt.endswith("\n"):
                    f.write("\n")
                n += len(stmt)
    return n


def export_apoc_full(session, out_path: Path) -> int:
    return _stream_apoc(
        session,
        "CALL apoc.export.cypher.all(null, {stream:true, format:'cypher-shell', "
        "useOptimizations:{type:'UNWIND_BATCH', unwindBatchSize:200}}) "
        "YIELD cypherStatements RETURN cypherStatements",
        out_path,
    )


def export_driver_json(session, out_path: Path) -> int:
    """Fallback (không APOC): dump mọi node + rel ra JSON (keyed theo elementId)."""
    nodes = [
        {"eid": r["eid"], "labels": r["labels"], "properties": dict(r["props"])}
        for r in session.run(
            "MATCH (n) RETURN elementId(n) AS eid, labels(n) AS labels, properties(n) AS props"
        )
    ]
    rels = [
        {"src": r["src"], "dst": r["dst"], "type": r["type"], "properties": dict(r["props"])}
        for r in session.run(
            "MATCH (a)-[r]->(b) RETURN elementId(a) AS src, elementId(b) AS dst, "
            "type(r) AS type, properties(r) AS props"
        )
    ]
    text = json.dumps({"nodes": nodes, "relationships": rels}, ensure_ascii=False, default=str)
    out_path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def copy_artifacts(dest: Path) -> list[dict]:
    copied = []
    files_dir = dest / "files"
    files_dir.mkdir(parents=True, exist_ok=True)
    for rel in COPY_FILES:
        src = _REPO / rel
        if src.exists():
            tgt = files_dir / rel
            tgt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, tgt)
            copied.append({"path": rel, "bytes": src.stat().st_size})
    for rel in COPY_DIRS:
        src = _REPO / rel
        if src.exists() and src.is_dir():
            shutil.copytree(src, files_dir / rel, dirs_exist_ok=True)
            nbytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
            copied.append({"path": rel + "/ (dir)", "bytes": nbytes})
    return copied


def do_backup(session, apoc_version: str | None, logic: dict, shared: dict, logic_dump: dict) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = BACKUP_ROOT / ts
    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n=== BACKUP -> {dest} ===")

    # 1) Snapshot toàn KG
    if apoc_version:
        full_path = dest / "graph_full.cypher"
        print("  APOC export (full graph, stream) ...")
        full_bytes = export_apoc_full(session, full_path)
        full_kind = "apoc.export.cypher.all"
    else:
        full_path = dest / "graph_full.json"
        print("  APOC không khả dụng -> fallback dump JSON (full) ...")
        full_bytes = export_driver_json(session, full_path)
        full_kind = "driver-json"
    print(f"  ✓ {full_path.name}  ({full_bytes / 1_048_576:.2f} MB)  [{full_kind}]")

    # 2) Artifact restore-được (driver, không cần APOC)
    logic_path = dest / "logic_layer.json"
    logic_path.write_text(
        json.dumps(logic_dump, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(
        f"  ✓ logic_layer.json  ({len(logic_dump['nodes'])} nodes, "
        f"{len(logic_dump['relationships'])} rels)  [restore-verified artifact]"
    )

    # 3) Ontology + processed artifacts
    copied = copy_artifacts(dest)
    print(f"  ✓ copied {len(copied)} artifact(s) -> {dest / 'files'}")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "neo4j": {"uri": URI, "database": DB},
        "apoc_version": apoc_version,
        "full_export": {"kind": full_kind, "file": full_path.name, "bytes": full_bytes},
        "logic_layer": {
            "file": "logic_layer.json",
            "fingerprint": fingerprint(logic_dump),
        },
        "logic_counts": logic,
        "shared_edges_baseline": shared,
        "copied_artifacts": copied,
        "restore_proven": None,  # set True/False sau khi --apply chứng minh
    }
    (dest / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("  ✓ manifest.json")
    return dest


def load_backup_manifest(backup_dir: Path) -> dict | None:
    mf = backup_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        data = json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return None
    logic_file = (data.get("logic_layer") or {}).get("file")
    if not logic_file or not (backup_dir / logic_file).exists():
        return None
    return data


# ---------------------------------------------------------------------------
# Delete (label-scoped, batched)
# ---------------------------------------------------------------------------


def delete_logic(session) -> int:
    total = 0
    q = (
        f"MATCH (n) WHERE {_logic_or('n')} "
        "WITH n LIMIT 1000 DETACH DELETE n RETURN count(n) AS c"
    )
    while True:
        c = session.run(q).single()["c"]
        total += c
        if c:
            print(f"  ... deleted batch {c}  (running total {total})")
        if c == 0:
            break
    return total


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _print_counts(title: str, logic: dict, shared: dict) -> None:
    print(f"\n{title}")
    for lab in LOGIC_LABELS:
        print(f"    {lab:<20} {logic[lab]:>7}")
    print(f"    {'TOTAL':<20} {logic['TOTAL']:>7}")
    print(f"  Cạnh semantic (bảo toàn): REQUIRES={shared['REQUIRES']}  DEFINES={shared['DEFINES']}")


def _fp_diff(base: dict, got: dict) -> list[str]:
    msgs = []
    if base["n_nodes"] != got["n_nodes"]:
        msgs.append(f"n_nodes {base['n_nodes']} != {got['n_nodes']}")
    if base["per_label"] != got["per_label"]:
        msgs.append(f"per_label {base['per_label']} != {got['per_label']}")
    if base["node_ids"] != got["node_ids"]:
        miss = sorted(set(base["node_ids"]) - set(got["node_ids"]))[:5]
        extra = sorted(set(got["node_ids"]) - set(base["node_ids"]))[:5]
        msgs.append(f"node_ids khác (thiếu {miss}, dư {extra})")
    if base["n_edges"] != got["n_edges"]:
        msgs.append(f"n_edges {base['n_edges']} != {got['n_edges']}")
    if base["edges"] != got["edges"]:
        miss = sorted(set(base["edges"]) - set(got["edges"]))[:5]
        msgs.append(f"edges khác (vd thiếu {miss})")
    return msgs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backup", action="store_true", help="Tạo backup (full + logic_layer.json).")
    ap.add_argument("--apply", action="store_true", help="Backup + prove-restore + XOÁ node logic.")
    ap.add_argument("--no-prove", action="store_true", help="Bỏ qua bước chứng minh restore khi --apply.")
    ap.add_argument("--use-backup", type=str, default=None, help="Dùng thư mục backup đã có (có manifest hợp lệ).")
    ap.add_argument("--restore", type=str, default=None, help="Nạp lại lớp logic từ <backup_dir>/logic_layer.json rồi thoát.")
    args = ap.parse_args()

    if not all([URI, USER, PWD]):
        print("FAIL: thiếu NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD trong .env", file=sys.stderr)
        return 1

    print(f"Connecting Neo4j {URI} (db={DB}) ...")
    driver = GraphDatabase.driver(URI, auth=(USER, PWD))
    try:
        driver.verify_connectivity()
        print("  ✓ Connected")
        with driver.session(database=DB) as s:

            # --- restore độc lập ---
            if args.restore:
                bdir = Path(args.restore)
                if not bdir.is_absolute():
                    bdir = _REPO / bdir
                data = load_backup_manifest(bdir)
                if data is None:
                    print(f"FAIL: {bdir} thiếu manifest.json/logic_layer.json hợp lệ.", file=sys.stderr)
                    return 1
                dump = json.loads((bdir / "logic_layer.json").read_text(encoding="utf-8"))
                print(f"\n=== RESTORE lớp logic từ {bdir} ===")
                nn, nr = restore_logic_subgraph(s, dump)
                got = fingerprint(dump_logic_subgraph(s))
                base = data["logic_layer"]["fingerprint"]
                diff = _fp_diff(base, got)
                _print_counts("=== SAU RESTORE ===", count_logic(s), count_shared_edges(s))
                if diff:
                    print("  ✗ Restore CHƯA khớp backup:\n    - " + "\n    - ".join(diff))
                    return 2
                print(f"  ✓ Restore khớp 100% backup ({nn} nodes, {nr} rels).")
                return 0

            apoc = probe_apoc(s)
            print(f"  APOC: {apoc or 'KHÔNG khả dụng (sẽ fallback driver JSON khi backup)'}")

            logic = count_logic(s)
            shared = count_shared_edges(s)
            _print_counts("=== BASELINE (read-only) ===", logic, shared)

            if not args.backup and not args.apply:
                print("\n[verify] Chỉ đọc — không đổi gì. Dùng --backup / --apply để hành động.")
                return 0

            logic_dump = dump_logic_subgraph(s)
            base_fp = fingerprint(logic_dump)

            # --- backup standalone ---
            if args.backup and not args.apply:
                if logic["TOTAL"] == 0:
                    print("\n[backup] Lưu ý: 0 node logic — vẫn tạo snapshot hiện trạng.")
                do_backup(s, apoc, logic, shared, logic_dump)
                print("\n=== DONE (backup) ===")
                return 0

            # --- apply ---
            if logic["TOTAL"] == 0:
                print("\n[apply] 0 node logic trong DB — không có gì để xoá.")
                if args.backup:
                    do_backup(s, apoc, logic, shared, logic_dump)
                print("\n=== DONE (nothing to delete) ===")
                return 0

            # backup (bắt buộc trước khi xoá)
            if args.use_backup:
                bdir = Path(args.use_backup)
                if not bdir.is_absolute():
                    bdir = _REPO / bdir
                data = load_backup_manifest(bdir)
                if data is None:
                    print(f"FAIL: --use-backup {bdir} không hợp lệ. Hãy --backup trước.", file=sys.stderr)
                    return 1
                print(f"\n[apply] Dùng backup sẵn có: {bdir}")
                backup_dir = bdir
            else:
                backup_dir = do_backup(s, apoc, logic, shared, logic_dump)

            # B1: xoá lần 1
            print("\n=== DELETE logic layer (DETACH DELETE theo LABEL) ===")
            deleted = delete_logic(s)
            print(f"  ✓ Đã xoá {deleted} node logic + cạnh kề.")
            after = count_logic(s)
            shared_after = count_shared_edges(s)
            if after["TOTAL"] != 0:
                print(f"  ✗ Còn {after['TOTAL']} node logic — DỪNG.", file=sys.stderr)
                return 2
            if shared_after != shared:
                print(
                    f"  ✗ Cạnh semantic đổi {shared} -> {shared_after} — DỪNG (đã có backup full).",
                    file=sys.stderr,
                )
                return 2
            print("  ✓ Xoá sạch; cạnh semantic REQUIRES/DEFINES nguyên vẹn.")

            restore_proven: bool | None = None
            if not args.no_prove:
                # B2: restore từ backup để CHỨNG MINH file backup dùng được
                print("\n=== PROVE RESTORE: nạp lại từ logic_layer.json rồi đối chiếu ===")
                dump = json.loads((backup_dir / "logic_layer.json").read_text(encoding="utf-8"))
                nn, nr = restore_logic_subgraph(s, dump)
                got_fp = fingerprint(dump_logic_subgraph(s))
                shared_restored = count_shared_edges(s)
                diff = _fp_diff(base_fp, got_fp)
                if diff or shared_restored != shared:
                    print("  ✗ RESTORE KHÔNG KHỚP — DỪNG, để DB ở trạng thái đã-restore (an toàn):")
                    for m in diff:
                        print(f"    - {m}")
                    if shared_restored != shared:
                        print(f"    - cạnh semantic {shared} -> {shared_restored}")
                    return 2
                restore_proven = True
                print(f"  ✓ RESTORE PROVEN — khớp 100% baseline ({nn} nodes, {nr} rels).")

                # B3: xoá lần cuối -> trạng thái mong muốn (sạch lớp logic)
                print("\n=== DELETE lần cuối (đưa về trạng thái sạch) ===")
                delete_logic(s)
                final = count_logic(s)
                shared_final = count_shared_edges(s)
                if final["TOTAL"] != 0 or shared_final != shared:
                    print("  ✗ Trạng thái cuối không như mong đợi — DỪNG.", file=sys.stderr)
                    return 2
                _print_counts("=== TRẠNG THÁI CUỐI ===", final, shared_final)
            else:
                _print_counts("=== TRẠNG THÁI CUỐI ===", after, shared_after)

            # cập nhật manifest với kết quả proof
            mf = backup_dir / "manifest.json"
            try:
                data = json.loads(mf.read_text(encoding="utf-8"))
                data["restore_proven"] = restore_proven
                mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            print(
                f"\n  ✓ Lớp logic đã gỡ sạch. Backup: {backup_dir}"
                + ("  (restore ĐÃ kiểm chứng)" if restore_proven else "")
            )
            print(
                "  → Bước tiếp: python -m offline.build_ontology_kg  "
                "(rebuild ontology_kg_full.json sạch)"
            )
            print("\n=== DONE (apply) ===")
            return 0
    finally:
        driver.close()


if __name__ == "__main__":
    sys.exit(main())
