"""B4 — Hợp nhất 5 nguồn (B1 + B2 + B3) → merged_graph.json + report.

Đầu vào:
    data/graph/interim/structured_law.json
    data/graph/interim/internal_refs.json
    data/graph/interim/external_refs.json
    data/graph/interim/definitions.json
    data/graph/interim/amendments.json
    data/graph/interim/llm_extractions/A*.json

Đầu ra:
    data/graph/processed/merged_graph.json        — cấu trúc {nodes: {label: [...]}, edges: {type: [...]}}
    data/graph/processed/extraction_summary.md    — báo cáo cho con người

NGUYÊN TẮC:
- Dedup semantic node theo canonical ID; gộp `mentioned_in` (union).
- Dedup edge theo khoá (src, dst, type, source_clause) → tránh duplicate.
- Validate cuối cùng (fail-fast):
  * Mọi semantic node có `mentioned_in` non-empty.
  * Mọi edge `src`/`dst` tồn tại trong nodes.
  * Mọi `source_clause` của edges là Clause/Point có thực.
  * Article 1..141 đủ + không trùng ID.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from src.legal_metadata import load_law_metadata, load_order

INTERIM = Path("data/graph/interim")
LLM_DIR = INTERIM / "llm_extractions"
OUT = Path("data/graph/processed/merged_graph.json")
REPORT = Path("data/graph/processed/extraction_summary.md")


# ---------------------------------------------------------------------------
# Load sources
# ---------------------------------------------------------------------------


def _load_json(path: Path):
    if not path.exists():
        sys.exit(f"FAIL: thiếu {path}. Chạy bước trước.")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _load_law_inputs() -> list[dict]:
    laws = load_law_metadata()
    out = []
    for law_id in load_order():
        meta = laws[law_id]
        structured_path = INTERIM / f"structured_law_{law_id}.json"
        if not structured_path.exists() and law_id == "L41_2024":
            structured_path = INTERIM / "structured_law.json"
        structured = _load_json(structured_path)
        out.append(
            {
                "law_id": law_id,
                "metadata": meta,
                "structured": structured,
                "internal_refs": _load_json(INTERIM / f"internal_refs_{law_id}.json"),
                "external_refs": _load_json(INTERIM / f"external_refs_{law_id}.json"),
                "definitions": _load_json(INTERIM / f"definitions_{law_id}.json"),
                "amendments": _load_json(INTERIM / f"amendments_{law_id}.json"),
            }
        )
    return out


def _llm_files_for(law_id: str) -> list[dict]:
    """Đọc file LLM extraction của 1 luật theo layout multi-law.

    Layout canonical (B3 mới): ``llm_extractions/<law_id>/A*.json``.

    Fallback cho L41 legacy: nếu subdir ``llm_extractions/L41_2024/`` chưa
    tồn tại nhưng các file phẳng ``llm_extractions/A*.json`` còn ở local
    (B3 cũ ghi ra đó), B4 vẫn đọc được. Pattern này mirror fallback đã có
    cho ``structured_law.json`` trong ``_load_law_inputs``.
    """
    subdir = LLM_DIR / law_id
    if subdir.exists():
        paths = sorted(subdir.glob("A*.json"), key=lambda p: int(p.stem[1:]))
    elif law_id == "L41_2024" and LLM_DIR.exists():
        paths = sorted(LLM_DIR.glob("A*.json"), key=lambda p: int(p.stem[1:]))
    else:
        return []
    out: list[dict] = []
    for fp in paths:
        with fp.open(encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def load_all() -> dict:
    """Load multi-law structural/rule sources into one dict."""
    laws_inputs = _load_law_inputs()
    llm_files: list[dict] = []
    for item in laws_inputs:
        llm_files.extend(_llm_files_for(item["law_id"]))
    return {"laws": laws_inputs, "llm_files": llm_files}


# ---------------------------------------------------------------------------
# Build nodes + edges
# ---------------------------------------------------------------------------

# Map LLM extraction key → Neo4j label
LLM_KEY_TO_LABEL = {
    "concepts": "LegalConcept",
    "subjects": "Subject",
    "organizations": "Organization",
    "roles": "Role",
    "benefits": "Benefit",
    "conditions": "Condition",
    "obligations": "Obligation",
    "rights": "Right",
    "prohibited_acts": "ProhibitedAct",
    "funds": "Fund",
}


class _Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, dict]] = defaultdict(dict)
        self.edges: dict[str, list[dict]] = defaultdict(list)
        self._edge_keys: set[tuple] = set()

    def add_node(self, label: str, node: dict) -> None:
        nid = node["id"]
        if nid in self.nodes[label]:
            # Merge: gộp mentioned_in nếu có
            existing = self.nodes[label][nid]
            for mi_field in ("mentioned_in",):
                if mi_field in node:
                    merged = list(dict.fromkeys((existing.get(mi_field) or []) + node[mi_field]))
                    existing[mi_field] = merged
            # Giữ description/definition dài hơn (chứa nhiều thông tin hơn)
            for tf in ("description", "definition"):
                if tf in node:
                    cur = existing.get(tf) or ""
                    new = node[tf] or ""
                    if len(new) > len(cur):
                        existing[tf] = new
        else:
            self.nodes[label][nid] = node

    def add_edge(self, etype: str, edge: dict, key: tuple | None = None) -> bool:
        """Trả True nếu thật sự thêm (không trùng)."""
        if key is None:
            key = (etype, edge["src"], edge["dst"], edge.get("source_clause", ""))
        if key in self._edge_keys:
            return False
        self._edge_keys.add(key)
        self.edges[etype].append(edge)
        return True


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_structural(g: _Graph, structured: dict, metadata=None) -> None:
    """Tạo nodes & edges structural từ B1."""
    law = structured["law"]
    law_id = law["id"]
    law_node = metadata.law_node if metadata is not None else law
    law_node = {**law_node, **{k: v for k, v in law.items() if v is not None}}
    g.add_node(
        "Law",
        law_node,
    )

    prev_article_id: str | None = None
    for ch in structured["chapters"]:
        g.add_node(
            "Chapter",
            {
                "id": ch["id"],
                "law_code": law_id,
                "number": ch["number"],
                "roman": ch["roman"],
                "title": ch["title"],
            },
        )
        g.add_edge("HAS_CHAPTER", {"src": law["id"], "dst": ch["id"]})

        for sec in ch["sections"]:
            g.add_node(
                "Section",
                {
                    "id": sec["id"],
                    "law_code": law_id,
                    "number": sec["number"],
                    "title": sec["title"],
                    "chapter_id": ch["id"],
                },
            )
            g.add_edge("HAS_SECTION", {"src": ch["id"], "dst": sec["id"]})

        for art in ch["articles"]:
            g.add_node(
                "Article",
                {
                    "id": art["id"],
                    "law_code": law_id,
                    "number": art["number"],
                    "title": art["title"],
                    "text": art["text"],
                    "chapter_id": ch["id"],
                    "section_id": art.get("section_id"),
                },
            )
            # Cạnh containment: Chapter -> Article (luôn có)
            g.add_edge("HAS_ARTICLE", {"src": ch["id"], "dst": art["id"]})
            g.add_edge("BELONGS_TO", {"src": art["id"], "dst": law_id})
            # Cạnh IN_SECTION nếu thuộc Mục
            if art.get("section_id"):
                g.add_edge(
                    "IN_SECTION",
                    {
                        "src": art["id"],
                        "dst": art["section_id"],
                    },
                )
            # NEXT theo thứ tự đọc
            if prev_article_id is not None:
                g.add_edge("NEXT", {"src": prev_article_id, "dst": art["id"]})
            prev_article_id = art["id"]

            for cl in art["clauses"]:
                g.add_node(
                    "Clause",
                    {
                        "id": cl["id"],
                        "law_code": law_id,
                        "number": cl["number"],
                        "text": cl["text"],
                        "article_id": art["id"],
                    },
                )
                g.add_edge("HAS_CLAUSE", {"src": art["id"], "dst": cl["id"]})
                for pt in cl["points"]:
                    g.add_node(
                        "Point",
                        {
                        "id": pt["id"],
                        "law_code": law_id,
                        "letter": pt["letter"],
                            "text": pt["text"],
                            "clause_id": cl["id"],
                        },
                    )
                    g.add_edge("HAS_POINT", {"src": cl["id"], "dst": pt["id"]})

            for tbl in art.get("tables", []):
                g.add_node(
                    "Table",
                    {
                        "id": tbl["id"],
                        "law_code": law_id,
                        "article_id": art["id"],
                        "rows_json": json.dumps(tbl["rows"], ensure_ascii=False),
                    },
                )
                g.add_edge("HAS_TABLE", {"src": art["id"], "dst": tbl["id"]})


def build_definitions(g: _Graph, definitions: list[dict]) -> None:
    for d in definitions:
        g.add_node(
            "LegalConcept",
            {
                "id": d["concept_id"],
                "term": d["term"],
                "definition": d["definition"],
                "defined_in": d["defined_in"],
                "mentioned_in": [d["defined_in"]],
            },
        )
        # DEFINES: từ Article (chứa định nghĩa) → Concept
        art_id = d["defined_in"].split(".K")[0]  # 'L41_2024.A3.K1' → 'L41_2024.A3'
        g.add_edge(
            "DEFINES",
            {
                "src": art_id,
                "dst": d["concept_id"],
                "source_clause": d["defined_in"],
                "source_text": d["span"][:300],
            },
        )


def build_external_refs(g: _Graph, external_refs: list[dict]) -> None:
    for r in external_refs:
        # ExternalLaw node (dedup theo dst id)
        node = {
            "id": r["dst"],
            "code": r["external_code"],
            "title": r["external_title"],
            "target_law": r.get("target_law"),
        }
        g.add_node("ExternalLaw", node)

        # Edge CITES_EXTERNAL: src (clause/article) → ExternalLaw
        # Khoá unique: (src, dst, source_clause, char_offset) — char_offset phân biệt 2 lần cite trong cùng clause
        key = ("CITES_EXTERNAL", r["src"], r["dst"], r["source_clause"], r["char_offset"])
        g.add_edge(
            "CITES_EXTERNAL",
            {
                "src": r["src"],
                "dst": r["dst"],
                "source_clause": r["source_clause"],
                "span": r["span"],
                "char_offset": r["char_offset"],
                "external_article": r["external_article"],
                "external_clause": r["external_clause"],
                "external_point": r["external_point"],
                "target_law": r.get("target_law"),
                "target_article_id": r.get("target_article_id"),
            },
            key=key,
        )


def build_cross_law_refs(g: _Graph, external_refs: list[dict]) -> None:
    """Create Clause/Point -> loaded Article refs when external source is in KG."""
    article_ids = set(g.nodes.get("Article", {}).keys())
    for r in external_refs:
        target_id = r.get("target_article_id")
        if not target_id or target_id not in article_ids:
            continue
        key = (
            "REFERS_TO",
            r["src"],
            target_id,
            r["source_clause"],
            r.get("char_offset"),
        )
        g.add_edge(
            "REFERS_TO",
            {
                "src": r["src"],
                "dst": target_id,
                "source_clause": r["source_clause"],
                "span": r["span"],
                "char_offset": r["char_offset"],
                "law": r.get("target_law"),
                "external_article": r.get("external_article"),
                "external_clause": r.get("external_clause"),
                "external_point": r.get("external_point"),
            },
            key=key,
        )


def build_amendments(g: _Graph, amendments: list[dict]) -> None:
    for a in amendments:
        action = a["action"]  # AMENDS / REPEALS / REPLACES
        node = {
            "id": a["dst"],
            "code": a["external_code"],
            "title": a["external_title"],
            "target_law": a.get("target_law"),
        }
        g.add_node("ExternalLaw", node)
        key = (action, a["src"], a["dst"], a["source_clause"], a.get("external_article"))
        g.add_edge(
            action,
            {
                "src": a["src"],
                "dst": a["dst"],
                "source_clause": a["source_clause"],
                "external_article": a["external_article"],
                "external_clause": a["external_clause"],
                "external_point": a["external_point"],
                "span_action": a["span_action"],
                "char_offset_action": a["char_offset_action"],
                "target_law": a.get("target_law"),
            },
            key=key,
        )


def build_law_relations(g: _Graph, law_inputs: list[dict]) -> None:
    law_ids = set(g.nodes.get("Law", {}).keys())
    for item in law_inputs:
        meta = item["metadata"]
        for target in meta.repeals:
            if target in law_ids:
                g.add_edge(
                    "REPEALS",
                    {
                        "src": meta.id,
                        "dst": target,
                        "source": "data/legal_metadata.yaml",
                    },
                    key=("REPEALS", meta.id, target, "law_metadata"),
                )


def build_internal_refs(g: _Graph, internal_refs: list[dict]) -> None:
    for r in internal_refs:
        key = ("REFERENCES", r["src"], r["dst"], r["source_clause"], r["char_offset"])
        g.add_edge(
            "REFERENCES",
            {
                "src": r["src"],
                "dst": r["dst"],
                "source_clause": r["source_clause"],
                "span": r["span"],
                "char_offset": r["char_offset"],
                "kind": r["kind"],
                "is_self": r.get("is_self", False),
            },
            key=key,
        )


def build_from_llm(g: _Graph, llm_files: list[dict]) -> None:
    """Gom entities + semantic_edges từ các file llm_extractions/A*.json."""
    for data in llm_files:
        if "skipped_reason" in data:
            continue
        ext = data["extraction"]

        for llm_key, label in LLM_KEY_TO_LABEL.items():
            for ent in ext.get(llm_key, []):
                node = dict(ent)
                # Đảm bảo mentioned_in unique
                node["mentioned_in"] = list(dict.fromkeys(node["mentioned_in"]))
                g.add_node(label, node)

        for ed in ext.get("semantic_edges", []):
            etype = ed["type"]
            # Edge unique theo (type, src, dst, source_clause, source_text)
            key = (etype, ed["src"], ed["dst"], ed["source_clause"], ed["source_text"])
            g.add_edge(
                etype,
                {
                    "src": ed["src"],
                    "dst": ed["dst"],
                    "source_clause": ed["source_clause"],
                    "source_text": ed["source_text"],
                },
                key=key,
            )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

# Semantic labels — bắt buộc có mentioned_in
SEMANTIC_LABELS = set(LLM_KEY_TO_LABEL.values()) | {"LegalConcept"}


def filter_orphan_edges(g: _Graph) -> dict[str, int]:
    """Loại edges có src/dst không tồn tại trong nodes (orphan).

    Hiện tượng phổ biến: LLM tạo edge với entity id mà chưa khai báo entity
    đó trong cùng output → để đảm bảo graph thống nhất, loại bỏ.

    Trả về {edge_type: count_dropped}.
    """
    all_ids: set[str] = set()
    for nodes in g.nodes.values():
        all_ids.update(nodes.keys())

    dropped: dict[str, int] = defaultdict(int)
    for etype, edges in list(g.edges.items()):
        kept = []
        for e in edges:
            if e["src"] not in all_ids or e["dst"] not in all_ids:
                dropped[etype] += 1
                continue
            kept.append(e)
        g.edges[etype] = kept
    return dict(dropped)


def validate(g: _Graph, law_inputs: list[dict] | None = None) -> list[str]:
    """Trả về list lỗi cứng (rỗng = OK). Đã filter orphan edges trước đó.

    Chỉ fail nếu:
    - Structural sai (Law/Chapter/Article không đủ).
    - Semantic node có mentioned_in rỗng.
    - source_clause của edge không phải Clause/Point có thực.
    """
    errors: list[str] = []

    # 1. Structural sanity
    if law_inputs:
        expected_laws = len(law_inputs)
        expected_chapters = sum(
            x["metadata"].expected_chapters or 0
            for x in law_inputs
            if x["metadata"].expected_chapters is not None
        )
        expected_articles = sum(
            x["metadata"].expected_articles or 0
            for x in law_inputs
            if x["metadata"].expected_articles is not None
        )
        if len(g.nodes.get("Law", {})) != expected_laws:
            errors.append(f"Phải có {expected_laws} Law node, có {len(g.nodes.get('Law', {}))}")
        if expected_chapters and len(g.nodes.get("Chapter", {})) != expected_chapters:
            errors.append(
                f"Phải có {expected_chapters} Chapter, có {len(g.nodes.get('Chapter', {}))}"
            )
        if expected_articles and len(g.nodes.get("Article", {})) != expected_articles:
            errors.append(
                f"Phải có {expected_articles} Article, có {len(g.nodes.get('Article', {}))}"
            )
    elif "Law" not in g.nodes:
        errors.append("Phải có ít nhất 1 Law node")

    # 2. Semantic nodes: mentioned_in non-empty
    for label in SEMANTIC_LABELS:
        for nid, node in g.nodes.get(label, {}).items():
            mi = node.get("mentioned_in", [])
            if not mi:
                errors.append(f"{label}/{nid} thiếu mentioned_in")
                if len(errors) > 20:
                    return errors

    # 3. Mọi source_clause của edges là Clause/Point/Article có thực
    structural_ids = (
        set(g.nodes.get("Clause", {}).keys())
        | set(g.nodes.get("Point", {}).keys())
        | set(g.nodes.get("Article", {}).keys())
    )
    bad_sc = 0
    for etype, edges in g.edges.items():
        for e in edges:
            sc = e.get("source_clause")
            if sc is not None and sc not in structural_ids:
                errors.append(f"Edge {etype} có source_clause không tồn tại: {sc}")
                bad_sc += 1
                if bad_sc > 20:
                    return errors
    return errors


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_graph(g: _Graph, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "nodes": {label: list(nodes_dict.values()) for label, nodes_dict in g.nodes.items()},
        "edges": dict(g.edges),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_report(g: _Graph, out_path: Path, drop_stats_total: dict | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Báo cáo trích xuất KG — Multi-law Phase 6\n")
    lines.append("## Tổng quan\n")
    total_nodes = sum(len(v) for v in g.nodes.values())
    total_edges = sum(len(v) for v in g.edges.values())
    lines.append(f"- **Tổng số node:** {total_nodes:,}")
    lines.append(f"- **Tổng số edge:** {total_edges:,}\n")

    lines.append("## Nodes theo label\n")
    lines.append("| Label | Số lượng |")
    lines.append("|---|---:|")
    for label in sorted(g.nodes.keys(), key=lambda x: -len(g.nodes[x])):
        lines.append(f"| `{label}` | {len(g.nodes[label]):,} |")
    lines.append("")

    lines.append("## Edges theo type\n")
    lines.append("| Type | Số lượng |")
    lines.append("|---|---:|")
    for etype in sorted(g.edges.keys(), key=lambda x: -len(g.edges[x])):
        lines.append(f"| `{etype}` | {len(g.edges[etype]):,} |")
    lines.append("")

    # Top entities theo degree
    def _degree(node_id: str) -> int:
        return sum(
            1
            for edges in g.edges.values()
            for e in edges
            if e["src"] == node_id or e["dst"] == node_id
        )

    lines.append("## Top 10 Subjects theo độ degree\n")
    subjs = list(g.nodes.get("Subject", {}).values())
    subjs_with_deg = sorted(
        ((s, _degree(s["id"])) for s in subjs),
        key=lambda x: -x[1],
    )[:10]
    lines.append("| Name | Degree | Xuất hiện trong (#Clause) |")
    lines.append("|---|---:|---:|")
    for s, deg in subjs_with_deg:
        lines.append(f"| {s['name']} | {deg} | {len(s.get('mentioned_in', []))} |")
    lines.append("")

    lines.append("## Top 10 Benefits\n")
    bens = list(g.nodes.get("Benefit", {}).values())
    bens_with_deg = sorted(
        ((b, _degree(b["id"])) for b in bens),
        key=lambda x: -x[1],
    )[:10]
    lines.append("| Name | Category | Degree |")
    lines.append("|---|---|---:|")
    for b, deg in bens_with_deg:
        lines.append(f"| {b['name']} | `{b.get('category', '?')}` | {deg} |")
    lines.append("")

    lines.append("## External laws được viện dẫn\n")
    lines.append("| Code | Title | #Cite |")
    lines.append("|---|---|---:|")
    for ext in g.nodes.get("ExternalLaw", {}).values():
        cite_count = sum(1 for e in g.edges.get("CITES_EXTERNAL", []) if e["dst"] == ext["id"])
        lines.append(f"| `{ext.get('code') or '—'}` | {ext['title']} | {cite_count} |")
    lines.append("")

    # Article degree (semantic)
    lines.append("## Article có ít/nhiều semantic edges nhất\n")
    art_edge_counts: dict[str, int] = defaultdict(int)
    semantic_types = {
        "ENTITLED_TO",
        "HAS_OBLIGATION",
        "HAS_RIGHT",
        "APPLIES_TO",
        "REQUIRES",
        "PAID_FROM",
        "MANAGES",
        "RESPONSIBLE_FOR",
        "PROHIBITED_BY",
        "DEFINES",
    }
    for et in semantic_types:
        for e in g.edges.get(et, []):
            sc = e.get("source_clause", "")
            if sc:
                art_id = ".".join(sc.split(".")[:2])  # L41_2024.A64
                art_edge_counts[art_id] += 1
    sorted_arts = sorted(art_edge_counts.items(), key=lambda x: -x[1])
    lines.append("**Top 10 Article có nhiều semantic edges nhất:**\n")
    lines.append("| Article | #edges |")
    lines.append("|---|---:|")
    for art_id, n in sorted_arts[:10]:
        art = g.nodes["Article"].get(art_id, {})
        lines.append(f"| {art_id} — {art.get('title', '?')[:60]} | {n} |")

    all_art_ids = set(g.nodes["Article"].keys())
    arts_no_semantic = sorted(all_art_ids - set(art_edge_counts.keys()))
    lines.append(
        f"\n**Số Article không có semantic edge nào:** "
        f"{len(arts_no_semantic)} / {len(all_art_ids)}"
    )
    if arts_no_semantic:
        lines.append("\n<details><summary>Danh sách (cần review B3 prompt hoặc rerun)</summary>\n")
        for aid in arts_no_semantic[:50]:
            art = g.nodes["Article"][aid]
            lines.append(f"- {aid}: {art.get('title', '?')}")
        if len(arts_no_semantic) > 50:
            lines.append(f"- ... và {len(arts_no_semantic) - 50} điều khác")
        lines.append("\n</details>")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("Loading sources...")
    src = load_all()
    print(f"  laws         : {len(src['laws'])}")
    for item in src["laws"]:
        print(
            f"    {item['law_id']:<10} "
            f"chapters={len(item['structured']['chapters'])} "
            f"internal_refs={len(item['internal_refs'])} "
            f"external_refs={len(item['external_refs'])} "
            f"definitions={len(item['definitions'])} "
            f"amendments={len(item['amendments'])}"
        )
    print(f"  llm files     : {len(src['llm_files'])}")

    g = _Graph()
    print("\nBuilding graph...")
    for item in src["laws"]:
        build_structural(g, item["structured"], item["metadata"])
    build_law_relations(g, src["laws"])
    for item in src["laws"]:
        build_definitions(g, item["definitions"])
        build_external_refs(g, item["external_refs"])
        build_cross_law_refs(g, item["external_refs"])
        build_amendments(g, item["amendments"])
        build_internal_refs(g, item["internal_refs"])
    build_from_llm(g, src["llm_files"])

    # Filter orphan edges (LLM tạo edge nhưng quên khai báo entity)
    dropped = filter_orphan_edges(g)
    n_dropped = sum(dropped.values())
    if n_dropped:
        print(f"\nLoại {n_dropped} orphan edges (src/dst không có entity tương ứng):")
        for etype, n in sorted(dropped.items(), key=lambda x: -x[1]):
            print(f"  {etype:<22} {n:>4}")

    print("\n=== STATS ===")
    print("Nodes:")
    for label in sorted(g.nodes.keys(), key=lambda x: -len(g.nodes[x])):
        print(f"  {label:<18} {len(g.nodes[label]):>5}")
    print("Edges:")
    for etype in sorted(g.edges.keys(), key=lambda x: -len(g.edges[x])):
        print(f"  {etype:<22} {len(g.edges[etype]):>5}")

    print("\n=== VALIDATE ===")
    errors = validate(g, src["laws"])
    if errors:
        for e in errors[:20]:
            print(f"  ✗ {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... và {len(errors) - 20} lỗi khác", file=sys.stderr)
        print(f"\nFAIL: {len(errors)} lỗi validation. Không lưu output.", file=sys.stderr)
        return 2
    print("  OK — graph thống nhất, mọi edge có src/dst hợp lệ + source_clause có thực.")

    write_graph(g, OUT)
    print(f"\nSaved: {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")

    write_report(g, REPORT)
    print(f"Saved: {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
