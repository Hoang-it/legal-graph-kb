"""Interactive REPL chat với hệ thống RAG.

Usage:
    python -m runtime.chat                  # mặc định
    python -m runtime.chat --top-k 12       # tăng số Clause retrieve
    python -m runtime.chat --no-verify      # bỏ verify citation (nhanh hơn)
    python -m runtime.chat --no-rich        # plain text, không màu

Trong REPL:
    /help               liệt kê lệnh
    /quit  | /exit      thoát
    /sources            xem các Clause được retrieve cho câu trước
    /verify             re-verify citation câu trước
    /verbose [on|off]   bật/tắt log debug
    /clear              clear màn hình
    /save <file>        lưu phiên chat ra file markdown
    <bất kỳ text khác>  → hỏi RAG
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Load env trước khi import bất kỳ thứ gì khác
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
if not (os.environ.get("OPENAI_BASE_URL") or "").strip():
    os.environ.pop("OPENAI_BASE_URL", None)


def _make_console(use_rich: bool):
    """Trả về (console, print_md, print_plain). Fallback nếu rich không có."""
    if use_rich:
        try:
            from rich.console import Console
            from rich.markdown import Markdown
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            console = Console()
            return console, Markdown, Panel, Table, Text
        except ImportError:
            pass
    return None, None, None, None, None


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

BANNER = """\
╔══════════════════════════════════════════════════════════════════════╗
║  Legal KG Chat — Luật Bảo hiểm xã hội 41/2024/QH15                  ║
║  Gõ /help để xem lệnh, /quit để thoát.                              ║
╚══════════════════════════════════════════════════════════════════════╝
"""

HELP = """\
Lệnh:
  /help               liệt kê lệnh
  /quit, /exit, /q    thoát
  /sources            xem các Clause được retrieve cho câu trước
  /verify             re-verify citation câu trước
  /verbose [on|off]   bật/tắt log debug (mặc định off)
  /clear              clear màn hình
  /save <file.md>     lưu lịch sử chat ra file Markdown
Mọi text khác → câu hỏi cho RAG.
"""


def run(top_k: int, auto_verify: bool, use_rich: bool):
    # Lazy import — sau khi env vars đã load
    print("Loading RAG pipeline (BGE-M3 + Neo4j + OpenAI)...", file=sys.stderr)
    from runtime.rag_query import RagPipeline

    pipeline = RagPipeline()
    # Pre-load embed model để câu đầu không chậm
    _ = pipeline.embed_model

    console, Markdown, Panel, Table, Text = _make_console(use_rich)
    rich_on = console is not None

    if rich_on:
        console.print(Panel(BANNER.strip(), border_style="cyan"))
    else:
        print(BANNER)

    history: list[dict] = []
    last_result = None
    verbose = False

    try:
        while True:
            try:
                if rich_on:
                    q = console.input("[bold cyan]bạn> [/]")
                else:
                    q = input("bạn> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break

            q = q.strip()
            if not q:
                continue

            # ---- Commands ----
            if q in ("/quit", "/exit", "/q"):
                break
            if q == "/help":
                print(HELP)
                continue
            if q == "/clear":
                os.system("cls" if os.name == "nt" else "clear")
                continue
            if q.startswith("/verbose"):
                parts = q.split()
                if len(parts) == 1:
                    verbose = not verbose
                else:
                    verbose = parts[1].lower() in ("on", "true", "1")
                print(f"verbose = {verbose}")
                continue
            if q == "/sources":
                if not last_result:
                    print("(chưa có câu nào)")
                    continue
                _show_sources(last_result, console, Table, rich_on)
                continue
            if q == "/verify":
                if not last_result:
                    print("(chưa có câu nào)")
                    continue
                v = pipeline.verify_citations(last_result.citation_ids)
                _show_verify(v, console, Table, rich_on)
                continue
            if q.startswith("/save"):
                parts = q.split(maxsplit=1)
                if len(parts) < 2:
                    print("Cú pháp: /save <file.md>")
                    continue
                _save_history(history, Path(parts[1]))
                print(f"Đã lưu {len(history)} câu vào {parts[1]}")
                continue
            if q.startswith("/"):
                print(f"Không hiểu lệnh: {q}. Gõ /help.")
                continue

            # ---- Hỏi RAG ----
            t0 = time.time()
            try:
                last_result = pipeline.ask(q, top_k=top_k, verbose=verbose)
            except KeyboardInterrupt:
                print("\n(huỷ)")
                continue
            except Exception as e:
                print(f"\n✗ Lỗi: {type(e).__name__}: {e}", file=sys.stderr)
                continue

            elapsed = time.time() - t0
            verified = pipeline.verify_citations(last_result.citation_ids) if auto_verify else {}
            _show_answer(q, last_result, verified, elapsed, console, Markdown, Panel, Text, rich_on)

            history.append(
                {
                    "q": q,
                    "answer": last_result.answer,
                    "citations": last_result.citations,
                    "citation_ids": last_result.citation_ids,
                    "verified": verified,
                    "n_hits": len(last_result.hits),
                    "elapsed_s": elapsed,
                }
            )

    finally:
        pipeline.close()
        if rich_on:
            console.print("\n[dim]Tạm biệt![/]")
        else:
            print("\nTạm biệt!")


# ---------------------------------------------------------------------------
# Pretty rendering
# ---------------------------------------------------------------------------


def _show_answer(q, result, verified, elapsed, console, Markdown, Panel, Text, rich_on):
    if rich_on:
        # Answer panel
        console.print()
        console.print(
            Panel(
                Markdown(result.answer),
                title=f"[bold green]Trả lời[/]  ({elapsed:.1f}s)",
                border_style="green",
                padding=(0, 1),
            )
        )
        # Citations
        if result.citations:
            txt = Text()
            for cit, cid in zip(result.citations, result.citation_ids, strict=False):
                if verified:
                    ok = verified.get(cid, False)
                    mark = "✓" if ok else "✗"
                    color = "green" if ok else "red"
                    txt.append(f"  {mark} {cit:<35} → {cid}\n", style=color)
                else:
                    txt.append(f"  • {cit:<35} → {cid}\n", style="yellow")
            console.print(Panel(txt, title="[bold]Citations[/]", border_style="yellow"))
        # Stats line
        verify_str = ""
        if verified:
            ok = sum(verified.values())
            verify_str = f"  verify {ok}/{len(verified)} ✓"
        console.print(
            f"[dim]vector hits: {len(result.hits)} | "
            f"semantic edges: {result.n_semantic_edges} | "
            f"refs: {result.n_refs}{verify_str}[/]\n"
        )
    else:
        print(f"\n{'─' * 70}")
        print(result.answer)
        print(f"{'─' * 70}")
        print(f"Citations: {result.citations}")
        if verified:
            ok = sum(verified.values())
            print(f"Verify: {ok}/{len(verified)} ✓")
        print(f"Time: {elapsed:.1f}s")
        print()


def _show_sources(result, console, Table, rich_on):
    if rich_on:
        tbl = Table(title="Top vector hits", show_lines=False)
        tbl.add_column("Score", justify="right", style="cyan")
        tbl.add_column("Clause ID", style="yellow")
        tbl.add_column("Điều", style="magenta")
        tbl.add_column("Text preview", overflow="fold")
        for h in result.hits[:10]:
            tbl.add_row(
                f"{h.score:.3f}",
                h.clause_id,
                f"{h.article_n}",
                h.text[:120] + ("..." if len(h.text) > 120 else ""),
            )
        console.print(tbl)
    else:
        print(f"\nTop {len(result.hits)} vector hits:")
        for h in result.hits[:10]:
            print(f"  {h.score:.3f}  {h.clause_id}: {h.text[:100]}")


def _show_verify(verified, console, Table, rich_on):
    if rich_on:
        tbl = Table(title="Citation verify")
        tbl.add_column("Citation ID", style="yellow")
        tbl.add_column("Tồn tại trong DB", justify="center")
        for cid, ok in verified.items():
            tbl.add_row(cid, "[green]✓[/]" if ok else "[red]✗[/]")
        console.print(tbl)
    else:
        for cid, ok in verified.items():
            print(f"  {'✓' if ok else '✗'} {cid}")


def _save_history(history, path: Path):
    lines = ["# Phiên chat Legal KG\n"]
    for i, h in enumerate(history, 1):
        lines.append(f"\n## Câu {i}: {h['q']}\n")
        lines.append(h["answer"])
        if h.get("citations"):
            lines.append(f"\n**Citations:** {', '.join(h['citations'])}")
        if h.get("verified"):
            ok = sum(h["verified"].values())
            lines.append(f"\n**Verify:** {ok}/{len(h['verified'])} ✓")
        lines.append(f"\n*({h['n_hits']} vector hits, {h['elapsed_s']:.1f}s)*\n")
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description="Interactive REPL cho Legal KG RAG.")
    p.add_argument("--top-k", type=int, default=8, help="Số Clause retrieve mỗi câu (default 8).")
    p.add_argument(
        "--no-verify", action="store_true", help="Bỏ qua verify citation ngược DB (nhanh hơn)."
    )
    p.add_argument("--no-rich", action="store_true", help="Tắt rich formatting, dùng plain text.")
    args = p.parse_args()

    try:
        run(top_k=args.top_k, auto_verify=not args.no_verify, use_rich=not args.no_rich)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
