"""Utilities for Phase 6 direct-Prolog extraction and runtime execution."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_ALIAS_PATH = Path("data/predicate_aliases.yaml")

BUILTIN_PREDICATES = {
    "legal_source",
    "true",
    "fail",
    "is",
    "member",
    "append",
    "length",
    "findall",
    "bagof",
    "setof",
    "sort",
    "is_list",
    "integer",
    "number",
    "atom",
    "nonvar",
    "var",
    "ground",
    "date",
    "format",
    "write",
    "nl",
}


@dataclass(frozen=True)
class PrologCheckResult:
    ok: bool
    returncode: int | None
    stdout: str = ""
    stderr: str = ""
    timeout: bool = False
    command: tuple[str, ...] = ()

    @property
    def diagnostic(self) -> str:
        if self.timeout:
            return "SWI-Prolog consult timed out"
        text = "\n".join(x for x in (self.stdout, self.stderr) if x).strip()
        return text or f"SWI-Prolog exited with returncode={self.returncode}"


@dataclass(frozen=True)
class PrologExecutionResult:
    success: bool
    status: str
    output: str = ""
    error: str = ""
    returncode: int | None = None
    timeout: bool = False
    command: tuple[str, ...] = ()


def load_predicate_aliases(path: Path | str = DEFAULT_ALIAS_PATH) -> dict[str, str]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    aliases = raw.get("aliases") or {}
    out: dict[str, str] = {}
    for canonical, values in aliases.items():
        canonical = str(canonical).strip()
        if not canonical:
            continue
        out[canonical] = canonical
        for alias in values or []:
            alias = str(alias).strip()
            if alias:
                out[alias] = canonical
    return out


def canonical_predicate(name: str, alias_map: dict[str, str] | None = None) -> str:
    alias_map = alias_map or load_predicate_aliases()
    name = (name or "").strip()
    return alias_map.get(name, name)


_PRED_RE = re.compile(r"(?<![A-Za-z0-9_])([a-z][a-z0-9_]*)\s*\(")
_NAMESPACED_SUFFIX_RE = re.compile(r"_(l\d+_\d{4})$", re.IGNORECASE)


def extract_predicates(source: str) -> list[dict[str, Any]]:
    """Conservatively extract predicate names and arity from Prolog source."""
    found: dict[tuple[str, int], dict[str, Any]] = {}
    for m in _PRED_RE.finditer(source or ""):
        name = m.group(1)
        if name in BUILTIN_PREDICATES:
            continue
        arity = _arity_at(source, m.end() - 1)
        found[(name, arity)] = {"name": name, "arity": arity}
    return list(found.values())


def _arity_at(source: str, open_paren_idx: int) -> int:
    depth = 0
    commas = 0
    in_single = False
    escaped = False
    for i in range(open_paren_idx, len(source)):
        ch = source[i]
        if in_single:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_single = False
            continue
        if ch == "'":
            in_single = True
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                inner = source[open_paren_idx + 1 : i].strip()
                return 0 if not inner else commas + 1
        elif ch == "," and depth == 1:
            commas += 1
    return 0


def namespace_predicate_name(
    name: str,
    law_code: str,
    alias_map: dict[str, str] | None = None,
) -> str:
    lower_law = law_code.lower()
    base = _NAMESPACED_SUFFIX_RE.sub("", name)
    canonical = canonical_predicate(base, alias_map)
    if canonical in BUILTIN_PREDICATES:
        return canonical
    return f"{canonical}_{lower_law}"


def namespace_prolog_source(
    source: str,
    law_code: str,
    alias_map: dict[str, str] | None = None,
    exclude: set[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Namespace domain predicates while preserving `legal_source/6`.

    This is lexical and conservative by design. It only rewrites lowercase
    predicate atoms immediately followed by `(` and never rewrites operators,
    variables, quoted strings, or known Prolog built-ins.
    """
    alias_map = alias_map or load_predicate_aliases()
    exclude = exclude or {"legal_source"}
    rewrites: list[dict[str, Any]] = []

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name in exclude or name in BUILTIN_PREDICATES:
            return match.group(0)
        namespaced = namespace_predicate_name(name, law_code, alias_map)
        if namespaced != name:
            rewrites.append({"from": name, "to": namespaced})
        return match.group(0).replace(name, namespaced, 1)

    return _PRED_RE.sub(repl, source or ""), rewrites


def consult_prolog(source: str, timeout_s: int = 5) -> PrologCheckResult:
    """Consult Prolog source using a real SWI-Prolog subprocess."""
    with tempfile.TemporaryDirectory(prefix="legal_prolog_") as td:
        path = Path(td) / "program.pl"
        path.write_text(source or "", encoding="utf-8")
        cmd = ("swipl", "--no-tty", "-q", "-t", "halt", str(path))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return PrologCheckResult(
                ok=False,
                returncode=None,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timeout=True,
                command=cmd,
            )
        return PrologCheckResult(
            ok=proc.returncode == 0 and "ERROR:" not in (proc.stderr or ""),
            returncode=proc.returncode,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            command=cmd,
        )


def execute_prolog_query(
    source: str,
    query: str,
    answer_var: str = "Trace",
    timeout_s: int = 5,
) -> PrologExecutionResult:
    query = sanitize_runtime_prolog_text((query or "").strip().rstrip("."))
    answer_var = (answer_var or "Trace").strip()
    if not query:
        return PrologExecutionResult(False, "invalid_query", error="empty query")
    consult = consult_prolog(source, timeout_s=timeout_s)
    if not consult.ok:
        return PrologExecutionResult(
            success=False,
            status="prolog_error",
            output=consult.stdout.strip(),
            error=consult.diagnostic,
            returncode=consult.returncode,
            timeout=consult.timeout,
            command=consult.command,
        )
    with tempfile.TemporaryDirectory(prefix="legal_prolog_exec_") as td:
        path = Path(td) / "program.pl"
        path.write_text(source or "", encoding="utf-8")
        goal = (
            f"((once(({query})) -> write_canonical({answer_var}), nl, halt(0); "
            "writeln('__NO_SOLUTION__'), halt(2)))"
        )
        cmd = ("swipl", "--no-tty", "-q", "-l", str(path), "-g", goal)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            return PrologExecutionResult(
                success=False,
                status="timeout",
                output=exc.stdout or "",
                error=exc.stderr or "SWI-Prolog execution timed out",
                timeout=True,
                command=cmd,
            )
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        output = stdout.strip()
        if proc.returncode == 0 and "__NO_SOLUTION__" not in stdout and "ERROR:" not in stderr:
            if output == "_" or re.fullmatch(r"_G\d+", output):
                return PrologExecutionResult(
                    success=False,
                    status="unbound_answer",
                    output=output,
                    error=stderr.strip(),
                    returncode=proc.returncode,
                    command=cmd,
                )
            return PrologExecutionResult(
                success=True,
                status="success",
                output=output,
                error=stderr.strip(),
                returncode=proc.returncode,
                command=cmd,
            )
        status = "prolog_error" if "ERROR:" in stderr else "no_solution"
        return PrologExecutionResult(
            success=False,
            status=status,
            output=stdout.strip(),
            error=(stderr or stdout).strip(),
            returncode=proc.returncode,
            command=cmd,
        )


def strip_json_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    return json.loads(strip_json_fence(text))


def compose_prolog(legal_sources_pl: str, prolog_source: str) -> str:
    parts = []
    if legal_sources_pl and legal_sources_pl.strip():
        parts.append(ensure_clause_periods(sanitize_prolog_atoms(legal_sources_pl.strip())))
    if prolog_source and prolog_source.strip():
        parts.append(ensure_clause_periods(sanitize_prolog_atoms(prolog_source.strip())))
    return "\n\n".join(parts)


def sanitize_prolog_atoms(source: str) -> str:
    """Deterministic lexical cleanup for known non-ASCII atom hazards."""
    out = []
    in_single = False
    escaped = False
    for ch in source or "":
        if in_single:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_single = False
            continue
        if ch == "'":
            in_single = True
            out.append(ch)
        elif ord(ch) > 127:
            out.append(_fold_non_ascii(ch))
        else:
            out.append(ch)
    return "".join(out)


def _fold_non_ascii(ch: str) -> str:
    if ch in {"đ", "Đ", "Ä‘", "Ä"}:
        return "d" if ch in {"đ", "Ä‘"} else "D"
    folded = unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode("ascii")
    return folded or "_"


_ATOM_PHRASE_RE = re.compile(
    r"(?P<prefix>[(,\[]\s*)"
    r"(?P<atom>[a-z][a-z0-9_]*(?:\s+[a-z][a-z0-9_]*)+)"
    r"(?P<suffix>\s*(?=[,\])]))"
)


def sanitize_runtime_prolog_text(source: str) -> str:
    """Clean user-generated facts/query while preserving stored legal rules."""
    text = sanitize_prolog_atoms(source or "")

    def repl(match: re.Match[str]) -> str:
        atom = re.sub(r"\s+", "_", match.group("atom").strip())
        return f"{match.group('prefix')}{atom}{match.group('suffix')}"

    previous = None
    while previous != text:
        previous = text
        text = _ATOM_PHRASE_RE.sub(repl, text)
    return text


def ensure_clause_periods(source: str) -> str:
    lines = []
    for raw in (source or "").splitlines():
        line = raw.rstrip()
        if not line:
            lines.append(line)
            continue
        if line.endswith((".", ":-", "->", ";", ",")):
            lines.append(line)
        elif re.match(r"^[a-z][a-z0-9_]*\s*\(.*\)\s*$", line):
            lines.append(line + ".")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_and_namespace_record(
    extraction: dict[str, Any],
    law_code: str,
    timeout_s: int = 5,
) -> dict[str, Any]:
    legal_sources_pl = str(extraction.get("legal_sources_pl") or "")
    prolog_source = str(extraction.get("prolog_source") or "")
    raw_program = compose_prolog(legal_sources_pl, prolog_source)
    if not prolog_source.strip() and legal_sources_pl.strip():
        sources_check = consult_prolog(compose_prolog(legal_sources_pl, ""), timeout_s=timeout_s)
        if not sources_check.ok:
            return {
                "ok": False,
                "stage": "legal_sources_consult",
                "diagnostic": sources_check.diagnostic,
                "stdout": sources_check.stdout,
                "stderr": sources_check.stderr,
            }
        return {
            "ok": True,
            "stage": "passed_no_rule",
            "no_prolog_rule": True,
            "prolog_source_namespaced": "",
            "legal_sources_pl": ensure_clause_periods(legal_sources_pl),
            "predicate_rewrites": [],
            "predicates_raw": [],
            "predicates_namespaced": [],
        }
    if not prolog_source.strip():
        return {
            "ok": False,
            "stage": "schema",
            "diagnostic": "missing prolog_source",
        }

    raw_check = consult_prolog(raw_program, timeout_s=timeout_s)
    if not raw_check.ok:
        return {
            "ok": False,
            "stage": "raw_consult",
            "diagnostic": raw_check.diagnostic,
            "stdout": raw_check.stdout,
            "stderr": raw_check.stderr,
        }

    alias_map = load_predicate_aliases()
    namespaced_prolog, rewrites = namespace_prolog_source(prolog_source, law_code, alias_map)
    namespaced_program = compose_prolog(legal_sources_pl, namespaced_prolog)
    namespaced_check = consult_prolog(namespaced_program, timeout_s=timeout_s)
    if not namespaced_check.ok:
        return {
            "ok": False,
            "stage": "namespaced_consult",
            "diagnostic": namespaced_check.diagnostic,
            "stdout": namespaced_check.stdout,
            "stderr": namespaced_check.stderr,
            "rewrites": rewrites,
        }

    return {
        "ok": True,
        "stage": "passed",
        "prolog_source_namespaced": namespaced_prolog,
        "legal_sources_pl": ensure_clause_periods(legal_sources_pl),
        "predicate_rewrites": rewrites,
        "predicates_raw": extract_predicates(prolog_source),
        "predicates_namespaced": extract_predicates(namespaced_prolog),
    }
