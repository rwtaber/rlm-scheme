"""Template store: S-expression parser for define-meta, template catalog, trigger/reject."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

_DEFAULT_TEMPLATE_DIR = os.environ.get("RLM_TEMPLATE_DIR", "templates")


# ---------------------------------------------------------------------------
# S-expression tokenizer and parser (~200 lines)
# Handles: parens, strings, numbers, booleans (#t/#f), symbols,
# quoted forms ('(...)), and ; line comments.
# Only used for parsing define-meta forms in .rkt template files.
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)                   |  # whitespace (skip)
    (?P<comment>;[^\n]*)          |  # line comment (skip)
    (?P<lparen>\()                |  # open paren
    (?P<rparen>\))                |  # close paren
    (?P<quote>')                  |  # quote shorthand
    (?P<string>"(?:[^"\\]|\\.)*") |  # double-quoted string
    (?P<atom>[^\s()"';]+)            # symbol, number, boolean
    """,
    re.VERBOSE,
)


def tokenize(source: str) -> list[tuple[str, str]]:
    """Tokenize an S-expression string into (type, value) pairs."""
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup
        value = m.group()
        if kind in ("ws", "comment"):
            continue
        assert kind is not None
        tokens.append((kind, value))
    return tokens


def parse_sexpr(tokens: list[tuple[str, str]], pos: int = 0) -> tuple[Any, int]:
    """Parse one S-expression from tokens starting at pos. Returns (value, new_pos)."""
    if pos >= len(tokens):
        raise ValueError("Unexpected end of input")
    kind, value = tokens[pos]

    if kind == "quote":
        # 'expr -> (quote expr)
        inner, new_pos = parse_sexpr(tokens, pos + 1)
        return inner, new_pos

    if kind == "lparen":
        items: list[Any] = []
        pos += 1
        while pos < len(tokens) and tokens[pos][0] != "rparen":
            item, pos = parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise ValueError("Unclosed parenthesis")
        return items, pos + 1  # skip rparen

    if kind == "rparen":
        raise ValueError("Unexpected ')'")

    if kind == "string":
        # Strip surrounding quotes and unescape
        return value[1:-1].replace('\\"', '"').replace("\\\\", "\\"), pos + 1

    if kind == "atom":
        return _parse_atom(value), pos + 1

    raise ValueError(f"Unexpected token: {kind} {value!r}")


def _parse_atom(value: str) -> Any:
    """Parse an atom: number, boolean, or symbol string."""
    if value == "#t":
        return True
    if value == "#f":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value  # symbol as string


def parse_all(source: str) -> list[Any]:
    """Parse all top-level S-expressions from source."""
    tokens = tokenize(source)
    results: list[Any] = []
    pos = 0
    while pos < len(tokens):
        expr, pos = parse_sexpr(tokens, pos)
        results.append(expr)
    return results


# ---------------------------------------------------------------------------
# Template metadata extraction
# ---------------------------------------------------------------------------

def extract_define_metas(source: str) -> dict[str, Any]:
    """Extract all (define-meta name value) forms from template source."""
    exprs = parse_all(source)
    metas: dict[str, Any] = {}
    for expr in exprs:
        if (
            isinstance(expr, list)
            and len(expr) >= 3
            and expr[0] == "define-meta"
            and isinstance(expr[1], str)
        ):
            metas[expr[1]] = expr[2]
    return metas


def extract_body(source: str) -> str:
    """Extract the template body (everything after define-meta forms)."""
    lines = source.split("\n")
    in_meta = False
    paren_depth = 0
    body_start = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("(define-meta"):
            in_meta = True
            paren_depth = 0
        if in_meta:
            paren_depth += stripped.count("(") - stripped.count(")")
            if paren_depth <= 0:
                in_meta = False
                body_start = i + 1
        elif not stripped or stripped.startswith(";") or stripped.startswith("#lang"):
            body_start = i + 1
        else:
            break

    return "\n".join(lines[body_start:])


def extract_slots(source: str) -> list[str]:
    """Find all {{slot_name}} markers in the template body."""
    return re.findall(r"\{\{(\w+)\}\}", source)


# ---------------------------------------------------------------------------
# Template record
# ---------------------------------------------------------------------------

class Template:
    """A loaded template with parsed metadata and body."""

    def __init__(self, path: Path, source: str) -> None:
        self.path = path
        self.source = source
        self.meta = extract_define_metas(source)
        self.body = extract_body(source)
        self.slot_markers = extract_slots(self.body)

    @property
    def name(self) -> str:
        return self.meta.get("name", self.path.stem)

    @property
    def version(self) -> str:
        return self.meta.get("version", "0.0.0")

    @property
    def summary(self) -> str:
        return self.meta.get("summary", "")

    @property
    def task_shapes(self) -> list[str]:
        raw = self.meta.get("task-shapes", [])
        return raw if isinstance(raw, list) else [raw]

    @property
    def data_shapes(self) -> list[str]:
        raw = self.meta.get("data-shapes", [])
        return raw if isinstance(raw, list) else [raw]

    @property
    def slots_schema(self) -> list[Any]:
        return self.meta.get("slots", [])

    @property
    def trigger(self) -> list[Any]:
        return self.meta.get("trigger", [])

    @property
    def reject(self) -> list[Any]:
        return self.meta.get("reject", [])

    @property
    def structural_profile(self) -> list[Any]:
        return self.meta.get("structural-profile", [])

    @property
    def verification_rules(self) -> list[Any]:
        return self.meta.get("verification-rules", [])

    @property
    def streamable(self) -> bool:
        return self.meta.get("streamable", False)

    @property
    def cacheable(self) -> bool:
        return self.meta.get("cacheable", False)

    @property
    def gates(self) -> list[Any]:
        return self.meta.get("gates", [])

    @property
    def output_schema(self) -> Any:
        return self.meta.get("output-schema", None)

    @property
    def budget_policy(self) -> list[Any]:
        return self.meta.get("budget-policy", [])

    @property
    def error_policies(self) -> list[Any]:
        return self.meta.get("error-policies", [])

    @property
    def uses_llm_generated_code(self) -> bool:
        return self.meta.get("uses-llm-generated-code", False)


# ---------------------------------------------------------------------------
# Template catalog
# ---------------------------------------------------------------------------

class TemplateStore:
    """Loads and indexes all .rkt templates from a directory."""

    def __init__(self, template_dir: str | Path = _DEFAULT_TEMPLATE_DIR) -> None:
        self.template_dir = Path(template_dir)
        self._templates: dict[str, Template] = {}
        if self.template_dir.exists():
            self.reload()

    def reload(self) -> None:
        """(Re-)load all templates from disk."""
        self._templates.clear()
        for path in sorted(self.template_dir.glob("*.rkt")):
            source = path.read_text()
            try:
                tpl = Template(path, source)
                self._templates[tpl.name] = tpl
            except Exception:
                # Skip malformed templates
                continue

    def get(self, name: str) -> Template | None:
        return self._templates.get(name)

    def list_names(self) -> list[str]:
        return sorted(self._templates.keys())

    def all_templates(self) -> list[Template]:
        return list(self._templates.values())

    def find_by_task_shape(self, shape: str) -> list[Template]:
        return [t for t in self._templates.values() if shape in t.task_shapes]

    def find_by_data_shape(self, shape: str) -> list[Template]:
        return [t for t in self._templates.values() if shape in t.data_shapes]

    def evaluate_triggers(
        self, template: Template, context: dict[str, Any]
    ) -> bool:
        """Evaluate a template's trigger conditions against classification context.

        Context keys: item_count, independent, output_type, has_second_phase,
        ordered, order_sensitive, requires_pairwise_comparison, etc.
        """
        trigger = template.trigger
        if not trigger:
            return True  # No trigger = always matches
        return all(_eval_condition(cond, context) for cond in trigger)

    def evaluate_rejects(
        self, template: Template, context: dict[str, Any]
    ) -> bool:
        """Evaluate reject conditions. Returns True if template should be REJECTED."""
        reject = template.reject
        if not reject:
            return False  # No reject = never rejected
        return any(_eval_condition(cond, context) for cond in reject)


def _eval_condition(cond: Any, context: dict[str, Any]) -> bool:
    """Evaluate a single trigger/reject condition expression."""
    if not isinstance(cond, list) or len(cond) < 2:
        return True  # malformed → pass

    op = cond[0]

    if op == ">":
        key, val = cond[1], cond[2]
        return context.get(key, 0) > val
    elif op == "<":
        key, val = cond[1], cond[2]
        return context.get(key, 0) < val
    elif op == ">=":
        key, val = cond[1], cond[2]
        return context.get(key, 0) >= val
    elif op == "eq?":
        key, val = cond[1], cond[2]
        return context.get(key) == val
    elif op == "not":
        return not _eval_condition(cond[1], context)
    elif op == "and":
        return all(_eval_condition(c, context) for c in cond[1:])
    elif op == "or":
        return any(_eval_condition(c, context) for c in cond[1:])
    else:
        return True  # unknown op → pass
