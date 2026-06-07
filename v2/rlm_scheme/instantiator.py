"""Template instantiator: slot validation, safe substitution, artifact creation."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

from .ids import ART_PREFIX, generate_id
from .models import ArtifactRecord, DataRef, InstantiatorInfo, StaticProfile
from .store import Store
from .template_store import Template

# Primitives allowed in instantiated artifacts (section 9).
ALLOWED_PRIMITIVES = frozenset({
    "llm-query", "llm-query-async", "await", "await-all", "await-any",
    "map-async", "parallel", "race",
    "tree-reduce", "fold-sequential",
    "sequence", "choose", "iterate-until",
    "gate", "recursive-spawn",
    "memoized", "with-validation", "try-fallback",
    "checkpoint", "restore", "tokens-used", "rate-limits", "heartbeat",
    "py-set!", "py-exec", "py-eval", "py-call",
    "syntax-e", "datum->syntax",
    "__context-ref", "__join-json",
    "finish",
    # Racket builtins allowed in templates
    "define", "let", "let*", "lambda", "if", "cond", "when", "unless",
    "begin", "and", "or", "not",
    "list", "cons", "car", "cdr", "map", "filter", "for/list",
    "hash", "hash-ref", "hash-set", "make-hash", "make-hasheq", "hasheq",
    "string-append", "format", "number->string",
    "null?", "pair?", "string?", "number?", "hash?",
    "equal?", "length", "append", "reverse",
    "in-list", "in-naturals", "in-range",
    "values", "apply",
    "remainder", "add1", "sub1",
    "string-append", "or",
})

# Dangerous identifiers that must not appear
FORBIDDEN_IDENTIFIERS = frozenset({
    "eval", "system", "require", "dynamic-require",
    "subprocess", "open-input-file", "open-output-file",
    "delete-file", "rename-file-or-directory",
    "tcp-connect", "tcp-listen", "udp-open-socket",
    "current-directory", "filesystem-change-evt",
})


class InstantiationError(Exception):
    """Raised when template instantiation fails."""
    pass


def instantiate(
    template: Template,
    slot_values: dict[str, Any],
    context_ids: list[str] | None = None,
    plan_id: str | None = None,
    store: Store | None = None,
) -> ArtifactRecord:
    """Instantiate a template with concrete slot values.

    1. Validate slot values against the template's slot schema
    2. Substitute {{slot}} markers with concrete values
    3. Verify no forbidden identifiers
    4. Hash the result
    5. Create and persist an ArtifactRecord
    """
    # Step 1: Validate slots
    _validate_slots(template, slot_values)

    # Step 2: Substitute
    code = _substitute_slots(template.body, slot_values, template.slots_schema)

    # Step 3: Check for forbidden identifiers
    _check_forbidden(code)

    # Step 4: Detect primitives used
    primitives_used = _detect_primitives(code)

    # Step 5: Hash
    code_bytes = code.encode("utf-8")
    code_hash = hashlib.sha256(code_bytes).hexdigest()

    # Step 6: Create artifact record
    art_id = generate_id(ART_PREFIX)

    # Persist code if store provided
    data_ref_path = f"artifacts/{art_id}/code.rkt"
    if store:
        store.save_raw("artifacts", art_id, "code.rkt", code_bytes)
        data_ref_path = str(store.root / "artifacts" / art_id / "code.rkt")

    # Build static profile from template metadata
    profile = _build_profile(template)

    record = ArtifactRecord(
        artifact_id=art_id,
        created_at=datetime.now(timezone.utc),
        plan_id=plan_id,
        context_ids=context_ids or [],
        template_name=template.name,
        template_version=template.version,
        slot_values=slot_values,
        instantiator=InstantiatorInfo(),
        generated_scheme_ref=DataRef(
            storage="filesystem",
            path=data_ref_path,
            hash=code_hash,
            bytes=len(code_bytes),
        ),
        primitives_used=primitives_used,
        static_profile=profile,
    )

    if store:
        store.save("artifacts", art_id, record.model_dump(mode="json"))

    return record


def get_instantiated_code(art_id: str, store: Store) -> str | None:
    """Retrieve the instantiated Scheme code for an artifact."""
    raw = store.load_raw("artifacts", art_id, "code.rkt")
    if raw is None:
        return None
    return raw.decode("utf-8")


def _validate_slots(template: Template, values: dict[str, Any]) -> None:
    """Validate slot values against the template's slot schema."""
    schema = template.slots_schema
    if not schema:
        return

    schema_names: set[str] = set()
    for slot_def in schema:
        if not isinstance(slot_def, list) or not slot_def:
            continue
        name = slot_def[0]
        schema_names.add(name)
        props = _slot_props(slot_def)

        required = props.get("required", False)
        has_default = "default" in props

        if name not in values:
            if required and not has_default:
                raise InstantiationError(
                    f"Required slot '{name}' not provided"
                )
            continue

        value = values[name]

        # Type check
        slot_type = props.get("type")
        if slot_type == "string" and not isinstance(value, str):
            raise InstantiationError(
                f"Slot '{name}' must be a string, got {type(value).__name__}"
            )
        if slot_type == "integer" and not isinstance(value, int):
            raise InstantiationError(
                f"Slot '{name}' must be an integer, got {type(value).__name__}"
            )
        if slot_type == "number" and not isinstance(value, (int, float)):
            raise InstantiationError(
                f"Slot '{name}' must be a number, got {type(value).__name__}"
            )
        if slot_type == "boolean" and not isinstance(value, bool):
            raise InstantiationError(
                f"Slot '{name}' must be a boolean, got {type(value).__name__}"
            )

        # Pattern check
        pattern = props.get("pattern")
        if pattern and isinstance(value, str) and not re.search(pattern, value):
            raise InstantiationError(
                f"Slot '{name}' value '{value}' does not match pattern '{pattern}'"
            )

        # Range checks
        if "min" in props and isinstance(value, (int, float)) and value < props["min"]:
            raise InstantiationError(
                f"Slot '{name}' value {value} below minimum {props['min']}"
            )
        if "max" in props and isinstance(value, (int, float)) and value > props["max"]:
            raise InstantiationError(
                f"Slot '{name}' value {value} above maximum {props['max']}"
            )

        # Min-length
        if "min-length" in props and isinstance(value, str) and len(value) < props["min-length"]:
            raise InstantiationError(
                f"Slot '{name}' length {len(value)} below minimum {props['min-length']}"
            )

    # Warn about unknown slots (not an error — allows forward compatibility)


def _slot_props(slot_def: list) -> dict[str, Any]:
    """Extract properties from a slot definition like (name (type string) (min 1))."""
    props: dict[str, Any] = {}
    for item in slot_def[1:]:
        if isinstance(item, list) and len(item) >= 2:
            key = item[0]
            value = item[1]
            props[key] = value
    return props


def _substitute_slots(
    body: str,
    values: dict[str, Any],
    schema: list[Any],
) -> str:
    """Replace {{slot_name}} markers with concrete values."""
    # Build defaults from schema
    defaults: dict[str, Any] = {}
    for slot_def in schema:
        if isinstance(slot_def, list) and slot_def:
            name = slot_def[0]
            props = _slot_props(slot_def)
            if "default" in props:
                defaults[name] = props["default"]

    def replacer(match: re.Match) -> str:
        name = match.group(1)
        if name in values:
            return _format_value(values[name])
        if name in defaults:
            return _format_value(defaults[name])
        raise InstantiationError(f"No value for slot '{name}'")

    return re.sub(r"\{\{(\w+)\}\}", replacer, body)


def _format_value(value: Any) -> str:
    """Format a Python value for Scheme code insertion."""
    if isinstance(value, bool):
        return "#t" if value else "#f"
    if isinstance(value, str):
        # Strings get quoted in templates via "{{slot}}" already,
        # so just return the raw value for string slots.
        # But if inserted into a non-quoted context, escape properly.
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "#f"
    return str(value)


def _check_forbidden(code: str) -> None:
    """Check that no forbidden identifiers appear in the code."""
    # Simple token scan — not perfect but catches obvious misuse
    tokens = re.findall(r"[a-zA-Z_][\w!?-]*", code)
    for token in tokens:
        if token in FORBIDDEN_IDENTIFIERS:
            raise InstantiationError(
                f"Forbidden identifier '{token}' in instantiated code"
            )


def _detect_primitives(code: str) -> list[str]:
    """Detect which RLM primitives are used in the code."""
    found: list[str] = []
    # Only check the actual RLM primitives (not builtins)
    rlm_primitives = {
        "llm-query", "llm-query-async", "await", "await-all", "await-any",
        "map-async", "parallel", "race",
        "tree-reduce", "fold-sequential",
        "sequence", "choose", "iterate-until",
        "gate", "recursive-spawn",
        "memoized", "with-validation", "try-fallback",
        "checkpoint", "restore", "tokens-used", "rate-limits", "heartbeat",
        "py-set!", "py-exec", "py-eval", "py-call",
        "syntax-e", "datum->syntax",
        "__context-ref", "__join-json",
        "finish",
    }
    for prim in rlm_primitives:
        # Match as a word boundary (roughly)
        if re.search(r"(?<![a-zA-Z_-])" + re.escape(prim) + r"(?![a-zA-Z_!?-])", code):
            found.append(prim)
    return sorted(found)


def _build_profile(template: Template) -> StaticProfile:
    """Build StaticProfile from template's structural-profile metadata."""
    sp = template.structural_profile
    if not sp:
        return StaticProfile()

    props: dict[str, Any] = {}
    for item in sp:
        if isinstance(item, list) and len(item) >= 2:
            props[item[0]] = item[1]

    return StaticProfile(
        expected_calls_formula=props.get("expected-calls"),
        max_concurrency=props.get("max-concurrency-slot") if isinstance(props.get("max-concurrency-slot"), int) else None,
        recursive_depth=props.get("recursive-depth", 0),
        uses_python_bridge=props.get("uses-python-bridge", False),
        uses_multimodal=props.get("uses-multimodal", False),
    )
