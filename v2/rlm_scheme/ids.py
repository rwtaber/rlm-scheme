"""ID generation and validation for durable records."""

import re
import secrets

CTX_PREFIX = "ctx_"
PLAN_PREFIX = "plan_"
ART_PREFIX = "art_"
DRY_PREFIX = "dry_"
VER_PREFIX = "ver_"
EXEC_PREFIX = "exec_"
CALL_PREFIX = "call_"
CACHE_PREFIX = "cache_"

ALL_PREFIXES = (
    CTX_PREFIX, PLAN_PREFIX, ART_PREFIX, DRY_PREFIX,
    VER_PREFIX, EXEC_PREFIX, CALL_PREFIX, CACHE_PREFIX,
)

_ID_RE = re.compile(r"^[a-z]+_[0-9a-f]{16}$")


def generate_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix."""
    if prefix not in ALL_PREFIXES:
        raise ValueError(f"Unknown prefix: {prefix!r}")
    return f"{prefix}{secrets.token_hex(8)}"


def validate_id(value: str, expected_prefix: str) -> bool:
    """Check that *value* is a well-formed ID with *expected_prefix*."""
    return (
        isinstance(value, str)
        and value.startswith(expected_prefix)
        and _ID_RE.match(value) is not None
    )
