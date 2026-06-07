"""Context-specific storage: data persistence, hashing, preview, JSONPath."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from .ids import CTX_PREFIX, generate_id
from .models import ContextMetadata, ContextRecord, DataRef
from .store import Store


class ContextStore:
    """Manages context records: stores data, computes hashes, generates previews."""

    def __init__(self, store: Store) -> None:
        self.store = store

    def create(
        self,
        data: Any,
        name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ContextRecord:
        """Create a new context record from raw data."""
        ctx_id = generate_id(CTX_PREFIX)

        # Serialize data to JSON bytes
        data_bytes = json.dumps(data, default=str).encode("utf-8")
        data_hash = hashlib.sha256(data_bytes).hexdigest()

        # Store the raw data
        data_path = self.store.save_raw("contexts", ctx_id, "data.json", data_bytes)

        # Build metadata
        meta = ContextMetadata(**(metadata or {}))
        if meta.item_count is None and isinstance(data, list):
            meta.item_count = len(data)
        if meta.data_shape is None:
            from .models import DataShape
            if isinstance(data, list):
                meta.data_shape = DataShape.FlatList
            elif isinstance(data, dict):
                meta.data_shape = DataShape.Singular
            else:
                meta.data_shape = DataShape.Unknown

        data_ref = DataRef(
            storage="filesystem",
            path=str(data_path),
            hash=data_hash,
            bytes=len(data_bytes),
        )

        record = ContextRecord(
            context_id=ctx_id,
            name=name,
            created_at=datetime.now(timezone.utc),
            data_ref=data_ref,
            metadata=meta,
        )

        # Persist the record
        self.store.save("contexts", ctx_id, record.model_dump(mode="json"))
        return record

    def get(self, context_id: str) -> ContextRecord | None:
        """Load a context record by ID."""
        data = self.store.load("contexts", context_id)
        if data is None:
            return None
        return ContextRecord.model_validate(data)

    def get_data(self, context_id: str) -> Any | None:
        """Load the raw context data."""
        raw = self.store.load_raw("contexts", context_id, "data.json")
        if raw is None:
            return None
        return json.loads(raw)

    def preview(self, context_id: str, max_chars: int = 200) -> str | None:
        """Generate a text preview of the context data."""
        data = self.get_data(context_id)
        if data is None:
            return None
        text = json.dumps(data, default=str)
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    def extract_path(self, context_id: str, json_path: str) -> Any:
        """Extract data using a simple JSONPath expression.

        Supported: "$" (root), "$.key", "$.key1.key2", etc.
        No array indexing or filters — sufficient for all template catalog paths.
        """
        data = self.get_data(context_id)
        if data is None:
            raise ValueError(f"Context {context_id} not found")
        return resolve_json_path(data, json_path)

    def list_ids(self) -> list[str]:
        return self.store.list_ids("contexts")

    def delete(self, context_id: str) -> bool:
        return self.store.delete("contexts", context_id)


def resolve_json_path(data: Any, path: str) -> Any:
    """Resolve a simple JSONPath expression against data.

    Supports: "$" (root), "$.key", "$.key1.key2.key3"
    """
    if path == "$":
        return data
    if not path.startswith("$."):
        raise ValueError(f"Invalid JSONPath: {path!r} (must start with '$' or '$.')")
    keys = path[2:].split(".")
    current = data
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                raise KeyError(f"Key {key!r} not found in path {path!r}")
            current = current[key]
        elif isinstance(current, list):
            try:
                idx = int(key)
                current = current[idx]
            except (ValueError, IndexError) as e:
                raise KeyError(f"Cannot index list with {key!r} in path {path!r}") from e
        else:
            raise KeyError(f"Cannot traverse {type(current).__name__} with key {key!r}")
    return current
