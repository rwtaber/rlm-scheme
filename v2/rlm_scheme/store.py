"""Generic filesystem JSON store for durable records."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_DEFAULT_ROOT = os.environ.get("RLM_STORE_ROOT", ".rlm_data")


class Store:
    """Simple filesystem store: one JSON file per record, organized by type."""

    def __init__(self, root: str | Path = _DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _dir(self, record_type: str) -> Path:
        d = self.root / record_type
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, record_type: str, record_id: str) -> Path:
        return self._dir(record_type) / f"{record_id}.json"

    def save(self, record_type: str, record_id: str, data: dict[str, Any]) -> Path:
        """Write a record to disk. Returns the file path."""
        path = self._path(record_type, record_id)
        path.write_text(json.dumps(data, default=str, indent=2))
        return path

    def load(self, record_type: str, record_id: str) -> dict[str, Any] | None:
        """Load a record from disk. Returns None if not found."""
        path = self._path(record_type, record_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def exists(self, record_type: str, record_id: str) -> bool:
        return self._path(record_type, record_id).exists()

    def list_ids(self, record_type: str) -> list[str]:
        """List all record IDs for a given type."""
        d = self._dir(record_type)
        return [p.stem for p in d.glob("*.json")]

    def delete(self, record_type: str, record_id: str) -> bool:
        """Delete a record. Returns True if it existed."""
        path = self._path(record_type, record_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self, record_type: str) -> int:
        """Delete all records of a given type. Returns count deleted."""
        d = self._dir(record_type)
        count = 0
        for p in d.glob("*.json"):
            p.unlink()
            count += 1
        return count

    def clear_all(self) -> dict[str, int]:
        """Delete all records. Returns counts per type."""
        counts: dict[str, int] = {}
        if not self.root.exists():
            return counts
        for d in self.root.iterdir():
            if d.is_dir():
                counts[d.name] = self.clear(d.name)
        return counts

    def save_raw(self, record_type: str, record_id: str, filename: str, data: bytes) -> Path:
        """Save raw bytes (e.g., context data) alongside a record."""
        d = self._dir(record_type) / record_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        path.write_bytes(data)
        return path

    def load_raw(self, record_type: str, record_id: str, filename: str) -> bytes | None:
        """Load raw bytes. Returns None if not found."""
        path = self._dir(record_type) / record_id / filename
        if not path.exists():
            return None
        return path.read_bytes()
