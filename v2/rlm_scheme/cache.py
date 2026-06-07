"""Content-addressed LLM result cache."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .ids import CACHE_PREFIX, generate_id
from .llm_provider import content_hash
from .models import CacheRecord, TokenUsage
from .store import Store


class LLMCache:
    """Content-addressed cache keyed by sha256(instruction, data, model, temperature)."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self._hits = 0
        self._misses = 0

    def lookup(
        self,
        instruction: str,
        data: Any,
        model: str,
        temperature: float,
    ) -> CacheRecord | None:
        """Look up a cached result. Returns None on miss."""
        key = content_hash(instruction, data, model, temperature)
        raw = self.store.load("cache", key)
        if raw is None:
            self._misses += 1
            return None
        self._hits += 1
        return CacheRecord.model_validate(raw)

    def store_result(
        self,
        instruction: str,
        data: Any,
        model: str,
        temperature: float,
        result: str,
        tokens: TokenUsage | None = None,
        json_mode: bool = False,
        execution_id: str | None = None,
        call_id: str | None = None,
    ) -> CacheRecord:
        """Store a result in the cache."""
        key = content_hash(instruction, data, model, temperature)
        inst_hash = content_hash(instruction, None, "", 0.0)
        data_hash = content_hash("", data, "", 0.0)

        record = CacheRecord(
            cache_key=key,
            created_at=datetime.now(timezone.utc),
            instruction_hash=inst_hash,
            data_hash=data_hash,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
            result=result,
            result_tokens=tokens or TokenUsage(),
            source_execution_id=execution_id,
            source_call_id=call_id,
        )
        self.store.save("cache", key, record.model_dump(mode="json"))
        return record

    def clear(self) -> int:
        """Clear all cached results. Returns count deleted."""
        return self.store.clear("cache")

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
