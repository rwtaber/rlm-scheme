"""Trace event collection and persistence."""

from __future__ import annotations

import time
from typing import Any

from .models import ScopeLogEntry, TraceDetail, TraceEvent
from .store import Store


class TraceCollector:
    """Collects trace events and scope log entries during execution."""

    def __init__(self, execution_id: str, artifact_id: str | None = None, plan_id: str | None = None) -> None:
        self.execution_id = execution_id
        self.artifact_id = artifact_id
        self.plan_id = plan_id
        self.events: list[TraceEvent] = []
        self.scope_log: list[ScopeLogEntry] = []
        self.stdout_parts: list[str] = []
        self._start_time = time.monotonic()

    def record_call_start(
        self,
        call_id: str,
        model: str,
        node_id: str | None = None,
        depth: int = 0,
    ) -> None:
        self.events.append(TraceEvent(
            type="call_start",
            call_id=call_id,
            model=model,
            node_id=node_id,
            depth=depth,
            elapsed_seconds=time.monotonic() - self._start_time,
        ))

    def record_call_end(
        self,
        call_id: str,
        tokens: int = 0,
        model: str | None = None,
    ) -> None:
        self.events.append(TraceEvent(
            type="call_end",
            call_id=call_id,
            tokens=tokens,
            model=model,
            elapsed_seconds=time.monotonic() - self._start_time,
        ))

    def record_call_error(
        self,
        call_id: str,
        error: str,
        model: str | None = None,
    ) -> None:
        self.events.append(TraceEvent(
            type="call_error",
            call_id=call_id,
            model=model,
            elapsed_seconds=time.monotonic() - self._start_time,
        ))

    def record_cache_hit(self, call_id: str) -> None:
        self.events.append(TraceEvent(
            type="cache_hit",
            call_id=call_id,
            elapsed_seconds=time.monotonic() - self._start_time,
        ))

    def record_gate(self, name: str, status: str) -> None:
        self.events.append(TraceEvent(
            type=f"gate_{status}",
            node_id=name,
            elapsed_seconds=time.monotonic() - self._start_time,
        ))

    def record_checkpoint(self, key: str) -> None:
        self.events.append(TraceEvent(
            type="checkpoint",
            node_id=key,
            elapsed_seconds=time.monotonic() - self._start_time,
        ))

    def log_scope(
        self,
        op: str,
        preview: str | None = None,
        scope: str | None = None,
        call_id: str | None = None,
    ) -> None:
        """Record a scope/provenance log entry (syntax-e unwrap, datum->syntax wrap, etc.)."""
        self.scope_log.append(ScopeLogEntry(
            op=op, preview=preview, scope=scope, call_id=call_id,
        ))

    def append_stdout(self, text: str) -> None:
        self.stdout_parts.append(text)

    def build(self) -> TraceDetail:
        return TraceDetail(
            artifact_id=self.artifact_id,
            plan_id=self.plan_id,
            events=self.events,
            scope_log=self.scope_log,
            stdout="".join(self.stdout_parts),
        )

    def persist(self, store: Store) -> None:
        """Save the trace to the store."""
        detail = self.build()
        store.save(
            "traces",
            self.execution_id,
            detail.model_dump(mode="json"),
        )

    @staticmethod
    def load(execution_id: str, store: Store) -> TraceDetail | None:
        data = store.load("traces", execution_id)
        if data is None:
            return None
        return TraceDetail.model_validate(data)
