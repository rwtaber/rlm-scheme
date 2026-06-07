"""Gate: human review checkpoints using asyncio.Event for suspend/resume."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from .models import ExecutionState, GateRecord


class GateManager:
    """Manages gate suspension/resume for executions."""

    def __init__(self) -> None:
        self._pending: dict[str, _PendingGate] = {}

    async def suspend(
        self,
        execution_id: str,
        gate_name: str,
        value: Any,
        message: str = "",
        required: bool = True,
    ) -> tuple[str, Any]:
        """Suspend execution at a gate. Returns (decision, value).

        Blocks until resume_execution is called.
        """
        key = f"{execution_id}:{gate_name}"
        event = asyncio.Event()
        self._pending[key] = _PendingGate(
            execution_id=execution_id,
            gate_name=gate_name,
            value=value,
            message=message,
            required=required,
            event=event,
        )
        await event.wait()

        pg = self._pending.pop(key, None)
        if pg is None:
            return "approve", value
        return pg.decision, value

    def resume(
        self,
        execution_id: str,
        gate_name: str,
        decision: str = "approve",
        reason: str | None = None,
    ) -> GateRecord | None:
        """Resume a suspended gate. Returns the GateRecord or None if not found."""
        key = f"{execution_id}:{gate_name}"
        pg = self._pending.get(key)
        if pg is None:
            return None

        pg.decision = decision
        pg.reason = reason
        pg.event.set()

        return GateRecord(
            name=gate_name,
            status="approved" if decision == "approve" else "rejected",
            decided_at=datetime.now(timezone.utc),
            reason=reason,
        )

    def get_pending(self, execution_id: str) -> list[dict[str, Any]]:
        """List pending gates for an execution."""
        result: list[dict[str, Any]] = []
        for key, pg in self._pending.items():
            if pg.execution_id == execution_id:
                result.append({
                    "name": pg.gate_name,
                    "message": pg.message,
                    "value_preview": str(pg.value)[:200],
                    "required": pg.required,
                })
        return result

    def cancel_all(self, execution_id: str) -> int:
        """Cancel all pending gates for an execution. Returns count cancelled."""
        to_cancel = [
            key for key, pg in self._pending.items()
            if pg.execution_id == execution_id
        ]
        for key in to_cancel:
            pg = self._pending[key]  # don't pop — suspend() will pop
            pg.decision = "reject"
            pg.reason = "execution cancelled"
            pg.event.set()
        return len(to_cancel)


class _PendingGate:
    """Internal state for a pending gate."""

    def __init__(
        self,
        execution_id: str,
        gate_name: str,
        value: Any,
        message: str,
        required: bool,
        event: asyncio.Event,
    ) -> None:
        self.execution_id = execution_id
        self.gate_name = gate_name
        self.value = value
        self.message = message
        self.required = required
        self.event = event
        self.decision: str = "approve"
        self.reason: str | None = None
