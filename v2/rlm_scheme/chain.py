"""Template chain execution with $previous references."""

from __future__ import annotations

from typing import Any

from .context_store import ContextStore
from .executor import Executor
from .models import (
    ChainStepResult,
    ExecuteResponse,
    ExecutionPolicy,
    TemplateChainStep,
)
from .store import Store
from .template_store import TemplateStore


class ChainExecutor:
    """Executes multi-step template chains with $previous resolution."""

    def __init__(self, executor: Executor, context_store: ContextStore) -> None:
        self.executor = executor
        self.context_store = context_store

    async def execute_chain(
        self,
        steps: list[TemplateChainStep],
        context_ids: list[str] | None = None,
        plan_id: str | None = None,
        policy: ExecutionPolicy | None = None,
        timeout: float | None = None,
    ) -> ExecuteResponse:
        """Execute a chain of template steps sequentially.

        Each step can reference $previous in slot values, which resolves
        to the context_id of the previous step's result.
        """
        context_ids = context_ids or []
        step_results: list[ChainStepResult] = []
        previous_context_id: str | None = None
        last_response: ExecuteResponse | None = None

        for i, step in enumerate(steps):
            template = self.executor.template_store.get(step.template_name)
            if template is None:
                return ExecuteResponse(
                    status="error",
                    errors=[f"Step {i}: template '{step.template_name}' not found"],
                )

            # Resolve $previous references in slot values
            resolved_slots = _resolve_previous(step.slot_values, previous_context_id)

            # Add context_ids if not already present
            step_context_ids = list(context_ids)
            if previous_context_id and previous_context_id not in step_context_ids:
                step_context_ids.append(previous_context_id)

            # Execute step
            response = await self.executor.execute(
                template=template,
                slot_values=resolved_slots,
                context_ids=step_context_ids,
                plan_id=plan_id,
                policy=policy,
                timeout=timeout,
            )

            if response.status == "error":
                response.errors = [f"Chain failed at step {i}: {e}" for e in (response.errors or [])]
                return response

            # Store result as intermediate context for next step
            intermediate_ctx = None
            if response.result and response.result.value is not None:
                rec = self.context_store.create(
                    response.result.value,
                    name=f"chain_step_{i}_{step.template_name}",
                )
                intermediate_ctx = rec.context_id
                previous_context_id = rec.context_id

            step_results.append(ChainStepResult(
                step=i,
                template=step.template_name,
                intermediate_context_id=intermediate_ctx,
                result=response.result.value if response.result else None,
            ))

            last_response = response

        # Build final response from last step
        if last_response:
            last_response.execution = last_response.execution  # preserve
            if last_response.execution:
                last_response.execution.chain_step_results = step_results
            return last_response

        return ExecuteResponse(
            status="error",
            errors=["Empty chain"],
        )


def _resolve_previous(
    slot_values: dict[str, Any],
    previous_context_id: str | None,
) -> dict[str, Any]:
    """Replace $previous references in slot values with the actual context_id."""
    resolved: dict[str, Any] = {}
    for key, value in slot_values.items():
        if isinstance(value, str) and "$previous" in value:
            if previous_context_id:
                resolved[key] = value.replace("$previous", previous_context_id)
            else:
                resolved[key] = value  # Leave as-is for first step
        else:
            resolved[key] = value
    return resolved
