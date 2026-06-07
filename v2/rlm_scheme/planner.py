"""Planner: template selection + LLM gap-fill + plan records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .classifier import classify
from .context_store import ContextStore
from .ids import PLAN_PREFIX, generate_id
from .llm_provider import LLMProvider
from .models import (
    AlternativeTemplate,
    Classification,
    PlannerInfo,
    PlanRecord,
    RecommendedAction,
    TemplateChainStep,
)
from .store import Store
from .template_store import Template, TemplateStore


class Planner:
    """Creates plan records: classifies task, selects template, fills content slots."""

    def __init__(
        self,
        store: Store,
        context_store: ContextStore,
        template_store: TemplateStore,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        self.store = store
        self.context_store = context_store
        self.template_store = template_store
        self.llm_provider = llm_provider

    async def plan(
        self,
        task: str,
        context_ids: list[str] | None = None,
        hints: dict[str, Any] | None = None,
    ) -> PlanRecord:
        """Create a plan for the given task.

        Level 1: Deterministic classification + template matching.
        Level 2: LLM gap-fill for content slots (instruction strings).
        """
        context_ids = context_ids or []
        hints = hints or {}

        # Gather metadata from contexts
        metadata = self._gather_metadata(context_ids, hints)

        # Level 1: Deterministic classification
        classification = classify(task, metadata, hints)

        # Level 1: Template selection via trigger/reject matching
        selected, alternatives = self._select_template(classification, metadata)

        # Build recommended action
        if selected:
            # Level 2: Fill content slots via LLM if provider available
            slot_values = await self._fill_slots(selected, task, context_ids, hints)
            recommended = RecommendedAction(
                kind="template_invocation",
                template_name=selected.name,
                template_version=selected.version,
                slot_values=slot_values,
            )
            planner_mode = "deterministic_with_llm_fill" if self.llm_provider else "deterministic"
        else:
            recommended = RecommendedAction(
                kind="no_template",
                description=f"No template matched for {classification.task_shape.value}/{classification.data_shape.value}",
            )
            planner_mode = "deterministic"

        plan_id = generate_id(PLAN_PREFIX)
        record = PlanRecord(
            plan_id=plan_id,
            created_at=datetime.now(timezone.utc),
            context_ids=context_ids,
            task=task,
            hints=hints,
            classification=classification,
            recommended=recommended,
            alternatives=[
                AlternativeTemplate(template_name=t.name, tradeoff=_tradeoff(t, selected))
                for t in alternatives
            ],
            planner=PlannerInfo(
                mode=planner_mode,
                model=None,
            ),
        )

        self.store.save("plans", plan_id, record.model_dump(mode="json"))
        return record

    def get(self, plan_id: str) -> PlanRecord | None:
        data = self.store.load("plans", plan_id)
        if data is None:
            return None
        return PlanRecord.model_validate(data)

    def _gather_metadata(
        self, context_ids: list[str], hints: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge context metadata with hints for classification."""
        meta: dict[str, Any] = {}
        for ctx_id in context_ids:
            rec = self.context_store.get(ctx_id)
            if rec and rec.metadata:
                m = rec.metadata.model_dump(exclude_none=True)
                meta.update(m)
        meta.update(hints)
        return meta

    def _select_template(
        self,
        classification: Classification,
        metadata: dict[str, Any],
    ) -> tuple[Template | None, list[Template]]:
        """Select best template and alternatives via trigger/reject matching."""
        task_shape = classification.task_shape.value
        candidates = self.template_store.find_by_task_shape(task_shape)

        # Evaluate triggers and rejects
        matching: list[Template] = []
        for tpl in candidates:
            if self.template_store.evaluate_rejects(tpl, metadata):
                continue
            if self.template_store.evaluate_triggers(tpl, metadata):
                matching.append(tpl)

        if not matching:
            # Fallback: any template with the right task shape
            matching = candidates

        if not matching:
            return None, []

        selected = matching[0]
        alternatives = matching[1:] if len(matching) > 1 else []
        return selected, alternatives

    async def _fill_slots(
        self,
        template: Template,
        task: str,
        context_ids: list[str],
        hints: dict[str, Any],
    ) -> dict[str, Any]:
        """Fill slot values. Required structural slots from hints/context,
        content slots (instructions) via LLM if available."""
        slot_values: dict[str, Any] = {}

        # Pre-fill from hints
        for key, value in hints.items():
            slot_values[key] = value

        # Fill context_id if single context
        if context_ids and "context_id" not in slot_values:
            slot_values["context_id"] = context_ids[0]

        # If LLM provider available, fill instruction slots
        if self.llm_provider and template.slots_schema:
            for slot_def in template.slots_schema:
                if not isinstance(slot_def, list) or not slot_def:
                    continue
                name = slot_def[0]
                if name in slot_values:
                    continue
                props = {}
                for item in slot_def[1:]:
                    if isinstance(item, list) and len(item) >= 2:
                        props[item[0]] = item[1]
                # Only fill string slots that look like instructions
                if props.get("type") == "string" and "instruction" in name.lower():
                    try:
                        result = await self.llm_provider.call(
                            instruction=(
                                f"Given the task: '{task}'\n"
                                f"Generate a clear, specific instruction for the '{name}' slot "
                                f"of the '{template.name}' template. "
                                f"Return only the instruction text, nothing else."
                            ),
                            data=None,
                            model="quality_text_model",
                        )
                        slot_values[name] = result.text.strip()
                    except Exception:
                        pass  # Leave unfilled — will be caught by instantiator validation

        return slot_values


def _tradeoff(alt: Template, selected: Template | None) -> str:
    """Generate a brief tradeoff description for an alternative template."""
    if selected is None:
        return "primary candidate"
    parts: list[str] = []
    if alt.streamable and not (selected and selected.streamable):
        parts.append("supports streaming")
    if alt.cacheable and not (selected and selected.cacheable):
        parts.append("supports caching")
    if alt.uses_llm_generated_code:
        parts.append("uses LLM-generated code")
    return "; ".join(parts) if parts else "alternative approach"
