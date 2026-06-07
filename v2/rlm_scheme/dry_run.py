"""Dry run: instantiate + simulate + estimate in one call."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from .ids import DRY_PREFIX, generate_id
from .instantiator import instantiate
from .models import (
    ArtifactSummary,
    CallGraphNode,
    CostEstimate,
    DryRunEstimate,
    DryRunRecord,
    DryRunResponse,
    DryRunSimulation,
    DryRunSummary,
    TokenEstimate,
)
from .store import Store
from .template_store import Template, TemplateStore


class DryRunner:
    """Simulates execution: instantiates artifact, estimates calls/cost."""

    def __init__(self, store: Store, template_store: TemplateStore) -> None:
        self.store = store
        self.template_store = template_store

    def run(
        self,
        template: Template,
        slot_values: dict[str, Any],
        context_ids: list[str] | None = None,
        plan_id: str | None = None,
        item_count: int | None = None,
    ) -> DryRunResponse:
        """Execute a dry run: instantiate + simulate + estimate."""
        # Instantiate
        artifact = instantiate(
            template, slot_values,
            context_ids=context_ids,
            plan_id=plan_id,
            store=self.store,
        )

        # Simulate
        n = item_count or _estimate_item_count(slot_values)
        profile = artifact.static_profile
        call_graph, summary = _simulate(template, n, slot_values)

        # Estimate tokens and cost
        estimate = _estimate(summary, slot_values)

        # Create dry run record
        dry_run_id = generate_id(DRY_PREFIX)
        record = DryRunRecord(
            dry_run_id=dry_run_id,
            created_at=datetime.now(timezone.utc),
            artifact_id=artifact.artifact_id,
            plan_id=plan_id,
            summary=DryRunSummary(
                llm_calls=summary["llm_calls"],
                max_concurrency=summary["max_concurrency"],
                recursive_depth=profile.recursive_depth,
                critical_path_calls=summary["critical_path"],
                checkpoints=summary.get("checkpoints", 0),
                python_phases=summary.get("python_phases", 0),
            ),
            call_graph=call_graph,
        )
        self.store.save("dry_runs", dry_run_id, record.model_dump(mode="json"))

        return DryRunResponse(
            dry_run_id=dry_run_id,
            plan_id=plan_id,
            artifact=ArtifactSummary(
                artifact_id=artifact.artifact_id,
                template_name=artifact.template_name,
                template_version=artifact.template_version,
                code_hash=artifact.generated_scheme_ref.hash,
                primitives_used=artifact.primitives_used,
            ),
            estimate=estimate,
            simulation=DryRunSimulation(
                llm_calls=summary["llm_calls"],
                max_concurrency=summary["max_concurrency"],
                recursive_depth=profile.recursive_depth,
                critical_path_calls=summary["critical_path"],
                checkpoints=summary.get("checkpoints", 0),
                python_phases=summary.get("python_phases", 0),
            ),
            call_graph=call_graph,
            output_schema=template.output_schema,
        )


def _estimate_item_count(slot_values: dict[str, Any]) -> int:
    """Estimate item count from slot values (default 10 for dry runs)."""
    return slot_values.get("item_count", 10)


def _simulate(
    template: Template,
    n: int,
    slot_values: dict[str, Any],
) -> tuple[list[CallGraphNode], dict[str, Any]]:
    """Simulate execution based on template's structural profile."""
    call_graph: list[CallGraphNode] = []
    summary: dict[str, Any] = {
        "llm_calls": 0,
        "max_concurrency": 0,
        "critical_path": 0,
        "checkpoints": 0,
        "python_phases": 0,
    }

    profile = template.structural_profile
    formula = None
    for item in profile:
        if isinstance(item, list) and len(item) >= 2:
            if item[0] == "expected-calls":
                formula = item[1]
            elif item[0] == "uses-python-bridge" and item[1]:
                summary["python_phases"] = 1

    max_concurrent = slot_values.get("max_concurrent", 20)
    branch_factor = slot_values.get("branch_factor", 5)

    if formula:
        summary["llm_calls"] = _eval_formula(formula, n, branch_factor)
    else:
        summary["llm_calls"] = n

    # Determine concurrency and critical path from template primitives
    primitives_used = set()
    body = template.body
    if "map-async" in body:
        primitives_used.add("map-async")
        summary["max_concurrency"] = min(n, max_concurrent)
        call_graph.append(CallGraphNode(
            node_id="map", primitive="map-async",
            calls=n, model=slot_values.get("map_model", "fast_text_model"),
            concurrency=min(n, max_concurrent),
        ))

    if "tree-reduce" in body:
        primitives_used.add("tree-reduce")
        reduce_calls = _tree_reduce_calls(n, branch_factor)
        call_graph.append(CallGraphNode(
            node_id="reduce", primitive="tree-reduce",
            calls=reduce_calls, model=slot_values.get("reduce_model", "quality_text_model"),
            branch_factor=branch_factor,
        ))
        summary["critical_path"] = 1 + math.ceil(math.log(max(n, 1)) / math.log(branch_factor)) if n > 1 else 1

    if "fold-sequential" in body:
        primitives_used.add("fold-sequential")
        call_graph.append(CallGraphNode(
            node_id="fold", primitive="fold-sequential",
            calls=n, model=slot_values.get("fold_model", "quality_text_model"),
        ))
        summary["critical_path"] = max(summary["critical_path"], n)

    if "iterate-until" in body:
        max_iter = slot_values.get("max_iterations", 3)
        call_graph.append(CallGraphNode(
            node_id="iterate", primitive="iterate-until",
            calls=max_iter, model=slot_values.get("model", "quality_text_model"),
        ))
        summary["critical_path"] = max(summary["critical_path"], max_iter * 2)

    if "llm-query" in body and not primitives_used:
        call_graph.append(CallGraphNode(
            node_id="query", primitive="llm-query",
            calls=1, model=slot_values.get("model", "fast_text_model"),
        ))
        summary["critical_path"] = max(summary["critical_path"], 1)

    if "parallel" in body or "race" in body:
        summary["max_concurrency"] = max(summary["max_concurrency"], 2)
        summary["critical_path"] = max(summary["critical_path"], 1)

    if summary["critical_path"] == 0:
        summary["critical_path"] = 1

    return call_graph, summary


def _tree_reduce_calls(n: int, branch_factor: int) -> int:
    """Calculate total calls in a tree reduction."""
    total = 0
    current = n
    while current > 1:
        groups = math.ceil(current / branch_factor)
        total += groups
        current = groups
    return total


def _eval_formula(formula: str, n: int, branch_factor: int) -> int:
    """Evaluate a call count formula like 'N + ceil(N/B) + ...'."""
    if formula == "1":
        return 1
    if formula == "N":
        return n

    # Try to parse common formulas
    total = 0
    if "N" in formula:
        total += n
    if "ceil(N/B)" in formula or "ceil(log_B(N))" in formula:
        total += _tree_reduce_calls(n, branch_factor)
    if "max_iterations" in formula:
        # Extract number
        m = re.search(r"(\d+)\s*\*\s*max_iterations", formula)
        if m:
            total = int(m.group(1)) * 3  # default max_iterations
        else:
            total = max(total, 3)

    return max(total, 1)


def _estimate(
    summary: dict[str, Any],
    slot_values: dict[str, Any],
) -> DryRunEstimate:
    """Estimate tokens and cost from simulation."""
    calls = summary["llm_calls"]
    # Rough estimates: 500 prompt tokens, 200 completion tokens per call
    est_prompt = calls * 500
    est_completion = calls * 200

    return DryRunEstimate(
        expected_llm_calls=calls,
        critical_path_calls=summary["critical_path"],
        max_concurrency=summary["max_concurrency"],
        models={
            slot_values.get("map_model", slot_values.get("model", "fast_text_model")): calls,
        },
        estimated_tokens=TokenEstimate(
            prompt=est_prompt,
            completion=est_completion,
            total=est_prompt + est_completion,
        ),
        estimated_cost_usd=CostEstimate(
            low=calls * 0.001,  # rough low estimate
            high=calls * 0.01,  # rough high estimate
        ),
    )
