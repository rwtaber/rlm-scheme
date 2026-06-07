"""FastMCP server wiring for RLM-Scheme v2."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from .cache import LLMCache
from .chain import ChainExecutor
from .context_store import ContextStore
from .dry_run import DryRunner
from .executor import Executor
from .gate import GateManager
from .llm_provider import DryRunProvider, LLMProvider
from .models import (
    ContextMetadata,
    ExecutionPolicy,
    ResetScope,
    TemplateInvocation,
)
from .planner import Planner
from .store import Store
from .template_store import TemplateStore
from .trace import TraceCollector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Component holder — initialised in lifespan
# ---------------------------------------------------------------------------

class _Components:
    store: Store
    context_store: ContextStore
    template_store: TemplateStore
    llm_provider: LLMProvider
    cache: LLMCache
    gate_manager: GateManager
    planner: Planner
    dry_runner: DryRunner
    executor: Executor
    chain_executor: ChainExecutor


_c = _Components()


def _init_components() -> None:
    """Create all components from environment / defaults."""
    data_dir = Path(os.environ.get("RLM_DATA_DIR", ".rlm_data"))
    template_dir = Path(os.environ.get("RLM_TEMPLATE_DIR", str(Path(__file__).parent.parent / "templates")))

    _c.store = Store(data_dir)
    _c.context_store = ContextStore(_c.store)
    _c.template_store = TemplateStore(template_dir)

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        _c.llm_provider = LLMProvider(api_key=api_key)
    else:
        logger.warning("No OPENAI_API_KEY — using DryRunProvider")
        _c.llm_provider = DryRunProvider()

    _c.cache = LLMCache(_c.store)
    _c.gate_manager = GateManager()

    _c.planner = Planner(
        _c.store, _c.context_store, _c.template_store,
        llm_provider=_c.llm_provider,
    )
    _c.dry_runner = DryRunner(_c.store, _c.template_store)
    _c.executor = Executor(
        store=_c.store,
        context_store=_c.context_store,
        template_store=_c.template_store,
        llm_provider=_c.llm_provider,
        cache=_c.cache,
        gate_manager=_c.gate_manager,
    )
    _c.chain_executor = ChainExecutor(_c.executor, _c.context_store)


@asynccontextmanager
async def lifespan(app: Any):
    _init_components()
    yield


# ---------------------------------------------------------------------------
# FastMCP app
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "rlm-scheme",
    instructions=(
        "Artifact-based LLM orchestration. "
        "Pipeline: load_context -> plan_strategy -> dry_run_strategy -> execute_strategy. "
        "Use get_execution_trace, get_status, cancel_call, resume_execution, reset_runtime to inspect/control."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _json_response(obj: Any) -> str:
    """Serialize a response to JSON string."""
    if hasattr(obj, "model_dump"):
        return json.dumps(obj.model_dump(mode="json"), default=str)
    return json.dumps(obj, default=str)


def _parse_json_param(value: str | None, name: str) -> dict[str, Any] | None:
    """Parse an optional JSON string parameter."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError(f"{name} must be a JSON object")
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {name}: {e}") from e


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def load_context(
    data: str,
    name: str | None = None,
    metadata_json: str | None = None,
) -> str:
    """Store input data and metadata; return context_id.

    Args:
        data: JSON-encoded data to store.
        name: Optional human-readable name.
        metadata_json: Optional JSON with metadata hints (data_shape, independent, etc.).
    """
    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError:
        parsed_data = data  # treat as raw string

    metadata_dict = _parse_json_param(metadata_json, "metadata_json")

    record = _c.context_store.create(parsed_data, name=name, metadata=metadata_dict)

    return _json_response({
        "status": "ok",
        "context_id": record.context_id,
        "name": record.name,
        "metadata": record.metadata.model_dump(mode="json") if record.metadata else None,
        "preview": _c.context_store.preview(record.context_id),
        "next_actions": [f"Call plan_strategy with context_id={record.context_id}"],
    })


@mcp.tool()
def get_context(
    context_id: str,
    include_preview: bool = True,
    include_data: bool = False,
) -> str:
    """Inspect metadata and optionally preview stored data.

    Args:
        context_id: The context to inspect.
        include_preview: Whether to include a data preview.
        include_data: Whether to include the full data (may be large).
    """
    record = _c.context_store.get(context_id)
    if record is None:
        return _json_response({"status": "error", "errors": [f"Context {context_id} not found"]})

    result: dict[str, Any] = {
        "status": "ok",
        "context_id": record.context_id,
        "name": record.name,
        "created_at": str(record.created_at),
        "metadata": record.metadata.model_dump(mode="json") if record.metadata else None,
    }
    if include_preview:
        result["preview"] = _c.context_store.preview(context_id)
    if include_data:
        data = _c.context_store.get_data(context_id)
        result["data"] = data
    return _json_response(result)


@mcp.tool()
async def plan_strategy(
    task: str,
    context_id: str | None = None,
    hints_json: str | None = None,
) -> str:
    """Classify task/data and return plan_id plus proposed template invocation.

    Args:
        task: Natural-language description of what to do.
        context_id: Optional context_id for the data to process.
        hints_json: Optional JSON with planning hints (independent, has_second_phase, etc.).
    """
    hints = _parse_json_param(hints_json, "hints_json") or {}
    context_ids = [context_id] if context_id else []

    plan = await _c.planner.plan(
        task=task,
        context_ids=context_ids,
        hints=hints,
    )

    return _json_response({
        "status": "ok",
        "plan_id": plan.plan_id,
        "classification": plan.classification.model_dump(mode="json"),
        "recommended": plan.recommended.model_dump(mode="json"),
        "alternatives": [a.model_dump(mode="json") for a in plan.alternatives],
        "next_actions": [f"Call dry_run_strategy with plan_id={plan.plan_id}"],
    })


@mcp.tool()
def dry_run_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    options_json: str | None = None,
) -> str:
    """Instantiate, simulate, and estimate without real LLM calls. Return dry_run_id.

    Args:
        plan_id: Plan to dry-run (loads template + slots from plan record).
        template_invocation_json: Direct template invocation JSON (alternative to plan_id).
        options_json: Optional JSON with dry-run options (item_count, etc.).
    """
    options = _parse_json_param(options_json, "options_json") or {}

    if plan_id:
        plan = _c.planner.get(plan_id)
        if plan is None:
            return _json_response({"status": "error", "errors": [f"Plan {plan_id} not found"]})
        template_name = plan.recommended.template_name
        slot_values = plan.recommended.slot_values or {}
    elif template_invocation_json:
        inv = _parse_json_param(template_invocation_json, "template_invocation_json")
        if inv is None:
            return _json_response({"status": "error", "errors": ["Invalid template_invocation_json"]})
        template_name = inv.get("template_name")
        slot_values = inv.get("slot_values", {})
    else:
        return _json_response({"status": "error", "errors": ["Provide plan_id or template_invocation_json"]})

    template = _c.template_store.get(template_name)
    if template is None:
        return _json_response({"status": "error", "errors": [f"Template '{template_name}' not found"]})

    result = _c.dry_runner.run(
        template,
        slot_values=slot_values,
        item_count=options.get("item_count"),
    )

    return _json_response({
        "status": "ok",
        "dry_run_id": result.dry_run_id,
        "plan_id": plan_id,
        "artifact": result.artifact.model_dump(mode="json") if result.artifact else None,
        "estimate": result.estimate.model_dump(mode="json") if result.estimate else None,
        "call_graph": [n.model_dump(mode="json") for n in result.call_graph],
        "output_schema": result.output_schema,
        "next_actions": [
            f"Call execute_strategy with plan_id={plan_id}" if plan_id
            else "Call execute_strategy with template_invocation_json"
        ],
    })


@mcp.tool()
async def execute_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    timeout_seconds: int | None = None,
    stream: bool = False,
    policy_json: str | None = None,
) -> str:
    """Instantiate, verify, and execute. Return execution_id.

    Args:
        plan_id: Plan to execute (loads template + slots from plan record).
        template_invocation_json: Direct template invocation JSON (alternative to plan_id).
        timeout_seconds: Maximum execution time in seconds.
        stream: Whether to stream results (not yet implemented).
        policy_json: Optional JSON with execution policy overrides.
    """
    policy_dict = _parse_json_param(policy_json, "policy_json")
    policy = ExecutionPolicy(**policy_dict) if policy_dict else None
    timeout = float(timeout_seconds) if timeout_seconds else None

    if plan_id:
        plan = _c.planner.get(plan_id)
        if plan is None:
            return _json_response({"status": "error", "errors": [f"Plan {plan_id} not found"]})
        template_name = plan.recommended.template_name
        slot_values = plan.recommended.slot_values or {}
        context_ids = plan.context_ids
        the_plan_id = plan_id

        # Check for chain steps
        if plan.recommended.steps:
            response = await _c.chain_executor.execute_chain(
                steps=plan.recommended.steps,
                context_ids=context_ids,
                plan_id=the_plan_id,
                policy=policy,
                timeout=timeout,
            )
            return _json_response(response.model_dump(mode="json"))
    elif template_invocation_json:
        inv = _parse_json_param(template_invocation_json, "template_invocation_json")
        if inv is None:
            return _json_response({"status": "error", "errors": ["Invalid template_invocation_json"]})
        template_name = inv.get("template_name")
        slot_values = inv.get("slot_values", {})
        context_ids = inv.get("context_ids", [])
        the_plan_id = None
    else:
        return _json_response({"status": "error", "errors": ["Provide plan_id or template_invocation_json"]})

    template = _c.template_store.get(template_name)
    if template is None:
        return _json_response({"status": "error", "errors": [f"Template '{template_name}' not found"]})

    response = await _c.executor.execute(
        template=template,
        slot_values=slot_values,
        context_ids=context_ids,
        plan_id=the_plan_id,
        policy=policy,
        timeout=timeout,
        stream=stream,
    )

    return _json_response(response.model_dump(mode="json"))


@mcp.tool()
def get_execution_trace(
    execution_id: str,
    include_scope_log: bool = True,
    include_calls: bool = True,
    include_stdout: bool = True,
) -> str:
    """Return call hierarchy, data flow, stdout, errors, token usage, and checkpoints.

    Args:
        execution_id: The execution to inspect.
        include_scope_log: Whether to include scope log entries.
        include_calls: Whether to include individual call events.
        include_stdout: Whether to include captured stdout.
    """
    detail = TraceCollector.load(execution_id, _c.store)
    if detail is None:
        return _json_response({"status": "error", "errors": [f"Trace for {execution_id} not found"]})
    result: dict[str, Any] = {
        "status": "ok",
        "execution_id": execution_id,
        "artifact_id": detail.artifact_id,
        "plan_id": detail.plan_id,
    }
    if include_calls:
        result["events"] = [e.model_dump(mode="json") for e in detail.events]
    if include_scope_log:
        result["scope_log"] = [s.model_dump(mode="json") for s in detail.scope_log]
    if include_stdout:
        result["stdout"] = detail.stdout

    return _json_response(result)


@mcp.tool()
def get_status(execution_id: str | None = None) -> str:
    """Return server/runtime/call status.

    Args:
        execution_id: Optional execution_id to get specific execution status.
    """
    if execution_id:
        record = _c.store.load("executions", execution_id)
        if record is None:
            return _json_response({"status": "error", "errors": [f"Execution {execution_id} not found"]})
        return _json_response({
            "status": "ok",
            "execution": record,
        })

    return _json_response({
        "status": "ok",
        "runtime": {
            "racket_alive": False,
            "python_bridge_alive": False,
        },
        "token_usage": {
            "prompt_tokens": _c.llm_provider.total_tokens.prompt,
            "completion_tokens": _c.llm_provider.total_tokens.completion,
            "total_tokens": _c.llm_provider.total_tokens.total,
            "calls": _c.llm_provider.total_calls,
        },
        "cache_stats": {
            "hits": _c.cache.hits,
            "misses": _c.cache.misses,
        },
    })


@mcp.tool()
def cancel_call(
    call_id: str | None = None,
    execution_id: str | None = None,
    reason: str | None = None,
) -> str:
    """Cancel one call or an entire execution.

    Args:
        call_id: Specific call to cancel.
        execution_id: Execution to cancel (cancels all pending gates).
        reason: Reason for cancellation.
    """
    cancelled: dict[str, Any] = {}

    if execution_id:
        gate_count = _c.gate_manager.cancel_all(execution_id)
        cancelled["gates_cancelled"] = gate_count
        cancelled["execution_id"] = execution_id

    if call_id:
        cancelled["call_id"] = call_id
        cancelled["note"] = "Call-level cancellation not yet implemented"

    if not cancelled:
        return _json_response({"status": "error", "errors": ["Provide call_id or execution_id"]})

    return _json_response({"status": "ok", "cancelled": cancelled})


@mcp.tool()
async def resume_execution(
    execution_id: str,
    gate: str,
    decision: str,
    reason: str | None = None,
) -> str:
    """Approve or reject a gate to resume or terminate a suspended execution.

    Args:
        execution_id: The suspended execution.
        gate: The gate name to resume.
        decision: "approve" or "reject".
        reason: Optional reason for the decision.
    """
    record = _c.gate_manager.resume(execution_id, gate, decision=decision, reason=reason)
    if record is None:
        return _json_response({
            "status": "error",
            "errors": [f"No pending gate '{gate}' for execution {execution_id}"],
        })

    return _json_response({
        "status": "ok",
        "execution_id": execution_id,
        "gate": record.model_dump(mode="json"),
    })


@mcp.tool()
def reset_runtime(scope: str = "session") -> str:
    """Reset sandbox state without deleting durable records by default.

    Args:
        scope: What to reset: "sandbox", "session", "all", "cache", "config".
    """
    cleared: dict[str, int] = {}

    if scope in ("cache", "session", "all"):
        cache_cleared = _c.cache.clear()
        cleared["cache_entries"] = cache_cleared

    if scope in ("session", "all"):
        _c.llm_provider.reset_accounting()
        cleared["token_accounting"] = 1

    if scope == "all":
        _c.store.clear_all()
        cleared["all_records"] = 1

    return _json_response({
        "status": "ok",
        "scope": scope,
        "cleared": cleared,
    })
