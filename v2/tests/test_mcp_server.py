"""Integration tests for the MCP server tools."""

import json
from pathlib import Path

import pytest

from rlm_scheme.mcp_server import (
    _c,
    _init_components,
    cancel_call,
    dry_run_strategy,
    execute_strategy,
    get_context,
    get_execution_trace,
    get_status,
    load_context,
    plan_strategy,
    reset_runtime,
    resume_execution,
)


@pytest.fixture(autouse=True)
def setup_components(tmp_path, monkeypatch):
    """Initialize components with test paths."""
    monkeypatch.setenv("RLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RLM_TEMPLATE_DIR", str(Path(__file__).parent.parent / "templates"))
    # No OPENAI_API_KEY → DryRunProvider
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _init_components()


def _parse(result: str) -> dict:
    return json.loads(result)


# ---------------------------------------------------------------------------
# load_context
# ---------------------------------------------------------------------------

class TestLoadContext:
    def test_load_json_data(self):
        result = _parse(load_context(data='[1, 2, 3]', name="test_list"))
        assert result["status"] == "ok"
        assert result["context_id"].startswith("ctx_")
        assert result["name"] == "test_list"

    def test_load_string_data(self):
        result = _parse(load_context(data="not valid json"))
        assert result["status"] == "ok"
        assert result["context_id"].startswith("ctx_")

    def test_load_with_metadata(self):
        meta = json.dumps({"data_shape": "FlatList", "independent": True})
        result = _parse(load_context(data='[{"a":1}]', metadata_json=meta))
        assert result["status"] == "ok"
        assert result["metadata"] is not None


# ---------------------------------------------------------------------------
# get_context
# ---------------------------------------------------------------------------

class TestGetContext:
    def test_get_existing(self):
        ctx = _parse(load_context(data='{"key": "value"}'))
        result = _parse(get_context(ctx["context_id"]))
        assert result["status"] == "ok"
        assert result["context_id"] == ctx["context_id"]
        assert "preview" in result

    def test_get_with_data(self):
        ctx = _parse(load_context(data='{"key": "value"}'))
        result = _parse(get_context(ctx["context_id"], include_data=True))
        assert result["data"] is not None

    def test_get_missing(self):
        result = _parse(get_context("ctx_nonexistent123456"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# plan_strategy
# ---------------------------------------------------------------------------

class TestPlanStrategy:
    async def test_plan_with_context(self):
        ctx = _parse(load_context(data='[{"id": 1}, {"id": 2}, {"id": 3}]'))
        result = _parse(await plan_strategy(
            task="Extract data from each item",
            context_id=ctx["context_id"],
            hints_json='{"independent": true}',
        ))
        assert result["status"] == "ok"
        assert result["plan_id"].startswith("plan_")
        assert result["classification"]["task_shape"] is not None
        assert result["recommended"]["template_name"] is not None

    async def test_plan_without_context(self):
        result = _parse(await plan_strategy(task="Summarize something"))
        assert result["status"] == "ok"
        assert result["plan_id"].startswith("plan_")


# ---------------------------------------------------------------------------
# dry_run_strategy
# ---------------------------------------------------------------------------

class TestDryRunStrategy:
    async def test_dry_run_from_plan(self):
        ctx = _parse(load_context(data='[{"id": 1}]'))
        plan = _parse(await plan_strategy(
            task="Summarize this",
            context_id=ctx["context_id"],
        ))
        result = _parse(dry_run_strategy(plan_id=plan["plan_id"]))
        assert result["status"] == "ok"
        assert result["dry_run_id"].startswith("dry_")
        assert result["artifact"] is not None

    def test_dry_run_direct_invocation(self):
        inv = json.dumps({
            "template_name": "direct_call",
            "slot_values": {
                "context_id": "ctx_0123456789abcdef",
                "instruction": "Test",
                "model": "fast_text_model",
            },
        })
        result = _parse(dry_run_strategy(template_invocation_json=inv))
        assert result["status"] == "ok"
        assert result["dry_run_id"].startswith("dry_")

    def test_dry_run_no_args_errors(self):
        result = _parse(dry_run_strategy())
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# execute_strategy
# ---------------------------------------------------------------------------

class TestExecuteStrategy:
    async def test_execute_from_plan(self):
        ctx = _parse(load_context(data='{"text": "Hello world"}'))
        plan = _parse(await plan_strategy(
            task="Summarize this text",
            context_id=ctx["context_id"],
        ))
        # Allow LLM-generated code since planner may select code_interpreter
        result = _parse(await execute_strategy(
            plan_id=plan["plan_id"],
            timeout_seconds=10,
            policy_json='{"allow_llm_generated_code": true}',
        ))
        assert result["status"] == "ok"
        assert result.get("execution_id") is not None

    async def test_execute_direct_invocation(self):
        ctx = _parse(load_context(data='{"text": "Test data"}'))
        inv = json.dumps({
            "template_name": "direct_call",
            "slot_values": {
                "context_id": ctx["context_id"],
                "instruction": "Process this",
                "model": "fast_text_model",
            },
            "context_ids": [ctx["context_id"]],
        })
        result = _parse(await execute_strategy(
            template_invocation_json=inv,
            timeout_seconds=10,
        ))
        assert result["status"] == "ok"

    async def test_execute_no_args_errors(self):
        result = _parse(await execute_strategy())
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# get_execution_trace
# ---------------------------------------------------------------------------

class TestGetExecutionTrace:
    async def test_trace_after_execution(self):
        ctx = _parse(load_context(data='{"text": "Trace test"}'))
        inv = json.dumps({
            "template_name": "direct_call",
            "slot_values": {
                "context_id": ctx["context_id"],
                "instruction": "Process",
                "model": "fast_text_model",
            },
            "context_ids": [ctx["context_id"]],
        })
        exec_result = _parse(await execute_strategy(
            template_invocation_json=inv,
            timeout_seconds=10,
        ))
        trace_result = _parse(get_execution_trace(exec_result["execution_id"]))
        assert trace_result["status"] == "ok"
        assert "events" in trace_result

    def test_trace_missing(self):
        result = _parse(get_execution_trace("exec_nonexistent12345"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    def test_general_status(self):
        result = _parse(get_status())
        assert result["status"] == "ok"
        assert "token_usage" in result

    async def test_execution_status(self):
        ctx = _parse(load_context(data='{"text": "Status test"}'))
        inv = json.dumps({
            "template_name": "direct_call",
            "slot_values": {
                "context_id": ctx["context_id"],
                "instruction": "Process",
                "model": "fast_text_model",
            },
            "context_ids": [ctx["context_id"]],
        })
        exec_result = _parse(await execute_strategy(
            template_invocation_json=inv,
            timeout_seconds=10,
        ))
        result = _parse(get_status(execution_id=exec_result["execution_id"]))
        assert result["status"] == "ok"
        assert "execution" in result

    def test_missing_execution_status(self):
        result = _parse(get_status(execution_id="exec_nonexistent12345"))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# cancel_call
# ---------------------------------------------------------------------------

class TestCancelCall:
    def test_cancel_no_args_errors(self):
        result = _parse(cancel_call())
        assert result["status"] == "error"

    def test_cancel_with_execution_id(self):
        result = _parse(cancel_call(execution_id="exec_0123456789abcdef"))
        assert result["status"] == "ok"
        assert result["cancelled"]["gates_cancelled"] == 0


# ---------------------------------------------------------------------------
# resume_execution
# ---------------------------------------------------------------------------

class TestResumeExecution:
    async def test_resume_no_pending_gate(self):
        result = _parse(await resume_execution(
            execution_id="exec_0123456789abcdef",
            gate="review",
            decision="approve",
        ))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# reset_runtime
# ---------------------------------------------------------------------------

class TestResetRuntime:
    def test_reset_session(self):
        result = _parse(reset_runtime(scope="session"))
        assert result["status"] == "ok"
        assert result["scope"] == "session"

    def test_reset_cache(self):
        result = _parse(reset_runtime(scope="cache"))
        assert result["status"] == "ok"

    def test_reset_all(self):
        result = _parse(reset_runtime(scope="all"))
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

class TestE2EPipeline:
    async def test_full_pipeline(self):
        """Test the complete load -> plan -> dry_run -> execute pipeline."""
        # 1. Load context
        ctx = _parse(load_context(
            data=json.dumps([{"id": i, "text": f"item {i}"} for i in range(5)]),
            name="test_batch",
        ))
        assert ctx["status"] == "ok"
        context_id = ctx["context_id"]

        # 2. Plan strategy
        plan = _parse(await plan_strategy(
            task="Extract key findings from each item",
            context_id=context_id,
            hints_json='{"independent": true}',
        ))
        assert plan["status"] == "ok"
        plan_id = plan["plan_id"]

        # 3. Dry run
        dry = _parse(dry_run_strategy(plan_id=plan_id))
        assert dry["status"] == "ok"

        # 4. Execute
        result = _parse(await execute_strategy(
            plan_id=plan_id,
            timeout_seconds=30,
            policy_json='{"allow_llm_generated_code": true}',
        ))
        assert result["status"] == "ok"
        execution_id = result.get("execution_id")
        assert execution_id is not None

        # 5. Get trace
        trace = _parse(get_execution_trace(execution_id))
        assert trace["status"] == "ok"

        # 6. Get status
        status = _parse(get_status(execution_id=execution_id))
        assert status["status"] == "ok"
