"""Tests for planner, dry run, and gate."""

import asyncio
from pathlib import Path

import pytest

from rlm_scheme.context_store import ContextStore
from rlm_scheme.dry_run import DryRunner
from rlm_scheme.gate import GateManager
from rlm_scheme.llm_provider import DryRunProvider
from rlm_scheme.models import TaskShape
from rlm_scheme.planner import Planner
from rlm_scheme.store import Store
from rlm_scheme.template_store import TemplateStore


TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


@pytest.fixture()
def env(tmp_path):
    """Set up a test environment with all stores."""
    store = Store(tmp_path / "data")
    ctx_store = ContextStore(store)
    tpl_store = TemplateStore(TEMPLATE_DIR)
    return store, ctx_store, tpl_store


# ---------------------------------------------------------------------------
# Planner tests
# ---------------------------------------------------------------------------

class TestPlanner:
    async def test_plan_batch_task(self, env):
        store, ctx_store, tpl_store = env
        # Create a context
        ctx = ctx_store.create([{"id": i, "text": f"item {i}"} for i in range(20)])

        planner = Planner(store, ctx_store, tpl_store)
        plan = await planner.plan(
            task="Extract key findings from each paper",
            context_ids=[ctx.context_id],
            hints={"independent": True},
        )

        assert plan.plan_id.startswith("plan_")
        assert plan.classification.task_shape == TaskShape.Batch
        assert plan.recommended.kind == "template_invocation"
        assert plan.recommended.template_name is not None

    async def test_plan_direct_task(self, env):
        store, ctx_store, tpl_store = env
        ctx = ctx_store.create({"text": "Hello world"})

        planner = Planner(store, ctx_store, tpl_store)
        plan = await planner.plan(
            task="Summarize this text",
            context_ids=[ctx.context_id],
        )

        assert plan.classification.task_shape == TaskShape.Direct
        assert plan.recommended.template_name in ("direct_call", "direct_json_extract", "code_interpreter")

    async def test_plan_with_llm_fill(self, env):
        store, ctx_store, tpl_store = env
        ctx = ctx_store.create([{"id": i} for i in range(10)])

        # Use dry run provider for LLM fill
        provider = DryRunProvider()
        planner = Planner(store, ctx_store, tpl_store, llm_provider=provider)
        plan = await planner.plan(
            task="Extract and summarize data from items",
            context_ids=[ctx.context_id],
            hints={"independent": True, "has_second_phase": True},
        )

        assert plan.planner.mode == "deterministic_with_llm_fill"
        # Should have filled instruction slots
        sv = plan.recommended.slot_values or {}
        assert "context_id" in sv

    async def test_plan_persisted(self, env):
        store, ctx_store, tpl_store = env
        ctx = ctx_store.create({"text": "test"})

        planner = Planner(store, ctx_store, tpl_store)
        plan = await planner.plan("Summarize", context_ids=[ctx.context_id])

        # Load back
        loaded = planner.get(plan.plan_id)
        assert loaded is not None
        assert loaded.plan_id == plan.plan_id
        assert loaded.task == "Summarize"


# ---------------------------------------------------------------------------
# Dry run tests
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_direct_call(self, env):
        store, _, tpl_store = env
        tpl = tpl_store.get("direct_call")
        assert tpl is not None

        runner = DryRunner(store, tpl_store)
        result = runner.run(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "instruction": "Summarize this",
                "model": "fast_text_model",
            },
        )

        assert result.dry_run_id.startswith("dry_")
        assert result.artifact is not None
        assert result.estimate is not None
        assert result.estimate.expected_llm_calls >= 1

    def test_dry_run_batch_extract_reduce(self, env):
        store, _, tpl_store = env
        tpl = tpl_store.get("batch_extract_reduce")
        assert tpl is not None

        runner = DryRunner(store, tpl_store)
        result = runner.run(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "map_instruction": "Extract claims from this paper.",
                "reduce_instruction": "Synthesize claims into a review.",
                "max_concurrent": 10,
                "branch_factor": 3,
            },
            item_count=30,
        )

        assert result.estimate.expected_llm_calls > 30  # N + reduce calls
        assert result.estimate.max_concurrency == 10
        assert len(result.call_graph) >= 2  # map + reduce nodes

    def test_dry_run_creates_artifact(self, env):
        store, _, tpl_store = env
        tpl = tpl_store.get("direct_call")
        assert tpl is not None

        runner = DryRunner(store, tpl_store)
        result = runner.run(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "instruction": "Test",
            },
        )

        assert result.artifact.artifact_id.startswith("art_")
        assert result.artifact.code_hash

    def test_dry_run_output_schema(self, env):
        store, _, tpl_store = env
        tpl = tpl_store.get("batch_extract_reduce")
        assert tpl is not None

        runner = DryRunner(store, tpl_store)
        result = runner.run(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "map_instruction": "Extract data from each item.",
                "reduce_instruction": "Combine into a report.",
            },
        )

        # batch_extract_reduce has an output_schema defined
        assert result.output_schema is not None


# ---------------------------------------------------------------------------
# Gate tests
# ---------------------------------------------------------------------------

class TestGate:
    async def test_suspend_and_resume(self):
        gm = GateManager()

        async def run_gate():
            decision, value = await gm.suspend(
                "exec_01", "review", {"data": "preview"},
                message="Please review",
            )
            return decision, value

        # Start gate in background
        task = asyncio.create_task(run_gate())
        await asyncio.sleep(0.01)  # Let it suspend

        # Check pending
        pending = gm.get_pending("exec_01")
        assert len(pending) == 1
        assert pending[0]["name"] == "review"

        # Resume
        record = gm.resume("exec_01", "review", decision="approve")
        assert record is not None
        assert record.status == "approved"

        decision, value = await task
        assert decision == "approve"
        assert value == {"data": "preview"}

    async def test_reject_gate(self):
        gm = GateManager()

        async def run_gate():
            return await gm.suspend("exec_01", "check", "value")

        task = asyncio.create_task(run_gate())
        await asyncio.sleep(0.01)

        record = gm.resume("exec_01", "check", decision="reject", reason="Not ready")
        assert record is not None
        assert record.status == "rejected"
        assert record.reason == "Not ready"

        decision, _ = await task
        assert decision == "reject"

    async def test_cancel_all(self):
        gm = GateManager()

        async def run_gate(name):
            return await gm.suspend("exec_01", name, "val")

        t1 = asyncio.create_task(run_gate("gate_a"))
        t2 = asyncio.create_task(run_gate("gate_b"))
        await asyncio.sleep(0.01)

        count = gm.cancel_all("exec_01")
        assert count == 2

        d1, _ = await t1
        d2, _ = await t2
        assert d1 == "reject"
        assert d2 == "reject"

    def test_resume_nonexistent(self):
        gm = GateManager()
        result = gm.resume("exec_99", "nonexistent")
        assert result is None
