"""Tests for template chain execution."""

from pathlib import Path

import pytest

from rlm_scheme.chain import ChainExecutor, _resolve_previous
from rlm_scheme.context_store import ContextStore
from rlm_scheme.executor import Executor
from rlm_scheme.gate import GateManager
from rlm_scheme.cache import LLMCache
from rlm_scheme.llm_provider import DryRunProvider
from rlm_scheme.models import ExecutionPolicy, TemplateChainStep
from rlm_scheme.store import Store
from rlm_scheme.template_store import TemplateStore


TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


@pytest.fixture()
def chain_env(tmp_path):
    """Set up chain test environment."""
    store = Store(tmp_path / "data")
    ctx_store = ContextStore(store)
    tpl_store = TemplateStore(TEMPLATE_DIR)
    provider = DryRunProvider()
    cache = LLMCache(store)
    gate = GateManager()
    executor = Executor(
        store=store,
        context_store=ctx_store,
        template_store=tpl_store,
        llm_provider=provider,
        cache=cache,
        gate_manager=gate,
    )
    chain_exec = ChainExecutor(executor, ctx_store)
    return chain_exec, executor, store, ctx_store, tpl_store


class TestResolvePrevious:
    """Test $previous reference resolution."""

    def test_no_previous_on_first_step(self):
        slots = {"input": "$previous", "instruction": "Process"}
        resolved = _resolve_previous(slots, None)
        # No previous available — left as-is
        assert resolved["input"] == "$previous"
        assert resolved["instruction"] == "Process"

    def test_replaces_previous(self):
        slots = {"input": "$previous", "instruction": "Process"}
        resolved = _resolve_previous(slots, "ctx_abc123def456")
        assert resolved["input"] == "ctx_abc123def456"
        assert resolved["instruction"] == "Process"

    def test_partial_replacement(self):
        slots = {"context_id": "prefix_$previous_suffix"}
        resolved = _resolve_previous(slots, "ctx_abc123")
        assert resolved["context_id"] == "prefix_ctx_abc123_suffix"

    def test_non_string_values_pass_through(self):
        slots = {"count": 5, "flag": True, "instruction": "$previous"}
        resolved = _resolve_previous(slots, "ctx_abc123")
        assert resolved["count"] == 5
        assert resolved["flag"] is True
        assert resolved["instruction"] == "ctx_abc123"


class TestChainExecution:
    """Test multi-step chain execution."""

    async def test_single_step_chain(self, chain_env):
        chain_exec, executor, store, ctx_store, tpl_store = chain_env
        ctx = ctx_store.create({"text": "Hello world"})
        tpl = tpl_store.get("direct_call")
        assert tpl is not None

        steps = [
            TemplateChainStep(
                template_name="direct_call",
                template_version=tpl.version,
                slot_values={
                    "context_id": ctx.context_id,
                    "instruction": "Summarize",
                    "model": "fast_text_model",
                },
            ),
        ]

        response = await chain_exec.execute_chain(
            steps=steps,
            context_ids=[ctx.context_id],
            timeout=10.0,
        )

        assert response.status == "ok"
        assert response.result is not None

    async def test_two_step_chain_with_previous(self, chain_env):
        chain_exec, executor, store, ctx_store, tpl_store = chain_env
        ctx = ctx_store.create({"text": "Source data"})
        tpl = tpl_store.get("direct_call")
        assert tpl is not None

        steps = [
            TemplateChainStep(
                template_name="direct_call",
                template_version=tpl.version,
                slot_values={
                    "context_id": ctx.context_id,
                    "instruction": "Extract key points",
                    "model": "fast_text_model",
                },
            ),
            TemplateChainStep(
                template_name="direct_call",
                template_version=tpl.version,
                slot_values={
                    "context_id": "$previous",
                    "instruction": "Summarize the key points",
                    "model": "fast_text_model",
                },
            ),
        ]

        response = await chain_exec.execute_chain(
            steps=steps,
            context_ids=[ctx.context_id],
            timeout=10.0,
        )

        assert response.status == "ok"
        assert response.result is not None
        # Chain step results should be recorded
        if response.execution and response.execution.chain_step_results:
            assert len(response.execution.chain_step_results) == 2
            assert response.execution.chain_step_results[0].step == 0
            assert response.execution.chain_step_results[1].step == 1

    async def test_chain_stores_intermediate_contexts(self, chain_env):
        chain_exec, executor, store, ctx_store, tpl_store = chain_env
        ctx = ctx_store.create({"text": "Test data"})
        tpl = tpl_store.get("direct_call")
        initial_ctx_count = len(ctx_store.list_ids())

        steps = [
            TemplateChainStep(
                template_name="direct_call",
                template_version=tpl.version,
                slot_values={
                    "context_id": ctx.context_id,
                    "instruction": "Step 1",
                    "model": "fast_text_model",
                },
            ),
            TemplateChainStep(
                template_name="direct_call",
                template_version=tpl.version,
                slot_values={
                    "context_id": "$previous",
                    "instruction": "Step 2",
                    "model": "fast_text_model",
                },
            ),
        ]

        response = await chain_exec.execute_chain(
            steps=steps,
            context_ids=[ctx.context_id],
            timeout=10.0,
        )

        assert response.status == "ok"
        # Should have created intermediate contexts
        final_ctx_count = len(ctx_store.list_ids())
        assert final_ctx_count > initial_ctx_count

    async def test_chain_error_on_missing_template(self, chain_env):
        chain_exec, executor, store, ctx_store, tpl_store = chain_env

        steps = [
            TemplateChainStep(
                template_name="nonexistent_template",
                template_version="0.1.0",
                slot_values={"instruction": "Test"},
            ),
        ]

        response = await chain_exec.execute_chain(steps=steps, timeout=10.0)
        assert response.status == "error"
        assert any("not found" in e for e in response.errors)

    async def test_empty_chain_returns_error(self, chain_env):
        chain_exec, executor, store, ctx_store, tpl_store = chain_env

        response = await chain_exec.execute_chain(steps=[], timeout=10.0)
        assert response.status == "error"
        assert any("Empty" in e for e in response.errors)

    async def test_chain_error_propagation(self, chain_env):
        """A chain should stop at the first error step."""
        chain_exec, executor, store, ctx_store, tpl_store = chain_env
        tpl = tpl_store.get("direct_call")

        steps = [
            TemplateChainStep(
                template_name="direct_call",
                template_version=tpl.version,
                slot_values={
                    "instruction": "Step 1",
                    "model": "fast_text_model",
                },
            ),
            TemplateChainStep(
                template_name="nonexistent_template",
                template_version="0.1.0",
                slot_values={"instruction": "This should not run"},
            ),
        ]

        response = await chain_exec.execute_chain(steps=steps, timeout=10.0)
        # The second step should fail (template not found)
        assert response.status == "error"
