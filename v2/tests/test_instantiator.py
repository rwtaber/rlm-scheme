"""Tests for classifier, instantiator, trace, and python bridge."""

from pathlib import Path

import pytest

from rlm_scheme.classifier import classify
from rlm_scheme.instantiator import (
    InstantiationError,
    get_instantiated_code,
    instantiate,
    _detect_primitives,
    _check_forbidden,
)
from rlm_scheme.models import DataShape, TaskShape
from rlm_scheme.store import Store
from rlm_scheme.template_store import TemplateStore
from rlm_scheme.trace import TraceCollector
from rlm_scheme.python_bridge import PythonBridge, PythonBridgeError


TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_direct_single_item(self):
        result = classify("Summarize this text", {"item_count": 1})
        assert result.task_shape == TaskShape.Direct
        assert result.confidence == 1.0

    def test_batch_many_independent(self):
        result = classify(
            "Extract data from each item",
            {"item_count": 50, "independent": True},
        )
        assert result.task_shape == TaskShape.Batch

    def test_classify_operation(self):
        result = classify(
            "Classify each email as spam or not",
            {"item_count": 100, "independent": True},
        )
        assert result.task_shape == TaskShape.Classify

    def test_validate_operation(self):
        result = classify(
            "Validate each record against the schema",
            {"item_count": 20, "independent": True},
        )
        assert result.task_shape == TaskShape.Validate

    def test_synthesize_ordered(self):
        result = classify(
            "Combine chapter summaries in order",
            {"item_count": 10, "independent": False, "ordered": True},
        )
        assert result.task_shape == TaskShape.Synthesize

    def test_refine(self):
        result = classify("Refine this essay", {"item_count": 1})
        assert result.task_shape == TaskShape.Refine

    def test_decompose(self):
        result = classify("Break down this problem into sub-tasks", {"item_count": 1})
        assert result.task_shape == TaskShape.Decompose

    def test_generate(self):
        result = classify("Generate a report", {"item_count": 1})
        assert result.task_shape == TaskShape.Generate

    def test_compare(self):
        result = classify("Compare these two approaches", {"item_count": 1})
        assert result.task_shape == TaskShape.Compare

    def test_composite_from_hints(self):
        result = classify(
            "Extract and then summarize",
            {"item_count": 10, "independent": True},
            hints={"multi_phase": True, "phases": ["Batch", "Synthesize"]},
        )
        assert result.task_shape == TaskShape.Composite

    def test_data_shape_flat_list(self):
        result = classify("Process items", {"item_count": 10, "independent": True})
        assert result.data_shape == DataShape.FlatList

    def test_data_shape_singular(self):
        result = classify("Summarize", {"item_count": 1})
        assert result.data_shape == DataShape.Singular

    def test_data_shape_multimodal(self):
        result = classify("Describe image", {"item_count": 1, "modality": ["image"]})
        assert result.data_shape == DataShape.Multimodal

    def test_hint_override(self):
        result = classify(
            "Custom task",
            {"item_count": 5},
            hints={"task_shape": "Pipeline", "data_shape": "Tabular"},
        )
        assert result.task_shape == TaskShape.Pipeline
        assert result.data_shape == DataShape.Tabular


# ---------------------------------------------------------------------------
# Instantiator tests
# ---------------------------------------------------------------------------

class TestInstantiator:
    @pytest.fixture()
    def store(self, tmp_path):
        return Store(tmp_path / "data")

    @pytest.fixture()
    def template_store(self):
        return TemplateStore(TEMPLATE_DIR)

    def test_instantiate_direct_call(self, store, template_store):
        tpl = template_store.get("direct_call")
        assert tpl is not None

        art = instantiate(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "instruction": "Summarize this text",
                "model": "fast_text_model",
            },
            store=store,
        )
        assert art.artifact_id.startswith("art_")
        assert art.template_name == "direct_call"
        assert "finish" in art.primitives_used

        # Check code was persisted
        code = get_instantiated_code(art.artifact_id, store)
        assert code is not None
        assert "ctx_0123456789abcdef" in code
        assert "Summarize this text" in code

    def test_instantiate_batch_extract_reduce(self, store, template_store):
        tpl = template_store.get("batch_extract_reduce")
        assert tpl is not None

        art = instantiate(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "map_instruction": "Extract the core claim from this paper.",
                "reduce_instruction": "Synthesize extracted claims into a review.",
                "max_concurrent": 10,
                "branch_factor": 3,
            },
            store=store,
        )
        assert "map-async" in art.primitives_used
        assert "tree-reduce" in art.primitives_used

    def test_missing_required_slot_raises(self, template_store):
        tpl = template_store.get("direct_call")
        assert tpl is not None

        with pytest.raises(InstantiationError, match="Required slot"):
            instantiate(tpl, slot_values={"context_id": "ctx_0123456789abcdef"})

    def test_invalid_pattern_raises(self, template_store):
        tpl = template_store.get("direct_call")
        assert tpl is not None

        with pytest.raises(InstantiationError, match="does not match pattern"):
            instantiate(
                tpl,
                slot_values={
                    "context_id": "bad_id",
                    "instruction": "test",
                },
            )

    def test_range_check(self, template_store):
        tpl = template_store.get("batch_map")
        assert tpl is not None

        with pytest.raises(InstantiationError, match="above maximum"):
            instantiate(
                tpl,
                slot_values={
                    "context_id": "ctx_0123456789abcdef",
                    "map_instruction": "test instruction",
                    "max_concurrent": 100,  # max is 50
                },
            )

    def test_forbidden_identifier_detection(self):
        with pytest.raises(InstantiationError, match="Forbidden identifier"):
            _check_forbidden("(eval (read))")

    def test_detect_primitives(self):
        code = '(map-async (lambda (x) (llm-query-async #:instruction "hi")) items)'
        prims = _detect_primitives(code)
        assert "map-async" in prims
        assert "llm-query-async" in prims

    def test_defaults_applied(self, store, template_store):
        tpl = template_store.get("direct_call")
        assert tpl is not None

        art = instantiate(
            tpl,
            slot_values={
                "context_id": "ctx_0123456789abcdef",
                "instruction": "test instruction",
            },
            store=store,
        )
        code = get_instantiated_code(art.artifact_id, store)
        assert code is not None
        assert "fast_text_model" in code  # default model


# ---------------------------------------------------------------------------
# Trace tests
# ---------------------------------------------------------------------------

class TestTrace:
    def test_collect_and_build(self):
        tc = TraceCollector("exec_test", artifact_id="art_test")
        tc.record_call_start("call_01", "fast_text_model")
        tc.record_call_end("call_01", tokens=150)
        tc.log_scope("syntax-e", preview="hello...", call_id="call_01")
        tc.append_stdout("output line\n")

        detail = tc.build()
        assert len(detail.events) == 2
        assert detail.events[0].type == "call_start"
        assert detail.events[1].type == "call_end"
        assert detail.events[1].tokens == 150
        assert len(detail.scope_log) == 1
        assert detail.stdout == "output line\n"

    def test_persist_and_load(self, tmp_path):
        store = Store(tmp_path / "data")
        tc = TraceCollector("exec_test", artifact_id="art_test")
        tc.record_call_start("call_01", "fast_text_model")
        tc.persist(store)

        loaded = TraceCollector.load("exec_test", store)
        assert loaded is not None
        assert len(loaded.events) == 1
        assert loaded.artifact_id == "art_test"


# ---------------------------------------------------------------------------
# Python bridge tests
# ---------------------------------------------------------------------------

class TestPythonBridge:
    def test_eval(self):
        bridge = PythonBridge(timeout=5.0)
        result = bridge.evaluate("1 + 2")
        assert result == 3

    def test_exec(self):
        bridge = PythonBridge(timeout=5.0)
        bridge.set_variable("items", [1, 2, 3])
        result = bridge.execute("print(sum(items))")
        assert "6" in result

    def test_set_variable(self):
        bridge = PythonBridge(timeout=5.0)
        bridge.set_variable("x", 42)
        result = bridge.evaluate("x * 2")
        assert result == 84

    def test_forbidden_import(self):
        bridge = PythonBridge(timeout=5.0)
        with pytest.raises(PythonBridgeError, match="Forbidden import"):
            bridge.execute("import os; os.listdir('.')")

    def test_alive(self):
        bridge = PythonBridge(timeout=5.0)
        assert bridge.alive

    def test_json_module_allowed(self):
        bridge = PythonBridge(timeout=5.0)
        result = bridge.evaluate('__import__("json").dumps({"a": 1})')
        assert '"a"' in result

    def test_timeout(self):
        bridge = PythonBridge(timeout=1.0)
        with pytest.raises(PythonBridgeError, match="timed out"):
            bridge.execute("import time; time.sleep(10)")
