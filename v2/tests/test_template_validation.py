"""Tests for template store: S-expression parser, template loading, trigger/reject."""

import os
from pathlib import Path

import pytest

from rlm_scheme.template_store import (
    Template,
    TemplateStore,
    extract_body,
    extract_define_metas,
    extract_slots,
    parse_all,
    parse_sexpr,
    tokenize,
    _eval_condition,
)


TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


class TestSExprParser:
    def test_tokenize_simple(self):
        tokens = tokenize('(define x 42)')
        kinds = [t[0] for t in tokens]
        assert kinds == ["lparen", "atom", "atom", "atom", "rparen"]

    def test_tokenize_string(self):
        tokens = tokenize('(name "hello world")')
        assert tokens[2] == ("string", '"hello world"')

    def test_tokenize_boolean(self):
        tokens = tokenize("#t #f")
        assert tokens[0] == ("atom", "#t")
        assert tokens[1] == ("atom", "#f")

    def test_tokenize_comment(self):
        tokens = tokenize('; this is a comment\n(x)')
        assert len(tokens) == 3  # lparen, atom, rparen

    def test_parse_atom_number(self):
        result, _ = parse_sexpr([("atom", "42")], 0)
        assert result == 42

    def test_parse_atom_float(self):
        result, _ = parse_sexpr([("atom", "3.14")], 0)
        assert result == 3.14

    def test_parse_atom_boolean(self):
        result, _ = parse_sexpr([("atom", "#t")], 0)
        assert result is True
        result, _ = parse_sexpr([("atom", "#f")], 0)
        assert result is False

    def test_parse_atom_symbol(self):
        result, _ = parse_sexpr([("atom", "hello")], 0)
        assert result == "hello"

    def test_parse_string(self):
        result, _ = parse_sexpr([("string", '"hello world"')], 0)
        assert result == "hello world"

    def test_parse_list(self):
        tokens = tokenize("(a b c)")
        result, _ = parse_sexpr(tokens, 0)
        assert result == ["a", "b", "c"]

    def test_parse_nested_list(self):
        tokens = tokenize("(a (b c) d)")
        result, _ = parse_sexpr(tokens, 0)
        assert result == ["a", ["b", "c"], "d"]

    def test_parse_quoted_list(self):
        tokens = tokenize("'(a b c)")
        result, _ = parse_sexpr(tokens, 0)
        assert result == ["a", "b", "c"]

    def test_parse_all_multiple(self):
        results = parse_all("(define x 1) (define y 2)")
        assert len(results) == 2
        assert results[0] == ["define", "x", 1]
        assert results[1] == ["define", "y", 2]


class TestExtractDefineMetas:
    def test_basic(self):
        source = """
(define-meta name "batch_map")
(define-meta version "1.0.0")
(define-meta streamable #t)
"""
        metas = extract_define_metas(source)
        assert metas["name"] == "batch_map"
        assert metas["version"] == "1.0.0"
        assert metas["streamable"] is True

    def test_list_value(self):
        source = '(define-meta task-shapes \'(Batch Synthesize))'
        metas = extract_define_metas(source)
        assert metas["task-shapes"] == ["Batch", "Synthesize"]

    def test_complex_slots(self):
        source = """
(define-meta slots
  '((context_id (type string) (pattern "^ctx_") (required #t))
    (max_concurrent (type integer) (min 1) (max 50) (default 20))))
"""
        metas = extract_define_metas(source)
        slots = metas["slots"]
        assert len(slots) == 2
        assert slots[0][0] == "context_id"
        assert slots[1][0] == "max_concurrent"


class TestExtractSlots:
    def test_finds_slots(self):
        body = """
(define items (__context-ref "{{context_id}}" "{{items_path}}"))
(llm-query #:instruction "{{map_instruction}}" #:model "{{map_model}}")
"""
        markers = extract_slots(body)
        assert set(markers) == {"context_id", "items_path", "map_instruction", "map_model"}


class TestExtractBody:
    def test_separates_meta_from_body(self):
        source = """(define-meta name "test")
(define-meta version "1.0.0")

(define x 42)
(finish x)
"""
        body = extract_body(source)
        assert "(define x 42)" in body
        assert "(finish x)" in body
        assert "define-meta" not in body


class TestTriggerRejectEval:
    def test_greater_than(self):
        assert _eval_condition([">", "item_count", 1], {"item_count": 10})
        assert not _eval_condition([">", "item_count", 1], {"item_count": 0})

    def test_eq(self):
        assert _eval_condition(["eq?", "independent", True], {"independent": True})
        assert not _eval_condition(["eq?", "independent", True], {"independent": False})

    def test_and(self):
        ctx = {"ordered": True, "order_sensitive": True}
        cond = ["and", ["eq?", "ordered", True], ["eq?", "order_sensitive", True]]
        assert _eval_condition(cond, ctx)

    def test_not(self):
        assert _eval_condition(["not", ["eq?", "x", True]], {"x": False})


class TestTemplateLoading:
    def test_load_all_templates(self):
        store = TemplateStore(TEMPLATE_DIR)
        names = store.list_names()
        assert len(names) == 16, f"Expected 16 templates, got {len(names)}: {names}"

    def test_each_template_has_required_meta(self):
        store = TemplateStore(TEMPLATE_DIR)
        for tpl in store.all_templates():
            assert tpl.name, f"Template at {tpl.path} has no name"
            assert tpl.version, f"Template {tpl.name} has no version"

    def test_batch_extract_reduce_details(self):
        store = TemplateStore(TEMPLATE_DIR)
        tpl = store.get("batch_extract_reduce")
        assert tpl is not None
        assert tpl.version == "1.0.0"
        assert "Batch" in tpl.task_shapes
        assert "Synthesize" in tpl.task_shapes
        assert tpl.streamable is True
        assert tpl.cacheable is True
        assert tpl.uses_llm_generated_code is False

    def test_template_slot_markers_match_schema(self):
        store = TemplateStore(TEMPLATE_DIR)
        tpl = store.get("batch_extract_reduce")
        assert tpl is not None
        # Slot markers in body should be subset of slot schema names
        schema_names = {s[0] for s in tpl.slots_schema if isinstance(s, list)}
        for marker in tpl.slot_markers:
            assert marker in schema_names, f"Slot marker {{{{{marker}}}}} not in schema: {schema_names}"

    def test_code_interpreter_uses_llm_code(self):
        store = TemplateStore(TEMPLATE_DIR)
        tpl = store.get("code_interpreter")
        assert tpl is not None
        assert tpl.uses_llm_generated_code is True

    def test_find_by_task_shape(self):
        store = TemplateStore(TEMPLATE_DIR)
        batch_templates = store.find_by_task_shape("Batch")
        names = [t.name for t in batch_templates]
        assert "batch_map" in names
        assert "batch_extract_reduce" in names

    def test_trigger_evaluation(self):
        store = TemplateStore(TEMPLATE_DIR)
        tpl = store.get("batch_extract_reduce")
        assert tpl is not None
        ctx = {
            "item_count": 10,
            "independent": True,
            "output_type": "one",
            "has_second_phase": True,
        }
        assert store.evaluate_triggers(tpl, ctx)

    def test_reject_evaluation(self):
        store = TemplateStore(TEMPLATE_DIR)
        tpl = store.get("batch_extract_reduce")
        assert tpl is not None
        ctx = {"ordered": True, "order_sensitive": True}
        assert store.evaluate_rejects(tpl, ctx)


class TestContextStore:
    """Test context store (imported here to keep test files aligned with plan batches)."""

    def test_create_and_get(self, tmp_path):
        from rlm_scheme.store import Store
        from rlm_scheme.context_store import ContextStore

        store = Store(tmp_path / "data")
        ctx_store = ContextStore(store)

        data = [{"id": 1, "text": "hello"}, {"id": 2, "text": "world"}]
        rec = ctx_store.create(data, name="test_context")

        assert rec.context_id.startswith("ctx_")
        assert rec.name == "test_context"
        assert rec.metadata.item_count == 2
        assert rec.metadata.data_shape.value == "FlatList"

        # Retrieve
        loaded = ctx_store.get(rec.context_id)
        assert loaded is not None
        assert loaded.context_id == rec.context_id

        # Data round-trip
        loaded_data = ctx_store.get_data(rec.context_id)
        assert loaded_data == data

    def test_preview(self, tmp_path):
        from rlm_scheme.store import Store
        from rlm_scheme.context_store import ContextStore

        store = Store(tmp_path / "data")
        ctx_store = ContextStore(store)

        data = {"key": "a" * 500}
        rec = ctx_store.create(data)
        preview = ctx_store.preview(rec.context_id, max_chars=50)
        assert preview is not None
        assert len(preview) <= 53  # 50 + "..."

    def test_json_path_extraction(self, tmp_path):
        from rlm_scheme.store import Store
        from rlm_scheme.context_store import ContextStore, resolve_json_path

        # Test resolve_json_path directly
        data = {"papers": [1, 2, 3], "meta": {"count": 3}}
        assert resolve_json_path(data, "$") == data
        assert resolve_json_path(data, "$.papers") == [1, 2, 3]
        assert resolve_json_path(data, "$.meta.count") == 3

        # Test via context store
        store = Store(tmp_path / "data")
        ctx_store = ContextStore(store)
        rec = ctx_store.create(data)
        extracted = ctx_store.extract_path(rec.context_id, "$.papers")
        assert extracted == [1, 2, 3]


class TestCache:
    def test_store_and_lookup(self, tmp_path):
        from rlm_scheme.store import Store
        from rlm_scheme.cache import LLMCache

        store = Store(tmp_path / "data")
        cache = LLMCache(store)

        # Miss
        result = cache.lookup("Extract data", {"items": [1]}, "fast_text_model", 0.0)
        assert result is None
        assert cache.misses == 1

        # Store
        cache.store_result(
            instruction="Extract data",
            data={"items": [1]},
            model="fast_text_model",
            temperature=0.0,
            result="extracted result",
        )

        # Hit
        result = cache.lookup("Extract data", {"items": [1]}, "fast_text_model", 0.0)
        assert result is not None
        assert result.result == "extracted result"
        assert cache.hits == 1

    def test_different_params_miss(self, tmp_path):
        from rlm_scheme.store import Store
        from rlm_scheme.cache import LLMCache

        store = Store(tmp_path / "data")
        cache = LLMCache(store)

        cache.store_result("instruction", "data", "model_a", 0.0, "result_a")
        # Different model = miss
        assert cache.lookup("instruction", "data", "model_b", 0.0) is None
        # Different temperature = miss
        assert cache.lookup("instruction", "data", "model_a", 0.7) is None


class TestStore:
    def test_crud(self, tmp_path):
        from rlm_scheme.store import Store

        store = Store(tmp_path / "data")
        store.save("test", "id1", {"key": "value"})
        assert store.exists("test", "id1")
        loaded = store.load("test", "id1")
        assert loaded == {"key": "value"}

        ids = store.list_ids("test")
        assert "id1" in ids

        store.delete("test", "id1")
        assert not store.exists("test", "id1")

    def test_clear_all(self, tmp_path):
        from rlm_scheme.store import Store

        store = Store(tmp_path / "data")
        store.save("type_a", "id1", {"a": 1})
        store.save("type_b", "id2", {"b": 2})
        counts = store.clear_all()
        assert counts["type_a"] == 1
        assert counts["type_b"] == 1
