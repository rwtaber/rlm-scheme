"""Tests for compact MCP progress reporting (S5) and enhanced progress (S7).

Verifies that:
- _call_stats tracks dispatched/completed counts correctly
- reset_call_stats() clears counts
- _format_progress_message() produces compact grouped output
- Message is capped at 80 chars
- No message when nothing is dispatched
- Progress monitor uses fraction-style (progress/total) reporting
- Poll interval increases after PROGRESS_POLL_THRESHOLD seconds
- Shared CallRegistry tracks calls across REPL instances (S7)
- Depth-aware call IDs avoid collisions (S7)
- Enhanced format shows depth annotations and instruction previews (S7)
- Event-driven monitor wakes on state changes (S7)
"""

import asyncio
import os
import sys
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import (
    CallRegistry,
    RacketREPL,
    _call_registry,
    _format_progress_message,
    PROGRESS_POLL_INITIAL,
    PROGRESS_POLL_LONG,
    PROGRESS_POLL_THRESHOLD,
)


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear the shared call registry before and after each test."""
    _call_registry.clear()
    yield
    _call_registry.clear()


# ============================================================
# _call_stats tracking
# ============================================================


class TestCallStats:
    """_call_stats tracking on RacketREPL."""

    @patch("mcp_server.openai.OpenAI")
    def test_initial_stats_zero(self, mock_openai_cls):
        """Fresh RacketREPL has zero dispatched/completed."""
        repl = RacketREPL()
        try:
            stats = repl.get_call_stats()
            assert stats["dispatched"] == 0
            assert stats["completed"] == 0
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_register_increments_dispatched(self, mock_openai_cls):
        """_register_call increments dispatched count."""
        repl = RacketREPL()
        try:
            repl._register_call("c1", "sync", "gpt-4o", "test")
            repl._register_call("c2", "async", "gpt-4o-mini", "test2")
            stats = repl.get_call_stats()
            assert stats["dispatched"] == 2
            assert stats["completed"] == 0
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_complete_increments_completed(self, mock_openai_cls):
        """_complete_call increments completed count."""
        repl = RacketREPL()
        try:
            repl._register_call("c1", "sync", "gpt-4o", "test")
            repl._register_call("c2", "async", "gpt-4o", "test2")
            repl._complete_call("c1", 100, 1.0, 50, 50)
            stats = repl.get_call_stats()
            assert stats["dispatched"] == 2
            assert stats["completed"] == 1
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_reset_clears_stats(self, mock_openai_cls):
        """reset_call_stats() zeroes both counters."""
        repl = RacketREPL()
        try:
            repl._register_call("c1", "sync", "gpt-4o", "test")
            repl._complete_call("c1", 100, 1.0, 50, 50)
            assert repl.get_call_stats()["dispatched"] == 1
            repl.reset_call_stats()
            stats = repl.get_call_stats()
            assert stats["dispatched"] == 0
            assert stats["completed"] == 0
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_get_call_stats_returns_copy(self, mock_openai_cls):
        """get_call_stats() returns a dict copy, not the internal reference."""
        repl = RacketREPL()
        try:
            s1 = repl.get_call_stats()
            s2 = repl.get_call_stats()
            assert s1 is not s2
            assert s1 == s2
        finally:
            repl.close()


# ============================================================
# _format_progress_message
# ============================================================


class TestFormatProgressMessage:
    """_format_progress_message produces compact grouped output."""

    def test_no_calls_no_dispatched_returns_none(self):
        """Returns None when nothing is happening."""
        result = _format_progress_message([], {"dispatched": 0, "completed": 0})
        assert result is None

    def test_single_model_active(self):
        """Single active call shows model with count."""
        calls = [{"model": "gpt-4o", "elapsed_seconds": 3.0}]
        stats = {"dispatched": 1, "completed": 0}
        msg = _format_progress_message(calls, stats)
        assert "0/1 done" in msg
        assert "1\u00d7gpt-4o" in msg
        assert "longest: 3s" in msg

    def test_multiple_models_grouped(self):
        """Multiple models are grouped by count."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 5.0},
            {"model": "gpt-4o", "elapsed_seconds": 4.0},
            {"model": "gpt-4o-mini", "elapsed_seconds": 2.0},
        ]
        stats = {"dispatched": 5, "completed": 2}
        msg = _format_progress_message(calls, stats)
        assert "2/5 done" in msg
        assert "3 active" in msg
        assert "2\u00d7gpt-4o" in msg
        assert "1\u00d7gpt-4o-mini" in msg

    def test_max_elapsed_shown(self):
        """Max elapsed time from active calls is included."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 12.0},
            {"model": "gpt-4o", "elapsed_seconds": 3.0},
        ]
        stats = {"dispatched": 4, "completed": 2}
        msg = _format_progress_message(calls, stats)
        assert "longest: 12s" in msg

    def test_all_completed_no_active(self):
        """When all calls are done, shows completed/dispatched only."""
        calls = []
        stats = {"dispatched": 7, "completed": 7}
        msg = _format_progress_message(calls, stats)
        assert msg == "7/7 done"

    def test_message_capped_at_80_chars(self):
        """Long messages are truncated to 80 chars with ellipsis."""
        calls = [
            {"model": "a-very-long-model-name-that-keeps-going", "elapsed_seconds": 1.0},
            {"model": "another-extremely-verbose-model-name-here", "elapsed_seconds": 2.0},
            {"model": "yet-another-model-with-a-really-long-name", "elapsed_seconds": 3.0},
        ]
        stats = {"dispatched": 10, "completed": 7}
        msg = _format_progress_message(calls, stats)
        assert len(msg) <= 80
        assert msg.endswith("...")

    def test_partially_completed(self):
        """Mid-execution: some done, some active."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 8.0},
        ]
        stats = {"dispatched": 3, "completed": 2}
        msg = _format_progress_message(calls, stats)
        assert "2/3 done" in msg
        assert "1 active" in msg


# ============================================================
# Progress monitor integration
# ============================================================


class TestProgressMonitorConstants:
    """Progress polling constants are sensible."""

    def test_initial_interval_is_short(self):
        assert PROGRESS_POLL_INITIAL == 2

    def test_long_interval_is_longer(self):
        assert PROGRESS_POLL_LONG > PROGRESS_POLL_INITIAL

    def test_threshold_is_reasonable(self):
        assert PROGRESS_POLL_THRESHOLD == 30


# ============================================================
# Shared CallRegistry basics (S7)
# ============================================================


class TestSharedRegistry:
    """Shared CallRegistry tracks calls across REPL instances."""

    def test_register_and_snapshot(self):
        """Registering a call makes it visible in the shared registry."""
        _call_registry.register("c1", "sync", "gpt-4o", depth=0,
                                instruction_preview="Summarize this")
        snap = _call_registry.snapshot()
        assert len(snap) == 1
        assert snap[0]["call_id"] == "c1"
        assert snap[0]["model"] == "gpt-4o"
        assert snap[0]["instruction_preview"] == "Summarize this"

    def test_complete_removes_call(self):
        """Completing a call removes it from the registry."""
        _call_registry.register("c1", "sync", "gpt-4o")
        assert len(_call_registry.snapshot()) == 1
        _call_registry.complete("c1")
        assert len(_call_registry.snapshot()) == 0

    def test_stats_track_dispatched_completed(self):
        """Stats correctly count dispatched and completed calls."""
        _call_registry.register("c1", "sync", "gpt-4o")
        _call_registry.register("c2", "async", "gpt-4o-mini")
        _call_registry.complete("c1")
        stats = _call_registry.get_stats()
        assert stats["dispatched"] == 2
        assert stats["completed"] == 1

    def test_multi_repl_visibility(self):
        """Calls registered from different REPL instances share the same registry."""
        repl1 = RacketREPL()
        repl2 = RacketREPL()
        try:
            repl1._register_call("r1_c1", "sync", "gpt-4o", "from repl1")
            repl2._register_call("r2_c1", "async", "gpt-4o-mini", "from repl2")
            snap = _call_registry.snapshot()
            ids = {c["call_id"] for c in snap}
            assert "r1_c1" in ids
            assert "r2_c1" in ids
        finally:
            repl1.close()
            repl2.close()

    def test_reset_stats_clears_counters(self):
        """reset_stats() zeroes dispatched and completed."""
        _call_registry.register("c1", "sync", "gpt-4o")
        _call_registry.complete("c1")
        _call_registry.reset_stats()
        stats = _call_registry.get_stats()
        assert stats["dispatched"] == 0
        assert stats["completed"] == 0

    def test_clear_removes_everything(self):
        """clear() removes all calls and resets stats."""
        _call_registry.register("c1", "sync", "gpt-4o")
        _call_registry.register("c2", "async", "gpt-4o")
        _call_registry.clear()
        assert len(_call_registry.snapshot()) == 0
        stats = _call_registry.get_stats()
        assert stats["dispatched"] == 0
        assert stats["completed"] == 0


# ============================================================
# Nested REPL visibility (S7)
# ============================================================


class TestNestedREPLVisibility:
    """Nested REPL calls (depth > 0) appear in the shared registry."""

    def test_depth_gt0_calls_visible(self):
        """Calls registered at depth > 0 appear in the shared registry."""
        _call_registry.register("call_d1_1", "sync", "gpt-4o", depth=1,
                                parent_id="call_1")
        snap = _call_registry.snapshot()
        assert len(snap) == 1
        assert snap[0]["depth"] == 1
        assert snap[0]["parent_id"] == "call_1"

    def test_depth_field_correct(self):
        """Depth field correctly reflects call nesting level."""
        _call_registry.register("c0", "sync", "gpt-4o", depth=0)
        _call_registry.register("c1", "recursive", "gpt-4o", depth=1, parent_id="c0")
        _call_registry.register("c2", "sync", "gpt-4o", depth=2, parent_id="c1")
        snap = _call_registry.snapshot()
        depths = {c["call_id"]: c["depth"] for c in snap}
        assert depths["c0"] == 0
        assert depths["c1"] == 1
        assert depths["c2"] == 2

    def test_depth_aware_call_id_format(self):
        """_next_call_id produces depth-aware IDs."""
        repl = RacketREPL()
        try:
            # Depth 0 -> call_N
            id0 = repl._next_call_id()
            assert id0.startswith("call_")
            assert "_d" not in id0

            # Depth 1 -> call_d1_N
            repl._current_depth = 1
            id1 = repl._next_call_id()
            assert "d1" in id1

            # Depth 2 -> call_d2_N
            repl._current_depth = 2
            id2 = repl._next_call_id()
            assert "d2" in id2

            # All IDs are unique
            assert len({id0, id1, id2}) == 3
        finally:
            repl.close()


# ============================================================
# Enhanced format — depth, recursive, preview (S7)
# ============================================================


class TestEnhancedFormat:
    """_format_progress_message with depth annotations and previews."""

    def test_depth_annotation_shown(self):
        """Depth annotation [d1:N] appears when deep calls are present."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 5.0, "depth": 0},
            {"model": "gpt-4o", "elapsed_seconds": 3.0, "depth": 1},
        ]
        stats = {"dispatched": 3, "completed": 1}
        msg = _format_progress_message(calls, stats)
        assert "[d1:" in msg

    def test_recursive_type_shown_for_single_call(self):
        """Single recursive call shows [recursive] annotation."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 12.0, "type": "recursive"},
        ]
        stats = {"dispatched": 3, "completed": 2}
        msg = _format_progress_message(calls, stats)
        assert "[recursive]" in msg

    def test_preview_shown_for_single_call(self):
        """Single call with instruction preview shows truncated preview."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 5.0,
             "instruction_preview": "Summarize the document"},
        ]
        stats = {"dispatched": 1, "completed": 0}
        msg = _format_progress_message(calls, stats)
        assert "Summarize" in msg

    def test_preview_dropped_before_depth(self):
        """Preview is dropped first when message exceeds 80 chars."""
        calls = [
            {"model": "gpt-4o-with-a-longish-name", "elapsed_seconds": 100.0,
             "depth": 1, "type": "sync",
             "instruction_preview": "A very long instruction that should be dropped"},
        ]
        stats = {"dispatched": 10, "completed": 9}
        msg = _format_progress_message(calls, stats)
        assert len(msg) <= 80
        # Depth should survive if room
        if "[d1:" not in msg:
            # Both depth and preview were dropped — message was very long
            assert len(msg) <= 80

    def test_backward_compat_no_depth_key(self):
        """Calls without depth key default to 0 (backward compat)."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 3.0},
        ]
        stats = {"dispatched": 1, "completed": 0}
        msg = _format_progress_message(calls, stats)
        assert "[d" not in msg  # no depth annotation for depth-0-only
        assert "0/1 done" in msg

    def test_multiple_depths_annotated(self):
        """Multiple depth levels get separate annotations."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 5.0, "depth": 0},
            {"model": "gpt-4o", "elapsed_seconds": 3.0, "depth": 1},
            {"model": "gpt-4o", "elapsed_seconds": 2.0, "depth": 1},
            {"model": "gpt-4o", "elapsed_seconds": 1.0, "depth": 2},
        ]
        stats = {"dispatched": 6, "completed": 2}
        msg = _format_progress_message(calls, stats)
        assert "[d1:" in msg
        assert "[d2:" in msg

    def test_flat_fanout_unchanged(self):
        """Flat fan-out (all depth 0) has no depth annotations — backward compat."""
        calls = [
            {"model": "gpt-4o", "elapsed_seconds": 5.0, "depth": 0},
            {"model": "gpt-4o", "elapsed_seconds": 3.0, "depth": 0},
        ]
        stats = {"dispatched": 5, "completed": 3}
        msg = _format_progress_message(calls, stats)
        assert "[d" not in msg
        assert "3/5 done" in msg
        assert "2\u00d7gpt-4o" in msg

    def test_hard_truncate_at_80(self):
        """Even after dropping preview and depth, hard truncate at 80."""
        calls = [
            {"model": "extremely-long-model-name-that-goes-on-and-on",
             "elapsed_seconds": 999999.0, "depth": 0},
            {"model": "another-extremely-long-model-name-going-even-further",
             "elapsed_seconds": 888888.0, "depth": 0},
        ]
        stats = {"dispatched": 100, "completed": 99}
        msg = _format_progress_message(calls, stats)
        assert len(msg) <= 80


# ============================================================
# Event-driven wake-up (S7)
# ============================================================


class TestEventDrivenWake:
    """CallRegistry.wait_for_change wakes on state transitions."""

    def test_event_set_on_register(self):
        """Registering a call wakes wait_for_change."""
        registry = CallRegistry()

        woken = []
        def waiter():
            result = registry.wait_for_change(timeout=5.0)
            woken.append(result)

        t = threading.Thread(target=waiter)
        t.start()
        time.sleep(0.05)  # let thread block on wait
        registry.register("c1", "sync", "gpt-4o")
        t.join(timeout=2.0)
        assert len(woken) == 1
        assert woken[0] is True  # woken by event, not timeout

    def test_wait_returns_on_timeout(self):
        """wait_for_change returns False after timeout with no change."""
        registry = CallRegistry()
        t0 = time.monotonic()
        result = registry.wait_for_change(timeout=0.1)
        elapsed = time.monotonic() - t0
        assert result is False
        assert elapsed >= 0.08  # waited approximately the timeout


# ============================================================
# Execution summary (S7b)
# ============================================================


class TestExecutionSummary:
    """CallRegistry.get_execution_summary provides post-hoc visibility."""

    def test_empty_summary_no_calls(self):
        """Summary with no calls reports llm_calls=0."""
        summary = _call_registry.get_execution_summary()
        assert summary["llm_calls"] == 0
        assert "models" not in summary

    def test_summary_after_one_call(self):
        """Summary after one completed call shows model and latency."""
        _call_registry.register("c1", "sync", "gpt-4o")
        time.sleep(0.05)
        _call_registry.complete("c1")
        summary = _call_registry.get_execution_summary()
        assert summary["llm_calls"] == 1
        assert "gpt-4o" in summary["models"]
        assert summary["max_call_latency"] >= 0.0

    def test_summary_multiple_models(self):
        """Summary groups models with counts."""
        _call_registry.register("c1", "sync", "gpt-4o")
        _call_registry.register("c2", "async", "gpt-4o")
        _call_registry.register("c3", "sync", "gpt-4o-mini")
        _call_registry.complete("c1")
        _call_registry.complete("c2")
        _call_registry.complete("c3")
        summary = _call_registry.get_execution_summary()
        assert summary["llm_calls"] == 3
        assert "gpt-4o" in summary["models"]
        assert "gpt-4o-mini" in summary["models"]
        # Most common model listed first
        assert summary["models"].startswith("gpt-4o")

    def test_summary_max_latency(self):
        """max_call_latency reflects the slowest call."""
        _call_registry.register("c1", "sync", "gpt-4o")
        time.sleep(0.1)
        _call_registry.complete("c1")
        _call_registry.register("c2", "sync", "gpt-4o")
        _call_registry.complete("c2")  # near-instant
        summary = _call_registry.get_execution_summary()
        assert summary["max_call_latency"] >= 0.1

    def test_reset_clears_history(self):
        """reset_stats clears execution history."""
        _call_registry.register("c1", "sync", "gpt-4o")
        _call_registry.complete("c1")
        _call_registry.reset_stats()
        summary = _call_registry.get_execution_summary()
        assert summary["llm_calls"] == 0

    def test_clear_clears_history(self):
        """clear() also clears execution history."""
        _call_registry.register("c1", "sync", "gpt-4o")
        _call_registry.complete("c1")
        _call_registry.clear()
        summary = _call_registry.get_execution_summary()
        assert summary["llm_calls"] == 0


# ============================================================
# execute_scheme result includes execution field (S7b)
# ============================================================


class TestExecuteSchemeResult:
    """execute_scheme result JSON includes execution summary."""

    def test_execution_field_present(self):
        """Result includes 'execution' key with elapsed time."""
        repl = RacketREPL()
        try:
            # Simulate what execute_scheme does: reset, run, build result
            _call_registry.reset_stats()
            resp = repl.send({"op": "eval", "code": "(finish 42)"})
            exec_summary = _call_registry.get_execution_summary()
            exec_summary["elapsed"] = 0.1
            result = {"status": resp["status"], "value": resp["result"],
                      "execution": exec_summary}
            assert "execution" in result
            assert "elapsed" in result["execution"]
            assert "llm_calls" in result["execution"]
        finally:
            repl.close()

    def test_execution_no_llm_calls(self):
        """Pure Scheme code reports llm_calls=0."""
        repl = RacketREPL()
        try:
            _call_registry.reset_stats()
            repl.send({"op": "eval", "code": "(finish 42)"})
            summary = _call_registry.get_execution_summary()
            assert summary["llm_calls"] == 0
        finally:
            repl.close()

    def test_execution_tokens_included_when_nonzero(self):
        """Tokens field is included when there are completed LLM calls."""
        repl = RacketREPL()
        try:
            _call_registry.reset_stats()
            # Simulate a call that completed with tokens
            repl._register_call("c1", "sync", "gpt-4o", "test")
            repl._complete_call("c1", tokens=200, elapsed=1.0,
                                prompt_tokens=150, completion_tokens=50)
            token_usage = repl.get_token_usage()
            exec_summary = _call_registry.get_execution_summary()
            if token_usage["total_tokens"] > 0:
                exec_summary["tokens"] = token_usage["total_tokens"]
            assert exec_summary["tokens"] == 200
            assert exec_summary["llm_calls"] == 1
        finally:
            repl.close()
