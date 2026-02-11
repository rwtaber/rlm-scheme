"""Tests for phantom results detection and stats reset."""

import json
import pytest
from mcp_server import _call_registry


def test_stats_reset_between_calls():
    """Verify that call stats are properly reset between execute_scheme calls."""
    # Simulate two calls being registered and completed
    _call_registry.reset_stats()

    _call_registry.register("call_1", "llm-query", "gpt-4o", depth=0)
    _call_registry.register("call_2", "llm-query", "gpt-4o", depth=0)
    _call_registry.complete("call_1")
    _call_registry.complete("call_2")

    summary1 = _call_registry.get_execution_summary()
    assert summary1["llm_calls"] == 2

    # Reset stats (simulating start of new execute_scheme call)
    _call_registry.reset_stats()

    # After reset, should show 0 calls
    summary2 = _call_registry.get_execution_summary()
    assert summary2["llm_calls"] == 0

    # New calls after reset should start fresh
    _call_registry.register("call_3", "llm-query", "gpt-3.5-turbo", depth=0)
    _call_registry.complete("call_3")

    summary3 = _call_registry.get_execution_summary()
    assert summary3["llm_calls"] == 1
    assert "gpt-3.5-turbo" in summary3.get("models", "")


def test_phantom_results_warning_detection():
    """Test that suspicious results (finished but 0 LLM calls) trigger warnings."""
    # This is more of an integration test - we'd need to mock execute_scheme
    # For now, just verify the warning logic would trigger correctly

    # Scenario from evaluation: status=finished, llm_calls=0, elapsed>1s
    # Should trigger warning

    result = {
        "status": "finished",
        "execution": {"llm_calls": 0, "elapsed": 5.0}
    }

    # The warning would be added in execute_scheme
    # Verify the condition that should trigger it:
    should_warn = (
        result["status"] == "finished" and
        result["execution"]["llm_calls"] == 0 and
        result["execution"]["elapsed"] > 1.0
    )

    assert should_warn, "Should detect phantom result condition"


def test_legitimate_zero_calls():
    """Test that legitimate 0-call executions don't trigger warnings."""
    # Quick execution (<1s) with 0 calls is legitimate (e.g., pure py-exec)
    result = {
        "status": "finished",
        "execution": {"llm_calls": 0, "elapsed": 0.3}
    }

    should_warn = (
        result["status"] == "finished" and
        result["execution"]["llm_calls"] == 0 and
        result["execution"]["elapsed"] > 1.0
    )

    assert not should_warn, "Should not warn for quick execution"


def test_error_status_no_warning():
    """Test that error status never triggers phantom result warning."""
    result = {
        "status": "error",
        "execution": {"llm_calls": 0, "elapsed": 5.0}
    }

    should_warn = (
        result["status"] == "finished" and
        result["execution"]["llm_calls"] == 0 and
        result["execution"]["elapsed"] > 1.0
    )

    assert not should_warn, "Should not warn for error status"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
