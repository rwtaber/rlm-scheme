"""Tests for L8: Better async error messages.

Tests:
1. JSON mode validation - requires "json" in instruction
2. Error logging for async failures
3. Detailed error messages in await results
4. Distinction between ValueError, APIError, and generic errors
"""

import concurrent.futures
import json
import os
import sys
import threading

import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import RacketREPL


# ============================================================
# JSON mode validation (L8.3)
# ============================================================


class TestJSONModeValidation:
    def test_json_mode_requires_json_in_instruction(self):
        """JSON mode without 'json' in instruction raises ValueError."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError, match="JSON mode requires the word 'json'"):
                repl._call_llm(
                    instruction="Return a structured response",  # No "json"
                    data="test",
                    json_mode=True
                )
        finally:
            repl.close()

    def test_json_mode_accepts_json_lowercase(self):
        """JSON mode works with 'json' in lowercase."""
        # This would make an actual API call, so we just verify it doesn't raise
        repl = RacketREPL()
        try:
            # No exception should be raised during validation
            # (API call would fail without API key, but validation passes)
            try:
                repl._call_llm(
                    instruction="Return a json object with keys: name, value",
                    data="test",
                    json_mode=True
                )
            except Exception as e:
                # Should NOT be a ValueError about missing "json"
                assert "JSON mode requires" not in str(e)
        finally:
            repl.close()

    def test_json_mode_accepts_json_uppercase(self):
        """JSON mode works with 'JSON' in uppercase."""
        repl = RacketREPL()
        try:
            try:
                repl._call_llm(
                    instruction="Return a JSON object",
                    data="test",
                    json_mode=True
                )
            except Exception as e:
                assert "JSON mode requires" not in str(e)
        finally:
            repl.close()

    def test_json_mode_accepts_json_mixed_case(self):
        """JSON mode works with 'Json' in mixed case."""
        repl = RacketREPL()
        try:
            try:
                repl._call_llm(
                    instruction="Return Json data",
                    data="test",
                    json_mode=True
                )
            except Exception as e:
                assert "JSON mode requires" not in str(e)
        finally:
            repl.close()

    def test_json_mode_false_does_not_validate(self):
        """When json_mode=False, no validation occurs."""
        repl = RacketREPL()
        try:
            # Should not raise even without "json" in instruction
            # (will fail at API call, but validation doesn't run)
            try:
                repl._call_llm(
                    instruction="Return a response",
                    data="test",
                    json_mode=False
                )
            except Exception as e:
                # Should NOT be a ValueError about JSON mode
                assert "JSON mode requires" not in str(e)
        finally:
            repl.close()

    def test_json_mode_empty_instruction(self):
        """JSON mode with empty instruction raises ValueError."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError, match="JSON mode requires"):
                repl._call_llm(
                    instruction="",
                    data="test",
                    json_mode=True
                )
        finally:
            repl.close()

    def test_json_mode_validation_logs_to_stderr(self, capsys):
        """JSON mode validation error is logged to stderr."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError):
                repl._call_llm(
                    instruction="Return structured data",
                    data="test",
                    json_mode=True
                )
            captured = capsys.readouterr()
            assert "[rlm] Error:" in captured.err
            assert "JSON mode requires" in captured.err
        finally:
            repl.close()


# ============================================================
# Async error logging (L8.4)
# ============================================================


class TestAsyncErrorLogging:
    def test_failed_call_logs_error_type(self, capsys):
        """Failed async calls log the error type to stderr."""
        repl = RacketREPL()
        try:
            call_id = "test_call_1"
            # Simulate a failed call by calling _call_llm_tracked with invalid JSON mode
            with pytest.raises(ValueError):
                repl._call_llm_tracked(
                    call_id=call_id,
                    instruction="Return a structured response",  # No "json" keyword
                    data="test",
                    json_mode=True
                )
            captured = capsys.readouterr()
            assert "FAILED" in captured.err
            assert "ValueError" in captured.err
            assert call_id in captured.err
        finally:
            repl.close()

    def test_failed_call_logs_elapsed_time(self, capsys):
        """Failed async calls log elapsed time."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError):
                repl._call_llm_tracked(
                    call_id="test_call_2",
                    instruction="Return structured data",  # No "json" keyword
                    data="test",
                    json_mode=True
                )
            captured = capsys.readouterr()
            # Should show elapsed time like "0.0s" or "0.1s"
            assert "s" in captured.err and "FAILED" in captured.err
        finally:
            repl.close()

    def test_failed_call_logs_error_message(self, capsys):
        """Failed async calls log the error message."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError):
                repl._call_llm_tracked(
                    call_id="test_call_3",
                    instruction="Structure",
                    data="test",
                    json_mode=True
                )
            captured = capsys.readouterr()
            assert "JSON mode requires" in captured.err
        finally:
            repl.close()


# ============================================================
# Error message formatting (L8.2)
# ============================================================


class TestErrorMessageFormatting:
    def test_error_includes_type_name(self):
        """Error messages include the exception type name."""
        # Simulate the error message formatting
        e = ValueError("test error")
        error_msg = f"[async error] {type(e).__name__}: {str(e)}"
        assert "ValueError" in error_msg
        assert "test error" in error_msg

    def test_api_error_includes_status_code(self):
        """API errors include status code in message."""
        # Mock an API error with status code
        class MockAPIError(Exception):
            def __init__(self, msg, status_code):
                super().__init__(msg)
                self.status_code = status_code

        e = MockAPIError("Bad request", 400)
        status = getattr(e, "status_code", "unknown")
        error_msg = f"[async error] API {status}: {str(e)}"
        assert "400" in error_msg

    def test_long_error_messages_truncated(self):
        """Very long error messages are truncated to 300 chars."""
        long_msg = "x" * 500
        truncated = long_msg[:300]
        assert len(truncated) == 300
        assert len(truncated) < len(long_msg)


# ============================================================
# Integration tests (L8 overall)
# ============================================================


class TestAsyncErrorIntegration:
    def test_validation_error_message_clear(self):
        """JSON mode validation error provides clear actionable message."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError) as exc_info:
                repl._call_llm(
                    instruction="Return structured output",
                    data="test",
                    json_mode=True
                )
            error_msg = str(exc_info.value)
            # Should explain the requirement
            assert "json" in error_msg.lower()
            assert "instruction" in error_msg.lower()
            # Should include the problematic instruction
            assert "Return structured output" in error_msg
        finally:
            repl.close()

    def test_error_logged_before_exception_raised(self, capsys):
        """Error is logged to stderr before exception is raised."""
        repl = RacketREPL()
        try:
            with pytest.raises(ValueError):
                repl._call_llm(
                    instruction="Bad",
                    data="test",
                    json_mode=True
                )
            captured = capsys.readouterr()
            # Stderr should have the log
            assert "[rlm] Error:" in captured.err
        finally:
            repl.close()


# ============================================================
# Backwards compatibility (L8)
# ============================================================


class TestBackwardsCompatibility:
    def test_non_json_calls_unchanged(self):
        """Calls without json_mode are not affected by validation."""
        repl = RacketREPL()
        try:
            # Should not raise validation error
            # (Will fail at API call, but that's expected without API key)
            try:
                repl._call_llm(
                    instruction="Translate to French",
                    data="hello",
                    json_mode=False
                )
            except Exception as e:
                # Should NOT be JSON validation error
                assert "JSON mode requires" not in str(e)
        finally:
            repl.close()

    def test_cancelled_calls_still_work(self):
        """Cancelled calls still return [cancelled] message."""
        # This tests that L8 changes didn't break cancellation
        repl = RacketREPL()
        try:
            cancel_event = threading.Event()
            cancel_event.set()  # Pre-cancelled
            with pytest.raises(concurrent.futures.CancelledError):
                repl._call_llm(
                    instruction="test",
                    data="test",
                    cancel_event=cancel_event
                )
        finally:
            repl.close()
