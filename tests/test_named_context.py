"""Tests for named context slots (Priority 1 improvement #5).

Tests that users can load and retrieve multiple datasets with clear names,
avoiding manual tracking with comments.
"""

import json
import os
import subprocess

import pytest

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")


def send_commands(*commands: dict, timeout: int = 30) -> list[dict]:
    """Send JSON commands to the Racket server, return responses."""
    input_text = "\n".join(json.dumps(cmd) for cmd in commands) + "\n"
    result = subprocess.run(
        ["racket", RACKET_SERVER],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    responses = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            responses.append(json.loads(line))
    return responses


# ============================================================
# Named context tests
# ============================================================


class TestNamedContext:
    """Test named context slots for managing multiple datasets."""

    def test_get_context_function_exists(self):
        """Verify get-context function is available."""
        responses = send_commands(
            {"op": "eval", "code": "(procedure? get-context)"}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "ok"

    def test_load_named_context(self):
        """Load context with a name and retrieve it."""
        responses = send_commands(
            {"op": "load-context", "data": "test data", "name": "test-dataset"},
            {"op": "eval", "code": '(finish (get-context "test-dataset"))'}
        )
        assert len(responses) == 2
        # First response: load confirmation
        assert responses[0]["status"] == "ok"
        assert "test-dataset" in responses[0]["result"]
        # Second response: retrieved data (finish changes status to "finished")
        assert responses[1]["status"] == "finished"
        assert responses[1]["result"] == "test data"

    def test_multiple_named_contexts(self):
        """Load multiple named contexts and retrieve them independently."""
        responses = send_commands(
            {"op": "load-context", "data": "gwas data", "name": "gwas"},
            {"op": "load-context", "data": "expression data", "name": "expression"},
            {"op": "load-context", "data": "network data", "name": "networks"},
            {"op": "eval", "code": '(finish (list (get-context "gwas") (get-context "expression") (get-context "networks")))'}
        )
        assert len(responses) == 4
        # All loads succeeded
        assert all(r["status"] == "ok" for r in responses[:3])
        # Retrieved all three correctly (finish changes status to "finished")
        assert responses[3]["status"] == "finished"
        result = responses[3]["result"]
        assert "gwas data" in result
        assert "expression data" in result
        assert "network data" in result

    def test_get_context_nonexistent_returns_false(self):
        """Getting a non-existent context returns #f."""
        responses = send_commands(
            {"op": "eval", "code": '(finish (get-context "nonexistent"))'}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "finished"
        assert responses[0]["result"] == "#f"

    def test_backward_compatibility_unnamed_context(self):
        """Loading context without name still works (backward compatibility)."""
        responses = send_commands(
            {"op": "load-context", "data": "default data"},
            {"op": "eval", "code": "(finish context)"}
        )
        assert len(responses) == 2
        assert responses[0]["status"] == "ok"
        assert responses[1]["status"] == "finished"
        assert responses[1]["result"] == "default data"

    def test_named_context_updates_default_context(self):
        """Loading a named context also updates the default 'context' variable."""
        responses = send_commands(
            {"op": "load-context", "data": "named data", "name": "my-data"},
            {"op": "eval", "code": "(finish context)"}
        )
        assert len(responses) == 2
        # The 'context' variable should have the most recent data
        assert responses[1]["result"] == "named data"

    def test_overwrite_named_context(self):
        """Loading context with same name overwrites previous value."""
        responses = send_commands(
            {"op": "load-context", "data": "first value", "name": "dataset"},
            {"op": "load-context", "data": "second value", "name": "dataset"},
            {"op": "eval", "code": '(finish (get-context "dataset"))'}
        )
        assert len(responses) == 3
        # Should have the second value
        assert responses[2]["result"] == "second value"

    def test_context_store_is_hash(self):
        """context-store is a hash table."""
        responses = send_commands(
            {"op": "eval", "code": "(finish (hash? context-store))"}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "#t"

    def test_get_context_with_special_characters(self):
        """Named contexts can use names with special characters."""
        responses = send_commands(
            {"op": "load-context", "data": "data", "name": "my-dataset_v2.0"},
            {"op": "eval", "code": '(finish (get-context "my-dataset_v2.0"))'}
        )
        assert len(responses) == 2
        assert responses[1]["result"] == "data"

    def test_reset_clears_named_contexts(self):
        """Reset command clears all named contexts."""
        responses = send_commands(
            {"op": "load-context", "data": "data", "name": "test"},
            {"op": "reset"},
            {"op": "eval", "code": '(finish (get-context "test"))'}
        )
        assert len(responses) == 3
        # After reset, context should be gone
        assert responses[2]["result"] == "#f"
