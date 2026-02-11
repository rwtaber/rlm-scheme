"""Tests for get_sandbox_state() MCP tool."""

import json
import os
import pytest
from mcp_server import get_backend, CHECKPOINT_DIR


def test_get_sandbox_state_basic():
    """Test that get_sandbox_state returns expected structure."""
    backend = get_backend()

    # Reset to clean state
    backend.send({"op": "reset"})

    # Define some variables
    backend.send({
        "op": "eval",
        "code": "(define my-var 42)\n(define result \"test\")"
    })

    # Get state via inspect-state (the racket operation)
    resp = backend.send({"op": "inspect-state"})

    assert resp["status"] == "ok"
    assert "variables" in resp
    assert isinstance(resp["variables"], list)
    assert "my-var" in resp["variables"]
    assert "result" in resp["variables"]
    assert "python_available" in resp
    assert isinstance(resp["python_available"], bool)


def test_get_sandbox_state_after_reset():
    """Test that state is cleared after reset."""
    backend = get_backend()

    # Define some variables
    backend.send({
        "op": "eval",
        "code": "(define temp-var 123)"
    })

    # Verify variable exists
    resp1 = backend.send({"op": "inspect-state"})
    assert "temp-var" in resp1["variables"]

    # Reset
    backend.send({"op": "reset"})

    # Check state after reset
    resp2 = backend.send({"op": "inspect-state"})
    assert "temp-var" not in resp2["variables"]


def test_get_sandbox_state_with_checkpoints():
    """Test that get_sandbox_state includes checkpoint information."""
    backend = get_backend()

    # Create a checkpoint
    backend.send({
        "op": "eval",
        "code": "(checkpoint \"test_checkpoint\" \"some data\")"
    })

    # The mcp_server.get_sandbox_state would read from CHECKPOINT_DIR
    # Here we just verify the checkpoint file exists
    checkpoint_file = os.path.join(CHECKPOINT_DIR, "test_checkpoint.json")
    assert os.path.exists(checkpoint_file)

    # Clean up
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)


def test_scaffold_variables_not_included():
    """Test that scaffold-provided variables are not in user variables."""
    backend = get_backend()

    # Reset to clean state
    backend.send({"op": "reset"})

    # Scaffold provides: context, llm-query, finish, etc.
    # These should NOT appear in user variables
    resp = backend.send({"op": "inspect-state"})
    variables = resp["variables"]

    # Check that scaffold names are excluded
    scaffold_names = ["context", "finish", "llm-query", "py-exec", "syntax-e"]
    for name in scaffold_names:
        assert name not in variables, f"Scaffold variable '{name}' should not be in user variables"


def test_python_bridge_availability():
    """Test that python_available reflects actual py_bridge state."""
    backend = get_backend()

    # After reset, Python bridge should be available
    backend.send({"op": "reset"})
    resp = backend.send({"op": "inspect-state"})

    # Python bridge should be running
    assert resp["python_available"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
