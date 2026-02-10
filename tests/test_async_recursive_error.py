"""Tests for llm-query-async + #:recursive error handling.

Verifies that using #:recursive with llm-query-async produces a clear error
message instead of a cryptic "unexpected keyword" error.
"""

import json
import os
import subprocess

import pytest

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")


def start_server():
    """Start a Racket server subprocess for interactive testing."""
    return subprocess.Popen(
        ["racket", RACKET_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )


def write_cmd(proc, cmd):
    """Write a JSON command to the server."""
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()


def read_msg(proc, timeout=10):
    """Read one JSON message from the server."""
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Server died")
    return json.loads(line.strip())


class TestAsyncRecursiveError:
    def test_async_with_recursive_gives_clear_error(self):
        """llm-query-async with #:recursive should give a clear error message."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define handle (llm-query-async
  #:instruction "test"
  #:data "test"
  #:recursive #t))
"""})
            result = read_msg(proc)

            # Should get an error
            assert result["status"] == "error"

            # Error message should mention:
            # 1. That #:recursive is not supported with async
            # 2. To use synchronous llm-query instead
            error_msg = result["message"].lower()
            assert "recursive" in error_msg or "#:recursive" in result["message"]
            assert "not supported" in error_msg or "cannot" in error_msg
            assert "llm-query" in error_msg or "synchronous" in error_msg

            # Should NOT be a generic "unexpected keyword" error
            assert "unexpected keyword" not in error_msg
        finally:
            proc.kill()

    def test_async_without_recursive_works(self):
        """llm-query-async without #:recursive should work normally."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define handle (llm-query-async
  #:instruction "test"
  #:data "test"))
(finish handle)
"""})

            # First expect the llm-query-async dispatch
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"

            # We don't need to actually complete the async call for this test
            # Just kill the process - the point was to verify it accepted the call
        finally:
            proc.kill()

    def test_recursive_false_explicitly_works(self):
        """llm-query-async with #:recursive #f should work (explicit false)."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define handle (llm-query-async
  #:instruction "test"
  #:data "test"
  #:recursive #f))
(finish handle)
"""})

            # First expect the llm-query-async dispatch
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
        finally:
            proc.kill()

    def test_error_message_mentions_solution(self):
        """Error message should suggest using llm-query instead."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(llm-query-async #:instruction "test" #:recursive #t)
"""})
            result = read_msg(proc)

            assert result["status"] == "error"
            # Should suggest using llm-query (the synchronous version)
            assert "llm-query" in result["message"]
            # Should be clear about the alternative
            assert ("synchronous" in result["message"].lower() or
                    "Use llm-query" in result["message"] or
                    "use llm-query" in result["message"])
        finally:
            proc.kill()
