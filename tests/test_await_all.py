"""Tests for await-all and await-all-syntax batch collection.

Tests the Racket sandbox's await-all (returns unwrapped strings) and
await-all-syntax (returns syntax objects) using the interactive callback
protocol. No real API calls.
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


def write_response(proc, resp):
    """Write a JSON response (for llm-query callbacks)."""
    proc.stdin.write(json.dumps(resp) + "\n")
    proc.stdin.flush()


# ============================================================
# await-all
# ============================================================


class TestAwaitAll:
    def test_await_all_single(self):
        """await-all with a single handle returns a list of one string."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h (llm-query-async #:instruction "test"))
(finish (await-all (list h)))
"""})

            # Read async dispatch
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            async_id = msg["id"]

            # Read await-batch
            msg2 = read_msg(proc)
            assert msg2["op"] == "await-batch"
            assert msg2["ids"] == [async_id]

            write_response(proc, {
                "results": [{
                    "result": "hello",
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            # Should be a list containing the unwrapped string
            assert "hello" in result["result"]
        finally:
            proc.kill()

    def test_await_all_multiple(self):
        """await-all collects 3 async results into a list of strings."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h1 (llm-query-async #:instruction "q1"))
(define h2 (llm-query-async #:instruction "q2"))
(define h3 (llm-query-async #:instruction "q3"))
(finish (await-all (list h1 h2 h3)))
"""})

            # Read 3 async dispatches
            ids = []
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Read await-batch and respond
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert msg["ids"] == ids
            write_response(proc, {
                "results": [
                    {"result": "answer1", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "answer2", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "answer3", "prompt_tokens": 5, "completion_tokens": 3}
                ]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "answer1" in result["result"]
            assert "answer2" in result["result"]
            assert "answer3" in result["result"]
        finally:
            proc.kill()

    def test_await_all_returns_strings_not_syntax(self):
        """await-all returns plain strings, so string-append works directly."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h1 (llm-query-async #:instruction "q1"))
(define h2 (llm-query-async #:instruction "q2"))
(define results (await-all (list h1 h2)))
(finish (string-append (car results) " " (cadr results)))
"""})

            # Read 2 dispatches
            ids = []
            for i in range(2):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Read await-batch and respond
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert msg["ids"] == ids
            write_response(proc, {
                "results": [
                    {"result": "word1", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "word2", "prompt_tokens": 5, "completion_tokens": 3}
                ]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "word1 word2"
        finally:
            proc.kill()

    def test_await_all_empty_list(self):
        """await-all with empty list returns empty list."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(finish (await-all '()))
"""})

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "()"
        finally:
            proc.kill()

    def test_await_all_with_map_pattern(self):
        """await-all works with the map-over-data fan-out pattern."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define data (list "a" "b" "c"))
(define handles (map (lambda (d) (llm-query-async #:instruction "process" #:data d)) data))
(define results (await-all handles))
(finish results)
"""})

            # Read 3 dispatches
            ids = []
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])
                assert msg["data"] in ["a", "b", "c"]

            # Read await-batch and respond
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert msg["ids"] == ids
            write_response(proc, {
                "results": [
                    {"result": "result-0", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "result-1", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "result-2", "prompt_tokens": 5, "completion_tokens": 3}
                ]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "result-0" in result["result"]
            assert "result-1" in result["result"]
            assert "result-2" in result["result"]
        finally:
            proc.kill()


# ============================================================
# await-all-syntax
# ============================================================


class TestAwaitAllSyntax:
    def test_await_all_syntax_returns_syntax_objects(self):
        """await-all-syntax returns syntax objects that need syntax-e to unwrap."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h1 (llm-query-async #:instruction "q1"))
(define h2 (llm-query-async #:instruction "q2"))
(define stx-results (await-all-syntax (list h1 h2)))
(finish (string-append (syntax-e (car stx-results)) " " (syntax-e (cadr stx-results))))
"""})

            # Read 2 dispatches
            ids = []
            for i in range(2):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Read await-batch and respond
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert msg["ids"] == ids
            write_response(proc, {
                "results": [
                    {"result": "stx1", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "stx2", "prompt_tokens": 5, "completion_tokens": 3}
                ]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "stx1 stx2"
        finally:
            proc.kill()


# ============================================================
# Protected bindings
# ============================================================


class TestAwaitAllProtection:
    def test_cannot_redefine_await_all(self):
        """await-all is a protected scaffold binding and cannot be redefined."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define (await-all x) "hacked")
"""})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "scaffold" in result["message"].lower() or "protected" in result["message"].lower() or "redefine" in result["message"].lower()
        finally:
            proc.kill()

    def test_cannot_redefine_await_all_syntax(self):
        """await-all-syntax is a protected scaffold binding."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define (await-all-syntax x) "hacked")
"""})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "scaffold" in result["message"].lower() or "protected" in result["message"].lower() or "redefine" in result["message"].lower()
        finally:
            proc.kill()
