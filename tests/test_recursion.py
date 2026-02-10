"""Tests for depth>1 recursive llm-query.

Tests the #:recursive keyword which gives sub-models their own Scheme sandbox.
Uses subprocess.Popen for Racket-level callback tests and RacketREPL for
Python-level integration tests.
"""

import json
import os
import subprocess
import sys
import time

import pytest

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")

# Add project root to path so we can import mcp_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import RacketREPL, MAX_RECURSION_DEPTH


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
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()


def read_msg(proc, timeout=10):
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Server died")
    return json.loads(line.strip())


def write_response(proc, resp):
    proc.stdin.write(json.dumps(resp) + "\n")
    proc.stdin.flush()


# ============================================================
# Racket-level: callback protocol for #:recursive
# ============================================================


class TestRecursiveCallback:
    def test_recursive_flag_forwarded(self):
        """#:recursive #t is forwarded as true in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test" #:recursive #t)))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["recursive"] is True

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_recursive_flag_default_false(self):
        """Without #:recursive, the callback has recursive=false."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["recursive"] is False

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_budget_included_in_callback(self):
        """Budget value is included in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(parameterize ([token-budget 500]) (finish (syntax-e (llm-query #:instruction "test" #:recursive #t))))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["budget"] == 500

            write_response(proc, {"result": "ok", "prompt_tokens": 10, "completion_tokens": 5})
            result = read_msg(proc)
            assert result["status"] == "finished"
        finally:
            proc.kill()

    def test_budget_null_when_infinite(self):
        """Budget is null in JSON when token-budget is +inf.0."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["budget"] is None

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_unsafe_raw_query_recursive(self):
        """#:recursive works on unsafe-raw-query too."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (unsafe-raw-query #:instruction "test" #:recursive #t))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["recursive"] is True

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()


# ============================================================
# Python-level: _extract_code
# ============================================================


class TestExtractCode:
    def test_extracts_from_scheme_fence(self):
        """Extracts code from ```scheme ... ``` fences."""
        text = "Here's the code:\n```scheme\n(finish 42)\n```\nDone."
        assert RacketREPL._extract_code(text) == "(finish 42)"

    def test_extracts_from_racket_fence(self):
        """Extracts code from ```racket ... ``` fences."""
        text = "```racket\n(finish 42)\n```"
        assert RacketREPL._extract_code(text) == "(finish 42)"

    def test_extracts_from_bare_fence(self):
        """Extracts code from ``` ... ``` fences with no language tag."""
        text = "Result:\n```\n(finish 42)\n```"
        assert RacketREPL._extract_code(text) == "(finish 42)"

    def test_returns_raw_when_no_fence(self):
        """Returns the full text stripped when no fences present."""
        text = "(finish 42)"
        assert RacketREPL._extract_code(text) == "(finish 42)"

    def test_multiline_code(self):
        """Handles multi-line code blocks."""
        text = "```scheme\n(define x 1)\n(define y 2)\n(finish (+ x y))\n```"
        expected = "(define x 1)\n(define y 2)\n(finish (+ x y))"
        assert RacketREPL._extract_code(text) == expected


# ============================================================
# Python-level: recursive call integration
# ============================================================


class TestRecursiveIntegration:
    def test_depth_limit_constant(self):
        """MAX_RECURSION_DEPTH is accessible and reasonable."""
        assert MAX_RECURSION_DEPTH >= 1
        assert MAX_RECURSION_DEPTH <= 10

    def test_nested_sandbox_lifecycle(self):
        """A nested RacketREPL can evaluate code and is cleaned up."""
        nested = RacketREPL()
        try:
            resp = nested.send({"op": "eval", "code": "(finish 42)"})
            assert resp["status"] == "finished"
            assert resp["result"] == "42"
        finally:
            nested.close()
        assert nested.proc is None

    def test_nested_sandbox_with_context(self):
        """A nested sandbox can load context and use it."""
        nested = RacketREPL()
        try:
            nested.send({"op": "load-context", "data": "hello world"})
            resp = nested.send({"op": "eval", "code": "(finish context)"})
            assert resp["status"] == "finished"
            assert resp["result"] == "hello world"
        finally:
            nested.close()

    def test_extract_and_eval(self):
        """Simulates the recursive flow: extract code from markdown, eval in nested sandbox."""
        llm_response = "Here is the code:\n```scheme\n(finish (+ 1 2))\n```"
        code = RacketREPL._extract_code(llm_response)

        nested = RacketREPL()
        try:
            resp = nested.send({"op": "eval", "code": code})
            assert resp["status"] == "finished"
            assert resp["result"] == "3"
        finally:
            nested.close()

    def test_current_depth_default(self):
        """Default _current_depth is 0."""
        repl = RacketREPL()
        try:
            assert repl._current_depth == 0
        finally:
            repl.close()

    def test_current_depth_propagation(self):
        """_current_depth can be set on a nested REPL."""
        nested = RacketREPL()
        nested._current_depth = 2
        try:
            assert nested._current_depth == 2
            resp = nested.send({"op": "eval", "code": "(finish 1)"})
            assert resp["status"] == "finished"
        finally:
            nested.close()

    def test_depth_limit_exceeded(self):
        """_call_llm_recursive returns error when depth exceeds limit."""
        repl = RacketREPL()
        repl._current_depth = MAX_RECURSION_DEPTH  # Already at max
        try:
            result = repl._call_llm_recursive("test", "", "")
            assert "[error]" in result["text"]
            assert "depth limit" in result["text"]
        finally:
            repl.close()
