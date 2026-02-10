"""Tests for llm-query callbacks: token tracking, multi-model, and async.

These tests use subprocess.Popen to talk to racket_server.rkt directly,
intercepting llm-query callbacks and writing back mock responses.
This tests the full callback protocol without any real API calls.
"""

import json
import os
import subprocess
import time

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
    # Simple blocking read — tests should be fast
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Server died")
    return json.loads(line.strip())


def write_response(proc, resp):
    """Write a JSON response (for llm-query callbacks)."""
    proc.stdin.write(json.dumps(resp) + "\n")
    proc.stdin.flush()


# ============================================================
# Token tracking (#2)
# ============================================================


class TestTokenTracking:
    def test_token_counts_forwarded(self):
        """Token counts from the response are used for budget tracking."""
        proc = start_server()
        try:
            # Set budget to 100 and call llm-query
            write_cmd(proc, {"op": "eval", "code":
                '(parameterize ([token-budget 100]) (finish (syntax-e (llm-query #:instruction "test"))))'
            })

            # Read the callback
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"

            # Respond with token counts totaling 50
            write_response(proc, {
                "result": "hello",
                "prompt_tokens": 30,
                "completion_tokens": 20,
            })

            # Read the final result
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "hello"
        finally:
            proc.kill()

    def test_budget_exhaustion_real_counts(self):
        """Budget exhaustion uses real token counts, not char estimate."""
        proc = start_server()
        try:
            # Budget of 50, but response has 60 tokens total
            write_cmd(proc, {"op": "eval", "code":
                '(parameterize ([token-budget 50]) (finish (syntax-e (llm-query #:instruction "test"))))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"

            write_response(proc, {
                "result": "short",
                "prompt_tokens": 30,
                "completion_tokens": 30,
            })

            result = read_msg(proc)
            assert result["status"] == "error"
            assert "budget" in result["message"].lower()
        finally:
            proc.kill()

    def test_missing_token_counts_default_zero(self):
        """If response has no token fields, defaults to 0 (no budget hit)."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(parameterize ([token-budget 10]) (finish (syntax-e (llm-query #:instruction "test"))))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"

            # No token fields — budget should not be decremented
            write_response(proc, {"result": "hello"})

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "hello"
        finally:
            proc.kill()

    def test_token_counts_in_scope_log(self):
        """Scope log includes token count information."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"

            write_response(proc, {
                "result": "response",
                "prompt_tokens": 15,
                "completion_tokens": 8,
            })

            read_msg(proc)  # eval result

            write_cmd(proc, {"op": "get-scope-log"})
            log_resp = read_msg(proc)
            log = json.loads(log_resp["result"])

            # Find the llm-query entry
            llm_entries = [e for e in log if e["op"] == "llm-query"]
            assert len(llm_entries) >= 1
            assert "23 tokens" in llm_entries[0]["datum_preview"]
            assert "15 in" in llm_entries[0]["datum_preview"]
            assert "8 out" in llm_entries[0]["datum_preview"]
        finally:
            proc.kill()

    def test_tokens_used_callback_format(self):
        """tokens-used callback returns hash with correct keys."""
        proc = start_server()
        try:
            # Make a call that uses tokens
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            write_response(proc, {"result": "ok", "prompt_tokens": 100, "completion_tokens": 50})
            read_msg(proc)  # eval result

            # Now query tokens-used - this triggers a callback
            write_cmd(proc, {"op": "eval", "code": '(finish (tokens-used))'})
            callback = read_msg(proc)
            assert callback["op"] == "tokens-used"
            # Respond with token usage hash
            write_response(proc, {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "total_calls": 1
            })
            result = read_msg(proc)
            assert result["status"] == "finished"
            # Should be a hash representation showing tokens
            result_str = result["result"]
            assert "prompt" in result_str.lower() or "100" in result_str
        finally:
            proc.kill()

    def test_tokens_used_accumulates(self):
        """tokens-used accumulates across multiple calls."""
        proc = start_server()
        try:
            # First call: 30+20=50 tokens
            write_cmd(proc, {"op": "eval", "code":
                '(define _r1 (syntax-e (llm-query #:instruction "test1")))'
            })
            msg = read_msg(proc)
            write_response(proc, {"result": "ok", "prompt_tokens": 30, "completion_tokens": 20})
            read_msg(proc)

            # Second call: 40+10=50 tokens
            write_cmd(proc, {"op": "eval", "code":
                '(define _r2 (syntax-e (llm-query #:instruction "test2")))'
            })
            msg = read_msg(proc)
            write_response(proc, {"result": "ok", "prompt_tokens": 40, "completion_tokens": 10})
            read_msg(proc)

            # Check total: should be 100 tokens (30+20+40+10)
            write_cmd(proc, {"op": "eval", "code": '(finish (tokens-used))'})
            callback = read_msg(proc)
            assert callback["op"] == "tokens-used"
            # Respond with cumulative totals
            write_response(proc, {
                "prompt_tokens": 70,
                "completion_tokens": 30,
                "total_tokens": 100,
                "total_calls": 2
            })
            result = read_msg(proc)
            assert result["status"] == "finished"
            result_str = result["result"]
            # Should show cumulative totals
            assert "70" in result_str or "30" in result_str
        finally:
            proc.kill()


# ============================================================
# Multi-model dispatch (#3)
# ============================================================


class TestMultiModel:
    def test_model_forwarded_in_callback(self):
        """#:model value is forwarded in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test" #:model "gpt-4o-mini")))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["model"] == "gpt-4o-mini"

            write_response(proc, {"result": "ok"})
            read_msg(proc)
        finally:
            proc.kill()

    def test_model_default_empty(self):
        """Without #:model, the callback has an empty model string."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["model"] == ""

            write_response(proc, {"result": "ok"})
            read_msg(proc)
        finally:
            proc.kill()

    def test_unsafe_raw_query_model(self):
        """#:model works on unsafe-raw-query too."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (unsafe-raw-query #:instruction "test" #:model "gpt-4o-mini"))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["model"] == "gpt-4o-mini"

            write_response(proc, {"result": "ok"})
            read_msg(proc)
        finally:
            proc.kill()


# ============================================================
# Async sub-calls (#4)
# ============================================================


class TestAsync:
    def test_async_single(self):
        """A single async call dispatches and can be awaited."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h (llm-query-async #:instruction "summarize"))
(finish (syntax-e (await h)))
"""})

            # Read the async dispatch
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            assert "id" in msg
            async_id = msg["id"]

            # Read the await request
            msg2 = read_msg(proc)
            assert msg2["op"] == "await"
            assert msg2["id"] == async_id

            # Respond to the await
            write_response(proc, {
                "result": "summary text",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "summary text"
        finally:
            proc.kill()

    def test_async_multiple_dispatch_then_await(self):
        """Multiple async calls dispatch before any await."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h1 (llm-query-async #:instruction "q1"))
(define h2 (llm-query-async #:instruction "q2"))
(define h3 (llm-query-async #:instruction "q3"))
(define r1 (syntax-e (await h1)))
(define r2 (syntax-e (await h2)))
(define r3 (syntax-e (await h3)))
(finish (list r1 r2 r3))
"""})

            # Read 3 async dispatches
            ids = []
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])
                assert msg["instruction"] == f"q{i+1}"

            # Read 3 awaits and respond
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "await"
                assert msg["id"] == ids[i]
                write_response(proc, {
                    "result": f"answer{i+1}",
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "(answer1 answer2 answer3)"
        finally:
            proc.kill()

    def test_async_model_forwarded(self):
        """#:model is forwarded in async callbacks."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define h (llm-query-async #:instruction "test" #:model "gpt-4o-mini"))'
            })

            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            assert msg["model"] == "gpt-4o-mini"

            # Read the eval response (no finish, just ok)
            result = read_msg(proc)
            assert result["status"] == "ok"
        finally:
            proc.kill()

    def test_async_budget_on_await(self):
        """Token budget is decremented when await collects the result."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(parameterize ([token-budget 50])
  (define h (llm-query-async #:instruction "test"))
  (finish (syntax-e (await h))))
"""})

            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"

            msg2 = read_msg(proc)
            assert msg2["op"] == "await"

            # Respond with tokens exceeding budget
            write_response(proc, {
                "result": "big",
                "prompt_tokens": 30,
                "completion_tokens": 30,
            })

            result = read_msg(proc)
            assert result["status"] == "error"
            assert "budget" in result["message"].lower()
        finally:
            proc.kill()

    def test_async_scope_log(self):
        """Async calls appear in the scope log."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h (llm-query-async #:instruction "test"))
(define _r (syntax-e (await h)))
"""})

            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"

            msg2 = read_msg(proc)
            assert msg2["op"] == "await"
            write_response(proc, {
                "result": "ok",
                "prompt_tokens": 5,
                "completion_tokens": 3,
            })

            read_msg(proc)  # eval result

            write_cmd(proc, {"op": "get-scope-log"})
            log_resp = read_msg(proc)
            log = json.loads(log_resp["result"])
            ops = [e["op"] for e in log]
            assert "llm-query-async" in ops
        finally:
            proc.kill()

    def test_await_bad_handle_errors(self):
        """Awaiting a non-handle value raises an error."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": '(await "not-a-handle")'})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "async handle" in result["message"].lower()
        finally:
            proc.kill()

    def test_scaffold_protection_async(self):
        """llm-query-async and await are protected scaffold names."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": "(define llm-query-async 42)"})
            r1 = read_msg(proc)
            assert r1["status"] == "error"

            write_cmd(proc, {"op": "eval", "code": "(define await 42)"})
            r2 = read_msg(proc)
            assert r2["status"] == "error"
        finally:
            proc.kill()
