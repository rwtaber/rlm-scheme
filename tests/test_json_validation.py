"""Tests for JSON mode validation (Priority 1 improvement #2).

Tests that #:json #t validates that 'json' appears in the instruction,
providing a clear error message to prevent cryptic API failures.
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


def eval_code(code: str) -> dict:
    """Send a single eval command."""
    responses = send_commands({"op": "eval", "code": code})
    assert len(responses) == 1
    return responses[0]


def start_server():
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


def read_msg(proc):
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Server died")
    return json.loads(line.strip())


def write_response(proc, resp):
    proc.stdin.write(json.dumps(resp) + "\n")
    proc.stdin.flush()


# ============================================================
# JSON mode validation tests
# ============================================================


class TestJsonModeValidation:
    """Test that #:json #t validates instruction contains 'json'."""

    def test_json_mode_without_json_in_instruction_sync(self):
        """Using #:json #t without 'json' in instruction gives clear error (sync)."""
        code = '(llm-query #:instruction "Return data" #:json #t)'
        r = eval_code(code)
        assert r["status"] == "error"
        assert "json" in r["message"].lower()
        assert "#:instruction" in r["message"]
        assert "OpenAI API requirement" in r["message"]

    def test_json_mode_without_json_in_instruction_async(self):
        """Using #:json #t without 'json' in instruction gives clear error (async)."""
        code = '(llm-query-async #:instruction "Return data" #:json #t)'
        r = eval_code(code)
        assert r["status"] == "error"
        assert "json" in r["message"].lower()
        assert "#:instruction" in r["message"]

    def test_json_mode_with_json_lowercase_passes(self):
        """#:json #t with 'json' in instruction (lowercase) passes validation."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "Return json" #:json #t)))'
            })
            msg = read_msg(proc)
            # Should reach the callback (not error before)
            assert msg["op"] == "llm-query"
            assert msg["json_mode"] is True

            write_response(proc, {"result": "{}", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_json_mode_with_json_uppercase_passes(self):
        """#:json #t with 'JSON' in instruction (uppercase) passes validation."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "Return JSON" #:json #t)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["json_mode"] is True

            write_response(proc, {"result": "{}", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_json_mode_with_json_mixed_case_passes(self):
        """#:json #t with 'Json' in instruction (mixed case) passes validation."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "Return a Json object" #:json #t)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["json_mode"] is True

            write_response(proc, {"result": "{}", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_json_mode_with_unsafe_raw_query(self):
        """Validation also applies to unsafe-raw-query."""
        code = '(unsafe-raw-query #:instruction "Return data" #:json #t)'
        r = eval_code(code)
        assert r["status"] == "error"
        assert "json" in r["message"].lower()

    def test_json_false_no_validation(self):
        """#:json #f or omitted doesn't require 'json' in instruction."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "Return data")))'
            })
            msg = read_msg(proc)
            # Should reach callback without error
            assert msg["op"] == "llm-query"
            assert msg["json_mode"] is False

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_empty_instruction_with_json_mode_fails(self):
        """Empty instruction with #:json #t fails validation."""
        code = '(llm-query #:instruction "" #:json #t)'
        r = eval_code(code)
        assert r["status"] == "error"
        assert "json" in r["message"].lower()
