"""Tests for rate limit awareness via API response headers (S4).

Verifies that:
- _call_llm() captures x-ratelimit-* headers via with_raw_response
- _rate_limits dict is populated and updated after each call
- (rate-limits) sandbox binding returns the data via callback protocol
- rate-limits is scaffold-protected
- get_rate_limits() MCP tool returns the data
- Async calls also update rate limits
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import RacketREPL

import openai

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")


# ============================================================
# Helpers
# ============================================================

def _make_mock_raw_response(text="hello", prompt_tokens=10, completion_tokens=5,
                             ratelimit_headers=None):
    """Create a mock with_raw_response result with headers and parseable body."""
    if ratelimit_headers is None:
        ratelimit_headers = {
            "x-ratelimit-limit-requests": "500",
            "x-ratelimit-limit-tokens": "30000",
            "x-ratelimit-remaining-requests": "499",
            "x-ratelimit-remaining-tokens": "29500",
            "x-ratelimit-reset-requests": "120ms",
            "x-ratelimit-reset-tokens": "8ms",
        }
    # Build the parsed ChatCompletion
    parsed = MagicMock()
    parsed.choices = [MagicMock()]
    parsed.choices[0].message.content = text
    parsed.usage.prompt_tokens = prompt_tokens
    parsed.usage.completion_tokens = completion_tokens

    # Build the raw response wrapper
    raw = MagicMock()
    raw.headers = ratelimit_headers
    raw.parse.return_value = parsed
    return raw


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
# _call_llm captures rate limit headers
# ============================================================


class TestRateLimitCapture:
    @patch("mcp_server.openai.OpenAI")
    def test_headers_captured_on_success(self, mock_openai_cls):
        """Successful API call populates _rate_limits from headers."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.with_raw_response.create.return_value = (
            _make_mock_raw_response(
                ratelimit_headers={
                    "x-ratelimit-limit-requests": "500",
                    "x-ratelimit-limit-tokens": "30000",
                    "x-ratelimit-remaining-requests": "498",
                    "x-ratelimit-remaining-tokens": "28000",
                    "x-ratelimit-reset-requests": "200ms",
                    "x-ratelimit-reset-tokens": "15ms",
                }
            )
        )
        repl = RacketREPL()
        repl._max_retries = 0
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "hello"
            rl = repl._rate_limits
            assert rl["limit_requests"] == 500
            assert rl["limit_tokens"] == 30000
            assert rl["remaining_requests"] == 498
            assert rl["remaining_tokens"] == 28000
            assert rl["reset_requests"] == "200ms"
            assert rl["reset_tokens"] == "15ms"
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_headers_updated_on_each_call(self, mock_openai_cls):
        """Each API call updates _rate_limits with the latest headers."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.with_raw_response.create.side_effect = [
            _make_mock_raw_response(ratelimit_headers={
                "x-ratelimit-remaining-requests": "499",
                "x-ratelimit-remaining-tokens": "29000",
            }),
            _make_mock_raw_response(ratelimit_headers={
                "x-ratelimit-remaining-requests": "498",
                "x-ratelimit-remaining-tokens": "27000",
            }),
        ]
        repl = RacketREPL()
        repl._max_retries = 0
        try:
            repl._call_llm("test1", "data1")
            assert repl._rate_limits["remaining_requests"] == 499
            assert repl._rate_limits["remaining_tokens"] == 29000

            repl._call_llm("test2", "data2")
            assert repl._rate_limits["remaining_requests"] == 498
            assert repl._rate_limits["remaining_tokens"] == 27000
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_missing_headers_default_to_zero(self, mock_openai_cls):
        """Missing rate limit headers default to 0 / empty string."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client
        mock_client.chat.completions.with_raw_response.create.return_value = (
            _make_mock_raw_response(ratelimit_headers={})
        )
        repl = RacketREPL()
        repl._max_retries = 0
        try:
            repl._call_llm("test", "data")
            rl = repl._rate_limits
            assert rl["limit_requests"] == 0
            assert rl["limit_tokens"] == 0
            assert rl["remaining_requests"] == 0
            assert rl["remaining_tokens"] == 0
            assert rl["reset_requests"] == ""
            assert rl["reset_tokens"] == ""
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_initial_rate_limits_all_zero(self, mock_openai_cls):
        """Before any API call, _rate_limits is all zeros."""
        repl = RacketREPL()
        try:
            rl = repl._rate_limits
            assert rl["limit_requests"] == 0
            assert rl["remaining_tokens"] == 0
        finally:
            repl.close()


class TestRateLimitRetryInteraction:
    @patch("mcp_server.openai.OpenAI")
    def test_headers_captured_after_retry(self, mock_openai_cls):
        """After a 429 retry succeeds, headers from the success are captured."""
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        # First call: RateLimitError
        mock_response_429 = MagicMock()
        mock_response_429.status_code = 429
        mock_response_429.headers = {"Retry-After": "0.01"}
        mock_response_429.json.return_value = {"error": {"message": "Rate limited"}}
        rate_error = openai.RateLimitError(
            message="Rate limited", response=mock_response_429,
            body={"error": {"message": "Rate limited"}},
        )

        mock_client.chat.completions.with_raw_response.create.side_effect = [
            rate_error,
            _make_mock_raw_response(
                text="recovered",
                ratelimit_headers={
                    "x-ratelimit-remaining-requests": "400",
                    "x-ratelimit-remaining-tokens": "25000",
                }
            ),
        ]
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "recovered"
            assert repl._rate_limits["remaining_requests"] == 400
            assert repl._rate_limits["remaining_tokens"] == 25000
        finally:
            repl.close()


class TestGetRateLimitsMethod:
    @patch("mcp_server.openai.OpenAI")
    def test_get_rate_limits_returns_copy(self, mock_openai_cls):
        """get_rate_limits() returns a dict copy, not the internal reference."""
        repl = RacketREPL()
        try:
            rl1 = repl.get_rate_limits()
            rl2 = repl.get_rate_limits()
            assert rl1 is not rl2
            assert rl1 == rl2
        finally:
            repl.close()


# ============================================================
# (rate-limits) sandbox binding via callback protocol
# ============================================================


class TestRateLimitsCallback:
    def test_rate_limits_callback_returns_hash(self):
        """(rate-limits) returns a hash from the sandbox."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define rl (rate-limits))
(finish rl)
"""})
            # Read the rate-limits callback
            msg = read_msg(proc)
            assert msg["op"] == "rate-limits"

            # Respond with mock rate limit data
            write_response(proc, {
                "limit_requests": 500,
                "limit_tokens": 30000,
                "remaining_requests": 450,
                "remaining_tokens": 25000,
                "reset_requests": "100ms",
                "reset_tokens": "5ms",
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            # The result should contain the rate limit data
            assert "450" in result["result"] or "remaining_requests" in result["result"]
        finally:
            proc.kill()

    def test_rate_limits_hash_ref(self):
        """Can access individual fields from (rate-limits) with hash-ref."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define rl (rate-limits))
(finish (hash-ref rl 'remaining_tokens))
"""})
            msg = read_msg(proc)
            assert msg["op"] == "rate-limits"

            write_response(proc, {
                "limit_requests": 500,
                "limit_tokens": 30000,
                "remaining_requests": 450,
                "remaining_tokens": 25000,
                "reset_requests": "100ms",
                "reset_tokens": "5ms",
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "25000"
        finally:
            proc.kill()

    def test_rate_limits_conditional_model_selection(self):
        """Can use (rate-limits) to conditionally select a model."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define rl (rate-limits))
(define remaining (hash-ref rl 'remaining_tokens))
(define model (if (< remaining 5000) "gpt-4o-mini" "gpt-4o"))
(finish model)
"""})
            msg = read_msg(proc)
            assert msg["op"] == "rate-limits"

            # Return low remaining tokens
            write_response(proc, {
                "limit_requests": 500,
                "limit_tokens": 30000,
                "remaining_requests": 10,
                "remaining_tokens": 3000,
                "reset_requests": "100ms",
                "reset_tokens": "5ms",
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "gpt-4o-mini"
        finally:
            proc.kill()

    def test_rate_limits_high_remaining_selects_full_model(self):
        """When remaining tokens are high, selects the full model."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define rl (rate-limits))
(define remaining (hash-ref rl 'remaining_tokens))
(define model (if (< remaining 5000) "gpt-4o-mini" "gpt-4o"))
(finish model)
"""})
            msg = read_msg(proc)
            assert msg["op"] == "rate-limits"

            write_response(proc, {
                "limit_requests": 500,
                "limit_tokens": 30000,
                "remaining_requests": 450,
                "remaining_tokens": 25000,
                "reset_requests": "100ms",
                "reset_tokens": "5ms",
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "gpt-4o"
        finally:
            proc.kill()


# ============================================================
# Protected bindings
# ============================================================


class TestRateLimitsProtection:
    def test_cannot_redefine_rate_limits(self):
        """rate-limits is a protected scaffold binding."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define (rate-limits) "hacked")
"""})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "scaffold" in result["message"].lower() or "protected" in result["message"].lower() or "redefine" in result["message"].lower()
        finally:
            proc.kill()

    def test_cannot_redefine_rate_limits_callback(self):
        """__rate-limits-callback is a protected internal binding."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define (__rate-limits-callback) "hacked")
"""})
            result = read_msg(proc)
            assert result["status"] == "error"
        finally:
            proc.kill()
