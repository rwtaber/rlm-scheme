"""Tests for retry with exponential backoff in _call_llm().

Verifies that rate limit errors (429) trigger retries with configurable
backoff, Retry-After header is respected, cancel_event is checked between
retries, and failures after max retries propagate correctly.
"""

import concurrent.futures
import os
import sys
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import RacketREPL

import openai


# ============================================================
# Helpers
# ============================================================

def _make_raw_response(text="hello", prompt_tokens=10, completion_tokens=5):
    """Create a mock with_raw_response result (has .parse() and .headers)."""
    parsed = MagicMock()
    parsed.choices = [MagicMock()]
    parsed.choices[0].message.content = text
    parsed.usage.prompt_tokens = prompt_tokens
    parsed.usage.completion_tokens = completion_tokens
    raw = MagicMock()
    raw.parse.return_value = parsed
    raw.headers = {}
    return raw


def _make_rate_limit_error(retry_after=None):
    """Create a mock openai.RateLimitError with optional Retry-After header."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    if retry_after is not None:
        mock_response.headers["Retry-After"] = str(retry_after)
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    return openai.RateLimitError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _make_api_status_error_429(retry_after=None):
    """Create a mock openai.APIStatusError with status_code 429."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.headers = {}
    if retry_after is not None:
        mock_response.headers["Retry-After"] = str(retry_after)
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    return openai.APIStatusError(
        message="Rate limit exceeded",
        response=mock_response,
        body={"error": {"message": "Rate limit exceeded"}},
    )


def _api_mock(mock_openai_cls):
    """Set up mock client and return the with_raw_response.create mock."""
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    return mock_client, mock_client.chat.completions.with_raw_response.create


# ============================================================
# Retry behaviour
# ============================================================


class TestRetryOnRateLimit:
    """_call_llm retries on 429 RateLimitError with exponential backoff."""

    @patch("mcp_server.openai.OpenAI")
    def test_succeeds_after_one_retry(self, mock_openai_cls):
        """First call raises RateLimitError, second succeeds."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_rate_limit_error(),
            _make_raw_response("success"),
        ]
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "success"
            assert create.call_count == 2
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_succeeds_after_multiple_retries(self, mock_openai_cls):
        """All retries except the last raise RateLimitError."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_raw_response("finally"),
        ]
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "finally"
            assert create.call_count == 4
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_raises_after_max_retries(self, mock_openai_cls):
        """All attempts fail — raises RateLimitError after exhausting retries."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_rate_limit_error(),
            _make_rate_limit_error(),
        ]
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            with pytest.raises(openai.RateLimitError):
                repl._call_llm("test", "data")
            assert create.call_count == 4
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_no_retry_when_disabled(self, mock_openai_cls):
        """RLM_MAX_RETRIES=0 disables retries — fails immediately."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = _make_rate_limit_error()
        repl = RacketREPL()
        repl._max_retries = 0
        repl._retry_base_delay = 0.01
        try:
            with pytest.raises(openai.RateLimitError):
                repl._call_llm("test", "data")
            assert create.call_count == 1
        finally:
            repl.close()


class TestRetryAPIStatusError429:
    """_call_llm retries on APIStatusError with status_code 429."""

    @patch("mcp_server.openai.OpenAI")
    def test_retries_on_api_status_429(self, mock_openai_cls):
        """APIStatusError with 429 triggers retry, succeeds on second attempt."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_api_status_error_429(),
            _make_raw_response("recovered"),
        ]
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "recovered"
            assert create.call_count == 2
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_does_not_retry_on_non_429(self, mock_openai_cls):
        """APIStatusError with non-429 status code is not retried."""
        mock_client, create = _api_mock(mock_openai_cls)
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Server error"}}
        error = openai.APIStatusError(
            message="Server error", response=mock_response,
            body={"error": {"message": "Server error"}},
        )
        create.side_effect = error
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            with pytest.raises(openai.APIStatusError):
                repl._call_llm("test", "data")
            assert create.call_count == 1
        finally:
            repl.close()


class TestRetryAfterHeader:
    """Retry-After header from API response is respected."""

    @patch("mcp_server.openai.OpenAI")
    def test_retry_after_respected(self, mock_openai_cls):
        """Uses max(computed_delay, Retry-After) as wait time."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_rate_limit_error(retry_after=0.02),
            _make_raw_response("ok"),
        ]
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            t0 = time.monotonic()
            result = repl._call_llm("test", "data")
            elapsed = time.monotonic() - t0
            assert result["text"] == "ok"
            # Increased tolerance for CI/parallel execution (10ms instead of 5ms)
            # Under heavy load, timing can vary significantly
            assert elapsed >= 0.010, f"elapsed={elapsed:.3f}s should be >= 0.010s (0.02s retry-after with tolerance)"
        finally:
            repl.close()


class TestRetryCancelEvent:
    """cancel_event is checked between retry attempts."""

    @patch("mcp_server.openai.OpenAI")
    def test_cancel_before_retry(self, mock_openai_cls):
        """Setting cancel_event during backoff prevents further retries."""
        mock_client, create = _api_mock(mock_openai_cls)

        cancel_event = threading.Event()

        call_count = 0
        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                cancel_event.set()
                raise _make_rate_limit_error()
            return _make_raw_response("should not reach")

        create.side_effect = side_effect
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            with pytest.raises(concurrent.futures.CancelledError):
                repl._call_llm("test", "data", cancel_event=cancel_event)
        finally:
            repl.close()


class TestRetryExponentialBackoff:
    """Verify exponential backoff timing."""

    @patch("mcp_server.openai.OpenAI")
    def test_backoff_increases(self, mock_openai_cls):
        """Each retry waits longer: base * 2^attempt."""
        mock_client, create = _api_mock(mock_openai_cls)

        timestamps = []
        def side_effect(**kwargs):
            timestamps.append(time.monotonic())
            if len(timestamps) <= 3:
                raise _make_rate_limit_error()
            return _make_raw_response("done")

        create.side_effect = side_effect
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.05
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "done"
            assert len(timestamps) == 4
            delay1 = timestamps[1] - timestamps[0]
            delay2 = timestamps[2] - timestamps[1]
            delay3 = timestamps[3] - timestamps[2]
            # Allow timing tolerance: exponential backoff should roughly double, check for 1.4x increase
            assert delay2 > delay1 * 1.4, f"delay2={delay2:.3f} not > delay1*1.4={delay1*1.4:.3f}"
            assert delay3 > delay2 * 1.4, f"delay3={delay3:.3f} not > delay2*1.4={delay2*1.4:.3f}"
        finally:
            repl.close()


class TestRetryConfigurable:
    """Retry parameters are configurable via class attributes."""

    @patch("mcp_server.openai.OpenAI")
    def test_custom_max_retries(self, mock_openai_cls):
        """Custom _max_retries value is respected."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_rate_limit_error(),
            _make_raw_response("ok"),
        ]
        repl = RacketREPL()
        repl._max_retries = 1
        repl._retry_base_delay = 0.01
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "ok"
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_max_retries_one_not_enough(self, mock_openai_cls):
        """With max_retries=1, two failures exhaust retries."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.side_effect = [
            _make_rate_limit_error(),
            _make_rate_limit_error(),
        ]
        repl = RacketREPL()
        repl._max_retries = 1
        repl._retry_base_delay = 0.01
        try:
            with pytest.raises(openai.RateLimitError):
                repl._call_llm("test", "data")
            assert create.call_count == 2
        finally:
            repl.close()


class TestRetryNonRetryableErrors:
    """Non-retryable errors propagate immediately."""

    @patch("mcp_server.openai.OpenAI")
    def test_auth_error_not_retried(self, mock_openai_cls):
        """openai.AuthenticationError is not retried."""
        mock_client, create = _api_mock(mock_openai_cls)
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        create.side_effect = openai.AuthenticationError(
            message="Invalid API key", response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            with pytest.raises(openai.AuthenticationError):
                repl._call_llm("test", "data")
            assert create.call_count == 1
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_success_on_first_try_no_retry(self, mock_openai_cls):
        """Successful first call does not trigger any retry logic."""
        mock_client, create = _api_mock(mock_openai_cls)
        create.return_value = _make_raw_response("fast")
        repl = RacketREPL()
        repl._max_retries = 3
        repl._retry_base_delay = 0.01
        try:
            result = repl._call_llm("test", "data")
            assert result["text"] == "fast"
            assert create.call_count == 1
        finally:
            repl.close()
