"""Tests for try/on-error error handling (Priority 1 improvement #6).

Tests that try/on-error gracefully handles failures in sub-model calls,
preventing single item failures from aborting entire batches.
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
# try/on-error tests
# ============================================================


class TestErrorHandling:
    """Test try/on-error error handling for robust pipelines."""

    def test_try_syntax_available(self):
        """Verify try syntax is available."""
        responses = send_commands(
            {"op": "eval", "code": """
(try
  (+ 1 2)
  on-error (lambda (err) "failed"))
(finish #t)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "finished"

    def test_try_catches_error(self):
        """try/on-error catches errors and invokes handler."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (error 'test "simulated error")
    on-error (lambda (err) "caught")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "finished"
        assert responses[0]["result"] == "caught"

    def test_try_passes_error_message(self):
        """Error handler receives the error message."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (error 'test "specific error message")
    on-error (lambda (err) err)))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "finished"
        assert "specific error message" in responses[0]["result"]

    def test_try_returns_normal_value_on_success(self):
        """try returns normal value when no error occurs."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (+ 10 20)
    on-error (lambda (err) "failed")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "finished"
        assert responses[0]["result"] == "30"

    def test_try_with_string_result(self):
        """try works with string results."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    "success value"
    on-error (lambda (err) "error value")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "success value"

    def test_try_with_division_by_zero(self):
        """try catches division by zero."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (/ 1 0)
    on-error (lambda (err) "div-by-zero")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "div-by-zero"

    def test_try_with_undefined_variable(self):
        """try catches undefined variable reference."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    nonexistent-variable
    on-error (lambda (err) "undefined")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "undefined"

    def test_try_in_map(self):
        """try/on-error works inside map for robust batch processing."""
        responses = send_commands(
            {"op": "eval", "code": """
(define items (list 1 0 2))  ; 0 will cause division error
(define results (map
  (lambda (x)
    (try
      (/ 10 x)
      on-error (lambda (err) -1)))
  items))
(finish results)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["status"] == "finished"
        result = responses[0]["result"]
        # Should have: (10 -1 5) - first and third succeed, second fails
        assert "10" in result
        assert "-1" in result
        assert "5" in result

    def test_try_returns_default_value(self):
        """try can return a default value on error."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (error 'test "fail")
    on-error (lambda (err) "DEFAULT")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "DEFAULT"

    def test_nested_try(self):
        """try can be nested for multi-level error handling."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (try
      (error 'inner "inner error")
      on-error (lambda (err) "inner caught"))
    on-error (lambda (err) "outer caught")))
(finish result)
"""}
        )
        assert len(responses) == 1
        # Inner try should catch it
        assert responses[0]["result"] == "inner caught"

    def test_try_with_complex_expression(self):
        """try works with complex expressions."""
        responses = send_commands(
            {"op": "eval", "code": """
(define result
  (try
    (let ([x 10]
          [y 20])
      (+ x y))
    on-error (lambda (err) "failed")))
(finish result)
"""}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "30"

    def test_try_documentation_example(self):
        """Verify documentation example syntax is valid."""
        # This doesn't make actual LLM calls, just verifies syntax
        responses = send_commands(
            {"op": "eval", "code": """
(define process-item (lambda (x) x))  ; Mock function
(define items (list "a" "b" "c"))
(define results (map
  (lambda (item)
    (try
      (process-item item)
      on-error (lambda (err) "FAILED")))
  items))
(finish (length results))
"""}
        )
        assert len(responses) == 1
        assert responses[0]["result"] == "3"
