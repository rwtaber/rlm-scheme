"""Tests for await-any: wait for first completed async handle.

Tests the await-any primitive which enables:
- Race patterns (try multiple strategies, use fastest)
- Progressive results (show results as they arrive)
- Timeout patterns (race against a timer)
- Dynamic scheduling
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


class TestAwaitAnyBasic:
    def test_await_any_returns_first_completed(self):
        """await-any returns the first completed result and remaining handles."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h1 (llm-query-async #:instruction "q1" #:data "a"))
(define h2 (llm-query-async #:instruction "q2" #:data "b"))
(call-with-values (lambda () (await-any (list h1 h2)))
  (lambda (first rest) (finish first)))
"""})
            # Read 2 dispatches
            ids = []
            for i in range(2):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Read await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"
            assert set(msg["ids"]) == set(ids)

            # Respond with first completion (ids[0])
            write_response(proc, {
                "completed_id": ids[0],
                "result": "first-result",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "remaining_ids": [ids[1]]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "first-result"
        finally:
            proc.kill()

    def test_await_any_with_empty_list_errors(self):
        """await-any on empty list raises an error."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(call-with-values (lambda () (await-any (list)))
  (lambda (first rest) (finish first)))
"""})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "cannot await-any on empty list" in result["message"]
        finally:
            proc.kill()

    def test_await_any_returns_remaining_handles(self):
        """await-any returns remaining handles that can be awaited later."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h1 (llm-query-async #:instruction "q1" #:data "a"))
(define h2 (llm-query-async #:instruction "q2" #:data "b"))
(define h3 (llm-query-async #:instruction "q3" #:data "c"))
(call-with-values (lambda () (await-any (list h1 h2 h3)))
  (lambda (first rest)
    (define remaining-results (await-all rest))
    (finish (string-append first " " (car remaining-results) " " (cadr remaining-results)))))
"""})
            # Read 3 dispatches
            ids = []
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Read await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"
            assert set(msg["ids"]) == set(ids)

            # First completion: ids[1]
            write_response(proc, {
                "completed_id": ids[1],
                "result": "second",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "remaining_ids": [ids[0], ids[2]]
            })

            # Read await-batch for remaining 2
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert set(msg["ids"]) == {ids[0], ids[2]}

            write_response(proc, {
                "results": [
                    {"result": "first", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "third", "prompt_tokens": 5, "completion_tokens": 3}
                ]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            # Result should contain all three in some order
            assert "second" in result["result"]
            assert "first" in result["result"]
            assert "third" in result["result"]
        finally:
            proc.kill()


class TestAwaitAnyRacePattern:
    def test_race_multiple_strategies(self):
        """Race pattern: try multiple approaches, use fastest."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
;; Try 3 different strategies, use whichever finishes first
(define algebraic (llm-query-async #:instruction "solve algebraically" #:data "x^2=4"))
(define numeric (llm-query-async #:instruction "solve numerically" #:data "x^2=4"))
(define graphical (llm-query-async #:instruction "solve graphically" #:data "x^2=4"))
(call-with-values (lambda () (await-any (list algebraic numeric graphical)))
  (lambda (fastest rest)
    (finish (string-append "Fastest approach: " fastest))))
"""})
            # Read 3 dispatches
            ids = []
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Read await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"

            # Numeric finishes first
            write_response(proc, {
                "completed_id": ids[1],
                "result": "x=2 or x=-2",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "remaining_ids": [ids[0], ids[2]]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "x=2 or x=-2" in result["result"]
        finally:
            proc.kill()


class TestAwaitAnyProgressiveResults:
    def test_progressive_results_pattern(self):
        """Progressive results: process items as they complete."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
;; Start 3 analyses
(define handles (list
  (llm-query-async #:instruction "analyze" #:data "doc1")
  (llm-query-async #:instruction "analyze" #:data "doc2")
  (llm-query-async #:instruction "analyze" #:data "doc3")))

;; Process as they arrive
(call-with-values (lambda () (await-any handles))
  (lambda (first rest1)
    (define result1 (string-append "Result 1: " first))
    (call-with-values (lambda () (await-any rest1))
      (lambda (second rest2)
        (define result2 (string-append " Result 2: " second))
        (call-with-values (lambda () (await-any rest2))
          (lambda (third rest3)
            (define result3 (string-append " Result 3: " third))
            (finish (string-append result1 result2 result3))))))))
"""})
            # Read 3 dispatches
            ids = []
            for i in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # First await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"
            write_response(proc, {
                "completed_id": ids[2],
                "result": "analysis-3",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "remaining_ids": [ids[0], ids[1]]
            })

            # Second await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"
            write_response(proc, {
                "completed_id": ids[0],
                "result": "analysis-1",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "remaining_ids": [ids[1]]
            })

            # Third await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"
            write_response(proc, {
                "completed_id": ids[1],
                "result": "analysis-2",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "remaining_ids": []
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "analysis-1" in result["result"]
            assert "analysis-2" in result["result"]
            assert "analysis-3" in result["result"]
        finally:
            proc.kill()


class TestAwaitAnySingleHandle:
    def test_await_any_with_single_handle(self):
        """await-any works with a single handle (edge case)."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define h (llm-query-async #:instruction "test"))
(call-with-values (lambda () (await-any (list h)))
  (lambda (result rest) (finish result)))
"""})
            # Read dispatch
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            handle_id = msg["id"]

            # Read await-any
            msg = read_msg(proc)
            assert msg["op"] == "await-any"
            assert msg["ids"] == [handle_id]

            write_response(proc, {
                "completed_id": handle_id,
                "result": "single-result",
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "remaining_ids": []
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "single-result"
        finally:
            proc.kill()
