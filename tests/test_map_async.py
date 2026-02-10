"""Tests for map-async parallel fan-out with optional concurrency limit.

Tests the Racket sandbox's map-async function using the interactive callback
protocol. Verifies unbatched and batched (max-concurrent) execution, empty
lists, and scaffold protection. No real API calls.
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


def respond_to_async_batch(proc, n, prefix="result"):
    """Handle n async dispatches followed by await-batch with mock responses."""
    ids = []
    for i in range(n):
        msg = read_msg(proc)
        assert msg["op"] == "llm-query-async", f"Expected llm-query-async, got {msg['op']}"
        ids.append(msg["id"])

    # Now expect a single await-batch message with all IDs
    msg = read_msg(proc)
    assert msg["op"] == "await-batch", f"Expected await-batch, got {msg['op']}"
    assert msg["ids"] == ids, f"Expected IDs {ids}, got {msg['ids']}"

    # Respond with all results at once
    results = []
    for i in range(n):
        results.append({
            "result": f"{prefix}-{i}",
            "prompt_tokens": 5,
            "completion_tokens": 3,
        })
    write_response(proc, {"results": results})
    return ids


def read_msg_skip_heartbeats(proc, timeout=10):
    """Read one JSON message, transparently handling heartbeat messages."""
    while True:
        msg = read_msg(proc, timeout)
        if msg.get("op") == "heartbeat":
            write_response(proc, {"ok": True})
            continue
        return msg


def respond_to_pipelined_batch(proc, init_count, total_count, prefix="result"):
    """Handle pipelined batching: initial dispatches, then await-any loop with incremental launches.

    Args:
        proc: The server process
        init_count: Number of items in initial window
        total_count: Total number of items to process
        prefix: Prefix for result strings
    """
    ids = []
    launched = 0

    # Read initial window dispatches
    for i in range(init_count):
        msg = read_msg_skip_heartbeats(proc)
        assert msg["op"] == "llm-query-async", f"Expected llm-query-async, got {msg['op']}"
        ids.append(msg["id"])
        launched += 1

    # Process pipelined: await-any, respond, expect new dispatch (if more items remain)
    completed = 0
    while completed < total_count:
        # Expect await-any with all currently active IDs
        msg = read_msg_skip_heartbeats(proc)
        assert msg["op"] == "await-any", f"Expected await-any, got {msg['op']}"
        assert set(msg["ids"]) == set(ids), f"Expected IDs {set(ids)}, got {set(msg['ids'])}"

        # Respond with first ID completing
        completed_id = ids[0]
        remaining_ids = ids[1:]
        write_response(proc, {
            "completed_id": completed_id,
            "result": f"{prefix}-{completed}",
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "remaining_ids": remaining_ids
        })
        ids = remaining_ids
        completed += 1

        # If more items remain to be launched, expect a dispatch
        if launched < total_count:
            msg = read_msg_skip_heartbeats(proc)
            assert msg["op"] == "llm-query-async", f"Expected llm-query-async at iteration {completed}, got {msg['op']}"
            ids.append(msg["id"])
            launched += 1


# ============================================================
# map-async without concurrency limit
# ============================================================


class TestMapAsync:
    def test_map_async_basic(self):
        """map-async applies fn to each item and returns results."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define items (list "a" "b" "c"))
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "process" #:data x))
  items))
(finish results)
"""})
            respond_to_async_batch(proc, 3)

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "result-0" in result["result"]
            assert "result-1" in result["result"]
            assert "result-2" in result["result"]
        finally:
            proc.kill()

    def test_map_async_single_item(self):
        """map-async with a single-element list works correctly."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "test" #:data x))
  (list "only")))
(finish results)
"""})
            respond_to_async_batch(proc, 1)

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "result-0" in result["result"]
        finally:
            proc.kill()

    def test_map_async_empty_list(self):
        """map-async with empty list returns empty list."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "test" #:data x))
  '()))
(finish results)
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "()"
        finally:
            proc.kill()

    def test_map_async_returns_strings(self):
        """map-async returns unwrapped strings that can be used directly."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "echo" #:data x))
  (list "hello" "world")))
(finish (string-append (car results) " " (cadr results)))
"""})
            ids = []
            for i in range(2):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])

            # Expect await-batch with both IDs
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert msg["ids"] == ids
            write_response(proc, {
                "results": [
                    {"result": "hello", "prompt_tokens": 5, "completion_tokens": 3},
                    {"result": "world", "prompt_tokens": 5, "completion_tokens": 3}
                ]
            })

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "hello world"
        finally:
            proc.kill()


# ============================================================
# map-async with #:max-concurrent
# ============================================================


class TestMapAsyncBatched:
    def test_max_concurrent_batches(self):
        """With #:max-concurrent 2, 5 items are processed with pipelined batching."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define items (list "a" "b" "c" "d" "e"))
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "process" #:data x))
  items
  #:max-concurrent 2))
(finish results)
"""})
            # Pipelined: initial window of 2, then process all 5
            respond_to_pipelined_batch(proc, init_count=2, total_count=5, prefix="result")

            result = read_msg_skip_heartbeats(proc)
            assert result["status"] == "finished"
            # Results should be in order
            for i in range(5):
                assert f"result-{i}" in result["result"]
        finally:
            proc.kill()

    def test_max_concurrent_equal_to_items(self):
        """When max-concurrent >= items, all launch at once (no batching)."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "test" #:data x))
  (list "a" "b" "c")
  #:max-concurrent 5))
(finish results)
"""})
            respond_to_async_batch(proc, 3)

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "result-0" in result["result"]
            assert "result-1" in result["result"]
            assert "result-2" in result["result"]
        finally:
            proc.kill()

    def test_max_concurrent_one(self):
        """max-concurrent 1 processes items sequentially with pipelining."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "seq" #:data x))
  (list "a" "b" "c")
  #:max-concurrent 1))
(finish results)
"""})
            # Pipelined with window of 1 (completely sequential)
            respond_to_pipelined_batch(proc, init_count=1, total_count=3, prefix="result")

            result = read_msg_skip_heartbeats(proc)
            assert result["status"] == "finished"
            for i in range(3):
                assert f"result-{i}" in result["result"]
        finally:
            proc.kill()

    def test_max_concurrent_preserves_order(self):
        """Results are returned in the same order as the input items."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define items (list "first" "second" "third" "fourth"))
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "order" #:data x))
  items
  #:max-concurrent 2))
(finish (car results))
"""})
            # Pipelined batching with window of 2
            respond_to_pipelined_batch(proc, init_count=2, total_count=4, prefix="ordered")

            result = read_msg_skip_heartbeats(proc)
            assert result["status"] == "finished"
            # First result should be "ordered-0" (preserved order)
            assert result["result"] == "ordered-0"
        finally:
            proc.kill()


# ============================================================
# Protected bindings
# ============================================================


class TestMapAsyncDefaultConcurrency:
    def test_default_max_concurrent_is_20(self):
        """Without #:max-concurrent, defaults to 20 with pipelined batching."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define items (list "a" "b" "c" "d" "e" "f" "g" "h" "i" "j" "k" "l" "m" "n" "o" "p" "q" "r" "s" "t" "u" "v" "w" "x" "y"))
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "process" #:data x))
  items))
(finish results)
"""})
            # With 25 items and default max-concurrent=20, pipelined batching with initial window of 20
            respond_to_pipelined_batch(proc, init_count=20, total_count=25, prefix="result")

            result = read_msg_skip_heartbeats(proc)
            assert result["status"] == "finished"
            # Verify all 25 results are present
            for i in range(25):
                assert f"result-{i}" in result["result"]
        finally:
            proc.kill()

    def test_default_batching_with_small_list(self):
        """With fewer than 10 items and no max-concurrent, launches all at once."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define items (list "a" "b" "c" "d" "e"))
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "test" #:data x))
  items))
(finish results)
"""})
            # With 5 items and default max-concurrent=10, all launch at once
            respond_to_async_batch(proc, 5)

            result = read_msg(proc)
            assert result["status"] == "finished"
            for i in range(5):
                assert f"result-{i}" in result["result"]
        finally:
            proc.kill()


class TestMapAsyncProtection:
    def test_cannot_redefine_map_async(self):
        """map-async is a protected scaffold binding."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define (map-async fn items) "hacked")
"""})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "scaffold" in result["message"].lower() or "protected" in result["message"].lower() or "redefine" in result["message"].lower()
        finally:
            proc.kill()
