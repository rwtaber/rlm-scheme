"""Tests for heartbeat mechanism and await timeout fix.

The heartbeat mechanism prevents false timeouts during long computations.
Racket sends {"op": "heartbeat"} to the MCP server, which resets the idle
timer and responds with {"ok": true}. This keeps the connection alive during
long fan-outs and user-initiated computations.

The await timeout fix ensures individual await calls use llm_timeout (not
the shorter Racket computation timeout), matching await-batch and await-any.
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
from mcp_server import RacketREPL


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
    """Write a JSON response (for callbacks)."""
    proc.stdin.write(json.dumps(resp) + "\n")
    proc.stdin.flush()


# ============================================================
# Heartbeat via Racket subprocess (direct protocol tests)
# ============================================================


class TestHeartbeatProtocol:
    def test_heartbeat_binding_exists(self):
        """(heartbeat) is available as a scaffold binding in the sandbox."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": "(heartbeat) (finish 42)"})
            # Racket sends heartbeat op to us
            msg = read_msg(proc)
            assert msg["op"] == "heartbeat"
            # Respond with ACK
            write_response(proc, {"ok": True})
            # Now get the final result
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "42"
        finally:
            proc.kill()

    def test_heartbeat_returns_void(self):
        """(heartbeat) returns void, doesn't interfere with computation."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define x 10)
(heartbeat)
(define y 20)
(finish (+ x y))
"""})
            # Handle heartbeat
            msg = read_msg(proc)
            assert msg["op"] == "heartbeat"
            write_response(proc, {"ok": True})
            # Get result
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "30"
        finally:
            proc.kill()

    def test_multiple_heartbeats(self):
        """Multiple heartbeats can be sent in sequence."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(heartbeat)
(heartbeat)
(heartbeat)
(finish "alive")
"""})
            for _ in range(3):
                msg = read_msg(proc)
                assert msg["op"] == "heartbeat"
                write_response(proc, {"ok": True})
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert result["result"] == "alive"
        finally:
            proc.kill()

    def test_heartbeat_is_protected(self):
        """heartbeat is a protected scaffold binding — cannot be redefined."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": '(define (heartbeat) "hacked")'})
            result = read_msg(proc)
            assert result["status"] == "error"
            assert "scaffold" in result["message"].lower() or "redefine" in result["message"].lower()
        finally:
            proc.kill()

    def test_heartbeat_interleaved_with_llm_query(self):
        """Heartbeats can be sent between LLM calls without confusion."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define r1 (llm-query-async #:instruction "test" #:data "a"))
(heartbeat)
(define r2 (llm-query-async #:instruction "test" #:data "b"))
(define results (await-all (list r1 r2)))
(finish results)
"""})
            # First: llm-query-async
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            id1 = msg["id"]

            # Second: heartbeat
            msg = read_msg(proc)
            assert msg["op"] == "heartbeat"
            write_response(proc, {"ok": True})

            # Third: llm-query-async
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            id2 = msg["id"]

            # Fourth: await-batch
            msg = read_msg(proc)
            assert msg["op"] == "await-batch"
            assert msg["ids"] == [id1, id2]
            write_response(proc, {"results": [
                {"result": "res-a", "prompt_tokens": 5, "completion_tokens": 3},
                {"result": "res-b", "prompt_tokens": 5, "completion_tokens": 3},
            ]})

            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "res-a" in result["result"]
            assert "res-b" in result["result"]
        finally:
            proc.kill()


# ============================================================
# Heartbeat in map-async (automatic heartbeats during fan-out)
# ============================================================


class TestMapAsyncHeartbeat:
    def _respond_to_pipelined_with_heartbeats(self, proc, init_count, total_count, prefix="result"):
        """Handle pipelined batch processing, expecting heartbeats at progress intervals."""
        ids = []
        launched = 0
        heartbeat_count = 0

        # Read initial window dispatches
        for i in range(init_count):
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async", f"Expected llm-query-async, got {msg['op']}"
            ids.append(msg["id"])
            launched += 1

        completed = 0
        while completed < total_count:
            msg = read_msg(proc)
            assert msg["op"] == "await-any", f"Expected await-any, got {msg['op']}"

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

            # After each completion, there may be a heartbeat (at progress intervals)
            # and/or a new llm-query-async dispatch. Read and handle them.
            if launched < total_count or completed == total_count:
                # Peek at next message - might be heartbeat or dispatch
                while True:
                    msg = read_msg(proc)
                    if msg["op"] == "heartbeat":
                        write_response(proc, {"ok": True})
                        heartbeat_count += 1
                    elif msg["op"] == "llm-query-async":
                        ids.append(msg["id"])
                        launched += 1
                        break
                    else:
                        # Must be the final result or next await-any
                        # Put it back by returning early
                        return msg, heartbeat_count

        return None, heartbeat_count

    def test_map_async_sends_heartbeats_during_pipelined(self):
        """map-async sends heartbeats during pipelined processing at progress intervals."""
        proc = start_server()
        try:
            # 15 items with max-concurrent 2 → pipelined processing
            # Progress interval = max(10, 15/10) = 10, so heartbeat at item 10 and 15
            items_code = " ".join(f'"{i}"' for i in range(15))
            write_cmd(proc, {"op": "eval", "code": f"""
(define items (list {items_code}))
(define results (map-async
  (lambda (x) (llm-query-async #:instruction "test" #:data x))
  items
  #:max-concurrent 2))
(finish (length results))
"""})

            ids = []
            launched = 0
            heartbeat_count = 0
            completed = 0

            # Read initial 2 dispatches
            for _ in range(2):
                msg = read_msg(proc)
                assert msg["op"] == "llm-query-async"
                ids.append(msg["id"])
                launched += 1

            # Process all items via await-any loop
            while completed < 15:
                msg = read_msg(proc)
                op = msg.get("op")

                if op == "heartbeat":
                    write_response(proc, {"ok": True})
                    heartbeat_count += 1
                    continue

                if op == "await-any":
                    completed_id = ids[0]
                    remaining_ids = ids[1:]
                    write_response(proc, {
                        "completed_id": completed_id,
                        "result": f"r-{completed}",
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "remaining_ids": remaining_ids
                    })
                    ids = remaining_ids
                    completed += 1
                    continue

                if op == "llm-query-async":
                    ids.append(msg["id"])
                    launched += 1
                    continue

                # Unexpected message
                raise AssertionError(f"Unexpected op: {op} in msg: {msg}")

            # Drain remaining messages (heartbeats) until final result
            while True:
                msg = read_msg(proc)
                op = msg.get("op")
                if op == "heartbeat":
                    write_response(proc, {"ok": True})
                    heartbeat_count += 1
                elif "status" in msg:
                    # This is the final result
                    break
                else:
                    raise AssertionError(f"Unexpected message: {msg}")

            assert msg["status"] == "finished"
            assert msg["result"] == "15"
            # At least 1 heartbeat should have been sent (at the 10-item mark)
            assert heartbeat_count >= 1, f"Expected at least 1 heartbeat, got {heartbeat_count}"
        finally:
            proc.kill()


# ============================================================
# Heartbeat via RacketREPL (MCP server integration)
# ============================================================


class TestHeartbeatMCPIntegration:
    def test_heartbeat_handled_by_send(self):
        """RacketREPL.send() handles heartbeat messages and continues."""
        repl = RacketREPL()
        try:
            # (heartbeat) sends a heartbeat op that send() must handle.
            # After handling it, send() should continue and return the finish result.
            resp = repl.send({"op": "eval", "code": "(heartbeat) (finish 42)"})
            assert resp["status"] == "finished"
            assert resp["result"] == "42"
        finally:
            repl.close()

    def test_heartbeat_resets_timeout(self):
        """Heartbeat prevents timeout even with a short timeout value.

        Uses a computation that interleaves heartbeats with sleeps.
        Each sleep is shorter than the timeout, but total time exceeds it.
        Without heartbeat resetting the timer, this would timeout.
        """
        repl = RacketREPL()
        try:
            # With timeout=4s, do 3 heartbeats with 2s sleeps between them.
            # Total: ~6s of computation, but never >4s idle between messages.
            resp = repl.send({"op": "eval", "code": """
(heartbeat)
(sleep 2)
(heartbeat)
(sleep 2)
(heartbeat)
(sleep 2)
(finish "survived")
"""}, timeout=4)
            assert resp["status"] == "finished"
            assert resp["result"] == "survived"
        finally:
            repl.close()

    def test_no_heartbeat_causes_timeout(self):
        """Without heartbeat, a long sleep exceeding timeout triggers TimeoutError."""
        repl = RacketREPL()
        try:
            # Sleep for 5s with a 3s timeout and no heartbeat → should timeout
            with pytest.raises(TimeoutError, match="timed out"):
                repl.send({"op": "eval", "code": "(sleep 5) (finish 42)"}, timeout=3)
        finally:
            repl.close()

    def test_heartbeat_preserves_state(self):
        """State defined before heartbeat persists after heartbeat."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": """
(define x 100)
(heartbeat)
(define y 200)
(heartbeat)
(finish (+ x y))
"""})
            assert resp["status"] == "finished"
            assert resp["result"] == "300"
        finally:
            repl.close()


# ============================================================
# Await timeout fix (uses llm_timeout, not timeout)
# ============================================================


class TestAwaitTimeoutFix:
    def test_await_uses_llm_timeout_not_computation_timeout(self):
        """Individual await should use llm_timeout (5x), not the shorter computation timeout.

        We verify this indirectly: with a short computation timeout but correct llm_timeout,
        a sync await should succeed even if the LLM call (simulated by the Racket subprocess
        protocol) takes longer than the computation timeout.
        """
        # This is tested at the code level - the fix changed line 3965 from
        # `timeout=timeout` to `timeout=llm_timeout`.
        # We verify the fix is in place by reading the source.
        import inspect
        source = inspect.getsource(RacketREPL.send)
        # The await handler should use llm_timeout for future.result()
        # Look for the pattern: in the "await" handler, future.result should use llm_timeout
        assert "future.result(timeout=llm_timeout)" in source, \
            "await handler should use llm_timeout, not timeout"

    def test_await_batch_uses_llm_timeout(self):
        """await-batch should also use llm_timeout (pre-existing, verify not regressed)."""
        import inspect
        source = inspect.getsource(RacketREPL.send)
        # await-batch uses concurrent.futures.wait with llm_timeout
        assert "timeout=llm_timeout," in source

    def test_await_any_uses_llm_timeout(self):
        """await-any should also use llm_timeout (pre-existing, verify not regressed)."""
        import inspect
        source = inspect.getsource(RacketREPL.send)
        # Count occurrences of llm_timeout in wait() calls — should be at least 2
        # (one for await-batch, one for await-any)
        count = source.count("timeout=llm_timeout")
        assert count >= 3, f"Expected at least 3 uses of llm_timeout (await, await-batch, await-any), found {count}"
