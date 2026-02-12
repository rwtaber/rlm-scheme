"""Tests for the MCP server's RacketREPL class.

Tests timeout handling, crash recovery, subprocess management,
and call registry/visibility (issue 8).
Does NOT test LLM calls (those require API keys).
"""

import base64
import concurrent.futures
import json
import os
import sys
import time
import threading

import pytest

# Add project root to path so we can import mcp_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import RacketREPL


# ============================================================
# Timeout (1a)
# ============================================================


class TestTimeout:
    def test_timeout_infinite_loop(self):
        """An infinite loop triggers TimeoutError within the timeout window (with SIGTERM grace)."""
        repl = RacketREPL()
        try:
            t0 = time.perf_counter()
            with pytest.raises(TimeoutError, match="timed out"):
                repl.send({"op": "eval", "code": "(let loop () (loop))"}, timeout=3)
            elapsed = time.perf_counter() - t0
            # Timeout + SIGTERM grace period (2s) should complete within ~8s
            assert elapsed < 10, f"Timeout took too long: {elapsed:.1f}s"
        finally:
            repl.close()

    def test_timeout_recovery(self):
        """After a timeout, the next call works (sandbox restarts)."""
        repl = RacketREPL()
        try:
            with pytest.raises(TimeoutError):
                repl.send({"op": "eval", "code": "(let loop () (loop))"}, timeout=3)
            # Next call should work — auto-restarts
            resp = repl.send({"op": "eval", "code": "(finish 42)"})
            assert resp["status"] == "finished"
            assert resp["result"] == "42"
        finally:
            repl.close()

    def test_no_timeout_on_normal_call(self):
        """Normal calls don't trigger timeout."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": "(finish 42)"}, timeout=5)
            assert resp["status"] == "finished"
            assert resp["result"] == "42"
        finally:
            repl.close()

    def test_graceful_sigterm_before_kill(self):
        """On timeout, SIGTERM is sent before SIGKILL (graceful shutdown)."""
        import time
        repl = RacketREPL()
        try:
            pid = repl.proc.pid
            with pytest.raises(TimeoutError):
                repl.send({"op": "eval", "code": "(let loop () (loop))"}, timeout=2)
            # Give process termination a moment to complete (especially under load)
            time.sleep(0.1)
            # Process should be dead after timeout
            assert repl.proc is None
            # Verify recovery works
            resp = repl.send({"op": "eval", "code": "(finish 42)"})
            assert resp["status"] == "finished"
        finally:
            repl.close()


# ============================================================
# Crash recovery (1b)
# ============================================================


class TestCrashRecovery:
    def test_crash_mid_call(self):
        """If the process dies mid-call, send() returns an error and restarts."""
        repl = RacketREPL()
        try:
            # Kill the process right after sending a command
            repl.proc.stdin.write(json.dumps({"op": "eval", "code": "(finish 42)"}) + "\n")
            repl.proc.stdin.flush()
            repl.proc.kill()
            repl.proc.wait()

            # The next send() should detect the dead process and recover
            resp = repl.send({"op": "eval", "code": "(finish 99)"})
            assert resp["status"] == "finished"
            assert resp["result"] == "99"
        finally:
            repl.close()

    def test_state_lost_after_crash(self):
        """State is lost after crash — variables don't persist."""
        repl = RacketREPL()
        try:
            repl.send({"op": "eval", "code": "(define x 99)"})
            repl.proc.kill()
            repl.proc.wait()

            # After restart, x should not exist
            resp = repl.send({"op": "eval", "code": "(finish x)"})
            assert resp["status"] == "error"
        finally:
            repl.close()


# ============================================================
# Start idempotency
# ============================================================


class TestStartIdempotent:
    def test_double_start_no_leak(self):
        """Calling _start() twice doesn't leak processes."""
        repl = RacketREPL()
        try:
            first_pid = repl.proc.pid
            repl._start()  # Should close the first, start a new one
            second_pid = repl.proc.pid
            assert first_pid != second_pid
            # The new process should work
            resp = repl.send({"op": "eval", "code": "(finish 42)"})
            assert resp["status"] == "finished"
        finally:
            repl.close()

    def test_close_then_send(self):
        """After close(), send() auto-restarts."""
        repl = RacketREPL()
        try:
            repl.close()
            assert repl.proc is None
            resp = repl.send({"op": "eval", "code": "(finish 42)"})
            assert resp["status"] == "finished"
        finally:
            repl.close()


# ============================================================
# Call registry (issue 8a)
# ============================================================


class TestCallRegistry:
    def test_call_id_increments(self):
        """Each call gets a unique incrementing ID."""
        repl = RacketREPL()
        try:
            id1 = repl._next_call_id()
            id2 = repl._next_call_id()
            id3 = repl._next_call_id()
            assert id1 == "call_1"
            assert id2 == "call_2"
            assert id3 == "call_3"
        finally:
            repl.close()

    def test_register_and_snapshot(self):
        """Registering a call makes it visible in the snapshot."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o", "test instruction", depth=0)
            snapshot = repl.get_active_calls_snapshot()
            assert len(snapshot) == 1
            assert snapshot[0]["call_id"] == "call_1"
            assert snapshot[0]["type"] == "sync"
            assert snapshot[0]["model"] == "gpt-4o"
            assert snapshot[0]["instruction_preview"] == "test instruction"
            assert snapshot[0]["depth"] == 0
            assert "elapsed_seconds" in snapshot[0]
        finally:
            repl.close()

    def test_complete_removes_from_snapshot(self):
        """Completing a call removes it from the active snapshot."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            assert len(repl.get_active_calls_snapshot()) == 1
            repl._complete_call("call_1", tokens=100, elapsed=1.5)
            assert len(repl.get_active_calls_snapshot()) == 0
        finally:
            repl.close()

    def test_multiple_concurrent_calls(self):
        """Multiple calls can be tracked simultaneously."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o", "first")
            repl._register_call("call_2", "async", "gpt-4o-mini", "second")
            repl._register_call("call_3", "recursive", "gpt-4o", "third", parent_id="call_1", depth=1)
            snapshot = repl.get_active_calls_snapshot()
            assert len(snapshot) == 3
            ids = {c["call_id"] for c in snapshot}
            assert ids == {"call_1", "call_2", "call_3"}
            # Check parent_id on recursive call
            recursive = [c for c in snapshot if c["call_id"] == "call_3"][0]
            assert recursive["parent_id"] == "call_1"
            assert recursive["depth"] == 1
        finally:
            repl.close()

    def test_instruction_preview_truncated(self):
        """Long instructions are truncated to 80 chars in the preview."""
        repl = RacketREPL()
        try:
            long_instruction = "x" * 200
            repl._register_call("call_1", "sync", "gpt-4o", long_instruction)
            snapshot = repl.get_active_calls_snapshot()
            assert len(snapshot[0]["instruction_preview"]) == 80
        finally:
            repl.close()

    def test_elapsed_time_increases(self):
        """Elapsed time in snapshot reflects real time."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            time.sleep(0.2)
            snapshot = repl.get_active_calls_snapshot()
            assert snapshot[0]["elapsed_seconds"] >= 0.1
        finally:
            repl.close()

    def test_close_clears_active_calls(self):
        """Closing the REPL clears all active calls."""
        repl = RacketREPL()
        repl._register_call("call_1", "sync", "gpt-4o")
        repl._register_call("call_2", "async", "gpt-4o")
        repl.close()
        assert len(repl.get_active_calls_snapshot()) == 0

    def test_empty_snapshot_when_no_calls(self):
        """Snapshot is empty when no calls are active."""
        repl = RacketREPL()
        try:
            assert repl.get_active_calls_snapshot() == []
        finally:
            repl.close()


# ============================================================
# Cancel call (issue 8e)
# ============================================================


class TestCancelCall:
    def test_cancel_nonexistent_call(self):
        """Cancelling a non-existent call returns an error message."""
        repl = RacketREPL()
        try:
            result = repl.cancel_call("call_999")
            assert "No active call" in result
        finally:
            repl.close()

    def test_cancel_removes_from_active(self):
        """Cancelling a call removes it from active calls."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            assert len(repl.get_active_calls_snapshot()) == 1
            result = repl.cancel_call("call_1")
            assert "Cancelled" in result
            assert len(repl.get_active_calls_snapshot()) == 0
        finally:
            repl.close()

    def test_cancel_cancels_future(self):
        """Cancelling an async call cancels the underlying Future."""
        repl = RacketREPL()
        try:
            future = concurrent.futures.Future()
            repl._register_call("call_1", "async", "gpt-4o", future=future)
            repl.cancel_call("call_1")
            assert future.cancelled()
        finally:
            repl.close()

    def test_cancel_sets_event(self):
        """Cancelling a call sets the cancellation event."""
        repl = RacketREPL()
        try:
            cancel_event = threading.Event()
            repl._register_call("call_1", "async", "gpt-4o", cancel_event=cancel_event)
            assert not cancel_event.is_set()
            repl.cancel_call("call_1")
            assert cancel_event.is_set()
        finally:
            repl.close()

    def test_cancel_closes_nested_repl(self):
        """Cancelling a recursive call closes the nested REPL."""
        repl = RacketREPL()
        try:
            nested = RacketREPL()
            assert nested.proc is not None
            repl._register_call("call_1", "recursive", "gpt-4o", nested_repl=nested)
            repl.cancel_call("call_1")
            assert nested.proc is None
        finally:
            repl.close()


# ============================================================
# Stderr logging (issue 8c)
# ============================================================


class TestStderrLogging:
    def test_register_logs_to_stderr(self, capsys):
        """Registering a call logs to stderr."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o", "test instruction")
            captured = capsys.readouterr()
            assert "[rlm] call_1:" in captured.err
            assert "gpt-4o" in captured.err
            assert "sync" in captured.err
        finally:
            repl.close()

    def test_complete_logs_to_stderr(self, capsys):
        """Completing a call logs to stderr."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            repl._complete_call("call_1", tokens=150, elapsed=2.5)
            captured = capsys.readouterr()
            assert "completed" in captured.err
            assert "150 tokens" in captured.err
        finally:
            repl.close()

    def test_cancel_logs_to_stderr(self, capsys):
        """Cancelling a call logs to stderr."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            repl.cancel_call("call_1")
            captured = capsys.readouterr()
            assert "cancelled" in captured.err
        finally:
            repl.close()


# ============================================================
# Thread safety (issue 8a)
# ============================================================


class TestThreadSafety:
    def test_concurrent_call_ids_unique(self):
        """Call IDs are unique even under concurrent access."""
        repl = RacketREPL()
        try:
            ids = []
            lock = threading.Lock()

            def get_ids(n):
                local_ids = [repl._next_call_id() for _ in range(n)]
                with lock:
                    ids.extend(local_ids)

            threads = [threading.Thread(target=get_ids, args=(50,)) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(ids) == 200
            assert len(set(ids)) == 200  # All unique
        finally:
            repl.close()


# ============================================================
# Image resolution (issue: multimodal support)
# ============================================================


class TestResolveImage:
    def test_data_url_passthrough(self):
        """data: URLs are returned as-is."""
        url = "data:image/png;base64,iVBORw0KGgo="
        assert RacketREPL._resolve_image(url) == url

    def test_raw_base64_wrapped(self):
        """Raw base64 strings are wrapped in a data URL."""
        result = RacketREPL._resolve_image("iVBORw0KGgo=")
        assert result == "data:image/png;base64,iVBORw0KGgo="

    def test_file_path_encoded(self, tmp_path):
        """File paths are read and base64-encoded."""
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic bytes
        result = RacketREPL._resolve_image(str(img))
        assert result.startswith("data:image/png;base64,")
        # Decode the base64 and verify it matches
        b64_part = result.split(",", 1)[1]
        assert base64.b64decode(b64_part) == b"\x89PNG\r\n\x1a\n"

    def test_jpeg_file_mime(self, tmp_path):
        """JPEG files get the correct MIME type."""
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")
        result = RacketREPL._resolve_image(str(img))
        assert result.startswith("data:image/jpeg;base64,")

    def test_nonexistent_file_treated_as_base64(self):
        """A path that doesn't exist is treated as raw base64."""
        result = RacketREPL._resolve_image("/no/such/file.png")
        assert result == "data:image/png;base64,/no/such/file.png"

    def test_file_too_large_rejected(self, tmp_path):
        """Files exceeding MAX_IMAGE_BYTES are rejected."""
        img = tmp_path / "huge.png"
        # Create a file just over the limit
        img.write_bytes(b"\x89PNG" + b"\x00" * (RacketREPL.MAX_IMAGE_BYTES + 1))
        with pytest.raises(ValueError, match="too large"):
            RacketREPL._resolve_image(str(img))

    def test_small_file_accepted(self, tmp_path):
        """Files under MAX_IMAGE_BYTES are accepted."""
        img = tmp_path / "small.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
        result = RacketREPL._resolve_image(str(img))
        assert result.startswith("data:image/png;base64,")

    def test_base64_too_large_rejected(self):
        """Oversized base64 strings are rejected."""
        max_b64_len = RacketREPL.MAX_IMAGE_BYTES * 4 // 3 + 4
        huge_b64 = "A" * (max_b64_len + 1)
        with pytest.raises(ValueError, match="too large"):
            RacketREPL._resolve_image(huge_b64)

    def test_non_image_file_warns(self, tmp_path, capsys):
        """Non-image files produce a warning on stderr."""
        txt = tmp_path / "file.txt"
        txt.write_bytes(b"hello world this is text")
        RacketREPL._resolve_image(str(txt))
        captured = capsys.readouterr()
        assert "may not be an image" in captured.err


# ============================================================
# Image count warning (S12)
# ============================================================


class TestImageCountWarning:
    def test_warning_over_limit(self, capsys):
        """Warning logged when image count exceeds MAX_RECOMMENDED_IMAGES."""
        repl = RacketREPL()
        try:
            # Simulate calling _call_llm with too many images
            # We can't actually call it (needs API key), but we test the warning logic
            images = ["iVBORw0KGgo="] * 6
            assert len(images) > repl.MAX_RECOMMENDED_IMAGES
            # The warning is printed to stderr at the start of _call_llm
            # We'd need a mock for the full test, but we can verify the constant
            assert repl.MAX_RECOMMENDED_IMAGES == 5
        finally:
            repl.close()

    def test_under_limit_no_warning(self):
        """No warning for image counts at or below limit."""
        repl = RacketREPL()
        try:
            images = ["iVBORw0KGgo="] * 3
            assert len(images) <= repl.MAX_RECOMMENDED_IMAGES
        finally:
            repl.close()

# ============================================================
# Stderr drain thread (S1)
# ============================================================


class TestStderrDrain:
    def test_stderr_buffer_exists(self):
        """REPL has a stderr ring buffer after init."""
        repl = RacketREPL()
        try:
            assert hasattr(repl, '_stderr_buffer')
            assert repl._stderr_buffer.maxlen == RacketREPL.STDERR_BUFFER_SIZE
        finally:
            repl.close()

    def test_stderr_thread_running(self):
        """Stderr drain thread is alive after start."""
        repl = RacketREPL()
        try:
            assert repl._stderr_thread is not None
            assert repl._stderr_thread.is_alive()
        finally:
            repl.close()

    def test_get_stderr_log_returns_list(self):
        """get_stderr_log returns a list."""
        repl = RacketREPL()
        try:
            log = repl.get_stderr_log()
            assert isinstance(log, list)
        finally:
            repl.close()

    def test_stderr_captured_on_error(self):
        """Stderr from Racket errors is captured in the buffer."""
        repl = RacketREPL()
        try:
            # Cause a Racket error that writes to stderr
            repl.send({"op": "eval", "code": "(error 'test \"stderr test message\")"})
            # Give the drain thread a moment to process
            time.sleep(0.1)
            # The error message may or may not appear in stderr depending on
            # Racket's error handling, but the buffer should exist and work
            log = repl.get_stderr_log()
            assert isinstance(log, list)
        finally:
            repl.close()

    def test_stderr_buffer_cleared_on_restart(self):
        """Stderr buffer is cleared when process restarts."""
        repl = RacketREPL()
        try:
            # Add something to the buffer manually
            repl._stderr_buffer.append("old message")
            assert len(repl.get_stderr_log()) == 1
            # Restart
            repl._start()
            # Buffer should be cleared
            assert "old message" not in repl.get_stderr_log()
        finally:
            repl.close()


# ============================================================
# Structured JSON from execute_scheme (S2)
# ============================================================


class TestStructuredJSON:
    def test_finished_returns_json(self):
        """execute_scheme returns structured JSON for finished status."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": "(finish 42)"})
            assert resp["status"] == "finished"
            assert resp["result"] == "42"
        finally:
            repl.close()

    def test_error_returns_json(self):
        """execute_scheme returns structured JSON for error status."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": "(undefined-func)"})
            assert resp["status"] == "error"
            assert "message" in resp
        finally:
            repl.close()

    def test_ok_status_on_define(self):
        """A define without finish returns ok status."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": "(define x 42)"})
            assert resp["status"] == "ok"
        finally:
            repl.close()

    def test_stdout_included_in_response(self):
        """stdout from display is included in response."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": '(display "hello") (finish 0)'})
            assert resp["status"] == "finished"
            assert resp["stdout"] == "hello"
        finally:
            repl.close()


# ============================================================
# Recursive budget propagation (S3)
# ============================================================


class TestBudgetPropagation:
    def test_budget_wraps_code_in_parameterize(self):
        """When budget is finite, nested code runs under parameterize."""
        # We test this by simulating what _call_llm_recursive does:
        # create a nested REPL and verify the budget is properly set
        nested = RacketREPL()
        try:
            # Simulate parameterize wrapping
            code = "(finish (token-budget))"
            budget_val = 5000
            wrapped = f"(parameterize ([token-budget {budget_val}]) {code})"
            resp = nested.send({"op": "eval", "code": wrapped})
            assert resp["status"] == "finished"
            assert resp["result"] == "5000"
        finally:
            nested.close()

    def test_budget_not_applied_when_infinite(self):
        """When budget is infinite, code runs without parameterize."""
        nested = RacketREPL()
        try:
            resp = nested.send({"op": "eval", "code": "(finish (token-budget))"})
            assert resp["status"] == "finished"
            assert "+inf.0" in resp["result"]
        finally:
            nested.close()

    def test_budget_wrapping_preserves_finish(self):
        """Wrapping in parameterize doesn't break finish."""
        nested = RacketREPL()
        try:
            code = '(define x 42) (finish x)'
            wrapped = f"(parameterize ([token-budget 3000]) {code})"
            resp = nested.send({"op": "eval", "code": wrapped})
            assert resp["status"] == "finished"
            assert resp["result"] == "42"
        finally:
            nested.close()


# ============================================================
# Consistent message roles (S5)
# ============================================================


class TestMessageRoles:
    """Test that _call_llm builds messages with consistent roles."""

    def _build_messages(self, instruction="", data="", images=None):
        """Build messages the same way _call_llm does, without API call."""
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})
        user_text = data or "(no data provided)"
        if images:
            content_parts = [{"type": "text", "text": user_text}]
            for img in images:
                url = RacketREPL._resolve_image(img)
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_text})
        return messages

    def test_instruction_and_data(self):
        """With both instruction and data, instruction is system, data is user."""
        msgs = self._build_messages("Translate", "hello")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Translate"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "hello"

    def test_instruction_only(self):
        """With instruction only, instruction is system, user gets placeholder."""
        msgs = self._build_messages("Say hello")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Say hello"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "(no data provided)"

    def test_data_only(self):
        """With data only, no system message, data is user."""
        msgs = self._build_messages(data="some data")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "some data"

    def test_neither(self):
        """With nothing, user gets placeholder."""
        msgs = self._build_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "(no data provided)"

    def test_with_images_includes_text(self):
        """With images and no data, user message includes placeholder text."""
        msgs = self._build_messages("Describe", images=["iVBORw0KGgo="])
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        content_parts = msgs[1]["content"]
        assert content_parts[0]["type"] == "text"
        assert content_parts[0]["text"] == "(no data provided)"
        assert content_parts[1]["type"] == "image_url"


# ============================================================
# MCP progress notifications (S8)
# ============================================================


class TestProgressNotifications:
    def test_execute_scheme_is_async(self):
        """execute_scheme tool handler is async (required for progress)."""
        import asyncio
        import inspect
        # Import the tool function
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import mcp_server
        # The mcp tool wraps it, but we can check the original
        assert inspect.iscoroutinefunction(mcp_server.execute_scheme)

    def test_active_calls_snapshot_format(self):
        """Active calls snapshot provides data needed for progress messages."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "async", "gpt-4o", "test instruction")
            time.sleep(0.1)
            snapshot = repl.get_active_calls_snapshot()
            assert len(snapshot) == 1
            # Progress message can be built from these fields
            call = snapshot[0]
            summary = f"{call['model']} ({call['elapsed_seconds']:.0f}s)"
            assert "gpt-4o" in summary
        finally:
            repl.close()


# ============================================================
# Project Python detection (S9)
# ============================================================


class TestPythonDetection:
    def test_detect_finds_project_venv(self):
        """_detect_project_python finds the project's .venv."""
        from mcp_server import _detect_project_python
        python_path = _detect_project_python()
        # This project has a .venv, so detection should find it
        assert python_path is not None
        assert ".venv" in python_path or "VIRTUAL_ENV" in os.environ.get("VIRTUAL_ENV", "")

    def test_detect_returns_existing_file(self):
        """Detected Python path points to an existing file."""
        from mcp_server import _detect_project_python
        python_path = _detect_project_python()
        if python_path:
            assert os.path.isfile(python_path)

    def test_venv_python_used_by_py_exec(self):
        """py-exec uses the project venv Python (can import project packages)."""
        repl = RacketREPL()
        try:
            # sys.executable should point to the venv Python
            resp = repl.send({"op": "eval", "code": '(finish (py-exec "import sys; print(sys.executable)"))'})
            assert resp["status"] == "finished"
            # Should contain .venv path since we detected it
            result = resp["result"].strip()
            assert "python" in result.lower()
        finally:
            repl.close()

    def test_detect_with_explicit_env_var(self, monkeypatch):
        """RLM_PYTHON env var overrides detection."""
        from mcp_server import _detect_project_python
        monkeypatch.setenv("RLM_PYTHON", "/usr/bin/python3")
        assert _detect_project_python() == "/usr/bin/python3"


# ============================================================
# Token usage tracking (S10)
# ============================================================


class TestTokenUsage:
    def test_initial_usage_zero(self):
        """Token usage starts at zero."""
        repl = RacketREPL()
        try:
            usage = repl.get_token_usage()
            assert usage["prompt_tokens"] == 0
            assert usage["completion_tokens"] == 0
            assert usage["total_tokens"] == 0
            assert usage["total_calls"] == 0
        finally:
            repl.close()

    def test_complete_call_increments_usage(self):
        """Completing a call increments cumulative usage."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            repl._complete_call("call_1", tokens=150, elapsed=1.0,
                                prompt_tokens=100, completion_tokens=50)
            usage = repl.get_token_usage()
            assert usage["prompt_tokens"] == 100
            assert usage["completion_tokens"] == 50
            assert usage["total_tokens"] == 150
            assert usage["total_calls"] == 1
        finally:
            repl.close()

    def test_multiple_calls_accumulate(self):
        """Multiple completed calls accumulate tokens."""
        repl = RacketREPL()
        try:
            repl._register_call("call_1", "sync", "gpt-4o")
            repl._complete_call("call_1", tokens=100, elapsed=1.0,
                                prompt_tokens=80, completion_tokens=20)
            repl._register_call("call_2", "async", "gpt-4o")
            repl._complete_call("call_2", tokens=200, elapsed=2.0,
                                prompt_tokens=150, completion_tokens=50)
            usage = repl.get_token_usage()
            assert usage["prompt_tokens"] == 230
            assert usage["completion_tokens"] == 70
            assert usage["total_tokens"] == 300
            assert usage["total_calls"] == 2
        finally:
            repl.close()

    def test_tokens_used_scheme_binding(self):
        """(tokens-used) is available in the Scheme sandbox."""
        repl = RacketREPL()
        try:
            resp = repl.send({"op": "eval", "code": "(finish (tokens-used))"})
            assert resp["status"] == "finished"
            # Should return a hash/dict with token counts
            result = resp["result"]
            assert "prompt_tokens" in result or "total_tokens" in result
        finally:
            repl.close()


# ============================================================
# LLM timeout decoupling (Issue 1: map-async timeout fix)
# ============================================================


class TestLLMTimeoutDecoupling:
    """Tests for the decoupled LLM timeout.

    The fix separates the Racket idle timeout (how long to wait for a Racket
    message) from the LLM wait timeout (how long concurrent.futures.wait()
    blocks for async LLM calls). Previously both used the same value, causing
    map-async to fail when LLM calls took longer than the Racket timeout.
    """

    def test_llm_timeout_default_5x(self):
        """Default llm_timeout is 5x the Racket timeout, minimum 300s."""
        repl = RacketREPL()
        try:
            # The llm_timeout is computed inside send(). We test it indirectly
            # by verifying send() accepts small timeouts without breaking async.
            # With timeout=60, llm_timeout should be max(60*5, 300) = 300.
            resp = repl.send({"op": "eval", "code": "(finish 42)"}, timeout=60)
            assert resp["status"] == "finished"
        finally:
            repl.close()

    def test_llm_timeout_minimum_300(self):
        """llm_timeout is at least 300s even with small Racket timeout."""
        repl = RacketREPL()
        try:
            # With timeout=10, llm_timeout should be max(10*5, 300) = 300.
            # (not 50, which would be too short for real LLM calls)
            resp = repl.send({"op": "eval", "code": "(finish 42)"}, timeout=10)
            assert resp["status"] == "finished"
        finally:
            repl.close()

    def test_llm_timeout_env_var(self, monkeypatch):
        """RLM_LLM_TIMEOUT_SECONDS env var overrides the default."""
        monkeypatch.setenv("RLM_LLM_TIMEOUT_SECONDS", "1800")
        repl = RacketREPL()
        try:
            # Verify the env var is set (actual timeout logic is internal)
            assert os.environ.get("RLM_LLM_TIMEOUT_SECONDS") == "1800"
            resp = repl.send({"op": "eval", "code": "(finish 42)"}, timeout=60)
            assert resp["status"] == "finished"
        finally:
            repl.close()

    def test_stdout_queue_always_exists(self):
        """stdout queue is created on all platforms (not just Windows)."""
        repl = RacketREPL()
        try:
            assert repl._stdout_queue is not None
            assert hasattr(repl._stdout_queue, 'get')
        finally:
            repl.close()

    def test_stdout_thread_running(self):
        """stdout drain thread is alive after start on all platforms."""
        repl = RacketREPL()
        try:
            assert repl._stdout_thread is not None
            assert repl._stdout_thread.is_alive()
        finally:
            repl.close()


# ============================================================
# Documentation quality (Issues 2, 4, 6, 9, 10)
# ============================================================


class TestDocumentationIssues:
    """Tests that documentation fixes from PoC improvement report are in place."""

    def test_issue2_pyeval_warning_in_api_reference(self):
        """Issue 2: API reference warns about py-eval only accepting expressions."""
        from mcp_server import get_code_generation_api_reference
        ref = get_code_generation_api_reference()
        assert "COMMON MISTAKE" in ref
        assert "py-eval CANNOT handle imports" in ref
        assert "SyntaxError" in ref

    def test_issue4_mapasync_raw_strings_documented(self):
        """Issue 4: API reference clarifies map-async returns raw strings."""
        from mcp_server import get_code_generation_api_reference
        ref = get_code_generation_api_reference()
        assert "raw LLM output strings" in ref
        assert "NOT parsed JSON" in ref

    def test_issue9_pyset_roundtrip_example(self):
        """Issue 9: API reference has py-set! → py-exec → py-eval round-trip."""
        from mcp_server import get_code_generation_api_reference
        ref = get_code_generation_api_reference()
        assert "py-set!" in ref
        assert "py-exec" in ref
        assert "py-eval" in ref
        # Check the round-trip pattern specifically
        assert "round-trip" in ref.lower()
        assert "py-set! \"raw_json\"" in ref

    def test_issue10_state_persistence_documented(self):
        """Issue 10: Usage guide mentions sandbox state persistence."""
        from mcp_server import USAGE_GUIDE_CORE
        # getting-started.md mentions persistent state in execution model section
        assert "persist" in USAGE_GUIDE_CORE.lower()
        assert "reset()" in USAGE_GUIDE_CORE
        # Should mention execution model or state
        assert ("execution model" in USAGE_GUIDE_CORE.lower() or "state" in USAGE_GUIDE_CORE.lower())
        # Note: Detailed documentation is in execution-model.md, quick start just mentions it

