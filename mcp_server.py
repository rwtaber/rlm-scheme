"""RLM-Scheme — MCP server for safe LLM orchestration.

Exposes a Racket sandbox with hygienic scope tracking as 8 MCP tools.
Claude Code launches this via .mcp.json; it manages racket_server.rkt
as a subprocess with interleaved llm-query callbacks.

Architecture:
  Claude Code --JSON-RPC/stdio--> mcp_server.py --JSON/stdin--> racket_server.rkt
                                                                    |
                                                                    +--JSON/stdin--> py_bridge.py
"""

import asyncio
import base64
import collections
import concurrent.futures
import json
import mimetypes
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv
load_dotenv()

import openai
from mcp.server.fastmcp import Context, FastMCP

IS_WINDOWS = platform.system() == "Windows"

_DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")

def _load_doc(relative_path: str) -> str:
    with open(os.path.join(_DOCS_DIR, relative_path), "r", encoding="utf-8") as f:
        return f.read()

mcp = FastMCP("scope")

MAX_RECURSION_DEPTH = 3

# L7: Checkpoint directory for persistent storage across timeouts
CHECKPOINT_DIR = os.path.join(os.getcwd(), ".rlm-scheme-checkpoints")

def _ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared call registry — visible across all RacketREPL instances (S7)
# ---------------------------------------------------------------------------

class CallRegistry:
    """Thread-safe shared registry for all in-flight LLM calls across REPL depths."""

    def __init__(self):
        self._calls: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._change_event = threading.Event()
        self._stats = {"dispatched": 0, "completed": 0}
        # Per-execution history of completed calls (model, latency)
        self._history: list[dict] = []

    def register(self, call_id: str, call_type: str, model: str,
                 depth: int = 0, instruction_preview: str = "",
                 parent_id: str | None = None) -> None:
        with self._lock:
            self._calls[call_id] = {
                "call_id": call_id,
                "type": call_type,
                "model": model,
                "depth": depth,
                "instruction_preview": instruction_preview,
                "parent_id": parent_id,
                "start_time": time.time(),
            }
            self._stats["dispatched"] += 1
        self._change_event.set()

    def complete(self, call_id: str) -> None:
        with self._lock:
            entry = self._calls.pop(call_id, None)
            self._stats["completed"] += 1
            if entry:
                self._history.append({
                    "model": entry["model"],
                    "latency": round(time.time() - entry["start_time"], 1),
                })
        self._change_event.set()

    def snapshot(self) -> list[dict]:
        with self._lock:
            now = time.time()
            return [
                {**c, "elapsed_seconds": round(now - c["start_time"], 1)}
                for c in self._calls.values()
            ]

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_execution_summary(self) -> dict:
        """Return a summary of the completed execution: call count, models, max latency."""
        with self._lock:
            stats = dict(self._stats)
            history = list(self._history)
        if not history:
            return {"llm_calls": 0}
        model_counts = collections.Counter(h["model"] for h in history)
        models_str = ", ".join(f"{m}\u00d7{n}" for m, n in model_counts.most_common())
        max_latency = max(h["latency"] for h in history)
        return {
            "llm_calls": stats["completed"],
            "models": models_str,
            "max_call_latency": max_latency,
        }

    def reset_stats(self) -> None:
        with self._lock:
            self._stats["dispatched"] = 0
            self._stats["completed"] = 0
            self._history.clear()

    def clear(self) -> None:
        with self._lock:
            self._calls.clear()
            self._stats["dispatched"] = 0
            self._stats["completed"] = 0
            self._history.clear()

    def wait_for_change(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for a state change. Returns True if woken early."""
        self._change_event.clear()
        return self._change_event.wait(timeout=timeout)


_call_registry = CallRegistry()


def _detect_project_python() -> str | None:
    """Detect the project's Python interpreter (venv, VIRTUAL_ENV, etc.)."""
    def _find_python_in_venv(venv_path: str) -> str | None:
        """Try to find Python executable in a venv (Unix or Windows)."""
        # Unix: bin/python3
        unix_candidate = os.path.join(venv_path, "bin", "python3")
        if os.path.isfile(unix_candidate):
            return unix_candidate
        # Windows: Scripts/python.exe
        windows_candidate = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.isfile(windows_candidate):
            return windows_candidate
        return None

    # 1. Explicit env var
    if os.environ.get("RLM_PYTHON"):
        return os.environ["RLM_PYTHON"]
    # 2. VIRTUAL_ENV env var (standard for activated venvs)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = _find_python_in_venv(venv)
        if candidate:
            return candidate
    # 3. .venv in working directory
    cwd_venv_path = os.path.join(os.getcwd(), ".venv")
    candidate = _find_python_in_venv(cwd_venv_path)
    if candidate:
        return candidate
    # 4. .venv relative to this file's directory
    project_venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    candidate = _find_python_in_venv(project_venv_path)
    if candidate:
        return candidate
    return None

# ---------------------------------------------------------------------------
# Usage guide & pattern documentation — loaded from docs/ at import time
# ---------------------------------------------------------------------------

USAGE_GUIDE_CORE = _load_doc("usage-guide.md")

_PATTERN_FILES = {
    1: "patterns/pattern-01-parallel-fan-out.md",
    2: "patterns/pattern-02-code-generation.md",
    3: "patterns/pattern-03-recursive-delegation.md",
    4: "patterns/pattern-04-critique-refine.md",
    5: "patterns/pattern-05-cumulative-fold.md",
    6: "patterns/pattern-06-meta-orchestration.md",
    7: "patterns/pattern-07-speculative-execution.md",
    8: "patterns/pattern-08-ensemble-voting.md",
    9: "patterns/pattern-09-active-learning.md",
    10: "patterns/pattern-10-tree-aggregation.md",
    11: "patterns/pattern-11-consensus-protocol.md",
    12: "patterns/pattern-12-backtracking-search.md",
    13: "patterns/pattern-13-anytime-algorithms.md",
    14: "patterns/pattern-14-memoization.md",
    15: "patterns/pattern-15-stream-processing.md",
    16: "patterns/pattern-16-multi-armed-bandit.md",
}
PATTERN_DETAILS = {pid: _load_doc(path) for pid, path in _PATTERN_FILES.items()}
# Pattern 16 includes the general reference sections (Parts IV & V) for backward compat
PATTERN_DETAILS[16] += _load_doc("primitive-reference.md") + _load_doc("best-practices.md")

_CODE_GEN_API_REF = _load_doc("api-reference.md")



# ---------------------------------------------------------------------------
# RacketREPL — manages the Racket subprocess
# ---------------------------------------------------------------------------

class RacketREPL:
    STDERR_BUFFER_SIZE = 200  # max lines to keep in ring buffer

    def __init__(self):
        self.proc = None
        # Configurable thread pool for async LLM calls
        max_workers = int(os.environ.get("RLM_MAX_WORKERS", "20"))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending = {}  # id -> Future
        self._current_depth = 0
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._active_calls = {}  # call_id -> call metadata dict
        self._active_calls_lock = threading.Lock()
        self._stderr_buffer = collections.deque(maxlen=self.STDERR_BUFFER_SIZE)
        self._stderr_thread = None
        # Threaded stdout reader (all platforms — ensures timeout covers full readline)
        self._stdout_queue = queue.Queue()
        self._stdout_thread = None
        # Rate limit state from most recent API response (S4)
        self._rate_limits = {
            "limit_requests": 0,
            "limit_tokens": 0,
            "remaining_requests": 0,
            "remaining_tokens": 0,
            "reset_requests": "",
            "reset_tokens": "",
        }
        # Cumulative token usage (S10)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0
        self._token_lock = threading.Lock()
        # Per-send() call stats for progress reporting (S5)
        self._call_stats = {"dispatched": 0, "completed": 0}
        self._call_stats_lock = threading.Lock()
        self._start()

    def _next_call_id(self) -> str:
        with self._call_counter_lock:
            self._call_counter += 1
            if self._current_depth > 0:
                return f"call_d{self._current_depth}_{self._call_counter}"
            return f"call_{self._call_counter}"

    def _register_call(self, call_id: str, call_type: str, model: str,
                       instruction: str = "", parent_id: str | None = None,
                       depth: int = 0, future: concurrent.futures.Future | None = None,
                       nested_repl: "RacketREPL | None" = None,
                       cancel_event: threading.Event | None = None) -> dict:
        meta = {
            "call_id": call_id,
            "type": call_type,
            "model": model,
            "depth": depth,
            "instruction_preview": instruction[:80] if instruction else "",
            "parent_id": parent_id,
            "start_time": time.time(),
            "future": future,
            "nested_repl": nested_repl,
            "cancel_event": cancel_event,
        }
        with self._active_calls_lock:
            self._active_calls[call_id] = meta
        with self._call_stats_lock:
            self._call_stats["dispatched"] += 1
        _call_registry.register(call_id, call_type, model, depth,
                                instruction[:80] if instruction else "", parent_id)
        print(f"[rlm] {call_id}: calling {model} ({call_type}, {len(instruction)} chars, depth {depth})...",
              file=sys.stderr, flush=True)
        return meta

    def _complete_call(self, call_id: str, tokens: int = 0, elapsed: float = 0,
                       prompt_tokens: int = 0, completion_tokens: int = 0):
        with self._active_calls_lock:
            self._active_calls.pop(call_id, None)
        with self._call_stats_lock:
            self._call_stats["completed"] += 1
        _call_registry.complete(call_id)
        with self._token_lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_calls += 1
        print(f"[rlm] {call_id}: completed ({tokens} tokens, {elapsed:.1f}s)",
              file=sys.stderr, flush=True)

    def get_token_usage(self) -> dict:
        """Return cumulative token usage across all completed calls."""
        with self._token_lock:
            return {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "total_calls": self._total_calls,
            }

    def _update_rate_limits(self, headers) -> None:
        """Update rate limit state from OpenAI response headers."""
        def _int(name: str) -> int:
            val = headers.get(name, "0")
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
        self._rate_limits = {
            "limit_requests": _int("x-ratelimit-limit-requests"),
            "limit_tokens": _int("x-ratelimit-limit-tokens"),
            "remaining_requests": _int("x-ratelimit-remaining-requests"),
            "remaining_tokens": _int("x-ratelimit-remaining-tokens"),
            "reset_requests": headers.get("x-ratelimit-reset-requests", ""),
            "reset_tokens": headers.get("x-ratelimit-reset-tokens", ""),
        }

    def get_rate_limits(self) -> dict:
        """Return rate limit state from the most recent API response."""
        return dict(self._rate_limits)

    def reset_call_stats(self):
        """Reset per-execution call stats (called at start of each execute_scheme)."""
        with self._call_stats_lock:
            self._call_stats["dispatched"] = 0
            self._call_stats["completed"] = 0

    def get_call_stats(self) -> dict:
        """Return dispatched/completed call counts for current execution."""
        with self._call_stats_lock:
            return dict(self._call_stats)

    def get_active_calls_snapshot(self) -> list[dict]:
        with self._active_calls_lock:
            now = time.time()
            return [
                {
                    "call_id": m["call_id"],
                    "type": m["type"],
                    "model": m["model"],
                    "depth": m["depth"],
                    "instruction_preview": m["instruction_preview"],
                    "parent_id": m["parent_id"],
                    "elapsed_seconds": round(now - m["start_time"], 1),
                }
                for m in self._active_calls.values()
            ]

    def cancel_call(self, call_id: str) -> str:
        with self._active_calls_lock:
            meta = self._active_calls.pop(call_id, None)
        if meta is None:
            return f"No active call with ID {call_id}"
        # Signal cancellation via event (checked before API call)
        if meta.get("cancel_event") is not None:
            meta["cancel_event"].set()
        # Cancel async futures
        if meta.get("future") is not None:
            meta["future"].cancel()
        # Kill nested REPL for recursive calls
        if meta.get("nested_repl") is not None:
            meta["nested_repl"].close()
        print(f"[rlm] {call_id}: cancelled by user", file=sys.stderr, flush=True)
        return f"Cancelled {call_id}"

    def _drain_stderr(self):
        """Daemon thread: continuously read stderr lines into ring buffer."""
        try:
            while self.proc and self.proc.stderr:
                line = self.proc.stderr.readline()
                if not line:
                    break
                stripped = line.rstrip("\n\r")
                if stripped:
                    self._stderr_buffer.append(stripped)
                    print(f"[racket-stderr] {stripped}", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            # Pipe closed — process is shutting down
            pass

    def get_stderr_log(self) -> list[str]:
        """Return the last N lines from Racket's stderr."""
        return list(self._stderr_buffer)

    def _drain_stdout(self):
        """Background thread: read lines from Racket stdout into queue.

        Used on all platforms to ensure _read_line timeout covers the full
        readline (not just select/poll). This prevents the race where select()
        fires on partial data but readline() blocks waiting for the newline.
        """
        proc = self.proc  # Capture reference — self.proc may become None on timeout
        try:
            while proc and proc.poll() is None:
                line = proc.stdout.readline()
                if line:
                    self._stdout_queue.put(("line", line))
                else:
                    break
        except Exception as e:
            self._stdout_queue.put(("error", str(e)))
        finally:
            self._stdout_queue.put(("eof", None))

    def _start(self):
        if self.proc is not None:
            self.close()
        server_rkt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "racket_server.rkt")
        env = os.environ.copy()
        # Pass detected project Python to Racket (S9: venv inheritance)
        python_path = _detect_project_python()
        if python_path:
            env["RLM_PYTHON"] = python_path
            print(f"[rlm] Using Python: {python_path}", file=sys.stderr, flush=True)
        self.proc = subprocess.Popen(
            ["racket", server_rkt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env,
        )
        # Use threaded stdout reader on all platforms for reliable timeout enforcement.
        # (selector + readline can miss timeout if partial data triggers select
        # but readline blocks waiting for newline on large messages)
        self._stdout_queue.queue.clear()
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        # Drain stderr in a daemon thread to prevent pipe buffer deadlock
        self._stderr_buffer.clear()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    SIGTERM_GRACE_SECONDS = 2

    def _read_line(self, timeout: float) -> str:
        """Read one line from Racket stdout with timeout.

        Uses a queue fed by a background thread on all platforms.
        This ensures the timeout covers the full readline, not just
        the select/poll check (which can fire on partial data).
        """
        try:
            msg_type, data = self._stdout_queue.get(timeout=timeout)
            if msg_type == "line":
                return data
            elif msg_type == "eof":
                raise RuntimeError("Racket process stdout closed")
            elif msg_type == "error":
                raise RuntimeError(f"Racket stdout reader error: {data}")
        except queue.Empty:
            # Timeout — Racket hasn't sent a message in `timeout` seconds.
            # Graceful shutdown: SIGTERM first, then SIGKILL after grace period.
            self.proc.terminate()
            try:
                self.proc.wait(timeout=self.SIGTERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            raise TimeoutError(f"Racket process timed out after {timeout}s")

    def send(self, cmd: dict, timeout: float = 300) -> dict:
        """Send a command, handle interleaved llm-query callbacks, return the final response.

        Args:
            timeout: Max seconds of idle time between Racket messages (computation timeout).
                     Does NOT limit LLM API call time — those use a separate, longer timeout.
        """
        # Decouple LLM wait timeout from Racket computation timeout.
        # LLM API calls (map-async, await-batch, await-any) can take much longer
        # than Racket computation. Using the same timeout for both was the root
        # cause of map-async failures: even generous timeouts (600s) could be
        # exceeded by parallel LLM batches, killing the Racket process.
        #
        # llm_timeout: max time to wait for LLM futures in concurrent.futures.wait()
        # Configurable via RLM_LLM_TIMEOUT_SECONDS env var.
        # Default: 5x the Racket timeout, minimum 300s (5 minutes).
        llm_timeout_env = os.environ.get("RLM_LLM_TIMEOUT_SECONDS")
        llm_timeout = float(llm_timeout_env) if llm_timeout_env else max(timeout * 5, 300)

        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self.proc.stdin.write(json.dumps(cmd) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self._read_line(timeout)
                if not line:
                    raise RuntimeError("Racket process died")
                msg = json.loads(line.strip())
                op = msg.get("op")

                if op == "llm-query":
                    llm_kwargs = {
                        "instruction": msg.get("instruction", ""),
                        "data": msg.get("data", ""),
                        "model": msg.get("model", ""),
                        "temperature": msg.get("temperature"),
                        "max_tokens": msg.get("max_tokens"),
                        "json_mode": msg.get("json_mode", False),
                        "images": msg.get("images"),
                    }
                    call_id = self._next_call_id()
                    model_name = llm_kwargs["model"] or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
                    call_type = "recursive" if msg.get("recursive") else "sync"
                    t0 = time.time()
                    self._register_call(call_id, call_type, model_name,
                                        llm_kwargs["instruction"],
                                        depth=self._current_depth)
                    try:
                        if msg.get("recursive"):
                            rec_result = self._call_llm_recursive(
                                llm_kwargs["instruction"],
                                llm_kwargs["data"],
                                llm_kwargs["model"],
                                msg.get("budget"),
                                _call_id=call_id,
                            )
                            total_tokens = rec_result["prompt_tokens"] + rec_result["completion_tokens"]
                            self._complete_call(call_id, total_tokens, time.time() - t0,
                                                rec_result["prompt_tokens"], rec_result["completion_tokens"])
                            self.proc.stdin.write(json.dumps({
                                "result": rec_result["text"],
                                "prompt_tokens": rec_result["prompt_tokens"],
                                "completion_tokens": rec_result["completion_tokens"],
                            }) + "\n")
                        else:
                            result = self._call_llm(**llm_kwargs)
                            total_tokens = result["prompt_tokens"] + result["completion_tokens"]
                            self._complete_call(call_id, total_tokens, time.time() - t0,
                                                result["prompt_tokens"], result["completion_tokens"])
                            self.proc.stdin.write(json.dumps({
                                "result": result["text"],
                                "prompt_tokens": result["prompt_tokens"],
                                "completion_tokens": result["completion_tokens"],
                            }) + "\n")
                    except Exception as exc:
                        self._complete_call(call_id, 0, time.time() - t0)
                        raise exc
                    self.proc.stdin.flush()

                elif op == "llm-query-async":
                    call_id = self._next_call_id()
                    model_name = msg.get("model", "") or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
                    cancel_event = threading.Event()
                    self._register_call(call_id, "async", model_name,
                                        msg.get("instruction", ""),
                                        depth=self._current_depth,
                                        cancel_event=cancel_event)
                    future = self._executor.submit(
                        self._call_llm_tracked, call_id,
                        msg.get("instruction", ""),
                        msg.get("data", ""),
                        msg.get("model", ""),
                        msg.get("temperature"),
                        msg.get("max_tokens"),
                        msg.get("json_mode", False),
                        msg.get("images"),
                        cancel_event,
                    )
                    self._pending[msg["id"]] = future
                    # Store future for cancellation
                    with self._active_calls_lock:
                        if call_id in self._active_calls:
                            self._active_calls[call_id]["future"] = future
                    # No response — Racket continues immediately

                elif op == "await":
                    future = self._pending.pop(msg["id"], None)
                    if future is None:
                        self.proc.stdin.write(json.dumps({
                            "result": "",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        }) + "\n")
                    else:
                        try:
                            result = future.result(timeout=llm_timeout)
                            self.proc.stdin.write(json.dumps({
                                "result": result["text"],
                                "prompt_tokens": result["prompt_tokens"],
                                "completion_tokens": result["completion_tokens"],
                            }) + "\n")
                        except concurrent.futures.CancelledError:
                            self.proc.stdin.write(json.dumps({
                                "result": "[cancelled] call was cancelled by user",
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                        except ValueError as e:
                            # L8: JSON mode validation errors get clear message
                            error_msg = f"[async error] {type(e).__name__}: {str(e)}"
                            self.proc.stdin.write(json.dumps({
                                "result": error_msg,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                        except openai.APIError as e:
                            # L8: OpenAI API errors with status code
                            status = getattr(e, "status_code", "unknown")
                            error_msg = f"[async error] API {status}: {str(e)[:300]}"
                            self.proc.stdin.write(json.dumps({
                                "result": error_msg,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                        except Exception as e:
                            # L8: Generic errors with type info
                            error_msg = f"[async error] {type(e).__name__}: {str(e)[:300]}"
                            self.proc.stdin.write(json.dumps({
                                "result": error_msg,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                    self.proc.stdin.flush()

                elif op == "await-batch":
                    # Parallel await: wait for multiple futures concurrently
                    future_ids = msg.get("ids", [])
                    futures_list = []
                    id_to_future = {}

                    for fid in future_ids:
                        future = self._pending.get(fid)
                        if future:
                            futures_list.append(future)
                            id_to_future[future] = fid

                    if not futures_list:
                        # No valid futures
                        self.proc.stdin.write(json.dumps({"results": []}) + "\n")
                        self.proc.stdin.flush()
                        continue

                    # Wait for all futures concurrently.
                    # Uses llm_timeout (not timeout) — LLM calls can take much longer
                    # than Racket computation. This is the key fix for map-async timeouts.
                    done, not_done = concurrent.futures.wait(
                        futures_list,
                        timeout=llm_timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )

                    # Build results in original order
                    results = []
                    for fid in future_ids:
                        future = self._pending.pop(fid, None)
                        if future is None:
                            results.append({
                                "result": "[error] future not found",
                                "prompt_tokens": 0,
                                "completion_tokens": 0
                            })
                        elif future not in done:
                            results.append({
                                "result": "[error] future timed out",
                                "prompt_tokens": 0,
                                "completion_tokens": 0
                            })
                        else:
                            try:
                                res = future.result(timeout=0)  # Already done
                                results.append({
                                    "result": res["text"],
                                    "prompt_tokens": res["prompt_tokens"],
                                    "completion_tokens": res["completion_tokens"]
                                })
                            except concurrent.futures.CancelledError:
                                results.append({
                                    "result": "[cancelled] call was cancelled by user",
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })
                            except ValueError as e:
                                error_msg = f"[async error] {type(e).__name__}: {str(e)}"
                                results.append({
                                    "result": error_msg,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })
                            except openai.APIError as e:
                                status = getattr(e, "status_code", "unknown")
                                error_msg = f"[async error] API {status}: {str(e)[:300]}"
                                results.append({
                                    "result": error_msg,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })
                            except Exception as e:
                                error_msg = f"[async error] {type(e).__name__}: {str(e)[:300]}"
                                results.append({
                                    "result": error_msg,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })

                    self.proc.stdin.write(json.dumps({"results": results}) + "\n")
                    self.proc.stdin.flush()

                elif op == "await-any":
                    # Wait for ANY future to complete (race pattern)
                    future_ids = msg.get("ids", [])
                    if not future_ids:
                        self.proc.stdin.write(json.dumps({
                            "error": "No futures provided",
                            "completed_id": None,
                            "remaining_ids": []
                        }) + "\n")
                        self.proc.stdin.flush()
                        continue

                    futures_list = []
                    id_to_future = {}

                    for fid in future_ids:
                        future = self._pending.get(fid)
                        if future:
                            futures_list.append(future)
                            id_to_future[future] = fid

                    if not futures_list:
                        self.proc.stdin.write(json.dumps({
                            "error": "No valid futures found",
                            "completed_id": None,
                            "remaining_ids": []
                        }) + "\n")
                        self.proc.stdin.flush()
                        continue

                    # Wait for FIRST completion.
                    # Uses llm_timeout — same decoupling as await-batch.
                    done, not_done = concurrent.futures.wait(
                        futures_list,
                        timeout=llm_timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    if not done:
                        # Timeout - no futures completed
                        self.proc.stdin.write(json.dumps({
                            "error": "Timeout waiting for any future",
                            "completed_id": None,
                            "remaining_ids": future_ids
                        }) + "\n")
                        self.proc.stdin.flush()
                        continue

                    # Get the completed future
                    completed_future = next(iter(done))
                    completed_id = id_to_future[completed_future]
                    self._pending.pop(completed_id)

                    # Get remaining IDs
                    remaining_ids = [id_to_future[f] for f in not_done]

                    # Extract result from completed future
                    try:
                        res = completed_future.result(timeout=0)  # Already done
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": res["text"],
                            "prompt_tokens": res["prompt_tokens"],
                            "completion_tokens": res["completion_tokens"],
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except concurrent.futures.CancelledError:
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": "[cancelled] call was cancelled by user",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except ValueError as e:
                        error_msg = f"[async error] {type(e).__name__}: {str(e)}"
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": error_msg,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except openai.APIError as e:
                        status = getattr(e, "status_code", "unknown")
                        error_msg = f"[async error] API {status}: {str(e)[:300]}"
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": error_msg,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except Exception as e:
                        error_msg = f"[async error] {type(e).__name__}: {str(e)[:300]}"
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": error_msg,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")

                    self.proc.stdin.flush()

                elif op == "tokens-used":
                    usage = self.get_token_usage()
                    self.proc.stdin.write(json.dumps(usage) + "\n")
                    self.proc.stdin.flush()

                elif op == "rate-limits":
                    self.proc.stdin.write(json.dumps(self._rate_limits) + "\n")
                    self.proc.stdin.flush()

                elif op == "checkpoint":
                    # L7: Save value to disk under key
                    key = msg.get("key", "")
                    value = msg.get("value")
                    if not key:
                        self.proc.stdin.write(json.dumps({"status": "error", "message": "key required"}) + "\n")
                    else:
                        try:
                            checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{key}.json")
                            with open(checkpoint_file, "w") as f:
                                json.dump(value, f, indent=2)
                            self.proc.stdin.write(json.dumps({"status": "ok"}) + "\n")
                            print(f"[rlm] Checkpoint saved: {key}", file=sys.stderr, flush=True)
                        except Exception as e:
                            self.proc.stdin.write(json.dumps({"status": "error", "message": str(e)}) + "\n")
                    self.proc.stdin.flush()

                elif op == "restore":
                    # L7: Load value from disk
                    key = msg.get("key", "")
                    if not key:
                        self.proc.stdin.write(json.dumps({"value": None}) + "\n")
                    else:
                        try:
                            checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{key}.json")
                            if os.path.exists(checkpoint_file):
                                with open(checkpoint_file, "r") as f:
                                    value = json.load(f)
                                self.proc.stdin.write(json.dumps({"value": value}) + "\n")
                                print(f"[rlm] Checkpoint restored: {key}", file=sys.stderr, flush=True)
                            else:
                                self.proc.stdin.write(json.dumps({"value": None}) + "\n")
                        except Exception as e:
                            print(f"[rlm] Checkpoint restore error ({key}): {e}", file=sys.stderr, flush=True)
                            self.proc.stdin.write(json.dumps({"value": None}) + "\n")
                    self.proc.stdin.flush()

                elif op == "heartbeat":
                    # Heartbeat from Racket — it's alive, just reset the idle timer.
                    # Sent automatically by map-async during long fan-outs, and
                    # available to user code via (heartbeat) for custom long computations.
                    self.proc.stdin.write(json.dumps({"ok": True}) + "\n")
                    self.proc.stdin.flush()
                    continue  # Loop back to _read_line with fresh timeout

                else:
                    return msg
        except TimeoutError:
            raise
        except (RuntimeError, BrokenPipeError, OSError) as e:
            if self.proc:
                self.proc.kill()
                self.proc = None
            stderr_log = self.get_stderr_log()
            stderr_text = "\n".join(stderr_log[-10:]) if stderr_log else "(no stderr)"
            self._start()
            return {
                "status": "error",
                "message": f"Racket process crashed: {e}. Stderr: {stderr_text}. Sandbox restarted — state was lost. Call load_context again if needed.",
            }

    MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
    IMAGE_MAGIC = {
        b"\x89PNG": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"GIF8": "image/gif",
        b"RIFF": "image/webp",  # WebP starts with RIFF
    }

    @staticmethod
    def _resolve_image(img: str) -> str:
        """Convert an image argument to a data URL.

        Accepts:
          - A data URL (data:image/...) — returned as-is.
          - A file path — read, base64-encoded, and wrapped in a data URL.
          - A raw base64 string — wrapped in a data URL (assumes PNG).

        Raises ValueError for files exceeding MAX_IMAGE_BYTES or
        non-image files (based on magic bytes).
        """
        if img.startswith("data:"):
            return img
        if os.path.isfile(img):
            file_size = os.path.getsize(img)
            if file_size > RacketREPL.MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Image file too large: {file_size} bytes "
                    f"(limit: {RacketREPL.MAX_IMAGE_BYTES // (1024*1024)} MB)"
                )
            with open(img, "rb") as f:
                data = f.read()
            # Validate magic bytes
            recognized = False
            for magic, _ in RacketREPL.IMAGE_MAGIC.items():
                if data[:len(magic)] == magic:
                    recognized = True
                    break
            if not recognized and len(data) > 4:
                print(f"[rlm] Warning: {img} may not be an image (unrecognized magic bytes)",
                      file=sys.stderr, flush=True)
            mime, _ = mimetypes.guess_type(img)
            if mime is None:
                mime = "image/png"
            b64 = base64.b64encode(data).decode()
            return f"data:{mime};base64,{b64}"
        # Raw base64 — check length (base64 of 20MB is ~27MB)
        max_b64_len = RacketREPL.MAX_IMAGE_BYTES * 4 // 3 + 4
        if len(img) > max_b64_len:
            raise ValueError(
                f"Base64 image data too large: {len(img)} chars "
                f"(limit: {max_b64_len} chars)"
            )
        return f"data:image/png;base64,{img}"

    MAX_RECOMMENDED_IMAGES = 5

    # Retry configuration — tuneable via environment variables.
    _max_retries: int = int(os.environ.get("RLM_MAX_RETRIES", "3"))
    _retry_base_delay: float = float(os.environ.get("RLM_RETRY_BASE_DELAY", "2.0"))

    def _call_llm(self, instruction: str, data: str, model: str = "",
                   temperature: float | None = None, max_tokens: int | None = None,
                   json_mode: bool = False,
                   images: list[str] | None = None,
                   cancel_event: threading.Event | None = None) -> dict:
        """Dispatch a sub-LLM call via the OpenAI API. Returns text + token counts."""
        if cancel_event and cancel_event.is_set():
            raise concurrent.futures.CancelledError("Call cancelled before API request")

        # L8: Validate JSON mode requirements (must have "json" in instruction)
        if json_mode and (not instruction or "json" not in instruction.lower()):
            error_msg = (
                "JSON mode requires the word 'json' in the instruction. "
                "OpenAI API will return 400 error without it. "
                f"Instruction: {instruction[:100] if instruction else '(empty)'}..."
            )
            print(f"[rlm] Error: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)

        if images and len(images) > self.MAX_RECOMMENDED_IMAGES:
            print(f"[rlm] Warning: {len(images)} images sent (recommended max: "
                  f"{self.MAX_RECOMMENDED_IMAGES}). Response quality may degrade.",
                  file=sys.stderr, flush=True)
        client = openai.OpenAI()
        model = model or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
        # Build messages — instruction is always system, data is always user.
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})

        # Build user message — multimodal when images are present.
        user_text = data or "(no data provided)"
        if images:
            content_parts = [{"type": "text", "text": user_text}]
            for img in images:
                url = self._resolve_image(img)
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_text})
        kwargs = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = self._max_retries
        base_delay = self._retry_base_delay

        for attempt in range(max_retries + 1):
            if cancel_event and cancel_event.is_set():
                raise concurrent.futures.CancelledError("Call cancelled before retry")
            try:
                raw = client.chat.completions.with_raw_response.create(**kwargs)
                resp = raw.parse()
                # Capture rate limit headers (S4)
                self._update_rate_limits(raw.headers)
                return {
                    "text": resp.choices[0].message.content,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            except openai.RateLimitError as e:
                if attempt >= max_retries:
                    raise
                # Respect Retry-After header if present.
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after_str = e.response.headers.get("Retry-After")
                    if retry_after_str:
                        try:
                            retry_after = float(retry_after_str)
                        except (ValueError, TypeError):
                            pass
                delay = base_delay * (2 ** attempt)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                print(f"[rlm] Rate limited (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {delay:.1f}s...",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
            except openai.APIStatusError as e:
                if e.status_code == 429 and attempt < max_retries:
                    retry_after = None
                    if hasattr(e, "response") and e.response is not None:
                        retry_after_str = e.response.headers.get("Retry-After")
                        if retry_after_str:
                            try:
                                retry_after = float(retry_after_str)
                            except (ValueError, TypeError):
                                pass
                    delay = base_delay * (2 ** attempt)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    print(f"[rlm] API 429 (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {delay:.1f}s...",
                          file=sys.stderr, flush=True)
                    time.sleep(delay)
                    continue
                raise

    def _call_llm_tracked(self, call_id: str, instruction: str, data: str,
                           model: str = "", temperature: float | None = None,
                           max_tokens: int | None = None,
                           json_mode: bool = False,
                           images: list[str] | None = None,
                           cancel_event: threading.Event | None = None) -> dict:
        """Like _call_llm but auto-completes the call registry entry when done."""
        t0 = time.time()
        try:
            result = self._call_llm(instruction, data, model, temperature, max_tokens, json_mode, images,
                                     cancel_event=cancel_event)
            total_tokens = result["prompt_tokens"] + result["completion_tokens"]
            self._complete_call(call_id, total_tokens, time.time() - t0,
                                result["prompt_tokens"], result["completion_tokens"])
            return result
        except Exception as e:
            # L8: Log async failures with details
            elapsed = time.time() - t0
            error_type = type(e).__name__
            print(f"[rlm] {call_id}: FAILED after {elapsed:.1f}s - {error_type}: {str(e)[:200]}",
                  file=sys.stderr, flush=True)
            self._complete_call(call_id, 0, elapsed)
            raise

    def _call_llm_recursive(self, instruction: str, data: str, model: str = "",
                             budget: float | None = None,
                             _call_id: str | None = None) -> dict:
        """Handle a recursive llm-query: call the sub-model, then run its Scheme code in a nested sandbox."""
        depth = self._current_depth + 1
        if depth > MAX_RECURSION_DEPTH:
            return {
                "text": f"[error] recursion depth limit ({MAX_RECURSION_DEPTH}) exceeded",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        # Build the prompt — include USAGE_GUIDE so the sub-model knows the sandbox API
        full_instruction = (
            f"{instruction}\n\n"
            f"You have access to a Scheme sandbox. Write Scheme code to accomplish the task.\n"
            f"Use (finish value) to return your result.\n\n"
            f"## Sandbox Reference\n{USAGE_GUIDE}"
        )

        # Call the sub-model to get Scheme code
        result = self._call_llm(full_instruction, data, model)
        scheme_code = self._extract_code(result["text"])
        total_prompt = result["prompt_tokens"]
        total_completion = result["completion_tokens"]

        # Spin up a nested sandbox
        nested = RacketREPL()
        nested._current_depth = depth
        # Store nested REPL in call registry for cancellation
        if _call_id:
            with self._active_calls_lock:
                if _call_id in self._active_calls:
                    self._active_calls[_call_id]["nested_repl"] = nested
        try:
            # Load context if data was provided
            if data:
                nested.send({"op": "load-context", "data": data})

            # Propagate budget if finite — wrap user code in parameterize
            budget_val = None
            if budget is not None and budget != "+inf.0":
                try:
                    bv = float(budget)
                    if bv < float("inf"):
                        budget_val = int(bv)
                except (ValueError, TypeError):
                    pass

            # Evaluate the sub-model's code, optionally under a budget
            if budget_val is not None:
                wrapped_code = f"(parameterize ([token-budget {budget_val}]) {scheme_code})"
                resp = nested.send({"op": "eval", "code": wrapped_code})
            else:
                resp = nested.send({"op": "eval", "code": scheme_code})

            if resp.get("status") == "finished":
                text = resp.get("result", "")
            elif resp.get("status") == "error":
                text = f"[sub-model error] {resp.get('message', 'unknown error')}"
            else:
                text = "(sub-model produced no output)"
        finally:
            nested.close()

        return {
            "text": text,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        }

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Scheme code from markdown fences if present."""
        match = re.search(r'```(?:scheme|racket)?\s*\n(.*?)```', text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def close(self):
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        with self._active_calls_lock:
            self._active_calls.clear()
        if self.proc:
            self.proc.terminate()
            self.proc = None


# ---------------------------------------------------------------------------
# Singleton backend
# ---------------------------------------------------------------------------

_backend = None


def get_backend() -> RacketREPL:
    global _backend
    if _backend is None:
        _ensure_checkpoint_dir()  # L7: Create checkpoint directory on init
        _backend = RacketREPL()
    return _backend


# ---------------------------------------------------------------------------
# Progress reporting (S5)
# ---------------------------------------------------------------------------

PROGRESS_POLL_INITIAL = 2   # seconds between progress updates
PROGRESS_POLL_LONG = 5      # seconds after threshold
PROGRESS_POLL_THRESHOLD = 30  # seconds before switching to long interval


def _format_progress_message(calls: list[dict], stats: dict) -> str | None:
    """Format a compact progress message from active calls and stats.

    Returns None when there's nothing to report (no calls dispatched).
    Progressive detail within 80-char cap:
    - Flat fan-out: "3/7 done — 2 active: gpt-4o×2 (longest: 45s)"
    - Recursive present: "3/7 done — 2 active: gpt-4o×2 [d1:1] (longest: 45s)"
    - Single recursive: "1/3 done — 1 active: gpt-4o [recursive] (12s)"
    - Single with room: "0/1 done — 1 active: gpt-4o 'Summarize...' (5s)"

    Truncation order: drop preview → drop depth → hard truncate at 77+"..."
    """
    dispatched = stats["dispatched"]
    completed = stats["completed"]
    if not calls and dispatched == 0:
        return None
    if not calls:
        return f"{completed}/{dispatched} done"

    model_counts = collections.Counter(c["model"] for c in calls)
    models_str = ", ".join(f"{n}\u00d7{m}" for m, n in model_counts.most_common())
    max_elapsed = max(c.get("elapsed_seconds", 0) for c in calls)

    # Depth annotation: count calls at depth > 0
    deep_calls = [c for c in calls if c.get("depth", 0) > 0]
    depth_counts = collections.Counter(c.get("depth", 0) for c in deep_calls)
    depth_str = ""
    if depth_counts:
        depth_str = " " + " ".join(
            f"[d{d}:{n}]" for d, n in sorted(depth_counts.items())
        )

    # Single-call preview: show type annotation or instruction preview
    preview_str = ""
    if len(calls) == 1:
        call = calls[0]
        if call.get("type") == "recursive":
            preview_str = " [recursive]"
        elif call.get("instruction_preview"):
            preview_str = f" '{call['instruction_preview'][:30]}...'"

    prefix = f"{completed}/{dispatched} done \u2014 {len(calls)} active: {models_str}"
    suffix = f" (longest: {max_elapsed:.0f}s)"

    # Try full message with all detail
    msg = f"{prefix}{depth_str}{preview_str}{suffix}"
    if len(msg) <= 80:
        return msg

    # Drop preview first
    msg = f"{prefix}{depth_str}{suffix}"
    if len(msg) <= 80:
        return msg

    # Drop depth
    msg = f"{prefix}{suffix}"
    if len(msg) <= 80:
        return msg

    # Hard truncate
    return msg[:77] + "..."


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

STDOUT_LIMIT = 2000


@mcp.tool()
def get_usage_guide() -> str:
    """Get essential guide with quick reference, decision framework, model selection guide, and pattern summaries. For detailed pattern implementations with complete code examples, call get_pattern_details() with specific pattern numbers. This guide helps you choose which patterns to use; get_pattern_details() gives you the full implementation details."""
    return USAGE_GUIDE_CORE


@mcp.tool()
def get_code_generation_api_reference() -> str:
    """Get condensed API reference for code-generating sub-models.

    When using Pattern 2 (Code Generation), sub-models don't automatically know
    the rlm-scheme API. Call this tool and include its output in your unsafe-raw-query
    #:data parameter so the sub-model generates correct syntax.

    This returns a minimal reference (~200 lines) optimized for inclusion in prompts.
    For the full guide with strategies and examples, use get_usage_guide instead."""

    return _CODE_GEN_API_REF


@mcp.tool()
def get_pattern_details(pattern_ids: int | list[int]) -> str:
    """Get detailed documentation for specific orchestration patterns with complete code examples.
    
    After using get_usage_guide() to choose which patterns fit your problem, call this tool
    to get full implementation details including:
    - Complete working code examples
    - Quantified improvements (latency, cost, quality metrics)
    - Optimization tips and best practices
    - Common mistakes to avoid
    - Pattern composition suggestions
    - Real-world use cases
    
    Args:
        pattern_ids: Single pattern number (1-16) or list of pattern numbers, e.g., [1, 4, 10]
        
    Available Patterns:
        1: Parallel Fan-Out (MapReduce)
        2: Code Generation (Meta-Programming)
        3: Recursive Delegation (Hierarchical Decomposition)
        4: Critique-Refine Loop
        5: Cumulative Fold (Sequential Synthesis)
        6: Meta-Orchestration (LLM Designs the Pipeline)
        7: Speculative Execution (Hedging)
        8: Ensemble Voting
        9: Active Learning (Budget-Optimized Quality)
        10: Tree Aggregation (Hierarchical Reduction)
        11: Consensus Protocol (Byzantine Fault Tolerance)
        12: Backtracking Search (Strategy Exploration)
        13: Anytime Algorithms (Progressive Refinement)
        14: Memoization (Content-Addressed Caching)
        15: Stream Processing (Constant Memory)
        16: Multi-Armed Bandit (Adaptive Model Selection)
    
    Example:
        get_pattern_details(1)  # Get Pattern 1 details
        get_pattern_details([1, 4, 10])  # Get multiple patterns
    """
    # Normalize to list
    if isinstance(pattern_ids, int):
        pattern_ids = [pattern_ids]
    
    result_parts = []
    for pid in pattern_ids:
        if pid in PATTERN_DETAILS:
            result_parts.append(PATTERN_DETAILS[pid])
        else:
            result_parts.append(f"Error: Pattern {pid} not found. Valid pattern IDs are 1-16.")

    return "\n\n---\n\n".join(result_parts)



@mcp.tool()
async def execute_scheme(code: str, timeout: int | None = None, ctx: Context = None) -> str:
    """Execute Scheme orchestration code. Use for strategy-driven LLM orchestration.

    AVAILABLE PATTERNS (16 total - choose based on your constraints):

    LATENCY: Parallel Fan-Out, Speculative Execution, Stream Processing
    QUALITY: Critique-Refine, Ensemble Voting, Consensus Protocol
    COST: Active Learning, Memoization, Multi-Armed Bandit
    STRUCTURE: Code Generation, Meta-Orchestration, Recursive Delegation, Tree Aggregation
    SPECIALIZED: Cumulative Fold, Backtracking Search, Anytime Algorithms

    Call get_usage_guide for complete details, decision framework, and code examples for all 16 patterns.

    Model selection: use gpt-4.1-nano ($0.10/1M) for fan-out and simple tasks, gpt-4o or gpt-4.1 for complex reasoning, o3-mini or o4-mini for math/logic. Always use the cheapest model that fits the task.

    State persists across calls. Capabilities: orchestrate LLM sub-calls with scope tracking, run Python code for file I/O and web requests (py-exec), process images (vision models), and fan out parallel work (map-async).

    Args:
        code: Scheme code to execute
        timeout: Optional timeout in seconds. If not specified, uses RLM_TIMEOUT_SECONDS env var (default 300).
        ctx: MCP context for progress reporting"""
    # Resolve timeout: parameter > env var > default 300
    if timeout is None:
        timeout = int(os.environ.get("RLM_TIMEOUT_SECONDS", "300"))

    backend = get_backend()
    backend.reset_call_stats()
    _call_registry.reset_stats()
    loop = asyncio.get_event_loop()
    t_start = time.monotonic()

    # Monitor active calls and report compact progress while send() blocks
    stop_monitor = threading.Event()
    timeout_warning_sent = [False]  # mutable cell for closure

    async def monitor_progress():
        start_time = time.monotonic()
        while not stop_monitor.is_set():
            elapsed = time.monotonic() - start_time

            # Warn at 80% of timeout
            if not timeout_warning_sent[0] and elapsed > timeout * 0.8:
                timeout_warning_sent[0] = True
                print(f"[rlm] Warning: {elapsed:.0f}s elapsed ({elapsed/timeout*100:.0f}% of {timeout}s timeout)",
                      file=sys.stderr, flush=True)

            interval = PROGRESS_POLL_LONG if elapsed > PROGRESS_POLL_THRESHOLD else PROGRESS_POLL_INITIAL
            # Event-driven: wake immediately on call start/complete, else poll
            await loop.run_in_executor(None, _call_registry.wait_for_change, interval)
            if stop_monitor.is_set():
                break
            calls = _call_registry.snapshot()
            stats = _call_registry.get_stats()
            msg = _format_progress_message(calls, stats)
            if msg is None:
                continue
            await ctx.report_progress(
                stats["completed"],
                total=max(stats["dispatched"], 1),
                message=msg,
            )

    monitor_task = asyncio.create_task(monitor_progress())

    try:
        resp = await loop.run_in_executor(
            None, lambda: backend.send({"op": "eval", "code": code}, timeout=timeout)
        )
    except TimeoutError as e:
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        stop_monitor.set()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    elapsed = round(time.monotonic() - t_start, 1)

    result = {"status": resp["status"]}
    if resp.get("stdout"):
        stdout = resp["stdout"]
        if len(stdout) > STDOUT_LIMIT:
            stdout = stdout[:STDOUT_LIMIT] + f"\n... ({len(resp['stdout'])} chars total, truncated)"
        result["stdout"] = stdout
    if resp["status"] == "finished":
        result["value"] = resp["result"]
    elif resp["status"] == "error":
        result["message"] = resp.get("message", "unknown error")

    # Execution summary — always included so caller has visibility
    exec_summary = _call_registry.get_execution_summary()
    exec_summary["elapsed"] = elapsed
    token_usage = backend.get_token_usage()
    if token_usage["total_tokens"] > 0:
        exec_summary["tokens"] = token_usage["total_tokens"]
    result["execution"] = exec_summary

    return json.dumps(result)


@mcp.tool()
def load_context(data: str, name: str | None = None) -> str:
    """Load input data into the sandbox. Available as `context` in Scheme and Python.

    Args:
        data: Text data to load (documents, code, CSV, JSON, etc.)
        name: Optional name for this context slot (e.g., "gwas-data", "expression").
              Use get-context to retrieve named contexts later.

    Named context slots (improvement #5) allow managing multiple datasets:
    - load_context(gwas_csv, "gwas-data")
    - load_context(expr_csv, "expression-data")
    - Later in Scheme: (get-context "gwas-data") or (get-context "expression-data")

    Strategy considerations after loading:
    - Data >100KB? → Use Pattern 1 (chunk via py-exec, parallel fan-out with map-async)
    - Unknown structure? → Use Pattern 2 (model inspects sample, generates analysis code)
    - Hierarchical? → Use Pattern 3 (recursive delegation to specialists)

    See get_usage_guide for strategy templates."""
    cmd = {"op": "load-context", "data": data}
    if name is not None:
        cmd["name"] = name
    resp = get_backend().send(cmd)
    if resp["status"] == "error":
        return f"[stderr] {resp['message']}"
    return resp.get("result", "context loaded")


@mcp.tool()
def get_scope_log() -> str:
    """Get the audit trail as JSON array. Each entry contains: op ('llm-query'|'syntax-e'|'datum->syntax'|'py-exec'|'py-eval'|'unsafe-*'), datum_preview (first 80 chars of data), and scope ('host'|'sandbox'|'sub-N'). Use to trace data flow and debug scope issues."""
    resp = get_backend().send({"op": "get-scope-log"})
    return resp.get("result", "[]")


@mcp.tool()
def reset() -> str:
    """Clear all sandbox state and start fresh. Call between unrelated tasks."""
    resp = get_backend().send({"op": "reset"})
    return resp.get("result", "sandbox reset")


@mcp.tool()
def get_status() -> str:
    """Get sandbox status: active calls, token usage, and rate limits. Non-blocking — safe to call any time."""
    backend = get_backend()
    return json.dumps({
        "active_calls": _call_registry.snapshot(),
        "token_usage": backend.get_token_usage(),
        "rate_limits": backend.get_rate_limits(),
    }, indent=2)


@mcp.tool()
def cancel_call(call_id: str) -> str:
    """Cancel an in-flight sub-model call by its ID. Use get_status to find call IDs. Cancels async futures and terminates nested REPLs for recursive calls. Returns immediately. Does not affect token accounting for already-completed work."""
    return get_backend().cancel_call(call_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
