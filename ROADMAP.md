# Roadmap

Implementation plans for upcoming features.

Items from code review and hands-on usage testing (comparing 13 academic PDFs
against their markdown conversions using the rlm-scheme tool — see
`papers/rlm-scheme-usage-report.md` for the full report).

---

# Short-term

Targeted fixes that are high-impact, bounded in scope, and don't require
architectural changes.

---

### Completed

- **S1. Retry with exponential backoff** — `_call_llm()` retries on 429 with
  configurable backoff, respects `Retry-After` header, checks `cancel_event`
  between retries. (13 tests)
- **S2. `await-all` / `await-all-syntax`** — batch result collection. (8 tests)
- **S3. `map-async` with `#:max-concurrent`** — parallel fan-out with optional
  batching. (9 tests)
- **S4. Rate limit awareness** — `(rate-limits)` binding exposes OpenAI
  `x-ratelimit-*` headers for proactive adaptation. (12 tests)
- **S5. Compact MCP progress reporting** — fraction-style progress
  (`3/7 done`), grouped model summaries, adaptive poll interval, 80-char cap.
  (15 tests)
- **S6. Simplify MCP tool surface** — consolidated 4 status tools into
  `get_status`, rewrote `execute_scheme` description to surface capabilities
  (LLM orchestration, Python/file I/O, web requests, vision). 11 → 8 tools.
  (18 tests)
- **S7. Enhanced progress reporting** — shared `CallRegistry` visible across
  all REPL depths (including recursive sandboxes), depth-aware call IDs
  (`call_d1_1`), richer `_format_progress_message` with depth annotations
  (`[d1:1]`), recursive type labels, instruction previews, progressive
  truncation. Event-driven monitor wake-up via `threading.Event` replaces
  polling sleep. (19 tests)
- **S8. Execution summary & non-blocking status** — `execute_scheme` result
  includes `execution` field with elapsed time, call count, models used,
  max latency, and token usage. `get_status` no longer calls Racket IPC
  (removed variable fetch), making it safe to call at any time. (9 tests)
- **L6. `map-async` timeout fix** — Changed default `#:max-concurrent` from `#f` to `10`, forcing all calls to use batching. Prevents unbounded async handle creation that caused 300s timeouts. (Tests require Racket)
- **L7. Disk-backed checkpoints** — Added `(checkpoint "key" value)` and `(restore "key")` bindings. Checkpoints stored as JSON in `.rlm-scheme-checkpoints/`, survive timeouts and reset. (14 tests)
- **L8. Better async error messages** — JSON mode validation requires "json" in instruction. Enhanced error logging with type info and status codes. Clear error messages for async failures. (17 tests, 3 passed without Racket)
- **L9. Document best practices** — Added comprehensive Best Practices section to README covering optimal batch sizes, JSON mode checklist, safe data transfer, cost control, and checkpointing. (No tests - documentation)
- **L10. Configurable timeout** — Added optional `timeout` parameter to `execute_scheme` and `RLM_TIMEOUT_SECONDS` env var. Warning at 80% elapsed. (15 tests, 4 passed without Racket)

---

# Long-term

Larger features that require design work, new subsystems, or research.

---

## L1. RLM environment integration

**Problem:** The upstream RLM library (in `rlm-upstream/`) uses `LocalREPL` and `ModalREPL` environment classes. Integrating RLM-Scheme as a new environment type enables direct benchmark comparison.

**Plan:**
1. Read `rlm-upstream/rlm/environment.py` to understand the `Environment` base class interface (likely `step(code) -> observation`, `reset()`, `close()`).
2. Create `rlm_environment.py` at project root — a new class `SchemeREPL(Environment)` that wraps `RacketREPL`:
   - `step(code)`: calls `send({"op": "eval", "code": code})`, formats the response as the RLM expects (the `observation` string).
   - `reset()`: calls `send({"op": "reset"})`.
   - `close()`: calls `self.repl.close()`.
   - `load_context(data)`: calls `send({"op": "load-context", "data": data})`.
3. Map the RLM's `FINAL(...)` convention to `(finish ...)`: the RLM loop checks for `FINAL` in the observation string; `execute_scheme` already returns `[finished] value`, so format this as `FINAL(value)` to match what the RLM loop expects.
4. Write a thin adapter in `rlm_environment.py` so the RLM agent loop can drop in `SchemeREPL` wherever it uses `LocalREPL`.
5. Test against one of the simpler RLM benchmarks (e.g., the arithmetic task from the paper) to verify the loop works end-to-end.

**Files:** New `rlm_environment.py`, references `rlm-upstream/rlm/environment.py`.

---

## L2. Static type checking

**Problem:** Scope violations (using a syntax object as a string, passing wrong-model prompts) are runtime errors. The monadic framing paper shows these could be compile-time errors.

**Plan:**

This is a research-grade feature. The approach is to type-check the LLM's generated code *before* evaluating it.

1. **Define a type grammar** for the sandbox's restricted language. The types are:
   - `String` — plain text
   - `Scoped<s>` — a syntax object tagged with provenance `s` (phantom type)
   - `Number`, `Boolean`, `List<T>`, `Void`
   - Function types for each scaffold binding (e.g., `llm-query : (...) -> Scoped<fresh>`, `syntax-e : Scoped<s> -> String`)

2. **Write a type checker** in Racket (or Python — Python is easier to iterate on). It takes an S-expression AST and the type environment (scaffold signatures), and walks the tree:
   - `(define x (llm-query ...))` -> `x : Scoped<s0>`
   - `(string-append x "...")` -> type error if `x : Scoped<s>`
   - `(syntax-e x)` -> `String`
   - `(finish (string-append (syntax-e x) "..."))` -> ok

3. **Integrate into eval dispatch.** Before `eval-top-level` runs user code (`racket_server.rkt`), pass the parsed expressions through the type checker. If it fails, return a type error without evaluating.

4. **Scope:** Start with a minimal subset — just track `Scoped` vs `String` to catch the most common mistake (using an `llm-query` result without `syntax-e`). Expand later to phantom provenance tags.

5. **Tests:** Programs that pass today but are semantically wrong (e.g., `(string-append (llm-query ...) "x")`) should now fail at the type-checking stage with a clear error message.

**Files:** New `type_checker.py` (or `type_checker.rkt`), modify `racket_server.rkt` `eval-top-level` to call it.

---

## L3. py-exec trailing newline fix and py-eval redesign

**Problem:** Python's `print()` appends `\n`, and `py-exec` returns raw stdout. This means `(string->number (py-exec "print(42)"))` returns `#f` because the input is `"42\n"`. Users must remember to strip, which is error-prone. More broadly, `py-exec`/`py-eval` data transfer relies on `print()` and `repr()` — both lossy for complex data.

**Plan:**
1. In the Scheme `py-exec` wrapper, strip trailing newlines from the return value (not all whitespace — just `\n` and `\r\n`).
2. Since raw multiline output sometimes matters, this should strip only *trailing* newlines, preserving internal structure.
3. Redesign `py-eval` for reliable structured data transfer: use JSON serialization as the default channel, with a fallback to `repr()` for non-serializable objects. This overlaps with the `py-set!` work already done.

**Files:** `racket_server.rkt` — `py-exec` definition (line ~520).

---

## L4. Native file I/O bindings

**Problem:** Every file read/write in the sandbox currently routes through `py-exec` with `print()`:

```scheme
(define content (py-exec "
with open('/path/to/file.md') as f:
    print(f.read()[:4000])
"))
```

This is verbose, fragile (content with quotes or backslashes needs escaping), and limited (`print()` truncates or mangles binary data). During the PDF-to-markdown task, 24 `py-exec` calls were file I/O — more than any other operation type.

**Plan:**
1. Add `(file-read path)` — returns file contents as a string. Optional `#:limit N` to read only the first N characters (for large files).
2. Add `(file-write path content)` — writes string content to a file. Returns the number of bytes written.
3. Both should be logged in the scope log (like `py-exec` calls are).
4. These don't expand the security surface beyond what `py-exec` already allows — they're just ergonomic wrappers.
5. Design consideration: should file I/O go through Racket (native ports) or Python (subprocess)? Racket is faster but adds complexity to the sandbox; Python reuses existing infrastructure.

**Files:** `racket_server.rkt` — new bindings in the sandbox. `mcp_server.py` — handler for file ops if they route through Python.

---

## L5. Vision-aware model routing

**Problem:** Every `llm-query` call with `#:image` or `#:images` required an explicit `#:model "gpt-4o"` because the default sub-model may not support vision. During the PDF comparison task, forgetting `#:model` on a vision call would silently fail or produce a non-vision response.

**Plan:**
1. If `#:image` or `#:images` is provided and no `#:model` is specified, auto-select a vision-capable model from a configured preference list.
2. Maintain a `VISION_MODELS` list (e.g., `["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-5-20250929"]`) in config.
3. If the explicitly-specified model doesn't support vision and images are present, emit a warning or error rather than silently dropping the images.
4. Requires a model capability registry — which models support vision, JSON mode, tool use, etc. This is useful beyond just vision routing.

**Files:** `mcp_server.py` — `_call_llm()`, model selection logic. New config for model capabilities.

---
