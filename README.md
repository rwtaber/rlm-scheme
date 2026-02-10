# RLM-Scheme

An alternative scaffold for the [Recursive Language Model](https://github.com/alexzhang13/rlm) architecture. Replaces the Python REPL with a Scheme sandbox that adds parallel sub-calls, multi-model routing, token budgets, generation control, structured JSON output, and recursive delegation. Runs as an MCP server so Claude Code can use it directly.

## Quick Start: Choose Your Orchestration Strategy

RLM-Scheme provides **6 orchestration strategies** for different problems. Choose based on your data and goals:

| Your Problem | Use This Strategy | Example Below |
|--------------|-------------------|---------------|
| **Data too large for one call** | Strategy 1: Parallel Fan-Out | ✓ See Example 1 |
| **Unknown data structure** | Strategy 2: Code Generation | ✓ See Example 2 |
| **Hierarchical data** | Strategy 3: Recursive Delegation | See [EXAMPLES.md](EXAMPLES.md) |
| **Quality uncertain** | Strategy 4: Critique-Refine Loop | ✓ See Example 2 |
| **Many perspectives** | Strategy 5: Cumulative Fold | See [EXAMPLES.md](EXAMPLES.md) |
| **Complex multi-phase** | Strategy 6: Meta-Orchestration | See [EXAMPLES.md](EXAMPLES.md) |

Call `get_usage_guide` for templates and detailed documentation for all 6 strategies.

## Example 1: Parallel Fan-Out (Strategy 1)

**Problem:** A 200-page report is too large for a single LLM call.

**Strategy:** Chunk in Python, summarize chunks in parallel with cheap model, synthesize with expensive model.

A 200-page report is loaded via `load_context`. This pipeline chunks it in Python, summarizes each chunk in parallel under a token budget, computes statistics, and synthesizes an executive summary. `py-set!` transfers data safely between Scheme and Python (no string escaping needed).

```scheme
;; 1. Chunk in Python — too large for a single sub-model call.
(define chunk-list (py-eval "[context[i:i+3000] for i in range(0, len(context), 3000)]"))

;; 2. Summarize each chunk in parallel under a token budget.
;;    Use map-async for efficient batching (10 chunks at a time).
(parameterize ([token-budget 15000])
  (define summaries (map-async
    (lambda (chunk)
      (llm-query-async
        #:instruction "Summarize this section in 2-3 sentences."
        #:data chunk
        #:model "gpt-4.1-nano"  ; Always use cheapest model for fan-out!
        #:temperature 0.3
        #:max-tokens 200))
    chunk-list
    #:max-concurrent 10))

  ;; 3. Transfer results to Python safely with py-set!.
  (py-set! "chunks" chunk-list)
  (define stats (py-exec "
import json
print(json.dumps({
  'num_chunks': len(chunks),
  'total_chars': sum(len(c) for c in chunks),
  'avg_chunk_len': sum(len(c) for c in chunks) // max(len(chunks), 1)
}))
"))

  ;; 4. Synthesize the independently-produced summaries.
  (py-set! "summaries" summaries)
  (define combined (py-exec "print('\\n---\\n'.join(summaries))"))
  (define final (syntax-e (llm-query
    #:instruction "Combine these section summaries into a coherent executive summary."
    #:data combined
    #:temperature 0.5)))

  ;; 5. Check token usage.
  (define usage (tokens-used))
  (finish (string-append final "\n\nStats: " stats "\nTokens: " usage)))
```

**Key pattern:** Cheap model for fan-out (`gpt-4.1-nano` @ $0.10/1M tokens), expensive model for synthesis (`gpt-4o` @ $2.50/1M tokens). This costs 10-20× less than using gpt-4o for everything.

This example demonstrates **Strategy 1** (parallel fan-out) with efficient batching (`map-async`), token budgets (`parameterize`/`tokens-used`), generation control (`#:temperature`, `#:max-tokens`), injection safety (syntax objects), safe data transfer (`py-set!`), and hybrid compute (`py-exec`/`py-eval`). After running, `get_scope_log` returns the full audit trail of every sub-model call.

## Example 2: Code Generation + Validation (Strategies 2 & 4)

**Problem:** Unknown data structure — need adaptive analysis approach.

**Strategy:** Let a model write the analysis code (Strategy 2), then validate and refine if needed (Strategy 4).

A more sophisticated pattern: generate analysis code, execute it, validate results, and recursively refine if needed.

```scheme
;; Step 1: Model writes its own analysis strategy based on data structure
(define sample (py-exec "print(context[:500])"))  ; Show data sample
(define analysis-code (unsafe-raw-query
  #:instruction "Write Scheme code that analyzes this data. Use py-exec for parsing, llm-query for insights. Define variable `analysis_result`. No (finish ...)."
  #:data sample
  #:temperature 0.0))

;; Step 2: Execute the generated strategy
(unsafe-exec-sub-output (datum->syntax #f analysis-code))

;; Step 3: Validate with a second model
(define validation (syntax-e
  (llm-query
    #:instruction "Check if this analysis has specific evidence and no hallucinations. Return JSON: {valid: bool, issues: [strings]}"
    #:data (py-eval "str(analysis_result)")
    #:json #t
    #:temperature 0.0)))

;; Step 4: Refine if invalid
(py-set! "v" validation)
(define is-valid (py-eval "import json; json.loads(v)['valid']"))
(define final-result
  (if is-valid
      (py-eval "analysis_result")
      (syntax-e  ; Recursive refinement with validation constraints
        (llm-query
          #:instruction (string-append "Revise this analysis addressing: " validation)
          #:data context
          #:recursive #t))))

(finish final-result)
```

**Key pattern:** **Strategy 2** (Code Generation) lets the analysis adapt to unknown data structures. **Strategy 4** (Critique-Refine Loop) ensures quality through validation and recursive refinement. The model writes its own strategy, then self-corrects.

**Important:** When using Strategy 2, call the MCP tool `get_code_generation_api_reference` and include its output in your prompt. Sub-models don't automatically know the rlm-scheme API syntax and will generate broken code without it.

See [EXAMPLES.md](EXAMPLES.md) for more patterns including Strategy 3 (Recursive Delegation), Strategy 5 (Cumulative Fold), and Strategy 6 (Meta-Orchestration).

## What this is

The [Recursive Language Model](https://github.com/alexzhang13/rlm) (Zhang et al. 2026) gives LLMs a REPL: instead of answering directly, the model writes code that loads data, makes sub-LLM calls, and builds up an answer programmatically. This lets it handle inputs larger than the context window and decompose complex problems into smaller pieces.

RLM-Scheme replaces the Python REPL with a Scheme sandbox. The model writes the same kind of orchestration code — load data, call sub-models, combine results, return an answer — but the runtime is more capable and more reliable.

### Capabilities

The original scaffold supports one model, one call at a time, at one level of depth, with no control over generation parameters.

| Feature | Original RLM | RLM-Scheme |
|---------|-------------|------------|
| Sub-model calls | `llm_query()`, sequential only | `llm-query`, `llm-query-async` with parallel fan-out |
| Model routing | Single model | Per-call `#:model` override |
| Generation control | None | Per-call `#:temperature` and `#:max-tokens` |
| Structured output | None | `#:json #t` for guaranteed valid JSON responses |
| Vision / images | None | `#:image` and `#:images` for multimodal sub-calls |
| Token control | None | `parameterize` scoped budgets with real API counts |
| Recursion depth | 1 (model writes code, sub-models answer) | Up to 3 (sub-models get their own sandboxes) |
| Computation | Python only | Scheme for orchestration + Python bridge for data |
| Audit trail | None | Full scope log of every call and crossing |
| Data transfer | N/A | `py-set!` for safe Scheme→Python transfer |
| Standard library | N/A | `racket/list`, `racket/string` in sandbox |
| Call visibility | None | Live call registry, stderr logging, cancel |
| Crash handling | None | Auto-restart with 60s timeout |

### Reliability

The original Python REPL has four failure modes that RLM-Scheme prevents:

- **Premature completion.** The Python scaffold uses regex to detect `FINAL("answer")` in the output. If the model's reasoning text mentions "FINAL", the scaffold captures it early — this happened in 29% of training turns. RLM-Scheme uses a real function call (`(finish value)`) — the word "finish" in a string does nothing.

- **Self-sabotage.** The Python scaffold shares one mutable namespace. The model can write `context = "oops"` and destroy its own input. RLM-Scheme protects all scaffold bindings — trying to redefine `context`, `finish`, or `llm-query` raises an error.

- **Prompt injection via sub-model responses.** In Python, sub-model responses are plain strings spliced into the next prompt. If the response contains "Ignore above instructions", it can hijack the pipeline. RLM-Scheme wraps every response in an opaque syntax object. The model must explicitly unwrap with `(syntax-e response)` before using it as text.

- **Silent model-routing bugs.** A prompt crafted for one model may behave differently on another. Every scope crossing is logged in an audit trail so you can trace exactly what data went where.

## Setup

### Prerequisites

- **Racket 8.x+**
- **Python 3.12+**
- **An OpenAI API key** — sub-model calls go through the OpenAI API

#### Installing Racket

| OS | Command |
|----|---------|
| **Linux (Debian/Ubuntu)** | `sudo apt install racket` |
| **macOS** | `brew install --cask racket` |
| **Windows** | `winget install Racket.Racket` |

Or download from [racket-lang.org/download](https://racket-lang.org/download/). Make sure `racket` is on your PATH — verify with `racket --version`.

> **Windows note:** The winget/installer may not add Racket to your PATH automatically. If `racket --version` doesn't work, add the install directory (typically `C:\Program Files\Racket`) to your system or user PATH.

### Install

```bash
git clone https://github.com/rwtaber/rlm-scheme.git
cd rlm-scheme
python -m venv .venv
```

Activate the virtual environment:

| OS | Command |
|----|---------|
| **Linux / macOS** | `source .venv/bin/activate` |
| **Windows (PowerShell)** | `.venv\Scripts\Activate.ps1` |
| **Windows (cmd)** | `.venv\Scripts\activate.bat` |

Then install dependencies:

```bash
pip install "mcp[cli]>=1.2.0" openai python-dotenv
```

### API key

Create a `.env` file in the project root (already in `.gitignore`):

```
OPENAI_API_KEY=sk-your-key-here
```

### Configure Claude Code

Copy `.mcp.json` to the project where you want to use it, updating the paths to match your system.

**Linux / macOS:**

```json
{
  "mcpServers": {
    "rlm-scheme": {
      "command": "/path/to/rlm-scheme/.venv/bin/python",
      "args": ["/path/to/rlm-scheme/mcp_server.py"],
      "cwd": "/path/to/rlm-scheme",
      "env": {
        "RLM_SUB_MODEL": "gpt-4o"
      }
    }
  }
}
```

**Windows:**

```json
{
  "mcpServers": {
    "rlm-scheme": {
      "command": "C:\\path\\to\\rlm-scheme\\.venv\\Scripts\\python.exe",
      "args": ["C:\\path\\to\\rlm-scheme\\mcp_server.py"],
      "cwd": "C:\\path\\to\\rlm-scheme",
      "env": {
        "RLM_SUB_MODEL": "gpt-4o"
      }
    }
  }
}
```

### Environment Variables

- **`RLM_SUB_MODEL`** — Default model for sub-calls (default: `gpt-4o`). Individual calls can override with `#:model`.
- **`RLM_TIMEOUT_SECONDS`** — Default timeout for `execute_scheme` in seconds (default: 300).
- **`RLM_MAX_WORKERS`** — Maximum concurrent async LLM calls (default: 10). Increase to 20-50 for bulk processing with cheap models like gpt-4.1-nano.
- **`RLM_PYTHON`** — Path to Python interpreter for py-exec/py-eval (default: auto-detected from venv).

### Verify

Activate the virtual environment (see above), then run:

```bash
pytest tests/
```

171 tests, all passing.

## More examples

### Classify and fix code issues

Multi-model routing: parallel fan-out review, cheap model for structured JSON classification, expensive model for the fix. Uses `map-async` for efficient parallel processing.

```scheme
;; 1. Fan-out: review code from multiple angles in parallel.
(define aspects (list "security vulnerabilities" "error handling gaps" "performance bottlenecks"))
(define reviews (map-async
  (lambda (aspect)
    (llm-query-async
      #:instruction (string-append "Analyze this code for " aspect ". List specific issues.")
      #:data context
      #:model "gpt-4o"))
  aspects))  ; Only 3 items, launch all at once

;; 2. Combine reviews in Python using py-set! (no format ~a needed).
(py-set! "aspects" aspects)
(py-set! "reviews" reviews)
(define combined (py-exec "
print('\\n\\n'.join(f'## {a}\\n{r}' for a, r in zip(aspects, reviews)))
"))

;; 3. Classify severity — cheap model, deterministic, structured JSON.
(define severity-json (syntax-e (llm-query
  #:instruction "Classify the most critical issue. Return JSON: {\"category\": \"security\"|\"bug\"|\"style\", \"summary\": \"one sentence\"}"
  #:data combined
  #:model "gpt-4o-mini"
  #:json #t
  #:temperature 0.0
  #:max-tokens 200)))

;; 4. Parse the JSON in Python and generate a targeted fix.
(py-set! "severity" severity-json)
(define category (py-exec "import json; print(json.loads(severity)['category'])"))
(define fix (syntax-e (llm-query
  #:instruction (string-append
    "The most critical issue is a " category " problem. "
    "Write a minimal fix. Return only the corrected code.")
  #:data context
  #:model "gpt-4o"
  #:temperature 0.0)))

(finish (string-append "## Classification\n" severity-json "\n\n## Fix\n" fix))
```

### Recursive delegation

Each sub-model gets its own Scheme sandbox via `#:recursive #t` and decides its own analysis strategy. The top level says *what* it wants, not *how* to do it.

```scheme
(define doc-a (py-exec "print(context.split('===SEPARATOR===')[0])"))
(define doc-b (py-exec "print(context.split('===SEPARATOR===')[1])"))

;; Each sub-model gets its own sandbox and decides its own strategy.
(define analysis-a (syntax-e (llm-query
  #:instruction "Analyze this document: main claims, methodology, findings, limitations. Chunk if needed."
  #:data doc-a
  #:recursive #t)))

(define analysis-b (syntax-e (llm-query
  #:instruction "Analyze this document: main claims, methodology, findings, limitations. Chunk if needed."
  #:data doc-b
  #:recursive #t)))

;; Compare the independently-produced analyses.
(define comparison (syntax-e (llm-query
  #:instruction "Compare these analyses. Where do they agree? Contradict? What gaps exist?"
  #:data (string-append "## Document A\n" analysis-a "\n\n## Document B\n" analysis-b))))

(finish comparison)
```

### Escape hatches — code generation and overrides

A sub-model writes Scheme code, then you execute it — a deliberate scope crossing. Shows every escape hatch.

```scheme
;; 1. Ask a sub-model to write code. Returns a plain string (no syntax wrapper).
(define generated-code (unsafe-raw-query
  #:instruction "Write Scheme code that defines `result` as the first 10 lines of context."
  #:data (substring context 0 200)
  #:temperature 0.0))

;; 2. Execute the generated code — a deliberate scope break.
(unsafe-exec-sub-output (datum->syntax #f generated-code))

;; 3. Other escape hatches:
;;    unsafe-interpolate: strip scope marks without logging as syntax-e.
(define raw (unsafe-interpolate (llm-query #:instruction "Say hello")))
;;    unsafe-overwrite: replace a protected binding mid-pipeline.
(unsafe-overwrite 'context "replacement data for next step")

;; 4. Return the variable defined by the generated code.
(finish-var "result")
```

## How it works

```
Claude Code  --JSON-RPC/stdio-->  mcp_server.py  --JSON/stdin-->  racket_server.rkt
                                                                     |
                                                                     +--JSON/stdin-->  py_bridge.py
```

**Claude Code** writes Scheme code and sends it to the MCP server via tool calls.

**`mcp_server.py`** is the entry point. It exposes 8 MCP tools over JSON-RPC and manages the Racket subprocess. When Scheme code calls `llm-query`, the Racket process sends a callback request back to Python, which calls the OpenAI API and returns the result. This callback loop is the core of the architecture — real API calls happen in Python while orchestration logic runs in the sandbox. All in-flight sub-model calls are tracked in a thread-safe call registry with unique IDs, and structured log lines are emitted to stderr for diagnostics.

**`racket_server.rkt`** is the sandbox. It creates a restricted Scheme evaluator with memory limits (256 MB), CPU timeout (30s), and no filesystem/network access. The scaffold bindings (`llm-query`, `finish`, `context`, etc.) are injected as host-side closures that can communicate with the MCP server but can't be redefined by user code.

**`py_bridge.py`** handles computation. When Scheme code calls `(py-exec "...")`, the request goes to an isolated Python subprocess with full stdlib access but no access to the sandbox.

### Data flow for a sub-model call

1. Claude Code sends Scheme code: `(finish (syntax-e (llm-query #:instruction "Summarize" #:data context)))`
2. `mcp_server.py` forwards to the Racket process
3. Racket evaluates the code, hits `llm-query`, writes a callback to stdout: `{"op":"llm-query","instruction":"Summarize","data":"..."}`
4. `mcp_server.py` reads the callback, calls `openai.chat.completions.create()`
5. Response goes back to Racket with token counts: `{"result":"Summary text...","prompt_tokens":150,"completion_tokens":42}`
6. Racket wraps the result, `syntax-e` unwraps it, `finish` returns: `{"status":"finished","result":"Summary text..."}`
7. Claude Code sees `[finished] Summary text...`

## Scaffold reference

The complete scaffold reference with all 24 bindings, model selection guidance, and 8 progressive examples is available via the `get_usage_guide` MCP tool. Call it before writing Scheme code.

**Quick reference** — Most commonly used bindings:
- `(finish value)` — Return result and halt
- `(llm-query #:instruction "..." #:data "...")` — Call sub-model (returns syntax object)
- `(syntax-e stx)` — Unwrap syntax object to string
- `(llm-query-async ...)` and `(await handle)` — Async sub-calls
- `(py-exec "code")` — Run Python code
- `(py-set! "name" value)` — Safe Scheme→Python transfer
- `(checkpoint "key" value)` and `(restore "key")` — Persistent storage
- `context` — Data loaded via `load_context`

See `get_usage_guide` for:
- All bindings with signatures and examples
- Model selection table (costs, context limits, best use cases)
- Parameter guidance (`#:temperature`, `#:json`, `#:image`, `#:recursive`)
- 8 complete examples from simple to advanced
- Best practices and rules

## MCP tools

Seven tools exposed to Claude Code:

| Tool | What it does |
|------|-------------|
| `get_usage_guide` | Returns the complete Scheme reference with examples. Call first. |
| `execute_scheme(code, timeout?)` | Run Scheme code in the sandbox. State persists across calls. Optional timeout parameter. |
| `load_context(data)` | Load input data as the `context` variable. |
| `get_scope_log` | Get the audit trail of all sub-model calls as JSON array. |
| `get_status` | Get current status: active calls (with IDs, models, depth, elapsed time), cumulative token usage, and API rate limits. |
| `cancel_call(call_id)` | Cancel an in-flight call by ID. Cancels async futures and kills nested REPLs. |
| `reset` | Clear all state. Call between unrelated tasks. |

**Note:** `get_status` consolidates monitoring functions — it returns active calls, token usage, and rate limits in one call.

## Best Practices

Based on real-world usage (processing 1,467 papers with 277 LLM calls in the AMR drug repurposing PoC):

### Parallel Orchestration

**Optimal batch size:** 8-10 concurrent calls per batch. `map-async` now defaults to 10.

```scheme
;; GOOD: Uses default batching (10 concurrent)
(define results (map-async
  (lambda (item) (llm-query-async #:instruction "Analyze" #:data item ...))
  items))

;; ALSO GOOD: Explicit batch size for finer control
(define results (map-async
  (lambda (item) (llm-query-async #:instruction "Analyze" #:data item ...))
  items
  #:max-concurrent 8))
```

**Pipeline large workloads:** For >50 items, split across multiple `execute_scheme` calls. Each call has a 300-second timeout. Process 40-50 items per call to stay well under the limit.

### JSON Mode Checklist

When using `#:json #t`:
1. ✅ Include the word "json" in the `#:instruction` string (OpenAI API requirement)
2. ✅ Use `#:temperature 0.0` for deterministic output (except o-series models)
3. ✅ Set reasonable `#:max-tokens` (100-300 for structured data)

```scheme
;; CORRECT
(define result (syntax-e
  (llm-query
    #:instruction "Extract keywords. Return JSON: {keywords: [strings]}"
    #:data context
    #:json #t
    #:temperature 0.0
    #:max-tokens 200)))
```

### Safe Data Transfer with py-set!

**Always use `py-set!` for LLM output → Python transfer.** Never embed LLM results in `py-exec` code strings.

```scheme
;; GOOD: py-set! handles all escaping automatically
(define text (syntax-e (llm-query #:instruction "Write a poem")))
(py-set! "poem" text)
(define word-count (py-exec "print(len(poem.split()))"))

;; BAD: Breaks on quotes, backslashes, unicode
;; (py-exec (string-append "poem = '" text "'"))  ;; DON'T DO THIS
```

### Cost Control

**Default to the cheapest model.** Fan-out over N chunks costs N×.

```scheme
;; Classification/extraction: gpt-4.1-nano ($0.10/1M input tokens)
(define categories (map-async
  (lambda (chunk)
    (llm-query-async
      #:instruction "Classify: science|tech|business|other. One word."
      #:data chunk
      #:model "gpt-4.1-nano"
      #:temperature 0.0
      #:max-tokens 10))
  chunks))

;; Complex reasoning: gpt-4o or gpt-4.1
(define synthesis (syntax-e
  (llm-query
    #:instruction "Synthesize these classifications into insights."
    #:data (py-eval "str(categories)")
    #:model "gpt-4o")))
```

### State Persistence

State persists across `execute_scheme` calls, but timeouts destroy it.

**For long pipelines:** Save intermediate results to disk via `py-exec`:

```scheme
;; After each major phase, checkpoint to disk
(py-set! "results" categorization-results)
(py-exec "
import json
with open('checkpoint.json', 'w') as f:
    json.dump(results, f)
print(f'Saved {len(results)} results')
")
```

### Rate Limit Awareness

Use `(rate-limits)` to adapt strategy proactively:

```scheme
(define rl (rate-limits))
(py-set! "rl" rl)
(define remaining-pct (py-eval "
int(100 * rl['remaining_tokens'] / max(rl['limit_tokens'], 1))
"))

(define model (if (> remaining-pct 50)
                  "gpt-4.1-nano"  ; Plenty of quota: use cheap model
                  "gpt-4o-mini"))  ; Low quota: consolidate calls
```

### Debugging

**Check stderr for diagnostics:**
```bash
# Racket server logs every call and completion
[rlm] call_1: calling gpt-4o-mini (llm-query, 150 chars, depth 0)...
[rlm] call_1: completed (523 tokens, 1.2s)
```

**Get the full audit trail:**
```python
# After execute_scheme
log = get_scope_log()  # Every sub-call, every scope crossing
```

## Files

| File | Language | Lines | Role |
|------|----------|-------|------|
| `racket_server.rkt` | Racket | 824 | Sandboxed Scheme REPL |
| `mcp_server.py` | Python | 1122 | MCP server + OpenAI API bridge + call registry |
| `py_bridge.py` | Python | 125 | Isolated Python subprocess |
| `tests/` | Python | 6 files | 211 tests |
| `.mcp.json` | JSON | 12 | Claude Code config |

## Tests

211 tests across 6 files. Run with `pytest tests/`.

- **test_racket_server.py** (72) — core eval, scaffold protection, parameterize, escape hatches, py-bridge, py-set!, stdout capture, non-void top-level, list/string functions
- **test_hygiene.py** (26) — adversarial tests: premature completion, namespace collision, prompt injection resistance, scope tracking
- **test_mcp_server.py** (65) — timeout, crash recovery, process lifecycle, call registry, cancel call, stderr logging, thread safety, image resolution, token tracking, progress notifications
- **test_callbacks.py** (14) — token tracking, multi-model dispatch, async protocol
- **test_recursion.py** (17) — recursive flag forwarding, code extraction, nested sandbox lifecycle, depth limits
- **test_api_params.py** (17) — temperature, max-tokens, JSON mode, image forwarding, combined parameters

## References

- Zhang et al. (2026). [Recursive Language Models.](https://github.com/alexzhang13/rlm) The architecture this builds on.
- Kohlbecker et al. (1986). *Hygienic Macro Expansion.* The scope discipline adapted for LLM pipelines.
- `papers/` contains the source PDFs and analysis notes.

## License

MIT
