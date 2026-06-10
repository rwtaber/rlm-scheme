# RLM-Scheme Greenfield Rewrite Plan — v2

**Status:** Normative specification. Supersedes `docs/GREENFIELD-REWRITE-PLAN.md` (v1) in full.
**Implementation root:** `v3/` (the existing `v2/` tree is the v1-plan implementation and is left untouched).
**Audience:** an implementing agent. This document is self-contained; no other document is required.

---

## 0. Changes from v1 (decision record)

| # | v1 problem | v2 decision |
|---|---|---|
| A | Racket layer was vestigial: `.rkt` templates were pattern-matched by a Python "runtime stub"; Scheme was never evaluated | **Real Racket runtime.** Template bodies are evaluated by a sandboxed Racket subprocess. Python hosts the MCP server, store, planner, cache, and LLM providers; Racket executes orchestration programs and calls back over a JSON-lines wire protocol. The Python pattern-matching interpreter is deleted from the design. |
| B | Injection guard blacklisted English words (`define`, `system`, `load`...) in instruction slots | **Removed.** There is no textual slot substitution anywhere in v2. Slot values are delivered to the runtime as a JSON data binding and read via `(slot 'name)`. Code and data never mix, so the injection class does not exist. No blacklist, no paren-balance check on prose. |
| C | `py-exec` claimed "isolated subprocess" but was in-process `exec` with an `__import__` shim | **Real subprocess.** Each `py-exec` call runs in a fresh `python -I -S` OS process with `resource.setrlimit` CPU/memory caps and host-side kill-on-timeout. The import allowlist is enforced inside the child as defense-in-depth, but the security boundary is the process. Limitations (no network namespace isolation) are stated honestly in Appendix B. |
| D | Cost estimates were hardcoded per template-name string | **Estimates by construction.** A dry run executes the actual template body in the Racket runtime in `simulate` mode with a counting stub provider. Call counts, concurrency, critical path, and recursion depth are measured, not predicted. Token/cost figures derive from measured calls × the model registry. `structural-profile` metadata is deleted. |
| E | MockLLMProvider returned `""`/`"{}"` — tests could pass without exercising data flow | **Content-bearing mock.** The test provider returns deterministic, distinguishable values derived from its inputs (Appendix C.4). Orchestration tests MUST assert on flowed content, not just completion. |
| F | Batch 7 wiring referenced classes and constructors that did not exist; response shapes drifted between batches | **Single source of truth.** Appendix I defines every constructor signature, public method, and MCP response shape. Batches reference Appendix I and MUST NOT restate shapes. |
| G | ~15 small self-contradictions (enum counts, ID prefixes, missing store namespace, schema-format conflicts, invalid example IDs) | **All reconciled** in Appendix A and Appendix I. `ResetScope` has exactly 7 values and `Store.reset` handles all 7. There are exactly 8 ID prefixes and 8 store namespaces (including `checkpoints`). Output schemas are JSON Schema (subset) carried as JSON strings — the alist schema notation is deleted. All example IDs in this document validate against the ID grammar. `confidence: 1.0` is replaced by `hints_complete: bool` + `assumed_fields: list`. |

Also removed relative to v1:

- **`syntax-e`/`datum->syntax` provenance machinery.** There is no real macro-hygiene mechanism behind it. Replaced by a per-call provenance entry in the execution trace (every LLM result is recorded with `call_id`, `node_id`, model, and a preview). Template authors treat all LLM output as untrusted data by default.
- **Alist output-schema notation** (Appendix D.4 of v1). One schema language: JSON Schema subset.

Retained from v1 (unchanged in spirit): the four-stage lifecycle, the 10-tool MCP surface, the 13/11 task/data-shape taxonomies, the Q0–Q9 classifier, the 16-template catalog, content-addressed LLM caching, gates, checkpoints, budget degradation, streaming, template chains.

---

## 1. Document conventions and precedence

1. **RFC-2119:** `[MUST]`, `[SHOULD]`, `[MAY]` carry their RFC-2119 meanings.
2. **Exact cardinalities are normative.** "Exactly 10 tools", "exactly 8 prefixes", etc. are testable claims; tests assert them.
3. **Precedence on conflict.** If any two statements in this document disagree, the order of authority is:
   1. Appendix A (taxonomies, IDs) and Appendix I (interfaces) — highest;
   2. Batch `Requirements` sections;
   3. Test listings inside batches;
   4. Prose, examples, and walkthroughs — lowest.
   An implementing agent that finds a conflict **[MUST]** record it in `v3/SPEC-DEVIATIONS.md` (one line: location of both statements, which won by precedence) rather than silently choosing. The build does not stop for conflicts; the log makes them auditable.
4. **Tests are derived from requirements**, never the reverse. A test that contradicts Appendix A or I is a spec bug, subject to rule 3.
5. All JSON examples use real values that validate against this spec (IDs match the grammar in A.6, enums match A.1–A.5).

---

## 2. System overview

RLM-Scheme is an MCP server for structured LLM orchestration. Agents never write Scheme. They describe tasks; the server classifies the task deterministically, selects a pre-audited orchestration template, simulates it, verifies it against policy, and executes it.

### 2.1 Two-process architecture

```
┌────────────────────────────  Python host (v3/rlm_scheme)  ───────────────────────────┐
│                                                                                      │
│  MCP server (FastMCP, exactly 10 tools)                                              │
│   ├─ ContextStore / Store (filesystem JSON, 8 namespaces)                            │
│   ├─ Classifier + Planner (deterministic trees, optional LLM gap-fill)               │
│   ├─ Instantiator (slot validation + canonicalization + hashing — NO substitution)   │
│   ├─ VerificationEngine (exactly 23 checks)                                          │
│   ├─ LLM providers + content-addressed cache + BudgetMonitor                         │
│   ├─ GateManager / CheckpointManager / StreamingNotifier / ChainExecutor             │
│   └─ PyExecSandbox (OS subprocess per call)                                          │
│                              │  JSON-lines over stdin/stdout (Appendix J)            │
└──────────────────────────────┼───────────────────────────────────────────────────────┘
                               │  one subprocess per run
┌──────────────────────────────▼──────────────  Racket runtime (v3/runtime)  ──────────┐
│  Trusted shim: wire I/O, primitives library (rlm/primitives), mode switch            │
│  Sandbox (racket/sandbox): evaluates the template body; only primitives are in scope │
│  Modes: simulate (stub provider, counts everything) | live (bridges calls to host)   │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

Division of responsibility:

- **Python owns all state and all I/O to the outside world**: storage, LLM API calls, cache, budget accounting, gates, checkpoints, MCP notifications, py-exec subprocesses.
- **Racket owns control flow**: it evaluates the template body — the real combinator structure (`map-async`, `tree-reduce`, `fold-sequential`, ...) — and emits requests (`llm_call`, `py_exec`, `checkpoint`, `gate_wait`, `partial_result`) to the host over the wire.
- **The same body runs in both modes.** `simulate` mode answers every `llm_call` instantly with a synthetic value and counts calls/concurrency/depth; `live` mode forwards to real providers. Dry-run numbers are therefore exact by construction for the given input.

### 2.2 Lifecycle and ID chain

```
load_context        → ctx_3f9a17b2c4d5e6f0
  plan_strategy     → plan_8a1b2c3d4e5f6071   (classification + template invocation or chain)
    dry_run_strategy→ dry_42d1e0f9a8b7c6d5    (creates art_… via simulated execution)
      execute_strategy → exec_19c8d7e6f5a4b3c2  (creates ver_…; real LLM calls happen here)
```

The happy path is exactly three tool calls after loading context. Artifact (`art_`) and verification (`ver_`) IDs are created internally and surfaced in responses for audit; agents never pass them between tools.

### 2.3 Key invariants

- **I-1 [MUST]** No template body is ever modified by slot values. The artifact = (template body, canonical slot values); its ID is a content hash (A.6).
- **I-2 [MUST]** No real LLM call occurs before a verification record with `decision: "pass"` exists for the artifact (Appendix E).
- **I-3 [MUST]** Dry-run statistics come from simulated execution of the same body that live execution runs.
- **I-4 [MUST]** The sandbox denies the template body all filesystem, network, and FFI access; the only effects available are the primitives exported by the trusted shim.
- **I-5 [MUST]** Every LLM call is recorded in the trace with `call_id`, `node_id`, model, token counts, and cache status.
- **I-6 [MUST]** The host enforces policy limits (calls, concurrency, tokens, cost, depth) at request time, independent of what the runtime does.

### 2.4 MCP tool surface (exactly 10 tools)

| # | Tool | Purpose |
|---|---|---|
| 1 | `load_context` | Store input data + structural metadata → `ctx_` |
| 2 | `plan_strategy` | Classify task, select template(s), fill slots → `plan_` |
| 3 | `dry_run_strategy` | Instantiate artifact(s) + simulated execution + cost estimate → `dry_` |
| 4 | `execute_strategy` | Verify + live execution → `exec_` |
| 5 | `resume_execution` | Resume from a gate decision or a checkpoint |
| 6 | `get_execution_trace` | Full event trace for an execution |
| 7 | `list_templates` | Catalog summary (Appendix G) |
| 8 | `describe_template` | One template's metadata, slots, and body |
| 9 | `get_record` | Fetch any stored record by ID (prefix-dispatched) |
| 10 | `reset` | Clear state by `ResetScope` |

Request/response shapes: Appendix I.3 (normative).

---
## Appendix A: Taxonomies and identifiers (normative)

### A.1 TaskShape — exactly 13 values

`Direct`, `Batch`, `Synthesize`, `Search`, `Refine`, `Compare`, `Classify`, `Pipeline`, `Generate`, `Decompose`, `Validate`, `Aggregate`, `Composite`.

### A.2 DataShape — exactly 11 values

`FlatList`, `Hierarchy`, `Singular`, `ChunkedSingular`, `Graph`, `TimeSeries`, `Tabular`, `Multimodal`, `Paired`, `KeyValue`, `Unknown`.

`Unknown` is the explicit fallback when metadata is insufficient; it is never silently coerced to `Singular`.

### A.3 ExecutionState — exactly 7 values, with transition table

`pending`, `verifying`, `running`, `suspended_gate`, `finished`, `failed`, `cancelled`.

| From \ To | verifying | running | suspended_gate | finished | failed | cancelled |
|---|---|---|---|---|---|---|
| `pending` | ✓ | — | — | — | ✓ | ✓ |
| `verifying` | — | ✓ | — | — | ✓ | ✓ |
| `running` | — | — | ✓ | ✓ | ✓ | ✓ |
| `suspended_gate` | — | ✓ | — | — | ✓ | ✓ |

`finished`, `failed`, `cancelled` are terminal. Any transition not marked ✓ **[MUST]** raise `InvalidStateTransition`.

### A.4 ErrorPolicy — exactly 3 values

`fail_fast` (first item error aborts the run), `skip_and_log` (item error → `null` result + trace event), `retry_then_skip` (per-call retry per C.3, then skip_and_log).

### A.5 ResetScope — exactly 7 values

| Scope | Clears |
|---|---|
| `contexts` | `contexts` namespace |
| `plans` | `plans` namespace |
| `executions` | `executions`, `dry_runs`, `verifications`, `checkpoints` namespaces |
| `artifacts` | `artifacts` namespace |
| `cache` | `cache` namespace |
| `gates` | pending gate records (GateManager state) |
| `all` | everything above |

`Store.reset(scope)` **[MUST]** implement all 7 branches and **[MUST]** raise `ValueError` for any other input (no implicit `None` return). Tests assert `len(ResetScope) == 7`.

### A.6 Identifiers — exactly 8 prefixes

Grammar: `^(ctx|plan|dry|exec|art|ver|call|ckpt)_[0-9a-f]{16}$`

| Prefix | Record | Generation |
|---|---|---|
| `ctx_` | ContextRecord | `secrets.token_hex(8)` |
| `plan_` | PlanRecord | `secrets.token_hex(8)` |
| `dry_` | DryRunRecord | `secrets.token_hex(8)` |
| `exec_` | ExecutionRecord | `secrets.token_hex(8)` |
| `art_` | ArtifactRecord | **first 16 hex chars of the artifact content hash** (D.5) — deterministic, enables cross-run cache reuse |
| `ver_` | VerificationRecord | `secrets.token_hex(8)` |
| `call_` | per-LLM-call trace ID | `secrets.token_hex(8)` |
| `ckpt_` | CheckpointRecord | `secrets.token_hex(8)` |

Tests assert exactly 8 prefixes and that every prefix's generated IDs match the grammar (16 hex chars — note `call_001a2b3c4d5e6f70`-style examples in this doc are 16 chars).

### A.7 Store namespaces — exactly 8

`contexts`, `plans`, `artifacts`, `dry_runs`, `verifications`, `executions`, `cache`, `checkpoints`.

`Store` **[MUST]** raise `ValueError` for any other namespace. (v1 omitted `checkpoints`, making Batch 8 unimplementable as written; v2 fixes this.)

---

## Appendix B: Primitive library (normative Racket signatures)

All primitives are exported by the trusted shim module `rlm/primitives` and are the **only** bindings (besides `racket/base` pure forms) visible inside the sandbox. They are implemented outside the sandbox and injected via the sandbox namespace (Batch 4). No template body may `require` anything.

### B.1 Core calls

```racket
(llm-query instruction input
           #:model alias                ; model-alias string, required
           #:temperature [t 0]
           #:json [json? #f]            ; request JSON output; result parsed to jsexpr
           #:max-tokens [n #f]
           #:node-id [node-id #f])      ; logical node label for the trace/call-graph
  → string? | jsexpr?

(llm-query-async ...same kwargs...) → promise?
(await p) (await-all lst) (await-any lst)
```

In `live` mode, `llm-query` sends an `llm_call` wire request and blocks the calling Racket thread until the host responds (the host serves it from cache or a real provider, enforcing budget). In `simulate` mode the shim answers locally with a synthetic value (B.6) and increments counters.

### B.2 Combinators

```racket
(map-async f items #:max-concurrent [k 10]
                   #:error-policy [p 'fail_fast]   ; A.4 values as symbols
                   #:node-id [id #f])               → list?
(parallel thunk ...)                                → list?
(race thunks #:timeout-seconds [s #f])              → any   ; first to finish wins
(tree-reduce f items #:branch-factor [b 5] #:node-id [id #f]) → any
(fold-sequential f init items #:checkpoint-every [n #f] #:node-id [id #f]) → any
(sequence expr ...)                                 ; = begin, for readability
(choose pred a b)
(iterate-until pred f init #:max-iterations n)      → any
(recursive-spawn f input #:max-depth d #:branch-factor [b 5]) → any
(memoized f)            ; content-keyed memo within one execution
(with-validation validate-f produce-f #:max-retries [r 2]) → any
(try-fallback primary-thunk fallback-thunk)
```

Structural facts (used by tests, measured — not assumed — in simulate mode):

- `tree-reduce` over N items with branch factor B performs `Σ ceil(N/B^i)` reduce calls for `i = 1..ceil(log_B N)` plus the N producer calls if `f` itself calls the LLM per group. For the canonical map+reduce template with N=100, B=5: 100 map calls + 20 + 4 + 1 reduce calls = **125**, critical path `1 + ceil(log_5 100)` = **4**.
- `map-async` concurrency never exceeds `#:max-concurrent`; the simulate-mode counter records the observed maximum.

### B.3 Control / effects

```racket
(gate payload #:label [l #f])    → jsexpr?   ; suspends execution: gate_wait wire op;
                                             ; resumes with the human/agent decision
(checkpoint data)                → string?   ; persists data host-side, returns ckpt_ id
(partial-result node-id index value)         ; streaming notification (no-op if stream off)
(finish value)                               ; terminates the program with this result
(py-exec code #:input [in 'null]
              #:allowed-imports [imports '()]
              #:timeout-seconds [s 30])      → jsexpr?
(py-eval expr-string)                        ; sugar: (py-exec (string-append "result = " expr-string))
```

### B.4 Data access

```racket
(slot name-symbol)                → any      ; reads the artifact's slot binding (jsexpr→racket)
(context-items context-id path)   → list?    ; items from a stored context ($ = whole array)
(join-values lst)                 → string?  ; canonical "\n---\n" join for reduce inputs
```

`slot` is the **entire** slot mechanism. Slot values arrive in the `run` wire message as a JSON object and are exposed read-only. **There is no marker substitution and no code generation from slot values** (decision B). A slot value containing `(define ...)`, backticks, or unbalanced parens is just a string.

### B.5 py-exec isolation contract (decision C)

- Each `py-exec` request is executed by the **host** in a fresh OS process: `python -I -S` (isolated mode, no site).
- The child applies, before user code runs: `resource.setrlimit(RLIMIT_CPU)` (= timeout + 1s), `RLIMIT_AS` (default 512 MiB, policy-overridable), `RLIMIT_NOFILE` (16). Code + input arrive on stdin as JSON; the child writes `{"ok": true, "result": ...}` or `{"ok": false, "error": ...}` to stdout and nothing else.
- The host enforces wall-clock timeout by killing the **process group** (`os.killpg`).
- The import allowlist is enforced in the child via an import hook — defense in depth only. **The security boundary is the process + rlimits.**
- **Honest limitation [MUST be documented in README]:** there is no network namespace isolation; a hostile snippet could attempt outbound connections. The `py_exec_policy` verification check (E, check 17) therefore requires explicit `allow_py_exec: true` in the execution policy, and templates using it declare `(uses-py-exec #t)`.

### B.6 Simulate mode semantics

In `simulate` mode the shim, instead of emitting `llm_call`:

- returns the JSON string `{"sim": true, "node": "<node-id>", "call": <n>}` (or that jsexpr when `#:json #t`) — content-bearing so combinator data flow is exercised;
- records: total calls, calls per node-id, calls per model alias, observed max concurrency, recursion depth, critical-path length (longest chain of dependent calls, tracked via a per-thread depth counter incremented around each call);
- estimates prompt tokens as `ceil(len(instruction + input) / 4)` and completion tokens as `#:max-tokens` if given, else the model's `default_completion_estimate` (C.2);
- `py-exec` is **not** executed in simulate mode; it is counted as one `python_phase` and returns `{"sim": true}`;
- `gate` returns `{"sim": true, "decision": "approved"}` and increments a `gates` counter; `checkpoint` increments `checkpoints` and returns `"ckpt_0000000000000000"`.

The `done` wire message in simulate mode carries the full counter block (Appendix J.4); this **is** the dry-run simulation result (decision D).

---

## Appendix C: Model registry and providers

### C.1 Registry file

`v3/config/models.json` — a JSON array of entries with **one** schema (v1 had two conflicting ones):

### C.2 ModelRegistryEntry (normative, also in Appendix I.1)

```json
{
  "alias": "fast_text_model",
  "provider": "anthropic",
  "model_id": "claude-haiku-4-5-20251001",
  "capabilities": ["text", "json"],
  "context_window_tokens": 200000,
  "max_output_tokens": 8192,
  "cost_per_million_input_usd": 1.00,
  "cost_per_million_output_usd": 5.00,
  "supports_json_mode": true,
  "default_temperature": 0,
  "default_completion_estimate": 1024
}
```

Required aliases at minimum: `fast_text_model`, `quality_text_model`, `vision_model`. Templates reference aliases only; resolution to `model_id` happens host-side at call time. Cost estimate for a dry run = Σ over measured calls of `(prompt_tokens × in_rate + completion_tokens × out_rate) / 1e6`, reported as `{low, high}` where low uses measured estimates and high multiplies completion tokens by 2.5×.

### C.3 Retry policy (host-side, all providers)

Up to 3 attempts; exponential backoff 1s/4s with ±20% jitter; retry on HTTP 429/5xx/timeout; never retry on 4xx ≠ 429. A 429 **[SHOULD]** also halve the effective concurrency for that model alias for 60s (adaptive backpressure — new in v2).

### C.4 MockLLMProvider (decision E)

The test provider **[MUST]** be deterministic and content-bearing:

```python
def complete(self, instruction: str, input_text: str, *, model: str,
             temperature: float, json_mode: bool) -> LLMResult:
    digest = sha256(f"{model}|{instruction}|{input_text}".encode()).hexdigest()[:12]
    if json_mode:
        value = json.dumps({"mock": digest, "instruction_head": instruction[:40],
                            "input_head": input_text[:40]})
    else:
        value = f"mock:{digest}:{instruction[:40]}"
    return LLMResult(value=value, prompt_tokens=..., completion_tokens=...)
```

Test rule **[MUST]**: at least one integration test per combinator asserts that *distinct inputs produce distinct flowed outputs* (e.g., tree-reduce over 10 distinct items yields a final value whose lineage differs when any item changes). A test suite that passes with a constant-output provider is non-conforming.

---
## Appendix D: Template format, instantiation, and hashing

### D.1 File format

One template = one `.rkt` file in `v3/templates/`. The first form **[MUST]** be `(define-meta ...)`; every remaining top-level form is the body, evaluated in order inside the sandbox. The body's final effect **[MUST]** be a call to `(finish value)`.

There are **no `{{slot}}` markers**. Bodies read parameters with `(slot 'name)`.

### D.2 define-meta grammar

```racket
(define-meta
  (name "batch_extract_reduce")            ; [MUST] match filename stem
  (version "2.0.0")                        ; [MUST] semver
  (task-shapes (Batch Synthesize Composite))   ; ⊆ A.1
  (data-shapes (FlatList Tabular))             ; ⊆ A.2
  (slots
    ;; (slot-name (type T) (required bool) [(default v)] [(min n)] [(max n)] [(description "...")])
    (context_id        (type context-ref) (required #t))
    (items_path        (type string)      (required #f) (default "$"))
    (map_instruction   (type string)      (required #t))
    (reduce_instruction (type string)     (required #t))
    (map_model         (type model-alias) (required #t))
    (reduce_model      (type model-alias) (required #t))
    (max_concurrent    (type integer)     (required #f) (default 10) (min 1) (max 100))
    (branch_factor     (type integer)     (required #f) (default 5)  (min 2) (max 20))
    (json_mode         (type boolean)     (required #f) (default #t)))
  (output-schema "{\"type\": \"string\"}")  ; JSON Schema subset, as a JSON string (D.4)
  (budget-policy (on-exceed switch_model) (fallback-model "fast_text_model"))
  (cacheable #t)
  (streamable #t)
  (uses-py-exec #f)
  (uses-llm-generated-code #f))
```

Slot types — exactly 6: `string`, `integer`, `number`, `boolean`, `model-alias`, `context-ref`. `model-alias` values must resolve in the registry; `context-ref` values must be valid `ctx_` IDs (or the literal `"$previous"` inside a chain step ≥ 1). Validation rules: required slots present; no undeclared slot names; type and min/max bounds respected. **That is the entire slot-validation rule set** — there is no content inspection of string slots (decision B).

### D.3 Reference template body (normative example): `batch_extract_reduce.rkt`

```racket
(define items (context-items (slot 'context_id) (slot 'items_path)))

(define extracted
  (map-async
    (lambda (item)
      (llm-query (slot 'map_instruction) item
                 #:model (slot 'map_model)
                 #:json (slot 'json_mode)
                 #:node-id "extract"))
    items
    #:max-concurrent (slot 'max_concurrent)
    #:error-policy 'retry_then_skip
    #:node-id "extract"))

(define report
  (tree-reduce
    (lambda (group)
      (llm-query (slot 'reduce_instruction) (join-values group)
                 #:model (slot 'reduce_model)
                 #:node-id "synthesize"))
    extracted
    #:branch-factor (slot 'branch_factor)
    #:node-id "synthesize"))

(finish report)
```

### D.4 Output schemas

One language: a **JSON Schema subset** serialized as a JSON string in `output-schema`. Supported keywords: `type` (`string|number|integer|boolean|object|array`), `properties`, `required`, `items`, `enum`. The host validates the final `result.value` against it post-execution (E, check 16 validates the schema itself; result validation failure marks the execution `failed` with the raw value preserved in the trace). The v1 alist notation and the v1 flat `(key . type)` validator are both deleted.

### D.5 Instantiation and the artifact hash

`Instantiator.instantiate(template, slot_values)` performs, in order:

1. Apply defaults for absent optional slots.
2. Validate per D.2. On failure raise `SlotValidationError` listing **all** violations.
3. Canonicalize slot values: `json.dumps(slot_values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
4. Compute `body_hash = sha256(template_body_text_utf8)` (body text = file content minus the `define-meta` form, whitespace preserved).
5. Compute the artifact hash:
   ```python
   artifact_hash = sha256(canonical_json({
       "template_name": meta.name,
       "template_version": meta.version,
       "body_hash": body_hash,
       "slot_values": canonical_slots,   # the dict, embedded canonically
   }))
   ```
6. `artifact_id = "art_" + artifact_hash[:16]` (A.6). Store the `ArtifactRecord` (I.1). Instantiating identical inputs twice yields byte-identical records (idempotent put).

Determinism is testable: a property-based test (Batch 3) asserts hash equality across repeated instantiation and hash inequality under any single-field perturbation.

### D.6 Template registration (load-time static checks)

At server start, `TemplateRegistry.load_all()` parses every `.rkt` file and **[MUST]** reject the catalog (refuse to serve) if any template fails:

- `define-meta` present, grammatical, name matches filename, version is semver;
- body reads only **declared** slots — static scan for `(slot 'x)` forms;
- body uses only allowlisted primitives — static scan of free identifiers in the body against the `rlm/primitives` export list plus a fixed pure-form allowlist (`define`, `lambda`, `let`, `let*`, `if`, `cond`, `quote`, `list`, `string-append`, `length`, `map`, `filter`, `and`, `or`, `not`, numeric/string/list ops from `racket/base`). Note: this scan runs on **trusted, repo-committed template bodies at registration time** — it is a code-review aid, not an input sanitizer; agent-supplied slot values are never scanned because they are never code;
- `uses-py-exec`/`uses-llm-generated-code` flags consistent with body (`py-exec` present ⇔ flag true);
- `output-schema` parses as JSON and uses only D.4 keywords.

The registry exposes exactly **16 templates** (Appendix G).

---

## Appendix E: Verification engine — exactly 23 checks

`VerificationEngine.verify(plan, artifact, dry_run, context, policy) -> VerificationRecord` runs all checks, never short-circuits, and returns `decision: "pass"` iff zero `fail`-severity results. Field names below refer to the records defined in Appendix I.1 — every input named here **exists** on those records (v1's checks read nonexistent attributes; v2's are reconciled).

| # | Name | Severity | Rule (inputs) |
|---|---|---|---|
| 1 | `artifact_exists` | fail | `artifact_id` resolves in store |
| 2 | `artifact_hash` | fail | recomputed D.5 hash == `artifact.artifact_hash` |
| 3 | `template_known` | fail | `(artifact.template_name, artifact.template_version)` in registry |
| 4 | `slot_schema` | fail | `artifact.slot_values` re-validate against current registry meta |
| 5 | `context_exists` | fail | every `context-ref` slot resolves (chains: `$previous` allowed for step ≥ 1) |
| 6 | `context_shape_compatible` | warn | `context.metadata.data_shape ∈ meta.data-shapes` (or `Unknown`) |
| 7 | `primitive_allowlist` | fail | registration-time scan result for this template version is `clean` (D.6) |
| 8 | `model_aliases_resolve` | fail | every `model-alias` slot value is in the registry |
| 9 | `call_count_limit` | fail | `dry_run.simulation.llm_calls ≤ policy.max_llm_calls` |
| 10 | `critical_path_limit` | warn | `dry_run.simulation.critical_path_calls ≤ policy.max_critical_path` |
| 11 | `concurrency_limit` | fail | `dry_run.simulation.max_concurrency ≤ policy.max_concurrency` |
| 12 | `token_budget` | fail | `dry_run.estimate.estimated_tokens.total ≤ policy.max_tokens` |
| 13 | `cost_budget` | fail | `dry_run.estimate.estimated_cost_usd.high ≤ policy.max_cost_usd` |
| 14 | `recursion_depth_limit` | fail | `dry_run.simulation.recursive_depth ≤ policy.max_recursion_depth` |
| 15 | `dry_run_fresh` | fail | a `DryRunRecord` exists for this `plan_id` whose `artifact_id` == this artifact (re-dry-run required after any slot change) |
| 16 | `output_schema_valid` | fail | `meta.output-schema` parses and uses only D.4 keywords |
| 17 | `py_exec_policy` | fail | `meta.uses-py-exec` ⇒ `policy.allow_py_exec` |
| 18 | `llm_generated_code_policy` | fail | `meta.uses-llm-generated-code` ⇒ `policy.allow_llm_generated_code` |
| 19 | `gate_policy` | fail | `dry_run.simulation.gates > 0` ⇒ `policy.allow_gates` |
| 20 | `timeout_sane` | fail | `1 ≤ timeout_seconds ≤ policy.max_timeout_seconds` (default cap 3600) |
| 21 | `checkpoint_writable` | warn | `dry_run.simulation.checkpoints > 0` ⇒ store `checkpoints` namespace writable |
| 22 | `cache_temperature` | warn | `meta.cacheable` and any slot-resolved temperature ≠ 0 ⇒ warn (cache keys include temperature, but nonzero temperature defeats reuse intent) |
| 23 | `budget_degradation_valid` | fail | `meta.budget-policy.fallback-model`, if set, resolves in the registry |

Policy defaults (used when `execute_strategy.policy_json` omits a field): `max_llm_calls 500`, `max_concurrency 50`, `max_critical_path 25`, `max_tokens 2_000_000`, `max_cost_usd 10.00`, `max_recursion_depth 5`, `allow_py_exec false`, `allow_llm_generated_code false`, `allow_gates true`, `max_timeout_seconds 3600`.

---
## Appendix F: Deterministic classifier

All classification operates on structured hint/metadata fields only. No LLM call is needed when hints are complete. The planner reports `hints_complete: bool` and `assumed_fields: [..]` — **never** a fabricated `confidence` number (v1's `confidence: 1.0` claim is deleted; deterministic ≠ correct).

### F.1 `classify_task_shape(hints) -> TaskShape`

Hint fields: `item_count: int|None`, `independent: bool|None`, `output_type: "one"|"list"|"per_item"|None`, `operation: str|None`, `has_second_phase: bool|None`, `sub_operations: list[str]|None`, `ordered: bool|None`.

```python
MANY_ITEMS_THRESHOLD = 3   # v2: two items is not a batch (v1 used 2)

def classify_task_shape(hints: dict) -> tuple[str, list[str]]:
    """Returns (task_shape, assumed_fields). Every hint read through _get()
    that was absent is recorded in assumed_fields."""
    assumed: list[str] = []
    def _get(key, default):
        if hints.get(key) is None:
            assumed.append(key)
            return default
        return hints[key]

    item_count       = _get("item_count", 0)
    independent      = _get("independent", True)
    output_type      = _get("output_type", "one")
    operation        = _get("operation", "other")
    has_second_phase = _get("has_second_phase", False)
    sub_operations   = _get("sub_operations", [])
    ordered          = _get("ordered", False)

    # Q9 (early): multiple distinct phases -> Composite
    if has_second_phase and len(sub_operations) >= 2:
        return "Composite", assumed

    # Q0: one small input, one output, one operation, no second phase
    if item_count <= 1 and output_type == "one" and not has_second_phase \
            and len(sub_operations) <= 1:
        if operation == "generate":                 return "Generate", assumed
        if operation in ("refine", "improve", "iterate"): return "Refine", assumed
        return "Direct", assumed

    # Q1: many items?
    if item_count >= MANY_ITEMS_THRESHOLD:
        if independent:                                       # Q2
            if operation in ("label",):                       return "Classify", assumed
            if operation in ("check", "grade"):               return "Validate", assumed
            if operation in ("aggregate", "compute", "stats"): return "Aggregate", assumed
            return "Batch", assumed                           # transform/extract/other
        # dependent items — v2 fix: Pipeline requires *stage* structure,
        # not merely item dependence (v1 wrongly sent dependent+unordered here)
        if len(sub_operations) >= 2:                          return "Pipeline", assumed
        return "Synthesize", assumed   # ordered or not: fold/tree chosen later

    # 2 items or item_count unknown — single/dual-input shapes (Q5–Q8)
    if operation == "generate" or (output_type == "list" and item_count == 0):
        return "Generate", assumed
    if operation in ("refine", "improve", "iterate"):
        return "Refine", assumed
    if operation in ("decompose", "split", "parse") and output_type in ("list", "per_item"):
        return "Decompose", assumed
    if operation in ("compare", "select", "choose", "rank"):
        return ("Search" if hints.get("latency_priority") else "Compare"), assumed
    if operation in ("aggregate", "compute", "stats"):
        return "Aggregate", assumed
    if output_type == "one" and item_count == 2:
        return "Synthesize", assumed
    return "Direct", assumed
```

(v2 changes vs v1: threshold 3; `extract` removed from the Decompose trigger — extraction of parts is Decompose only via `decompose|split|parse`, while `extract` over many items is Batch; dependent-unordered no longer falls into Pipeline; Generate/Refine reachable from the Q0 branch.)

### F.2 `classify_data_shape(metadata) -> DataShapeResult`

Unchanged from v1's F.2 logic with three amendments: (1) the fallback return is `Unknown`, not `Singular`; (2) an agent-provided `data_shape` hint is honored only if it is one of the 11 A.2 values, else ignored with a trace note; (3) `Graph` is classified when `metadata.edges` is a non-empty list (v1 listed Graph in the enum but never produced it). Concurrency hint: `min(item_count, 20)` when `item_count > 50`, else `item_count`.

### F.3 `select_template(task_shape, data_shape, hints) -> Selection`

Returns `Selection(template_name, slot_flags: dict, chain: list|None)` — v1 returned a bare string and smuggled distinctions through comments. Branches that selected the same template on both arms in v1 (`_select_compare`, `_select_aggregate`, Refine's dead Q2) are collapsed: the discriminating hint now sets a `slot_flags` entry instead of pretending to branch.

| TaskShape | Decision | Result |
|---|---|---|
| Direct | `fits_context` false → raise `ReclassificationNeeded`; `output_type ∈ {list, per_item}` → `direct_json_extract`; else `direct_call` (flag `needs_compute` when operation ∈ {aggregate, compute, stats}) |
| Batch | per-item output: `ambiguous_items` → `tiered_review`, else `batch_map` (flag `memoize` when `likely_duplicates`); combined output: `ordered` → `batch_extract_fold`; `ambiguous_items` → `tiered_review`; else `batch_extract_reduce` |
| Synthesize | `fits_context` → `direct_call`; `ordered` → `ordered_synthesis_fold` (flag `summarize_accumulator` when `accumulator_large`); else `tree_synthesis` |
| Search | finite candidates: `latency_priority` → `race_candidates`, else `compare_candidates`; infinite → `refine_until_valid` |
| Refine | `has_testable_pred` → `refine_until_valid`, else `bounded_critique_refine` |
| Compare | `compare_candidates`, flags `compare_target`, `select_or_synthesize` |
| Classify | one item → `direct_call`; `ambiguous_items` → `tiered_review`; else `batch_map` (flag `aggregate_report` when `has_second_phase`) |
| Pipeline | `stages_distinct` false → `ReclassificationNeeded("Batch")`; else build a chain from `sub_operations` (each step selected recursively as Composite does) |
| Generate | until-condition → `refine_until_valid`; fixed count: `items_consistent` → `ordered_synthesis_fold`, else `batch_map` (flag `dedup` when `items_unique`) |
| Decompose | not `one_pass` → `recursive_decompose`; `process_parts_after` → `decompose_then_batch`; else `direct_json_extract` (flag `known_boundary`) |
| Validate | not `same_rubric` → `ordered_synthesis_fold`; `ambiguous_items` → `tiered_review`; else `batch_map` |
| Aggregate | `tabular_extract_aggregate` (flag `interpret` when not `pure_computation`; flag `groupby` when `grouped_report`) |
| Composite | chain: classify each `sub_operations[i]` via the op→shape map (extract/transform→Batch, label→Classify, check→Validate, synthesize→Synthesize, refine→Refine, compare→Compare, aggregate→Aggregate, decompose→Decompose, generate→Generate) and select each step with `has_second_phase=False`; step ≥ 1 input slot = `"$previous"` |

`ReclassificationNeeded` propagates to the `plan_strategy` response as `status: "needs_reclassification"` with guidance text — it is not a server error.

---

## Appendix G: Template catalog — exactly 16 templates

| Template | TaskShapes | DataShapes | Composition | Streamable | Cacheable |
|---|---|---|---|---|---|
| `direct_call` | Direct, Synthesize, Classify | Singular, KeyValue | `llm-query` | No | Yes |
| `direct_json_extract` | Direct, Decompose | Singular, KeyValue | `llm-query #:json` + `with-validation` | No | Yes |
| `batch_map` | Batch, Classify, Validate, Generate | FlatList, Paired, Multimodal | `map-async` | Yes | Yes |
| `batch_extract_reduce` | Batch, Synthesize, Composite | FlatList, Tabular | `map-async` + `tree-reduce` | Yes | Yes |
| `batch_extract_fold` | Batch, Synthesize | FlatList, TimeSeries, ChunkedSingular | `map-async` + `fold-sequential` | Yes | Yes |
| `ordered_synthesis_fold` | Synthesize, Generate, Validate | FlatList, ChunkedSingular, TimeSeries | `fold-sequential` + optional `checkpoint` | Yes | Yes |
| `tree_synthesis` | Synthesize | FlatList, Hierarchy | `tree-reduce` | Yes | Yes |
| `compare_candidates` | Compare, Search | FlatList, Paired | `parallel` + selector (`py-exec` or `llm-query`) | No | Yes |
| `race_candidates` | Search | FlatList | `race` | No | No |
| `refine_until_valid` | Refine, Search, Generate | Singular | `iterate-until` + `with-validation` | No | No |
| `bounded_critique_refine` | Refine | Singular | `iterate-until` (critique/refine pair) | No | No |
| `tiered_review` | Batch, Classify, Validate | FlatList, Tabular | cheap `map-async` + filter + expensive `map-async` | Yes | Yes |
| `tabular_extract_aggregate` | Aggregate | Tabular, FlatList | `map-async` + `py-exec` | Yes | Yes |
| `decompose_then_batch` | Decompose, Composite | Singular, Hierarchy | `llm-query #:json` + `map-async` | Yes | Yes |
| `recursive_decompose` | Decompose | Hierarchy, Singular | `recursive-spawn` | No | No |
| `code_interpreter` | Direct, Aggregate | Singular, Tabular | `llm-query` + `py-exec` + `with-validation` + `iterate-until` | No | No |

Non-cacheable rationale: `race_candidates` is timing-nondeterministic; the two refine templates and `recursive_decompose` have iteration-dependent state (resume via checkpoints, not result cache); `code_interpreter` runs LLM-generated code (`uses-llm-generated-code #t`, gated by check 18). Every "Yes/No" cell is asserted by a registry test against the template's `define-meta` flags.

---
## Appendix H: End-to-end walkthrough (happy path)

Task: "Extract ACE2 protein mentions from 100 research papers and synthesize a report." All IDs below validate against A.6.

```
→ load_context(data="[{\"id\":\"paper_001\",...}]", name="ace2_papers",
               metadata_json="{\"data_shape\":\"FlatList\",\"item_count\":100,\"independent\":true}")
← { "status": "ok", "context_id": "ctx_3f9a17b2c4d5e6f0", "name": "ace2_papers",
    "preview": "[{\"id\":\"paper_001\",\"text\":\"Recent studies on ACE2...",
    "next_actions": ["Call plan_strategy with context_id=ctx_3f9a17b2c4d5e6f0"] }

→ plan_strategy(task="Extract ACE2 mentions from each paper and synthesize one report.",
                context_id="ctx_3f9a17b2c4d5e6f0",
                hints_json="{\"item_count\":100,\"independent\":true,\"output_type\":\"one\",
                             \"has_second_phase\":true,\"sub_operations\":[\"extract\",\"synthesize\"]}")
← { "status": "ok", "plan_id": "plan_8a1b2c3d4e5f6071",
    "classification": { "task_shape": "Composite", "constituent_shapes": ["Batch","Synthesize"],
                        "data_shape": "FlatList", "hints_complete": true, "assumed_fields": [],
                        "rationale": "Independent items, combined output, second phase." },
    "recommended": { "kind": "template_invocation",
                     "template_name": "batch_extract_reduce", "template_version": "2.0.0",
                     "slot_values": { "context_id": "ctx_3f9a17b2c4d5e6f0", "items_path": "$",
                       "map_instruction": "Extract ACE2 mentions with evidence as JSON.",
                       "reduce_instruction": "Synthesize ACE2 findings into a report with citations.",
                       "map_model": "fast_text_model", "reduce_model": "quality_text_model",
                       "max_concurrent": 20, "branch_factor": 5, "json_mode": true } },
    "alternatives": [ { "template_name": "batch_extract_fold",
                        "tradeoff": "Preserves paper order; higher latency." } ],
    "next_actions": ["Call dry_run_strategy(plan_id=plan_8a1b2c3d4e5f6071)"] }

→ dry_run_strategy(plan_id="plan_8a1b2c3d4e5f6071")
  # Host spawns the Racket runtime in simulate mode, evaluates batch_extract_reduce.rkt
  # with a counting stub provider, and MEASURES the call graph (decision D — not predicted).
← { "status": "ok", "dry_run_id": "dry_42d1e0f9a8b7c6d5", "plan_id": "plan_8a1b2c3d4e5f6071",
    "artifact": { "artifact_id": "art_a1b2c3d4e5f60718", "template_name": "batch_extract_reduce",
                  "template_version": "2.0.0", "artifact_hash": "a1b2c3d4e5f60718...",
                  "primitives_used": ["map-async","tree-reduce","llm-query"] },
    "simulation": { "llm_calls": 125, "critical_path_calls": 4, "max_concurrency": 20,
                    "recursive_depth": 0, "checkpoints": 0, "python_phases": 0, "gates": 0,
                    "calls_by_model": { "fast_text_model": 100, "quality_text_model": 25 } },
    "estimate": { "estimated_tokens": { "prompt": 100000, "completion": 31250, "total": 131250 },
                  "estimated_cost_usd": { "low": 0.26, "high": 0.65 }, "cache_hits_expected": 0 },
    "call_graph": [ { "node_id": "extract", "primitive": "map-async", "calls": 100 },
                    { "node_id": "synthesize", "primitive": "tree-reduce", "calls": 25 } ],
    "warnings": [], "next_actions": ["Call execute_strategy(plan_id=plan_8a1b2c3d4e5f6071)"] }

→ execute_strategy(plan_id="plan_8a1b2c3d4e5f6071", timeout_seconds=900,
                   policy_json="{\"max_llm_calls\":500,\"max_concurrency\":50}")
  # Host runs all 23 checks (E), records ver_ with decision=pass (invariant I-2),
  # then spawns the Racket runtime in LIVE mode; real llm_call wire requests are served
  # by providers/cache under budget enforcement.
← { "status": "ok", "execution_id": "exec_19c8d7e6f5a4b3c2", "artifact_id": "art_a1b2c3d4e5f60718",
    "verification": { "verification_id": "ver_5d6e7f8091a2b3c4", "decision": "pass",
                      "failed_checks": [], "warnings": [] },
    "result": { "value": "ACE2 (Angiotensin-Converting Enzyme 2) findings across 100 papers...",
                "schema_valid": true },
    "execution": { "state": "finished", "elapsed_seconds": 182.4, "llm_calls": 125,
                   "cache_hits": 0, "tokens": 131250, "checkpoints_written": 0 },
    "next_actions": ["Call get_execution_trace(execution_id=exec_19c8d7e6f5a4b3c2)"] }
```

ID chain:

```
ctx_3f9a17b2c4d5e6f0
  → plan_8a1b2c3d4e5f6071
    → dry_42d1e0f9a8b7c6d5      (creates art_a1b2c3d4e5f60718 via simulated execution)
      → exec_19c8d7e6f5a4b3c2   (creates ver_5d6e7f8091a2b3c4; real calls here)
```

Chains, streaming, and cross-run cache reuse work exactly as in v1's H.2 walkthrough, with one change: the dry run's `cache_hits_expected` is computed by replaying each simulated `llm_call`'s content cache-key against the live cache (the simulate provider produces the same key material a live call would, since keys exclude the response). A second run that changes only `reduce_instruction` reports `cache_hits_expected: 100` for the unchanged map phase.

---

## Appendix I: Interface reference (single source of truth — decision F)

Batches reference this appendix and **[MUST NOT]** restate signatures or response shapes. Where a batch and this appendix disagree, this appendix wins (precedence rule 1, §1).

### I.1 Records (Pydantic v2, strict; `v3/rlm_scheme/models.py`)

```python
class ContextRecord(BaseModel):
    context_id: str            # ctx_
    name: str
    data: list | dict          # parsed JSON payload
    metadata: ContextMetadata
    created_at: float

class ContextMetadata(BaseModel):
    data_shape: DataShape = DataShape.Unknown
    item_count: int = 0
    independent: bool = True
    ordered: bool = False
    modality: list[str] = ["text"]
    # plus any additional structural fields from F.2 (all optional)

class PlanRecord(BaseModel):
    plan_id: str               # plan_
    context_id: str
    task: str
    classification: Classification
    recommended: TemplateInvocation | TemplateChain   # discriminated by .kind
    alternatives: list[Alternative] = []
    created_at: float

class Classification(BaseModel):
    task_shape: TaskShape
    constituent_shapes: list[TaskShape] = []
    data_shape: DataShape
    hints_complete: bool
    assumed_fields: list[str] = []
    rationale: str

class TemplateInvocation(BaseModel):
    kind: Literal["template_invocation"] = "template_invocation"
    template_name: str
    template_version: str
    slot_values: dict

class TemplateChain(BaseModel):
    kind: Literal["template_chain"] = "template_chain"
    steps: list[ChainStep]     # ChainStep has step:int, template_name, template_version, slot_values

class ArtifactRecord(BaseModel):
    artifact_id: str                 # art_ = "art_" + artifact_hash[:16]
    template_name: str
    template_version: str
    body_hash: str                   # sha256 of body text
    artifact_hash: str               # full D.5 hash
    slot_values: dict                # canonical
    primitives_used: list[str]
    uses_py_exec: bool
    uses_llm_generated_code: bool

class DryRunRecord(BaseModel):
    dry_run_id: str            # dry_
    plan_id: str
    artifact_id: str           # for chains: artifact_ids: list[str] also present
    simulation: SimulationStats
    estimate: CostEstimate
    call_graph: list[CallGraphNode]
    warnings: list[str] = []

class SimulationStats(BaseModel):
    llm_calls: int
    critical_path_calls: int
    max_concurrency: int
    recursive_depth: int
    checkpoints: int
    python_phases: int
    gates: int
    calls_by_model: dict[str, int]

class CostEstimate(BaseModel):
    estimated_tokens: TokenEstimate         # prompt, completion, total
    estimated_cost_usd: CostRange           # low, high
    cache_hits_expected: int = 0

class VerificationRecord(BaseModel):
    verification_id: str       # ver_
    artifact_id: str
    decision: Literal["pass", "fail"]
    checks: list[CheckResult]  # name, status(pass|warn|fail), message
    failed_checks: list[str] = []
    warnings: list[str] = []

class ExecutionRecord(BaseModel):
    execution_id: str          # exec_
    artifact_id: str
    plan_id: str
    verification_id: str
    state: ExecutionState
    result: ExecutionResult | None
    stats: ExecutionStats
    gate: GateInfo | None = None        # set when state == suspended_gate
    created_at: float

class CheckpointRecord(BaseModel):
    checkpoint_id: str         # ckpt_
    execution_id: str
    node_id: str | None
    data: object
    created_at: float

class ModelRegistryEntry(BaseModel):   # C.2 — the ONLY registry schema
    alias: str
    provider: str
    model_id: str
    capabilities: list[str]
    context_window_tokens: int
    max_output_tokens: int
    cost_per_million_input_usd: float
    cost_per_million_output_usd: float
    supports_json_mode: bool
    default_temperature: float = 0
    default_completion_estimate: int = 1024
```

### I.2 Component constructors (the ONLY valid signatures)

```python
Store(root: Path)
ContextStore(store: Store)
TemplateRegistry(template_dir: Path)                       # .load_all() at startup
Instantiator(registry: TemplateRegistry)
Classifier()                                               # pure functions, no deps
Planner(classifier: Classifier, registry: TemplateRegistry,
        llm_provider: LLMProvider | None)                  # provider only for slot gap-fill
LLMCache(store: Store)
BudgetMonitor(policy: ExecutionPolicy)
RacketRuntime(runtime_dir: Path)                           # spawns/owns the subprocess
DryRunner(store: Store, instantiator: Instantiator, runtime: RacketRuntime, registry: TemplateRegistry)
VerificationEngine(store: Store, registry: TemplateRegistry)
PyExecSandbox(python_bin: str = "python3")
GateManager(store: Store)
CheckpointManager(store: Store)
StreamingNotifier(mcp_session)                             # no-op if stream disabled
Executor(store: Store, registry: TemplateRegistry, instantiator: Instantiator,
         runtime: RacketRuntime, verification: VerificationEngine, dry_runner: DryRunner,
         llm_provider: LLMProvider, cache: LLMCache, gate_manager: GateManager,
         checkpoint_manager: CheckpointManager, py_sandbox: PyExecSandbox,
         notifier: StreamingNotifier)
ChainExecutor(executor: Executor, store: Store)
```

There is exactly **one** `RacketRuntime`, **one** `Executor`. No module-level singletons referencing undefined classes (v1's Batch 7 defect). Wiring lives in `v3/rlm_scheme/app.py::build_app()` which constructs each component once and injects it; tests construct via `build_app(tmp_path)`.

### I.3 MCP tool response envelope

Every tool returns `{"status": "ok" | "error" | "needs_reclassification" | "suspended", ...}`. On `error`: `{"status":"error","error_code": str,"message": str}`. The per-tool `ok` shapes are exactly those shown in the Appendix H walkthrough; `resume_execution` returns the same shape as `execute_strategy`. No tool returns a shape not defined here or in H.

---

## Appendix J: Host↔Runtime wire protocol (decision A, the core of the rewrite)

JSON Lines (one JSON object per line) over the runtime subprocess's stdin/stdout. stderr is captured for diagnostics only. The host is the client; the runtime never initiates a connection.

### J.1 Startup

Host spawns: `racket v3/runtime/main.rkt`. Runtime emits `{"type":"ready","protocol":"1.0"}` then waits.

### J.2 Host → runtime: `run`

```json
{ "type": "run",
  "mode": "simulate" | "live",
  "artifact_id": "art_a1b2c3d4e5f60718",
  "body": "<template body source text>",
  "slot_values": { "context_id": "ctx_3f9a17b2c4d5e6f0", "...": "..." },
  "contexts": { "ctx_3f9a17b2c4d5e6f0": [ ... ] },   // referenced context data, inlined
  "limits": { "max_recursion_depth": 5 } }
```

The runtime loads `body` into a `racket/sandbox` evaluator whose namespace exposes **only** `rlm/primitives` + the pure allowlist (D.6). `slot_values`/`contexts` are bound as data. The body cannot open files, sockets, or FFI (invariant I-4).

### J.3 Runtime → host: effect requests (live mode)

Each blocks the issuing Racket thread until the host replies with a matching `id`.

```json
{ "type":"llm_call", "id":"call_001a2b3c4d5e6f70", "node_id":"extract",
  "instruction":"...", "input":"...", "model":"fast_text_model",
  "temperature":0, "json":true, "max_tokens":null }
   → host reply: { "type":"llm_result", "id":"call_001a2b3c4d5e6f70",
                   "value":"...", "prompt_tokens":900, "completion_tokens":300, "cache_hit":false }

{ "type":"py_exec", "id":"...", "code":"...", "input": ..., "allowed_imports":["json"],
  "timeout_seconds":30 }
   → { "type":"py_result", "id":"...", "ok":true, "result": ... }

{ "type":"checkpoint", "id":"...", "node_id":"synthesize", "data": ... }
   → { "type":"checkpoint_ack", "id":"...", "checkpoint_id":"ckpt_77a8b9c0d1e2f304" }

{ "type":"gate", "id":"...", "label":"review", "payload": ... }
   → execution suspends; host returns to MCP caller with status "suspended".
     On resume_execution, host replies { "type":"gate_decision", "id":"...", "decision": ... }

{ "type":"partial_result", "node_id":"extract", "index":7, "value":"..." }   // fire-and-forget
```

Concurrency: the runtime may have many `llm_call` requests in flight (up to a combinator's `max_concurrent`); the host matches replies by `id` and enforces global concurrency/budget regardless of what the runtime requests (invariant I-6). If a policy limit would be exceeded, the host replies `{"type":"llm_error","id":...,"error_code":"budget_exceeded"}` and the runtime surfaces it per the active `error-policy`.

### J.4 Runtime → host: terminal

```json
{ "type":"done", "value": <finish value>,
  "stats": { "llm_calls":125, "critical_path_calls":4, "max_concurrency":20,
             "recursive_depth":0, "checkpoints":0, "python_phases":0, "gates":0,
             "calls_by_model": {"fast_text_model":100,"quality_text_model":25} } }
{ "type":"error", "error_code":"...", "message":"...", "trace":"..." }
```

In simulate mode there are no effect requests; the runtime answers calls internally (B.6) and emits only `done` with the measured `stats`. **The `stats` block is identical in shape between simulate and live**, so dry-run and execution report through one code path (invariant I-3).

### J.5 Lifecycle and faults

- One subprocess per `run`; the host tears it down on `done`/`error`/timeout via process-group kill.
- A crashed runtime (no `done` within `timeout_seconds`, or stdout EOF) → execution `failed`, `error_code: "runtime_crash"`, stderr attached to the trace.
- The host **[MUST NOT]** trust runtime-reported stats for policy enforcement during live mode — it independently counts the `llm_call` requests it served. The `done.stats` are reconciled against the host's count; a mismatch is a `warn` in the trace.

---
## Build plan (Batches 0–11)

Each batch lists **Purpose**, **Depends-On**, **Files**, **Requirements** (RFC-2119, R-x.y tagged), **Tests**, **Acceptance gates**. Dependency labels are exact (v1 mislabeled several). Every batch ends green (all its tests + all prior tests pass) before the next begins. Build everything under `v3/`.

### Batch 0 — Foundations: enums, IDs, records, registry

- **Purpose:** the vocabulary every later batch imports.
- **Depends-On:** none.
- **Files:** `v3/rlm_scheme/enums.py`, `ids.py`, `models.py`, `config/models.json`, `v3/pyproject.toml`.
- **Requirements:**
  - R-0.1 [MUST] Define all enums per A.1–A.5: TaskShape(13), DataShape(11), ExecutionState(7), ErrorPolicy(3), ResetScope(7).
  - R-0.2 [MUST] `ExecutionState` exposes `can_transition(a, b) -> bool` per the A.3 table; illegal transitions raise `InvalidStateTransition`.
  - R-0.3 [MUST] `ids.py` defines the 8 prefixes (A.6), `new_id(prefix)`, and `validate_id(s)` against `^(ctx|plan|dry|exec|art|ver|call|ckpt)_[0-9a-f]{16}$`. `art_` IDs are derived (D.5), not random.
  - R-0.4 [MUST] All records in I.1 as strict Pydantic v2 models.
  - R-0.5 [MUST] `ModelRegistryEntry` is the only registry schema (C.2); `config/models.json` ships `fast_text_model`, `quality_text_model`, `vision_model`.
- **Tests:** `len()` of each enum equals its cardinality; transition table matrix; `validate_id` accepts/rejects a fixture set (incl. 15-char reject, wrong-prefix reject); every record round-trips JSON.
- **Acceptance gates:** `len(ResetScope)==7`, `len(TaskShape)==13`, exactly 8 prefixes enumerated, all model entries validate.

### Batch 1 — Store and ContextStore

- **Depends-On:** Batch 0.
- **Files:** `v3/rlm_scheme/store.py`, `context_store.py`.
- **Requirements:**
  - R-1.1 [MUST] `Store.NAMESPACES` = the 8 names in A.7; any other namespace → `ValueError`.
  - R-1.2 [MUST] `put/get/exists/list/delete` per namespace; JSON-on-filesystem under `root/<namespace>/<id>.json`; idempotent put.
  - R-1.3 [MUST] `Store.reset(scope)` implements all 7 ResetScope branches (A.5); unknown scope → `ValueError` (no implicit `None`).
  - R-1.4 [MUST] `ContextStore.load(data, name, metadata)` parses JSON, classifies data shape if absent (Batch 2 hook; until then store `Unknown`), returns `ContextRecord`; `context-items(id, path)` supports `$` and `$.field`.
- **Tests:** namespace guard; reset matrix (each scope clears exactly its namespaces, leaves others); round-trip; `context-items` paths.
- **Acceptance gates:** all 8 namespaces creatable; `reset("all")` empties everything; `reset("executions")` clears executions+dry_runs+verifications+checkpoints only.

### Batch 2 — Classifier

- **Depends-On:** Batch 0.
- **Files:** `v3/rlm_scheme/classifier.py`.
- **Requirements:**
  - R-2.1 [MUST] `classify_task_shape(hints) -> (TaskShape, assumed_fields)` per F.1 exactly (threshold 3; the v2 corrections).
  - R-2.2 [MUST] `classify_data_shape(metadata) -> DataShapeResult` per F.2; fallback `Unknown`; `Graph` from non-empty `edges`.
  - R-2.3 [MUST] `select_template(task_shape, data_shape, hints) -> Selection` per F.3; `ReclassificationNeeded` where specified.
  - R-2.4 [MUST] No LLM calls in this module (pure, deterministic).
- **Tests:** a table-driven fixture (≥ 2 cases per TaskShape) asserting shape + assumed_fields; reclassification raises for Direct-overflow and non-distinct Pipeline; collapsed branches return correct `slot_flags`.
- **Acceptance gates:** every TaskShape and DataShape reachable by some fixture; classifier importable with zero non-stdlib deps.

### Batch 3 — Template registry, parser, instantiator

- **Depends-On:** Batches 0–1.
- **Files:** `v3/rlm_scheme/template_store.py`, `instantiator.py`, `v3/templates/*.rkt` (all 16).
- **Requirements:**
  - R-3.1 [MUST] Parse `define-meta` per D.2 into a `TemplateMeta`; body text = file minus the meta form.
  - R-3.2 [MUST] `TemplateRegistry.load_all()` runs the D.6 static checks and refuses to serve on any failure; exposes exactly 16 templates.
  - R-3.3 [MUST] `Instantiator.instantiate(meta, slot_values)` performs the D.5 steps: defaults, validate (all violations), canonicalize, hash, `art_` id. **No text substitution; no slot-content inspection.**
  - R-3.4 [MUST] Slot validation covers exactly the 6 slot types; `model-alias` resolves in registry; `context-ref` matches `ctx_` or `"$previous"`.
  - R-3.5 [MUST] Write all 16 template bodies (G), each reading params via `(slot 'x)`.
- **Tests:** property-based determinism (repeat → equal hash; single-field perturb → unequal); `art_` id == `art_`+hash[:16]; every template loads and passes static checks; required-slot-missing lists all; an unknown slot name rejected; a string slot containing `(define x)` is accepted unchanged (decision B regression test).
- **Acceptance gates:** registry reports 16; instantiation idempotent; no blacklist code exists anywhere (`grep -ri "DANGEROUS" v3/` returns nothing).

### Batch 4 — Racket runtime: shim, primitives, sandbox, simulate mode

- **Depends-On:** Batches 0, 3.
- **Files:** `v3/runtime/main.rkt`, `v3/runtime/primitives.rkt`, `v3/runtime/wire.rkt`, `v3/runtime/sandbox.rkt`, `v3/rlm_scheme/runtime.py`.
- **Requirements:**
  - R-4.1 [MUST] `main.rkt` implements the J protocol: emit `ready`, accept `run`, dispatch effects, emit `done`/`error`.
  - R-4.2 [MUST] `primitives.rkt` implements every B primitive. In simulate mode they answer locally and update the J.4 stat counters (B.6); in live mode they marshal J.3 requests and block on replies.
  - R-4.3 [MUST] `sandbox.rkt` builds a `racket/sandbox` evaluator that exposes only `rlm/primitives` + the D.6 pure allowlist; filesystem/network/FFI denied (invariant I-4). A body attempting `(require racket/system)` or file I/O fails at sandbox build/eval.
  - R-4.4 [MUST] `runtime.py::RacketRuntime` spawns the subprocess, performs the handshake, sends `run`, services effect callbacks via injected handlers, returns `(value, stats)`; tears down via process-group kill on done/error/timeout (J.5).
  - R-4.5 [MUST] Critical-path and max-concurrency are *measured* in simulate mode (B.6), not declared.
- **Tests (require a Racket toolchain in CI):** `batch_extract_reduce` in simulate over 100 items yields `llm_calls==125`, `critical_path_calls==4`, `max_concurrency<=20`; sandbox escape attempts fail; a deliberately crashing body → `runtime_crash`; tree-reduce formula for several (N,B) pairs.
- **Acceptance gates:** all 16 bodies run to `done` in simulate mode with sane stats; no body can read a file.

### Batch 5 — LLM providers, content-addressed cache, budget monitor

- **Depends-On:** Batches 0–1.
- **Files:** `v3/rlm_scheme/llm_provider.py`, `cache.py`, `budget.py`.
- **Requirements:**
  - R-5.1 [MUST] `LLMProvider` protocol + a real provider (Anthropic) and `MockLLMProvider` per C.4 (deterministic, content-bearing).
  - R-5.2 [MUST] Retry/backoff/adaptive-concurrency per C.3.
  - R-5.3 [MUST] `LLMCache` key = sha256 of `{instruction, input_text, model_id, temperature, json_mode}` (response excluded); stored in the `cache` namespace; `get/put`.
  - R-5.4 [MUST] `BudgetMonitor` tracks calls/tokens/cost/concurrency against `ExecutionPolicy`; `would_exceed(req)` consulted by the host before serving an `llm_call` (invariant I-6); degradation per `budget-policy` (`switch_model` to fallback, or `checkpoint_and_stop`).
- **Tests:** mock determinism + the C.4 distinct-output rule; cache hit on identical key, miss on any field change; budget trip switches model then stops; retry honors 429 then succeeds (fake transport).
- **Acceptance gates:** a suite using a constant-output provider FAILS (proving E's anti-trivial rule); cache reuse demonstrably zeroes cost on a repeat.

### Batch 6 — Dry runner (simulated execution → estimates)

- **Depends-On:** Batches 3, 4, 5.
- **Files:** `v3/rlm_scheme/dry_run.py`.
- **Requirements:**
  - R-6.1 [MUST] `DryRunner.run(plan)` instantiates the artifact(s), runs the runtime in **simulate** mode, and builds `DryRunRecord` from the measured `stats` (decision D — no hardcoded per-name formulas; no `structural-profile`).
  - R-6.2 [MUST] Token/cost estimates from `stats.calls_by_model` × registry (C.2); `cache_hits_expected` by replaying simulated call keys against the live cache.
  - R-6.3 [MUST] For chains, simulate each step with `$previous` bound to a synthetic context of the predecessor's simulated output; aggregate stats.
- **Tests:** `batch_extract_reduce` dry run → 125 calls, cost within tolerance; adding a 17th hypothetical template needs zero dry-runner edits (parametrized test instantiates an arbitrary registered template); chain dry run aggregates two steps; second run predicts 100 cache hits.
- **Acceptance gate:** dry-run stats equal the live execution's host-counted calls for the same artifact (cross-checked in Batch 8).

### Batch 7 — Verification engine

- **Depends-On:** Batches 1, 3, 5, 6.
- **Files:** `v3/rlm_scheme/verification.py`.
- **Requirements:**
  - R-7.1 [MUST] Implement all 23 checks (E) reading only fields that exist in I.1; never short-circuit; `decision=pass` iff zero `fail`.
  - R-7.2 [MUST] Policy defaults per E.
  - R-7.3 [MUST] Persist `VerificationRecord` (`ver_`).
- **Tests:** one pass + one fail fixture per check (46 cases); `dry_run_fresh` fails when slot values changed since the dry run; `py_exec_policy`/`llm_generated_code_policy`/`gate_policy` gate correctly; a clean plan passes all 23.
- **Acceptance gates:** exactly 23 checks registered; no check references an attribute absent from I.1 (static assertion over the record fields).

### Batch 8 — Executor, py-exec sandbox, gates, checkpoints, streaming

- **Depends-On:** Batches 4–7.
- **Files:** `v3/rlm_scheme/executor.py`, `python_bridge.py` (PyExecSandbox), `gate.py`, `checkpoint.py`, `trace.py`, `streaming.py`.
- **Requirements:**
  - R-8.1 [MUST] `Executor.execute(plan, policy, runtime_options)`: dry-run-fresh check → run all 23 checks → on pass set state `verifying`→`running`, spawn runtime in **live** mode, service J.3 effects.
  - R-8.2 [MUST] Invariant I-2: no live `llm_call` is served before a `pass` verification exists.
  - R-8.3 [MUST] `PyExecSandbox` runs each `py_exec` in a fresh `python -I -S` subprocess with rlimits + process-group kill (B.5, decision C). In-process `exec` is forbidden (`grep` gate).
  - R-8.4 [MUST] `GateManager` persists a gate, sets state `suspended_gate`, returns `status:"suspended"`; `resume_execution` injects the decision and resumes.
  - R-8.5 [MUST] `CheckpointManager` writes `CheckpointRecord` to the `checkpoints` namespace (A.7); resume-from-checkpoint supported.
  - R-8.6 [MUST] Host independently counts served `llm_call`s for budget (I-6) and reconciles against `done.stats` (J.5).
  - R-8.7 [MUST] Trace records every call (`call_`), py_exec, checkpoint, gate, and partial_result; `get_execution_trace` returns it.
  - R-8.8 [MUST] Streaming: when `stream:true`, forward `partial_result` as MCP notifications.
- **Tests:** end-to-end `batch_extract_reduce` with MockLLM → finished, 125 host-counted calls == dry-run estimate (closes Batch 6 gate); py-exec computes a sum in a subprocess; py-exec timeout is killed; py-exec without `allow_py_exec` blocked at verification; gate suspends + resumes; checkpoint write+resume; sandbox host-side budget trip returns `budget_exceeded`.
- **Acceptance gates:** no in-process `exec(`/`eval(` of py-exec code anywhere; full happy path (H) green with MockLLM.

### Batch 9 — Chain executor

- **Depends-On:** Batch 8.
- **Files:** `v3/rlm_scheme/chain.py`.
- **Requirements:**
  - R-9.1 [MUST] Execute `TemplateChain` steps in order; step 0 runs normally; each step ≥ 1 resolves `"$previous"` to an auto-created context `ctx_…` holding the prior step's output before instantiation.
  - R-9.2 [MUST] Per-step verification + dry-run-fresh; aggregate stats and per-step results in the response.
  - R-9.3 [MUST] Cache reuse across steps/runs via the content cache (Batch 5).
- **Tests:** two-step chain (`batch_map`→`tree_synthesis`) end-to-end; `$previous` resolves to the right context; second run with changed final step hits cache on step 0.
- **Acceptance gate:** chain walkthrough (H, chained variant) green.

### Batch 10 — MCP server (the 10 tools) and app wiring

- **Depends-On:** Batches 1–9.
- **Files:** `v3/rlm_scheme/mcp_server.py`, `app.py`.
- **Requirements:**
  - R-10.1 [MUST] Expose exactly the 10 tools (§2.4) via FastMCP with the I.3 response envelope and the H shapes.
  - R-10.2 [MUST] `app.py::build_app(root)` constructs each component once per I.2 and injects it; no module-level singletons; no references to undefined classes.
  - R-10.3 [MUST] Tool errors return the I.3 error envelope, never raw exceptions; `ReclassificationNeeded` → `status:"needs_reclassification"`; gate → `status:"suspended"`.
- **Tests:** each tool round-trips against `build_app(tmp_path)` with MockLLM; tool count == 10; the full H sequence over the MCP surface produces the documented shapes (IDs match A.6 grammar).
- **Acceptance gates:** exactly 10 tools registered; `build_app` wires with no errors; every response validates against I.

### Batch 11 — Docs, examples, CI

- **Depends-On:** Batch 10.
- **Files:** `v3/README.md`, `v3/examples/*.py`, CI config.
- **Requirements:**
  - R-11.1 [MUST] README documents the two-process architecture, the py-exec isolation **limitation** (B.5), and how to run with a real Racket toolchain.
  - R-11.2 [MUST] All README example IDs validate against A.6 (v1 shipped `dr_789`, `plan_xyz` — forbidden here).
  - R-11.3 [MUST] CI runs the full suite including Racket-dependent Batch 4/8 tests; a job asserts the spec-deviation log (§1 rule 3) is empty or reviewed.
- **Tests:** doctest/lint the README examples for ID validity; CI green on a clean checkout.
- **Acceptance gates:** CI green; README examples valid; `SPEC-DEVIATIONS.md` reviewed.

---

## Appendix K: Definition of done

The build is complete when, on a clean checkout with a Racket toolchain and `MockLLMProvider`:

1. All batch tests pass (0–11).
2. The cardinality assertions hold: 13 task shapes, 11 data shapes, 7 execution states, 3 error policies, 7 reset scopes, 8 ID prefixes, 8 store namespaces, 10 MCP tools, 16 templates, 23 verification checks, 6 slot types.
3. The happy path (H) and the chained/streaming/cache-reuse path run end-to-end.
4. Dry-run stats equal host-counted live stats for the same artifact (I-3).
5. No live LLM call precedes a passing verification (I-2).
6. No blacklist code, no in-process `exec` of py-exec code, no module-level singletons referencing undefined classes (the three v1 anti-patterns), each enforced by a grep-style gate.
7. `SPEC-DEVIATIONS.md` is empty or every entry is reviewed and justified.


