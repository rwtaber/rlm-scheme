# RLM-Scheme Greenfield Implementation Plan

**Status:** Normative implementation plan.
**Audience:** an implementing agent.

This document defines a fresh implementation of RLM-Scheme: an MCP server for structured LLM orchestration. Agents do not write Scheme. They load context, ask the server to plan a strategy, dry-run that strategy, and execute only after verification passes.

The central design is intentionally narrow:

- **Python owns state and I/O.** MCP tools, persistent store, planning, verification, LLM providers, cache, budget accounting, and Python code execution all live in Python.
- **Racket owns orchestration control flow.** Template bodies are Racket S-expressions evaluated in a sandboxed subprocess.
- **Code and data never mix.** Slot values are passed as JSON data and read with `(slot 'name)`. Template files never contain textual slot markers.
- **Dry runs execute the same body as live runs.** Simulation mode returns synthetic LLM results and measures calls, concurrency, recursion depth, gates, checkpoints, and Python phases.
- **Simple suspension model.** Gates are live-process pauses only. Checkpoints are persisted audit/restart data, not transparent continuations.
- **Template parsing uses S-expressions, not regular expressions.** Metadata and template bodies are parsed structurally.

---

## 1. Precedence and Conventions

1. `[MUST]`, `[SHOULD]`, and `[MAY]` carry their RFC-2119 meanings.
2. Exact cardinalities are normative and tested.
3. If two statements conflict, this precedence order applies:
   1. Appendix A and Appendix H.
   2. Batch requirements.
   3. Batch tests.
   4. Prose and examples.
4. Any conflict found during implementation **[MUST]** be recorded in `SPEC-DEVIATIONS.md` with both locations and the winning rule.
5. All examples **[MUST]** use IDs that match Appendix A.6.

---

## 2. System Overview

### 2.1 Runtime Architecture

```
Python host
  MCP tools
  Store / ContextStore
  TemplateRegistry / Instantiator
  Classifier / Planner
  DryRunner / VerificationEngine / Executor
  LLM providers / cache / budget monitor
  GateManager / CheckpointManager / TraceStore
  PyExecSandbox
        |
        | JSON Lines over stdin/stdout
        v
Racket runtime subprocess
  wire protocol
  primitive library
  sandboxed template evaluator
```

Python sends a `run` message to Racket with:

- mode: `simulate` or `live`;
- artifact ID;
- parsed template body source;
- slot values;
- referenced context data;
- execution limits.

Racket evaluates the template body. In simulate mode, primitives return synthetic values locally. In live mode, effectful primitives send JSON-line requests to Python and block until Python replies.

### 2.2 User-Facing Lifecycle

```
load_context         -> ctx_...
plan_strategy        -> plan_...
dry_run_strategy     -> dry_... and art_...
execute_strategy     -> ver_... and exec_...
get_execution_trace  -> trace for exec_...
```

No live LLM call may happen before verification passes.

### 2.3 MCP Tools

The server exposes exactly 10 tools:

| Tool | Purpose |
|---|---|
| `load_context` | Store JSON input data and metadata. |
| `plan_strategy` | Classify the task and select template invocation or chain. |
| `dry_run_strategy` | Instantiate artifact(s), simulate execution, estimate cost. |
| `execute_strategy` | Verify and execute live. |
| `resume_execution` | Resume a currently suspended gate. |
| `get_execution_trace` | Return trace events for an execution. |
| `list_templates` | Return template catalog summary. |
| `describe_template` | Return metadata and body for one template. |
| `get_record` | Fetch a stored record by ID. |
| `reset` | Clear state by reset scope. |

---

## Appendix A: Taxonomies and Identifiers

### A.1 TaskShape: exactly 13

`Direct`, `Batch`, `Synthesize`, `Search`, `Refine`, `Compare`, `Classify`, `Pipeline`, `Generate`, `Decompose`, `Validate`, `Aggregate`, `Composite`.

### A.2 DataShape: exactly 11

`FlatList`, `Hierarchy`, `Singular`, `ChunkedSingular`, `Graph`, `TimeSeries`, `Tabular`, `Multimodal`, `Paired`, `KeyValue`, `Unknown`.

`Unknown` is the fallback when metadata is insufficient.

### A.3 ExecutionState: exactly 7

`pending`, `verifying`, `running`, `suspended_gate`, `finished`, `failed`, `cancelled`.

Allowed transitions:

| From | To |
|---|---|
| `pending` | `verifying`, `failed`, `cancelled` |
| `verifying` | `running`, `failed`, `cancelled` |
| `running` | `suspended_gate`, `finished`, `failed`, `cancelled` |
| `suspended_gate` | `running`, `failed`, `cancelled` |

All other transitions **[MUST]** raise `InvalidStateTransition`.

### A.4 ErrorPolicy: exactly 3

`fail_fast`, `skip_and_log`, `retry_then_skip`.

### A.5 ResetScope: exactly 7

| Scope | Clears |
|---|---|
| `contexts` | `contexts` |
| `plans` | `plans` |
| `executions` | `executions`, `dry_runs`, `verifications`, `checkpoints`, `traces` |
| `artifacts` | `artifacts` |
| `cache` | `cache` |
| `gates` | pending in-memory gate records |
| `all` | all namespaces and pending in-memory gate records |

### A.6 IDs: exactly 9 prefixes

Grammar: `^(ctx|plan|dry|exec|art|ver|call|ckpt|gate)_[0-9a-f]{16}$`

| Prefix | Record | Generation |
|---|---|---|
| `ctx_` | `ContextRecord` | random 16 hex |
| `plan_` | `PlanRecord` | random 16 hex |
| `dry_` | `DryRunRecord` | random 16 hex |
| `exec_` | `ExecutionRecord` | random 16 hex |
| `art_` | `ArtifactRecord` | first 16 hex of artifact hash |
| `ver_` | `VerificationRecord` | random 16 hex |
| `call_` | LLM call trace ID | random 16 hex |
| `ckpt_` | `CheckpointRecord` | random 16 hex |
| `gate_` | `GateInfo` | random 16 hex |

### A.7 Store namespaces: exactly 9

`contexts`, `plans`, `artifacts`, `dry_runs`, `verifications`, `executions`, `cache`, `checkpoints`, `traces`.

Gates are intentionally not a store namespace. A pending gate belongs to the live `GateManager` process table and is lost if the server process dies. This matches Appendix G.4: gates are not resumable after process death.

---

## Appendix B: Template Language

### B.1 File Format

One template is one `.rkt` file under `templates/`.

The first top-level form **[MUST]** be:

```racket
(define-meta
  (name "batch_extract_reduce")
  (version "1.0.0")
  (task-shapes (Batch Synthesize Composite))
  (data-shapes (FlatList Tabular))
  (slots
    (context_id (type context-ref) (required #t))
    (items_path (type string) (required #f) (default "$"))
    (map_instruction (type string) (required #t))
    (reduce_instruction (type string) (required #t))
    (map_model (type model-alias) (required #f) (default "fast_text_model"))
    (reduce_model (type model-alias) (required #f) (default "quality_text_model"))
    (max_concurrent (type integer) (required #f) (default 10) (min 1) (max 100))
    (branch_factor (type integer) (required #f) (default 5) (min 2) (max 20))
    (json_mode (type boolean) (required #f) (default #t)))
  (output-schema "{\"type\":\"string\"}")
  (budget-policy (on-exceed fail))
  (cacheable #t)
  (streamable #t)
  (uses-py-exec #f)
  (uses-llm-generated-code #f))
```

Every remaining top-level form is the executable template body. The final effect **[MUST]** be `(finish value)`.

There are no `{{slot}}` markers. Parameters are read only through `(slot 'name)`.

### B.2 Slot Types: exactly 6

`string`, `integer`, `number`, `boolean`, `model-alias`, `context-ref`.

Validation rules:

- required slots must be present;
- undeclared slots are rejected;
- type checks and numeric min/max bounds are enforced;
- `model-alias` must exist in the model registry;
- `context-ref` must be a valid `ctx_` ID, except `"$previous"` is allowed inside chain steps after step 0;
- string slot contents are never inspected for code-like text.

### B.3 S-Expression Parsing

Template metadata **[MUST]** be parsed as S-expressions.

Implementation requirements:

- tokenize and parse balanced S-expressions structurally;
- preserve body text exactly after the first metadata form for hashing;
- walk parsed forms to find `(slot 'name)` references;
- walk parsed forms to find free identifiers;
- do not use regular expressions for S-expression parsing or body analysis.

Using regular expressions for simple non-S-expression tasks, such as validating ID strings, is allowed.

### B.4 Primitive Library

Templates run with only these primitives plus a small pure-form allowlist from `racket/base`.

```racket
(llm-query instruction input
           #:model alias
           #:temperature [t 0]
           #:json [json? #f]
           #:max-tokens [n #f]
           #:node-id [node-id #f])

(llm-query-async ...same kwargs...)
(await p)
(await-all promises)
(await-any promises)

(map-async f items #:max-concurrent [k 10]
                   #:error-policy [p 'fail_fast]
                   #:node-id [id #f])
(parallel thunk ...)
(race thunks #:timeout-seconds [s #f])
(tree-reduce f items #:branch-factor [b 5] #:node-id [id #f])
(fold-sequential f init items #:checkpoint-every [n #f] #:node-id [id #f])
(sequence expr ...)
(choose pred a b)
(iterate-until pred f init #:max-iterations n)
(recursive-spawn f input #:max-depth d #:branch-factor [b 5])
(memoized f)
(with-validation validate-f produce-f #:max-retries [r 2])
(try-fallback primary-thunk fallback-thunk)

(gate payload #:label [label #f])
(checkpoint data #:node-id [node-id #f])
(partial-result node-id index value)
(finish value)

(py-exec code #:input [input 'null]
              #:allowed-imports [imports '()]
              #:timeout-seconds [seconds 30])
(py-eval expr-string)

(slot name-symbol)
(context-items context-id path)
(join-values values)
```

`py-eval` is syntax sugar for `py-exec` and is governed by the same policy. Any template that uses `py-eval` **[MUST]** declare `(uses-py-exec #t)`.

### B.5 Output Schema Subset

`output-schema` is a JSON string containing a restricted JSON Schema object.

Allowed keywords:

- `type`: one of `string`, `number`, `integer`, `boolean`, `object`, `array`, `null`;
- `properties`: object mapping property names to schemas;
- `required`: array of property-name strings;
- `items`: schema for array items;
- `enum`: array of scalar values;
- `additionalProperties`: boolean.

No other JSON Schema keywords are supported. A schema using an unsupported keyword fails `output_schema_valid`. Final execution results are validated against this subset after `(finish value)`.

### B.6 Reference Template Body

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

---

## Appendix C: Models, Providers, Cache, and Budget

### C.1 Model Registry

`config/models.json` is a JSON array of `ModelRegistryEntry`.

Required aliases:

- `fast_text_model`
- `quality_text_model`
- `vision_model`

Templates reference aliases only. Providers resolve aliases to concrete model IDs at call time. `vision_model` is reserved for multimodal contexts: when `DataShape.Multimodal` is selected and a template has a generic model slot, the planner defaults that slot to `vision_model` unless the caller supplies another alias.

### C.2 Provider Retry

Provider calls use up to 3 attempts. Retry on HTTP 429, HTTP 5xx, and timeout. Do not retry other HTTP 4xx responses. Backoff is 1s then 4s with jitter.

### C.3 MockLLMProvider

The mock provider **[MUST]** be deterministic and content-bearing. Distinct `(model, instruction, input)` triples must produce distinct output. Tests must fail if a constant-output provider is substituted for combinator integration tests.

### C.4 Cache Keys

`LLMCache` key material:

```json
{
  "instruction": "...",
  "input_text": "...",
  "model_id": "...",
  "temperature": 0,
  "json_mode": true,
  "max_tokens": null
}
```

The response is not part of the key.

### C.5 Budget Policy

The simplest runtime behavior is normative:

- if a live LLM request would exceed policy, Python replies with `budget_exceeded`;
- the runtime applies the active `ErrorPolicy`;
- model switching is not performed implicitly.

Templates may declare a fallback model for future use, but this implementation records it only as metadata and verifies that the alias exists. Runtime model changes would make the verified artifact less clear, so they are intentionally out of scope.

### C.6 Token and Cost Estimation

Each simulated `llm-query` emits a `SimulatedCall`. Python applies this estimate:

```python
prompt_tokens = ceil((call.instruction_chars + call.input_chars) / 4)
completion_tokens = max_tokens if max_tokens is not None else model.default_completion_estimate
```

The runtime computes `instruction_chars` from the instruction string length and `input_chars` from the canonical input string length. Canonical input means JSON with sorted keys and compact separators for structured values; strings are used directly.

Dry-run totals:

```python
prompt_tokens(call) = ceil((call.instruction_chars + call.input_chars) / 4)
completion_tokens(call) = call.max_tokens if call.max_tokens is not None else model.default_completion_estimate
prompt_total = sum(prompt_tokens(call) for call in calls)
completion_total = sum(completion_tokens(call) for call in calls)
token_total = prompt_total + completion_total
low_cost = sum((prompt_tokens(call) * input_rate + completion_tokens(call) * output_rate) / 1_000_000 for call in calls)
high_cost = sum((prompt_tokens(call) * input_rate + high_completion_tokens(call) * output_rate) / 1_000_000 for call in calls)
```

The high-cost completion term is capped at `model.max_output_tokens`:

```python
high_completion_tokens(call) = min(ceil(completion_tokens(call) * 2.5), model.max_output_tokens)
```

`input_rate` and `output_rate` come from `ModelRegistryEntry`. `py-exec`, gates, checkpoints, and partial results do not add tokens.

---

## Appendix D: Instantiation, Hashing, and Dry Runs

### D.1 Instantiation

`Instantiator.instantiate(template, slot_values)`:

1. Applies defaults.
2. Validates slots.
3. Canonicalizes slot values with `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
4. Computes `body_hash = sha256(body_text_utf8)`.
5. Computes `artifact_hash = sha256(canonical_json({template_name, template_version, body_hash, slot_values}))`.
6. Stores `ArtifactRecord`.

`artifact_id = "art_" + artifact_hash[:16]`.

Identical inputs must produce identical artifact records.

### D.2 Simulate Mode

In simulate mode:

- `llm-query` returns a synthetic content-bearing string or JSON value;
- `py-exec` is counted but not run;
- `gate` is counted and returns `{"sim": true, "decision": "approved"}`;
- `checkpoint` is counted and returns `ckpt_0000000000000000`;
- `partial-result` is counted as a trace event if tracing is enabled.

The terminal `done` message includes `stats: SimulationStats` and `calls: list[SimulatedCall]` as defined in Appendix H. Python builds `DryRunRecord.call_graph` and `CostEstimate` from `calls`; it does not infer those values from aggregate stats alone.

Simulation is deterministic and estimates upper-bound work for dynamic control forms:

- `llm-query`: counts one call at the current dependency depth, records model alias and token estimate.
- `llm-query-async`: starts the same simulated work in a logical concurrent branch.
- `await`: waits for one branch; the branch's depth contributes to the current continuation.
- `await-all`: waits for all branches; all started branches count, and continuation depth uses the maximum branch depth.
- `await-any`: all already-started branches count; the winner is the first branch in creation order, and continuation depth uses that branch depth.
- `map-async`: starts every item, bounded by `#:max-concurrent`; all items count; observed max concurrency is `min(item_count, max_concurrent)`.
- `parallel`: all thunks count; continuation depth uses the maximum branch depth.
- `race`: starts all thunks; all started thunks count; the winner is the first thunk in source order.
- `tree-reduce`: runs reduce levels until one value remains; each group reducer call counts if it calls effects. Critical path is producer path plus one reducer path per tree level.
- `fold-sequential`: runs items in order; effect depths add sequentially.
- `iterate-until`: simulate mode runs exactly `#:max-iterations` iterations, regardless of synthetic predicate result.
- `with-validation`: simulate mode runs `produce-f` and `validate-f` exactly `max_retries + 1` times.
- `try-fallback`: simulate mode runs both branches and returns the primary branch result; this is an upper-bound estimate.
- `recursive-spawn`: simulate mode expands a full tree through `max-depth` with `branch-factor`; `recursive_depth` is the maximum depth reached. If the supplied function does not recurse, only the called body effects count.
- `memoized`: repeated calls with the same canonicalized function identity and argument count once in simulate mode; later hits return the first synthetic result and do not add LLM calls or token estimates.

`critical_path_calls` is the longest dependent chain of simulated LLM calls. Parallel branches contribute their maximum branch path, not their sum. Sequential forms contribute the sum of their dependent paths.

Cache-hit prediction is deliberately simple for now: dry runs report `cache_hits_expected = 0`. Accurate replay of simulated call keys can be added later without changing the public lifecycle.

---

## Appendix E: Verification Engine

`VerificationEngine.verify(plan, artifact, dry_run, policy) -> VerificationRecord` runs all checks and never short-circuits. The decision is `pass` iff no check has status `fail`.

Exactly 22 checks:

| # | Name | Severity | Rule |
|---|---|---|---|
| 1 | `artifact_exists` | fail | artifact is stored |
| 2 | `artifact_hash` | fail | recomputed hash matches |
| 3 | `template_known` | fail | template name/version exists |
| 4 | `slot_schema` | fail | slots revalidate |
| 5 | `context_exists` | fail | all context-ref slots resolve |
| 6 | `context_shape_compatible` | warn | context shape matches template or is `Unknown` |
| 7 | `primitive_allowlist` | fail | template registration scan was clean |
| 8 | `model_aliases_resolve` | fail | all model aliases exist |
| 9 | `call_count_limit` | fail | simulated calls <= policy |
| 10 | `critical_path_limit` | warn | simulated critical path <= policy |
| 11 | `concurrency_limit` | fail | simulated concurrency <= policy |
| 12 | `token_budget` | fail | estimated tokens <= policy |
| 13 | `cost_budget` | fail | estimated high cost <= policy |
| 14 | `recursion_depth_limit` | fail | simulated depth <= policy |
| 15 | `dry_run_fresh` | fail | dry run artifact matches current plan |
| 16 | `output_schema_valid` | fail | output schema conforms to the B.5 subset |
| 17 | `py_exec_policy` | fail | py-exec requires policy allow |
| 18 | `llm_generated_code_policy` | fail | generated code requires policy allow |
| 19 | `gate_policy` | fail | gates require policy allow |
| 20 | `timeout_sane` | fail | timeout is within policy |
| 21 | `checkpoint_writable` | warn | checkpoints namespace is writable |
| 22 | `fallback_model_valid` | fail | declared fallback model exists if present |

Policy defaults:

```python
max_llm_calls = 500
max_concurrency = 50
max_critical_path = 25
max_tokens = 2_000_000
max_cost_usd = 10.00
max_recursion_depth = 5
allow_py_exec = False
allow_llm_generated_code = False
allow_gates = True
max_timeout_seconds = 3600
```

---

## Appendix F: Classifier and Planner

### F.1 Task Classification

`classify_task_shape(hints) -> tuple[TaskShape, list[str]]`.

Hint fields:

- `item_count`
- `independent`
- `output_type`: `one`, `list`, or `per_item`
- `operation`
- `has_second_phase`
- `sub_operations`
- `ordered`
- `latency_priority`
- `ambiguous_items`
- `has_testable_predicate`
- `candidate_count`
- `until_condition`
- `recursive`
- `process_parts`
- `estimated_context_tokens`

Rules:

1. If `has_second_phase` and at least two `sub_operations`, return `Composite`.
2. If `item_count <= 1`, `output_type == "one"`, and no second phase:
   - `operation == "generate"` -> `Generate`;
   - `operation in {"refine", "improve", "iterate"}` -> `Refine`;
   - otherwise `Direct`.
3. If `item_count >= 3` and `independent`:
   - `label` -> `Classify`;
   - `check` or `grade` -> `Validate`;
   - `aggregate`, `compute`, or `stats` -> `Aggregate`;
   - otherwise `Batch`.
4. If `item_count >= 3` and not `independent`:
   - at least two `sub_operations` -> `Pipeline`;
   - otherwise `Synthesize`.
5. For one or two items, or unknown item count:
   - `generate` -> `Generate`;
   - `refine`, `improve`, `iterate` -> `Refine`;
   - `decompose`, `split`, `parse` with list-like output -> `Decompose`;
   - `compare`, `select`, `choose`, `rank` -> `Search` if latency priority else `Compare`;
   - `aggregate`, `compute`, `stats` -> `Aggregate`;
   - two items with one output -> `Synthesize`;
   - otherwise `Direct`.

Every absent hint read by the classifier is added to `assumed_fields`.

### F.2 Data Shape Classification

`classify_data_shape(data, metadata) -> DataShape`.

Rules, in order:

1. If `metadata.data_shape` is one of Appendix A.2, return it.
2. If `metadata.edges` is a non-empty list, return `Graph`.
3. If `data` is a list:
   - empty list -> `FlatList`;
   - list of two-element lists/objects with `left/right` or `a/b` keys -> `Paired`;
   - `metadata.ordered == True` and temporal keys are present -> `TimeSeries`;
   - list of objects with `children` or `parent` keys -> `Hierarchy`;
   - any item has non-text modality metadata -> `Multimodal`;
   - list of objects with identical scalar keys and at least 2 rows -> `Tabular`;
   - otherwise `FlatList`.
4. If `data` is a dict:
   - contains `nodes` and `edges` -> `Graph`;
   - contains `chunks` list -> `ChunkedSingular`;
   - all values are scalar -> `KeyValue`;
   - contains nested `children` -> `Hierarchy`;
   - otherwise `Singular`.
5. If `data` is a string:
   - metadata says chunked -> `ChunkedSingular`;
   - otherwise `Singular`.
6. Otherwise return `Unknown`.

### F.3 Template Selection

The planner chooses from exactly 16 templates:

| Template | Shapes | Composition |
|---|---|---|
| `direct_call` | Direct, Synthesize, Classify | one LLM call |
| `direct_json_extract` | Direct, Decompose | JSON LLM call + validation |
| `batch_map` | Batch, Classify, Validate, Generate | `map-async` |
| `batch_extract_reduce` | Batch, Synthesize, Composite | `map-async` + `tree-reduce` |
| `batch_extract_fold` | Batch, Synthesize | `map-async` + `fold-sequential` |
| `ordered_synthesis_fold` | Synthesize, Generate, Validate | `fold-sequential` |
| `tree_synthesis` | Synthesize | `tree-reduce` |
| `compare_candidates` | Compare, Search | parallel comparison |
| `race_candidates` | Search | race |
| `refine_until_valid` | Refine, Search, Generate | bounded iteration |
| `bounded_critique_refine` | Refine | critique/refine loop |
| `tiered_review` | Batch, Classify, Validate | cheap pass then expensive pass |
| `tabular_extract_aggregate` | Aggregate | extract then py-exec aggregate |
| `decompose_then_batch` | Decompose, Composite | decompose then map |
| `recursive_decompose` | Decompose | recursive spawn |
| `code_interpreter` | Direct, Aggregate | LLM code + py-exec |

Selection rules:

- Direct: list output -> `direct_json_extract`, otherwise `direct_call`.
- Batch: per-item output -> `batch_map`; ordered combined output -> `batch_extract_fold`; `ambiguous_items == true` -> `tiered_review`; otherwise `batch_extract_reduce`.
- Synthesize: ordered -> `ordered_synthesis_fold`; hierarchical -> `tree_synthesis`; otherwise `direct_call` when context fits, else `tree_synthesis`.
- Search: finite candidates + latency priority -> `race_candidates`; finite candidates -> `compare_candidates`; otherwise `refine_until_valid`.
- Refine: `has_testable_predicate == true` -> `refine_until_valid`; otherwise `bounded_critique_refine`.
- Compare: `compare_candidates`.
- Classify: one item -> `direct_call`; `ambiguous_items == true` -> `tiered_review`; otherwise `batch_map`.
- Pipeline: build a chain from sub-operations.
- Generate: `until_condition == true` -> `refine_until_valid`; otherwise `batch_map`.
- Decompose: `recursive == true` -> `recursive_decompose`; `process_parts == true` -> `decompose_then_batch`; otherwise `direct_json_extract`.
- Validate: `ambiguous_items == true` -> `tiered_review`; otherwise `batch_map`.
- Aggregate: `tabular_extract_aggregate`.
- Composite: use `batch_extract_reduce` for extract/synthesize two-phase tasks; otherwise build a chain.

Derived predicates:

- `finite candidates` means `candidate_count is not None and candidate_count > 0`.
- `hierarchical` means `data_shape == Hierarchy`.
- `one item` means `item_count == 1`.
- `context fits` means `estimated_context_tokens <= floor(quality_text_model.context_window_tokens * 0.8)`. If `estimated_context_tokens` is absent, derive it as `ceil(len(canonical_json(context.data)) / 4)`.

---

## Appendix G: Runtime Protocol

JSON Lines over stdin/stdout.

### G.1 Startup

Runtime emits:

```json
{"type":"ready","protocol":"1.0"}
```

### G.2 Run

Host sends:

```json
{
  "type": "run",
  "mode": "simulate",
  "artifact_id": "art_a1b2c3d4e5f60718",
  "body": "(define ...)",
  "slot_values": {},
  "contexts": {},
  "limits": {"max_recursion_depth": 5}
}
```

### G.3 Live Effect Requests

Runtime may send:

```json
{"type":"llm_call","id":"call_001a2b3c4d5e6f70","node_id":"extract","instruction":"...","input":"...","model":"fast_text_model","temperature":0,"json":true,"max_tokens":null}
{"type":"py_exec","id":"call_111a2b3c4d5e6f70","code":"...","input":{},"allowed_imports":["json"],"timeout_seconds":30}
{"type":"checkpoint","id":"ckpt_77a8b9c0d1e2f300","node_id":"fold","data":{}}
{"type":"gate","id":"gate_77a8b9c0d1e2f300","label":"review","payload":{}}
{"type":"partial_result","node_id":"extract","index":7,"value":{}}
```

Python replies with matching IDs for `llm_call`, `py_exec`, `checkpoint`, and `gate`. `partial_result` is fire-and-forget and has no reply.

### G.4 Gates

The simplest behavior is normative:

- a gate suspends a live execution and keeps the Racket subprocess alive;
- `execute_strategy` returns `status: "suspended"` with `execution_id` and `gate`;
- `resume_execution` supplies a decision to the waiting subprocess;
- gate suspension has a timeout; timeout fails the execution;
- gates are not resumable after process death.

### G.5 Checkpoints

Checkpoints persist data and trace location. They do not capture Racket continuations. Resume-from-checkpoint means starting a new execution from a checkpoint-aware template or chain step, not restoring an arbitrary stack frame.

### G.6 Terminal Messages

```json
{"type":"done","value":{},"stats":{"llm_calls":1,"critical_path_calls":1,"max_concurrency":1,"recursive_depth":0,"checkpoints":0,"python_phases":0,"gates":0,"calls_by_model":{"fast_text_model":1}},"calls":[{"call_id":"call_001a2b3c4d5e6f70","node_id":"extract","model":"fast_text_model","instruction_chars":12,"input_chars":42,"max_tokens":null,"json_mode":true,"depth":1}]}
{"type":"error","error_code":"runtime_error","message":"...","trace":"..."}
```

The `done.stats` object has the same shape in live and simulate mode: `SimulationStats`. In simulate mode, `calls` is the complete list of simulated LLM calls. In live mode, `calls` may be omitted because Python already records served calls. Python independently counts live LLM calls and reconciles its count with runtime stats.

---

## Appendix H: Python Interfaces

All Pydantic models use strict validation.

```python
class ContextMetadata(BaseModel):
    data_shape: DataShape = DataShape.Unknown
    item_count: int = 0
    independent: bool = True
    ordered: bool = False
    modality: list[str] = ["text"]
    edges: list[dict] | None = None
    chunked: bool = False
    extra: dict[str, Any] = {}

class ContextRecord(BaseModel):
    context_id: str
    name: str
    data: list | dict | str | int | float | bool | None
    metadata: ContextMetadata
    created_at: float

class Classification(BaseModel):
    task_shape: TaskShape
    constituent_shapes: list[TaskShape] = []
    data_shape: DataShape
    hints_complete: bool
    assumed_fields: list[str] = []
    rationale: str = ""

class Alternative(BaseModel):
    template_name: str
    tradeoff: str

class TemplateInvocation(BaseModel):
    kind: Literal["template_invocation"] = "template_invocation"
    template_name: str
    template_version: str
    slot_values: dict[str, Any]

class ChainStep(BaseModel):
    step: int
    template_name: str
    template_version: str
    slot_values: dict[str, Any]

class TemplateChain(BaseModel):
    kind: Literal["template_chain"] = "template_chain"
    steps: list[ChainStep]

class PlanRecord(BaseModel):
    plan_id: str
    context_id: str
    task: str
    hints: dict[str, Any] = {}
    classification: Classification
    recommended: TemplateInvocation | TemplateChain
    alternatives: list[Alternative] = []
    created_at: float

class ArtifactRecord(BaseModel):
    artifact_id: str
    template_name: str
    template_version: str
    body_hash: str
    artifact_hash: str
    slot_values: dict[str, Any]
    primitives_used: list[str]
    uses_py_exec: bool
    uses_llm_generated_code: bool

class CallGraphNode(BaseModel):
    node_id: str
    primitive: str
    calls: int
    model: str | None = None

class TokenEstimate(BaseModel):
    prompt: int
    completion: int
    total: int

class CostRange(BaseModel):
    low: float
    high: float

class CostEstimate(BaseModel):
    estimated_tokens: TokenEstimate
    estimated_cost_usd: CostRange
    cache_hits_expected: int = 0

class SimulationStats(BaseModel):
    llm_calls: int
    critical_path_calls: int
    max_concurrency: int
    recursive_depth: int
    checkpoints: int
    python_phases: int
    gates: int
    calls_by_model: dict[str, int]

class SimulatedCall(BaseModel):
    call_id: str
    node_id: str | None = None
    model: str
    instruction_chars: int
    input_chars: int
    max_tokens: int | None = None
    json_mode: bool = False
    depth: int

class DryRunRecord(BaseModel):
    dry_run_id: str
    plan_id: str
    artifact_id: str
    simulation: SimulationStats
    estimate: CostEstimate
    call_graph: list[CallGraphNode]
    calls: list[SimulatedCall] = []
    warnings: list[str] = []
    created_at: float

class CheckResult(BaseModel):
    name: str
    status: Literal["pass", "warn", "fail"]
    severity: Literal["warn", "fail"]
    message: str

class VerificationRecord(BaseModel):
    verification_id: str
    artifact_id: str
    dry_run_id: str
    decision: Literal["pass", "fail"]
    checks: list[CheckResult]
    failed_checks: list[str] = []
    warnings: list[str] = []
    created_at: float

class ExecutionResult(BaseModel):
    value: Any
    schema_valid: bool = True

class ExecutionStats(BaseModel):
    elapsed_seconds: float = 0
    llm_calls: int = 0
    cache_hits: int = 0
    tokens: int = 0
    checkpoints_written: int = 0
    python_phases: int = 0

class GateInfo(BaseModel):
    gate_id: str
    execution_id: str
    label: str | None = None
    payload: Any
    created_at: float
    timeout_seconds: int

class ExecutionRecord(BaseModel):
    execution_id: str
    artifact_id: str
    plan_id: str
    dry_run_id: str
    verification_id: str | None = None
    state: ExecutionState
    result: ExecutionResult | None = None
    stats: ExecutionStats = ExecutionStats()
    gate: GateInfo | None = None
    error: dict[str, Any] | None = None
    created_at: float
    completed_at: float | None = None

class CheckpointRecord(BaseModel):
    checkpoint_id: str
    execution_id: str
    node_id: str | None
    data: Any
    created_at: float

class TraceEvent(BaseModel):
    ts: float
    type: str
    execution_id: str | None = None
    call_id: str | None = None
    node_id: str | None = None
    model: str | None = None
    tokens: int | None = None
    cache_hit: bool | None = None
    payload: Any = None

class TraceRecord(BaseModel):
    execution_id: str
    events: list[TraceEvent]

class ExecutionPolicy(BaseModel):
    max_llm_calls: int = 500
    max_concurrency: int = 50
    max_critical_path: int = 25
    max_tokens: int = 2_000_000
    max_cost_usd: float = 10.00
    max_recursion_depth: int = 5
    allow_py_exec: bool = False
    allow_llm_generated_code: bool = False
    allow_gates: bool = True
    max_timeout_seconds: int = 3600

class ModelRegistryEntry(BaseModel):
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

### H.1 MCP Response Envelope and Tool Schemas

Every MCP tool returns a JSON object with a `status` field.

Allowed status values:

- `ok`: successful non-suspended result;
- `error`: tool failed before producing a valid result;
- `needs_reclassification`: planner could not select a template from the supplied hints;
- `suspended`: execution paused at a live gate.

Error envelope:

```json
{"status":"error","error_code":"string","message":"string"}
```

Tool contracts:

| Tool | Required input | Optional input | `ok` output |
|---|---|---|---|
| `load_context` | `data_json` | `name`, `metadata_json` | `context_id`, `name`, `metadata`, `preview`, `next_actions` |
| `plan_strategy` | `task`, `context_id` | `hints_json` | `plan_id`, `classification`, `recommended`, `alternatives`, `next_actions` |
| `dry_run_strategy` | `plan_id` | none | `dry_run_id`, `plan_id`, `artifact`, `simulation`, `estimate`, `call_graph`, `warnings`, `next_actions` |
| `execute_strategy` | `plan_id` | `dry_run_id`, `policy_json`, `timeout_seconds`, `stream` | `execution_id`, `artifact_id`, `verification`, `result`, `execution`, `next_actions` |
| `resume_execution` | `execution_id`, `gate_id`, `decision_json` | none | same shape as `execute_strategy` |
| `get_execution_trace` | `execution_id` | none | `execution_id`, `trace` |
| `list_templates` | none | none | `templates` |
| `describe_template` | `template_name` | `template_version` | `template` |
| `get_record` | `record_id` | none | `record_id`, `namespace`, `record` |
| `reset` | `scope` | none | `scope`, `cleared` |

`execute_strategy` selects the latest dry run for `plan_id` when `dry_run_id` is omitted. If no fresh dry run exists, it returns `error_code: "dry_run_required"`. `resume_execution` is valid only while the stored execution state is `suspended_gate`.

Constructors:

```python
Store(root: Path)
ContextStore(store: Store)
TemplateRegistry(template_dir: Path, model_registry: ModelRegistry)
Instantiator(store: Store, registry: TemplateRegistry)
Classifier()
Planner(classifier: Classifier, registry: TemplateRegistry, llm_provider: LLMProvider | None)
LLMCache(store: Store)
BudgetMonitor(policy: ExecutionPolicy)
RacketRuntime(runtime_dir: Path)
DryRunner(store: Store, instantiator: Instantiator, runtime: RacketRuntime, registry: TemplateRegistry)
VerificationEngine(store: Store, registry: TemplateRegistry)
PyExecSandbox(python_bin: str = "python3")
GateManager()
CheckpointManager(store: Store)
TraceStore(store: Store)
Executor(...)
ChainExecutor(executor: Executor, store: Store)
```

All components are built once in `rlm_scheme/app.py::build_app(root: Path)`.

---

## Appendix I: Build Batches

Each batch must end with all previous tests still passing.

### Batch 0: Foundations

Files:

- `rlm_scheme/enums.py`
- `rlm_scheme/ids.py`
- `rlm_scheme/models.py`
- `config/models.json`
- `pyproject.toml`

Requirements:

- implement Appendix A enums and ID helpers;
- implement Appendix H models;
- ship required model aliases.

Tests:

- enum cardinalities;
- transition matrix;
- ID validation fixtures;
- model JSON round trips.

### Batch 1: Store and Contexts

Files:

- `rlm_scheme/store.py`
- `rlm_scheme/context_store.py`

Requirements:

- filesystem JSON store with Appendix A.7 namespaces;
- namespace guard;
- reset scopes;
- context loading and `context-items` path support for `$` and `$.field`.

Tests:

- namespace guard;
- reset matrix;
- context round trip;
- path extraction.

### Batch 2: Classifier and Planner

Files:

- `rlm_scheme/classifier.py`
- `rlm_scheme/planner.py`

Requirements:

- implement Appendix F task and data classification;
- implement template selection;
- no LLM calls unless slot gap-fill is explicitly needed.

Tests:

- at least two task fixtures per `TaskShape`;
- fixtures for every `DataShape`;
- selection table tests.

### Batch 3: Template Registry and Instantiator

Files:

- `rlm_scheme/sexpr.py`
- `rlm_scheme/template_store.py`
- `rlm_scheme/instantiator.py`
- `templates/*.rkt`

Requirements:

- S-expression parser;
- metadata parsing;
- body preservation for hashing;
- slot validation;
- all 16 templates.

Tests:

- parser nested-form fixtures;
- no regex use in S-expression parser/body analysis;
- all templates load;
- all templates use `(slot 'name)`;
- artifact hash determinism.

### Batch 4: Racket Runtime and Simulation

Files:

- `runtime/main.rkt`
- `runtime/primitives.rkt`
- `runtime/sandbox.rkt`
- `runtime/wire.rkt`
- `rlm_scheme/runtime.py`

Requirements:

- JSON-lines protocol;
- sandboxed evaluation;
- simulate mode counters;
- process-group cleanup.

Tests:

- handshake;
- sandbox escape attempts fail;
- `batch_extract_reduce` over 100 items reports 125 calls;
- runtime crash is reported cleanly.

### Batch 5: Providers, Cache, Budget, PyExec

Files:

- `rlm_scheme/llm_provider.py`
- `rlm_scheme/cache.py`
- `rlm_scheme/budget.py`
- `rlm_scheme/python_bridge.py`

Requirements:

- provider protocol;
- deterministic mock provider;
- cache key;
- budget monitor;
- `py-exec` in fresh `python -I -S` process with rlimits and timeout kill.

Tests:

- mock distinct-output tests;
- cache hit/miss fixtures;
- budget exceeded behavior;
- py-exec timeout kill.

### Batch 6: Dry Runner and Verification

Files:

- `rlm_scheme/dry_run.py`
- `rlm_scheme/verification.py`

Requirements:

- dry run by simulated runtime execution;
- cost estimate from measured calls;
- all 22 verification checks.

Tests:

- dry-run stats for canonical templates;
- one pass and fail fixture per verification check;
- no live calls before verification.

### Batch 7: Executor, Gates, Checkpoints, Trace

Files:

- `rlm_scheme/executor.py`
- `rlm_scheme/gate.py`
- `rlm_scheme/checkpoint.py`
- `rlm_scheme/trace.py`

Requirements:

- execute only after verification pass;
- live effect handling;
- live-process gate suspension and resume;
- checkpoint persistence;
- trace events.

Tests:

- end-to-end execution with `MockLLMProvider`;
- gate suspend/resume;
- gate timeout failure;
- checkpoint record written;
- host-counted calls reconcile with runtime stats.

### Batch 8: Chain Executor

Files:

- `rlm_scheme/chain.py`

Requirements:

- execute chain steps in order;
- step output becomes a new context;
- `"$previous"` resolves for later steps.

Tests:

- two-step chain end to end;
- `$previous` binding test;
- per-step stats aggregation.

### Batch 9: MCP Server and App Wiring

Files:

- `rlm_scheme/mcp_server.py`
- `rlm_scheme/app.py`

Requirements:

- expose exactly 10 MCP tools;
- construct components through `build_app(root)`;
- no module-level singletons;
- consistent response envelope.

Tests:

- tool count;
- each tool round trip;
- happy path through MCP surface.

### Batch 10: Docs and CI

Files:

- `README.md`
- `examples/*`
- CI config

Requirements:

- document architecture;
- document py-exec limitation: no network namespace isolation;
- document Racket requirement;
- validate example IDs.

Tests:

- README/example ID validation;
- full suite in CI.

---

## Appendix J: Definition of Done

The implementation is done when:

1. All batch tests pass.
2. Cardinality tests pass: 13 task shapes, 11 data shapes, 7 execution states, 3 error policies, 7 reset scopes, 9 ID prefixes, 9 store namespaces, 10 MCP tools, 16 templates, 22 verification checks, 6 slot types.
3. The happy path runs end to end with `MockLLMProvider`.
4. A two-step chain runs end to end.
5. A gate suspends and resumes while its runtime process remains alive.
6. Checkpoints are persisted and visible in traces.
7. No live LLM call occurs before a passing verification record exists.
8. Template parsing/body analysis does not use regex.
9. `SPEC-DEVIATIONS.md` is empty or every entry is reviewed.
