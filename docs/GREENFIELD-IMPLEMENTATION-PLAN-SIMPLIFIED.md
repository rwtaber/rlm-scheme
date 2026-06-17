# RLM-Scheme Greenfield Implementation Plan

**Status:** new v1 implementation plan.  
**Audience:** human maintainers and coding agents.  
**Goal:** implement a small, auditable MCP server for bounded long-context LLM workflows.

This document replaces the older greenfield plan for a first implementation. It keeps the useful core and removes features that can wait:

- no arbitrary generated code execution;
- no template substitution system;
- no durable gate/checkpoint resume;
- no multi-device scheduling optimizer;
- no imported sub-program library;
- no large verification checklist whose items overlap.

The v1 system accepts a model-authored program in a tiny total S-expression language, binds it to a context, injects host-owned resource parameters, dry-runs it to produce upper bounds, verifies those bounds against policy, then executes it once.

---

## 1. Core Invariant

No live model call may occur until all of these are true:

1. The context has been loaded into the host store.
2. The authored program parses into the v1 AST.
3. Every referenced slot and host parameter resolves.
4. The program uses only allowed forms and primitives.
5. Recursion, if present, is structurally decreasing.
6. The dry run has produced an upper bound for calls, tokens, cost, and depth.
7. Verification has passed against the exact program, exact slots, exact host parameters, exact model registry, exact policy, and exact runtime version.

Execution is deterministic except for `leaf-call` results. Re-running the same verified artifact produces the same call structure unless context data or runtime versions change, both of which invalidate verification.

---

## 2. Runtime Shape

```text
MCP client / coding agent
        |
        v
Python host
  - MCP tools
  - Store and ContextStore
  - Program parser and validator
  - Parameter planner
  - Dry runner
  - Verifier
  - Executor
  - Provider adapters
  - Trace store
        |
        | JSON Lines
        v
Racket runtime
  - AST interpreter
  - total combinators
  - effect requests only
```

Python owns state, policy, provider calls, cost accounting, persistence, and MCP. Racket interprets the verified AST and requests effects from Python. Racket never receives a whole context unless the whole context is already under the verified leaf threshold.

---

## 3. Stored Objects

ID format:

```text
^(ctx|prog|plan|dry|art|ver|exec|call)_[0-9a-f]{16}$
```

Store namespaces:

- `contexts`
- `programs`
- `plans`
- `artifacts`
- `dry_runs`
- `verifications`
- `executions`
- `traces`
- `cache`

Records:

```python
class ContextRecord(BaseModel):
    context_id: str
    data_ref: str                 # store path or inline store key
    data_shape: Literal["text", "items", "json"]
    size_chars: int
    item_count: int | None = None
    schema: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    created_at: float

class ProgramRecord(BaseModel):
    program_id: str
    source: str
    ast_hash: str
    created_at: float

class PlanRecord(BaseModel):
    plan_id: str
    context_id: str
    program_id: str
    task: str
    slot_values: dict[str, Any]
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    planner_parameters: dict[str, Any]
    model_registry_hash: str
    policy_hash: str
    runtime_version: str
    created_at: float

class ArtifactRecord(BaseModel):
    artifact_id: str
    plan_id: str
    artifact_hash: str
    program_ast_hash: str
    slot_values_hash: str
    planner_parameters_hash: str
    schemas_hash: str
    model_registry_hash: str
    policy_hash: str
    runtime_version: str

class DryRunRecord(BaseModel):
    dry_run_id: str
    artifact_id: str
    stats: dict[str, Any]
    calls: list[dict[str, Any]]
    cost: dict[str, Any]
    bounds: dict[str, Any]
    warnings: list[str] = []

class VerificationRecord(BaseModel):
    verification_id: str
    artifact_id: str
    dry_run_id: str
    decision: Literal["pass", "fail"]
    checks: list[dict[str, Any]]

class ExecutionRecord(BaseModel):
    execution_id: str
    artifact_id: str
    verification_id: str
    state: Literal["running", "succeeded", "failed", "cancelled"]
    result: Any | None = None
    stats: dict[str, Any] = {}
    created_at: float
    completed_at: float | None = None
```

`artifact_hash` is computed from the exact AST, slot values, host parameters, input/output schemas, model registry hash, policy hash, and runtime version. Verification is invalid if any of these change.

---

## 4. Context Model

`load_context` stores data outside the prompt. Context access uses explicit units:

- `size_chars`: Unicode code points in the stored text representation.
- `estimated_tokens`: `ceil(size_chars / 4)`.
- `range.start` and `range.length`: character offsets for text contexts.
- item indexes: zero-based indexes for `items` contexts.

Supported context shapes:

- `text`: one contiguous text blob.
- `items`: ordered independent items, each serializable to text.
- `json`: one JSON value; v1 may treat it as text unless a primitive explicitly supports JSON access.

Context reads:

```text
peek(context, start, length) -> Text
slice(context, start, length) -> Text
context-items(context) -> List[ItemRef]
item-text(item_ref) -> Text
```

`ItemRef` carries metadata only: context id, item index, and item size in characters. `size-tokens` on an `ItemRef` uses that metadata. `size-tokens` on `List[ItemRef]` is the sum of the item sizes. `item-text` is the only primitive that reads an item's bytes.

Host limits:

- `max_context_read_chars` defaults to `leaf_threshold_tokens * 4`.
- A single read that exceeds `max_context_read_chars` fails verification in dry run or fails execution if somehow reached live.
- `context-items` returns item handles, not item bytes.

The Racket run message contains only context metadata and handles. Actual bytes cross the pipe only through bounded `context_read` replies.

---

## 5. Model Registry

```python
class ModelRegistryEntry(BaseModel):
    alias: str
    provider: str
    model_id: str
    backend: Literal["mock", "remote", "openai_compatible", "ollama"] = "remote"
    endpoint: str | None = None
    context_window_tokens: int
    reliable_input_tokens: int
    input_rate: float
    output_rate: float
    default_completion_tokens: int
    max_output_tokens: int
    max_concurrency: int = 1
    fallback_alias: str | None = None
```

Rules:

- `reliable_input_tokens <= context_window_tokens`.
- Leaf sizing uses `reliable_input_tokens`, not `context_window_tokens`.
- `fallback_alias` is used only for `retry_then_escalate`.
- v1 does not optimize across physical devices. `max_concurrency` is only a provider throttle for execution and a dry-run latency hint.

---

## 6. Program Language

The model authors a single S-expression program. The host parses it into an AST. There is no textual substitution and no host `eval`.

### 6.1 Grammar

```text
program       ::= expr

expr          ::= literal
                | var
                | slot-ref
                | param-ref
                | if
                | let
                | lambda
                | application
                | fix

literal       ::= string | number | boolean | null | quoted-data
quoted-data   ::= (quote datum)

slot-ref      ::= (slot symbol)
param-ref     ::= (param symbol)

if            ::= (if expr expr expr)
let           ::= (let ((identifier expr) ...) expr)
lambda        ::= (lambda (identifier ...) expr)
fix           ::= (fix identifier (lambda (identifier) expr))

application   ::= (operator expr ...)
operator      ::= primitive-name | identifier
```

`quoted-data` may contain only JSON-like data: strings, numbers, booleans, null, lists, and objects represented as association lists. It may not contain a form that is later evaluated as code.

### 6.2 Names

An identifier is valid only if it is:

- bound by `let`, `lambda`, or `fix`;
- a primitive name in the allow-list;
- used inside `(slot name)` or `(param name)`.

There are no macros, dynamic operator lookup, arbitrary module imports, `eval`, `read`, file I/O, network I/O, subprocesses, or host callbacks other than declared effects.

### 6.3 Host Parameters

The model must refer to host-owned resource values with `(param name)`.

Required parameters:

- `leaf_threshold_tokens`
- `split_factor`
- `max_depth`
- `leaf_model`
- `fallback_model`

Optional parameters:

- `temperature`
- `max_output_tokens`
- `error_policy`

The host injects parameters into an immutable environment before dry run and live execution. The program cannot assign to them. The artifact hash includes the exact parameter map.

### 6.4 Primitive Allow-List

Core primitives:

| Primitive | Signature | Effect |
|---|---|---|
| `size-tokens` | `Text or ItemRef or List -> Integer` | none |
| `<=` | `Number x Number -> Boolean` | none |
| `+`, `-`, `*`, `/` | numeric operations | none |
| `list` | `Any... -> List[Any]` | none |
| `first`, `rest`, `empty?` | list operations | none |
| `split-text` | `Text x Integer -> List[Text]` | none |
| `split-items` | `List[ItemRef] x Integer -> List[List[ItemRef]]` | none |
| `map` | `(A -> B) x List[A] -> List[B]` | inherited |
| `filter` | `(A -> Boolean) x List[A] -> List[A]` | inherited |
| `reduce` | `(B x A -> B) x B x List[A] -> B` | inherited |
| `concat` | `List[Text] -> Text` | none |
| `format-prompt` | `Text x Any -> Text` | none |
| `peek` | `ContextRef x Integer x Integer -> Text` | `CONTEXT_READ` |
| `slice` | `ContextRef x Integer x Integer -> Text` | `CONTEXT_READ` |
| `context-items` | `ContextRef -> List[ItemRef]` | `CONTEXT_READ` |
| `item-text` | `ItemRef -> Text` | `CONTEXT_READ` |
| `leaf-call` | `ModelAlias x InstructionText x InputText x Schema -> Value` | `LLM` |
| `validate` | `Schema x Value -> Boolean` | none |

Deferred primitives:

- `py-exec`
- `gate`
- `checkpoint`
- `race`
- `memoized`
- embeddings
- imported sub-programs

Deferred primitives must not exist in the v1 allow-list.

---

## 7. Typing and Schemas

v1 uses structural types plus a small JSON Schema subset.

Runtime types:

```text
Any
Null
Boolean
Number
Integer
Text
Schema
ContextRef
ItemRef
ModelAlias
List[T]
Function[A, B]
```

Allowed JSON Schema keywords:

- `type`
- `properties`
- `required`
- `items`
- `enum`
- `additionalProperties`

Unsupported keywords fail schema validation.

Compatibility:

- `Any` accepts anything.
- Exact scalar types must match, except `integer` is compatible with `number` as producer-to-consumer.
- For objects, the producer must provide all consumer-required properties with compatible schemas.
- For arrays, item schemas must be compatible.
- `additionalProperties: false` is enforced during validation.

Every `leaf-call` output is validated against its schema before flowing downstream. If validation fails, the active error policy applies.

For static typing, `leaf-call` returns the runtime type implied by its output schema. For example, a schema `{"type":"string"}` gives `Text`, and `{"type":"array","items":{"type":"string"}}` gives `List[Text]`.

Error policies:

- `fail_fast`: fail execution immediately.
- `retry_then_fail`: retry once on the same model, then fail.
- `retry_then_escalate`: retry once on the same model, then retry once on `fallback_alias`, then fail.

Dry run cost bounds assume the worst case for the chosen policy.

---

## 8. Termination Rules

`fix` is the only recursion form.

A recursive program is valid only if the verifier can prove this syntactic shape:

```text
(fix self
  (lambda (x)
    (if (<= (size-tokens x) (param leaf_threshold_tokens))
        base-expr-with-no-self-call
        recursive-expr-where-every-self-call-receives-a-proper-part-of-x)))
```

Accepted recursive arguments:

- an element produced by `(split-text x (param split_factor))`;
- an element produced by `(split-items x (param split_factor))`;
- a filtered subset of those elements;
- a mapped value whose mapper is proven non-growing by primitive rule.

Rejected recursive arguments:

- `x` itself;
- a concatenation or merge of split parts;
- a value from a slot, parameter, leaf call, or context read unrelated to `x`;
- any expression whose size relation to `x` is unknown.

Split rules:

- `(param split_factor) >= 2`.
- If `size-tokens(x) > leaf_threshold_tokens`, every child produced by `split-text` or `split-items` must have estimated size strictly less than `x`.
- Maximum simulated depth must be `<= (param max_depth)` and `<= policy.max_recursion_depth`.

If the verifier cannot prove descent, the program is rejected. It is not repaired automatically.

---

## 9. Host Parameter Planning

The model owns control-flow shape. The host owns resource parameters.

Inputs:

- context metadata;
- program AST;
- model registry;
- policy;
- optional hints.

Host chooses:

```python
leaf_threshold_tokens = floor(0.7 * leaf_model.reliable_input_tokens)
split_factor = first candidate in [2, 3, 4, 5, 8, 10] that passes policy
max_depth = policy.max_recursion_depth
leaf_model = policy.default_leaf_model unless the submitted hints request an allowed alias
fallback_model = leaf_model.fallback_alias
temperature = 0
max_output_tokens = leaf_model.default_completion_tokens
error_policy = policy.default_error_policy
```

Policy may allow overrides, but the host must clamp them:

- `leaf_threshold_tokens <= 0.9 * reliable_input_tokens`;
- `max_output_tokens <= model.max_output_tokens`;
- `split_factor` must be one of the allowed candidates;
- model alias must exist and be allowed by policy.

The host never rewrites the algorithm. If no parameter set satisfies policy, `plan_strategy` returns structured errors.

---

## 10. Dry Run

Dry run interprets the exact AST with the exact slots and parameters in simulate mode.

Simulation rules:

- `leaf-call` records a simulated call and returns a synthetic value matching the requested schema.
- `CONTEXT_READ` returns metadata-sized synthetic text or item handles, not live bytes.
- `map` simulates every element.
- `filter` is keep-all unless its predicate is statically evaluable.
- `reduce` simulates every reduction step.
- `retry_then_escalate` is not simulated inline; dry run adds worst-case retry and fallback calls to the high estimate.

Dry run output:

```python
class DryRunStats(BaseModel):
    llm_calls_low: int
    llm_calls_high: int
    context_reads: int
    recursive_depth: int
    max_concurrency: int
    estimated_wall_clock_seconds: float

class SimulatedCall(BaseModel):
    call_id: str
    model: str
    prompt_tokens: int
    completion_tokens_low: int
    completion_tokens_high: int
    output_schema: dict[str, Any]
    depth: int

class CostEstimate(BaseModel):
    prompt_tokens: int
    completion_tokens_low: int
    completion_tokens_high: int
    cost_usd_low: float
    cost_usd_high: float
```

Guarantee:

- Live calls must be `<= llm_calls_high`.
- Live cost must be `<= cost_usd_high` if providers charge according to registry rates and return no more than requested max tokens.
- Exact dry-run/live call equality is required only in deterministic test fixtures with no dynamic filter selectivity and no retries.

---

## 11. Verification

Verification runs after dry run and before execution. It runs every check and reports all failures.

Checks:

| Name | Severity | Rule |
|---|---|---|
| `artifact_integrity` | fail | artifact hash matches AST, slots, parameters, schemas, model registry hash, policy hash, and runtime version |
| `context_exists` | fail | bound context IDs exist |
| `slots_resolve` | fail | every `(slot name)` has a value |
| `params_resolve` | fail | every `(param name)` has a host value |
| `schema_valid` | fail | input/output/leaf schemas use the supported subset |
| `types_compatible` | fail | primitive arguments and boundary schemas are compatible |
| `effects_allowed` | fail | inferred effects are allowed by policy |
| `language_closed` | fail | AST uses only v1 forms and allow-listed primitives |
| `termination_bound` | fail | every `fix` satisfies structural descent and max depth |
| `model_aliases_resolve` | fail | every model alias exists and is policy-allowed |
| `budget_bound` | fail | high calls, tokens, and cost are within policy |
| `dry_run_fresh` | fail | dry run is for this exact artifact and current runtime version |
| `latency_estimate` | warn | estimated wall clock exceeds policy target |

Policy defaults:

```python
max_llm_calls = 500
max_prompt_tokens = 2_000_000
max_completion_tokens = 500_000
max_cost_usd = 10.00
max_recursion_depth = 5
max_context_read_chars = 32_000
max_wall_clock_seconds_target = 1800
allowed_effects = {"LLM", "CONTEXT_READ"}
default_error_policy = "retry_then_escalate"
default_leaf_model = "local_fast"
allowed_models = {"local_fast", "cloud_quality"}
```

`latency_estimate` is a warning in v1 because provider throughput is often noisy. Calls, tokens, cost, effects, language closure, and termination are hard gates.

---

## 12. Execution

`execute_strategy` requires:

- a `plan_id`;
- a matching fresh `dry_run_id`;
- no failed verification checks.

Execution steps:

1. Recompute artifact hash.
2. Verify dry run freshness.
3. Create an `ExecutionRecord`.
4. Start the Racket runtime with exact AST, slots, parameters, context metadata, and limits.
5. Service effect requests from Python.
6. Validate every `leaf-call` output schema.
7. Apply error policy on validation or provider failure.
8. Record trace events.
9. Store final result or failure.

Cancellation is best-effort:

- Python stops accepting new effect requests.
- In-flight provider calls may finish or be abandoned depending on adapter support.
- Execution state becomes `cancelled`.

No v1 resume exists after failure or process death. A user may execute the same verified artifact again, which creates a new execution record.

---

## 13. JSON Lines Protocol

Startup:

```json
{"type":"ready","protocol":"1.0","runtime_version":"rlm-scheme-racket-v1"}
```

Run request:

```json
{
  "type": "run",
  "mode": "simulate",
  "artifact_id": "art_0123456789abcdef",
  "ast": ["..."],
  "slot_values": {},
  "parameters": {},
  "contexts": {
    "ctx_0123456789abcdef": {
      "data_shape": "text",
      "size_chars": 1280000,
      "item_count": null
    }
  },
  "limits": {
    "max_recursion_depth": 5,
    "max_context_read_chars": 32000
  }
}
```

Effect requests:

```json
{"type":"context_read","id":"call_0000000000000001","context_id":"ctx_0123456789abcdef","op":"slice","range":{"start":0,"length":4000}}
{"type":"llm_call","id":"call_0000000000000002","model":"local_fast","instruction":"...","input":"...","output_schema":{"type":"object"},"max_tokens":512,"temperature":0}
```

Effect replies:

```json
{"type":"context_read_result","id":"call_0000000000000001","text":"...","size_chars":4000}
{"type":"llm_call_result","id":"call_0000000000000002","value":{},"usage":{"prompt_tokens":1000,"completion_tokens":120}}
{"type":"effect_error","id":"call_0000000000000002","error_code":"provider_error","message":"..."}
```

Terminal messages:

```json
{"type":"done","value":{},"stats":{},"calls":[]}
{"type":"error","error_code":"runtime_error","message":"...","trace":[]}
```

The runtime must not print non-protocol text to stdout.

---

## 14. MCP Tools

Exactly seven v1 tools:

| Tool | Purpose |
|---|---|
| `load_context` | Store context data and return `context_id`. |
| `plan_strategy` | Accept authored program, bind slots, choose host parameters, and return `plan_id`. |
| `dry_run_strategy` | Simulate exact artifact and return `dry_run_id`, bounds, and cost. |
| `execute_strategy` | Verify and execute once. |
| `get_execution_trace` | Return trace for an execution. |
| `get_record` | Return any stored record by ID. |
| `reset` | Clear store scopes for tests/development. |

Response envelope:

```json
{"status":"ok","...":"..."}
{"status":"error","error_code":"...","message":"...","details":{}}
```

`plan_strategy`, `dry_run_strategy`, and `execute_strategy` all return structured errors suitable for repair by the caller. The host does not run its own autonomous repair loop in v1.

---

## 15. Trace Events

Every execution records:

- execution start/end;
- context read requests and byte counts;
- model call request metadata;
- model call usage;
- schema validation failures;
- retries and escalations;
- final result or failure.

Trace records must omit full prompt and full model output by default. They may include hashes and short previews. A debug policy may enable full trace capture for local testing only.

---

## 16. Build Batches

### Batch 0: Foundations

Files:

- `rlm_scheme/ids.py`
- `rlm_scheme/store.py`
- `rlm_scheme/models.py`
- `config/models.json`

Acceptance:

- ID grammar tests.
- Store namespace tests.
- Model registry validation tests.
- Stable hash/canonical JSON tests.

### Batch 1: Parser and AST

Files:

- `rlm_scheme/sexpr.py`
- `rlm_scheme/ast.py`
- `rlm_scheme/program_validation.py`

Acceptance:

- Parse valid v1 forms.
- Reject macros, dynamic operators, free variables, `eval`, unknown primitives, and malformed `quote`.
- Resolve slots and params without textual substitution.
- No regex-based S-expression parsing.

### Batch 2: Schemas and Context Store

Files:

- `rlm_scheme/schema.py`
- `rlm_scheme/context_store.py`

Acceptance:

- Supported JSON Schema subset validates.
- Unsupported keywords fail.
- Schema compatibility tests pass.
- Text, items, and JSON contexts load and report metadata.
- Bounded reads enforce `max_context_read_chars`.

### Batch 3: Parameter Planning and Artifacts

Files:

- `rlm_scheme/planner.py`
- `rlm_scheme/artifacts.py`

Acceptance:

- Host chooses `leaf_threshold_tokens`, `split_factor`, model aliases, and error policy.
- Host clamps overrides.
- Artifact hash changes when AST, slots, parameters, schemas, model registry, policy, or runtime version changes.
- Structured planning errors are returned.

### Batch 4: Racket Runtime

Files:

- `runtime/main.rkt`
- `runtime/interpreter.rkt`
- `runtime/combinators.rkt`
- `runtime/wire.rkt`
- `rlm_scheme/runtime.py`

Acceptance:

- Handshake works.
- AST runs without Racket `eval`.
- Core primitives work.
- Simulate mode returns `done`, stats, and calls.
- Runtime stdout is protocol-only.

### Batch 5: Dry Run and Verification

Files:

- `rlm_scheme/dry_run.py`
- `rlm_scheme/verification.py`
- `rlm_scheme/cost.py`

Acceptance:

- Dry run computes low/high calls and cost.
- Unknown filters are keep-all.
- Recursive descent proofs accept valid split recursion and reject non-decreasing recursion.
- All verification checks run and report all failures.
- No live provider call occurs during planning, dry run, or failed verification.

### Batch 6: Providers and Execution

Files:

- `rlm_scheme/providers.py`
- `rlm_scheme/executor.py`
- `rlm_scheme/trace.py`
- `rlm_scheme/cache.py`

Acceptance:

- Mock provider is deterministic.
- Remote/OpenAI-compatible/Ollama adapters share one provider interface.
- Leaf outputs are schema-validated.
- Retry and escalation policies work.
- Live calls never exceed dry-run high call bound in tests.
- Trace records effects and usage.

### Batch 7: MCP Server and Docs

Files:

- `rlm_scheme/mcp_server.py`
- `rlm_scheme/app.py`
- `README.md`
- `examples/*.rkt`

Acceptance:

- Exactly seven tools are exposed.
- Happy path: load context -> plan -> dry run -> execute -> trace.
- Example programs parse and run with mock provider.
- README documents v1 scope and deferred features.

---

## 17. Example Programs

### 17.1 Direct Call

```scheme
(leaf-call
  (param leaf_model)
  (slot instruction)
  (slice (slot context) 0 (* 4 (param leaf_threshold_tokens)))
  (slot output_schema))
```

### 17.2 Recursive Chunk Summarize

```scheme
((fix solve
   (lambda (x)
     (if (<= (size-tokens x) (param leaf_threshold_tokens))
         (leaf-call
           (param leaf_model)
           (slot leaf_instruction)
           (concat (map item-text x))
           (slot text_schema))
         (leaf-call
           (param leaf_model)
           (slot compose_instruction)
           (concat
             (map solve
               (split-items x (param split_factor))))
           (slot text_schema)))))
 (context-items (slot context)))
```

This is the preferred v1 shape for large text: load the document as an `items` context where each item is a chunk. The program recurses over item handles, and bytes are read only at bounded leaves through `item-text`. In this example, `text_schema` is `{"type":"string"}`, so both recursive branches return `Text`.

### 17.3 Item Map Then Symbolic Concat

```scheme
(concat
  (map
    (lambda (item)
      (leaf-call
        (param leaf_model)
        (slot leaf_instruction)
        (item-text item)
        (slot text_schema)))
    (context-items (slot context))))
```

---

## 18. Deferred Features

These are intentionally out of v1:

- arbitrary Python execution;
- durable checkpoints and resume;
- human gates;
- race/await async primitives;
- embeddings and vector indexes;
- multi-device load/swap optimization;
- neural reduce cost optimizers;
- imported audited sub-program library;
- automatic program repair loop;
- full prompt/output trace capture by default;
- multimodal inputs.

Add deferred features only after v1 passes the definition of done and the new feature has its own verification contract.

---

## 19. Definition of Done

v1 is done when:

1. All batch acceptance tests pass.
2. No live provider call can happen before passing verification.
3. The parser accepts only the v1 grammar.
4. S-expression parsing and AST analysis do not use regex.
5. Host parameters enter through `(param name)` and are included in artifact identity.
6. Artifact identity includes AST, slots, schemas, host parameters, model registry hash, policy hash, and runtime version.
7. Context bytes cross to Racket only through bounded `CONTEXT_READ`.
8. Recursive programs must prove structural descent or fail verification.
9. Dry-run high calls/cost are enforced as hard bounds.
10. Live execution is tested to stay within dry-run high bounds.
11. Leaf outputs are schema-validated before downstream use.
12. Retry and escalation are counted and traced.
13. Exactly seven MCP tools are exposed.
14. Example programs run against the mock provider.
15. Deferred features are absent from the v1 allow-list.
