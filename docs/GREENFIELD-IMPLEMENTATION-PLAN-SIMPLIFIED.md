# RLM-Scheme Greenfield Implementation Plan

**Status:** normative implementation plan.  
**Audience:** human maintainers and coding agents.  
**Goal:** implement a small, auditable MCP server for bounded long-context LLM workflows.

## Orientation

RLM-Scheme is an MCP server for auditable long-context LLM computation. It keeps large inputs outside the model window, lets a caller submit a small typed orchestration program, dry-runs that program to compute resource bounds, verifies it against policy, and only then executes it.

The key move is to separate **strategy shape** from **unsafe execution**. A model may author the control-flow shape, but it can only use a tiny total S-expression language. The host owns the safety-critical values: leaf size, split factor, model aliases, context-read limits, schemas, policy, and budgets. The program is parsed, typed, parameterized, simulated, and verified before any live model call is allowed.

The system is intentionally not a general agent runtime. It does not run arbitrary generated code, substitute text into templates, import unverified sub-programs, or resume durable human gates. Its job is narrower: bounded decomposition, mapping, filtering, reduction, validation, and synthesis over data too large or too risky to send to one prompt.

The execution lifecycle is:

```text
load_context -> plan_strategy -> dry_run_strategy -> execute_strategy -> get_execution_trace
```

`load_context` stores data in the host. `plan_strategy` accepts an authored program and binds host parameters. `dry_run_strategy` simulates the exact artifact and produces conservative bounds. `execute_strategy` verifies those bounds and executes once only if every hard check passes.

---

## 1. Core Invariant

No live model call may occur until all of these are true:

1. The context has been loaded into the host store.
2. The authored program parses into the implementation AST.
3. Every referenced slot and host parameter resolves.
4. The program uses only allowed forms and primitives.
5. Recursion, if present, is structurally decreasing.
6. The dry run has produced an upper bound for calls, tokens, cost, and depth.
7. Verification has passed against the exact program, exact slots, exact host parameters, exact model registry, exact policy, and exact runtime version.

Execution is deterministic except for `leaf-call` results. Re-running the same verified artifact produces the same call structure. Contexts are immutable, so the only way to change input data is to load it again under a new `context_id`, which yields a new artifact; a runtime-version change likewise invalidates verification.

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

`cache` holds an optional provider-response cache keyed by `(model, instruction, input, schema, params)`; it dedupes identical live calls and is never consulted in dry run.

Records:

```python
class ContextRecord(BaseModel):
    context_id: str
    storage: Literal["memory", "file"]
    data_ref: str                 # memory store key, source path, or chunk store path
    data_shape: Literal["text", "items", "json"]
    size_chars: int
    item_count: int | None = None
    loader: dict[str, Any]
    warnings: list[str] = []
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
    stats: DryRunStats              # §10
    calls: list[SimulatedCall]      # §10
    cost: CostEstimate              # §10
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

`load_context` stores data outside the prompt and returns a structured load result. **Contexts are immutable:** loading new data yields a new `context_id`; stored bytes never change under an existing id, which is why the artifact identifies context by id rather than by content hash.

Contexts are fully addressable by the host, but they are not required to be memory-backed. Small inline contexts may live in memory. Large files and directories should be file-backed: the host stores item metadata and offsets, then reads bounded slices lazily when `slice` or `item-text` is evaluated. Racket receives only handles, metadata, and bounded read replies.

`load_context` response:

```python
class LoadContextResult(BaseModel):
    context_id: str
    data_shape: Literal["text", "items", "json"]
    storage: Literal["memory", "file"]
    size_chars: int
    estimated_tokens: int
    item_count: int | None = None
    skipped: list[dict[str, Any]] = []
    warnings: list[str] = []
```

Context access uses explicit units:

- `size_chars`: Unicode code points in the stored text representation.
- `estimated_tokens`: `ceil(size_chars / 4)`.
- `range.start` and `range.length`: character offsets for text contexts.
- item indexes: zero-based indexes for `items` contexts.

Supported context shapes:

- `text`: one contiguous text blob.
- `items`: ordered independent items, each serializable to text.
- `json`: one JSON value; the implementation may treat it as text unless a primitive explicitly supports JSON access.

`load_context` accepts inline data or a host-readable loader descriptor. Loader kinds:

- `inline`: caller supplies the data directly.
- `file`: host loads one file, optionally chunked into an `items` context.
- `directory`: host loads matching files under a directory, optionally chunked into an `items` context.

Inline loader:

```json
{
  "kind": "inline",
  "data": "text, JSON, or an array of item strings"
}
```

File loader:

```json
{
  "kind": "file",
  "path": "/absolute/path/server.log",
  "encoding": "utf-8",
  "encoding_errors": "replace",
  "chunk_chars": 16000,
  "chunk_overlap_chars": 0
}
```

Directory loader:

```json
{
  "kind": "directory",
  "root": "/absolute/path",
  "include": ["**/*.py", "**/*.rkt", "**/*.md"],
  "exclude": [".git/**", ".venv/**", "__pycache__/**"],
  "encoding": "utf-8",
  "encoding_errors": "replace",
  "chunk_chars": 16000
}
```

Loader rules:

- `path` / `root` must resolve under one of `policy.allowed_context_roots`; otherwise `load_context` fails;
- `encoding` defaults to `utf-8`;
- `encoding_errors` is either `strict` or `replace`, default `replace`;
- `chunk_chars` and `chunk_overlap_chars` are measured in decoded Unicode code points;
- `chunk_overlap_chars` must be smaller than `chunk_chars`;
- chunking preserves order and may split lines or paragraphs unless a future loader option says otherwise;
- `size_chars` is the total decoded character count, excluding overlap duplication;
- for file-backed contexts, chunk metadata records source path, chunk index, char start, char end, and size;
- binary files are rejected by the `file` loader and skipped by the `directory` loader unless an explicit text encoding succeeds.

Directory loading adds these deterministic rules:

- the host expands `include`/`exclude` globs under `root`;
- paths are sorted lexicographically by relative path;
- files larger than `chunk_chars` are split into ordered chunk items;
- every item stores a stable label: `relative/path` for a whole file, or `relative/path#chunk-N` for a chunk;
- stored context bytes are immutable after `context_id` creation.

Context reads:

```text
slice(context, start, length) -> Text
context-items(context) -> List[ItemRef]
item-text(item_ref) -> Text
item-label(item_ref) -> Text
```

`ItemRef` carries metadata only: context id, item index, label, source path if any, chunk index if any, and item size in characters. `size-tokens` on an `ItemRef` uses that metadata. `size-tokens` on `List[ItemRef]` is the sum of the item sizes. `item-label` reads only metadata. `item-text` is the only primitive that reads an item's bytes.

Host limits:

- `max_context_read_chars` is a host parameter equal to `leaf_threshold_tokens * 4`. It is included in the artifact and carried in the run-message `limits`. There is no separate policy constant for it.
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
- This plan does not optimize across physical devices. `max_concurrency` is only a provider throttle for execution.

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
operator      ::= primitive-name | identifier | lambda | fix
```

The `operator` of an application must evaluate to a `Function`; the verifier checks its arity and argument types. This permits applying a `lambda` or `fix` directly (as in §17.2).

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
- `max_context_read_chars`
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
| `slice` | `ContextRef x Integer x Integer -> Text` | `CONTEXT_READ` |
| `context-items` | `ContextRef -> List[ItemRef]` | `CONTEXT_READ` |
| `item-text` | `ItemRef -> Text` | `CONTEXT_READ` |
| `item-label` | `ItemRef -> Text` | none |
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

Out-of-scope primitives must not exist in the implementation allow-list.

---

## 7. Typing and Schemas

The implementation uses structural types plus a small JSON Schema subset.

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

**Slot and param types.** Param types are fixed: `leaf_threshold_tokens`, `split_factor`, `max_depth`, `max_output_tokens` are `Integer`; `temperature` is `Number`; `leaf_model`, `fallback_model` are `ModelAlias`; `error_policy` is an enum `Text`. Slot types are inferred from their use sites in the AST — e.g. the first argument of `context-items`/`slice` must be a `ContextRef`, and the schema argument of `leaf-call` must be a `Schema`. The verifier requires all uses of a slot to agree on one type and checks the bound `slot_values` value is consistent with it.

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
- a filtered subset of those elements.

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

Each `split_factor` candidate `k` is scored with the closed form `depth = ceil(log_k(ceil(estimated_tokens / leaf_threshold_tokens)))` and `leaf_calls = k^depth`. The host takes the first candidate whose `depth <= max_recursion_depth` and whose `leaf_calls` (times the error-policy worst-case factor) and token total are within policy. The dry run later confirms exact counts; if the chosen parameters fail simulation, `dry_run_strategy` returns structured errors. If they fail verification, `execute_strategy` returns structured errors without executing.

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
- `filter` is keep-all unless its predicate is statically evaluable — i.e. a closed expression over the simulated value's metadata using `size-tokens`, `<=`, arithmetic, `empty?`, `first`, `rest`, and literals, making no `leaf-call`, reading no context bytes, and not depending on synthetic leaf output.
- `reduce` simulates every reduction step.
- retries are not simulated inline; dry run folds their worst case into the high estimate: `retry_then_fail` adds one retry call per leaf, and `retry_then_escalate` adds one retry plus one fallback call per leaf.

Dry run output:

```python
class DryRunStats(BaseModel):
    llm_calls_low: int
    llm_calls_high: int
    context_reads: int
    recursive_depth: int

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
| `language_closed` | fail | AST uses only implementation forms and allow-listed primitives |
| `termination_bound` | fail | every `fix` satisfies structural descent and max depth |
| `model_aliases_resolve` | fail | every model alias exists and is policy-allowed |
| `budget_bound` | fail | high calls, tokens, and cost are within policy |
| `dry_run_fresh` | fail | dry run is for this exact artifact and current runtime version |

Policy defaults:

```python
max_llm_calls = 500
max_prompt_tokens = 2_000_000
max_completion_tokens = 500_000
max_cost_usd = 10.00
max_recursion_depth = 5
max_context_bytes = 10_000_000_000
max_context_items = 1_000_000
max_item_chars = 64_000
allowed_effects = {"LLM", "CONTEXT_READ"}
default_error_policy = "retry_then_escalate"
default_leaf_model = "local_fast"
allowed_models = {"local_fast", "cloud_quality"}
allowed_context_roots = ["/home/rwt/Code"]
```

All checks are hard gates: calls, tokens, cost, effects, language closure, and termination must pass. There is no separate verify tool — `execute_strategy` runs all checks after confirming a fresh dry run and refuses to execute unless `decision == pass`.

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

There is no resume after failure or process death. A user may execute the same verified artifact again, which creates a new execution record.

---

## 13. JSON Lines Protocol

Startup:

```json
{"type":"ready","protocol":"1.0","runtime_version":"rlm-scheme-racket-1"}
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
    "max_context_read_chars": 22400
  }
}
```

Effect requests:

```json
{"type":"context_read","id":"call_0000000000000001","context_id":"ctx_0123456789abcdef","op":"slice","range":{"start":0,"length":4000}}
{"type":"context_read","id":"call_0000000000000003","context_id":"ctx_0123456789abcdef","op":"context-items"}
{"type":"context_read","id":"call_0000000000000004","context_id":"ctx_0123456789abcdef","op":"item-text","item_index":7}
{"type":"llm_call","id":"call_0000000000000002","model":"local_fast","instruction":"...","input":"...","output_schema":{"type":"object"},"max_tokens":512,"temperature":0}
```

`op` is one of `slice`, `context-items`, `item-text`; `slice` carries a `range`, `item-text` carries an `item_index`, and `context-items` carries neither. `item-label` is metadata-only and is evaluated inside the runtime from the `ItemRef`; it does not require a context-read effect.

Effect replies:

```json
{"type":"context_read_result","id":"call_0000000000000001","text":"...","size_chars":4000}
{"type":"context_read_result","id":"call_0000000000000003","items":[{"index":0,"size_chars":1234}]}
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

Exactly seven tools:

| Tool | Signature |
|---|---|
| `load_context` | `(data_or_loader, data_shape, schema?, metadata?) -> LoadContextResult` |
| `plan_strategy` | `(context_id, program_source, task, slot_values, hints?) -> plan_id` |
| `dry_run_strategy` | `(plan_id) -> dry_run_id, bounds, cost` |
| `execute_strategy` | `(plan_id, dry_run_id) -> execution_id` |
| `get_execution_trace` | `(execution_id) -> trace` |
| `get_record` | `(id) -> record` |
| `reset` | `(scope) -> ok` |

`plan_strategy` parses the program and materializes the `ProgramRecord`, `PlanRecord`, and the content-addressed `ArtifactRecord` (from AST + slots + host parameters + schemas + registry/policy/runtime hashes). `dry_run_strategy` simulates that artifact. `execute_strategy` runs verification internally — after confirming a fresh dry run — and executes only if `decision == pass`.

Response envelope:

```json
{"status":"ok","...":"..."}
{"status":"error","error_code":"...","message":"...","details":{}}
```

`plan_strategy`, `dry_run_strategy`, and `execute_strategy` all return structured errors suitable for repair by the caller. The host does not run its own autonomous repair loop.

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

- Parse valid implementation forms.
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
- Inline text, items, and JSON contexts load and report metadata.
- File loader streams large files, chunks by decoded character offsets, supports overlap, and does not require reading the whole file into memory.
- Directory loader expands files deterministically, applies include/exclude globs, chunks large files, and assigns stable item labels.
- Loader policy rejects roots outside `allowed_context_roots`, excessive total bytes, excessive item counts, and chunks larger than `max_item_chars`.
- Binary file behavior is deterministic: file loader rejects binary input; directory loader records skipped binary files.
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
- `item-label` returns metadata without a context-read effect.
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
- Repository plan-review worked example runs with a temporary directory and mock provider.
- README documents implementation scope and out-of-scope features.

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

This is the preferred shape for large text: load the document as an `items` context where each item is a chunk. The program recurses over item handles, and bytes are read only at bounded leaves through `item-text`. In this example, `text_schema` is `{"type":"string"}`, so both recursive branches return `Text`.

### 17.3 Item Map Then Symbolic Concat

```scheme
(concat
  (map
    (lambda (item)
      (leaf-call
        (param leaf_model)
        (slot leaf_instruction)
        (concat
          (list
            "Item: " (item-label item) "\n\n"
            (item-text item)))
        (slot text_schema)))
    (context-items (slot context))))
```

---

## 18. Worked Example: Repository Plan Review

Problem:

> Given a repository and an implementation plan, identify files that conflict with the plan, missing implementation work, and tests that should be added.

This is realistic because the repository may contain hundreds of files. The model should not receive the whole repository in one prompt, and the runtime should not execute arbitrary generated code while inspecting it.

### 18.1 Load the Repository Directory

The caller loads the repository as an `items` context:

```json
{
  "data_or_loader": {
    "kind": "directory",
    "root": "/home/rwt/Code/rlm-scheme",
    "include": ["**/*.py", "**/*.rkt", "**/*.md", "config/**/*.json"],
    "exclude": [".git/**", ".venv/**", "__pycache__/**", ".pytest_cache/**"],
    "chunk_chars": 16000
  },
  "data_shape": "items",
  "metadata": {"purpose": "repo-review"}
}
```

The host expands the directory into ordered `ItemRef`s. Each item has a label such as `rlm_scheme/runtime.py` or `docs/GREENFIELD-IMPLEMENTATION-PLAN-SIMPLIFIED.md#chunk-2`.

### 18.2 Load the Plan Text

The caller loads the plan as `text` if it fits the reliable-input budget, or as `items` if it is large. For this example, assume a bounded plan summary is supplied as a slot rather than read from context:

```json
{
  "plan_summary": "Implement the MCP tools load_context, plan_strategy, dry_run_strategy, execute_strategy, get_execution_trace, get_record, and reset. Use a closed S-expression AST, host parameters, immutable contexts, directory item loading, dry-run high bounds, and verification inside execute_strategy."
}
```

### 18.3 Leaf Prompt

The leaf instruction is ordinary slot data:

```text
You are reviewing one repository item against the implementation plan.
Return a concise text block with these headings:
LABEL
RELEVANT
FINDINGS
REQUIRED CHANGES
TESTS

Use RELEVANT: false when the item does not affect the plan.
Do not invent files or APIs. Base the answer only on the plan summary and item text.
```

The leaf input is constructed by the program from bounded data:

```text
PLAN SUMMARY:
<plan_summary slot>

ITEM:
<item-label>

ITEM TEXT:
<item-text>
```

### 18.4 Program

```scheme
(concat
  (map
    (lambda (item)
      (leaf-call
        (param leaf_model)
        (slot review_instruction)
        (concat
          (list
            "PLAN SUMMARY:\n"
            (slot plan_summary)
            "\n\nITEM:\n"
            (item-label item)
            "\n\nITEM TEXT:\n"
            (item-text item)))
        (slot text_schema)))
    (context-items (slot repo_context))))
```

`text_schema` is:

```json
{"type": "string"}
```

The `plan_strategy` call binds these slots:

```json
{
  "repo_context": "ctx_repo",
  "plan_summary": "Implement the MCP tools load_context, plan_strategy, dry_run_strategy, execute_strategy, get_execution_trace, get_record, and reset. Use a closed S-expression AST, host parameters, immutable contexts, directory item loading, dry-run high bounds, and verification inside execute_strategy.",
  "review_instruction": "You are reviewing one repository item against the implementation plan. Return a concise text block with headings LABEL, RELEVANT, FINDINGS, REQUIRED CHANGES, and TESTS. Use RELEVANT: false when the item does not affect the plan. Do not invent files or APIs. Base the answer only on the plan summary and item text.",
  "text_schema": {"type": "string"}
}
```

### 18.5 What Each Tool Does

1. `load_context` creates `ctx_repo` from the directory loader.
2. `plan_strategy` parses the program, checks the slot bindings above, infers slot types, chooses host parameters, and creates the artifact.
3. `dry_run_strategy` simulates one leaf call per repository item and adds retry/escalation worst-case calls to the high bound.
4. `execute_strategy` verifies artifact integrity, language closure, context existence, slot and parameter resolution, schema validity, effects, model aliases, termination, budget, and dry-run freshness.
5. During execution, each `item-text` request reads only one bounded item. Each `leaf-call` receives the exact instruction above and the constructed input containing the plan summary, item label, and item text.
6. Each model output is validated against `text_schema`. Invalid output follows the configured retry/escalation policy.
7. `get_execution_trace` returns item reads, model call metadata, usage, validation failures, retries, escalations, and final status.

This example intentionally uses symbolic `concat` as the final composition, so every leaf returns `Text`. A structured-object variant would require adding an allow-listed symbolic combiner such as `json-lines` or `collect`, plus its typing and verification rules.

---

## 19. Worked Example: Large Text File Review

Problem:

> Given a 3 GB server log, find recurring error patterns and produce per-chunk observations without loading the whole file into an LLM context or process memory.

### 19.1 Load the File

```json
{
  "data_or_loader": {
    "kind": "file",
    "path": "/storage/logs/server.log",
    "encoding": "utf-8",
    "encoding_errors": "replace",
    "chunk_chars": 16000,
    "chunk_overlap_chars": 1000
  },
  "data_shape": "items",
  "metadata": {"purpose": "log-review"}
}
```

The host validates that `/storage/logs/server.log` is under `allowed_context_roots`, streams the file, creates ordered chunk items such as `server.log#chunk-000001`, and records decoded char offsets. The context may be file-backed; only metadata and offsets need to stay resident.

### 19.2 Program Shape

```scheme
(concat
  (map
    (lambda (chunk)
      (leaf-call
        (param leaf_model)
        (slot analyze_instruction)
        (concat
          (list
            "CHUNK:\n"
            (item-label chunk)
            "\n\nLOG TEXT:\n"
            (item-text chunk)))
        (slot text_schema)))
    (context-items (slot log_context))))
```

The leaf model sees one bounded chunk at a time. Dry run counts one low-bound call per chunk and high-bound retry/escalation calls according to policy.

---

## 20. Out of Scope

These features are intentionally out of scope for this plan:

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

Add an out-of-scope feature only after the implementation passes the definition of done and the new feature has its own verification contract.

---

## 21. Definition of Done

The implementation is done when:

1. All batch acceptance tests pass.
2. No live provider call can happen before passing verification.
3. The parser accepts only the implementation grammar.
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
15. Out-of-scope features are absent from the implementation allow-list.
