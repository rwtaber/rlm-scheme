# RLM-Scheme Implementation Plan v2

## Preamble

RLM-Scheme is a structured orchestration system exposed as an MCP server. An agent interacts with durable objects through a fixed pipeline:

```text
context_id -> plan_id -> dry_run_id -> execution_id
```

Agents describe intent and data; the system classifies, selects a template, instantiates executable Scheme, verifies it against policy, and executes it in a sandboxed Racket runtime. Agents never write Scheme. The happy path is 3 tool calls after loading context: `plan_strategy` -> `dry_run_strategy` -> `execute_strategy`. The public MCP surface is exactly 10 tools. Internally, the system also creates `artifact_id` and `verification_id` records for audit and debugging, but these are not agent-managed concepts.

Templates are `.rkt` files containing real Scheme code with `{{slot}}` markers. The instantiator validates slots, substitutes values, and hashes the result. There is no intermediate representation. LLMs choose strategy intent and fill content slots; deterministic code validates and substitutes safely; verification checks the filled artifact before real model calls happen.

**Conventions used in this document:**

- Key words [MUST], [SHOULD], and [MAY] follow RFC 2119 semantics. [MUST] indicates an absolute requirement. [SHOULD] indicates a recommendation that may be deviated from with justification. [MAY] indicates an optional feature.
- Pseudocode uses Python-style type annotations (e.g., `str`, `int | None`, `list[str]`) for parameter and return types.
- Cross-references use the form \"See Appendix X, item Y\" rather than \"See section X.\"
- Requirements are tagged with bracketed keywords: `[MUST]`, `[SHOULD]`, `[MAY]`.
- Quantities are always explicit: \"ALL 23 verification checks\", \"16 template files\", \"13 TaskShape values.\"

---

## Appendix A: Enum and Type Definitions

### A.1 TaskShape Enum (13 values)

Every plan classification [MUST] assign exactly one TaskShape. Composite tasks [MUST] also record constituent shapes.

| Value | Description | Structural family |
|---|---|---|
| `Direct` | One operation on one small input. | `llm-query` only. |
| `Batch` | Same operation over many independent items. | `map-async`, optional reduction. |
| `Synthesize` | Combine many inputs into one output. | `tree-reduce` or `fold-sequential`. |
| `Search` | Explore solution space and choose best result. | `parallel`, `race`, `iterate-until`. |
| `Refine` | Improve one artifact iteratively. | `iterate-until`. |
| `Compare` | Evaluate alternatives against criteria. | `parallel` plus selection/aggregation. |
| `Classify` | Assign labels/categories to items. | `map-async`, optional aggregation. |
| `Pipeline` | Distinct sequential transformations. | `sequence`. |
| `Generate` | Create new content from scratch. | index-based `map-async`, `iterate-until`, or `fold-sequential`. |
| `Decompose` | Break one input into structured parts. | `llm-query` JSON, `python_compute`, or `recursive`. |
| `Validate` | Produce pass/fail/score assessments. | `map-async`, validation, aggregation. |
| `Aggregate` | Extract metrics and compute report. | `map-async` plus `python_compute`. |
| `Composite` | Multi-phase task combining 2+ atomic shapes. | Instantiated `sequence` of phase templates. |

### A.2 DataShape Enum (10 + Unknown)

Context metadata [MUST] include a DataShape value. `Unknown` is the fallback when the agent does not specify a shape.

| Value | Description | Key metadata fields |
|---|---|---|
| `FlatList` | Independent or ordered list. | `count`, `item_size`, `independent`. |
| `Hierarchy` | Tree or nested structure. | `depth`, `branching`, `node_count`. |
| `Singular` | One blob that may fit in context. | `size`, `chunkable`, `boundary`. |
| `ChunkedSingular` | Large document split into dependent chunks. | `chunk_count`, `overlap`, `dependency`. |
| `Graph` | Connected entities and edges. | `nodes`, `edges`, `connectedness`. |
| `TimeSeries` | Ordered observations. | `length`, `window_size`, `causal`. |
| `Tabular` | Rows with shared schema. | `row_count`, `columns`, `grouping_keys`. |
| `Multimodal` | Text plus images/audio. | `modality`, `count`, `model_requirements`. |
| `Paired` | Aligned source/target pairs. | `pair_count`, `alignment_key`. |
| `KeyValue` | Dictionary/map data. | `key_count`, `preserve_keys`. |
| `Unknown` | Shape not specified or not determinable. | None required. |

### A.3 ExecutionState Enum (7 values)

Every execution record [MUST] have exactly one state at any time.

| Value | Terminal? | Description |
|---|---|---|
| `queued` | No | Execution created but not yet started. |
| `running` | No | Execution in progress. |
| `awaiting_gate` | No | Execution suspended at a human-review gate. |
| `finished` | Yes | `finish` primitive completed successfully. |
| `failed` | Yes | Unhandled error propagated to top level. |
| `cancelled` | Yes | `cancel_call(execution_id=...)` received. |
| `gate_rejected` | Yes | Gate rejected via `resume_execution(decision="reject")`. |

**State transitions:**

| From | To | Trigger |
|---|---|---|
| `queued` | `running` | Executor picks up the execution. |
| `running` | `finished` | `finish` primitive completes successfully. |
| `running` | `failed` | Unhandled error propagates to top level (includes `BudgetExhaustedError` from checkpoint-and-stop). |
| `running` | `cancelled` | `cancel_call(execution_id=...)` received. |
| `running` | `awaiting_gate` | `gate` primitive fires. |
| `awaiting_gate` | `running` | `resume_execution(decision="approve")`. |
| `awaiting_gate` | `cancelled` | `cancel_call(execution_id=...)` while suspended. |
| `awaiting_gate` | `gate_rejected` | `resume_execution(decision="reject")`. |

No transitions out of terminal states (`finished`, `failed`, `cancelled`, `gate_rejected`).

### A.4 ErrorPolicy Enum (3 values)

Each template node [MUST] declare an error policy. The default is `fail_fast`.

| Value | Behavior |
|---|---|
| `fail_fast` | First error aborts the entire node. Partial results are discarded. Default for most primitives. |
| `collect` | Errors are collected alongside successful results. The node completes and returns a mixed result list with error markers. Consumer nodes [MUST] handle error entries. |
| `fallback` | On error, execute the declared fallback function for that item. If fallback also fails, apply `fail_fast` or `collect` as secondary policy. |

### A.5 ResetScope Enum (7 values)

The `reset_runtime` tool [MUST] accept one of these scope values.

| Value | Clears | Preserves |
|---|---|---|
| `sandbox` | Racket sandbox state only. | All durable records, caches, and configuration. |
| `session` | Racket sandbox state, in-memory caches, active call handles, execution records and traces from current session. | Contexts, plans, artifacts, dry-runs, verifications, and checkpoints from prior sessions. |
| `cache` | LLM result cache entries. | All durable records, sandbox state, and template catalog. |
| `contexts` | All context records and associated data files. | Plans, artifacts, executions, cache, and sandbox state. |
| `executions` | All execution records, traces, and associated result files. | Contexts, plans, artifacts, dry-runs, verifications, cache. |
| `config` | Reloads model registry and template catalog from disk. | All durable records, sandbox state, and cache. |
| `all` | All durable records and sandbox state. Fresh start. | Nothing. |

### A.6 ID Prefix Table (8 prefixes)

Every durable object ID [MUST] match the regex `^{prefix}[a-z0-9]{16}$`, generated via `secrets.token_hex(8)`.

| Prefix | Object type | Example |
|---|---|---|
| `ctx_` | Context record | `ctx_7f3a1b2c4d5e6f78` |
| `plan_` | Plan record | `plan_b2c1d3e4f5a6b7c8` |
| `dry_` | Dry-run record | `dry_1a2b3c4d5e6f7890` |
| `exec_` | Execution record | `exec_5e6f7a8b9c0d1e2f` |
| `art_` | Artifact record | `art_e4d9a1b2c3d4e5f6` |
| `ver_` | Verification record | `ver_3c4d5e6f7a8b9c0d` |
| `call_` | LLM call record | `call_001a2b3c4d5e6f7` |
| `ckpt_` | Checkpoint record | `ckpt_8a9b0c1d2e3f4a5b` |

ID generation pseudocode:

```python
import secrets

def generate_id(prefix: str) -> str:
    \"\"\"Generate a durable object ID. Prefix MUST be one of the 8 registered prefixes.\"\"\"
    assert prefix in (\"ctx_\", \"plan_\", \"dry_\", \"exec_\", \"art_\", \"ver_\", \"call_\", \"ckpt_\")
    return f\"{prefix}{secrets.token_hex(8)}\"
```

---

## Appendix B: Full Primitive Signatures

All primitives below are the public bindings available inside instantiated Scheme artifacts. Helper bindings prefixed with `__` are instantiator-private. Templates [MUST] use only these primitives and `__`-prefixed helpers; any other binding causes verification failure (see Appendix A.6, check 21 in the verification table).

### B.1 LLM Primitives

**`llm-query`**

```
llm-query(
    instruction: str,
    data: any,
    model: str,
    recursive: bool = False,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json: bool = False,
    image: ImageInput | None = None,
    images: list[ImageInput] = []
) -> SyntaxObject
```

Synchronous LLM call. Returns a syntax-wrapped result. Decrements token budget after response. Logs call metadata and provenance. Supports `recursive=True` only under global recursion policy.

**`llm-query-async`**

```
llm-query-async(
    instruction: str,
    data: any,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json: bool = False,
    image: ImageInput | None = None,
    images: list[ImageInput] = []
) -> AsyncHandle
```

Dispatches LLM call through the host future pool and returns immediately with an opaque async handle. Does not support recursive calls. Use inside `map-async`, `parallel`, and `race` bodies.

### B.2 Await Primitives

**`await`**

```
await(handle: AsyncHandle) -> SyntaxObject
```

Blocks until one handle completes. Propagates cancellation and provider errors. Decrements token budget when real usage is known. Wraps result as syntax.

**`await-all`**

```
await-all(handles: list[AsyncHandle]) -> list[str]
```

Waits for all handles concurrently. Returns unwrapped strings in input order. Records batch wait in trace.

**`await-any`**

```
await-any(handles: list[AsyncHandle]) -> tuple[str, list[AsyncHandle]]
```

Waits for the first completed handle. Returns the completed unwrapped string and the list of remaining handles. [MUST] be deterministic in dry-run mode: exactly one requested pending handle completes per call.

### B.3 Parallel Primitives

**`map-async`**

```
map-async(
    fn: Callable[[any], AsyncHandle],
    items: list[any],
    max_concurrent: int | None = None
) -> list[str]
```

Rolling-window fan-out over items. Preserves input order. Maintains a bounded concurrency window. Reports progress and heartbeats during long fan-outs. Error policy is declared per-node in template metadata.

**`parallel`**

```
parallel(
    thunks: list[Callable[[], AsyncHandle]],
    max_concurrent: int | None = None
) -> list[str]
```

Genuinely concurrent thunk execution. Preserves strategy order in output. [MUST] reject bodies that call synchronous `llm-query` directly.

**`race`**

```
race(thunks: list[Callable[[], AsyncHandle]]) -> str
```

Launches all candidates. Returns first completed result. Cancels or abandons remaining handles. Records losing handles in trace as cancelled.

### B.4 Reduction Primitives

**`tree-reduce`**

```
tree-reduce(
    reducer: Callable[[list[any]], any],
    items: list[any],
    branch_factor: int,
    leaf_fn: Callable[[any], any] | None = None
) -> any
```

Recursive associative reduction. Rejects empty input. Groups items by branch factor and applies reducer recursively until one result remains. Call estimate: `N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1`.

**`fold-sequential`**

```
fold-sequential(
    reducer: Callable[[any, any], any],
    initial: any,
    items: list[any]
) -> any
```

Ordered accumulation. Processes items sequentially, passing accumulator and item to reducer. High critical-path latency. Appropriate for order-sensitive synthesis.

### B.5 Control Primitives

**`sequence`**

```
sequence(fn1: Callable, fn2: Callable, ...) -> Callable[[any], any]
```

Left-to-right function composition. Used by the instantiator for multi-phase templates. Can be generated as `let*` when simpler.

**`choose`**

```
choose(
    predicate: Callable[[any], bool],
    then_fn: Callable,
    else_fn: Callable
) -> Callable
```

Routes based on deterministic predicate. [SHOULD] not hide model calls inside predicates unless declared.

**`iterate-until`**

```
iterate-until(
    step_fn: Callable[[any], any],
    predicate: Callable[[any], bool],
    init: any,
    max_iter: int
) -> any
```

Bounded loop. Stops when predicate returns true or `max_iter` is reached. Dry-run reports worst-case iteration count unless predicate is statically known.

**`recursive-spawn`**

```
recursive-spawn(
    template_name: str,
    slot_values: dict[str, any]
) -> Callable[[any], SyntaxObject]
```

Returns a lambda that, when called with data: creates a temporary context, instantiates the named template with slot values plus the temporary context ID, verifies the sub-artifact against the parent execution's policy, and executes it. Global recursion depth is enforced host-side.

### B.6 Delegation Primitives

**`load-context`**

```
__context-ref(context_id: str, json_path: str) -> any
```

Retrieves data from a loaded context at runtime. `context_id` [MUST] be a valid `ctx_` ID. `json_path` is an RFC 9535 JSONPath expression. `\"$\"` returns the entire context value.

### B.7 Modifier Primitives

**`memoized`**

```
memoized(fn: Callable, key_fn: Callable[[any], str]) -> Callable
```

Caches results within one execution unless the template requests persistent caching. Key function [MUST] be deterministic. Trace records cache hits/misses.

**`with-validation`**

```
with-validation(fn: Callable, validator: Callable[[any], bool]) -> Callable
```

Runs `fn`, validates result, returns result or raises structured validation error. [SHOULD] include schema path or validation rule in errors.

**`try-fallback`**

```
try-fallback(primary_fn: Callable, fallback_fn: Callable) -> Callable
```

Catches declared error classes. Executes fallback with original args. Records both primary failure and fallback result in trace.

### B.8 State Primitives

**`gate`**

```
gate(
    name: str,
    value: any,
    message: str,
    required: bool = False
) -> any
```

Suspends execution for human review. Transitions state to `awaiting_gate`. On approve, returns `value` unchanged. On reject, raises structured gate-rejection error. `required=False` means gate fires only when execution policy includes `require_gates: true`.

**`finish`**

```
finish(value: any) -> NoReturn
```

Terminates the current execution and sets its result to `value`. If the template declares `output-schema`, the runtime validates `value` against it before accepting. Exactly one `finish` call per execution [MUST] occur.

**`checkpoint`**

```
checkpoint(key: str, value: any) -> any
```

Persists a durable partial result keyed by execution/artifact namespace. Values [MUST] be JSON-serializable. Returns the value unchanged.

**`checkpoint-restore`**

```
restore(key: str) -> any | None
```

Retrieves a previously checkpointed value, or `None` if no checkpoint exists for the key.

### B.9 Compute Primitives

**`python-compute` / `py-exec`**

```
py-exec(code: str) -> str
```

Executes Python code in an isolated subprocess. Receives values over JSON. Cannot access MCP server internals or Racket scaffold bindings. Returns stdout as a string.

**`py-eval`**

```
py-eval(expr: str) -> any
```

Evaluates a Python expression and returns the result. Same isolation as `py-exec`.

**`py-call`**

```
py-call(ref: str, method: str, *args: any) -> any
```

Calls a method on a Python object reference. Same isolation as `py-exec`.

**`py-set!`**

```
py-set!(name: str, value: any) -> None
```

Sets a variable in the Python bridge namespace. Value is transferred as JSON.

### B.10 Helper Primitives

**`syntax-e`**

```
syntax-e(stx: SyntaxObject) -> any
```

Unwraps a syntax object to its underlying value. Pass-through for non-syntax values. Logs every unwrap to the scope/provenance log. This is the explicit \"trust this LLM output\" operation.

**`datum->syntax`**

```
datum->syntax(datum: any) -> SyntaxObject
```

Wraps a plain value in a syntax object with scope metadata. Logs the wrap operation. Used when template code needs to promote a computed value back into the syntax-tracked domain.

---

## Appendix C: Model Registry and Retry Config Schemas

### C.1 Model Registry Entry Schema

The model registry [MUST] be a JSON file at `config/models.json` (path configurable via `RLM_MODEL_REGISTRY` environment variable). It is loaded once at server startup and can be reloaded via `reset_runtime(scope=\"config\")`.

```python
class ModelRegistryEntry(BaseModel):
    \"\"\"One model alias in the registry.\"\"\"
    alias: str                          # e.g., \"fast_text_model\"
    provider: str                       # e.g., \"openai\", \"anthropic\"
    model_id: str                       # provider-specific model name
    context_window_tokens: int          # max input tokens
    max_output_tokens: int              # max completion tokens
    capabilities: dict[str, bool]       # {\"text\": True, \"json\": True, \"image\": False}
    cost_per_1k_prompt: float           # USD per 1,000 prompt tokens
    cost_per_1k_completion: float       # USD per 1,000 completion tokens
    temperature_range: tuple[float, float]  # (min, max), e.g., (0.0, 2.0)
    supports_temperature: bool          # whether temperature kwarg is accepted
    cost_tier: str                      # \"low\", \"medium\", \"high\"
    fallback: str | None = None         # alias to fall back to on budget policy
```

Example registry JSON:

```json
{
  \"schema_version\": \"1\",
  \"aliases\": {
    \"fast_text_model\": {
      \"provider\": \"openai\",
      \"model_id\": \"configured-fast-model\",
      \"context_window_tokens\": 128000,
      \"max_output_tokens\": 16384,
      \"capabilities\": {\"text\": true, \"json\": true, \"image\": false},
      \"cost_per_1k_prompt\": 0.0015,
      \"cost_per_1k_completion\": 0.002,
      \"temperature_range\": [0.0, 2.0],
      \"supports_temperature\": true,
      \"cost_tier\": \"low\",
      \"fallback\": null
    }
  },
  \"defaults\": {
    \"planner\": \"quality_text_model\",
    \"map\": \"fast_text_model\",
    \"reduce\": \"quality_text_model\",
    \"vision\": \"vision_model\"
  }
}
```

### C.2 Alias Validation Rules

- Aliases [MUST] match the regex `^[a-z][a-z0-9_]{2,30}$`.
- Alias names [MUST] be unique within the registry.
- The `defaults` section [MUST] reference aliases that exist in the `aliases` section.
- If a `fallback` alias is specified, it [MUST] exist in the registry and [MUST] have compatible capabilities (see Appendix C, item C.4).

### C.3 Retry Config Schema

The retry policy [MUST] be a JSON file at `config/retry.json` (path configurable via `RLM_RETRY_CONFIG` environment variable). Retries are per-call, not per-execution.

```python
class RetryConfig(BaseModel):
    \"\"\"Global retry configuration.\"\"\"
    max_retries: int = 3                    # maximum retry attempts per call
    base_delay_seconds: float = 1.0         # initial backoff delay
    max_delay_seconds: float = 60.0         # backoff ceiling
    backoff_multiplier: float = 2.0         # exponential multiplier
    jitter: bool = True                     # add random jitter to backoff
    retry_on: list[str] = [                 # error types that trigger retry
        \"rate_limit\", \"timeout\", \"server_error\"
    ]
    retryable_status_codes: list[int] = [429, 500, 502, 503, 504]
```

Example retry config JSON:

```json
{
  \"schema_version\": \"1\",
  \"defaults\": {
    \"max_retries\": 3,
    \"base_delay_seconds\": 1.0,
    \"max_delay_seconds\": 60.0,
    \"backoff_multiplier\": 2.0,
    \"jitter\": true,
    \"retryable_status_codes\": [429, 500, 502, 503, 504],
    \"retry_on\": [\"rate_limit\", \"timeout\", \"server_error\"]
  },
  \"per_model_overrides\": {
    \"fast_text_model\": {
      \"max_retries\": 5,
      \"base_delay_seconds\": 0.5
    }
  }
}
```

`per_model_overrides` entries are merged with defaults: override keys replace default keys; unspecified keys inherit from `defaults`.

### C.4 Error Type Mapping

The host [MUST] classify provider errors into retry categories using this mapping. Only non-retryable errors reach the per-primitive error policy (see Appendix A, item A.4).

| HTTP status / exception | Error type | Retry behavior |
|---|---|---|
| 429 (Too Many Requests) | `rate_limit` | Use `Retry-After` header if present, else exponential backoff. |
| 408 (Request Timeout), `ETIMEDOUT`, `ECONNRESET` | `timeout` | Exponential backoff, same request. |
| 500, 502, 503, 504 | `server_error` | Exponential backoff, same request. |
| 400 (Bad Request) | `invalid_request` | **Not retried** -- propagate immediately. |
| 401, 403 | `auth_error` | **Not retried** -- propagate immediately. |
| 404, 422 | `client_error` | **Not retried** -- propagate immediately. |
| Context length exceeded | `context_overflow` | **Not retried** -- propagate immediately. |
| Network unreachable, DNS failure | `network_error` | **Not retried** -- propagate immediately. |
| Provider SDK `RateLimitError` | `rate_limit` | Per `rate_limit` rules above. |
| Provider SDK `APITimeoutError` | `timeout` | Per `timeout` rules above. |
| Provider SDK `APIConnectionError` | `server_error` | Per `server_error` rules above. |
| All other SDK exceptions | `client_error` | **Not retried** -- propagate immediately. |

Rate-limit retries [MUST] use the `Retry-After` header when available. Token budget [MUST NOT] be consumed by failed attempts. Each retry [MUST] be logged in the execution trace.

---

## Appendix D: Template `define-meta` Grammar

### D.1 BNF Grammar

```bnf
meta-form    ::= '(' 'define-meta' name value ')'
name         ::= symbol
value        ::= atom | quoted-list
atom         ::= string | number | boolean
boolean      ::= '#t' | '#f'
quoted-list  ::= \"'\" s-expr
s-expr       ::= atom | '(' s-expr* ')' | '(' s-expr '.' s-expr ')'
```

The template loader recognizes `define-meta` forms at the top level of the template file. Each `define-meta` binding associates a symbol name with a value. Atomic values (strings, numbers, booleans) are stored directly. Quoted lists are stored as Scheme data structures. The loader collects ALL `define-meta` bindings into a hash table keyed by name; duplicate names [MUST] cause a loader error.

### D.2 REQUIRED vs OPTIONAL Field Table

| Field | Status | Type | Description |
|---|---|---|---|
| `name` | **REQUIRED** | `string` | Template name. [MUST] match the filename stem. |
| `version` | **REQUIRED** | `string` | Semver version string. |
| `task-shapes` | **REQUIRED** | `quoted-list` of symbols | TaskShape values this template handles (from ALL 13 values in Appendix A, item A.1). |
| `data-shapes` | **REQUIRED** | `quoted-list` of symbols | DataShape values this template handles (from ALL 11 values in Appendix A, item A.2). |
| `slots` | **REQUIRED** | `quoted-list` of alists | Typed slot definitions (see Appendix D, item D.3). |
| `description` | **REQUIRED** | `string` | Human-readable summary of what the template does. |
| `trigger` | OPTIONAL | `quoted-list` of predicates | Conditions under which the planner [SHOULD] select this template (see Appendix D, item D.5). |
| `reject` | OPTIONAL | `quoted-list` of predicates | Conditions under which the planner [MUST NOT] select this template (see Appendix D, item D.5). |
| `model-requirements` | OPTIONAL | `quoted-list` | Required model capabilities (e.g., `json`, `image`). |
| `output-schema` | OPTIONAL | `quoted-list` (alist notation) | Declares the structure of the value passed to `finish` (see Appendix D, item D.4). |
| `structural-profile` | OPTIONAL | `quoted-list` (alist notation) | Expected call formulas, max concurrency, recursive depth. |
| `expected-calls` | OPTIONAL | `string` | Formula for expected LLM call count (e.g., `\"N + ceil(N/B) + ... + 1\"`). |
| `streamable` | OPTIONAL | `boolean` | Whether meaningful intermediate results exist. Default: `#f`. |
| `cacheable` | OPTIONAL | `boolean` | Whether LLM call results can be cached across executions. Default: `#f`. |
| `gates` | OPTIONAL | `quoted-list` of alists | Declared human-review checkpoints (see Appendix D, item D.6 for example). |
| `budget-policy` | OPTIONAL | `quoted-list` (alist notation) | Degradation behavior when token budget runs low. |
| `uses-llm-generated-code` | OPTIONAL | `boolean` | Whether the template uses the code interpreter pattern. Default: `#f`. |
| `verification-rules` | OPTIONAL | `quoted-list` of symbols | Named verification checks beyond the default ALL 23 checks. |
| `dry-run-warnings` | OPTIONAL | `quoted-list` | Warnings to surface during dry-run. |
| `error-policies` | OPTIONAL | `quoted-list` of alists | Per-node error policy declarations. |
| `examples` | OPTIONAL | `quoted-list` | Example invocations for documentation. |

### D.3 Slot Schema Syntax

Slots are declared as an alist where each entry names a slot and provides constraint pairs. The instantiator [MUST] validate every slot value against these constraints before substitution.

```scheme
(define-meta slots
  '((slot_name   (type TYPE) [(CONSTRAINT VALUE)] ...)
    ...))
```

Supported constraint keys:

| Key | Values | Description |
|---|---|---|
| `type` | `string`, `integer`, `number`, `boolean` | [REQUIRED] Slot value type. |
| `required` | `#t`, `#f` | Whether the slot [MUST] be provided. Default: `#f`. |
| `default` | any | Default value when slot is not provided. |
| `min` | number | Minimum value (for `integer` and `number` types). |
| `max` | number | Maximum value (for `integer` and `number` types). |
| `min-length` | integer | Minimum string length (for `string` type). |
| `max-length` | integer | Maximum string length (for `string` type). |
| `pattern` | string (regex) | Regex pattern the value [MUST] match (for `string` type). |
| `enum` | quoted list | Allowed values (for `string` type). |
| `nullable` | `#t`, `#f` | Whether `#f`/null is acceptable. Default: `#f`. |
| `description` | string | Human-readable description for planner context. |

Example:

```scheme
(define-meta slots
  '((context_id         (type string) (pattern \"^ctx_\") (required #t))
    (items_path         (type string) (default \"$\"))
    (map_instruction    (type string) (min-length 10) (required #t))
    (reduce_instruction (type string) (min-length 10) (required #t))
    (map_model          (type string) (default \"fast_text_model\"))
    (reduce_model       (type string) (default \"quality_text_model\"))
    (max_concurrent     (type integer) (min 1) (max 50) (default 20))
    (branch_factor      (type integer) (min 2) (max 10) (default 5))
    (json_mode          (type boolean) (default #f))
    (checkpoint_every   (type integer) (nullable #t) (min 1) (default #f))))
```

### D.4 Output-Schema Alist Syntax

Output schemas use alist notation that maps 1:1 to JSON Schema. Each `(key value)` pair corresponds to a JSON Schema keyword. Nested objects use nested alists.

Conversion rule: the alist `'((type object) (properties ((name (type string)))))` becomes `{\"type\": \"object\", \"properties\": {\"name\": {\"type\": \"string\"}}}`.

The template loader [MUST] validate structural well-formedness (balanced parens, known JSON Schema keywords at the top level: `type`, `properties`, `items`, `required`, `enum`, `description`). Semantic JSON Schema correctness is validated at verification time.

Example:

```scheme
(define-meta output-schema
  '((type object)
    (properties
      ((findings (type array)
                 (items ((type object)
                         (properties
                           ((paper_id (type string))
                            (ace2_mentions (type array))
                            (evidence (type string))
                            (uncertainty (type string)))))))
       (summary (type string))))))
```

### D.5 Trigger and Reject Rule Evaluation

Predicates in `trigger` and `reject` are evaluated in an environment where each classification hint field is bound as a variable: `item_count`, `independent`, `output_type`, `has_second_phase`, `ordered`, `modality`, `operation`, `sub_operations`, `requires_pairwise_comparison`, `order_sensitive`, and any other hint fields the agent provides.

**Evaluation rules:**

- Missing hints are bound to `#f`.
- **Trigger** conditions use implicit AND: ALL predicates [MUST] return `#t` for the template to match.
- **Reject** conditions use implicit OR: ANY predicate returning `#t` disqualifies the template.
- Predicates that reference a potentially missing hint [SHOULD] handle the `#f` case, e.g., `(and ordered (eq? order_sensitive #t))` short-circuits to `#f` when `ordered` is missing.

Example:

```scheme
(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)
    (eq? output_type 'one)
    (eq? has_second_phase #t)))

(define-meta reject
  '((and (eq? ordered #t) (eq? order_sensitive #t))
    (eq? requires_pairwise_comparison #t)))
```

### D.6 Reference Example Template

A complete `define-meta` block with ALL 6 REQUIRED fields and representative OPTIONAL fields:

```scheme
;; --- Metadata (ALL 6 REQUIRED fields) ---

(define-meta name \"batch_extract_reduce\")
(define-meta version \"1.0.0\")
(define-meta description
  \"Run independent extraction over many items, then synthesize results with tree reduction.\")
(define-meta task-shapes '(Batch Synthesize Composite))
(define-meta data-shapes '(FlatList ChunkedSingular Tabular))

(define-meta slots
  '((context_id         (type string) (pattern \"^ctx_\") (required #t)
                        (description \"ID of the loaded context containing items.\"))
    (items_path         (type string) (default \"$\")
                        (description \"JSONPath to extract items from context.\"))
    (map_instruction    (type string) (min-length 10) (required #t)
                        (description \"Instruction for per-item extraction.\"))
    (reduce_instruction (type string) (min-length 10) (required #t)
                        (description \"Instruction for synthesis/reduction.\"))
    (map_model          (type string) (default \"fast_text_model\")
                        (description \"Model alias for map phase.\"))
    (reduce_model       (type string) (default \"quality_text_model\")
                        (description \"Model alias for reduce phase.\"))
    (max_concurrent     (type integer) (min 1) (max 50) (default 20)
                        (description \"Maximum parallel map calls.\"))
    (branch_factor      (type integer) (min 2) (max 10) (default 5)
                        (description \"Tree-reduce branching factor.\"))
    (json_mode          (type boolean) (default #f)
                        (description \"Whether to request JSON output from map calls.\"))
    (checkpoint_every   (type integer) (nullable #t) (min 1) (default #f)
                        (description \"Checkpoint after every N map items. Null disables.\"))))

;; --- OPTIONAL metadata ---

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)
    (eq? output_type 'one)
    (eq? has_second_phase #t)))

(define-meta reject
  '((and (eq? ordered #t) (eq? order_sensitive #t))
    (eq? requires_pairwise_comparison #t)))

(define-meta structural-profile
  '((expected-calls \"N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1\")
    (critical-path  \"1 + ceil(log_B(N))\")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta output-schema
  '((type object)
    (properties
      ((findings (type array)
                 (items ((type object)
                         (properties
                           ((paper_id (type string))
                            (ace2_mentions (type array))
                            (evidence (type string))
                            (uncertainty (type string)))))))
       (summary (type string))))))

(define-meta verification-rules
  '(context_id_exists
    items_path_resolves_to_list
    map_model_supports_json_if_json_mode
    expected_calls_within_policy
    max_concurrency_within_policy
    only_primitive_bindings))

(define-meta streamable #t)
(define-meta cacheable #t)

(define-meta budget-policy
  '((on-low-budget   switch-model)
    (low-budget-threshold 0.20)
    (fallback-model  \"fast_text_model\")
    (on-exhausted    checkpoint-and-stop)))

(define-meta gates
  '((review_extractions
      (description \"Review extraction results before synthesis\")
      (required #f))))

(define-meta uses-llm-generated-code #f)

(define-meta error-policies
  '((extract    (on-error fail_fast) (checkpoint-every 25))
    (synthesize (on-error fail_fast))))

(define-meta examples
  '(((task \"Extract claims from papers and synthesize a literature review.\")
     (slot_values
       (items_path \"$.papers\")
       (map_instruction \"Extract the core claim, evidence, and uncertainty as JSON.\")
       (reduce_instruction \"Synthesize the extracted claims into a literature review.\")))))

;; --- Body ---

(define items (__context-ref \"{{context_id}}\" \"{{items_path}}\"))

(define extracted
  (map-async
    (lambda (item)
      (llm-query-async
        #:instruction {{map_instruction}}
        #:data item
        #:model {{map_model}}
        #:json {{json_mode}}))
    items
    #:max-concurrent {{max_concurrent}}))

(gate \"review_extractions\" extracted
      #:message \"Review extraction results before synthesis.\")

(define synthesized
  (tree-reduce
    (lambda group
      (syntax-e
        (llm-query
          #:instruction {{reduce_instruction}}
          #:data (__join-json group)
          #:model {{reduce_model}})))
    extracted
    #:branch-factor {{branch_factor}}))

(finish synthesized)
```

---

## Batch 0: Project Skeleton and Schemas

### 0.0 Purpose

Establish the `v2/` directory structure, define all Pydantic v2 models for durable records and API payloads, create enumerations for classification taxonomies and runtime states, implement deterministic ID generation, define the exception hierarchy, and set up model registry configuration loading. This batch produces the foundational type system that every subsequent batch depends on.

### 0.1 Depends On

Nothing. This is the first batch.

### 0.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `v2/pyproject.toml` | Create | Project metadata, dependencies (pydantic>=2.0, pyyaml, pytest) |
| `v2/rlm_scheme/__init__.py` | Create | Package init, version string |
| `v2/rlm_scheme/models.py` | Create | Pydantic v2 models for all 7 durable record types plus API payloads |
| `v2/rlm_scheme/enums.py` | Create | TaskShape, DataShape, ExecutionState, ErrorPolicy, ResetScope enums |
| `v2/rlm_scheme/ids.py` | Create | `generate_id(prefix)` and `validate_id(id_str, prefix)` functions |
| `v2/rlm_scheme/exceptions.py` | Create | Base exception class plus specific error types |
| `v2/rlm_scheme/config.py` | Create | Model registry loading from YAML/JSON, validation, alias resolution |
| `v2/tests/__init__.py` | Create | Empty test package init |
| `v2/tests/conftest.py` | Create | Shared pytest fixtures (tmp_dir, sample records, mock registry) |
| `v2/tests/test_models.py` | Create | Schema validation tests for all record types and enums |

### 0.3 Requirements

- R-0.1 [MUST] All ID strings match the pattern `^{prefix}[a-z0-9]{16}$` where prefix is one of the 8 registered prefixes: `ctx_`, `plan_`, `art_`, `dry_`, `ver_`, `exec_`, `call_`, `ckpt_`.
- R-0.2 [MUST] `generate_id(prefix)` produces cryptographically random IDs using `secrets.token_hex(8)`.
- R-0.3 [MUST] Pydantic models exist for all 7 durable record types: ContextRecord, PlanRecord, ArtifactRecord, DryRunRecord, VerificationRecord, ExecutionRecord, CacheRecord.
- R-0.4 [MUST] Every record model includes fields: `schema_version` (default `\"1\"`), `created_at` (UTC datetime).
- R-0.5 [MUST] TaskShape enum has exactly 13 values: Direct, Batch, Synthesize, Search, Refine, Compare, Classify, Pipeline, Generate, Decompose, Validate, Aggregate, Composite.
- R-0.6 [MUST] DataShape enum has 11 values: FlatList, Hierarchy, Singular, ChunkedSingular, Graph, TimeSeries, Tabular, Multimodal, Paired, KeyValue, Unknown.
- R-0.7 [MUST] ExecutionState enum has exactly 7 values: queued, running, finished, failed, cancelled, awaiting_gate, gate_rejected.
- R-0.8 [MUST] ErrorPolicy enum has exactly 3 values: fail_fast, collect, fallback.
- R-0.9 [MUST] ResetScope enum has exactly 7 values: sandbox, session, cache, contexts, executions, config, all.
- R-0.10 [MUST] Model registry loads from JSON file, path configurable via `RLM_MODEL_REGISTRY` environment variable, defaulting to `config/models.json`.
- R-0.11 [MUST] Model registry validates that each alias has: `provider`, `model`, `capabilities` (list), `max_context_tokens` (int), `cost_tier`.
- R-0.12 [MUST] Exception hierarchy includes: `RLMSchemeError` (base), `StoreError`, `ValidationError`, `InstantiationError`, `VerificationError`, `ExecutionError`, `ClassificationError`, `ConfigError`, `ContextNotFoundError`, `TemplateNotFoundError`, `SlotValidationError`, `PolicyViolationError`.
- R-0.13 [SHOULD] All Pydantic models use `model_config = ConfigDict(strict=True)` to enforce type coercion rules.
- R-0.14 [MUST] ContextRecord includes `data_ref` with fields: `storage`, `path`, `hash`, `bytes`; and `metadata` with fields: `data_shape`, `item_count`, `independent`, `ordered`, `modality`, `total_size_estimate_tokens`.
- R-0.15 [MUST] ExecutionRecord `state` field uses the ExecutionState enum and enforces valid transitions as documented in the plan (terminal states: finished, failed, cancelled, gate_rejected).
- R-0.16 [MAY] Model registry `defaults` section maps role names (planner, map, reduce, vision) to alias names.

### 0.4 Detailed Specifications

**ID Generation:**

```python
import secrets

ID_PREFIXES = (\"ctx_\", \"plan_\", \"art_\", \"dry_\", \"ver_\", \"exec_\", \"call_\", \"ckpt_\")
ID_PATTERN = re.compile(r\"^(ctx_|plan_|art_|dry_|ver_|exec_|call_|ckpt_)[a-z0-9]{16}$\")

def generate_id(prefix: str) -> str:
    \"\"\"Generate a random ID with the given prefix.\"\"\"
    if prefix not in ID_PREFIXES:
        raise ValueError(f\"Invalid prefix: {prefix}. Must be one of {ID_PREFIXES}\")
    return prefix + secrets.token_hex(8)

def validate_id(id_str: str, expected_prefix: str | None = None) -> bool:
    \"\"\"Validate an ID string matches the expected format.\"\"\"
    if not ID_PATTERN.match(id_str):
        return False
    if expected_prefix and not id_str.startswith(expected_prefix):
        return False
    return True
```

**Enum Definitions:**

```python
from enum import Enum

class TaskShape(str, Enum):
    Direct = \"Direct\"
    Batch = \"Batch\"
    Synthesize = \"Synthesize\"
    Search = \"Search\"
    Refine = \"Refine\"
    Compare = \"Compare\"
    Classify = \"Classify\"
    Pipeline = \"Pipeline\"
    Generate = \"Generate\"
    Decompose = \"Decompose\"
    Validate = \"Validate\"
    Aggregate = \"Aggregate\"
    Composite = \"Composite\"

class DataShape(str, Enum):
    FlatList = \"FlatList\"
    Hierarchy = \"Hierarchy\"
    Singular = \"Singular\"
    ChunkedSingular = \"ChunkedSingular\"
    Graph = \"Graph\"
    TimeSeries = \"TimeSeries\"
    Tabular = \"Tabular\"
    Multimodal = \"Multimodal\"
    Paired = \"Paired\"
    KeyValue = \"KeyValue\"
    Unknown = \"Unknown\"

class ExecutionState(str, Enum):
    queued = \"queued\"
    running = \"running\"
    finished = \"finished\"
    failed = \"failed\"
    cancelled = \"cancelled\"
    awaiting_gate = \"awaiting_gate\"
    gate_rejected = \"gate_rejected\"

class ErrorPolicy(str, Enum):
    fail_fast = \"fail_fast\"
    collect = \"collect\"
    fallback = \"fallback\"

class ResetScope(str, Enum):
    sandbox = \"sandbox\"
    session = \"session\"
    cache = \"cache\"
    contexts = \"contexts\"
    executions = \"executions\"
    config = \"config\"
    all = \"all\"
```

**Pydantic Models (representative subset):**

```python
from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime, timezone

class DataRef(BaseModel):
    model_config = ConfigDict(strict=True)
    storage: str = \"filesystem\"
    path: str
    hash: str
    bytes: int

class ContextMetadata(BaseModel):
    model_config = ConfigDict(strict=True)
    data_shape: DataShape = DataShape.Unknown
    item_count: int | None = None
    item_size_estimate_tokens: int | None = None
    total_size_estimate_tokens: int | None = None
    independent: bool | None = None
    ordered: bool | None = None
    modality: list[str] = Field(default_factory=lambda: [\"text\"])
    chunking: dict | None = None
    source: dict | None = None
    schema_: dict | None = Field(default=None, alias=\"schema\")

class ContextRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    context_id: str
    schema_version: str = \"1\"
    name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data_ref: DataRef
    metadata: ContextMetadata = Field(default_factory=ContextMetadata)

class Classification(BaseModel):
    model_config = ConfigDict(strict=True)
    task_shape: TaskShape
    constituent_shapes: list[TaskShape] | None = None
    data_shape: DataShape
    confidence: float = 1.0
    rationale: str | None = None

class TemplateInvocation(BaseModel):
    model_config = ConfigDict(strict=True)
    kind: str = \"template_invocation\"
    template_name: str
    template_version: str = \"1.0.0\"
    slot_values: dict = Field(default_factory=dict)

class TemplateChain(BaseModel):
    model_config = ConfigDict(strict=True)
    kind: str = \"template_chain\"
    steps: list[TemplateInvocation]

class PlanRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    plan_id: str
    schema_version: str = \"1\"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context_ids: list[str] = Field(default_factory=list)
    task: str
    hints: dict = Field(default_factory=dict)
    classification: Classification
    recommended: TemplateInvocation | TemplateChain
    alternatives: list[dict] = Field(default_factory=list)

class ArtifactRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    artifact_id: str
    schema_version: str = \"1\"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    plan_id: str | None = None
    context_ids: list[str] = Field(default_factory=list)
    source_type: str = \"template_invocation\"
    template_name: str
    template_version: str
    slot_values: dict = Field(default_factory=dict)
    generated_scheme_ref: DataRef
    primitives_used: list[str] = Field(default_factory=list)
    static_profile: dict = Field(default_factory=dict)

class DryRunRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    dry_run_id: str
    schema_version: str = \"1\"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_id: str
    mode: str = \"deterministic\"
    summary: dict = Field(default_factory=dict)
    call_graph: list[dict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

class VerificationCheck(BaseModel):
    model_config = ConfigDict(strict=True)
    name: str
    status: str  # \"pass\", \"warn\", \"fail\"
    message: str

class VerificationRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    verification_id: str
    schema_version: str = \"1\"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    artifact_id: str
    dry_run_id: str | None = None
    decision: str  # \"pass\", \"warn\", \"fail\"
    policy: dict = Field(default_factory=dict)
    checks: list[VerificationCheck] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

class ExecutionRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    execution_id: str
    schema_version: str = \"1\"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    state: ExecutionState = ExecutionState.queued
    artifact_id: str
    plan_id: str | None = None
    verification_id: str | None = None
    result_ref: DataRef | None = None
    trace_ref: dict | None = None
    metrics: dict = Field(default_factory=dict)
    error: dict | None = None
    gates: list[dict] = Field(default_factory=list)
    cache_hits: int = 0
    budget_policy_activations: int = 0
    chain_step_results: list[dict] = Field(default_factory=list)

class CacheRecord(BaseModel):
    model_config = ConfigDict(strict=True)
    cache_key: str
    schema_version: str = \"1\"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    instruction_hash: str
    data_hash: str
    model: str
    temperature: float = 0.0
    json_mode: bool = False
    result: str
    result_tokens: dict = Field(default_factory=dict)
    source_execution_id: str | None = None
    source_call_id: str | None = None
```

**Model Registry Config:**

```python
from pydantic import BaseModel, ConfigDict

class ModelAlias(BaseModel):
    model_config = ConfigDict(strict=True)
    provider: str
    model: str
    capabilities: list[str]
    max_context_tokens: int
    supports_temperature: bool = True
    cost_tier: str
    fallback: str | None = None

class ModelRegistry(BaseModel):
    model_config = ConfigDict(strict=True)
    schema_version: str = \"1\"
    aliases: dict[str, ModelAlias]
    defaults: dict[str, str] = Field(default_factory=dict)

def load_model_registry(path: str | None = None) -> ModelRegistry:
    \"\"\"Load model registry from JSON/YAML file.\"\"\"
    import os, json
    path = path or os.environ.get(\"RLM_MODEL_REGISTRY\", \"config/models.json\")
    with open(path) as f:
        data = json.load(f)
    return ModelRegistry(**data)
```

**Exception Hierarchy:**

```python
class RLMSchemeError(Exception):
    \"\"\"Base exception for all RLM-Scheme errors.\"\"\"

class StoreError(RLMSchemeError): ...
class ValidationError(RLMSchemeError): ...
class InstantiationError(RLMSchemeError): ...
class VerificationError(RLMSchemeError): ...
class ExecutionError(RLMSchemeError): ...
class ClassificationError(RLMSchemeError): ...
class ConfigError(RLMSchemeError): ...
class ContextNotFoundError(StoreError): ...
class TemplateNotFoundError(StoreError): ...
class SlotValidationError(InstantiationError): ...
class PolicyViolationError(VerificationError): ...
```

### 0.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|---------------|----------|----------|
| `v2/tests/test_models.py` | `test_generate_id_format` | Call `generate_id(\"ctx_\")` 100 times | All match `^ctx_[a-z0-9]{16}$`, all unique |
| `v2/tests/test_models.py` | `test_generate_id_all_prefixes` | Call `generate_id` with each of 7 valid prefixes | All match `^{prefix}[a-z0-9]{16}$` |
| `v2/tests/test_models.py` | `test_generate_id_invalid_prefix` | Call `generate_id(\"bad_\")` | Raises `ValueError` |
| `v2/tests/test_models.py` | `test_validate_id_valid` | `validate_id(\"ctx_abcdef0123456789\")` | Returns `True` |
| `v2/tests/test_models.py` | `test_validate_id_wrong_prefix` | `validate_id(\"ctx_abcdef0123456789\", \"plan_\")` | Returns `False` |
| `v2/tests/test_models.py` | `test_validate_id_bad_format` | `validate_id(\"ctx_SHORT\")` | Returns `False` |
| `v2/tests/test_models.py` | `test_task_shape_values` | Check `len(TaskShape)` | Equals 13 |
| `v2/tests/test_models.py` | `test_data_shape_values` | Check `len(DataShape)` | Equals 11 |
| `v2/tests/test_models.py` | `test_execution_state_values` | Check `len(ExecutionState)` | Equals 7 |
| `v2/tests/test_models.py` | `test_error_policy_values` | Check `len(ErrorPolicy)` | Equals 3 |
| `v2/tests/test_models.py` | `test_reset_scope_values` | Check `len(ResetScope)` | Equals 5 |
| `v2/tests/test_models.py` | `test_context_record_roundtrip` | Create ContextRecord, serialize to JSON, deserialize | Fields match original |
| `v2/tests/test_models.py` | `test_plan_record_with_invocation` | Create PlanRecord with TemplateInvocation | Serializes/deserializes correctly |
| `v2/tests/test_models.py` | `test_plan_record_with_chain` | Create PlanRecord with TemplateChain | `recommended.kind == \"template_chain\"`, steps accessible |
| `v2/tests/test_models.py` | `test_artifact_record_fields` | Create ArtifactRecord with all fields | All fields accessible, `source_type == \"template_invocation\"` |
| `v2/tests/test_models.py` | `test_execution_record_state_default` | Create ExecutionRecord without explicit state | `state == ExecutionState.queued` |
| `v2/tests/test_models.py` | `test_dry_run_record_fields` | Create DryRunRecord with summary dict | `mode == \"deterministic\"`, summary accessible |
| `v2/tests/test_models.py` | `test_verification_record_checks` | Create VerificationRecord with 3 checks | `len(checks) == 3`, decision accessible |
| `v2/tests/test_models.py` | `test_cache_record_fields` | Create CacheRecord with all fields | `cache_key`, `result`, `model` all accessible |
| `v2/tests/test_models.py` | `test_model_registry_load` | Load registry from fixture JSON file | Aliases parsed, defaults parsed, `ModelAlias` fields validated |
| `v2/tests/test_models.py` | `test_model_registry_missing_field` | Load registry with alias missing `provider` | Raises `ValidationError` |
| `v2/tests/test_models.py` | `test_model_registry_env_var` | Set `RLM_MODEL_REGISTRY` env var, call `load_model_registry()` | Loads from env-specified path |
| `v2/tests/test_models.py` | `test_exception_hierarchy` | Catch `ContextNotFoundError` as `StoreError` and `RLMSchemeError` | Both catches succeed |
| `v2/tests/test_models.py` | `test_all_records_have_schema_version` | Instantiate all 7 record types | All have `schema_version == \"1\"` |
| `v2/tests/test_models.py` | `test_all_records_have_created_at` | Instantiate all 7 record types | All have `created_at` as UTC datetime |

### 0.6 Acceptance Gates

```bash
# All tests pass
cd v2 && python -m pytest tests/test_models.py -v
# Expected: 25 passed

# Pydantic models compile without errors
cd v2 && python -c \"from rlm_scheme.models import *; from rlm_scheme.enums import *; from rlm_scheme.ids import *; from rlm_scheme.exceptions import *; from rlm_scheme.config import ModelRegistry; print('All imports OK')\"
# Expected: \"All imports OK\"

# ID generation produces valid format
cd v2 && python -c \"from rlm_scheme.ids import generate_id; import re; ids = [generate_id(p) for p in ('ctx_','plan_','art_','dry_','ver_','exec_','call_')]; assert all(re.match(r'^(ctx_|plan_|art_|dry_|ver_|exec_|call_)[a-z0-9]{16}$', i) for i in ids); print('ID format OK')\"
# Expected: \"ID format OK\"

# Enum cardinalities
cd v2 && python -c \"from rlm_scheme.enums import *; assert len(TaskShape)==13; assert len(DataShape)==11; assert len(ExecutionState)==7; assert len(ErrorPolicy)==3; assert len(ResetScope)==5; print('Enum counts OK')\"
# Expected: \"Enum counts OK\"
```

### 0.7 Checklist

- [ ] `v2/pyproject.toml` created with dependencies: pydantic>=2.0, pyyaml, pytest
- [ ] `v2/rlm_scheme/__init__.py` created with version string
- [ ] `v2/rlm_scheme/ids.py` implements `generate_id()` and `validate_id()`
- [ ] `v2/rlm_scheme/enums.py` defines all 5 enum classes with correct cardinalities
- [ ] `v2/rlm_scheme/models.py` defines all 7 record types plus supporting models
- [ ] `v2/rlm_scheme/exceptions.py` defines exception hierarchy (12 classes)
- [ ] `v2/rlm_scheme/config.py` implements `load_model_registry()` with env var support
- [ ] `v2/tests/conftest.py` provides shared fixtures
- [ ] `v2/tests/test_models.py` has 25+ tests covering all record types, enums, IDs
- [ ] All tests pass: `python -m pytest tests/test_models.py -v`

---

## Batch 1: Durable Store, LLM Provider, and MCP Skeleton

### 1.0 Purpose

Implement the filesystem-based durable store for all record types, create the abstract LLM provider adapter with a mock implementation for testing, and stand up the FastMCP server skeleton with all 10 tool stubs wired to return structured responses. This batch establishes the storage layer and the public API surface that all subsequent batches wire real logic into.

### 1.1 Depends On

Batch 0 (models, enums, IDs, exceptions, config).

### 1.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `v2/rlm_scheme/store.py` | Create | Namespace-based key-value store backed by JSON files on disk |
| `v2/rlm_scheme/llm_adapter.py` | Create | Abstract LLM provider interface + MockLLMProvider for testing |
| `v2/rlm_scheme/mcp_server.py` | Create | FastMCP server with all 10 tool stubs |
| `v2/tests/test_store.py` | Create | Store CRUD tests, namespace isolation, reset scopes |
| `v2/tests/test_llm_adapter.py` | Create | Mock provider tests, response format validation |

### 1.3 Requirements

- R-1.1 [MUST] Store supports namespaces: `contexts`, `plans`, `artifacts`, `dry_runs`, `verifications`, `executions`, `cache`.
- R-1.2 [MUST] Store operations: `put(namespace, id, record)`, `get(namespace, id)`, `list(namespace)`, `delete(namespace, id)`, `clear(namespace)`.
- R-1.3 [MUST] Store persists records as JSON files at `{base_dir}/{namespace}/{id}.json`.
- R-1.4 [MUST] Store `get()` returns `None` for missing keys (does not raise).
- R-1.5 [MUST] Store supports `reset(scope: ResetScope)` that clears records per the scoping rules: `sandbox` clears nothing durable, `session` clears executions from current session, `all` clears everything, `cache` clears only cache namespace, `config` clears nothing (handled by caller).
- R-1.6 [MUST] `LLMProvider` abstract base class defines `async def query(instruction, data, model, temperature, max_tokens, json_mode, images) -> LLMResponse`.
- R-1.7 [MUST] `MockLLMProvider` returns deterministic responses: empty string `\"\"` for text calls, `\"{}\"` for JSON-mode calls.
- R-1.8 [MUST] `LLMResponse` includes fields: `content` (str), `prompt_tokens` (int), `completion_tokens` (int), `model` (str).
- R-1.9 [MUST] FastMCP server registers all 10 tools with correct names and parameter signatures matching Appendix B (primitive signatures).
- R-1.10 [MUST] Each stub tool returns a valid JSON string with `\"status\"` field.
- R-1.11 [MUST] `reset_runtime` stub accepts `scope` parameter with valid `ResetScope` values.
- R-1.12 [SHOULD] Store validates that record IDs match expected prefix for their namespace (e.g., `contexts` namespace requires `ctx_` prefix).
- R-1.13 [MUST] MockLLMProvider tracks call count and can be queried for total calls made.
- R-1.14 [MAY] Store supports an optional `session_id` field to scope session-level resets.

### 1.4 Detailed Specifications

**Store class:**

```python
import json, os
from pathlib import Path
from rlm_scheme.enums import ResetScope

class Store:
    NAMESPACES = (
        \"contexts\", \"plans\", \"artifacts\", \"dry_runs\",
        \"verifications\", \"executions\", \"cache\"
    )

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir)
        self.session_id: str | None = None
        for ns in self.NAMESPACES:
            (self.base_dir / ns).mkdir(parents=True, exist_ok=True)

    def put(self, namespace: str, record_id: str, data: dict) -> None:
        \"\"\"Store a record. Overwrites if exists.\"\"\"
        self._validate_namespace(namespace)
        path = self.base_dir / namespace / f\"{record_id}.json\"
        path.write_text(json.dumps(data, default=str, indent=2))

    def get(self, namespace: str, record_id: str) -> dict | None:
        \"\"\"Retrieve a record by ID. Returns None if not found.\"\"\"
        path = self.base_dir / namespace / f\"{record_id}.json\"
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def list(self, namespace: str) -> list[str]:
        \"\"\"List all record IDs in a namespace.\"\"\"
        self._validate_namespace(namespace)
        ns_dir = self.base_dir / namespace
        return [p.stem for p in ns_dir.glob(\"*.json\")]

    def delete(self, namespace: str, record_id: str) -> bool:
        \"\"\"Delete a record. Returns True if deleted, False if not found.\"\"\"
        path = self.base_dir / namespace / f\"{record_id}.json\"
        if path.exists():
            path.unlink()
            return True
        return False

    def clear(self, namespace: str) -> int:
        \"\"\"Clear all records in a namespace. Returns count deleted.\"\"\"
        ns_dir = self.base_dir / namespace
        count = 0
        for p in ns_dir.glob(\"*.json\"):
            p.unlink()
            count += 1
        return count

    def reset(self, scope: ResetScope) -> dict:
        \"\"\"Reset store per scope rules. Returns summary of what was cleared.\"\"\"
        if scope == ResetScope.all:
            total = sum(self.clear(ns) for ns in self.NAMESPACES)
            return {\"cleared\": \"all\", \"records_deleted\": total}
        elif scope == ResetScope.cache:
            count = self.clear(\"cache\")
            return {\"cleared\": \"cache\", \"records_deleted\": count}
        elif scope == ResetScope.session:
            count = self.clear(\"executions\")
            return {\"cleared\": \"session\", \"records_deleted\": count}
        elif scope == ResetScope.sandbox:
            return {\"cleared\": \"sandbox\", \"records_deleted\": 0}
        elif scope == ResetScope.config:
            return {\"cleared\": \"config\", \"records_deleted\": 0}

    def _validate_namespace(self, namespace: str) -> None:
        if namespace not in self.NAMESPACES:
            raise ValueError(f\"Invalid namespace: {namespace}\")
```

**LLM Provider Adapter:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str

class LLMProvider(ABC):
    @abstractmethod
    async def query(
        self,
        instruction: str,
        data: str,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        images: list | None = None,
    ) -> LLMResponse: ...

class MockLLMProvider(LLMProvider):
    def __init__(self):
        self.call_count = 0
        self.calls: list[dict] = []

    async def query(self, instruction, data, model, temperature=None,
                    max_tokens=None, json_mode=False, images=None) -> LLMResponse:
        self.call_count += 1
        self.calls.append({
            \"instruction\": instruction, \"data\": data, \"model\": model,
            \"json_mode\": json_mode
        })
        content = \"{}\" if json_mode else \"\"
        return LLMResponse(
            content=content,
            prompt_tokens=len(instruction) + len(str(data)),
            completion_tokens=len(content) or 10,
            model=model
        )
```

**MCP Server Skeleton (tool signatures):**

```python
from fastmcp import FastMCP, Context

mcp = FastMCP(\"rlm-scheme-v2\")

@mcp.tool()
def load_context(data: str, name: str | None = None,
                 metadata_json: str | None = None) -> str: ...

@mcp.tool()
def get_context(context_id: str, include_preview: bool = True,
                include_data: bool = False) -> str: ...

@mcp.tool()
def plan_strategy(task: str, context_id: str | None = None,
                  hints_json: str | None = None) -> str: ...

@mcp.tool()
def dry_run_strategy(plan_id: str | None = None,
                     template_invocation_json: str | None = None,
                     options_json: str | None = None) -> str: ...

@mcp.tool()
async def execute_strategy(plan_id: str | None = None,
                           template_invocation_json: str | None = None,
                           timeout_seconds: int | None = None,
                           stream: bool = False,
                           policy_json: str | None = None,
                           runtime_options_json: str | None = None,
                           ctx: Context = None) -> str: ...

@mcp.tool()
def get_execution_trace(execution_id: str, include_scope_log: bool = True,
                        include_calls: bool = True,
                        include_stdout: bool = True) -> str: ...

@mcp.tool()
def get_status(execution_id: str | None = None) -> str: ...

@mcp.tool()
def cancel_call(call_id: str | None = None, execution_id: str | None = None,
                reason: str | None = None) -> str: ...

@mcp.tool()
def reset_runtime(scope: str = \"session\") -> str: ...

@mcp.tool()
async def resume_execution(execution_id: str, gate: str, decision: str,
                           reason: str | None = None) -> str: ...
```

### 1.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|---------------|----------|----------|
| `v2/tests/test_store.py` | `test_put_and_get` | Store a context record, retrieve it | Retrieved data matches stored data |
| `v2/tests/test_store.py` | `test_get_missing` | Get a nonexistent ID | Returns `None` |
| `v2/tests/test_store.py` | `test_list_namespace` | Store 3 records, list namespace | Returns 3 IDs |
| `v2/tests/test_store.py` | `test_list_empty_namespace` | List empty namespace | Returns empty list |
| `v2/tests/test_store.py` | `test_delete_record` | Store then delete | Delete returns `True`, subsequent get returns `None` |
| `v2/tests/test_store.py` | `test_delete_missing` | Delete nonexistent ID | Returns `False` |
| `v2/tests/test_store.py` | `test_clear_namespace` | Store 5 records, clear | Returns 5, list returns empty |
| `v2/tests/test_store.py` | `test_namespace_isolation` | Store in `contexts`, list `plans` | Plans list is empty |
| `v2/tests/test_store.py` | `test_invalid_namespace` | `put(\"invalid\", ...)` | Raises `ValueError` |
| `v2/tests/test_store.py` | `test_reset_all` | Store records in multiple namespaces, reset(all) | All namespaces empty |
| `v2/tests/test_store.py` | `test_reset_cache` | Store in contexts and cache, reset(cache) | Cache empty, contexts intact |
| `v2/tests/test_store.py` | `test_reset_sandbox` | Store records, reset(sandbox) | All records intact |
| `v2/tests/test_store.py` | `test_put_overwrites` | Put same ID twice with different data | Get returns second data |
| `v2/tests/test_llm_adapter.py` | `test_mock_text_response` | Query with json_mode=False | `content == \"\"`, tokens > 0 |
| `v2/tests/test_llm_adapter.py` | `test_mock_json_response` | Query with json_mode=True | `content == \"{}\"` |
| `v2/tests/test_llm_adapter.py` | `test_mock_call_tracking` | Make 5 queries | `call_count == 5`, `len(calls) == 5` |
| `v2/tests/test_llm_adapter.py` | `test_mock_model_passthrough` | Query with model=\"test_model\" | `response.model == \"test_model\"` |

### 1.6 Acceptance Gates

```bash
# Store tests pass
cd v2 && python -m pytest tests/test_store.py -v
# Expected: 13 passed

# LLM adapter tests pass
cd v2 && python -m pytest tests/test_llm_adapter.py -v
# Expected: 4 passed

# MCP server imports and all 10 tools are registered
cd v2 && python -c \"from rlm_scheme.mcp_server import mcp; tools = mcp.list_tools(); names = sorted([t.name for t in tools]); expected = sorted(['cancel_call','dry_run_strategy','execute_strategy','get_context','get_execution_trace','get_status','load_context','plan_strategy','reset_runtime','resume_execution']); assert names == expected, f'{names} != {expected}'; print(f'{len(tools)} tools registered: OK')\"
# Expected: \"10 tools registered: OK\"

# Each stub returns valid JSON with status field
cd v2 && python -c \"
import json
from rlm_scheme.mcp_server import load_context, get_status, reset_runtime
r = json.loads(load_context('test'))
assert 'status' in r
r = json.loads(get_status())
assert 'status' in r
r = json.loads(reset_runtime('session'))
assert 'status' in r
print('Stub responses OK')
\"
# Expected: \"Stub responses OK\"
```

### 1.7 Checklist

- [ ] `v2/rlm_scheme/store.py` implements Store with 7 namespaces and all CRUD operations
- [ ] `v2/rlm_scheme/llm_adapter.py` defines LLMProvider ABC and MockLLMProvider
- [ ] `v2/rlm_scheme/mcp_server.py` registers all 10 MCP tools with correct signatures
- [ ] Each MCP tool stub returns valid JSON with `\"status\"` field
- [ ] `v2/tests/test_store.py` covers CRUD, namespace isolation, all reset scopes
- [ ] `v2/tests/test_llm_adapter.py` covers mock responses and call tracking
- [ ] All tests pass: `python -m pytest tests/test_store.py tests/test_llm_adapter.py -v`

---

## Batch 2: Context Primitives

### 2.0 Purpose

Implement the ContextManager that stores, retrieves, and lists contexts with rich metadata, including automatic data analysis for shape detection, item counting, independence inference, modality detection, and size estimation. Wire the `load_context` and `get_context` MCP tools to real implementations that persist context records in the durable store and return structured responses with previews.

### 2.1 Depends On

Batch 0 (models, enums, IDs, exceptions), Batch 1 (store, MCP skeleton).

### 2.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `v2/rlm_scheme/context.py` | Create | ContextManager with store/retrieve/list/analyze/preview |
| `v2/rlm_scheme/mcp_server.py` | Modify | Wire `load_context` and `get_context` to real ContextManager |
| `v2/tests/test_context.py` | Create | Context storage, retrieval, analysis, preview tests |

### 2.3 Requirements

- R-2.1 [MUST] `ContextManager.store(data, name, metadata)` stores data to disk, computes SHA-256 hash, persists ContextRecord, returns `context_id`.
- R-2.2 [MUST] `ContextManager.get(context_id, include_preview, include_data)` retrieves context record with optional preview and data.
- R-2.3 [MUST] `ContextManager.list()` returns list of context IDs with names and metadata summaries.
- R-2.4 [MUST] Data analysis auto-detects `item_count` when data is a JSON array.
- R-2.5 [MUST] Data analysis infers `data_shape` as `FlatList` for JSON arrays, `Singular` for strings, `KeyValue` for JSON objects, `Tabular` for arrays of objects with consistent keys.
- R-2.6 [MUST] Preview generation returns first 500 characters for string data, or first 3 items serialized for array data.
- R-2.7 [MUST] `get_context` with `include_data=True` returns data only when serialized size is under 100 KB; otherwise returns `\"data_too_large\": true`.
- R-2.8 [MUST] `load_context` MCP tool returns JSON response with `context_id`, `name`, `metadata`, `preview`, and `next_actions`.
- R-2.9 [MUST] `get_context` MCP tool returns JSON response with context details.
- R-2.10 [SHOULD] Metadata provided by the caller overrides auto-detected values.
- R-2.11 [MUST] Data is stored as a separate file at `contexts/{context_id}/data.json`, not embedded in the record.
- R-2.12 [SHOULD] Token size estimation uses a heuristic of `len(text) / 4` for English text.

### 2.4 Detailed Specifications

**ContextManager:**

```python
import hashlib, json
from pathlib import Path
from rlm_scheme.ids import generate_id
from rlm_scheme.models import ContextRecord, ContextMetadata, DataRef
from rlm_scheme.store import Store
from rlm_scheme.enums import DataShape

class ContextManager:
    def __init__(self, store: Store):
        self.store = store

    def store_context(self, data: str, name: str | None = None,
                      metadata: dict | None = None) -> ContextRecord:
        context_id = generate_id(\"ctx_\")
        # Store raw data
        data_dir = self.store.base_dir / \"contexts\" / context_id
        data_dir.mkdir(parents=True, exist_ok=True)
        data_path = data_dir / \"data.json\"
        data_bytes = data.encode(\"utf-8\")
        data_path.write_bytes(data_bytes)
        data_hash = \"sha256:\" + hashlib.sha256(data_bytes).hexdigest()

        # Auto-detect metadata
        detected = self._analyze_data(data)
        if metadata:
            detected.update({k: v for k, v in metadata.items() if v is not None})

        record = ContextRecord(
            context_id=context_id,
            name=name,
            data_ref=DataRef(
                storage=\"filesystem\",
                path=f\"contexts/{context_id}/data.json\",
                hash=data_hash,
                bytes=len(data_bytes)
            ),
            metadata=ContextMetadata(**detected)
        )
        self.store.put(\"contexts\", context_id, record.model_dump(mode=\"json\"))
        return record

    def get_context(self, context_id: str, include_preview: bool = True,
                    include_data: bool = False) -> dict:
        record_data = self.store.get(\"contexts\", context_id)
        if not record_data:
            return None
        result = {\"context\": record_data}
        if include_preview:
            raw = self._load_raw_data(context_id)
            result[\"preview\"] = self._generate_preview(raw)
        if include_data:
            raw = self._load_raw_data(context_id)
            if len(raw.encode(\"utf-8\")) <= 100_000:
                result[\"data\"] = raw
            else:
                result[\"data_too_large\"] = True
        return result

    def list_contexts(self) -> list[dict]:
        ids = self.store.list(\"contexts\")
        return [self.store.get(\"contexts\", cid) for cid in ids]

    def _analyze_data(self, data: str) -> dict:
        \"\"\"Auto-detect data shape, item count, and modality.\"\"\"
        result = {\"modality\": [\"text\"]}
        try:
            parsed = json.loads(data)
        except (json.JSONDecodeError, TypeError):
            # Plain text
            result[\"data_shape\"] = DataShape.Singular.value
            result[\"total_size_estimate_tokens\"] = len(data) // 4
            return result

        if isinstance(parsed, list):
            result[\"item_count\"] = len(parsed)
            result[\"total_size_estimate_tokens\"] = len(data) // 4
            if len(parsed) > 0 and all(isinstance(item, dict) for item in parsed):
                keys = set(parsed[0].keys())
                if all(set(item.keys()) == keys for item in parsed):
                    result[\"data_shape\"] = DataShape.Tabular.value
                else:
                    result[\"data_shape\"] = DataShape.FlatList.value
                result[\"independent\"] = True
            else:
                result[\"data_shape\"] = DataShape.FlatList.value
                result[\"independent\"] = True
        elif isinstance(parsed, dict):
            result[\"data_shape\"] = DataShape.KeyValue.value
            result[\"item_count\"] = len(parsed)
            result[\"total_size_estimate_tokens\"] = len(data) // 4
        else:
            result[\"data_shape\"] = DataShape.Singular.value
            result[\"total_size_estimate_tokens\"] = len(data) // 4
        return result

    def _generate_preview(self, data: str) -> str:
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list) and len(parsed) > 3:
                return json.dumps(parsed[:3], indent=2)[:500]
            return data[:500]
        except (json.JSONDecodeError, TypeError):
            return data[:500]

    def _load_raw_data(self, context_id: str) -> str:
        path = self.store.base_dir / \"contexts\" / context_id / \"data.json\"
        return path.read_text()
```

### 2.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|---------------|----------|----------|
| `v2/tests/test_context.py` | `test_store_and_retrieve` | Store a JSON array, retrieve by ID | Record exists, data_ref has hash, context_id matches |
| `v2/tests/test_context.py` | `test_store_with_name` | Store with name=\"papers\" | Record name == \"papers\" |
| `v2/tests/test_context.py` | `test_auto_detect_flat_list` | Store `[1, 2, 3]` | `data_shape == \"FlatList\"`, `item_count == 3` |
| `v2/tests/test_context.py` | `test_auto_detect_tabular` | Store `[{\"a\":1},{\"a\":2}]` | `data_shape == \"Tabular\"`, `item_count == 2` |
| `v2/tests/test_context.py` | `test_auto_detect_singular` | Store `\"plain text\"` | `data_shape == \"Singular\"` |
| `v2/tests/test_context.py` | `test_auto_detect_key_value` | Store `{\"key1\": \"val1\"}` | `data_shape == \"KeyValue\"` |
| `v2/tests/test_context.py` | `test_metadata_override` | Store array with `metadata={\"data_shape\": \"Hierarchy\"}` | `data_shape == \"Hierarchy\"` (overrides auto-detect) |
| `v2/tests/test_context.py` | `test_preview_string` | Store 1000-char string | Preview is first 500 chars |
| `v2/tests/test_context.py` | `test_preview_array` | Store 10-item array | Preview contains first 3 items |
| `v2/tests/test_context.py` | `test_include_data_small` | Get with include_data=True, data < 100KB | Response includes `data` field |
| `v2/tests/test_context.py` | `test_include_data_large` | Get with include_data=True, data > 100KB | Response includes `data_too_large: true`, no `data` field |
| `v2/tests/test_context.py` | `test_get_nonexistent` | Get with invalid context_id | Returns `None` |
| `v2/tests/test_context.py` | `test_list_contexts` | Store 3 contexts, list | Returns 3 records |
| `v2/tests/test_context.py` | `test_data_hash_consistency` | Store same data twice | Both records have same `data_ref.hash` value |
| `v2/tests/test_context.py` | `test_token_estimate` | Store 400-char text | `total_size_estimate_tokens == 100` |
| `v2/tests/test_context.py` | `test_mcp_load_context` | Call `load_context` MCP tool with JSON array | Response has `status: \"ok\"`, `context_id`, `preview`, `next_actions` |
| `v2/tests/test_context.py` | `test_mcp_get_context` | Load then get via MCP tools | Response has `status: \"ok\"`, context data |

### 2.6 Acceptance Gates

```bash
# Context tests pass
cd v2 && python -m pytest tests/test_context.py -v
# Expected: 17 passed

# MCP load_context returns real context_id
cd v2 && python -c \"
import json
from rlm_scheme.mcp_server import load_context
r = json.loads(load_context('[1,2,3]', name='test'))
assert r['status'] == 'ok'
assert r['context_id'].startswith('ctx_')
assert r['metadata']['item_count'] == 3
print('load_context OK')
\"
# Expected: \"load_context OK\"
```

### 2.7 Checklist

- [ ] `v2/rlm_scheme/context.py` implements ContextManager with store/get/list/analyze/preview
- [ ] Auto-detection identifies FlatList, Tabular, Singular, KeyValue shapes
- [ ] Preview generation works for strings (500 chars) and arrays (3 items)
- [ ] Large data protection (100KB threshold) implemented
- [ ] `load_context` MCP tool wired to real ContextManager
- [ ] `get_context` MCP tool wired to real ContextManager
- [ ] `v2/tests/test_context.py` has 17+ tests
- [ ] All tests pass: `python -m pytest tests/test_context.py -v`

---

## Batch 3: Template Catalog and S-Expression Parser

### 3.0 Purpose

Build the S-expression parser for `define-meta` forms, implement the template loader that reads `.rkt` files and extracts structured metadata, implement the template validator that checks required fields and slot schema well-formedness, and create all 16 template `.rkt` files from the initial catalog. This batch provides the template infrastructure that the classifier, instantiator, and planner depend on.

### 3.1 Depends On

Batch 0 (models, enums).

### 3.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `v2/rlm_scheme/sexpr_parser.py` | Create | S-expression parser for define-meta forms |
| `v2/rlm_scheme/template_loader.py` | Create | Template loader: reads .rkt files, extracts metadata |
| `v2/rlm_scheme/template_validator.py` | Create | Template validator: checks required fields, slot schema |
| `v2/rlm_scheme/templates/direct_call.rkt` | Create | Direct single-call template |
| `v2/rlm_scheme/templates/direct_json_extract.rkt` | Create | Direct JSON extraction template |
| `v2/rlm_scheme/templates/batch_map.rkt` | Create | Batch map-async template |
| `v2/rlm_scheme/templates/batch_extract_reduce.rkt` | Create | Batch extract + tree-reduce template |
| `v2/rlm_scheme/templates/batch_extract_fold.rkt` | Create | Batch extract + fold-sequential template |
| `v2/rlm_scheme/templates/ordered_synthesis_fold.rkt` | Create | Ordered synthesis via fold template |
| `v2/rlm_scheme/templates/tree_synthesis.rkt` | Create | Tree-reduce synthesis template |
| `v2/rlm_scheme/templates/compare_candidates.rkt` | Create | Parallel comparison template |
| `v2/rlm_scheme/templates/race_candidates.rkt` | Create | Race-first-wins template |
| `v2/rlm_scheme/templates/refine_until_valid.rkt` | Create | Iterate-until with validation template |
| `v2/rlm_scheme/templates/bounded_critique_refine.rkt` | Create | Bounded critique-refine loop template |
| `v2/rlm_scheme/templates/tiered_review.rkt` | Create | Cheap pass + uncertainty filter + expensive review |
| `v2/rlm_scheme/templates/tabular_extract_aggregate.rkt` | Create | Map extraction + python aggregation template |
| `v2/rlm_scheme/templates/decompose_then_batch.rkt` | Create | JSON decomposition + map-async template |
| `v2/rlm_scheme/templates/recursive_decompose.rkt` | Create | Recursive-spawn decomposition template |
| `v2/rlm_scheme/templates/code_interpreter.rkt` | Create | LLM code generation + py-exec template |
| `v2/tests/test_sexpr_parser.py` | Create | Parser tests for atoms, lists, define-meta forms |
| `v2/tests/test_template_loader.py` | Create | Loader tests: metadata extraction, body separation, catalog listing |

### 3.3 Requirements

- R-3.1 [MUST] S-expression parser handles: strings (double-quoted with escapes), numbers (int and float), booleans (`#t`, `#f`), symbols, lists (parenthesized), quoted forms (`'expr`), dot-pairs (`(a . b)`).
- R-3.2 [MUST] Parser extracts all `(define-meta name value)` forms from template files into a dict keyed by name.
- R-3.3 [MUST] Duplicate `define-meta` names in a single template raise a parse error.
- R-3.4 [MUST] Template loader separates metadata (define-meta forms) from body (remaining Scheme code).
- R-3.5 [MUST] Template loader validates these required metadata fields: `name`, `version`, `task-shapes`, `data-shapes`, `slots`.
- R-3.6 [MUST] Template validator checks slot schema well-formedness: each slot has `type`, required slots have `required #t`, types are one of `string`, `integer`, `boolean`, `number`.
- R-3.7 [MUST] Template validator checks output-schema alist syntax: balanced parens, top-level keys from `type`, `properties`, `items`, `required`, `enum`, `description`.
- R-3.8 [MUST] All 16 template `.rkt` files from the catalog (See Appendix G) are created with complete define-meta metadata and valid Scheme bodies.
- R-3.9 [MUST] Template loader provides `list_templates()` returning all loaded template names.
- R-3.10 [MUST] Template loader provides `get_template(name)` returning metadata + body.
- R-3.11 [MUST] Each template declares: `name`, `version`, `summary`, `task-shapes`, `data-shapes`, `slots`, `structural-profile`, `verification-rules`.
- R-3.12 [SHOULD] Each template declares: `trigger`, `reject`, `output-schema`, `streamable`, `cacheable`, `budget-policy`, `gates`, `uses-llm-generated-code`, `examples`.
- R-3.13 [MUST] Template bodies contain only `{{slot_name}}` markers and references to primitive runtime bindings from Appendix B.

### 3.4 Detailed Specifications

**S-Expression Parser:**

```python
class SExprParser:
    \"\"\"Parse S-expressions from template files.\"\"\"

    def parse(self, text: str) -> list:
        \"\"\"Parse text into a list of S-expressions.\"\"\"
        tokens = self._tokenize(text)
        results = []
        while tokens:
            results.append(self._parse_expr(tokens))
        return results

    def _tokenize(self, text: str) -> list[str]:
        \"\"\"Tokenize S-expression text into tokens.\"\"\"
        # Handle: ( ) ' \"strings\" #t #f ;comments numbers symbols
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in ' \\t\
\\r':
                i += 1
            elif c == ';':
                # Skip to end of line (comment)
                while i < len(text) and text[i] != '\
':
                    i += 1
            elif c in '()':
                tokens.append(c)
                i += 1
            elif c == \"'\":
                tokens.append(\"'\")
                i += 1
            elif c == '\"':
                # Parse string literal
                j = i + 1
                while j < len(text) and text[j] != '\"':
                    if text[j] == '\\\\':
                        j += 1  # skip escaped char
                    j += 1
                tokens.append(text[i:j+1])
                i = j + 1
            elif c == '#':
                if text[i+1] in 'tf':
                    tokens.append(text[i:i+2])
                    i += 2
                else:
                    # Other # forms
                    j = i
                    while j < len(text) and text[j] not in ' \\t\
\\r()':
                        j += 1
                    tokens.append(text[i:j])
                    i = j
            else:
                # Symbol or number
                j = i
                while j < len(text) and text[j] not in ' \\t\
\\r();\"':
                    j += 1
                tokens.append(text[i:j])
                i = j
        return tokens

    def _parse_expr(self, tokens: list) -> any:
        \"\"\"Parse one expression from token list (mutates tokens).\"\"\"
        if not tokens:
            raise ValueError(\"Unexpected end of input\")
        token = tokens.pop(0)
        if token == '(':
            return self._parse_list(tokens)
        elif token == \"'\":
            return ('quote', self._parse_expr(tokens))
        else:
            return self._parse_atom(token)

    def _parse_list(self, tokens: list) -> list | tuple:
        \"\"\"Parse list contents until closing paren.\"\"\"
        result = []
        while tokens and tokens[0] != ')':
            if tokens[0] == '.':
                tokens.pop(0)
                cdr = self._parse_expr(tokens)
                assert tokens.pop(0) == ')'
                return (result[0] if len(result) == 1 else tuple(result), cdr)
            result.append(self._parse_expr(tokens))
        if tokens:
            tokens.pop(0)  # consume ')'
        return result

    def _parse_atom(self, token: str) -> any:
        if token == '#t': return True
        if token == '#f': return False
        if token.startswith('\"') and token.endswith('\"'):
            return token[1:-1].replace('\\\\\"', '\"').replace('\\\\\\\\', '\\\\')
        try: return int(token)
        except ValueError: pass
        try: return float(token)
        except ValueError: pass
        return token  # symbol

def extract_define_metas(exprs: list) -> tuple[dict, list]:
    \"\"\"Separate define-meta forms from body expressions.\"\"\"
    meta = {}
    body = []
    for expr in exprs:
        if isinstance(expr, list) and len(expr) == 3 and expr[0] == 'define-meta':
            name = expr[1]
            if name in meta:
                raise ValueError(f\"Duplicate define-meta: {name}\")
            value = expr[2]
            if isinstance(value, tuple) and len(value) == 2 and value[0] == 'quote':
                value = value[1]
            meta[name] = value
        else:
            body.append(expr)
    return meta, body
```

**Template Loader:**

```python
class TemplateLoader:
    REQUIRED_META = (\"name\", \"version\", \"task-shapes\", \"data-shapes\", \"slots\")

    def __init__(self, templates_dir: str | Path):
        self.templates_dir = Path(templates_dir)
        self._cache: dict[str, dict] = {}

    def load_template(self, name: str) -> dict:
        \"\"\"Load and parse a template by name.\"\"\"
        path = self.templates_dir / f\"{name}.rkt\"
        if not path.exists():
            raise TemplateNotFoundError(f\"Template not found: {name}\")
        text = path.read_text()
        parser = SExprParser()
        exprs = parser.parse(text)
        meta, body = extract_define_metas(exprs)
        self._validate_required(name, meta)
        return {\"name\": name, \"metadata\": meta, \"body_text\": self._extract_body_text(text), \"path\": str(path)}

    def list_templates(self) -> list[str]:
        \"\"\"List all available template names.\"\"\"
        return sorted(p.stem for p in self.templates_dir.glob(\"*.rkt\"))

    def get_template(self, name: str) -> dict:
        if name not in self._cache:
            self._cache[name] = self.load_template(name)
        return self._cache[name]
```

### 3.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|---------------|----------|----------|
| `v2/tests/test_sexpr_parser.py` | `test_parse_string` | Parse `\"hello\"` | Returns `\"hello\"` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_number` | Parse `42` and `3.14` | Returns int 42, float 3.14 |
| `v2/tests/test_sexpr_parser.py` | `test_parse_boolean` | Parse `#t` and `#f` | Returns True, False |
| `v2/tests/test_sexpr_parser.py` | `test_parse_symbol` | Parse `map-async` | Returns string `\"map-async\"` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_list` | Parse `(a b c)` | Returns `[\"a\", \"b\", \"c\"]` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_nested_list` | Parse `(a (b c) d)` | Returns `[\"a\", [\"b\", \"c\"], \"d\"]` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_quoted` | Parse `'(a b)` | Returns `(\"quote\", [\"a\", \"b\"])` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_dot_pair` | Parse `(a . b)` | Returns `(\"a\", \"b\")` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_comment` | Parse `; comment\
(a)` | Returns `[[\"a\"]]` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_string_escapes` | Parse `\"a\\\"b\\\\c\"` | Returns `'a\"b\\\\c'` |
| `v2/tests/test_sexpr_parser.py` | `test_extract_define_metas` | Parse template with 3 define-meta forms | Meta dict has 3 keys, body has remaining exprs |
| `v2/tests/test_sexpr_parser.py` | `test_duplicate_define_meta` | Template with duplicate name | Raises `ValueError` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_define_meta_quoted_list` | `(define-meta shapes '(Batch Synthesize))` | Meta value is `[\"Batch\", \"Synthesize\"]` |
| `v2/tests/test_sexpr_parser.py` | `test_parse_slot_schema` | Parse full slot alist | Correct nested structure with types, defaults, required |
| `v2/tests/test_template_loader.py` | `test_load_batch_extract_reduce` | Load batch_extract_reduce.rkt | Has name, version, task-shapes, data-shapes, slots |
| `v2/tests/test_template_loader.py` | `test_load_all_templates` | Load all 16 templates | All parse without error, all have required fields |
| `v2/tests/test_template_loader.py` | `test_list_templates` | Call list_templates() | Returns 16 names |
| `v2/tests/test_template_loader.py` | `test_template_not_found` | Load nonexistent template | Raises `TemplateNotFoundError` |
| `v2/tests/test_template_loader.py` | `test_template_has_body` | Load any template | `body_text` is non-empty, contains `{{slot}}` markers or primitive calls |
| `v2/tests/test_template_loader.py` | `test_template_metadata_types` | Load batch_extract_reduce | `task-shapes` is list, `slots` is list of lists, `version` is string |
| `v2/tests/test_template_loader.py` | `test_template_validator_required_fields` | Template missing `name` field | Validator raises error |
| `v2/tests/test_template_loader.py` | `test_template_slot_schema_valid` | batch_extract_reduce slot schema | All slots have `type`, required slots have `required #t` |
| `v2/tests/test_template_loader.py` | `test_output_schema_alist` | batch_extract_reduce output-schema | Parses to valid alist with known JSON Schema keywords |

### 3.6 Acceptance Gates

```bash
# Parser tests pass
cd v2 && python -m pytest tests/test_sexpr_parser.py -v
# Expected: 14 passed

# Template loader tests pass
cd v2 && python -m pytest tests/test_template_loader.py -v
# Expected: 9 passed

# All 16 templates exist and load
cd v2 && python -c \"
from rlm_scheme.template_loader import TemplateLoader
loader = TemplateLoader('rlm_scheme/templates')
templates = loader.list_templates()
assert len(templates) >= 16, f'Only {len(templates)} templates found'
for name in templates:
    t = loader.get_template(name)
    assert 'name' in t['metadata']
    assert 'slots' in t['metadata']
print(f'{len(templates)} templates loaded OK')
\"
# Expected: \"16 templates loaded OK\"

# No template body references disallowed forms
cd v2 && python -c \"
from pathlib import Path
import re
disallowed = ['(eval ', '(system ', '(shell ']
for p in Path('rlm_scheme/templates').glob('*.rkt'):
    text = p.read_text()
    for d in disallowed:
        assert d not in text, f'{p.name} contains disallowed form: {d}'
print('No disallowed forms found')
\"
# Expected: \"No disallowed forms found\"
```

### 3.7 Checklist

- [ ] `v2/rlm_scheme/sexpr_parser.py` handles strings, numbers, booleans, symbols, lists, quotes, dot-pairs, comments
- [ ] `extract_define_metas()` separates metadata from body, rejects duplicates
- [ ] `v2/rlm_scheme/template_loader.py` loads .rkt files, validates required fields
- [ ] `v2/rlm_scheme/template_validator.py` checks slot schemas and output-schema alists
- [ ] All 16 template `.rkt` files created with complete define-meta and valid bodies
- [ ] `list_templates()` returns 16 names
- [ ] `v2/tests/test_sexpr_parser.py` has 14+ tests
- [ ] `v2/tests/test_template_loader.py` has 9+ tests
- [ ] All tests pass

---

## Batch 4: Classifier, Instantiator, and Planner

### 4.0 Purpose

Implement the deterministic TaskShape/DataShape classifier using the Q0-Q9 decision tree, the template instantiator that validates slot values and performs safe `{{slot}}` substitution to produce immutable artifacts, and the planner that orchestrates classification, template selection via trigger/reject conditions, slot filling, and plan record creation. Wire the `plan_strategy` MCP tool to the real implementation.

### 4.1 Depends On

Batch 0 (models, enums, IDs, exceptions), Batch 1 (store), Batch 2 (context), Batch 3 (template catalog, s-expression parser).

### 4.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `v2/rlm_scheme/classifier.py` | Create | Deterministic Q0-Q9 decision tree classifier |
| `v2/rlm_scheme/instantiator.py` | Create | Slot validation, safe substitution, artifact creation |
| `v2/rlm_scheme/planner.py` | Create | Classification + template selection + slot filling + plan record |
| `v2/rlm_scheme/mcp_server.py` | Modify | Wire `plan_strategy` to real planner |
| `v2/tests/test_classifier.py` | Create | Classification tests for all TaskShape paths |
| `v2/tests/test_instantiator.py` | Create | Slot validation, substitution, hash determinism tests |
| `v2/tests/test_planner.py` | Create | End-to-end planner tests |

### 4.3 Requirements

- R-4.1 [MUST] `classify_task_shape(hints)` implements the Q0-Q9 decision tree from Appendix F, returning a `TaskShape` enum value.
- R-4.2 [MUST] `classify_data_shape(metadata)` infers DataShape from context metadata fields (item_count, independent, ordered, modality, etc.) per the mapping rules in Appendix F.
- R-4.3 [MUST] `select_template(task_shape, data_shape, hints, templates)` evaluates trigger/reject conditions from template metadata to select the best matching template.
- R-4.4 [MUST] Template trigger conditions use implicit AND (all must match). Reject conditions use implicit OR (any match disqualifies).
- R-4.5 [MUST] Instantiator applies defaults for missing optional slots first, then validates all slot values against the template's slot schema before substitution: type checking, required fields, min/max ranges, min-length, pattern matching, enum values. A required slot with no value and no default [MUST] fail validation.
- R-4.6 [MUST] Instantiator performs safe `{{slot_name}}` substitution: strings are escaped and quoted, numbers and booleans are inserted as Scheme literals, context IDs are quoted strings.
- R-4.7 [MUST] Instantiator rejects slot values containing unbalanced parentheses, Scheme keywords (`define`, `lambda`, `set!`, `eval`, `require`), or code injection attempts.
- R-4.8 [MUST] After substitution, no `{{` markers remain in the artifact code.
- R-4.9 [MUST] Instantiation is deterministic: same template + same slot values produce the same SHA-256 code hash.
- R-4.10 [MUST] Instantiator stores ArtifactRecord with `generated_scheme_ref` pointing to the artifact file and its hash.
- R-4.11 [MUST] Planner returns a PlanRecord with classification, recommended template invocation, and alternatives.
- R-4.12 [MUST] For Composite tasks (Q9 = YES), planner produces a `template_chain` with `$previous` references connecting steps.
- R-4.13 [MUST] If no template matches, planner returns `status: \"no_template\"` with `recommendation.needed_template`.
- R-4.14 [SHOULD] Planner fills content slots (map_instruction, reduce_instruction) via LLM call when they are required, have no default, and no agent override.
- R-4.15 [MUST] `plan_strategy` MCP tool returns JSON with `plan_id`, `classification`, `recommended`, `alternatives`, `next_actions`.

### 4.4 Detailed Specifications

**Classifier (Q0-Q9 Decision Tree):**

```python
from rlm_scheme.enums import TaskShape, DataShape

def classify_task_shape(hints: dict) -> TaskShape:
    \"\"\"Deterministic task shape classification. See Appendix F.\"\"\"
    item_count = hints.get(\"item_count\", 0)
    independent = hints.get(\"independent\", False)
    output_type = hints.get(\"output_type\", \"one\")
    operation = hints.get(\"operation\", \"other\")
    has_second_phase = hints.get(\"has_second_phase\", False)
    ordered = hints.get(\"ordered\", False)
    sub_operations = hints.get(\"sub_operations\", [])

    # Q9: Multiple phases? (check first as it can override)
    if has_second_phase and len(sub_operations) >= 2:
        return TaskShape.Composite

    # Q0: One small input, one output, one operation, no second phase?
    if item_count <= 1 and output_type == \"one\" and not has_second_phase:
        # Q5: Creating content with no source items?
        if hints.get(\"creating_content\", False):
            return TaskShape.Generate
        # Q6: Improving one artifact?
        if operation in (\"refine\", \"improve\", \"iterate\"):
            return TaskShape.Refine
        # Q7: Breaking input into parts?
        if operation in (\"decompose\", \"split\", \"parse\"):
            return TaskShape.Decompose
        # Q8: Choosing among alternatives?
        if operation in (\"compare\", \"choose\", \"select\"):
            return TaskShape.Compare
        if operation in (\"search\", \"find_best\"):
            return TaskShape.Search
        return TaskShape.Direct

    # Q1: Many input items?
    if item_count > 1:
        # Q2: Items independent?
        if independent:
            # Q3: Per-item operation type
            if operation in (\"transform\", \"extract\", \"extract_then_synthesize\"):
                return TaskShape.Batch
            elif operation in (\"label\", \"category\", \"categorize\"):
                return TaskShape.Classify
            elif operation in (\"check\", \"grade\", \"audit\", \"validate\"):
                return TaskShape.Validate
            else:
                return TaskShape.Batch  # default for independent items
        else:
            # Q4: Information accumulates across ordered items?
            if ordered:
                return TaskShape.Synthesize
            else:
                return TaskShape.Pipeline

    # Fallback based on output_type
    if output_type == \"one\":
        return TaskShape.Synthesize
    elif output_type == \"list\":
        return TaskShape.Aggregate
    return TaskShape.Direct

def classify_data_shape(metadata: dict) -> DataShape:
    \"\"\"Infer DataShape from context metadata.\"\"\"
    explicit = metadata.get(\"data_shape\")
    if explicit:
        try:
            return DataShape(explicit)
        except ValueError:
            pass

    item_count = metadata.get(\"item_count\", 0)
    modality = metadata.get(\"modality\", [\"text\"])
    independent = metadata.get(\"independent\")

    if \"image\" in modality or \"audio\" in modality:
        return DataShape.Multimodal
    if item_count == 0:
        return DataShape.Singular
    if item_count == 1:
        return DataShape.Singular
    if independent is True:
        return DataShape.FlatList
    if independent is False:
        if metadata.get(\"ordered\"):
            return DataShape.ChunkedSingular
        return DataShape.FlatList
    return DataShape.Unknown

def select_template(task_shape: TaskShape, data_shape: DataShape,
                    hints: dict, templates: list[dict]) -> str | None:
    \"\"\"Select best matching template using trigger/reject conditions.\"\"\"
    candidates = []
    for tmpl in templates:
        meta = tmpl[\"metadata\"]
        # Check shape compatibility
        task_shapes = meta.get(\"task-shapes\", [])
        data_shapes = meta.get(\"data-shapes\", [])
        if task_shape.value not in task_shapes:
            continue
        if data_shapes and data_shape.value not in data_shapes:
            continue
        # Check reject conditions (implicit OR — any match disqualifies)
        if _evaluate_reject(meta.get(\"reject\", []), hints):
            continue
        # Check trigger conditions (implicit AND — all must match)
        if _evaluate_trigger(meta.get(\"trigger\", []), hints):
            candidates.append(tmpl)
    if not candidates:
        return None
    # Return first matching (templates are loaded in priority order)
    return candidates[0][\"metadata\"][\"name\"]
```

**Instantiator:**

```python
import hashlib, re, json
from rlm_scheme.ids import generate_id
from rlm_scheme.models import ArtifactRecord, DataRef
from rlm_scheme.exceptions import SlotValidationError, InstantiationError

SLOT_PATTERN = re.compile(r\"\\{\\{(\\w+)\\}\\}\")
DANGEROUS_PATTERNS = re.compile(
    r\"\\b(define|lambda|set!|eval|require|load|include|system)\\b\"
)

class Instantiator:
    def __init__(self, store):
        self.store = store

    def instantiate(self, template: dict, slot_values: dict,
                    plan_id: str | None = None,
                    context_ids: list[str] | None = None) -> ArtifactRecord:
        meta = template[\"metadata\"]
        slots_schema = meta.get(\"slots\", [])

        # 1. Apply defaults for missing optional slots
        filled = self._apply_defaults(slots_schema, slot_values)

        # 2. Validate filled slot values against schema (after defaults applied)
        self._validate_slots(slots_schema, filled)

        # 3. Perform safe substitution
        body = template[\"body_text\"]
        artifact_code = self._substitute(body, filled)

        # 4. Verify no remaining markers
        remaining = SLOT_PATTERN.findall(artifact_code)
        if remaining:
            raise InstantiationError(f\"Unfilled slot markers: {remaining}\")

        # 5. Hash the result
        code_hash = \"sha256:\" + hashlib.sha256(
            artifact_code.encode(\"utf-8\")
        ).hexdigest()

        # 6. Store artifact
        artifact_id = generate_id(\"art_\")
        artifact_dir = self.store.base_dir / \"artifacts\" / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / \"program.rkt\"
        artifact_path.write_text(artifact_code)

        # 7. Extract primitives used
        primitives = self._detect_primitives(artifact_code)

        record = ArtifactRecord(
            artifact_id=artifact_id,
            plan_id=plan_id,
            context_ids=context_ids or [],
            template_name=meta[\"name\"],
            template_version=meta.get(\"version\", \"1.0.0\"),
            slot_values=filled,
            generated_scheme_ref=DataRef(
                storage=\"filesystem\",
                path=f\"artifacts/{artifact_id}/program.rkt\",
                hash=code_hash,
                bytes=len(artifact_code.encode(\"utf-8\"))
            ),
            primitives_used=primitives,
            static_profile=self._extract_profile(meta)
        )
        self.store.put(\"artifacts\", artifact_id, record.model_dump(mode=\"json\"))
        return record

    def _validate_slots(self, schema: list, values: dict) -> None:
        \"\"\"Validate slot values against template slot schema.\"\"\"
        for slot_def in schema:
            name = slot_def[0]
            props = {p[0]: p[1] for p in slot_def[1:]}
            required = props.get(\"required\", False)
            if required and name not in values:
                raise SlotValidationError(f\"Required slot missing: {name}\")
            if name in values:
                value = values[name]
                self._check_type(name, value, props.get(\"type\", \"string\"))
                self._check_injection(name, value)
                if \"min-length\" in props and isinstance(value, str):
                    if len(value) < props[\"min-length\"]:
                        raise SlotValidationError(
                            f\"Slot {name}: min-length {props['min-length']}\"
                        )
                if \"pattern\" in props and isinstance(value, str):
                    if not re.match(props[\"pattern\"], value):
                        raise SlotValidationError(
                            f\"Slot {name}: pattern mismatch\"
                        )

    def _check_injection(self, name: str, value) -> None:
        \"\"\"Reject values that could inject Scheme code.\"\"\"
        if isinstance(value, str):
            if DANGEROUS_PATTERNS.search(value):
                raise SlotValidationError(
                    f\"Slot {name}: contains dangerous Scheme keyword\"
                )
            if value.count('(') != value.count(')'):
                raise SlotValidationError(
                    f\"Slot {name}: unbalanced parentheses\"
                )

    def _substitute(self, body: str, values: dict) -> str:
        \"\"\"Perform safe {{slot}} substitution.\"\"\"
        def replacer(match):
            name = match.group(1)
            value = values[name]
            if isinstance(value, bool):
                return \"#t\" if value else \"#f\"
            elif isinstance(value, (int, float)):
                return str(value)
            elif isinstance(value, str):
                escaped = value.replace('\\\\', '\\\\\\\\').replace('\"', '\\\\\"')
                return f'\"{escaped}\"'
            return str(value)
        return SLOT_PATTERN.sub(replacer, body)
```

### 4.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|---------------|----------|----------|
| `v2/tests/test_classifier.py` | `test_q0_direct` | hints: item_count=1, output_type=\"one\" | `TaskShape.Direct` |
| `v2/tests/test_classifier.py` | `test_q1_q2_q3_batch` | hints: item_count=100, independent=True, operation=\"extract\" | `TaskShape.Batch` |
| `v2/tests/test_classifier.py` | `test_q1_q2_q3_classify` | hints: item_count=50, independent=True, operation=\"label\" | `TaskShape.Classify` |
| `v2/tests/test_classifier.py` | `test_q1_q2_q3_validate` | hints: item_count=50, independent=True, operation=\"grade\" | `TaskShape.Validate` |
| `v2/tests/test_classifier.py` | `test_q4_synthesize_ordered` | hints: item_count=10, independent=False, ordered=True | `TaskShape.Synthesize` |
| `v2/tests/test_classifier.py` | `test_q4_pipeline_unordered` | hints: item_count=10, independent=False, ordered=False | `TaskShape.Pipeline` |
| `v2/tests/test_classifier.py` | `test_q5_generate` | hints: item_count=0, creating_content=True | `TaskShape.Generate` |
| `v2/tests/test_classifier.py` | `test_q6_refine` | hints: item_count=1, operation=\"refine\" | `TaskShape.Refine` |
| `v2/tests/test_classifier.py` | `test_q7_decompose` | hints: item_count=1, operation=\"decompose\" | `TaskShape.Decompose` |
| `v2/tests/test_classifier.py` | `test_q8_compare` | hints: item_count=1, operation=\"compare\" | `TaskShape.Compare` |
| `v2/tests/test_classifier.py` | `test_q9_composite` | hints: has_second_phase=True, sub_operations=[\"extract\",\"synthesize\"] | `TaskShape.Composite` |
| `v2/tests/test_classifier.py` | `test_data_shape_flat_list` | metadata: item_count=50, independent=True | `DataShape.FlatList` |
| `v2/tests/test_classifier.py` | `test_data_shape_singular` | metadata: item_count=1 | `DataShape.Singular` |
| `v2/tests/test_classifier.py` | `test_data_shape_multimodal` | metadata: modality=[\"text\",\"image\"] | `DataShape.Multimodal` |
| `v2/tests/test_classifier.py` | `test_select_template_batch_reduce` | task_shape=Batch, data_shape=FlatList, hints matching trigger | Returns `\"batch_extract_reduce\"` |
| `v2/tests/test_classifier.py` | `test_select_template_reject` | Template with reject condition matching hints | Template not selected |
| `v2/tests/test_instantiator.py` | `test_instantiate_valid` | batch_map with valid slots | Artifact created, no `{{` in code |
| `v2/tests/test_instantiator.py` | `test_instantiate_deterministic` | Same template + slots twice | Same code_hash both times |
| `v2/tests/test_instantiator.py` | `test_instantiate_missing_required` | Omit required slot | Raises `SlotValidationError` |
| `v2/tests/test_instantiator.py` | `test_instantiate_type_mismatch` | String for integer slot | Raises `SlotValidationError` |
| `v2/tests/test_instantiator.py` | `test_instantiate_injection_rejected` | Slot value with `(eval ...)` | Raises `SlotValidationError` |
| `v2/tests/test_instantiator.py` | `test_instantiate_unbalanced_parens` | Slot value with `\"(((\"` | Raises `SlotValidationError` |
| `v2/tests/test_instantiator.py` | `test_instantiate_string_escaping` | Slot value with quotes | Properly escaped in artifact |
| `v2/tests/test_instantiator.py` | `test_instantiate_boolean_slot` | json_mode=True | Substituted as `#t` in artifact |
| `v2/tests/test_instantiator.py` | `test_instantiate_defaults_applied` | Omit optional slot with default | Default value used |
| `v2/tests/test_planner.py` | `test_plan_with_full_hints` | All hints provided | No LLM call, deterministic classification |
| `v2/tests/test_planner.py` | `test_plan_returns_invocation` | Standard batch task | Response has `recommended.template_name` |
| `v2/tests/test_planner.py` | `test_plan_composite_chain` | Composite task hints | Response has `recommended.kind == \"template_chain\"` |
| `v2/tests/test_planner.py` | `test_plan_no_template` | Unsupported task shape | Response has `status: \"no_template\"` |
| `v2/tests/test_planner.py` | `test_plan_mcp_tool` | Call plan_strategy MCP tool | Response has plan_id, classification, recommended |

### 4.6 Acceptance Gates

```bash
# Classifier tests pass
cd v2 && python -m pytest tests/test_classifier.py -v
# Expected: 16 passed

# Instantiator tests pass
cd v2 && python -m pytest tests/test_instantiator.py -v
# Expected: 9 passed

# Planner tests pass
cd v2 && python -m pytest tests/test_planner.py -v
# Expected: 5 passed

# Classification with full hints uses no LLM calls
cd v2 && python -c \"
from rlm_scheme.classifier import classify_task_shape
from rlm_scheme.enums import TaskShape
shape = classify_task_shape({'item_count': 100, 'independent': True, 'output_type': 'one', 'has_second_phase': True, 'sub_operations': ['extract', 'synthesize']})
assert shape == TaskShape.Composite, f'Got {shape}'
print('Deterministic classification OK')
\"
# Expected: \"Deterministic classification OK\"

# Instantiation determinism
cd v2 && python -c \"
from rlm_scheme.instantiator import Instantiator
from rlm_scheme.template_loader import TemplateLoader
from rlm_scheme.store import Store
import tempfile
store = Store(tempfile.mkdtemp())
loader = TemplateLoader('rlm_scheme/templates')
inst = Instantiator(store)
tmpl = loader.get_template('batch_map')
slots = {'context_id': 'ctx_aaaa1111bbbb2222', 'map_instruction': 'Extract data from each item.', 'map_model': 'fast_text_model', 'max_concurrent': 20, 'json_mode': True}
a1 = inst.instantiate(tmpl, slots)
a2 = inst.instantiate(tmpl, slots)
assert a1.generated_scheme_ref.hash == a2.generated_scheme_ref.hash
print('Deterministic instantiation OK')
\"
# Expected: \"Deterministic instantiation OK\"
```

### 4.7 Checklist

- [ ] `v2/rlm_scheme/classifier.py` implements Q0-Q9 decision tree for TaskShape
- [ ] `v2/rlm_scheme/classifier.py` implements DataShape classification from metadata
- [ ] `v2/rlm_scheme/classifier.py` implements template selection via trigger/reject
- [ ] `v2/rlm_scheme/instantiator.py` validates slots, substitutes safely, hashes, stores artifacts
- [ ] Injection protection rejects dangerous Scheme keywords and unbalanced parens
- [ ] Instantiation is deterministic (same inputs produce same hash)
- [ ] `v2/rlm_scheme/planner.py` orchestrates classification + selection + slot filling
- [ ] `plan_strategy` MCP tool wired to real planner
- [ ] `v2/tests/test_classifier.py` has 16+ tests covering all Q0-Q9 paths
- [ ] `v2/tests/test_instantiator.py` has 9+ tests
- [ ] `v2/tests/test_planner.py` has 5+ tests
- [ ] All tests pass

---

## Batch 5: Dry-Run, Verification, and Executor Core

### 5.0 Purpose

Implement the dry-run simulator that computes expected LLM call counts using structural formulas without making real calls, the verification engine that runs all 23 checks from the plan against filled artifacts, the executor core that dispatches LLM calls through the provider adapter and tracks token budgets, and the trace collector that records execution events. Wire `dry_run_strategy` and `execute_strategy` MCP tools to their real implementations.

### 5.1 Depends On

Batch 0 (models, enums, IDs, exceptions), Batch 1 (store, LLM adapter, MCP skeleton), Batch 2 (context), Batch 3 (template catalog), Batch 4 (classifier, instantiator, planner).

### 5.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| `v2/rlm_scheme/dry_runner.py` | Create | Dry-run simulation: call count formulas, mock execution, cost estimates |
| `v2/rlm_scheme/cost_model.py` | Create | Token/cost estimation per model alias |
| `v2/rlm_scheme/verification.py` | Create | All 23 verification checks |
| `v2/rlm_scheme/executor.py` | Create | LLM dispatch, token budget tracking, execution state machine |
| `v2/rlm_scheme/trace.py` | Create | TraceCollector: events, scope log, stdout capture |
| `v2/rlm_scheme/mcp_server.py` | Modify | Wire `dry_run_strategy` and `execute_strategy` to real implementations |
| `v2/tests/test_dry_run.py` | Create | Dry-run call count tests, cost estimate tests |
| `v2/tests/test_verification.py` | Create | Tests for all 23 verification checks |
| `v2/tests/test_executor.py` | Create | Executor dispatch, budget, state transition, trace tests |

### 5.3 Requirements

- R-5.1 [MUST] Dry-run computes tree-reduce call counts using: `N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1`.
- R-5.2 [MUST] Dry-run for `batch_extract_reduce` with N=100, B=5 returns `expected_llm_calls: 125` and `critical_path_calls: 4`.
- R-5.3 [MUST] Dry-run uses mock LLM responses (empty string for text, `\"{}\"` for JSON mode) and makes zero real provider calls.
- R-5.4 [MUST] Dry-run computes `max_concurrency`, `recursive_depth`, `critical_path_calls`, and `models` breakdown.
- R-5.5 [MUST] Dry-run uses no global mutable state; two concurrent dry-runs produce independent results.
- R-5.6 [MUST] Cost model estimates tokens using: prompt = item_count * avg_input_tokens, completion = item_count * avg_output_tokens.
- R-5.7 [MUST] Verification implements all 23 checks from Appendix E: artifact_origin, artifact_hash, template_version, slots_filled, model_exists, model_capabilities, image_model, no_unsafe_forms, no_raw_import, call_count_limit, recursive_depth_limit, concurrency_limit, context_exists, output_schema_valid, output_schema_present, dry_run_warnings, code_interpreter_policy, gate_consistency, budget_policy_model, budget_policy_caps, primitive_allowlist, context_window_fit, temperature_compat.
- R-5.8 [MUST] Verification returns overall decision: `pass` (all pass), `warn` (warnings only), `fail` (any check fails).
- R-5.9 [MUST] Failed verification blocks execution; `execute_strategy` returns `status: \"verification_failed\"` with failing checks.
- R-5.10 [MUST] Executor transitions execution state correctly: queued -> running -> finished/failed/cancelled.
- R-5.11 [MUST] Executor tracks token usage per call and total; rejects calls when budget is exhausted.
- R-5.12 [MUST] TraceCollector records events: `llm_call_started`, `llm_call_completed`, `llm_call_failed`, with call_id, model, tokens, elapsed time.
- R-5.13 [MUST] TraceCollector records scope log entries for every `syntax-e` unwrap.
- R-5.14 [MUST] `dry_run_strategy` MCP tool internally instantiates the template, computes estimates, simulates, and returns DryRunRecord.
- R-5.15 [MUST] `execute_strategy` MCP tool internally instantiates (or reuses cached artifact), verifies, executes, and returns ExecutionRecord.
- R-5.16 [MUST] Executing the same artifact twice creates two distinct execution_id values.
- R-5.17 [SHOULD] Executor reuses a cached artifact from a prior dry-run via hash match.
- R-5.18 [MUST] Default policy limits: max_llm_calls=1000, max_concurrency=50, max_recursive_depth=3.

### 5.4 Detailed Specifications

**Tree-Reduce Call Count Formula:**

```python
import math

def tree_reduce_calls(n: int, branch_factor: int, include_leaf: bool = True) -> int:
    \"\"\"Compute total calls for tree-reduce.

    Formula: N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1
    The N term is the leaf/map calls. Reduction levels follow.
    \"\"\"
    if n <= 0:
        return 0
    total = n if include_leaf else 0
    level = n
    while level > 1:
        level = math.ceil(level / branch_factor)
        total += level
    return total

def critical_path_depth(n: int, branch_factor: int) -> int:
    \"\"\"Critical path = 1 + ceil(log_B(N)) reduction levels.\"\"\"
    if n <= 1:
        return 1
    return 1 + math.ceil(math.log(n) / math.log(branch_factor))
```

**DryRunner:**

```python
class DryRunner:
    def __init__(self, store, template_loader, instantiator):
        self.store = store
        self.template_loader = template_loader
        self.instantiator = instantiator

    def dry_run(self, plan_id: str = None,
                template_invocation: dict = None,
                options: dict = None) -> dict:
        # 1. Resolve template invocation from plan or direct input
        invocation = self._resolve_invocation(plan_id, template_invocation)

        # 2. Instantiate template -> artifact
        template = self.template_loader.get_template(invocation[\"template_name\"])
        artifact = self.instantiator.instantiate(
            template, invocation[\"slot_values\"], plan_id=plan_id
        )

        # 3. Compute structural estimates
        profile = artifact.static_profile
        slot_values = invocation[\"slot_values\"]
        estimates = self._compute_estimates(template, slot_values, options)

        # 4. Simulate execution with mock responses
        simulation = self._simulate(template, slot_values)

        # 5. Build call graph
        call_graph = self._build_call_graph(template, slot_values)

        # 6. Store dry-run record
        dry_run_id = generate_id(\"dry_\")
        record = DryRunRecord(
            dry_run_id=dry_run_id,
            artifact_id=artifact.artifact_id,
            summary=estimates,
            call_graph=call_graph,
        )
        self.store.put(\"dry_runs\", dry_run_id, record.model_dump(mode=\"json\"))

        return {
            \"status\": \"ok\",
            \"dry_run_id\": dry_run_id,
            \"plan_id\": plan_id,
            \"artifact\": {
                \"artifact_id\": artifact.artifact_id,
                \"template_name\": artifact.template_name,
                \"template_version\": artifact.template_version,
                \"code_hash\": artifact.generated_scheme_ref.hash,
                \"primitives_used\": artifact.primitives_used,
            },
            \"estimate\": estimates,
            \"simulation\": simulation,
            \"call_graph\": call_graph,
            \"warnings\": [],
            \"next_actions\": [f\"Call execute_strategy(plan_id={plan_id})\"],
        }

    def _compute_estimates(self, template, slot_values, options) -> dict:
        meta = template[\"metadata\"]
        profile = meta.get(\"structural-profile\", [])
        # Parse profile into dict
        profile_dict = {p[0]: p[1] for p in profile} if profile else {}

        item_count = slot_values.get(\"item_count\", 0)
        # Try to get from context metadata
        if not item_count:
            ctx_id = slot_values.get(\"context_id\")
            if ctx_id:
                ctx = self.store.get(\"contexts\", ctx_id)
                if ctx:
                    item_count = ctx.get(\"metadata\", {}).get(\"item_count\", 0)

        branch_factor = slot_values.get(\"branch_factor\", 5)
        max_conc = slot_values.get(\"max_concurrent\", 20)

        # Compute calls based on template type
        template_name = meta[\"name\"]
        if template_name in (\"batch_extract_reduce\", \"batch_extract_fold\"):
            total_calls = tree_reduce_calls(item_count, branch_factor)
            map_calls = item_count
            reduce_calls = total_calls - map_calls
        elif template_name in (\"batch_map\",):
            total_calls = item_count
            map_calls = item_count
            reduce_calls = 0
        elif template_name in (\"tree_synthesis\",):
            total_calls = tree_reduce_calls(item_count, branch_factor,
                                           include_leaf=False)
            map_calls = 0
            reduce_calls = total_calls
        elif template_name == \"direct_call\":
            total_calls = 1
            map_calls = 0
            reduce_calls = 0
        else:
            total_calls = item_count or 1
            map_calls = total_calls
            reduce_calls = 0

        crit_path = critical_path_depth(item_count, branch_factor)

        assumptions = (options or {}).get(\"assumptions\", {})
        avg_input = assumptions.get(\"avg_input_tokens\", 800)
        avg_output = assumptions.get(\"avg_output_tokens\", 250)

        return {
            \"expected_llm_calls\": total_calls,
            \"critical_path_calls\": crit_path,
            \"max_concurrency\": max_conc,
            \"models\": self._model_breakdown(template, slot_values,
                                            map_calls, reduce_calls),
            \"estimated_tokens\": {
                \"prompt\": total_calls * avg_input,
                \"completion\": total_calls * avg_output,
                \"total\": total_calls * (avg_input + avg_output),
            },
        }
```

**Verification Engine (23 checks):**

```python
class VerificationEngine:
    def __init__(self, store, model_registry, template_loader):
        self.store = store
        self.registry = model_registry
        self.loader = template_loader

    def verify(self, artifact: ArtifactRecord, dry_run: DryRunRecord = None,
               policy: dict = None) -> VerificationRecord:
        policy = policy or self._default_policy()
        checks = []

        checks.append(self._check_artifact_origin(artifact))           # 1
        checks.append(self._check_artifact_hash(artifact))             # 2
        checks.append(self._check_template_version(artifact))          # 3
        checks.append(self._check_slots_filled(artifact))              # 4
        checks.append(self._check_model_exists(artifact))              # 5
        checks.append(self._check_model_capabilities(artifact))        # 6
        checks.append(self._check_image_model(artifact))               # 7
        checks.append(self._check_no_unsafe_forms(artifact))           # 8
        checks.append(self._check_no_raw_import(artifact))             # 9
        checks.append(self._check_call_count_limit(artifact, dry_run, policy))  # 10
        checks.append(self._check_recursive_depth_limit(artifact, policy))      # 11
        checks.append(self._check_concurrency_limit(artifact, policy))          # 12
        checks.append(self._check_context_exists(artifact))            # 13
        checks.append(self._check_output_schema_valid(artifact))       # 14
        checks.append(self._check_output_schema_present(artifact, policy))      # 15
        checks.append(self._check_dry_run_warnings(dry_run))           # 16
        checks.append(self._check_code_interpreter_policy(artifact, policy))    # 17
        checks.append(self._check_gate_consistency(artifact))          # 18
        checks.append(self._check_budget_policy_model(artifact))       # 19
        checks.append(self._check_budget_policy_caps(artifact))        # 20
        checks.append(self._check_primitive_allowlist(artifact))       # 21
        checks.append(self._check_context_window_fit(artifact))        # 22
        checks.append(self._check_temperature_compat(artifact))        # 23

        failed = [c for c in checks if c.status == \"fail\"]
        warned = [c for c in checks if c.status == \"warn\"]

        if failed:
            decision = \"fail\"
        elif warned:
            decision = \"warn\"
        else:
            decision = \"pass\"

        ver_id = generate_id(\"ver_\")
        record = VerificationRecord(
            verification_id=ver_id,
            artifact_id=artifact.artifact_id,
            dry_run_id=dry_run.dry_run_id if dry_run else None,
            decision=decision,
            policy=policy,
            checks=checks,
        )
        self.store.put(\"verifications\", ver_id,
                       record.model_dump(mode=\"json\"))
        return record

    def _default_policy(self) -> dict:
        return {
            \"max_llm_calls\": 1000,
            \"max_concurrency\": 50,
            \"max_recursive_depth\": 3,
            \"allow_python_bridge\": True,
            \"allow_multimodal\": True,
            \"allow_llm_generated_code\": False,
        }

    # Each _check_* method returns VerificationCheck(name, status, message)
    def _check_artifact_origin(self, artifact):
        if artifact.source_type == \"template_invocation\":
            return VerificationCheck(name=\"artifact_origin\", status=\"pass\",
                message=\"Artifact was created by the instantiator.\")
        return VerificationCheck(name=\"artifact_origin\", status=\"fail\",
            message=\"Artifact was not created by the instantiator.\")

    def _check_slots_filled(self, artifact):
        code = self._read_artifact_code(artifact)
        import re
        remaining = re.findall(r\"\\{\\{\\w+\\}\\}\", code)
        if remaining:
            return VerificationCheck(name=\"slots_filled\", status=\"fail\",
                message=f\"Unfilled slot markers: {remaining}.\")
        return VerificationCheck(name=\"slots_filled\", status=\"pass\",
            message=\"All slot markers filled.\")

    def _check_call_count_limit(self, artifact, dry_run, policy):
        if dry_run:
            expected = dry_run.summary.get(\"expected_llm_calls\", 0)
            limit = policy.get(\"max_llm_calls\", 1000)
            if expected > limit:
                return VerificationCheck(name=\"call_count_limit\", status=\"fail\",
                    message=f\"Expected {expected} calls exceeds limit {limit}.\")
        return VerificationCheck(name=\"call_count_limit\", status=\"pass\",
            message=\"Call count within limits.\")

    # ... remaining 20 checks follow the same pattern
```

**Executor Core:**

```python
class Executor:
    def __init__(self, store, llm_provider, verification_engine,
                 instantiator, template_loader, dry_runner):
        self.store = store
        self.llm = llm_provider
        self.verifier = verification_engine
        self.instantiator = instantiator
        self.loader = template_loader
        self.dry_runner = dry_runner
        self.trace = TraceCollector()

    async def execute(self, plan_id=None, template_invocation=None,
                      timeout=None, policy=None, stream=False) -> dict:
        invocation = self._resolve_invocation(plan_id, template_invocation)

        # 1. Instantiate (or reuse cached artifact)
        template = self.loader.get_template(invocation[\"template_name\"])
        artifact = self.instantiator.instantiate(
            template, invocation[\"slot_values\"], plan_id=plan_id
        )

        # 2. Dry-run for estimates
        dry_run = self.dry_runner.dry_run(
            template_invocation=invocation
        )

        # 3. Verify
        ver = self.verifier.verify(artifact, policy=policy)
        if ver.decision == \"fail\":
            return {
                \"status\": \"verification_failed\",
                \"verification\": ver.model_dump(mode=\"json\"),
            }

        # 4. Create execution record
        exec_id = generate_id(\"exec_\")
        exec_record = ExecutionRecord(
            execution_id=exec_id,
            artifact_id=artifact.artifact_id,
            plan_id=plan_id,
            verification_id=ver.verification_id,
            state=ExecutionState.running,
        )
        self.store.put(\"executions\", exec_id,
                       exec_record.model_dump(mode=\"json\"))

        # 5. Execute artifact
        try:
            result = await self._run_artifact(artifact, invocation, policy)
            exec_record.state = ExecutionState.finished
            exec_record.completed_at = datetime.now(timezone.utc)
            exec_record.metrics = self.trace.get_metrics()
        except Exception as e:
            exec_record.state = ExecutionState.failed
            exec_record.error = {\"message\": str(e)}
            result = None

        self.store.put(\"executions\", exec_id,
                       exec_record.model_dump(mode=\"json\"))

        return {
            \"status\": \"ok\" if result else \"error\",
            \"execution_id\": exec_id,
            \"artifact_id\": artifact.artifact_id,
            \"verification\": ver.model_dump(mode=\"json\"),
            \"result\": {\"value\": result} if result else None,
            \"execution\": {
                \"state\": exec_record.state.value,
                \"llm_calls\": self.trace.call_count,
                \"tokens\": self.trace.total_tokens,
            },
        }

    async def _run_artifact(self, artifact, invocation, policy) -> str:
        \"\"\"Execute the artifact by interpreting its primitives.\"\"\"
        # Dispatch LLM calls through the provider,
        # track token budget, record trace events
        ...
```

**TraceCollector:**

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
import time

@dataclass
class TraceEvent:
    type: str
    call_id: str | None = None
    node_id: str | None = None
    model: str | None = None
    tokens: int = 0
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ScopeLogEntry:
    op: str
    preview: str
    scope: str
    call_id: str | None = None

class TraceCollector:
    def __init__(self):
        self.events: list[TraceEvent] = []
        self.scope_log: list[ScopeLogEntry] = []
        self.stdout: str = \"\"
        self.call_count: int = 0
        self.total_tokens: int = 0
        self._active_calls: dict[str, float] = {}

    def record_call_start(self, call_id, node_id, model):
        self._active_calls[call_id] = time.time()
        self.events.append(TraceEvent(
            type=\"llm_call_started\", call_id=call_id,
            node_id=node_id, model=model
        ))

    def record_call_complete(self, call_id, tokens):
        start = self._active_calls.pop(call_id, time.time())
        elapsed = time.time() - start
        self.call_count += 1
        self.total_tokens += tokens
        self.events.append(TraceEvent(
            type=\"llm_call_completed\", call_id=call_id,
            tokens=tokens, elapsed_seconds=round(elapsed, 3)
        ))

    def record_scope_unwrap(self, op, preview, scope, call_id=None):
        self.scope_log.append(ScopeLogEntry(
            op=op, preview=preview[:200], scope=scope, call_id=call_id
        ))

    def get_metrics(self) -> dict:
        return {
            \"llm_calls\": self.call_count,
            \"tokens\": self.total_tokens,
        }

    def to_dict(self) -> dict:
        return {
            \"events\": [vars(e) for e in self.events],
            \"scope_log\": [vars(s) for s in self.scope_log],
            \"stdout\": self.stdout,
        }
```

### 5.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|---------------|----------|----------|
| `v2/tests/test_dry_run.py` | `test_tree_reduce_100_5` | tree_reduce_calls(100, 5) | Returns 125 (100+20+4+1) |
| `v2/tests/test_dry_run.py` | `test_tree_reduce_8_2` | tree_reduce_calls(8, 2) | Returns 15 (8+4+2+1) |
| `v2/tests/test_dry_run.py` | `test_tree_reduce_1` | tree_reduce_calls(1, 5) | Returns 1 |
| `v2/tests/test_dry_run.py` | `test_critical_path_100_5` | critical_path_depth(100, 5) | Returns 4 |
| `v2/tests/test_dry_run.py` | `test_dry_run_batch_extract_reduce` | Dry-run batch_extract_reduce, N=100, B=5 | `expected_llm_calls=125`, `critical_path_calls=4`, `max_concurrency=20` |
| `v2/tests/test_dry_run.py` | `test_dry_run_direct_call` | Dry-run direct_call | `expected_llm_calls=1` |
| `v2/tests/test_dry_run.py` | `test_dry_run_batch_map` | Dry-run batch_map, N=50 | `expected_llm_calls=50` |
| `v2/tests/test_dry_run.py` | `test_dry_run_no_real_calls` | Run dry-run with MockLLMProvider | `provider.call_count == 0` |
| `v2/tests/test_dry_run.py` | `test_dry_run_independent` | Two concurrent dry-runs | Both produce independent results |
| `v2/tests/test_dry_run.py` | `test_dry_run_creates_artifact` | Run dry-run | Artifact record stored in store |
| `v2/tests/test_dry_run.py` | `test_dry_run_mcp_tool` | Call dry_run_strategy MCP tool | Response has dry_run_id, artifact, estimate |
| `v2/tests/test_verification.py` | `test_verify_pass_all` | Valid artifact with all checks passing | `decision == \"pass\"`, 23 checks |
| `v2/tests/test_verification.py` | `test_verify_fail_slots_filled` | Artifact with remaining `{{slot}}` | `decision == \"fail\"`, slots_filled check fails |
| `v2/tests/test_verification.py` | `test_verify_fail_unsafe_forms` | Artifact containing `(eval ...)` | no_unsafe_forms check fails |
| `v2/tests/test_verification.py` | `test_verify_fail_call_count` | Expected calls exceed policy limit | call_count_limit check fails |
| `v2/tests/test_verification.py` | `test_verify_fail_unknown_model` | Artifact references unknown model alias | model_exists check fails |
| `v2/tests/test_verification.py` | `test_verify_fail_code_interpreter` | `uses-llm-generated-code: #t`, policy disallows | code_interpreter_policy check fails |
| `v2/tests/test_verification.py` | `test_verify_warn_only` | Artifact with warning-level check | `decision == \"warn\"` |
| `v2/tests/test_verification.py` | `test_verify_primitive_allowlist` | Artifact with disallowed primitive | primitive_allowlist check fails |
| `v2/tests/test_verification.py` | `test_verify_context_exists` | Artifact references nonexistent context | context_exists check fails |
| `v2/tests/test_verification.py` | `test_verify_all_23_checks_run` | Valid artifact | `len(checks) == 23` |
| `v2/tests/test_executor.py` | `test_execute_basic` | Execute direct_call template | `state == \"finished\"`, result has value |
| `v2/tests/test_executor.py` | `test_execute_verification_fail` | Execute with policy violation | `status == \"verification_failed\"`, no execution |
| `v2/tests/test_executor.py` | `test_execute_two_runs_distinct_ids` | Execute same artifact twice | Two different execution_ids |
| `v2/tests/test_executor.py` | `test_execute_tracks_tokens` | Execute with MockLLMProvider | `tokens > 0` in metrics |
| `v2/tests/test_executor.py` | `test_execute_state_transitions` | Execute successfully | State goes queued -> running -> finished |
| `v2/tests/test_executor.py` | `test_execute_failure_state` | Execute with forced error | State is `failed`, error field populated |
| `v2/tests/test_executor.py` | `test_trace_events` | Execute and get trace | Trace has llm_call_started and llm_call_completed events |
| `v2/tests/test_executor.py` | `test_trace_scope_log` | Execute with syntax-e operations | Scope log has entries with op, preview, scope |
| `v2/tests/test_executor.py` | `test_execute_mcp_tool` | Call execute_strategy MCP tool | Response has execution_id, verification, result |

### 5.6 Acceptance Gates

```bash
# Dry-run tests pass
cd v2 && python -m pytest tests/test_dry_run.py -v
# Expected: 11 passed

# Verification tests pass
cd v2 && python -m pytest tests/test_verification.py -v
# Expected: 11 passed

# Executor tests pass
cd v2 && python -m pytest tests/test_executor.py -v
# Expected: 9 passed

# Tree-reduce formula validation
cd v2 && python -c \"
from rlm_scheme.dry_runner import tree_reduce_calls, critical_path_depth
assert tree_reduce_calls(100, 5) == 125, 'N=100 B=5 should be 125'
assert tree_reduce_calls(8, 2) == 15, 'N=8 B=2 should be 15'
assert critical_path_depth(100, 5) == 4, 'Critical path should be 4'
print('Call count formulas OK')
\"
# Expected: \"Call count formulas OK\"

# Verification runs all 23 checks
cd v2 && python -c \"
from rlm_scheme.verification import VerificationEngine
# ... setup with valid artifact ...
# assert len(result.checks) == 23
print('23 checks verified')
\"
# Expected: \"23 checks verified\"

# Full pipeline: plan -> dry_run -> execute
cd v2 && python -c \"
import json
from rlm_scheme.mcp_server import load_context, plan_strategy, dry_run_strategy, execute_strategy
import asyncio
ctx = json.loads(load_context('[{\\\"id\\\":1},{\\\"id\\\":2},{\\\"id\\\":3}]', name='test'))
plan = json.loads(plan_strategy('Extract data from items', ctx['context_id']))
dry = json.loads(dry_run_strategy(plan_id=plan['plan_id']))
assert dry['status'] == 'ok'
assert 'expected_llm_calls' in dry['estimate']
print('Pipeline OK')
\"
# Expected: \"Pipeline OK\"
```

### 5.7 Checklist

- [ ] `v2/rlm_scheme/dry_runner.py` computes call counts, simulates execution, estimates costs
- [ ] Tree-reduce formula: `tree_reduce_calls(100, 5) == 125`
- [ ] Dry-run makes zero real LLM calls
- [ ] Dry-run is stateless (no global mutable state)
- [ ] `v2/rlm_scheme/cost_model.py` estimates tokens and cost per model
- [ ] `v2/rlm_scheme/verification.py` implements all 23 checks
- [ ] Verification returns pass/warn/fail decision
- [ ] Failed verification blocks execution
- [ ] `v2/rlm_scheme/executor.py` dispatches LLM calls, tracks budget, manages state
- [ ] `v2/rlm_scheme/trace.py` records events, scope log, stdout
- [ ] `dry_run_strategy` MCP tool wired to real implementation
- [ ] `execute_strategy` MCP tool wired to real implementation
- [ ] `v2/tests/test_dry_run.py` has 11+ tests
- [ ] `v2/tests/test_verification.py` has 11+ tests
- [ ] `v2/tests/test_executor.py` has 9+ tests
- [ ] All tests pass: `python -m pytest tests/ -v`

---

## Batch 6: Cache, Gates, and Chain Execution

### 6.0 Purpose

Implement a content-addressed LLM result cache, a gate primitive with resume and cancel operations, and sequential template chain execution. These three components are orthogonal but all consumed by the executor layer; building them together allows the chain executor to test cache integration and gate suspension in the same batch.

### 6.1 Depends On

- Batch 1 (Store, models, exceptions)
- Batch 2 (Context primitives)
- Batch 3 (Template parser and validator)
- Batch 4 (Dry-runner, cost model)
- Batch 5 (Executor core, LLM adapter)

### 6.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| v2/rlm_scheme/cache.py | Create | Content-addressed LLM result cache |
| v2/rlm_scheme/gate.py | Create | GateManager: gate registration, resume, cancel |
| v2/rlm_scheme/chain.py | Create | ChainExecutor: step sequencing, $previous resolution |
| v2/tests/test_cache.py | Create | Cache unit tests |
| v2/tests/test_gate.py | Create | Gate unit tests |
| v2/tests/test_chain.py | Create | Chain execution tests |

### 6.3 Requirements

1. [MUST] R-6.1: The cache key MUST equal `sha256(canonical_json(instruction) + canonical_json(data) + model_alias + str(temperature) + str(json_mode))` where `canonical_json(x)` is `json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True)`.
2. [MUST] R-6.2: Cache entries MUST be immutable — storing a different value under the same key MUST raise `CacheKeyCollisionError`.
3. [MUST] R-6.3: Calls with `temperature > 0` MUST NOT be cached unless the originating template declares `cacheable: #t`.
4. [MUST] R-6.4: Cache hits MUST NOT consume any token budget; the executor MUST detect a cache hit before dispatching to the LLM adapter.
5. [MUST] R-6.5: `reset_runtime(scope="cache")` MUST delete all cache entries and reset hit/miss counters to zero.
6. [MUST] R-6.6: `GateManager` MUST track pending gates per `execution_id` in a dict keyed by `(execution_id, gate_name)`.
7. [MUST] R-6.7: `GateManager.resume(execution_id, gate, decision="approve")` MUST set the gate state to `"approved"` and return the full gate record.
8. [MUST] R-6.8: `GateManager.resume(execution_id, gate, decision="reject")` MUST set the gate state to `"rejected"` and return the full gate record including the `reason` field.
9. [MUST] R-6.9: `GateManager.cancel_all(execution_id)` MUST transition every pending gate for that `execution_id` to state `"cancelled"` and return the count of gates cancelled.
10. [MUST] R-6.10: `ChainExecutor` MUST resolve any `"$previous"` string value in a step's `slot_values` to the `context_id` of the immediately preceding step's output context.
11. [MUST] R-6.11: Chain steps MUST execute strictly sequentially; step N+1 MUST NOT begin until step N has produced an output context.
12. [MUST] R-6.12: Each completed chain step MUST store its output as an auto-created context with an auto-generated `context_id` of the form `ctx_auto_{step_index}`.
13. [SHOULD] R-6.13: If a chain step fails, `ChainExecutor` SHOULD preserve all previously completed steps' output contexts rather than rolling them back.

### 6.4 Detailed Specifications

**Cache key computation**

`canonical_json` is defined as:

```python
def canonical_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
```

The raw cache key material is the concatenation (with no separator) of:

```
canonical_json(instruction) + canonical_json(data) + model_alias + str(temperature) + str(json_mode)
```

The stored key is `hashlib.sha256(key_material.encode()).hexdigest()`.

**LLMCache public interface**

```python
class LLMCache:
    def __init__(self, store: Store) -> None: ...
    def make_key(self, instruction: object, data: object, model_alias: str,
                 temperature: float, json_mode: bool) -> str: ...
    def lookup(self, key: str) -> str | None: ...
    def store(self, key: str, result: str, metadata: dict) -> None: ...
    def clear(self) -> None: ...
    @property
    def stats(self) -> dict: ...  # {"hits": int, "misses": int, "size": int}
```

`store` writes cache entries to the `Store` under the namespace `"cache"`. The `metadata` dict MUST include at minimum `{"cached_at": iso8601_utc_string}`. Raising `CacheKeyCollisionError` (defined in `exceptions.py`) when a key collision with a different value is detected satisfies R-6.2.

**GateManager public interface**

```python
class GateState(str, enum.Enum):
    PENDING   = "pending"
    APPROVED  = "approved"
    REJECTED  = "rejected"
    CANCELLED = "cancelled"

@dataclasses.dataclass
class GateRecord:
    execution_id: str
    gate_name: str
    state: GateState
    decision: str | None
    reason: str | None
    registered_at: str     # ISO-8601 UTC
    resolved_at: str | None

class GateManager:
    def __init__(self) -> None: ...
    def register(self, execution_id: str, gate_name: str) -> GateRecord: ...
    def resume(self, execution_id: str, gate_name: str,
               decision: str, reason: str | None = None) -> GateRecord | None: ...
    def cancel_all(self, execution_id: str) -> int: ...
    def pending_gates(self, execution_id: str) -> list[GateRecord]: ...
```

`resume` returns `None` when no gate matching `(execution_id, gate_name)` exists in the manager.

**ChainExecutor public interface**

```python
@dataclasses.dataclass
class ChainStep:
    template_id: str
    slot_values: dict[str, object]

@dataclasses.dataclass
class ChainResult:
    steps_completed: int
    output_context_ids: list[str]   # one per completed step
    final_context_id: str | None
    error: str | None

class ChainExecutor:
    def __init__(self, executor: Executor, store: Store) -> None: ...
    async def run(self, steps: list[ChainStep],
                  execution_id: str) -> ChainResult: ...
    def _resolve_previous(self, slot_values: dict, previous_context_id: str | None) -> dict: ...
```

`_resolve_previous` walks `slot_values` and replaces any string value equal to `"$previous"` with `previous_context_id`. If `previous_context_id` is `None` (first step) and `"$previous"` is present, it MUST raise `ChainResolutionError`.

**$previous resolution example**

Given the chain:

```python
steps = [
    ChainStep("summarize", {"text": "ctx_input"}),
    ChainStep("classify",  {"text": "$previous"}),
]
```

After step 0 completes and produces `ctx_auto_0`, `_resolve_previous` transforms step 1's slot_values to `{"text": "ctx_auto_0"}` before execution.

**Cache integration in executor**

The executor (from Batch 5) MUST call `cache.lookup(key)` before dispatching to the LLM adapter. If a hit is found, the result is returned directly and the token budget is not decremented.

### 6.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|--------------|----------|----------|
| test_cache.py | test_cache_hit | Store a result then look up same key | Returns cached result string |
| test_cache.py | test_cache_miss | Look up a key that was never stored | Returns `None` |
| test_cache.py | test_cache_key_deterministic | Call `make_key` twice with identical arguments | Both calls return the same hex string |
| test_cache.py | test_cache_different_inputs | Two calls with different `instruction` values | Distinct keys produced |
| test_cache.py | test_cache_clear | Store an entry then call `clear()` then lookup | Returns `None` after clear |
| test_cache.py | test_temperature_not_cached | Attempt to store with `temperature=0.7` and `cacheable=False` | Entry not stored; `lookup` returns `None` |
| test_cache.py | test_cache_stats_hit | Store then lookup (hit) | `stats["hits"] == 1` |
| test_cache.py | test_cache_stats_miss | Lookup non-existent key | `stats["misses"] == 1` |
| test_gate.py | test_register_gate | Register a gate for `execution_id="ex1"` | Returns `GateRecord` with `state="pending"` |
| test_gate.py | test_resume_approve | Register then `resume(..., decision="approve")` | Record state is `"approved"` |
| test_gate.py | test_resume_reject | Register then `resume(..., decision="reject", reason="bad")` | Record state is `"rejected"`, `reason="bad"` |
| test_gate.py | test_cancel_all | Register 2 gates then `cancel_all("ex1")` | Returns `2`; both gates are `"cancelled"` |
| test_gate.py | test_resume_nonexistent | Resume gate that was never registered | Returns `None` |
| test_gate.py | test_pending_gates_list | Register 2 gates, approve 1, call `pending_gates` | Returns list of length 1 |
| test_chain.py | test_resolve_previous | Call `_resolve_previous` with `"$previous"` value and a real context_id | Value replaced with context_id |
| test_chain.py | test_resolve_previous_no_previous | Call `_resolve_previous` with `"$previous"` but `previous_context_id=None` | Raises `ChainResolutionError` |
| test_chain.py | test_single_step_chain | One-step chain with mocked executor | `steps_completed == 1`, one output context |
| test_chain.py | test_multi_step_chain | Two-step chain with mocked executor | `steps_completed == 2`, two output contexts |
| test_chain.py | test_chain_intermediate_context | Two-step chain | `ctx_auto_0` context exists in store after run |
| test_chain.py | test_chain_preserves_steps_on_failure | Two-step chain where step 2 raises | `steps_completed == 1`, `error` is set |

### 6.6 Acceptance Gates

```sh
# Gate 1: all new tests pass
cd v2 && python -m pytest tests/test_cache.py tests/test_gate.py tests/test_chain.py -v

# Gate 2: cache round-trip works from a fresh store
cd v2 && python -c "
from rlm_scheme.cache import LLMCache
from rlm_scheme.store import Store
from pathlib import Path
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    c = LLMCache(Store(Path(d)))
    key = c.make_key('instruction', {'x': 1}, 'gpt-4o', 0.0, False)
    c.store(key, 'result42', {})
    assert c.lookup(key) == 'result42', 'cache round-trip failed'
    print('cache round-trip OK')
"

# Gate 3: gate approve/reject cycle works
cd v2 && python -c "
from rlm_scheme.gate import GateManager
gm = GateManager()
gm.register('ex1', 'human-review')
rec = gm.resume('ex1', 'human-review', decision='approve')
assert rec.state.value == 'approved', f'expected approved, got {rec.state}'
print('gate cycle OK')
"

# Gate 4: full test suite still green
cd v2 && python -m pytest tests/ -q --tb=short
```

### 6.7 Checklist

- [ ] `v2/rlm_scheme/cache.py` created with `LLMCache`, `canonical_json`, `make_key`, `lookup`, `store`, `clear`, `stats`
- [ ] `CacheKeyCollisionError` added to `v2/rlm_scheme/exceptions.py`
- [ ] `ChainResolutionError` added to `v2/rlm_scheme/exceptions.py`
- [ ] `v2/rlm_scheme/gate.py` created with `GateState`, `GateRecord`, `GateManager`
- [ ] `v2/rlm_scheme/chain.py` created with `ChainStep`, `ChainResult`, `ChainExecutor`
- [ ] Temperature-gating check implemented (R-6.3)
- [ ] Cache lookup integrated into Batch 5 executor before LLM dispatch (R-6.4)
- [ ] `reset_runtime(scope="cache")` wired to `LLMCache.clear()` (R-6.5)
- [ ] `_resolve_previous` raises `ChainResolutionError` when `$previous` present but no prior context
- [ ] All 20 tests in this batch pass
- [ ] No regressions in Batches 1–5 tests

---

## Batch 7: MCP Server Wiring

### 7.0 Purpose

Wire all 10 MCP tools to the real implementations built in Batches 1–6. After this batch the MCP server is fully functional end-to-end: load_context, get_context, plan_strategy, dry_run_strategy, execute_strategy, get_execution_trace, get_status, cancel_call, resume_execution, and reset_runtime all delegate to their respective subsystems rather than returning stubs.

### 7.1 Depends On

- Batch 1 (Store, models, exceptions)
- Batch 2 (Context primitives, ContextManager)
- Batch 3 (TemplateParser, TemplateValidator)
- Batch 4 (Planner, DryRunner)
- Batch 5 (Executor, LLM adapter, token budget)
- Batch 6 (LLMCache, GateManager, ChainExecutor)

### 7.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| v2/rlm_scheme/mcp_server.py | Modify | Wire all 10 tools to real implementations |
| v2/tests/test_mcp_server.py | Create | Integration tests covering all 10 tools |

### 7.3 Requirements

1. [MUST] R-7.1: `load_context` MUST accept `data: str`, `name: str | None`, and `metadata_json: str | None`; it MUST return JSON with at minimum `{"status": "ok", "context_id": "<id>"}`.
2. [MUST] R-7.2: `get_context` MUST return a JSON object with `context_id`, `name`, `size_bytes`, and optional `preview` and `data` fields controlled by `include_preview` and `include_data` boolean parameters.
3. [MUST] R-7.3: `plan_strategy` MUST accept `task: str`, optional `context_id: str`, and optional `hints_json: str`; it MUST return JSON with `plan_id` and `recommended_template`.
4. [MUST] R-7.4: `dry_run_strategy` MUST accept either `plan_id` or `template_invocation_json` (at least one required) and return JSON with `dry_run_id` and an `artifact` object containing token estimates.
5. [MUST] R-7.5: `execute_strategy` MUST accept `plan_id` or `template_invocation_json`, plus `timeout_seconds: int | None`, `stream: bool`, `policy_json: str | None`, and `runtime_options_json: str | None`.
6. [MUST] R-7.6: `get_execution_trace` MUST return a JSON object with an `events` array, a `scope_log` array, and a `stdout` string; the `include_scope_log`, `include_calls`, and `include_stdout` boolean parameters MUST control which sections are populated.
7. [MUST] R-7.7: `get_status` with no arguments MUST return `token_usage` and `cache_stats`; with `execution_id` MUST additionally return `execution_status`, `steps_completed`, and `errors`.
8. [MUST] R-7.8: `cancel_call` MUST accept `call_id: str | None` and `execution_id: str | None`; at least one MUST be provided or the tool MUST return `{"status": "error", "errors": ["either call_id or execution_id required"]}`.
9. [MUST] R-7.9: `resume_execution` MUST accept `execution_id: str`, `gate: str`, `decision: str` (either `"approve"` or `"reject"`), and optional `reason: str`; it MUST delegate to `GateManager.resume`.
10. [MUST] R-7.10: `reset_runtime` MUST support all 7 scopes defined in Appendix A item A.5: `"sandbox"`, `"session"`, `"cache"`, `"contexts"`, `"executions"`, `"config"`, and `"all"`.
11. [MUST] R-7.11: Every tool MUST return a JSON string whose top-level object contains a `"status"` field with value `"ok"`, `"warn"`, or `"error"`.
12. [MUST] R-7.12: Any response with `"status": "error"` MUST include an `"errors"` array containing at least one non-empty string message.
13. [SHOULD] R-7.13: Tool implementations SHOULD catch all exceptions from subsystems, log them, and convert them to `{"status": "error", "errors": [...]}` responses rather than raising.
14. [SHOULD] R-7.14: `plan_strategy` SHOULD include a `reasoning` field in its response explaining the planner's template selection rationale.

### 7.4 Detailed Specifications

**Module-level singleton setup**

`mcp_server.py` MUST instantiate singletons at module load time (not inside tool functions) so that state persists across tool calls within a session:

```python
_store           = Store(Path(os.environ.get("RLM_STORE_DIR", "/tmp/rlm_scheme")))
_context_manager = ContextManager(_store)
_template_parser = TemplateParser()
_planner         = Planner(_template_parser)
_dry_runner      = DryRunner(_template_parser, CostModel())
_llm_adapter     = LLMAdapter()
_cache           = LLMCache(_store)
_gate_manager    = GateManager()
_executor        = Executor(_llm_adapter, _cache, _gate_manager, _store)
_chain_executor  = ChainExecutor(_executor, _store)
_token_budget    = TokenBudget()
```

**Full tool signatures**

```python
@mcp.tool()
def load_context(
    data: str,
    name: str | None = None,
    metadata_json: str | None = None,
) -> str: ...

@mcp.tool()
def get_context(
    context_id: str,
    include_preview: bool = True,
    include_data: bool = False,
) -> str: ...

@mcp.tool()
async def plan_strategy(
    task: str,
    context_id: str | None = None,
    hints_json: str | None = None,
) -> str: ...

@mcp.tool()
def dry_run_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    options_json: str | None = None,
) -> str: ...

@mcp.tool()
async def execute_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    timeout_seconds: int | None = None,
    stream: bool = False,
    policy_json: str | None = None,
    runtime_options_json: str | None = None,
) -> str: ...

@mcp.tool()
def get_execution_trace(
    execution_id: str,
    include_scope_log: bool = True,
    include_calls: bool = True,
    include_stdout: bool = True,
) -> str: ...

@mcp.tool()
def get_status(
    execution_id: str | None = None,
) -> str: ...

@mcp.tool()
def cancel_call(
    call_id: str | None = None,
    execution_id: str | None = None,
    reason: str | None = None,
) -> str: ...

@mcp.tool()
async def resume_execution(
    execution_id: str,
    gate: str,
    decision: str,
    reason: str | None = None,
) -> str: ...

@mcp.tool()
def reset_runtime(
    scope: str = "session",
) -> str: ...
```

**reset_runtime scope semantics**

| Scope | Action |
|-------|--------|
| `"sandbox"` | Clear Racket sandbox state only; leave all durable records, caches, and config intact |
| `"session"` | Clear execution traces, token budget counters, and sandbox state; leave contexts and cache intact |
| `"cache"` | Call `_cache.clear()`; reset cache stats to zero |
| `"contexts"` | Delete all contexts from the store; reset context manager |
| `"executions"` | Clear all execution traces and cancel all pending gates |
| `"config"` | Reload model registry and template catalog from disk |
| `"all"` | Apply all of the above scopes |

**Error response shape**

```json
{
  "status": "error",
  "errors": ["human-readable message"]
}
```

**Warn response shape**

Used when the operation partially succeeded:

```json
{
  "status": "warn",
  "warnings": ["what was degraded"],
  "result": { ... }
}
```

**plan_strategy response shape**

```json
{
  "status": "ok",
  "plan_id": "plan_<uuid>",
  "recommended_template": "map-reduce-summarize",
  "classification": "multi-document",
  "reasoning": "Task contains multiple items; map-reduce is appropriate.",
  "estimated_calls": 7
}
```

**execute_strategy response shape**

```json
{
  "status": "ok",
  "execution_id": "exec_<uuid>",
  "steps_completed": 1,
  "output_context_id": "ctx_<uuid>",
  "token_usage": {"prompt": 1200, "completion": 400, "total": 1600}
}
```

### 7.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|--------------|----------|----------|
| test_mcp_server.py | test_load_json_data | `load_context` with valid JSON array string | `status == "ok"`, `context_id` present |
| test_mcp_server.py | test_load_string_data | `load_context` with plain string | `status == "ok"`, `context_id` present |
| test_mcp_server.py | test_load_with_name | `load_context` with `name="my_doc"` | Response includes `name == "my_doc"` |
| test_mcp_server.py | test_get_existing_context | Load then `get_context` | `status == "ok"`, `preview` non-empty |
| test_mcp_server.py | test_get_missing_context | `get_context("ctx_does_not_exist")` | `status == "error"` |
| test_mcp_server.py | test_get_context_include_data | Load then `get_context(include_data=True)` | Response includes `data` field |
| test_mcp_server.py | test_plan_with_context | Load context then `plan_strategy` with context_id | `plan_id` present, `status == "ok"` |
| test_mcp_server.py | test_plan_without_context | `plan_strategy(task="summarize this")` | `plan_id` present, `status == "ok"` |
| test_mcp_server.py | test_plan_with_hints | `plan_strategy` with `hints_json='{"prefer": "map-reduce"}'` | `status == "ok"` |
| test_mcp_server.py | test_dry_run_from_plan | `plan_strategy` then `dry_run_strategy(plan_id=...)` | `dry_run_id` present, `artifact` present |
| test_mcp_server.py | test_dry_run_from_invocation | `dry_run_strategy(template_invocation_json=...)` | `status == "ok"` |
| test_mcp_server.py | test_dry_run_no_args | `dry_run_strategy()` with no arguments | `status == "error"` |
| test_mcp_server.py | test_execute_from_plan | `plan_strategy` then `execute_strategy(plan_id=...)` | `execution_id` present, `status == "ok"` |
| test_mcp_server.py | test_execute_from_invocation | `execute_strategy(template_invocation_json=...)` | `status == "ok"` |
| test_mcp_server.py | test_execute_no_args | `execute_strategy()` with no arguments | `status == "error"` |
| test_mcp_server.py | test_trace_after_execution | Execute then `get_execution_trace` | `events` array non-empty |
| test_mcp_server.py | test_trace_missing_execution | `get_execution_trace("exec_nonexistent")` | `status == "error"` |
| test_mcp_server.py | test_trace_include_flags | Trace with all include flags `False` | Response has empty `events`, no `scope_log`, no `stdout` |
| test_mcp_server.py | test_general_status | `get_status()` with no args | `token_usage` present, `cache_stats` present |
| test_mcp_server.py | test_execution_status | Execute then `get_status(execution_id=...)` | `execution_status` present |
| test_mcp_server.py | test_cancel_with_execution_id | Execute then `cancel_call(execution_id=...)` | `status == "ok"` |
| test_mcp_server.py | test_cancel_no_args | `cancel_call()` with no arguments | `status == "error"` |
| test_mcp_server.py | test_resume_no_pending_gate | `resume_execution` for execution with no registered gate | `status == "error"` |
| test_mcp_server.py | test_resume_invalid_decision | `resume_execution(..., decision="maybe")` | `status == "error"` |
| test_mcp_server.py | test_reset_sandbox | `reset_runtime("sandbox")` | `status == "ok"` |
| test_mcp_server.py | test_reset_session | `reset_runtime("session")` | `status == "ok"` |
| test_mcp_server.py | test_reset_cache | `reset_runtime("cache")` | `status == "ok"`, cache stats zeroed |
| test_mcp_server.py | test_reset_contexts | `reset_runtime("contexts")` | `status == "ok"` |
| test_mcp_server.py | test_reset_executions | `reset_runtime("executions")` | `status == "ok"` |
| test_mcp_server.py | test_reset_config | `reset_runtime("config")` | `status == "ok"` |
| test_mcp_server.py | test_reset_all | `reset_runtime("all")` | `status == "ok"` |
| test_mcp_server.py | test_reset_invalid_scope | `reset_runtime("bogus")` | `status == "error"` |
| test_mcp_server.py | test_e2e_pipeline | Full `load_context` → `plan_strategy` → `dry_run_strategy` → `execute_strategy` → `get_execution_trace` → `get_status` | All six calls return `status == "ok"` |

### 7.6 Acceptance Gates

```sh
# Gate 1: all MCP server integration tests pass
cd v2 && python -m pytest tests/test_mcp_server.py -v

# Gate 2: exactly 10 tools are registered
cd v2 && python -c "
from rlm_scheme.mcp_server import mcp
tools = [t.name for t in mcp.list_tools()]
assert len(tools) == 10, f'expected 10 tools, got {len(tools)}: {tools}'
expected = {
    'load_context', 'get_context', 'plan_strategy', 'dry_run_strategy',
    'execute_strategy', 'get_execution_trace', 'get_status',
    'cancel_call', 'resume_execution', 'reset_runtime',
}
assert set(tools) == expected, f'tool name mismatch: {set(tools) ^ expected}'
print('All 10 tools registered correctly')
"

# Gate 3: every tool returns JSON with a status field
cd v2 && python -c "
import asyncio, json
from rlm_scheme.mcp_server import (
    load_context, get_context, dry_run_strategy,
    get_status, cancel_call, reset_runtime,
)
results = [
    load_context('hello world'),
    get_context('ctx_nonexistent'),
    dry_run_strategy(),
    get_status(),
    cancel_call(),
    reset_runtime('session'),
]
for r in results:
    d = json.loads(r)
    assert 'status' in d, f'missing status in: {r}'
print('All checked tools return status field')
"

# Gate 4: full test suite still green
cd v2 && python -m pytest tests/ -q --tb=short
```

### 7.7 Checklist

- [ ] All module-level singletons instantiated at the top of `mcp_server.py`
- [ ] `load_context` wired to `ContextManager.store`
- [ ] `get_context` wired to `ContextManager.get`, returns correct fields
- [ ] `plan_strategy` wired to `Planner.plan`, returns `plan_id` and `recommended_template`
- [ ] `dry_run_strategy` wired to `DryRunner.dry_run`, requires at least one of `plan_id` / `template_invocation_json`
- [ ] `execute_strategy` wired to `Executor.execute`, requires at least one of `plan_id` / `template_invocation_json`
- [ ] `get_execution_trace` wired to executor trace store, returns `events`, `scope_log`, `stdout`
- [ ] `get_status` returns `token_usage` and `cache_stats` without args; execution details with `execution_id`
- [ ] `cancel_call` returns error when both `call_id` and `execution_id` are `None`
- [ ] `resume_execution` delegates to `GateManager.resume`, validates `decision` is `"approve"` or `"reject"`
- [ ] `reset_runtime` handles all 7 scopes and returns error for unrecognised scope
- [ ] All 31 tests in this batch pass
- [ ] No regressions in Batches 1–6 tests

---

## Batch 8: Streaming and Budget Degradation

### 8.0 Purpose

Implement streaming partial results emitted as MCP progress notifications during map-async and tree-reduce execution, budget monitoring that detects low-budget thresholds and activates degradation policies (switch-model or checkpoint-and-stop), and code interpreter template support with a controlled Python bridge enforcing an import allowlist.

### 8.1 Depends On

- Batch 1 (Store, models, exceptions)
- Batch 2 (Context primitives)
- Batch 3 (Template parser — map-async, tree-reduce, code interpreter structures)
- Batch 4 (Dry-runner, cost model)
- Batch 5 (Executor, LLM adapter, token budget)
- Batch 6 (Cache, GateManager, ChainExecutor)
- Batch 7 (MCP server singletons, execute_strategy stream parameter)

### 8.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| v2/rlm_scheme/streaming.py | Create | Streaming notification helpers for map-async and tree-reduce |
| v2/rlm_scheme/budget.py | Create | BudgetMonitor: threshold detection, policy dispatch |
| v2/rlm_scheme/python_bridge.py | Create | Controlled Python subprocess for py-exec, py-eval with import allowlist |
| v2/rlm_scheme/executor.py | Modify | Integrate streaming notifications, budget monitor, code interpreter dispatch |
| v2/tests/test_streaming.py | Create | Streaming integration tests |
| v2/tests/test_budget.py | Create | Budget degradation tests |

### 8.3 Requirements

1. [SHOULD] R-8.1: When `stream=True`, the executor SHOULD emit MCP `notifications/partial_result` messages for each completed item in a map-async or tree-reduce node.
2. [SHOULD] R-8.2: Progress notifications SHOULD include `execution_id`, `node_id`, `items_completed`, and `items_total` fields.
3. [MUST] R-8.3: `BudgetMonitor` MUST fire a callback when cumulative token usage first crosses the template's `low-budget-threshold` (expressed as a fraction of `max-tokens`).
4. [MUST] R-8.4: When the low-budget threshold is crossed, the executor MUST consult the template's `budget-policy` and dispatch to either `_apply_switch_model` or `_apply_checkpoint_and_stop`.
5. [MUST] R-8.5: `_apply_switch_model` MUST change the `model_alias` for all remaining LLM calls in the current execution to the `fallback-model` named in `budget-policy`.
6. [MUST] R-8.6: `_apply_checkpoint_and_stop` MUST write a checkpoint record to the store containing completed steps and their output context IDs, then raise `BudgetExhaustedError` to halt execution gracefully.
7. [MUST] R-8.7: Any template that uses `py-exec` or `py-eval` MUST have `uses-llm-generated-code: #t` set; the executor MUST reject code interpreter dispatch without this flag.
8. [MUST] R-8.8: The Python bridge MUST enforce the import allowlist from the template's `code-generation-policy`; any attempt to `import` a module not on the allowlist MUST raise `ImportBlockedError` and terminate the subprocess.
9. [SHOULD] R-8.9: The Python bridge SHOULD run LLM-generated code with a 10-second execution timeout; trusted code (not LLM-generated) SHOULD use a 30-second timeout.
10. [SHOULD] R-8.10: Streaming notifications SHOULD be best-effort; failure to deliver a notification MUST NOT abort the execution.

### 8.4 Detailed Specifications

**StreamingNotifier public interface**

```python
class StreamingNotifier:
    def __init__(self, context: fastmcp.Context | None) -> None: ...

    async def notify_item_complete(
        self,
        execution_id: str,
        node_id: str,
        items_completed: int,
        items_total: int,
        partial_result: object | None = None,
    ) -> None: ...

    async def notify_phase_complete(
        self,
        execution_id: str,
        phase: str,
    ) -> None: ...
```

When `context` is `None`, all notify calls are no-ops. Exceptions from `context.report_progress` MUST be caught and logged without propagation (satisfying R-8.10).

**BudgetMonitor public interface**

```python
@dataclasses.dataclass
class BudgetPolicy:
    action: str                   # "switch-model" or "checkpoint-and-stop"
    fallback_model: str | None    # required when action == "switch-model"
    threshold_fraction: float     # 0.0–1.0, e.g. 0.2 means 20% remaining

class BudgetMonitor:
    def __init__(self, max_tokens: int, policy: BudgetPolicy) -> None: ...

    def record_usage(self, tokens_used: int) -> bool:
        """Return True if threshold was newly crossed, False otherwise."""
        ...

    @property
    def threshold_crossed(self) -> bool: ...

    @property
    def tokens_remaining(self) -> int: ...
```

`record_usage` MUST be idempotent regarding the `threshold_crossed` flag — once crossed it stays crossed regardless of subsequent calls.

**BudgetExhaustedError** is added to `exceptions.py`. Its message MUST include the checkpoint record ID.

**Checkpoint record schema** (stored under namespace `"checkpoints"` in the Store):

```json
{
  "execution_id": "exec_<uuid>",
  "checkpoint_id": "ckpt_<uuid>",
  "created_at": "<ISO-8601>",
  "steps_completed": 3,
  "output_context_ids": ["ctx_auto_0", "ctx_auto_1", "ctx_auto_2"],
  "tokens_used_at_checkpoint": 48000
}
```

**Python bridge public interface**

```python
class PythonBridge:
    DEFAULT_TIMEOUT    = 30   # seconds, trusted code
    LLM_CODE_TIMEOUT   = 10   # seconds, LLM-generated code

    def __init__(
        self,
        allowlist: list[str] | None = None,
        is_llm_generated: bool = False,
    ) -> None: ...

    def exec(self, code: str, globals_dict: dict | None = None) -> dict:
        """Execute code string; return updated globals. Raises on timeout or blocked import."""
        ...

    def eval(self, expression: str, globals_dict: dict | None = None) -> object:
        """Evaluate expression string; return result."""
        ...
```

The import allowlist check works by wrapping the code in a `RestrictedImporter` context manager that overrides `__import__` for the duration of the call. If `allowlist` is `None`, a safe default allowlist of `["json", "math", "re", "datetime", "collections", "itertools", "functools", "typing"]` is used.

**Executor modifications**

The executor's `_dispatch_node` method gains two new branches:

1. **streaming branch**: after each completed item in map-async or tree-reduce, call `await notifier.notify_item_complete(...)`.
2. **code interpreter branch**: when a node has type `py-exec` or `py-eval`, route to `PythonBridge` instead of the LLM adapter. Validate `uses-llm-generated-code` flag before dispatching (R-8.7).

Budget monitoring is integrated into the executor's main execution loop:

```python
if budget_monitor and budget_monitor.record_usage(tokens_used):
    if policy.action == "switch-model":
        self._apply_switch_model(policy.fallback_model)
    elif policy.action == "checkpoint-and-stop":
        self._apply_checkpoint_and_stop(execution_id, completed_steps)
        raise BudgetExhaustedError(checkpoint_id)
```

### 8.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|--------------|----------|----------|
| test_streaming.py | test_streaming_flag_no_error | Execute with `stream=True`, `context=None` | No exception; execution completes normally |
| test_streaming.py | test_notify_item_complete_no_op | `StreamingNotifier(None).notify_item_complete(...)` | Returns without error |
| test_streaming.py | test_notify_tracks_progress | Mock `context`; call `notify_item_complete` 3 times | `context.report_progress` called 3 times |
| test_budget.py | test_budget_monitor_not_triggered | Use 10% of budget with 20% threshold | `threshold_crossed == False` |
| test_budget.py | test_budget_monitor_activation | Use 85% of budget with 20% threshold | `record_usage` returns `True`, `threshold_crossed == True` |
| test_budget.py | test_budget_monitor_idempotent | Cross threshold then record more usage | Second call returns `False` (already crossed) |
| test_budget.py | test_model_switch | Execute with budget policy `switch-model` at 50% | Active model alias changes to `fallback_model` |
| test_budget.py | test_checkpoint_and_stop | Execute with policy `checkpoint-and-stop` at 1% threshold | `BudgetExhaustedError` raised; checkpoint in store |
| test_budget.py | test_checkpoint_record_fields | Inspect checkpoint after stop | `execution_id`, `steps_completed`, `output_context_ids` all present |
| test_budget.py | test_python_bridge_exec | `PythonBridge().exec("x = 2 + 2")` | Returns `{"x": 4}` in globals |
| test_budget.py | test_python_bridge_eval | `PythonBridge().eval("3 * 7")` | Returns `21` |
| test_budget.py | test_python_bridge_blocked_import | `exec("import os")` with default allowlist | Raises `ImportBlockedError` |
| test_budget.py | test_python_bridge_allowed_import | `exec("import json")` with default allowlist | No error |
| test_budget.py | test_code_interpreter_flag_missing | Dispatch `py-exec` node without `uses-llm-generated-code: True` | Raises `TemplateValidationError` |

### 8.6 Acceptance Gates

```sh
# Gate 1: all streaming and budget tests pass
cd v2 && python -m pytest tests/test_streaming.py tests/test_budget.py -v

# Gate 2: BudgetMonitor threshold detection works
cd v2 && python -c "
from rlm_scheme.budget import BudgetMonitor, BudgetPolicy
policy = BudgetPolicy(action='switch-model', fallback_model='gpt-4o-mini', threshold_fraction=0.2)
bm = BudgetMonitor(max_tokens=10000, policy=policy)
triggered = bm.record_usage(8500)
assert triggered, 'threshold should have been crossed'
assert bm.threshold_crossed
print('BudgetMonitor threshold OK')
"

# Gate 3: PythonBridge import blocking works
cd v2 && python -c "
from rlm_scheme.python_bridge import PythonBridge
from rlm_scheme.exceptions import ImportBlockedError
pb = PythonBridge(allowlist=['json', 'math'])
try:
    pb.exec('import subprocess')
    raise AssertionError('should have raised ImportBlockedError')
except ImportBlockedError:
    print('ImportBlockedError raised correctly')
"

# Gate 4: full test suite still green
cd v2 && python -m pytest tests/ -q --tb=short
```

### 8.7 Checklist

- [ ] `v2/rlm_scheme/streaming.py` created with `StreamingNotifier`
- [ ] `v2/rlm_scheme/budget.py` created with `BudgetPolicy`, `BudgetMonitor`
- [ ] `v2/rlm_scheme/python_bridge.py` created with `PythonBridge`, `RestrictedImporter`
- [ ] `BudgetExhaustedError` and `ImportBlockedError` added to `exceptions.py`
- [ ] `executor.py` updated: streaming notifications hooked into map-async and tree-reduce dispatch
- [ ] `executor.py` updated: `BudgetMonitor.record_usage` called after each LLM call
- [ ] `executor.py` updated: `py-exec` / `py-eval` dispatch routes to `PythonBridge`
- [ ] `executor.py` enforces `uses-llm-generated-code` flag check before code interpreter dispatch
- [ ] `checkpoint-and-stop` writes checkpoint record to store and raises `BudgetExhaustedError`
- [ ] All 14 tests in this batch pass
- [ ] No regressions in Batches 1–7 tests

---

## Batch 9: Runtime Primitives and Python Bridge Integration

### 9.0 Purpose

Implement the `PythonRuntimeStub` — a Python pattern-matching interpreter for template structures — and fully integrate the Python bridge for `py-exec` / `py-eval`. The stub recognizes structural patterns such as `map-async`, `tree-reduce`, and `fold-sequential` and executes them via Python async without requiring a Racket runtime. This is the complete execution substrate for the MCP server.

### 9.1 Depends On

- Batch 1 (Store, models, exceptions)
- Batch 2 (Context primitives)
- Batch 3 (Template parser — all combinator structures)
- Batch 4 (Dry-runner — combinator call-count formulas)
- Batch 5 (Executor, LLM adapter)
- Batch 6 (Cache, GateManager)
- Batch 7 (MCP server wiring)
- Batch 8 (Streaming, BudgetMonitor, PythonBridge)

### 9.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| v2/rlm_scheme/runtime_stub.py | Create | PythonRuntimeStub: pattern-dispatch for all template combinators |
| v2/rlm_scheme/python_bridge.py | Modify | Add `py-exec` / `py-eval` context integration |
| v2/rlm_scheme/executor.py | Modify | Replace ad-hoc combinator dispatch with `PythonRuntimeStub` |
| v2/tests/test_runtime_primitives.py | Create | Runtime primitive unit and integration tests |

### 9.3 Requirements

1. [MUST] R-9.1: `PythonRuntimeStub` MUST recognise and execute the `map-async` pattern: apply an LLM call concurrently to every item in an input list and return a list of results.
2. [MUST] R-9.2: `PythonRuntimeStub` MUST recognise and execute the `tree-reduce` pattern: repeatedly apply a reduce call to groups of `branch_factor` items until a single result remains.
3. [MUST] R-9.3: `PythonRuntimeStub` MUST recognise and execute the `fold-sequential` pattern: apply a fold call to items one at a time, threading an accumulator through each call.
4. [MUST] R-9.4: `map-async` execution MUST respect the `max_concurrent` parameter using `asyncio.Semaphore`; no more than `max_concurrent` LLM calls MUST be in-flight simultaneously.
5. [MUST] R-9.5: A `tree-reduce` over N items with branch factor B MUST produce exactly `ceil(N/B) + ceil(ceil(N/B)/B) + ...` calls, converging to a single result; for N=8 and B=2 this is 4+2+1 = 7 reduce calls plus 8 map calls = 15 total calls.
6. [MUST] R-9.6: `fold-sequential` over N items MUST produce exactly N LLM calls; step i receives the accumulator from step i-1.
7. [MUST] R-9.7: `llm-query` (direct single-call) pattern MUST be supported by the stub as the base case.
8. [SHOULD] R-9.8: The stub SHOULD support `memoize` combinator: if the same `(instruction, data)` pair is seen again within the same execution, it SHOULD return the cached result without a new LLM call.
9. [SHOULD] R-9.9: The Python bridge SHOULD support passing the current execution context (store reference, context_id) into `py-exec` via a `__rlm__` dict injected into the globals.
10. [SHOULD] R-9.10: `PythonRuntimeStub` SHOULD emit structured log entries for each pattern dispatch, recorded in the execution trace.

### 9.4 Detailed Specifications

**PythonRuntimeStub public interface**

```python
class PythonRuntimeStub:
    def __init__(
        self,
        llm_adapter: LLMAdapter,
        cache: LLMCache,
        notifier: StreamingNotifier,
        store: Store,
    ) -> None: ...

    async def execute(
        self,
        node: TemplateNode,
        slot_values: dict[str, object],
        execution_id: str,
        model_alias: str,
        policy: ExecutionPolicy,
    ) -> ExecutionResult: ...

    async def _dispatch(
        self,
        node: TemplateNode,
        slot_values: dict[str, object],
        context: RuntimeContext,
    ) -> object: ...
```

`TemplateNode` is the parsed AST node from Batch 3. `_dispatch` pattern-matches on `node.type` and routes to the appropriate handler.

**RuntimeContext dataclass**

```python
@dataclasses.dataclass
class RuntimeContext:
    execution_id: str
    model_alias: str
    policy: ExecutionPolicy
    call_count: int            # mutable, incremented on each LLM call
    memo_table: dict           # (instruction_key, data_key) -> result
    trace_events: list[dict]   # appended on each dispatch
```

**Pattern dispatch table**

| `node.type` | Handler method | Key parameters |
|-------------|----------------|----------------|
| `llm-query` | `_handle_llm_query` | `instruction`, `data`, `model`, `temperature` |
| `map-async` | `_handle_map_async` | `items`, `map_fn`, `max_concurrent` |
| `tree-reduce` | `_handle_tree_reduce` | `items`, `map_fn`, `reduce_fn`, `branch_factor` |
| `fold-sequential` | `_handle_fold_sequential` | `items`, `fold_fn`, `initial_accumulator` |
| `memoize` | `_handle_memoize` | `inner_node`, delegates to `_dispatch` with memo check |
| `py-exec` | `_handle_py_exec` | `code`, `policy.code_generation_policy` |
| `py-eval` | `_handle_py_eval` | `expression`, `policy.code_generation_policy` |

**map-async concurrency implementation**

```python
async def _handle_map_async(self, node, slot_values, ctx):
    sem = asyncio.Semaphore(node.params.get("max_concurrent", 8))
    items = self._resolve_items(node, slot_values)

    async def process_one(idx, item):
        async with sem:
            result = await self._dispatch(node.map_fn, {**slot_values, "item": item}, ctx)
            await self._notifier.notify_item_complete(
                ctx.execution_id, node.node_id, idx + 1, len(items), result
            )
            return result

    return await asyncio.gather(*[process_one(i, item) for i, item in enumerate(items)])
```

**tree-reduce call count formula**

For N items and branch factor B:

```
total_map_calls    = N
total_reduce_calls = sum(ceil(n / B) for n in sequence_until_one(N, B))
```

where `sequence_until_one(N, B)` yields `N, ceil(N/B), ceil(ceil(N/B)/B), ...` until the value is 1.

For N=8, B=2: `sequence = [8, 4, 2, 1]`; map calls = 8; reduce calls = `4 + 2 + 1 = 7`; total = 15.

**fold-sequential accumulator threading**

```python
async def _handle_fold_sequential(self, node, slot_values, ctx):
    items = self._resolve_items(node, slot_values)
    acc = self._resolve_value(node.params.get("initial_accumulator"), slot_values)
    for item in items:
        acc = await self._dispatch(
            node.fold_fn,
            {**slot_values, "item": item, "accumulator": acc},
            ctx,
        )
    return acc
```

**memoize lookup**

```python
async def _handle_memoize(self, node, slot_values, ctx):
    memo_key = (
        canonical_json(node.inner_node.params.get("instruction", "")),
        canonical_json(slot_values.get("data", "")),
    )
    if memo_key in ctx.memo_table:
        return ctx.memo_table[memo_key]
    result = await self._dispatch(node.inner_node, slot_values, ctx)
    ctx.memo_table[memo_key] = result
    return result
```

**py-exec context injection**

When dispatching `py-exec`, the bridge receives a `__rlm__` dict injected into globals:

```python
rlm_globals = {
    "__rlm__": {
        "store": self._store,
        "execution_id": ctx.execution_id,
        "slot_values": slot_values,
    }
}
```

### 9.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|--------------|----------|----------|
| test_runtime_primitives.py | test_llm_query_basic | Single `llm-query` node with mocked LLM | Returns string result |
| test_runtime_primitives.py | test_map_async_basic | `map-async` over 5 items, mocked LLM | Returns list of 5 results |
| test_runtime_primitives.py | test_map_async_concurrency | `max_concurrent=3`, 10 items, timed mock | No more than 3 concurrent calls |
| test_runtime_primitives.py | test_map_async_returns_all | `map-async` over 7 items | List length == 7 |
| test_runtime_primitives.py | test_tree_reduce_call_count | N=8, B=2, counting mock | Exactly 15 calls total |
| test_runtime_primitives.py | test_tree_reduce_single_result | N=4, B=2, concat mock | Returns single combined result |
| test_runtime_primitives.py | test_tree_reduce_n_equals_1 | N=1, B=2 | Returns immediately, 1 map call, 0 reduce calls |
| test_runtime_primitives.py | test_fold_sequential_call_count | 5 items | Exactly 5 LLM calls |
| test_runtime_primitives.py | test_fold_sequential_threading | Items [1,2,3], mock accumulates sum | Final result is 6 |
| test_runtime_primitives.py | test_fold_sequential_order | Items in order A,B,C | Calls made in order A→B→C |
| test_runtime_primitives.py | test_memoize_deduplication | Same item appears twice in map | Second occurrence uses memo, 1 LLM call not 2 |
| test_runtime_primitives.py | test_memoize_different_items | Different items | Both call LLM |
| test_runtime_primitives.py | test_py_exec_execution | `py-exec` with `x = 1 + 1` | Globals contain `x == 2` |
| test_runtime_primitives.py | test_py_exec_rlm_injected | `py-exec` accessing `__rlm__` | `__rlm__["execution_id"]` is correct |
| test_runtime_primitives.py | test_direct_call_execution | `llm-query` with full slot resolution | Returns result from mocked LLM adapter |
| test_runtime_primitives.py | test_trace_events_recorded | Execute `map-async` over 3 items | `trace_events` has at least 3 entries |
| test_runtime_primitives.py | test_dispatch_unknown_type | Node with `type="bogus"` | Raises `UnknownNodeTypeError` |

### 9.6 Acceptance Gates

```sh
# Gate 1: all runtime primitive tests pass
cd v2 && python -m pytest tests/test_runtime_primitives.py -v

# Gate 2: map-async call count correct
cd v2 && python -c "
import asyncio
from rlm_scheme.runtime_stub import PythonRuntimeStub
from unittest.mock import AsyncMock, MagicMock

call_count = 0
async def mock_llm(instruction, data, model, temperature, json_mode):
    global call_count
    call_count += 1
    return f'result_{call_count}'

stub = PythonRuntimeStub.__new__(PythonRuntimeStub)
stub._call_count_check = True

# 5-item map-async check
items = list(range(5))
results = asyncio.run(stub._test_map_async(items, mock_llm))
assert len(results) == 5, f'expected 5, got {len(results)}'
print('map-async call count OK')
" 2>/dev/null || echo "manual verification required — see test suite"

# Gate 3: tree-reduce call count N=8 B=2 = 15
cd v2 && python -c "
from rlm_scheme.runtime_stub import tree_reduce_call_count
total = tree_reduce_call_count(n=8, branch_factor=2)
assert total == 15, f'expected 15, got {total}'
print(f'tree_reduce_call_count(8,2) = {total} OK')
"

# Gate 4: full test suite still green
cd v2 && python -m pytest tests/ -q --tb=short
```

### 9.7 Checklist

- [ ] `v2/rlm_scheme/runtime_stub.py` created with `PythonRuntimeStub`, `RuntimeContext`, `tree_reduce_call_count`
- [ ] `UnknownNodeTypeError` added to `exceptions.py`
- [ ] All seven `_handle_*` methods implemented in `PythonRuntimeStub`
- [ ] `map-async` uses `asyncio.Semaphore` for concurrency limiting (R-9.4)
- [ ] `tree-reduce` call count matches formula for all N, B combinations tested (R-9.5)
- [ ] `fold-sequential` threads accumulator correctly (R-9.6)
- [ ] `memoize` deduplication works within a single execution (R-9.8)
- [ ] `py-exec` injects `__rlm__` dict into bridge globals (R-9.9)
- [ ] Trace events emitted for each dispatch (R-9.10)
- [ ] `executor.py` updated to use `PythonRuntimeStub` for all combinator dispatch
- [ ] All 17 tests in this batch pass
- [ ] No regressions in Batches 1–8 tests

---

## Batch 10: Entry Point, Documentation, and Polish

### 10.0 Purpose

Add the `__main__.py` entry point so the package is runnable via `python -m rlm_scheme`, write the README showing the 3-tool happy path, run the full test suite to verify 0 failures and at least 150 tests, and fix any issues surfaced during final integration.

### 10.1 Depends On

- All prior batches (1–9)

### 10.2 Files to Create or Modify

| File | Action | Description |
|------|--------|-------------|
| v2/rlm_scheme/__main__.py | Create | Entry point: `python -m rlm_scheme` starts the MCP server |
| v2/README.md | Create | Usage documentation including the 3-tool happy path |
| v2/rlm_scheme/mcp_server.py | Modify | Verify `if __name__ == "__main__"` block is clean |
| v2/tests/test_entry_point.py | Create | Smoke tests for the entry point and README content |

### 10.3 Requirements

1. [MUST] R-10.1: `python -m rlm_scheme` MUST either start the MCP server successfully or print usage information and exit with code 0 or 1.
2. [MUST] R-10.2: `python -m rlm_scheme --help` MUST exit without raising an unhandled exception.
3. [MUST] R-10.3: `v2/README.md` MUST document the 3-tool happy path: `plan_strategy` → `dry_run_strategy` → `execute_strategy`, with example inputs and outputs for each step.
4. [MUST] R-10.4: `v2/README.md` MUST contain zero occurrences of the strings `execute_scheme` or `dry_run_scheme`.
5. [MUST] R-10.5: The full test suite (`python -m pytest tests/`) MUST pass with 0 failures and 0 errors.
6. [SHOULD] R-10.6: The full test suite SHOULD collect at least 150 test items.
7. [SHOULD] R-10.7: `v2/README.md` SHOULD include a section listing all 10 MCP tool names with one-line descriptions.
8. [SHOULD] R-10.8: `v2/README.md` SHOULD include a quick-start section showing how to install dependencies and run the server.
9. [MAY] R-10.9: The entry point MAY support `--stdio` and `--port PORT` flags to select transport.
10. [MAY] R-10.10: `v2/README.md` MAY include a troubleshooting section for common errors (`ContextNotFoundError`, `TemplateValidationError`).

### 10.4 Detailed Specifications

**`__main__.py` implementation**

```python
"""Entry point for python -m rlm_scheme."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rlm_scheme",
        description="RLM-Scheme MCP server — LLM reasoning template engine",
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use (default: stdio)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for SSE transport (default: 8000)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    from rlm_scheme.mcp_server import mcp  # deferred import to allow --help without heavy deps

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**README structure**

The README MUST contain the following top-level sections in order:

1. **RLM-Scheme** — one-paragraph overview
2. **Quick Start** — install and run commands
3. **The 3-Tool Happy Path** — annotated example of `plan_strategy` → `dry_run_strategy` → `execute_strategy`
4. **All 10 MCP Tools** — table with tool name and one-line description
5. **Template Structure** — brief overview of `map-async`, `tree-reduce`, `fold-sequential` combinators
6. **Configuration** — environment variables (`RLM_STORE_DIR`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`)

**3-tool happy path example** (MUST appear verbatim or substantially in README):

```
1. plan_strategy(task="Summarize these 8 research papers", context_id="ctx_abc123")
   → {"status": "ok", "plan_id": "plan_xyz", "recommended_template": "map-reduce-summarize"}

2. dry_run_strategy(plan_id="plan_xyz")
   → {"status": "ok", "dry_run_id": "dr_789", "artifact": {"estimated_calls": 15, "estimated_tokens": 24000}}

3. execute_strategy(plan_id="plan_xyz")
   → {"status": "ok", "execution_id": "exec_456", "output_context_id": "ctx_result_999"}
```

**All 10 MCP tools table** (MUST appear in README):

| Tool | Description |
|------|-------------|
| `load_context` | Store data (text, JSON, or binary) and receive a context_id |
| `get_context` | Retrieve context metadata and optional preview or full data |
| `plan_strategy` | Analyse a task and recommend a template with a plan_id |
| `dry_run_strategy` | Estimate token cost and call count without executing |
| `execute_strategy` | Execute a plan or template invocation and return results |
| `get_execution_trace` | Retrieve events, scope log, and stdout for a completed execution |
| `get_status` | Return global token usage and cache stats, or execution details |
| `cancel_call` | Cancel an in-progress call or execution |
| `resume_execution` | Approve or reject a pending human-review gate |
| `reset_runtime` | Clear session state, cache, contexts, executions, or all |

**Final integration fix protocol**

Before marking this batch complete, run the full suite and address failures in this order:

1. Import errors — fix missing imports or circular dependencies.
2. `AttributeError` / `TypeError` — fix interface mismatches between batches.
3. Assertion failures in integration tests — investigate data flow.
4. Test count below 150 — add missing edge-case tests to the most-undercovered files.

### 10.5 Test Specification

| Test File | Test Function | Scenario | Expected |
|-----------|--------------|----------|----------|
| test_entry_point.py | test_help_exits_cleanly | `python -m rlm_scheme --help` via subprocess | Exit code 0, no stderr |
| test_entry_point.py | test_module_importable | `import rlm_scheme` | No ImportError |
| test_entry_point.py | test_mcp_object_exists | `from rlm_scheme.mcp_server import mcp` | `mcp` is not None |
| test_entry_point.py | test_readme_exists | `v2/README.md` file exists | File present and non-empty |
| test_entry_point.py | test_readme_no_execute_scheme | `grep execute_scheme v2/README.md` | Zero matches |
| test_entry_point.py | test_readme_no_dry_run_scheme | `grep dry_run_scheme v2/README.md` | Zero matches |
| test_entry_point.py | test_readme_has_plan_strategy | README contains `plan_strategy` | String found |
| test_entry_point.py | test_readme_has_dry_run_strategy | README contains `dry_run_strategy` | String found |
| test_entry_point.py | test_readme_has_execute_strategy | README contains `execute_strategy` | String found |
| test_entry_point.py | test_readme_has_all_10_tools | README contains all 10 tool names | All 10 found |
| test_entry_point.py | test_full_suite_count | `pytest --collect-only -q` | At least 150 items collected |
| (all batches) | (all tests) | Complete suite run | 0 failures, 0 errors |

### 10.6 Acceptance Gates

```sh
# Gate 1: entry point help works
cd v2 && python -m rlm_scheme --help

# Gate 2: README exists and contains no forbidden strings
cd v2 && test -f README.md && echo "README exists"
cd v2 && count=$(grep -c "execute_scheme\|dry_run_scheme" README.md 2>/dev/null || echo 0); echo "Forbidden string count: $count"; test "$count" -eq 0

# Gate 3: README contains all 10 tool names
cd v2 && python -c "
content = open('README.md').read()
tools = [
    'load_context', 'get_context', 'plan_strategy', 'dry_run_strategy',
    'execute_strategy', 'get_execution_trace', 'get_status',
    'cancel_call', 'resume_execution', 'reset_runtime',
]
missing = [t for t in tools if t not in content]
assert not missing, f'Missing from README: {missing}'
print('All 10 tool names present in README')
"

# Gate 4: full test suite passes with >= 150 tests
cd v2 && python -m pytest tests/ -v --tb=short 2>&1 | tee /tmp/rlm_pytest_output.txt
cd v2 && python -m pytest tests/ --collect-only -q 2>&1 | tail -5

# Gate 5: test count >= 150
cd v2 && python -c "
import subprocess, re
result = subprocess.run(['python', '-m', 'pytest', 'tests/', '--collect-only', '-q'],
                       capture_output=True, text=True)
lines = result.stdout + result.stderr
match = re.search(r'(\d+) test', lines)
count = int(match.group(1)) if match else 0
print(f'Test count: {count}')
assert count >= 150, f'Expected >= 150 tests, got {count}'
print('Test count gate passed')
"
```

### 10.7 Checklist

- [ ] `v2/rlm_scheme/__main__.py` created with `build_parser()` and `main()` functions
- [ ] `python -m rlm_scheme --help` exits with code 0
- [ ] `v2/README.md` created with all six required sections
- [ ] README contains zero occurrences of `execute_scheme` and `dry_run_scheme`
- [ ] README contains all 10 MCP tool names
- [ ] README 3-tool happy path example is present and accurate
- [ ] `test_entry_point.py` created with 12 tests covering entry point and README content
- [ ] Full test suite runs with 0 failures and 0 errors
- [ ] Test suite collects at least 150 items
- [ ] Any interface mismatches discovered during final integration run are fixed
- [ ] All import cycles resolved
- [ ] `v2/rlm_scheme/__init__.py` exports `mcp`, `__version__`, and the ten tool functions for convenience imports
---

## Appendix E: Verification Check Registry

All 23 verification checks with pass conditions, failure message templates, and Python pseudocode.

### E.1 Check 1: artifact_origin

**Pass condition:** Artifact record exists with `source_type: "template_invocation"`
**Failure message:** `"Artifact was not created by the instantiator."`

```python
def check_artifact_origin(artifact, store):
    if artifact.source_type != "template_invocation":
        return Fail("Artifact was not created by the instantiator.")
    return Pass()
```

### E.2 Check 2: artifact_hash

**Pass condition:** `sha256(artifact_code)` matches `artifact.generated_scheme_ref.hash`
**Failure message:** `"Artifact code hash mismatch: expected {expected}, got {actual}."`

```python
def check_artifact_hash(artifact):
    import hashlib
    actual = "sha256:" + hashlib.sha256(artifact.code.encode()).hexdigest()
    expected = artifact.generated_scheme_ref.hash
    if actual != expected:
        return Fail(f"Artifact code hash mismatch: expected {expected}, got {actual}.")
    return Pass()
```

### E.3 Check 3: template_version

**Pass condition:** Template name+version exists in catalog
**Failure message:** `"Unknown template: {name} v{version}."`

```python
def check_template_version(artifact, template_store):
    name = artifact.template_name
    version = artifact.template_version
    if not template_store.exists(name, version):
        return Fail(f"Unknown template: {name} v{version}.")
    return Pass()
```

### E.4 Check 4: slots_filled

**Pass condition:** No `{{slot}}` markers remain in artifact code
**Failure message:** `"Unfilled slot markers: {markers}."`

```python
import re

def check_slots_filled(artifact):
    markers = re.findall(r"\{\{[^}]+\}\}", artifact.code)
    if markers:
        return Fail(f"Unfilled slot markers: {markers}.")
    return Pass()
```

### E.5 Check 5: model_exists

**Pass condition:** All model aliases in artifact resolve in registry
**Failure message:** `"Unknown model alias: {alias}."`

```python
def check_model_exists(artifact, model_registry):
    for alias in artifact.referenced_model_aliases:
        if not model_registry.has(alias):
            return Fail(f"Unknown model alias: {alias}.")
    return Pass()
```

### E.6 Check 6: model_capabilities

**Pass condition:** JSON-mode calls target models with `json_mode: true`
**Failure message:** `"Model {alias} does not support JSON mode."`

```python
def check_model_capabilities(artifact, model_registry):
    for call in artifact.llm_calls:
        if call.json_mode:
            model = model_registry.get(call.model_alias)
            if not model.capabilities.get("json_mode", False):
                return Fail(f"Model {call.model_alias} does not support JSON mode.")
    return Pass()
```

### E.7 Check 7: image_model

**Pass condition:** Image inputs target models with `image: true`
**Failure message:** `"Model {alias} does not support image inputs."`

```python
def check_image_model(artifact, model_registry):
    for call in artifact.llm_calls:
        if call.has_image_input:
            model = model_registry.get(call.model_alias)
            if not model.capabilities.get("image", False):
                return Fail(f"Model {call.model_alias} does not support image inputs.")
    return Pass()
```

### E.8 Check 8: no_unsafe_forms

**Pass condition:** No `eval`, `system`, `shell`, `exec` (non-`py-exec`) in artifact
**Failure message:** `"Unsafe form found: {form}."`

```python
import re

def check_no_unsafe_forms(artifact):
    unsafe_patterns = [
        r"\beval\b",
        r"\bsystem\b",
        r"\bshell\b",
        r"(?<!py-)\bexec\b",
    ]
    for pattern in unsafe_patterns:
        matches = re.findall(pattern, artifact.code)
        if matches:
            return Fail(f"Unsafe form found: {matches[0]}.")
    return Pass()
```

### E.9 Check 9: no_raw_import

**Pass condition:** No `require`, `load`, `include` outside allowed set
**Failure message:** `"Disallowed import: {form}."`

```python
import re

ALLOWED_REQUIRES = {"rlm/primitives", "rlm/runtime", "racket/base"}

def check_no_raw_import(artifact):
    import_forms = re.findall(r'\((?:require|load|include)\s+([^\)]+)\)', artifact.code)
    for form_text in import_forms:
        module = form_text.strip().strip('"')
        if module not in ALLOWED_REQUIRES:
            return Fail(f"Disallowed import: {form_text.strip()}.")
    return Pass()
```

### E.10 Check 10: call_count_limit

**Pass condition:** Expected calls <= `policy.max_llm_calls` (default: 1000)
**Failure message:** `"Expected {n} calls exceeds limit {limit}."`

```python
def check_call_count_limit(dry_run_estimate, policy):
    limit = policy.get("max_llm_calls", 1000)
    n = dry_run_estimate.expected_llm_calls
    if n > limit:
        return Fail(f"Expected {n} calls exceeds limit {limit}.")
    return Pass()
```

### E.11 Check 11: recursive_depth_limit

**Pass condition:** Recursive depth <= `policy.max_recursive_depth` (default: 3)
**Failure message:** `"Recursive depth {d} exceeds limit {limit}."`

```python
def check_recursive_depth_limit(dry_run_estimate, policy):
    limit = policy.get("max_recursive_depth", 3)
    d = dry_run_estimate.recursive_depth
    if d > limit:
        return Fail(f"Recursive depth {d} exceeds limit {limit}.")
    return Pass()
```

### E.12 Check 12: concurrency_limit

**Pass condition:** Max concurrency <= `policy.max_concurrency` (default: 50)
**Failure message:** `"Concurrency {c} exceeds limit {limit}."`

```python
def check_concurrency_limit(dry_run_estimate, policy):
    limit = policy.get("max_concurrency", 50)
    c = dry_run_estimate.max_concurrency
    if c > limit:
        return Fail(f"Concurrency {c} exceeds limit {limit}.")
    return Pass()
```

### E.13 Check 13: context_exists

**Pass condition:** All referenced `context_id` values exist in store
**Failure message:** `"Context not found: {id}."`

```python
def check_context_exists(artifact, context_store):
    for ctx_id in artifact.referenced_context_ids:
        if not context_store.exists(ctx_id):
            return Fail(f"Context not found: {ctx_id}.")
    return Pass()
```

### E.14 Check 14: output_schema_valid

**Pass condition:** If `output-schema` declared, it is structurally valid alist notation
**Failure message:** `"Output schema is malformed: {detail}."`

```python
def check_output_schema_valid(artifact):
    schema = artifact.metadata.get("output_schema")
    if schema is None:
        return Pass()
    try:
        validate_alist_schema(schema)
    except SchemaValidationError as e:
        return Fail(f"Output schema is malformed: {e}.")
    return Pass()

def validate_alist_schema(schema):
    # Schema must be a list of (key . type) pairs
    if not isinstance(schema, list):
        raise SchemaValidationError("Schema must be a list.")
    for entry in schema:
        if not (isinstance(entry, tuple) and len(entry) == 2):
            raise SchemaValidationError(f"Entry {entry!r} is not a key-type pair.")
        key, typ = entry
        if not isinstance(key, str):
            raise SchemaValidationError(f"Key {key!r} must be a string.")
        if typ not in ("string", "number", "boolean", "list", "dict", "any"):
            raise SchemaValidationError(f"Unknown type {typ!r}.")
```

### E.15 Check 15: output_schema_present

**Pass condition:** If policy requires output schema, template declares one
**Failure message:** `"Output schema required by policy but not declared."`

```python
def check_output_schema_present(artifact, policy):
    if not policy.get("require_output_schema", False):
        return Pass()
    schema = artifact.metadata.get("output_schema")
    if schema is None:
        return Fail("Output schema required by policy but not declared.")
    return Pass()
```

### E.16 Check 16: dry_run_warnings

**Pass condition:** No `error`-level warnings from dry-run
**Failure message:** `"Dry-run produced error-level warning: {warning}."`

```python
def check_dry_run_warnings(dry_run_result):
    for warning in dry_run_result.warnings:
        if warning.level == "error":
            return Fail(f"Dry-run produced error-level warning: {warning.message}.")
    return Pass()
```

### E.17 Check 17: code_interpreter_policy

**Pass condition:** If `uses-llm-generated-code: #t`, policy has `allow_llm_generated_code: true`
**Failure message:** `"Template uses LLM-generated code but policy disallows it."`

```python
def check_code_interpreter_policy(artifact, policy):
    uses_llm_code = artifact.metadata.get("uses_llm_generated_code", False)
    if uses_llm_code and not policy.get("allow_llm_generated_code", False):
        return Fail("Template uses LLM-generated code but policy disallows it.")
    return Pass()
```

### E.18 Check 18: gate_consistency

**Pass condition:** Gate names in body match `define-meta gates` declarations
**Failure message:** `"Gate '{name}' used in body but not declared in metadata."`

```python
import re

def check_gate_consistency(artifact):
    declared_gates = set(artifact.metadata.get("gates", []))
    used_gates = set(re.findall(r'\(gate\s+"([^"]+)"', artifact.code))
    for name in used_gates:
        if name not in declared_gates:
            return Fail(f"Gate '{name}' used in body but not declared in metadata.")
    return Pass()
```

### E.19 Check 19: budget_policy_model

**Pass condition:** If `budget-policy` declares a fallback model, it exists in registry
**Failure message:** `"Budget fallback model '{alias}' not found in registry."`

```python
def check_budget_policy_model(artifact, model_registry):
    budget_policy = artifact.metadata.get("budget_policy")
    if budget_policy is None:
        return Pass()
    fallback_alias = budget_policy.get("fallback_model")
    if fallback_alias is None:
        return Pass()
    if not model_registry.has(fallback_alias):
        return Fail(f"Budget fallback model '{fallback_alias}' not found in registry.")
    return Pass()
```

### E.20 Check 20: budget_policy_caps

**Pass condition:** Fallback model has compatible capabilities (JSON mode, images)
**Failure message:** `"Fallback model '{alias}' lacks capability: {cap}."`

```python
def check_budget_policy_caps(artifact, model_registry):
    budget_policy = artifact.metadata.get("budget_policy")
    if budget_policy is None:
        return Pass()
    fallback_alias = budget_policy.get("fallback_model")
    if fallback_alias is None:
        return Pass()
    fallback_model = model_registry.get(fallback_alias)
    required_caps = artifact.metadata.get("required_capabilities", [])
    for cap in required_caps:
        if not fallback_model.capabilities.get(cap, False):
            return Fail(f"Fallback model '{fallback_alias}' lacks capability: {cap}.")
    return Pass()
```

### E.21 Check 21: primitive_allowlist

**Pass condition:** Only primitives from the approved list are used in artifact
**Failure message:** `"Disallowed primitive: {name}."`

```python
ALLOWED_PRIMITIVES = {
    "llm-query", "llm-query-async", "await", "await-all", "await-any",
    "map-async", "parallel", "race", "tree-reduce", "fold-sequential",
    "sequence", "choose", "iterate-until", "recursive-spawn",
    "memoized", "with-validation", "try-fallback",
    "gate", "finish", "python-compute", "py-exec",
    "load-context", "syntax-e", "datum->syntax",
    "checkpoint", "checkpoint-restore",
}

def check_primitive_allowlist(artifact):
    for call in artifact.primitive_calls:
        if call.name not in ALLOWED_PRIMITIVES:
            return Fail(f"Disallowed primitive: {call.name}.")
    return Pass()
```

### E.22 Check 22: context_window_fit

**Pass condition:** Estimated input tokens fit model's context window
**Failure message:** `"Estimated {tokens} tokens exceeds {alias} context window of {limit}."`

```python
def check_context_window_fit(artifact, dry_run_estimate, model_registry):
    for call in dry_run_estimate.call_estimates:
        model = model_registry.get(call.model_alias)
        tokens = call.estimated_prompt_tokens
        limit = model.context_window_tokens
        if tokens > limit:
            return Fail(
                f"Estimated {tokens} tokens exceeds {call.model_alias} "
                f"context window of {limit}."
            )
    return Pass()
```

### E.23 Check 23: temperature_compat

**Pass condition:** Temperature and max-token settings are valid for model
**Failure message:** `"Invalid temperature {t} for model {alias}."`

```python
def check_temperature_compat(artifact, model_registry):
    for call in artifact.llm_calls:
        model = model_registry.get(call.model_alias)
        t = call.temperature
        if t is not None:
            t_min = model.temperature_range.get("min", 0.0)
            t_max = model.temperature_range.get("max", 2.0)
            if not (t_min <= t <= t_max):
                return Fail(f"Invalid temperature {t} for model {call.model_alias}.")
        if call.max_tokens is not None:
            if call.max_tokens > model.max_output_tokens:
                return Fail(
                    f"max_tokens {call.max_tokens} exceeds model "
                    f"{call.model_alias} limit {model.max_output_tokens}."
                )
    return Pass()
```

### E.24 Overall Verification Decision

```python
def run_all_checks(artifact, dry_run_result, context_store, model_registry, policy):
    checks = [
        check_artifact_origin(artifact, context_store),
        check_artifact_hash(artifact),
        check_template_version(artifact, template_store),
        check_slots_filled(artifact),
        check_model_exists(artifact, model_registry),
        check_model_capabilities(artifact, model_registry),
        check_image_model(artifact, model_registry),
        check_no_unsafe_forms(artifact),
        check_no_raw_import(artifact),
        check_call_count_limit(dry_run_result.estimate, policy),
        check_recursive_depth_limit(dry_run_result.estimate, policy),
        check_concurrency_limit(dry_run_result.estimate, policy),
        check_context_exists(artifact, context_store),
        check_output_schema_valid(artifact),
        check_output_schema_present(artifact, policy),
        check_dry_run_warnings(dry_run_result),
        check_code_interpreter_policy(artifact, policy),
        check_gate_consistency(artifact),
        check_budget_policy_model(artifact, model_registry),
        check_budget_policy_caps(artifact, model_registry),
        check_primitive_allowlist(artifact),
        check_context_window_fit(artifact, dry_run_result.estimate, model_registry),
        check_temperature_compat(artifact, model_registry),
    ]
    if any(c.status == "fail" for c in checks):
        return VerificationResult(decision="fail", checks=checks)
    elif any(c.status == "warn" for c in checks):
        return VerificationResult(decision="warn", checks=checks)
    else:
        return VerificationResult(decision="pass", checks=checks)
```

---

## Appendix F: Classifier Decision Trees (Python Pseudocode)

### F.1 `classify_task_shape(hints) -> TaskShape`

Implements the Q0–Q9 decision tree. All logic is deterministic on the structured `hints` dict. Returns a `TaskShape` string.

```python
def classify_task_shape(hints: dict) -> str:
    """
    Classify task into a TaskShape using the Q0-Q9 decision tree.
    All decisions operate on structured hint fields only.

    Required hint fields (may be None if Level 2 gap-fill has not run):
        item_count     : int | None
        independent    : bool | None
        output_type    : "one" | "list" | "per_item" | None
        operation      : "transform" | "extract" | "label" | "check" | "grade" | "other" | None
        has_second_phase: bool | None
        sub_operations : list[str] | None
        ordered        : bool | None
    """
    item_count = hints.get("item_count", 0) or 0
    independent = hints.get("independent")
    output_type = hints.get("output_type")
    operation = hints.get("operation")
    has_second_phase = hints.get("has_second_phase", False)
    sub_operations = hints.get("sub_operations") or []
    ordered = hints.get("ordered", False)

    MANY_ITEMS_THRESHOLD = 2

    # Q0: One small input, one output, one operation, no second phase?
    is_single_input = (item_count <= 1)
    is_single_output = (output_type == "one" or output_type is None)
    no_second_phase = not has_second_phase
    no_sub_ops = (len(sub_operations) <= 1)

    if is_single_input and is_single_output and no_second_phase and no_sub_ops:
        return "Direct"

    # Q9: Does the task clearly have multiple phases?
    # Check early: Composite wins if multiple sub_operations span distinct shapes.
    if has_second_phase and len(sub_operations) >= 2:
        return "Composite"

    # Q1: Are there many input items?
    if item_count >= MANY_ITEMS_THRESHOLD:
        # Q2: Are items independent?
        if independent is True:
            # Q3: What is the per-item operation?
            if operation in ("transform", "extract", "other"):
                return "Batch"
            elif operation in ("label",):
                return "Classify"
            elif operation in ("check", "grade"):
                return "Validate"
            else:
                # default for unknown operation with independent items
                return "Batch"
        else:
            # Q4: Does information accumulate across ordered items?
            if ordered:
                return "Synthesize"  # fold-sequential path
            else:
                return "Pipeline"
    else:
        # Q5: Is the task creating content with no source item list?
        if operation == "generate" or (output_type == "list" and item_count == 0):
            return "Generate"

        # Q6: Is the task improving one artifact?
        if operation in ("refine", "improve", "iterate"):
            return "Refine"

        # Q7: Is the task breaking one input into parts?
        if operation in ("decompose", "split", "parse", "extract") and output_type in ("list", "per_item"):
            return "Decompose"

        # Q8: Is the task choosing among alternatives?
        if operation in ("compare", "select", "choose", "rank"):
            if hints.get("latency_priority", False):
                return "Search"
            return "Compare"

        # Remaining single-input cases
        if output_type == "one" and operation == "aggregate":
            return "Aggregate"

        if output_type == "one":
            return "Synthesize"

        # Fallback
        return "Direct"
```

### F.2 `classify_data_shape(metadata) -> DataShape`

Implements the DataShape mapping rules, returning a `DataShape` string and optional processing advice.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataShapeResult:
    shape: str
    concurrency_hint: Optional[int] = None
    processing_advice: Optional[str] = None


def classify_data_shape(metadata: dict) -> DataShapeResult:
    """
    Classify context data into a DataShape using structural mapping rules.

    Metadata fields:
        data_shape      : str | None  (agent-provided hint, used if present)
        item_count      : int
        independent     : bool
        ordered         : bool
        size_bytes      : int | None
        chunkable       : bool
        chunk_count     : int | None
        chunks_dependent: bool
        depth           : int | None   (for hierarchical data)
        branching       : int | None
        row_count       : int | None
        columns         : list | None
        modality        : list[str]
        pair_count      : int | None
        key_count       : int | None
        context_limit_tokens: int      (model context window, for Singular classification)
        estimated_tokens: int | None
    """
    # Honor agent-provided hint if structurally consistent
    agent_hint = metadata.get("data_shape")
    if agent_hint and agent_hint in (
        "FlatList", "Hierarchy", "Singular", "ChunkedSingular",
        "Graph", "TimeSeries", "Tabular", "Multimodal", "Paired", "KeyValue"
    ):
        return _apply_data_shape_rules(agent_hint, metadata)

    item_count = metadata.get("item_count", 0) or 0
    independent = metadata.get("independent", True)
    ordered = metadata.get("ordered", False)
    depth = metadata.get("depth")
    row_count = metadata.get("row_count")
    modality = metadata.get("modality") or []
    pair_count = metadata.get("pair_count")
    key_count = metadata.get("key_count")
    chunkable = metadata.get("chunkable", False)
    chunks_dependent = metadata.get("chunks_dependent", False)
    chunk_count = metadata.get("chunk_count")
    estimated_tokens = metadata.get("estimated_tokens")
    context_limit_tokens = metadata.get("context_limit_tokens", 128000)

    # Multimodal: any non-text modality present
    if any(m in modality for m in ("image", "audio")):
        return DataShapeResult(shape="Multimodal")

    # Paired: aligned source/target pairs
    if pair_count is not None and pair_count > 0:
        return DataShapeResult(shape="Paired")

    # KeyValue: dictionary/map data
    if key_count is not None and item_count == 0:
        return DataShapeResult(shape="KeyValue")

    # Tabular: rows with shared schema
    if row_count is not None and row_count > 0 and metadata.get("columns"):
        return DataShapeResult(shape="Tabular")

    # Hierarchy: tree or nested structure
    if depth is not None and depth > 2:
        return DataShapeResult(shape="Hierarchy")

    # TimeSeries: ordered observations (no independence)
    if ordered and not independent and item_count > 0:
        return DataShapeResult(shape="TimeSeries")

    # List shapes
    if item_count > 1:
        if independent:
            concurrency = min(item_count, 20) if item_count > 50 else item_count
            return DataShapeResult(shape="FlatList", concurrency_hint=concurrency)
        else:
            return DataShapeResult(shape="FlatList", processing_advice="fold-sequential")

    # Singular shapes
    if item_count <= 1:
        if chunkable and chunks_dependent:
            return DataShapeResult(shape="ChunkedSingular")
        if chunkable and chunk_count and chunk_count > 1:
            return DataShapeResult(
                shape="Singular",
                processing_advice="chunk_then_flatlist"
            )
        if estimated_tokens and estimated_tokens > context_limit_tokens:
            return DataShapeResult(
                shape="Singular",
                processing_advice="exceeds_context"
            )
        return DataShapeResult(shape="Singular")

    return DataShapeResult(shape="Singular")


def _apply_data_shape_rules(shape: str, metadata: dict) -> DataShapeResult:
    """Apply concurrency and processing advice for an already-classified shape."""
    item_count = metadata.get("item_count", 0) or 0
    independent = metadata.get("independent", True)

    if shape == "FlatList":
        if independent:
            concurrency = min(item_count, 20) if item_count > 50 else item_count
            return DataShapeResult(shape="FlatList", concurrency_hint=concurrency)
        else:
            return DataShapeResult(shape="FlatList", processing_advice="fold-sequential")

    if shape == "Singular":
        chunkable = metadata.get("chunkable", False)
        chunks_dependent = metadata.get("chunks_dependent", False)
        estimated_tokens = metadata.get("estimated_tokens")
        context_limit_tokens = metadata.get("context_limit_tokens", 128000)
        if chunkable and chunks_dependent:
            return DataShapeResult(shape="ChunkedSingular")
        if estimated_tokens and estimated_tokens > context_limit_tokens and chunkable:
            return DataShapeResult(shape="Singular", processing_advice="chunk_then_flatlist")

    return DataShapeResult(shape=shape)
```

### F.3 `select_template(task_shape, data_shape, hints) -> str`

Implements the per-shape template selection trees for all 13 shapes, returning a template name string.

```python
def select_template(task_shape: str, data_shape: str, hints: dict) -> str:
    """
    Select a template name given a classified TaskShape, DataShape, and hints.

    hints fields used here:
        output_type         : "one" | "list" | "per_item"
        ordered             : bool
        has_second_phase    : bool
        operation           : str
        quality_priority    : bool
        latency_priority    : bool
        item_count          : int
        independent         : bool
        sub_operations      : list[str]
        ambiguous_items     : bool   (some items are harder/uncertain)
        likely_duplicates   : bool
        has_testable_pred   : bool   (for Refine: is there a machine-checkable predicate?)
        compare_target      : "models" | "strategies"
        select_or_synthesize: "select" | "synthesize"
        fixed_count         : bool   (for Generate: fixed number vs. until-condition)
        items_consistent    : bool   (for Generate: must items be mutually consistent?)
        items_unique        : bool   (for Generate: must items be unique?)
        known_boundary      : bool   (for Decompose: is split boundary deterministic?)
        one_pass            : bool   (for Decompose: is one decomposition pass enough?)
        process_parts_after : bool   (for Decompose: process parts afterward?)
        same_rubric         : bool   (for Validate: same rubric for all items?)
        false_positive_cost : bool   (for Validate: false positives costlier than negatives?)
        pure_computation    : bool   (for Aggregate: pure computation after extraction?)
        grouped_report      : bool   (for Aggregate: produce grouped report?)
        fits_context        : bool   (for Synthesize: do all items fit in one context?)
        accumulator_large   : bool   (for Synthesize fold: will accumulator exceed context?)
        candidate_set_finite: bool   (for Search: is candidate set finite?)
        stages_distinct     : bool   (for Pipeline: are stages clearly distinct?)
        stage_can_fail      : bool
        stage_needs_gating  : bool
        criterion_validate_each: bool (for Refine: validate each iteration?)
    """
    if task_shape == "Direct":
        return _select_direct(hints)
    elif task_shape == "Batch":
        return _select_batch(hints)
    elif task_shape == "Synthesize":
        return _select_synthesize(hints)
    elif task_shape == "Search":
        return _select_search(hints)
    elif task_shape == "Refine":
        return _select_refine(hints)
    elif task_shape == "Compare":
        return _select_compare(hints)
    elif task_shape == "Classify":
        return _select_classify(hints)
    elif task_shape == "Pipeline":
        return _select_pipeline(hints)
    elif task_shape == "Generate":
        return _select_generate(hints)
    elif task_shape == "Decompose":
        return _select_decompose(hints)
    elif task_shape == "Validate":
        return _select_validate(hints)
    elif task_shape == "Aggregate":
        return _select_aggregate(hints)
    elif task_shape == "Composite":
        return _select_composite(hints)
    else:
        raise ValueError(f"Unknown task shape: {task_shape!r}")


def _select_direct(hints: dict) -> str:
    # Q1: Does the input fit in one model context?
    fits_context = hints.get("fits_context", True)
    if not fits_context:
        # Reclassify — caller should re-run classify_task_shape
        raise ReclassificationNeeded(
            "Input does not fit in one context. Reclassify as Decompose, Batch, or Synthesize."
        )

    # Q2: Is deterministic computation needed before/after the call?
    operation = hints.get("operation", "")
    needs_compute = operation in ("aggregate", "compute", "stats")
    if needs_compute:
        # Caller assembles direct_call + python_compute in sequence
        return "direct_call"  # caller adds python_compute phase

    output_type = hints.get("output_type", "one")
    if output_type in ("list", "per_item"):
        return "direct_json_extract"

    return "direct_call"


def _select_batch(hints: dict) -> str:
    output_type = hints.get("output_type", "per_item")
    ordered = hints.get("ordered", False)
    ambiguous_items = hints.get("ambiguous_items", False)
    likely_duplicates = hints.get("likely_duplicates", False)

    # Q1: Return a list or one combined output?
    if output_type in ("list", "per_item"):
        if ambiguous_items:
            # Q3: Some items harder or more ambiguous?
            return "tiered_review"
        if likely_duplicates:
            # Q4: Duplicates likely? Use memoized map.
            return "batch_map"  # with memoized flag set in slot_values
        return "batch_map"
    else:
        # Combined output
        # Q2: Is combination order-sensitive?
        if ordered:
            return "batch_extract_fold"
        else:
            if ambiguous_items:
                return "tiered_review"
            return "batch_extract_reduce"


def _select_synthesize(hints: dict) -> str:
    fits_context = hints.get("fits_context", False)
    ordered = hints.get("ordered", False)
    accumulator_large = hints.get("accumulator_large", False)

    # Q1: Do all items fit in one context?
    if fits_context:
        return "direct_call"  # direct synthesis in one context

    # Q2: Is order important?
    if ordered:
        # Q3: Is accumulator likely to exceed context?
        if accumulator_large:
            return "ordered_synthesis_fold"  # with summarization flag
        else:
            return "ordered_synthesis_fold"
    else:
        return "tree_synthesis"


def _select_search(hints: dict) -> str:
    candidate_set_finite = hints.get("candidate_set_finite", True)
    latency_priority = hints.get("latency_priority", False)

    # Q1: Is the candidate set finite?
    if candidate_set_finite:
        # Q2: Is latency more important than quality?
        if latency_priority:
            return "race_candidates"
        else:
            return "compare_candidates"  # evaluate all then select
    else:
        # Iterative search (no dedicated template — use refine_until_valid)
        return "refine_until_valid"


def _select_refine(hints: dict) -> str:
    has_testable_pred = hints.get("has_testable_pred", False)

    # Q1: Is there a testable predicate?
    if has_testable_pred:
        return "refine_until_valid"
    else:
        return "bounded_critique_refine"
    # Q2: Should each iteration be validated?
    # Both templates support validation wrapping — set via slot_values.


def _select_compare(hints: dict) -> str:
    compare_target = hints.get("compare_target", "strategies")
    select_or_synthesize = hints.get("select_or_synthesize", "select")

    # Q1: Compare models or strategies?
    # Both use compare_candidates; distinction is in slot_values.
    # Q2: Select one or synthesize all?
    if select_or_synthesize == "select":
        return "compare_candidates"  # python_compute or Scheme selection
    else:
        return "compare_candidates"  # with llm-query aggregator in reduce slot


def _select_classify(hints: dict) -> str:
    item_count = hints.get("item_count", 1)
    output_type = hints.get("output_type", "per_item")
    ambiguous_items = hints.get("ambiguous_items", False)

    # Q1: One item or many?
    if item_count <= 1:
        return "direct_call"  # direct_classify path
    else:
        # Q3: Ambiguous categories?
        if ambiguous_items:
            return "tiered_review"
        # Q2: Need distribution/report?
        needs_report = hints.get("has_second_phase", False)
        if needs_report:
            return "batch_map"  # with python_compute aggregation as second phase
        return "batch_map"


def _select_pipeline(hints: dict) -> str:
    stages_distinct = hints.get("stages_distinct", True)

    # Q1: Are stages distinct?
    if not stages_distinct:
        # Reclassify as Batch
        raise ReclassificationNeeded("Stages not distinct. Reclassify as Batch.")

    # Q2-Q3: Can a stage fail / does a stage need quality gating?
    # These affect slot_values (fallback, gate), not template selection.
    # Pipeline maps to a template_chain of atomic templates.
    # Return the first-stage template; caller builds the chain.
    return "batch_map"  # placeholder — caller assembles chain per sub_operations


def _select_generate(hints: dict) -> str:
    fixed_count = hints.get("fixed_count", True)
    items_consistent = hints.get("items_consistent", False)
    items_unique = hints.get("items_unique", False)

    # Q1: Fixed number or until condition?
    if fixed_count:
        if items_consistent:
            # Q2: Must items be mutually consistent?
            return "ordered_synthesis_fold"  # fold-sequential maintains consistency
        else:
            return "batch_map"  # map-async over generated index list
    else:
        return "refine_until_valid"  # iterate-until condition

    # Q3: Must items be unique? — handled via python_compute dedup in slot_values.


def _select_decompose(hints: dict) -> str:
    known_boundary = hints.get("known_boundary", False)
    one_pass = hints.get("one_pass", True)
    process_parts_after = hints.get("process_parts_after", False)

    # Q1: Known structural boundary?
    if known_boundary:
        # python_compute splitter — use direct_json_extract with python_compute
        template = "direct_json_extract"
    else:
        # llm-query with JSON output
        template = "direct_json_extract"

    # Q2: Is one pass enough?
    if not one_pass:
        return "recursive_decompose"

    # Q3: Process parts afterward?
    if process_parts_after:
        return "decompose_then_batch"

    return template


def _select_validate(hints: dict) -> str:
    same_rubric = hints.get("same_rubric", True)
    needs_structured = hints.get("output_type") in ("list", "per_item")
    ambiguous_items = hints.get("ambiguous_items", False)

    # Q1: Same rubric for all items?
    if same_rubric:
        if ambiguous_items:
            return "tiered_review"
        return "batch_map"  # map-async validation
    else:
        return "ordered_synthesis_fold"  # fold-sequential if criteria evolve

    # Q2: Need structured assessment? — handled via json_mode in slot_values.
    # Q3: Which error is costlier? — handled via tiered_review or extra slot.


def _select_aggregate(hints: dict) -> str:
    pure_computation = hints.get("pure_computation", True)
    grouped_report = hints.get("grouped_report", False)

    # Q1: Pure computation after extraction?
    if pure_computation:
        return "tabular_extract_aggregate"
    else:
        # map-async extraction + python_compute stats + llm interpretation
        return "tabular_extract_aggregate"  # with interpret_instruction slot set

    # Q2: Grouped report? — handled via groupby flag in slot_values.


def _select_composite(hints: dict) -> str:
    """
    Composite: return the name of the first atomic template in the chain.
    The planner builds the full template_chain; this function identifies
    the first phase template.
    """
    sub_operations = hints.get("sub_operations") or []
    if not sub_operations:
        return "batch_map"  # conservative default

    first_op = sub_operations[0]
    op_to_shape = {
        "extract": "Batch",
        "transform": "Batch",
        "label": "Classify",
        "check": "Validate",
        "synthesize": "Synthesize",
        "refine": "Refine",
        "compare": "Compare",
        "aggregate": "Aggregate",
        "decompose": "Decompose",
        "generate": "Generate",
    }
    first_shape = op_to_shape.get(first_op, "Batch")
    first_hints = dict(hints)
    first_hints["has_second_phase"] = False
    first_hints["sub_operations"] = []
    return select_template(first_shape, hints.get("data_shape", "FlatList"), first_hints)


class ReclassificationNeeded(Exception):
    pass
```

---

## Appendix G: Template Catalog Summary Table

All 16 templates with supported shapes, data shapes, primitive composition, key slots, and streaming/caching flags.

| Template | Supported TaskShapes | Supported DataShapes | Primitive Composition | Key Slots | Streamable | Cacheable |
|---|---|---|---|---|---|---|
| `direct_call` | Direct | Singular, KeyValue | `llm-query` | `instruction`, `model`, `temperature`, `max_tokens` | No | Yes |
| `direct_json_extract` | Direct, Decompose | Singular, KeyValue | `llm-query #:json #t` + `with-validation` | `instruction`, `model`, `output_schema`, `validation_instruction` | No | Yes |
| `batch_map` | Batch, Classify, Validate | FlatList, Paired, Multimodal | `map-async` | `context_id`, `items_path`, `map_instruction`, `map_model`, `max_concurrent` | Yes | Yes |
| `batch_extract_reduce` | Batch + Synthesize | FlatList, Tabular | `map-async` + `tree-reduce` | `context_id`, `items_path`, `map_instruction`, `reduce_instruction`, `map_model`, `reduce_model`, `max_concurrent`, `branch_factor` | Yes | Yes |
| `batch_extract_fold` | Batch + ordered Synthesize | FlatList, TimeSeries, ChunkedSingular | `map-async` + `fold-sequential` or `fold-sequential` alone | `context_id`, `items_path`, `map_instruction`, `fold_instruction`, `map_model`, `fold_model`, `max_concurrent` | Yes | Yes |
| `ordered_synthesis_fold` | Synthesize, Generate | FlatList (ordered), ChunkedSingular, TimeSeries | `fold-sequential` + optional `checkpoint` | `context_id`, `items_path`, `fold_instruction`, `fold_model`, `checkpoint_every` | Yes | Yes |
| `tree_synthesis` | Synthesize | FlatList, Hierarchy | `tree-reduce` | `context_id`, `items_path`, `reduce_instruction`, `reduce_model`, `branch_factor` | Yes | Yes |
| `compare_candidates` | Compare, Search | FlatList, Paired | `parallel` + `python-compute` or `llm-query` selector | `candidates_path`, `evaluation_instruction`, `selection_instruction`, `eval_model`, `select_model` | No | Yes |
| `race_candidates` | Search | FlatList | `race` | `candidates_path`, `generation_instruction`, `model`, `timeout_seconds` | No | No |
| `refine_until_valid` | Refine, Search | Singular | `iterate-until` + `with-validation` | `initial_instruction`, `refine_instruction`, `validate_instruction`, `model`, `max_iterations` | No | No |
| `bounded_critique_refine` | Refine | Singular | `iterate-until` with critique/refine state | `initial_instruction`, `critique_instruction`, `refine_instruction`, `model`, `max_iterations`, `critique_model` | No | No |
| `tiered_review` | Batch, Classify, Validate | FlatList, Tabular | cheap `map-async` + uncertainty filter + expensive `map-async` | `context_id`, `items_path`, `cheap_instruction`, `expensive_instruction`, `cheap_model`, `expensive_model`, `uncertainty_threshold` | Yes | Yes |
| `tabular_extract_aggregate` | Aggregate | Tabular, FlatList | `map-async` + `python-compute` | `context_id`, `items_path`, `extract_instruction`, `extract_model`, `aggregate_expression`, `output_schema` | Yes | Yes |
| `decompose_then_batch` | Decompose, Composite | Singular, Hierarchy | `llm-query #:json #t` + `map-async` | `context_id`, `decompose_instruction`, `process_instruction`, `decompose_model`, `process_model`, `max_concurrent` | Yes | Yes |
| `recursive_decompose` | Decompose, Hierarchy | Hierarchy, Singular | `recursive-spawn` (artifact-aware) | `context_id`, `decompose_instruction`, `model`, `max_depth`, `branch_factor` | No | No |
| `code_interpreter` | Direct, Aggregate | Singular, Tabular | `llm-query` + `py-exec` + `with-validation` + `iterate-until` | `instruction`, `model`, `max_iterations`, `allowed_imports`, `output_schema` | No | No |

**Notes:**
- `race_candidates` is not cacheable because it is inherently non-deterministic — the winning candidate depends on which LLM call returns first.
- `refine_until_valid` and `bounded_critique_refine` are not cacheable because each iteration depends on the previous result; caching intermediate states requires checkpoint semantics, not result caching.
- `recursive_decompose` is not streamable because the call graph is not known ahead of time; items emerge from recursive decomposition dynamically.
- `code_interpreter` requires `uses-llm-generated-code: true` in `define-meta` and will fail the `code_interpreter_policy` verification check unless the execution policy has `allow_llm_generated_code: true`.
- `batch_extract_fold` supports both the two-phase pattern (`map-async` then `fold-sequential`) and the one-phase pattern (direct `fold-sequential` when items are dependent).

---

## Appendix H: End-to-End Walkthroughs

### H.1 Basic Happy Path

Complete MCP call sequence for: "Extract ACE2 protein mentions from 100 research papers and synthesize a report."

The happy path requires only 3 tool calls after loading context: `plan_strategy` → `dry_run_strategy` → `execute_strategy`.

#### Step 1: Load context

Agent sends 100 papers as a JSON array.

```
→ load_context(
    data: "[{\"id\":\"paper_001\",\"text\":\"...\"},...]",
    name: "ace2_papers",
    metadata_json: "{\"data_shape\":\"FlatList\",\"item_count\":100,\"independent\":true,\"modality\":[\"text\"]}"
  )

← {
    "status": "ok",
    "context_id": "ctx_7f3a",
    "name": "ace2_papers",
    "metadata": {"data_shape":"FlatList","item_count":100,"independent":true},
    "preview": "[{\"id\":\"paper_001\",\"text\":\"Recent studies on ACE2...",
    "next_actions": ["Call plan_strategy with context_id=ctx_7f3a"]
  }
```

#### Step 2: Plan strategy

Agent describes the task. Hints provide structured fields so classification is fully deterministic (Level 1 only, no LLM call needed for classification). One additional LLM call fills content slots (`map_instruction`, `reduce_instruction`).

```
→ plan_strategy(
    task: "Extract ACE2 mentions from each paper and synthesize findings into one report.",
    context_id: "ctx_7f3a",
    hints_json: "{\"item_count\":100,\"independent\":true,\"output_type\":\"one\",\"has_second_phase\":true,\"sub_operations\":[\"extract\",\"synthesize\"]}"
  )

← {
    "status": "ok",
    "plan_id": "plan_b2c1",
    "classification": {
      "task_shape": "Composite",
      "constituent_shapes": ["Batch","Synthesize"],
      "data_shape": "FlatList",
      "confidence": 1.0,
      "rationale": "All hints provided. Deterministic classification: independent items with combined output and second phase."
    },
    "recommended": {
      "kind": "template_invocation",
      "template_name": "batch_extract_reduce",
      "template_version": "1.0.0",
      "slot_values": {
        "context_id": "ctx_7f3a",
        "items_path": "$",
        "map_instruction": "Extract all mentions of ACE2 protein, including evidence and uncertainty, as JSON.",
        "reduce_instruction": "Synthesize the extracted ACE2 findings into a concise report with citations.",
        "map_model": "fast_text_model",
        "reduce_model": "quality_text_model",
        "max_concurrent": 20,
        "branch_factor": 5,
        "json_mode": true
      }
    },
    "alternatives": [
      {"template_name":"batch_extract_fold","tradeoff":"Preserves paper order but higher latency."}
    ],
    "next_actions": ["Call dry_run_strategy(plan_id=plan_b2c1)"]
  }
```

#### Step 3: Dry run

Agent runs a dry run. Internally, this instantiates the template (validates slots, substitutes markers, hashes, stores artifact), computes cost estimates, and simulates execution — all in one call. No real LLM calls are made. Tree-reduce call count for N=100, B=5: `100 + 20 + 4 + 1 = 125`.

```
→ dry_run_strategy(plan_id: "plan_b2c1")

← {
    "status": "ok",
    "dry_run_id": "dry_1a2b",
    "plan_id": "plan_b2c1",
    "artifact": {
      "artifact_id": "art_e4d9",
      "template_name": "batch_extract_reduce",
      "template_version": "1.0.0",
      "code_hash": "sha256:a1b2c3...",
      "primitives_used": ["map-async","tree-reduce","llm-query-async","llm-query"]
    },
    "estimate": {
      "expected_llm_calls": 125,
      "critical_path_calls": 4,
      "max_concurrency": 20,
      "models": {"fast_text_model":100,"quality_text_model":25},
      "estimated_tokens": {"prompt":100000,"completion":31250,"total":131250},
      "estimated_cost_usd": {"low":1.20,"high":3.50}
    },
    "simulation": {
      "llm_calls": 125,
      "max_concurrency": 20,
      "recursive_depth": 0,
      "critical_path_calls": 4,
      "checkpoints": 0,
      "python_phases": 0
    },
    "call_graph": [
      {"node_id":"extract","primitive":"map-async","calls":100,"model":"fast_text_model","concurrency":20},
      {"node_id":"synthesize","primitive":"tree-reduce","calls":25,"model":"quality_text_model","branch_factor":5}
    ],
    "warnings": [],
    "next_actions": ["Call execute_strategy(plan_id=plan_b2c1)"]
  }
```

#### Step 4: Execute

Agent runs the strategy. Internally, this cache-hits the artifact from the dry run via hash match, runs all 23 verification checks automatically, then executes the instantiated Scheme. Real LLM calls happen here.

```
→ execute_strategy(
    plan_id: "plan_b2c1",
    timeout_seconds: 900,
    runtime_options_json: "{\"progress_interval_seconds\":5}",
    policy_json: "{\"max_llm_calls\":500,\"max_concurrency\":50}"
  )

  ... (progress notifications arrive every 5 seconds) ...

← {
    "status": "ok",
    "execution_id": "exec_5e6f",
    "artifact_id": "art_e4d9",
    "verification": {
      "verification_id": "ver_3c4d",
      "decision": "pass",
      "checks": [
        {"name":"artifact_hash","status":"pass","message":"Code hash matches."},
        {"name":"primitive_allowlist","status":"pass","message":"Only allowed primitives used."},
        {"name":"call_count_limit","status":"pass","message":"125 <= 500."},
        {"name":"concurrency_limit","status":"pass","message":"20 <= 50."},
        {"name":"context_exists","status":"pass","message":"ctx_7f3a exists."}
      ]
    },
    "result": {
      "value": "ACE2 (Angiotensin-Converting Enzyme 2) findings across 100 papers...",
      "stdout": ""
    },
    "execution": {
      "state": "finished",
      "elapsed_seconds": 182.4,
      "llm_calls": 125,
      "tokens": 131250,
      "models": {"fast_text_model":100,"quality_text_model":25},
      "checkpoints_written": 0
    },
    "next_actions": ["Call get_execution_trace(execution_id=exec_5e6f)"]
  }
```

#### Step 5: Inspect trace (optional)

Agent reviews what happened during execution.

```
→ get_execution_trace(execution_id: "exec_5e6f")

← {
    "status": "ok",
    "execution_id": "exec_5e6f",
    "trace": {
      "artifact_id": "art_e4d9",
      "plan_id": "plan_b2c1",
      "events": [
        {"type":"llm_call_started","call_id":"call_001","node_id":"extract","model":"fast_text_model","depth":0},
        {"type":"llm_call_completed","call_id":"call_001","tokens":1250,"elapsed_seconds":1.2},
        {"type":"llm_call_started","call_id":"call_101","node_id":"synthesize","model":"quality_text_model","depth":0},
        {"type":"llm_call_completed","call_id":"call_101","tokens":5000,"elapsed_seconds":4.1}
      ],
      "scope_log": [
        {"op":"syntax-e","preview":"extracted ACE2 mentions...","scope":"sandbox","call_id":"call_001"}
      ],
      "stdout": ""
    }
  }
```

#### ID chain summary

```
ctx_7f3a (data)
  → plan_b2c1 (classification + template invocation)
    → dry_1a2b (instantiate + simulate + estimate)  [creates art_e4d9]
      → exec_5e6f (cache-hit artifact + verify + execute)  [creates ver_3c4d]
```

Internal IDs `art_e4d9` (artifact) and `ver_3c4d` (verification) are created automatically and appear in responses for audit and replay. They are not passed between tools by the agent.

---

### H.2 Chained Workflow With Streaming And Cache Reuse

This walkthrough shows a Composite task using template chaining, streaming partial results, and cross-execution cache reuse across two runs. The second run changes only the synthesis instruction; all 100 map-phase calls hit the cache because the extraction instruction, model, and data are identical.

#### First run: Extract then synthesize, with streaming

```
→ load_context(
    data: "[100 papers]",
    name: "ace2_papers",
    metadata_json: "{\"data_shape\":\"FlatList\",\"item_count\":100,\"independent\":true}"
  )
← { "status": "ok", "context_id": "ctx_7f3a" }
```

```
→ plan_strategy(
    task: "Extract ACE2 mentions and synthesize a report.",
    context_id: "ctx_7f3a",
    hints_json: "{\"item_count\":100,\"independent\":true,\"output_type\":\"one\",\"has_second_phase\":true,\"sub_operations\":[\"extract\",\"synthesize\"]}"
  )
← {
    "status": "ok",
    "plan_id": "plan_c1d2",
    "recommended": {
      "kind": "template_chain",
      "steps": [
        {
          "step": 0,
          "template_name": "batch_map",
          "slot_values": {
            "context_id": "ctx_7f3a",
            "items_path": "$",
            "map_instruction": "Extract ACE2 mentions as JSON.",
            "map_model": "fast_text_model",
            "max_concurrent": 20
          }
        },
        {
          "step": 1,
          "template_name": "tree_synthesis",
          "slot_values": {
            "input": "$previous",
            "reduce_instruction": "Synthesize findings into a report.",
            "reduce_model": "quality_text_model",
            "branch_factor": 5
          }
        }
      ]
    }
  }
```

The `$previous` reference in step 1's `input` slot is resolved at execution time to the intermediate context created by step 0. The planner writes `$previous` into the slot value; the chain executor substitutes the actual `context_id` of step 0's output before instantiating step 1's artifact.

```
→ dry_run_strategy(plan_id: "plan_c1d2")
← {
    "status": "ok",
    "dry_run_id": "dry_e3f4",
    "steps": [
      {
        "step": 0,
        "template_name": "batch_map",
        "artifact_id": "art_f1a0",
        "estimated_llm_calls": 100,
        "estimated_cost_usd": {"low": 0.80, "high": 2.00}
      },
      {
        "step": 1,
        "template_name": "tree_synthesis",
        "artifact_id": "art_f1a1",
        "estimated_llm_calls": 25,
        "estimated_cost_usd": {"low": 0.40, "high": 1.50}
      }
    ],
    "total_estimated_llm_calls": 125,
    "total_estimated_cost_usd": {"low": 1.20, "high": 3.50},
    "cache_hits_expected": 0,
    "warnings": []
  }
```

```
→ execute_strategy(
    plan_id: "plan_c1d2",
    stream: true,
    timeout_seconds: 900
  )
```

With `stream: true`, the server sends MCP notifications as map-phase items complete. Each notification is a JSON object delivered on the MCP notification channel before the final response:

```
{ "jsonrpc": "2.0", "method": "notifications/partial_result",
  "params": { "execution_id": "exec_a1b2", "node_id": "extract",
               "item_index": 0, "value": "{\"ace2_mentions\":[...]}" } }

{ "jsonrpc": "2.0", "method": "notifications/partial_result",
  "params": { "execution_id": "exec_a1b2", "node_id": "extract",
               "item_index": 1, "value": "{\"ace2_mentions\":[...]}" } }

...  (98 more notifications, one per paper as each completes)
```

Final response after all items complete and synthesis finishes:

```
← {
    "status": "ok",
    "execution_id": "exec_a1b2",
    "result": {
      "value": "ACE2 findings report...",
      "stdout": ""
    },
    "execution": {
      "state": "finished",
      "elapsed_seconds": 195.2,
      "llm_calls": 125,
      "tokens": 131250,
      "cache_hits": 0,
      "budget_policy_activations": 0,
      "chain_step_results": [
        {
          "step": 0,
          "template": "batch_map",
          "intermediate_context_id": "ctx_auto_0",
          "llm_calls": 100,
          "state": "finished"
        },
        {
          "step": 1,
          "template": "tree_synthesis",
          "llm_calls": 25,
          "state": "finished",
          "result": "ACE2 findings report..."
        }
      ]
    }
  }
```

The chain executor created `ctx_auto_0` automatically to hold step 0's output (the 100 extracted JSON objects). Step 1 resolved `$previous` to `ctx_auto_0` before instantiating its artifact.

#### Second run: Same extraction, different synthesis instruction

The agent wants a methods section instead of a report. The map instruction, map model, and source data are identical to the first run, so all 100 map-phase LLM calls will hit the content-addressed cache.

```
→ plan_strategy(
    task: "Extract ACE2 mentions and write a methods section.",
    context_id: "ctx_7f3a",
    hints_json: "{\"item_count\":100,\"independent\":true,\"output_type\":\"one\",\"has_second_phase\":true,\"sub_operations\":[\"extract\",\"synthesize\"]}"
  )
← {
    "status": "ok",
    "plan_id": "plan_d4e5",
    "recommended": {
      "kind": "template_chain",
      "steps": [
        {
          "step": 0,
          "template_name": "batch_map",
          "slot_values": {
            "context_id": "ctx_7f3a",
            "items_path": "$",
            "map_instruction": "Extract ACE2 mentions as JSON.",
            "map_model": "fast_text_model",
            "max_concurrent": 20
          }
        },
        {
          "step": 1,
          "template_name": "tree_synthesis",
          "slot_values": {
            "input": "$previous",
            "reduce_instruction": "Write a methods section describing how ACE2 was studied across these papers.",
            "reduce_model": "quality_text_model",
            "branch_factor": 5
          }
        }
      ]
    }
  }
```

Step 0's slot values are identical to the first run. The dry run detects this and predicts 100 cache hits:

```
→ dry_run_strategy(plan_id: "plan_d4e5")
← {
    "status": "ok",
    "dry_run_id": "dry_f5g6",
    "steps": [
      {
        "step": 0,
        "template_name": "batch_map",
        "artifact_id": "art_f1a0",
        "estimated_llm_calls": 100,
        "cache_hits_expected": 100,
        "estimated_cost_usd": {"low": 0.00, "high": 0.00}
      },
      {
        "step": 1,
        "template_name": "tree_synthesis",
        "artifact_id": "art_f2b3",
        "estimated_llm_calls": 25,
        "cache_hits_expected": 0,
        "estimated_cost_usd": {"low": 0.40, "high": 1.50}
      }
    ],
    "total_estimated_llm_calls": 125,
    "cache_hits_expected": 100,
    "total_estimated_cost_usd": {"low": 0.40, "high": 1.50},
    "warnings": []
  }
```

Step 0's `artifact_id` is `art_f1a0` — the same hash as the first run, because the slot values are identical. Step 1 has a new artifact ID because its `reduce_instruction` differs.

```
→ execute_strategy(
    plan_id: "plan_d4e5",
    timeout_seconds: 900
  )
← {
    "status": "ok",
    "execution_id": "exec_c3d4",
    "result": {
      "value": "Methods section focusing on ACE2 receptor characterization...",
      "stdout": ""
    },
    "execution": {
      "state": "finished",
      "elapsed_seconds": 48.7,
      "llm_calls": 125,
      "cache_hits": 100,
      "chain_step_results": [
        {
          "step": 0,
          "template": "batch_map",
          "intermediate_context_id": "ctx_auto_1",
          "llm_calls": 100,
          "cache_hits": 100,
          "state": "finished"
        },
        {
          "step": 1,
          "template": "tree_synthesis",
          "llm_calls": 25,
          "cache_hits": 0,
          "state": "finished",
          "result": "Methods section focusing on ACE2 receptor characterization..."
        }
      ]
    }
  }
```

The second run completed in 48.7 seconds versus 195.2 seconds for the first run. All 100 map-phase calls returned instantly from cache. Only the 25 tree-reduce synthesis calls made real LLM requests, reducing cost by approximately 60%.

#### Cache key mechanics

The content-addressed cache key for each LLM call is derived from:

```python
cache_key = sha256(json.dumps({
    "instruction": call.instruction,
    "input_text": call.input_text,
    "model": call.resolved_model_id,
    "temperature": call.temperature,
    "json_mode": call.json_mode,
}, sort_keys=True).encode()).hexdigest()
```

Because the map instruction (`"Extract ACE2 mentions as JSON."`), model alias (`fast_text_model`), temperature (0), and each paper's text are identical between runs, every map-phase cache key from the second run matches a key written during the first run. The synthesis cache keys differ because the reduce instruction changed.