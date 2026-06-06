# RLM-Scheme Greenfield Rewrite Plan

Complete implementation plan for rewriting RLM-Scheme from scratch. No backward
compatibility required. The existing codebase serves only as a feature inventory
for runtime behavior that should be preserved — see section 21 (Success Criteria)
for the full list.

---

## 1. North Star

The MCP server should expose a structured orchestration system, not a raw code
execution system.

An agent interacts with durable objects:

```text
context_id -> plan_id -> dry_run_id -> execution_id
```

Each stage has a clear responsibility:

| ID | Meaning |
|---|---|
| `context_id` | Stored input data plus metadata: shape, item count, modality, independence, size estimates, and optional names. |
| `plan_id` | Task intent and planning record: objective, constraints, inferred TaskShape/DataShape, selected template, and rationale. |
| `dry_run_id` | Structural simulation: expected calls, fan-out, recursive depth, model mix, token/cost estimates, warnings, and failure risks. |
| `execution_id` | One real execution attempt: result, stdout, trace, call metrics, token usage, errors, checkpoints, and status history. |

Internally, the system also creates `artifact_id` (instantiated Scheme with
code hash) and `verification_id` (pre-execution checks) records. These
appear in responses for audit and debugging but are not agent-managed
concepts — agents do not create or pass them between tools.

Normal agent flow:

1. `load_context(data, name, metadata)` stores large input and returns a
   `context_id`.
2. `plan_strategy(task, context_id, hints)` classifies the work and returns a
   `plan_id` plus a template invocation (or a template chain for multi-phase
   Composite tasks).
3. `dry_run_strategy(plan_id)` instantiates, simulates, and returns a
   `dry_run_id` with call estimates and cost projections.
4. `execute_strategy(plan_id, timeout, stream)` instantiates (or reuses a
   cached artifact), verifies, and executes. Returns an `execution_id`.
   With `stream=true`, partial results are delivered via MCP notifications
   as items complete. Execution may pause at declared gates for human review;
   `resume_execution` approves or rejects.
5. `get_execution_trace(execution_id)`, `get_status`, `cancel_call`, and
   `reset_runtime` inspect or control work.

Instantiation (slot validation + safe substitution + hashing) and verification
(policy checks) happen automatically inside `dry_run_strategy` and
`execute_strategy`. If either step fails, the tool returns a structured
error. There are no separate tools for instantiation, estimation, or
verification.

Cross-execution memoization caches LLM results by content hash. Repeated
runs with identical inputs (same instruction, data, model, temperature)
reuse cached results automatically, making iterative refinement cheap.

Scheme is internal instantiated code. It may be inspectable through dry-run or
execution responses for debugging, but agents should not submit arbitrary
Scheme strings to the public MCP API.

---

## 2. What Templates Store

Templates are the bridge between high-level planning and executable Scheme.
They should be data, not prompts that ask an LLM to write code.

There are two levels in the instantiation pipeline:

```text
Template (Scheme code with {{slot}} markers)
    ↓  instantiator validates slots, substitutes values, hashes result
Artifact (executable Scheme run by the Racket sandbox)
```

A **template** is a `.rkt` file containing real Scheme code that uses
primitive runtime bindings directly. Content-specific values are represented
as `{{slot_name}}` markers — typed holes that the instantiator fills with
concrete values. For example, a `batch_extract_reduce` template contains
`map-async` and `tree-reduce` calls but uses `{{map_instruction}}`,
`{{map_model}}`, and `{{max_concurrent}}` markers where content-specific
values belong.

Templates are Scheme, not JSON node graphs. There is no intermediate
representation between the template and the executable artifact. The instantiator
validates slot values against the template's slot schema, performs safe
substitution of `{{slot}}` markers with concrete values, and hashes the
result. The output is executable Scheme that runs directly in the Racket
sandbox.

A template file is pure Scheme. Metadata lives in `define-meta` forms at
the top; the executable body follows. No JSON, no block comments — one
language throughout.

`define-meta` is a custom form recognized by the template loader, not
standard Racket `define`. The loader collects all `define-meta` bindings
into a metadata hash before evaluating the body.

**`define-meta` grammar:**

```bnf
meta-form    ::= '(' 'define-meta' name value ')'
name         ::= symbol
value        ::= atom | quoted-list
atom         ::= string | number | boolean
boolean      ::= '#t' | '#f'
quoted-list  ::= "'" s-expr
s-expr       ::= atom | '(' s-expr* ')' | '(' s-expr '.' s-expr ')'
```

The loader recognizes `define-meta` forms at the top level of the template file. Each `define-meta` binding associates a symbol name with a value. Atomic values (strings, numbers, booleans) are stored directly. Quoted lists are stored as Scheme data structures. The loader collects all `define-meta` bindings into a hash table keyed by name; duplicate names are an error.

Template metadata includes:

- `name` and `version`,
- supported TaskShape/DataShape combinations,
- trigger and rejection conditions (quoted Scheme predicates evaluated
  against classification hints),
- typed slots with defaults, enums, ranges, required fields, and
  descriptions (alists instead of JSON Schema),
- model requirements such as JSON mode or image support,
- output shape and schema contract (alist notation declaring the structure
  of the value passed to `finish`),
- expected call formulas and structural profiles,
- verification rules and dry-run warnings,
- `streamable` flag — whether meaningful intermediate results exist
  (e.g., completed map items),
- `cacheable` flag — whether LLM call results can be cached across
  executions for content-addressed reuse,
- `gates` — declared human-review checkpoints that suspend execution,
- `budget-policy` — degradation behavior (model switching, checkpoint-and-stop)
  when the token budget runs low,
- `uses-llm-generated-code` flag — whether the template uses the code
  interpreter pattern (`llm-query` → `py-exec`), requiring explicit
  policy approval.

**Output-schema alist syntax:** Output schemas use alist notation that maps
1:1 to JSON Schema. Each `(key value)` pair in the alist corresponds to a
JSON Schema keyword. Nested objects use nested alists. The conversion rule
is: `'((type object) (properties ((name (type string))))) ` becomes
`{"type": "object", "properties": {"name": {"type": "string"}}}`. The
template loader validates structural well-formedness (balanced parens, known
JSON Schema keywords at the top level: `type`, `properties`, `items`,
`required`, `enum`, `description`) but does not validate semantic JSON
Schema correctness — that happens at verification time.

The Scheme body uses only:

- primitive runtime bindings (section 9),
- instantiator-owned helper bindings (prefixed with `__`),
- `{{slot_name}}` markers that the instantiator substitutes before execution.

The planner reads template metadata and fills slots. The instantiator validates
slot values, substitutes them into the template body, and stores the result
as an immutable artifact. Agents interact only with templates (via
`plan_strategy`); instantiation happens internally when they call
`dry_run_strategy` or `execute_strategy`.

This division is important:

- LLMs choose strategy intent and content slots (template selection + slot
  filling).
- Deterministic code validates slots and substitutes them safely
  (instantiation). No code generation or IR translation is involved.
- Verification checks the instantiated artifact before real model calls happen.

---

## 3. Public MCP API

The greenfield server should expose a small, artifact-based MCP surface (10 tools).
Instantiation, estimation, and verification happen internally — agents do not
need separate tools for these steps.

| Tool | Purpose |
|---|---|
| `load_context(data, name=None, metadata=None)` | Store input data and metadata; return `context_id`. |
| `get_context(context_id)` | Inspect metadata and optionally preview stored data. |
| `plan_strategy(task, context_id=None, hints=None)` | Classify task/data and return `plan_id` plus proposed template invocation. |
| `dry_run_strategy(plan_id=None, template_invocation=None)` | Instantiate, simulate, and estimate without real LLM calls. Return `dry_run_id`. |
| `execute_strategy(plan_id=None, template_invocation=None, timeout=None)` | Instantiate, verify, and execute. Return `execution_id`. |
| `get_execution_trace(execution_id)` | Return call hierarchy, data flow, stdout, errors, token usage, and checkpoints. |
| `get_status(execution_id=None)` | Return server/runtime/call status. |
| `cancel_call(call_id=None, execution_id=None)` | Cancel one call or an entire execution. |
| `resume_execution(execution_id, gate, decision, reason=None)` | Approve or reject a gate to resume or terminate a suspended execution. |
| `reset_runtime(scope="session")` | Reset sandbox state without deleting durable records by default. |

Target FastMCP function signatures:

```python
def load_context(
    data: str,
    name: str | None = None,
    metadata_json: str | None = None,
) -> str: ...

def get_context(
    context_id: str,
    include_preview: bool = True,
    include_data: bool = False,
) -> str: ...

def plan_strategy(
    task: str,
    context_id: str | None = None,
    hints_json: str | None = None,
) -> str: ...

def dry_run_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    options_json: str | None = None,
) -> str: ...

async def execute_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    timeout_seconds: int | None = None,
    stream: bool = False,
    runtime_options_json: str | None = None,
    ctx: Context = None,
) -> str: ...

def get_execution_trace(
    execution_id: str,
    include_scope_log: bool = True,
    include_calls: bool = True,
    include_stdout: bool = True,
) -> str: ...

def get_status(execution_id: str | None = None) -> str: ...

def cancel_call(
    call_id: str | None = None,
    execution_id: str | None = None,
    reason: str | None = None,
) -> str: ...

def reset_runtime(scope: str = "session") -> str: ...

async def resume_execution(
    execution_id: str,
    gate: str,
    decision: str,
    reason: str | None = None,
) -> str: ...
```

`*_json` parameters are JSON strings because MCP clients vary in how reliably
they support nested structured arguments. The server should parse and validate
them into typed internal models immediately.

At least one of `plan_id` or `template_invocation_json` is required for
`dry_run_strategy` and `execute_strategy`. If `plan_id` is provided and
already contains a recommended template invocation, the tool uses it
directly.

Do not expose these as public tools:

- `execute_scheme(code, ...)`,
- `dry_run_scheme(code, ...)`,
- arbitrary raw code import,
- public unsafe interpolation/overwrite/eval helpers.

Internal test helpers may still invoke lower-level runtime functions, but the
MCP contract should only expose artifact-based orchestration.

---

## 4. MCP Request And Response Schemas

All public MCP tools should return JSON strings. Each response should include a
stable top-level `status` field so agents can handle errors mechanically.

Common response shape:

```json
{
  "status": "ok | warn | error",
  "id": "optional primary id",
  "warnings": [],
  "errors": [],
  "next_actions": []
}
```

Errors should be structured:

```json
{
  "status": "error",
  "error": {
    "code": "verification_failed",
    "message": "Artifact failed verification.",
    "details": {
      "artifact_id": "art_...",
      "failed_checks": ["call_count_limit"]
    },
    "retryable": false
  }
}
```

### 4.1 `load_context`

Purpose: store input data and metadata so plans reference data by ID instead of
copying it through every tool call.

Request:

```json
{
  "data": "string | JSON-serializable value",
  "name": "optional human-readable name",
  "metadata": {
    "data_shape": "FlatList | Hierarchy | Singular | ChunkedSingular | Graph | TimeSeries | Tabular | Multimodal | Paired | KeyValue | Unknown",
    "item_count": 100,
    "item_size_estimate_tokens": 500,
    "total_size_estimate_tokens": 50000,
    "independent": true,
    "ordered": false,
    "modality": ["text"],
    "chunking": {
      "chunk_count": 100,
      "overlap_tokens": 100,
      "boundary": "paragraph"
    },
    "source": {
      "kind": "inline | file | url | generated",
      "uri": "optional source identifier"
    },
    "schema": {
      "type": "optional JSON schema or table schema"
    }
  }
}
```

Response:

```json
{
  "status": "ok",
  "context_id": "ctx_01HX...",
  "name": "papers",
  "metadata": {
    "data_shape": "FlatList",
    "item_count": 100,
    "total_size_estimate_tokens": 50000,
    "independent": true,
    "modality": ["text"]
  },
  "preview": "first 500 characters or structured preview",
  "next_actions": [
    "Call plan_strategy with context_id=ctx_01HX..."
  ]
}
```

### 4.2 `get_context`

Request:

```json
{
  "context_id": "ctx_01HX...",
  "include_preview": true,
  "include_data": false
}
```

Response:

```json
{
  "status": "ok",
  "context": {
    "context_id": "ctx_01HX...",
    "name": "papers",
    "created_at": "2026-06-03T12:00:00Z",
    "metadata": {},
    "preview": "...",
    "data_hash": "sha256:..."
  }
}
```

`include_data=true` should be allowed only for contexts under 100 KB
(serialized JSON size) or when the caller sets an explicit debug flag.
Large context retrieval should default to previews and metadata. The server
returns `"data_too_large": true` in the response when `include_data=true`
is requested but the context exceeds 100 KB, and omits the data field.

### 4.3 `plan_strategy`

Purpose: classify task/data, choose a template, and persist a planning record.

Request:

```json
{
  "task": "Analyze every paper for ACE2 mentions and synthesize findings.",
  "context_id": "ctx_01HX...",
  "hints": {
    "task_shape": null,
    "data_shape": "FlatList",
    "item_count": 100,
    "independent": true,
    "output_type": "one",
    "operation": "extract_then_synthesize",
    "has_second_phase": true,
    "sub_operations": ["extract", "synthesize"],
    "priority": "balanced",
    "latency_priority": "medium",
    "quality_priority": "high",
    "budget_limit_usd": 5.0,
    "max_concurrent": 20,
    "preferred_models": {
      "map": "fast_text_model",
      "reduce": "quality_text_model"
    }
  }
}
```

Response:

```json
{
  "status": "ok",
  "plan_id": "plan_01HX...",
  "classification": {
    "task_shape": "Composite",
    "constituent_shapes": ["Batch", "Synthesize"],
    "data_shape": "FlatList",
    "confidence": 0.92,
    "rationale": "Independent per-paper extraction followed by one synthesis output."
  },
  "recommended": {
    "kind": "template_invocation",
    "template_name": "batch_extract_reduce",
    "template_version": "1.0.0",
    "slot_values": {
      "context_id": "ctx_01HX...",
      "map_instruction": "Extract ACE2 mentions, evidence, and uncertainty as JSON.",
      "reduce_instruction": "Synthesize ACE2 findings into a concise report.",
      "map_model": "fast_text_model",
      "reduce_model": "quality_text_model",
      "max_concurrent": 20,
      "branch_factor": 5,
      "json_mode": true
    }
  },
  "alternatives": [
    {
      "template_name": "batch_extract_fold",
      "tradeoff": "Preserves order but has higher latency."
    }
  ],
  "next_actions": [
    "Call dry_run_strategy(plan_id=plan_01HX...)",
    "Call execute_strategy(plan_id=plan_01HX...)"
  ]
}
```

For Composite tasks, `plan_strategy` can return a template chain instead of a
single template invocation:

```json
{
  "status": "ok",
  "plan_id": "plan_01HX...",
  "classification": {
    "task_shape": "Composite",
    "constituent_shapes": ["Batch", "Synthesize"],
    "data_shape": "FlatList",
    "confidence": 0.95
  },
  "recommended": {
    "kind": "template_chain",
    "steps": [
      {
        "template_name": "batch_map",
        "template_version": "1.0.0",
        "slot_values": {
          "context_id": "ctx_01HX...",
          "map_instruction": "Extract ACE2 mentions as JSON.",
          "map_model": "fast_text_model",
          "max_concurrent": 20,
          "json_mode": true
        }
      },
      {
        "template_name": "tree_synthesis",
        "template_version": "1.0.0",
        "slot_values": {
          "input": "$previous",
          "reduce_instruction": "Synthesize findings into a report.",
          "reduce_model": "quality_text_model",
          "branch_factor": 5
        }
      }
    ]
  },
  "next_actions": [
    "Call dry_run_strategy(plan_id=plan_01HX...)"
  ]
}
```

`$previous` in a step's `slot_values` resolves to the output of the preceding
step, stored automatically as an intermediate context during execution. See
section 11.8 for full chain execution semantics, including failure handling,
gates in chains, and dry-run behavior.

Planner output must not include raw Scheme. If no template fits, the planner
returns a structured `no_template` response:

```json
{
  "status": "no_template",
  "plan_id": "plan_01HX...",
  "classification": {
    "task_shape": "Pipeline",
    "data_shape": "TimeSeries",
    "confidence": 0.85
  },
  "recommendation": {
    "description": "No existing template handles Pipeline tasks over TimeSeries data with causal dependencies.",
    "needed_template": {
      "task_shapes": ["Pipeline"],
      "data_shapes": ["TimeSeries"],
      "primitives_likely": ["fold-sequential", "py-exec"],
      "slot_suggestions": ["window_size", "causal_model"]
    }
  },
  "next_actions": [
    "Create a new template matching the recommendation above.",
    "Or reclassify the task with different hints."
  ]
}
```

### 4.4 `dry_run_strategy`

Purpose: instantiate a plan or template invocation into an artifact, simulate
execution, and return the dry-run results along with cost estimates and
artifact details — all in one call.

Internally, this tool:

1. Instantiates the template invocation (validates slots, substitutes `{{slot}}`
   markers, hashes, stores the artifact record).
2. Computes a static cost estimate from the structural profile.
3. Simulates execution with mock LLM responses.

If instantiation fails (invalid slots, unknown template, remaining markers),
the tool returns a structured error. The agent does not need to handle
instantiation as a separate step.

Request:

```json
{
  "plan_id": "plan_01HX...",
  "template_invocation": null,
  "options": {
    "mock_prefix": "[dry-run]",
    "deterministic_await_any": true,
    "max_simulated_items": 1000,
    "assumptions": {
      "item_count": 100,
      "avg_input_tokens": 800,
      "avg_output_tokens": 250
    }
  }
}
```

At least one of `plan_id` or `template_invocation` is required. If `plan_id`
is provided and already contains a recommended template invocation, the tool
uses it directly. `template_invocation` overrides the plan's recommendation.

Response:

```json
{
  "status": "ok",
  "dry_run_id": "dry_01HX...",
  "plan_id": "plan_01HX...",
  "artifact": {
    "artifact_id": "art_01HX...",
    "template_name": "batch_extract_reduce",
    "template_version": "1.0.0",
    "code_hash": "sha256:...",
    "primitives_used": ["map-async", "tree-reduce", "llm-query-async", "llm-query"]
  },
  "estimate": {
    "expected_llm_calls": 125,
    "critical_path_calls": 4,
    "max_concurrency": 20,
    "models": {
      "fast_text_model": 100,
      "quality_text_model": 25
    },
    "estimated_tokens": {
      "prompt": 100000,
      "completion": 31250,
      "total": 131250
    },
    "estimated_cost_usd": {
      "low": 1.20,
      "high": 3.50
    }
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
    {
      "node_id": "map.extract",
      "primitive": "map-async",
      "calls": 100,
      "model": "fast_text_model",
      "concurrency": 20
    },
    {
      "node_id": "reduce.synthesize",
      "primitive": "tree-reduce",
      "calls": 25,
      "model": "quality_text_model",
      "branch_factor": 5
    }
  ],
  "warnings": [],
  "next_actions": [
    "Call execute_strategy(plan_id=plan_01HX...)"
  ]
}
```

The response also includes `output_schema` from the template's `define-meta`
when present, and `cache_hits_expected` when a matching artifact from a prior
execution is found in the LLM result cache.

For template chains, the dry-run response includes a `steps` array with
per-step estimates and validates that each step's output schema is compatible
with the next step's input expectations. Aggregate totals (`total_estimated_llm_calls`,
`total_estimated_cost_usd`) cover the entire chain.

### 4.5 `execute_strategy`

Purpose: instantiate (if not already instantiated), verify against policy, and execute
the strategy — all in one call.

Internally, this tool:

1. Instantiates the template invocation (validates slots, substitutes markers,
   hashes). If the same artifact was already instantiated (e.g. by a prior
   dry run), the cached artifact is reused automatically via hash match.
2. Runs verification checks automatically (hash integrity, primitive
   allowlist, policy limits). If verification fails, returns a structured
   error — the agent does not call a separate verify tool.
3. Executes the instantiated Scheme in the sandbox.

Request:

```json
{
  "plan_id": "plan_01HX...",
  "template_invocation": null,
  "timeout_seconds": 900,
  "runtime_options": {
    "progress_interval_seconds": 2,
    "checkpoint_prefix": "ace2-run",
    "max_stdout_chars": 4000
  },
  "policy": {
    "max_llm_calls": 500,
    "max_concurrency": 50,
    "max_recursive_depth": 3,
    "allow_python_bridge": true,
    "allow_multimodal": true
  }
}
```

At least one of `plan_id` or `template_invocation` is required. If `policy`
is omitted, server defaults apply.

Response:

```json
{
  "status": "ok",
  "execution_id": "exec_01HX...",
  "artifact_id": "art_01HX...",
  "verification": {
    "verification_id": "ver_01HX...",
    "decision": "pass",
    "checks": [
      {
        "name": "artifact_hash",
        "status": "pass",
        "message": "Generated code hash matches stored artifact."
      },
      {
        "name": "primitive_allowlist",
        "status": "pass",
        "message": "Only primitive runtime names are used."
      },
      {
        "name": "policy_limits",
        "status": "pass",
        "message": "Estimated calls (125) within limit (500)."
      }
    ]
  },
  "result": {
    "value": "final answer or JSON value",
    "stdout": "optional truncated stdout"
  },
  "execution": {
    "state": "finished",
    "elapsed_seconds": 182.4,
    "llm_calls": 125,
    "tokens": 131250,
    "models": {
      "fast_text_model": 100,
      "quality_text_model": 25
    },
    "checkpoints_written": 0
  },
  "next_actions": [
    "Call get_execution_trace(execution_id=exec_01HX...)"
  ]
}
```

If verification fails, the response has `status: "verification_failed"` with
the failing checks and no execution is attempted. The agent can adjust
policy limits or the template invocation and retry.

**Streaming:** When `stream=true`, the server emits `notifications/partial_result`
messages during execution:

```json
{
  "type": "notifications/partial_result",
  "execution_id": "exec_01HX...",
  "node_id": "extract",
  "primitive": "map-async",
  "item_index": 42,
  "items_completed": 43,
  "items_total": 100,
  "value": "{ ... extracted data ... }"
}
```

The final response still contains the complete result. Agents that don't
support notifications get the same behavior as without streaming.

**Gates:** If the template declares a gate and execution reaches it, the response
returns early with `"state": "awaiting_gate"`:

```json
{
  "status": "ok",
  "execution_id": "exec_01HX...",
  "execution": {
    "state": "awaiting_gate",
    "gate": {
      "name": "review_extractions",
      "message": "Review 100 extractions before synthesis.",
      "value_preview": "[{\"paper_id\":\"paper_001\",...}, ...]"
    }
  },
  "next_actions": [
    "Call resume_execution(execution_id=exec_01HX..., gate=review_extractions, decision=approve)"
  ]
}
```

The agent or human reviews the gate data and calls `resume_execution` to continue
or reject. See section 4.9.

**Cache and budget metrics:** The execution response `metrics` object also includes
`cache_hits` (number of LLM calls satisfied from cross-execution cache) and
`budget_policy_activations` (number of times the budget policy triggered, e.g.
model switching or checkpoint-and-stop).

### 4.6 `get_execution_trace`

Request:

```json
{
  "execution_id": "exec_01HX...",
  "include_scope_log": true,
  "include_calls": true,
  "include_stdout": true
}
```

Response:

```json
{
  "status": "ok",
  "execution_id": "exec_01HX...",
  "trace": {
    "artifact_id": "art_01HX...",
    "plan_id": "plan_01HX...",
    "events": [
      {
        "type": "llm_call_started",
        "call_id": "call_001",
        "node_id": "map.extract",
        "model": "fast_text_model",
        "depth": 0
      }
    ],
    "scope_log": [
      {
        "op": "syntax-e",
        "preview": "extracted result...",
        "scope": "sandbox",
        "call_id": "call_001"
      }
    ],
    "stdout": ""
  }
}
```

### 4.7 `get_status`

Request:

```json
{
  "execution_id": "exec_01HX..."
}
```

Response:

```json
{
  "status": "ok",
  "runtime": {
    "racket_alive": true,
    "python_bridge_alive": true,
    "sandbox_memory_limit_mb": 256
  },
  "active_calls": [
    {
      "call_id": "call_001",
      "execution_id": "exec_01HX...",
      "type": "async",
      "model": "fast_text_model",
      "elapsed_seconds": 12.3,
      "depth": 0,
      "instruction_preview": "Extract ACE2..."
    }
  ],
  "token_usage": {
    "prompt_tokens": 10000,
    "completion_tokens": 2500,
    "total_tokens": 12500,
    "calls": 10
  },
  "rate_limits": {
    "remaining_requests": 490,
    "remaining_tokens": 900000,
    "reset_requests": "..."
  }
}
```

### 4.8 `cancel_call`

Request:

```json
{
  "call_id": "call_001",
  "execution_id": null,
  "reason": "user requested cancellation"
}
```

Response:

```json
{
  "status": "ok",
  "cancelled": {
    "call_ids": ["call_001"],
    "execution_id": "exec_01HX..."
  }
}
```

If `execution_id` is provided, cancel all active and queued calls for that
execution and mark the execution as `cancelled`.

### 4.9 `resume_execution`

Purpose: approve or reject a gate to resume or terminate a suspended execution.

Request:

```json
{
  "execution_id": "exec_01HX...",
  "gate": "review_extractions",
  "decision": "approve",
  "reason": null
}
```

Response (approved — execution resumes and completes):

```json
{
  "status": "ok",
  "execution_id": "exec_01HX...",
  "gate": {
    "name": "review_extractions",
    "decision": "approve",
    "resumed_at": "2026-06-03T12:06:00Z"
  },
  "result": {
    "value": "final synthesized report...",
    "stdout": ""
  },
  "execution": {
    "state": "finished",
    "elapsed_seconds": 195.1,
    "llm_calls": 125,
    "tokens": 131250
  }
}
```

Response (rejected — execution terminates):

```json
{
  "status": "ok",
  "execution_id": "exec_01HX...",
  "gate": {
    "name": "review_extractions",
    "decision": "reject",
    "reason": "Too many false positives in extractions."
  },
  "execution": {
    "state": "gate_rejected",
    "elapsed_seconds": 90.2,
    "llm_calls": 100,
    "tokens": 100000
  }
}
```

If `decision` is `"reject"`, the execution terminates with state `"gate_rejected"`.
Completed work up to the gate is preserved in the execution record and trace.

---

## 5. Durable Record Schemas

The API schemas above are request/response contracts. The server should also
store durable records with explicit schemas so history, verification, and
replay are reliable.

**Note on internal records:** Artifact records (5.3) and verification records
(5.5) are created internally by `dry_run_strategy` and `execute_strategy` —
there are no dedicated MCP tools for creating them. The `artifact_id` and
`verification_id` appear in tool responses for traceability and are stored
durably for audit/replay, but agents never create or pass them as primary
inputs.

### 5.1 Context Record

```json
{
  "context_id": "ctx_01HX...",
  "schema_version": "1",
  "name": "papers",
  "created_at": "2026-06-03T12:00:00Z",
  "data_ref": {
    "storage": "filesystem",
    "path": "contexts/ctx_01HX/data.json",
    "hash": "sha256:...",
    "bytes": 1234567
  },
  "metadata": {
    "data_shape": "FlatList",
    "item_count": 100,
    "independent": true,
    "ordered": false,
    "modality": ["text"],
    "total_size_estimate_tokens": 50000
  }
}
```

### 5.2 Plan Record

```json
{
  "plan_id": "plan_01HX...",
  "schema_version": "1",
  "created_at": "2026-06-03T12:01:00Z",
  "context_ids": ["ctx_01HX..."],
  "task": "Analyze every paper for ACE2 mentions and synthesize findings.",
  "hints": {},
  "classification": {
    "task_shape": "Composite",
    "constituent_shapes": ["Batch", "Synthesize"],
    "data_shape": "FlatList",
    "confidence": 0.92
  },
  "recommended": {
    "kind": "template_invocation",
    "template_name": "batch_extract_reduce",
    "template_version": "1.0.0",
    "slot_values": {}
  },
  "alternatives": [],
  "planner": {
    "mode": "deterministic_with_llm_fill",
    "model": "quality_text_model",
    "prompt_hash": "sha256:..."
  }
}
```

### 5.3 Artifact Record

```json
{
  "artifact_id": "art_01HX...",
  "schema_version": "1",
  "created_at": "2026-06-03T12:02:00Z",
  "plan_id": "plan_01HX...",
  "context_ids": ["ctx_01HX..."],
  "source_type": "template_invocation",
  "template_name": "batch_extract_reduce",
  "template_version": "1.0.0",
  "slot_values": {},
  "instantiator": {
    "name": "rlm-scheme-template-instantiator",
    "version": "0.1.0"
  },
  "generated_scheme_ref": {
    "path": "artifacts/art_01HX/program.rkt",
    "hash": "sha256:..."
  },
  "primitives_used": ["map-async", "tree-reduce"],
  "static_profile": {
    "expected_calls_formula": "N + ceil(N/B) + ... + 1",
    "max_concurrency": 20,
    "recursive_depth": 0
  }
}
```

Artifacts should be immutable. Any edit creates a new `artifact_id`. Since
templates are Scheme, the artifact's `generated_scheme_ref` points to the
template with all `{{slot}}` markers replaced — no intermediate
representation is stored.

### 5.4 Dry-Run Record

```json
{
  "dry_run_id": "dry_01HX...",
  "schema_version": "1",
  "created_at": "2026-06-03T12:03:00Z",
  "artifact_id": "art_01HX...",
  "mode": "deterministic",
  "summary": {
    "llm_calls": 125,
    "max_concurrency": 20,
    "recursive_depth": 0,
    "critical_path_calls": 4
  },
  "call_graph": [],
  "warnings": []
}
```

### 5.5 Verification Record

```json
{
  "verification_id": "ver_01HX...",
  "schema_version": "1",
  "created_at": "2026-06-03T12:04:00Z",
  "artifact_id": "art_01HX...",
  "dry_run_id": "dry_01HX...",
  "decision": "pass | warn | fail",
  "policy": {},
  "checks": [
    {
      "name": "primitive_allowlist",
      "status": "pass",
      "message": "Only primitive runtime bindings used."
    }
  ],
  "warnings": [],
  "errors": []
}
```

### 5.6 Execution Record

```json
{
  "execution_id": "exec_01HX...",
  "schema_version": "1",
  "created_at": "2026-06-03T12:05:00Z",
  "completed_at": "2026-06-03T12:08:00Z",
  "state": "queued | running | finished | failed | cancelled | awaiting_gate | gate_rejected",
  "artifact_id": "art_01HX...",
  "plan_id": "plan_01HX...",
  "verification_id": "ver_01HX...",
  "result_ref": {
    "path": "executions/exec_01HX/result.json",
    "hash": "sha256:..."
  },
  "trace_ref": {
    "path": "executions/exec_01HX/trace.jsonl"
  },
  "metrics": {
    "elapsed_seconds": 182.4,
    "llm_calls": 125,
    "tokens": 131250,
    "max_concurrency_observed": 20
  },
  "error": null
}
```

**State transitions:**

| From | To | Trigger |
|---|---|---|
| `queued` | `running` | Executor picks up the execution |
| `running` | `finished` | `finish` primitive completes successfully |
| `running` | `failed` | Unhandled error propagates to top level |
| `running` | `cancelled` | `cancel_call(execution_id=...)` received |
| `running` | `awaiting_gate` | `gate` primitive fires |
| `awaiting_gate` | `running` | `resume_execution(decision="approve")` |
| `awaiting_gate` | `gate_rejected` | `resume_execution(decision="reject")` |
| `awaiting_gate` | `cancelled` | `cancel_call(execution_id=...)` while suspended |

Terminal states: `finished`, `failed`, `cancelled`, `gate_rejected`. No
transitions out of terminal states. An execution in `awaiting_gate` that
receives no `resume_execution` remains suspended indefinitely (see gate
timeout decision in section 20).

Additional fields for advanced features:

- `gates`: array of gate records `[{"name": "...", "status": "pending | approved | rejected", "decided_at": "...", "reason": "..."}]`
- `cache_hits`: integer count of LLM calls served from cross-execution cache
- `budget_policy_activations`: integer count of times budget policy triggered (model switch, checkpoint-and-stop)
- `chain_step_results`: array of per-step results for chain executions (intermediate context IDs and step outcomes)

### 5.7 Cache Record

```json
{
  "cache_key": "sha256:...",
  "schema_version": "1",
  "created_at": "2026-06-03T12:05:30Z",
  "instruction_hash": "sha256:...",
  "data_hash": "sha256:...",
  "model": "fast_text_model",
  "temperature": 0,
  "json_mode": false,
  "result": "...",
  "result_tokens": {
    "prompt": 1000,
    "completion": 250,
    "total": 1250
  },
  "source_execution_id": "exec_01HX...",
  "source_call_id": "call_042"
}
```

Cache records are content-addressed: `cache_key = sha256(instruction + data +
model + temperature + json_mode)`. Same inputs always produce the same key.
Cache entries are immutable — a hit returns the stored result without calling
the provider.

Temperature > 0 calls are not cached by default (non-deterministic output).
Templates can override this with `cacheable: #t` in their metadata when
non-determinism is acceptable for caching (e.g., the template handles
variability via validation).

---

## 6. Slot Substitution Model

Templates are Scheme files with `{{slot_name}}` markers. The instantiator fills
these markers with concrete values to produce executable artifacts. There is
no intermediate node graph or resolved template representation.

### Slot Markers

Slots use double-brace syntax: `{{slot_name}}`. The instantiator substitutes
each marker with the corresponding value from the template invocation's
`slot_values`. Markers can appear anywhere in the Scheme body where a
literal value would be valid:

- String positions: `#:instruction {{map_instruction}}` → `#:instruction "Extract ACE2 mentions..."`
- Numeric positions: `#:max-concurrent {{max_concurrent}}` → `#:max-concurrent 20`
- Boolean positions: `#:json {{json_mode}}` → `#:json #t`
- Identifier positions: `#:model {{map_model}}` → `#:model "fast_text_model"`

### Substitution Rules

1. All `{{slot}}` markers must have corresponding values in `slot_values`.
   Missing required slots are instantiation errors.
2. Slot values are type-checked against the template's `slot_schema` before
   substitution.
3. String values are escaped and quoted. Numeric and boolean values are
   inserted as Scheme literals. Context IDs are inserted as quoted strings.
4. Substitution is safe — values cannot inject arbitrary Scheme code. The
   instantiator rejects slot values that contain unbalanced parentheses, Scheme
   keywords, or other code injection attempts.
5. After substitution, the result must be syntactically valid Scheme that
   uses only primitive runtime bindings and instantiator-owned helpers.

### Context References

Templates access loaded context data via the `__context-ref` helper:

```scheme
(define items (__context-ref "{{context_id}}" "{{items_path}}"))
```

`__context-ref` takes a context ID and a JSONPath expression
([RFC 9535](https://www.rfc-editor.org/rfc/rfc9535)) and returns the
extracted data at runtime. The instantiator validates that `context_id` slots
contain valid context ID patterns and that `items_path` slots contain valid
JSONPath expressions.

### Allowed Primitives In Templates

Template bodies may only use the primitive runtime bindings listed in
section 9. Specifically:

- LLM calls: `llm-query`, `llm-query-async`
- Await: `await`, `await-all`, `await-any`
- Parallel: `map-async`, `parallel`, `race`
- Reduction: `tree-reduce`, `fold-sequential`
- Control: `sequence`, `choose`, `iterate-until`
- Delegation: `recursive-spawn`
- Modifiers: `memoized`, `with-validation`, `try-fallback`
- State: `checkpoint`, `restore`, `tokens-used`, `rate-limits`, `heartbeat`
- Compute: `py-exec`, `py-eval`, `py-call`, `py-set!`
- Helpers: `__context-ref`, `__join-json`, `finish`, `syntax-e`, `datum->syntax`

Explicitly disallowed in templates:

- any name not in the primitive runtime basis (section 9),
- string `eval`,
- shell commands,
- filesystem access outside declared context/artifact/checkpoint stores.

---

## 7. Example Template

Templates are `.rkt` files with `define-meta` forms for metadata and Scheme
code for the executable body — one language throughout.

File:

```text
templates/batch_extract_reduce.rkt
```

Template:

```scheme
;; --- Metadata ---

(define-meta name "batch_extract_reduce")
(define-meta version "1.0.0")
(define-meta summary
  "Run independent extraction over many items, then synthesize results with tree reduction.")
(define-meta task-shapes '(Batch Synthesize Composite))
(define-meta data-shapes '(FlatList ChunkedSingular Tabular))
(define-meta output-shape 'one)

(define-meta trigger
  '((> item_count 1)
    (eq? independent #t)
    (eq? output_type 'one)
    (eq? has_second_phase #t)))

(define-meta reject
  '((and (eq? ordered #t) (eq? order_sensitive #t))
    (eq? requires_pairwise_comparison #t)))

(define-meta slots
  '((context_id         (type string) (pattern "^ctx_") (required #t))
    (items_path         (type string) (default "$.items"))
    (map_instruction    (type string) (min-length 10) (required #t))
    (reduce_instruction (type string) (min-length 10) (required #t))
    (map_model          (type string) (default "fast_text_model"))
    (reduce_model       (type string) (default "quality_text_model"))
    (max_concurrent     (type integer) (min 1) (max 50) (default 20))
    (branch_factor      (type integer) (min 2) (max 10) (default 5))
    (json_mode          (type boolean) (default #f))
    (checkpoint_every   (type integer) (nullable #t) (min 1) (default #f))))

(define-meta structural-profile
  '((expected-calls "N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1")
    (critical-path  "1 + ceil(log_B(N))")
    (max-concurrency-slot max_concurrent)
    (recursive-depth 0)
    (uses-python-bridge #f)
    (uses-multimodal #f)))

(define-meta verification-rules
  '(context_id_exists
    items_path_resolves_to_list
    map_model_supports_json_if_json_mode
    expected_calls_within_policy
    max_concurrency_within_policy
    only_primitive_bindings))

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

(define-meta streamable #t)
(define-meta cacheable #t)

(define-meta budget-policy
  '((on-low-budget   switch-model)
    (low-budget-threshold 0.20)
    (fallback-model  "fast_text_model")
    (on-exhausted    checkpoint-and-stop)))

(define-meta gates
  '((review_extractions
      (description "Review extraction results before synthesis")
      (required #f))))

(define-meta uses-llm-generated-code #f)

(define-meta examples
  '(((task "Extract claims from papers and synthesize a literature review.")
     (slot_values
       (items_path "$.papers")
       (map_instruction "Extract the core claim, evidence, and uncertainty as JSON.")
       (reduce_instruction "Synthesize the extracted claims into a literature review.")))))

;; --- Body ---
;; Scheme code with {{slot}} markers. The instantiator substitutes slot values
;; to produce the executable artifact.

(define items (__context-ref "{{context_id}}" "{{items_path}}"))

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

;; Optional gate — suspends for human review when policy requires it.
(gate "review_extractions" extracted
      #:message "Review extraction results before synthesis.")

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

Example template invocation (what the planner produces):

```json
{
  "template_name": "batch_extract_reduce",
  "template_version": "1.0.0",
  "slot_values": {
    "context_id": "ctx_01HX...",
    "items_path": "$.papers",
    "map_instruction": "Extract ACE2 mentions, evidence, and uncertainty as JSON.",
    "reduce_instruction": "Synthesize ACE2 findings into a report with citations to source IDs.",
    "map_model": "fast_text_model",
    "reduce_model": "quality_text_model",
    "max_concurrent": 20,
    "branch_factor": 5,
    "json_mode": true
  }
}
```

Example instantiated artifact (after slot substitution):

```scheme
(define items (__context-ref "ctx_01HX..." "$.papers"))

(define extracted
  (map-async
    (lambda (item)
      (llm-query-async
        #:instruction "Extract ACE2 mentions, evidence, and uncertainty as JSON."
        #:data item
        #:model "fast_text_model"
        #:json #t))
    items
    #:max-concurrent 20))

(define synthesized
  (tree-reduce
    (lambda group
      (syntax-e
        (llm-query
          #:instruction "Synthesize ACE2 findings into a report with citations to source IDs."
          #:data (__join-json group)
          #:model "quality_text_model")))
    extracted
    #:branch-factor 5))

(finish synthesized)
```

The instantiated artifact is the template with all `{{slot}}` markers replaced
by concrete values. It uses only primitive runtime bindings and
instantiator-owned helper bindings.

---

## 8. Model Registry

Templates should refer to model aliases, not hardcoded provider model names.
The server resolves aliases at instantiation or execution time through a model
registry.

The registry is a JSON configuration file at `config/models.json` (path
configurable via environment variable `RLM_MODEL_REGISTRY`). It is loaded once
at server startup and can be reloaded via `reset_runtime(scope="config")`.
There is no MCP tool to modify it — registry changes are an operator concern,
not an agent concern.

Example registry (`config/models.json`):

```json
{
  "schema_version": "1",
  "aliases": {
    "fast_text_model": {
      "provider": "openai",
      "model": "configured-fast-model",
      "capabilities": ["text", "json"],
      "max_context_tokens": 128000,
      "supports_temperature": true,
      "cost_tier": "low"
    },
    "quality_text_model": {
      "provider": "openai",
      "model": "configured-quality-model",
      "capabilities": ["text", "json"],
      "max_context_tokens": 128000,
      "supports_temperature": true,
      "cost_tier": "high",
      "fallback": "fast_text_model"
    },
    "vision_model": {
      "provider": "openai",
      "model": "configured-vision-model",
      "capabilities": ["text", "json", "image"],
      "max_context_tokens": 128000,
      "supports_temperature": true,
      "cost_tier": "high"
    }
  },
  "defaults": {
    "planner": "quality_text_model",
    "map": "fast_text_model",
    "reduce": "quality_text_model",
    "vision": "vision_model"
  }
}
```

Verification should check aliases against the registry:

- alias exists,
- required capabilities are present,
- JSON mode is supported when requested,
- image inputs target an image-capable alias,
- context estimates fit the alias context window,
- temperature/max-token settings are compatible with the resolved model.
- fallback alias exists and has compatible capabilities when budget-policy
  references it.

Provider model names should live in configuration, not in templates or planner
prompts. Documentation examples should use aliases unless they are describing
provider configuration.

---

## 9. Runtime Basis

The Racket runtime should be small and primitive-only. Higher-level patterns
are expressed as template-level compositions of primitives.

### Keep As Runtime Primitives

| Group | Primitive | Notes |
|---|---|---|
| LLM | `llm-query` | Synchronous call; returns syntax-wrapped result. |
| LLM | `llm-query-async` | Starts async call; returns handle only. |
| Await | `await` | Await one handle. |
| Await | `await-all` | Await all handles. |
| Await | `await-any` | Await first completed handle and return remaining handles. |
| Parallel | `map-async` | Rolling-window fan-out with `#:max-concurrent`. |
| Parallel | `parallel` | Concurrent async thunk execution, not sequential `map`. |
| Parallel | `race` | First completed async thunk wins. |
| Reduction | `tree-reduce` | Recursive associative reduction. |
| Reduction | `fold-sequential` | Ordered accumulation. |
| Control | `sequence` | Function pipeline. |
| Control | `choose` | Conditional dispatch. |
| Control | `iterate-until` | Bounded loop. |
| Control | `gate` | Suspend execution for human review. |
| Delegation | `recursive-spawn` | Nested orchestration with global depth limit. |
| Modifier | `memoized` | Cache by explicit key. |
| Modifier | `with-validation` | Wrap result validation. |
| Modifier | `try-fallback` | Error recovery. |
| State | `checkpoint` / `restore` | Durable partial results. |
| State | `tokens-used` / `rate-limits` | Runtime accounting. |
| State | `heartbeat` | Keep long executions alive. |
| Compute | `py-exec` / `py-eval` / `py-call` / `py-set!` | Controlled Python bridge for parsing, aggregation, and local computation. |

### Primitive Signatures And Semantics

The signatures below are the target public bindings inside instantiated
Scheme artifacts. Some helper bindings can be private and prefixed with `__`.

#### `llm-query`

```scheme
(llm-query #:instruction string
           #:data any
           #:model string
           #:recursive boolean
           #:temperature number-or-#f
           #:max-tokens integer-or-#f
           #:json boolean
           #:image image-or-#f
           #:images list)
  -> syntax-object
```

Semantics:

- calls the host model provider synchronously,
- returns a syntax object,
- decrements token budget after response,
- logs call metadata and provenance,
- supports image inputs only when model capabilities allow them,
- supports `#:recursive #t` only under global recursion policy.

Instantiation rule: use this for reduce/synthesis/refinement steps where the next
Scheme expression needs the completed value immediately.

#### `llm-query-async`

```scheme
(llm-query-async #:instruction string
                 #:data any
                 #:model string
                 #:temperature number-or-#f
                 #:max-tokens integer-or-#f
                 #:json boolean
                 #:image image-or-#f
                 #:images list)
  -> async-handle
```

Semantics:

- dispatches model call through the host future pool,
- returns immediately with an opaque async handle,
- does not support recursive calls,
- validates image/model compatibility before dispatch when possible,
- records call metadata in the execution registry.

Instantiation rule: use this inside `map-async`, `parallel`, and `race` bodies.

#### `await`

```scheme
(await async-handle) -> syntax-object
```

Semantics:

- blocks until one handle completes,
- propagates cancellation and provider errors,
- decrements token budget when the real usage is known,
- wraps result as syntax.

#### `await-all`

```scheme
(await-all (list async-handle ...)) -> (list string ...)
```

Semantics:

- waits for all handles concurrently,
- returns unwrapped strings in input order,
- records batch wait in trace.

If syntax preservation is needed, the runtime may also expose a private
instantiator-only `__await-all-syntax`.

#### `await-any`

```scheme
(await-any (list async-handle ...)) -> (values string (list async-handle ...))
```

Semantics:

- waits for the first completed handle,
- returns the completed unwrapped string and reconstructed remaining handles,
- must be deterministic in dry-run mode: exactly one requested pending handle
  completes per `await-any` call.

#### `map-async`

```scheme
(map-async (lambda (item) async-handle)
           items
           #:max-concurrent integer-or-#f)
  -> (list string ...)
```

Semantics:

- preserves input order,
- maintains a rolling concurrency window,
- validates that the lambda returns async handles,
- reports progress and heartbeats during long fan-outs,
- propagates per-item errors according to instantiator-selected error policy.

Error policy should be explicit in the template:

```json
{
  "on_item_error": "fail | collect | fallback",
  "checkpoint_every": 25
}
```

#### `parallel`

```scheme
(parallel (list (lambda () async-handle) ...)
          #:max-concurrent integer-or-#f)
  -> (list string ...)
```

Semantics:

- genuinely concurrent, not sequential thunk invocation,
- accepts thunks that return async handles,
- preserves strategy order in output,
- uses the same concurrency policy as `map-async`.

The instantiator should reject `parallel` bodies that call synchronous
`llm-query` directly unless it rewrites them into async equivalents.

#### `race`

```scheme
(race (list (lambda () async-handle) ...)) -> string
```

Semantics:

- launches all candidates,
- returns first completed result,
- cancels or abandons remaining handles according to runtime policy,
- records losing handles in trace as cancelled or ignored.

#### `tree-reduce`

```scheme
(tree-reduce reducer items
             #:branch-factor integer
             #:leaf-fn procedure)
  -> any
```

Semantics:

- rejects empty input,
- optionally applies `leaf-fn`,
- groups items by branch factor,
- applies reducer recursively until one result remains,
- suitable only for associative or order-insensitive reductions.

Call estimate:

```text
N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1
```

The `N` term is present when the tree reduction follows a map/leaf LLM call.
For a pure reduce over already computed values, omit the leaf-call term.

#### `fold-sequential`

```scheme
(fold-sequential reducer initial items) -> any
```

Semantics:

- processes items in order,
- passes accumulator and item to reducer,
- has high critical-path latency,
- appropriate for order-sensitive synthesis and rolling summaries.

#### `sequence`

```scheme
(sequence fn1 fn2 ...) -> (lambda (input) output)
```

Semantics:

- left-to-right function composition,
- used by instantiator for multi-phase templates,
- can be generated as `let*` when simpler.

#### `choose`

```scheme
(choose predicate then-fn else-fn) -> procedure
```

Semantics:

- routes based on deterministic predicate or predicate function,
- should not hide model calls inside predicates unless declared.

#### `iterate-until`

```scheme
(iterate-until step-fn predicate init #:max-iter integer) -> any
```

Semantics:

- bounded loop,
- stops when predicate returns true or `max-iter` is reached,
- used for refine/critique loops,
- dry-run reports worst-case iteration count unless predicate is statically
  known.

#### `gate`

```scheme
(gate name value #:message string #:required boolean) -> value
```

Semantics:

- suspends execution and transitions state to `"awaiting_gate"`,
- stores the gate name, value preview, and message in the execution record,
- waits for `resume_execution` to approve or reject,
- on approve, returns `value` unchanged — downstream code sees the same data,
- on reject, raises a structured gate-rejection error,
- `#:required #t` means the gate always fires; `#:required #f` means it fires
  only when the execution policy includes `require_gates: true`,
- in dry-run mode, gates are recorded in the simulation output but do not
  suspend — the dry-run reports `"gates": ["review_extractions"]` so the
  agent knows execution will pause.

Gate is a pass-through — it does not transform data. Templates place gates
between phases where human review adds value (e.g., between extraction and
synthesis).

**Gate lifecycle:**

```text
Template declares gate       →  Verification checks gate names match
  ↓                               define-meta gates declarations
Execution reaches gate call  →  State transitions to awaiting_gate
  ↓                               Value + message stored in execution record
Agent calls resume_execution
  ├─ decision=approve        →  State returns to running, gate returns value
  └─ decision=reject         →  State transitions to gate_rejected,
                                  structured error raised, partial work preserved
```

Gates with `#:required #f` only fire when the execution policy includes
`require_gates: true`. This allows the same template to run interactively
(with gates) or autonomously (skipping gates). In dry-run mode, gates are
recorded but never suspend — the response lists gate names so the agent
knows to expect pauses during real execution.

#### `recursive-spawn`

```scheme
(recursive-spawn strategy-ref) -> (lambda (data) syntax-object)
```

Semantics:

- delegates to a nested artifact/sub-strategy,
- global recursion depth is enforced host-side,
- inherits explicitly passed context refs only,
- appears as recursive depth in dry-run and trace.

Do not include a public `#:depth` keyword unless it controls real enforcement.

#### `memoized`

```scheme
(memoized fn #:key-fn key-fn) -> procedure
```

Semantics:

- caches within one execution unless the template requests persistent caching,
- key function must be deterministic,
- trace should record cache hits/misses for LLM-call avoidance.

#### `with-validation`

```scheme
(with-validation fn validator) -> procedure
```

Semantics:

- runs `fn`,
- validates result,
- returns result or raises structured validation error,
- should include schema path or validation rule in errors.

#### `try-fallback`

```scheme
(try-fallback primary-fn fallback-fn) -> procedure
```

Semantics:

- catches declared error classes,
- executes fallback with the original args,
- records both primary failure and fallback result in trace.

#### `checkpoint` / `restore`

```scheme
(checkpoint key value) -> value
(restore key) -> value-or-#f
```

Semantics:

- keys are namespaced by execution/artifact unless explicitly shared,
- values must be JSON-serializable,
- checkpoints should be visible in execution trace and status.

#### `tokens-used` / `rate-limits` / `heartbeat`

```scheme
(tokens-used) -> hash
(rate-limits) -> hash
(heartbeat) -> void
```

Semantics:

- expose host accounting to instantiated strategies,
- `heartbeat` should also be emitted automatically by long primitives.

#### Python Bridge

```scheme
(py-set! name value) -> void
(py-exec code) -> string
(py-eval expr) -> any
(py-call ref method . args) -> any
```

Semantics:

- runs in an isolated Python subprocess,
- receives values over JSON, not string interpolation,
- can access declared context values,
- cannot access MCP server internals or Racket scaffold bindings,
- should be used only by trusted templates.

Allowed Python bridge use cases:

- JSON parsing,
- schema validation,
- table aggregation,
- grouping,
- deduplication,
- statistics,
- deterministic uncertainty filtering.

Disallowed by default:

- arbitrary filesystem writes outside artifact/checkpoint stores,
- subprocess/shell execution,
- network access,
- importing project secrets,
- mutating durable records except through host APIs.

### Error Propagation Model

Each primitive must define how errors are handled. Templates declare an error
policy per primitive usage in their metadata; the instantiator validates the
corresponding error handling in the template body.

**Error policies** (declared per-node in templates):

| Policy | Behavior |
|---|---|
| `fail_fast` | First error aborts the entire node. Partial results are discarded. Default for most primitives. |
| `collect` | Errors are collected alongside successful results. The node completes and returns a mixed result list with error markers. Consumer nodes must handle error entries. |
| `fallback` | On error, execute the declared fallback function for that item. If fallback also fails, apply `fail_fast` or `collect` as secondary policy. |

**Per-primitive error semantics:**

| Primitive | Default policy | Error behavior |
|---|---|---|
| `llm-query` | `fail_fast` | Provider errors, timeouts, and token budget exhaustion propagate as structured errors. Rate-limit errors trigger retry (see retry policy). |
| `llm-query-async` | `fail_fast` | Error is captured in the future. Surfaces when the handle is awaited. |
| `await` | `fail_fast` | Re-raises the error from the awaited handle. |
| `await-all` | `fail_fast` | If any handle errored, raises the first error. With `collect`, returns error markers in position (see error marker format below). |
| `await-any` | `fail_fast` | If the completed handle errored, raises immediately. Remaining handles are still cancellable. |
| `map-async` | `fail_fast` | First item error cancels remaining in-flight items and raises. With `collect`, continues all items and returns mixed results. With `fallback`, retries failed items with fallback function. |
| `parallel` | `fail_fast` | First thunk error cancels remaining and raises. With `collect`, completes all thunks. |
| `race` | `fail_fast` | If first completed is an error, raises immediately. Remaining are cancelled. Does not wait for a successful result. |
| `tree-reduce` | `fail_fast` | Error at any level aborts the tree. Partial reductions from completed levels are lost. |
| `fold-sequential` | `fail_fast` | Error on any item aborts. Accumulator state up to the error is available in checkpoint if checkpointing is enabled. |
| `iterate-until` | `fail_fast` | Error on any iteration aborts. Last successful state is available if checkpointed. |
| `recursive-spawn` | `fail_fast` | Sub-artifact error propagates to parent. |
| `with-validation` | `fail_fast` | Validation failure is a structured error, not an exception. The template decides whether to retry or propagate. |
| `try-fallback` | N/A | This IS the error recovery primitive. Primary error triggers fallback. If fallback also fails, the error propagates. |

**Error marker format:** When a primitive uses the `collect` error policy,
failed items are represented in the result list as error marker objects:

```json
{
  "__error": true,
  "message": "Provider returned 500: Internal Server Error",
  "item_index": 42,
  "call_id": "call_043",
  "retries_attempted": 3,
  "error_type": "server_error"
}
```

Consumer nodes (e.g., `tree-reduce` after `map-async`) must handle error
markers explicitly — they are not filtered out automatically. Templates
should use `with-validation` or Python bridge filtering to separate
successful results from error markers before passing to reduction phases.

**Rate-limit and transient errors** are handled by the retry policy (see
section 11.6) before the error policy applies. Only non-retryable errors reach
the per-primitive error handling.

**Checkpoints and partial recovery:** Templates that process large item lists
should declare `checkpoint_every: N`. When `map-async` or `fold-sequential`
checkpoints, a `restore` call on re-execution skips already-completed items.
This interacts with error policies — `fail_fast` with checkpointing means the
failed execution can be retried from the last checkpoint, not from scratch.

### Common Patterns As Template Compositions

The runtime does not provide pre-composed patterns. Common orchestration
patterns are expressed as template-level compositions of primitives:

| Pattern | Composed from |
|---|---|
| Fan-out-aggregate | `map-async` + `tree-reduce` or `fold-sequential` |
| Critique-refine | `iterate-until` with generate/critique/refine state |
| Ensemble | `parallel` + aggregation logic |
| Vote | `parallel` + majority/plurality/consensus selection |
| Tiered review | cheap `map-async` → filter → expensive review |
| Active learning | cheap `map-async` → uncertainty filter → expensive `map-async` |
| Code interpreter | `llm-query` + `py-exec` + `with-validation` + `iterate-until` |

### Privileged Runtime Hooks

The runtime does not expose `unsafe-interpolate`, `unsafe-overwrite`, or
`unsafe-exec-sub-output` as public bindings. If the instantiator needs
privileged hooks, they are private host-generated forms that templates
cannot request directly.

---

## 10. Preserved Feature Inventory

The rewrite should not accidentally lose these current capabilities.

### Context Handling

Preserve large context support as a first-class feature, not just a variable
called `context`.

Required behavior:

- context objects get stable `context_id`s,
- named contexts remain possible,
- metadata captures DataShape, item count, modality, chunking, independence,
  token estimates, and source information,
- Python bridge receives context when needed,
- planner can classify from metadata without reading all data,
- artifacts reference contexts by ID instead of embedding large payloads.

### Hygiene And Provenance

Preserve the core RLM-Scheme safety idea:

- LLM results are syntax-wrapped by default,
- unwrapping is explicit and logged,
- `datum->syntax` wrapping is explicit and logged,
- scope/provenance logs are attached to execution traces,
- generated artifacts cannot overwrite runtime scaffold bindings.

The greenfield implementation can simplify mechanics, but it should keep the
observable guarantee: model output is data until explicitly unwrapped.

### Async Execution

Preserve the async callback architecture:

- `llm-query-async` returns handles,
- `await`, `await-all`, and `await-any` work with real futures,
- `map-async` uses bounded concurrency and rolling completion,
- long fan-outs report progress and heartbeat,
- cancellation works for queued, active, and nested calls.

`parallel` must be genuinely concurrent in the rewrite. It should require
thunks that return async handles or instantiate into equivalent async structure.

### Runtime Accounting

Preserve:

- token usage tracking,
- scoped token budgets,
- rate-limit header tracking,
- retry behavior for rate limits/transient failures,
- per-call model/latency/tokens/error records,
- execution summaries.

### Long-Running Workflow Support

Preserve:

- checkpoint/restore,
- heartbeat,
- progress messages,
- call registry,
- cancellation,
- execution trace retrieval after completion or failure.

These are essential for large recursive workflows.

### Multimodal Support

Preserve image support:

- file paths, data URLs, and base64 images,
- MIME validation by magic bytes,
- max image size checks,
- warnings for too many images,
- model capability checks in templates.

### Python Bridge

Keep a controlled Python bridge because it is useful for:

- JSON parsing and validation,
- tabular aggregation,
- statistics,
- deterministic filtering,
- grouping and deduplication,
- local computation that should not consume LLM tokens.

The bridge should not become an unrestricted public escape hatch. Templates
should declare when Python computation is required, and the instantiator should
generate constrained bridge calls.

### Recursive Delegation

Preserve recursive LLM orchestration, but make it artifact-aware:

- recursive calls instantiate sub-strategies, not arbitrary model-written Scheme,
- global recursion depth is enforced in one place,
- nested executions inherit context references intentionally,
- recursive depth appears in dry-run and trace output.

Remove dead or misleading APIs such as `recursive-spawn #:depth` unless the
keyword is wired to real enforcement.

---

## 11. Architecture Components

### 11.1 Durable Store

Start with filesystem JSON records for simplicity, but design the schema so it
can move to SQLite, PGlite, or another embedded database.

Store:

- contexts,
- plans,
- artifacts,
- dry-runs,
- verification records,
- executions,
- traces,
- checkpoints,
- cache entries.

Every record should include:

- ID,
- version,
- creation timestamp,
- parent IDs,
- schema version,
- status,
- warnings/errors.

**State lifecycle and persistence scoping:**

All durable records are persistent by default — they survive server restarts
and `reset_runtime` calls. The `reset_runtime` tool accepts a `scope` parameter
that controls what is cleared:

| Scope | Clears | Preserves |
|---|---|---|
| `"sandbox"` | Racket sandbox state, in-memory caches, active call handles. | All durable records (contexts, plans, artifacts, dry-runs, verifications, executions, traces, checkpoints). |
| `"session"` | Everything in `"sandbox"` plus execution records and traces from the current server session. | Contexts, plans, artifacts, dry-runs, verifications, and checkpoints from prior sessions. |
| `"all"` | All durable records and sandbox state. Fresh start. | Nothing. |
| `"cache"` | LLM result cache entries. | All durable records, sandbox state, and template catalog. |
| `"config"` | Reloads model registry and template catalog from disk. | All durable records and sandbox state. |

Contexts, plans, and artifacts are long-lived — they represent reusable work
products. An agent can re-execute an artifact multiple times, creating new
execution records each time. Dry-run and verification records are linked to
specific artifacts and remain valid as long as the artifact exists.

Execution records and traces are the most voluminous. They can be pruned by
age or count without affecting the ability to create new executions from
existing artifacts.

Checkpoints are scoped to their execution but survive `"sandbox"` resets so
that failed long-running workflows can be resumed.

### 11.2 Slot Substitution

See section 6 for the slot substitution model, allowed primitives, and
disallowed operations. The instantiator fills `{{slot}}` markers in template
Scheme code with concrete values — there is no intermediate node graph or
resolved template representation.

### 11.3 Template Catalog

See section 7 for the full template schema, and section 15 for the initial
template catalog. Templates live as `.rkt` files with `define-meta` forms:

```text
templates/
  batch_extract_reduce.rkt
  batch_map.rkt
  ordered_synthesis_fold.rkt
  compare_candidates.rkt
  refine_until_valid.rkt
  tiered_review.rkt
  ...
```

Template validation is a developer/CI concern. Runtime verification assumes
trusted templates are structurally valid, but still verifies filled artifacts.

### 11.4 Template Instantiation (internal library)

The instantiator validates slot values and substitutes them into template Scheme
code to produce immutable artifacts. It is invoked internally by
`dry_run_strategy` and `execute_strategy` — there is no dedicated
`compile_strategy` MCP tool.

Responsibilities:

- parse template `define-meta` forms and Scheme body,
- validate slot values against the template's `slots` metadata,
- substitute `{{slot}}` markers with safe, type-appropriate values,
- reject values that could inject arbitrary Scheme code,
- verify all markers are resolved and only allowed primitives are used,
- calculate static structural profiles from template metadata,
- hash the resulting Scheme code,
- store artifact metadata.

The instantiator should be deterministic: same inputs, same artifact hash.

The instantiator does NOT:

- generate Scheme code from a node graph,
- translate between an IR and Scheme,
- produce source maps (the artifact IS the template with filled slots,
  so line numbers correspond directly).

### 11.5 Racket Runtime

The Racket runtime should be a sandboxed execution engine, not the planning
interface.

Responsibilities:

- evaluate instantiated Scheme artifacts,
- enforce resource limits,
- preserve syntax hygiene,
- call back to Python host for LLM/Python/checkpoint/rate-limit operations,
- emit stdout, scope logs, and trace events,
- protect scaffold bindings.

### 11.6 Python Host

The Python MCP server owns orchestration state around the Racket runtime.

Responsibilities:

- MCP tools,
- durable object store,
- OpenAI or model-provider calls,
- async futures and cancellation,
- dry-run mode,
- verification,
- progress/status reporting,
- trace assembly,
- checkpoint persistence,
- image resolution,
- Python bridge process management.

**Retry policy:** The host retries transient provider errors (rate limits,
server errors, timeouts) before surfacing errors to the per-primitive error
policy. The retry policy is configurable via `config/retry.json` (path
configurable via `RLM_RETRY_CONFIG`):

```json
{
  "schema_version": "1",
  "defaults": {
    "max_retries": 3,
    "initial_backoff_seconds": 1.0,
    "max_backoff_seconds": 60.0,
    "backoff_multiplier": 2.0,
    "retryable_status_codes": [429, 500, 502, 503, 504],
    "retryable_error_types": ["rate_limit", "timeout", "server_error"]
  },
  "per_model_overrides": {
    "fast_text_model": {
      "max_retries": 5,
      "initial_backoff_seconds": 0.5
    }
  }
}
```

Retries are per-call, not per-execution. Each retry is logged in the
execution trace. Rate-limit retries use the `Retry-After` header when
available, falling back to exponential backoff. Token budget is not consumed
by failed attempts. If all retries are exhausted, the error propagates to
the per-primitive error policy (see section 9, Error Propagation Model).

**Error type mapping:** The host classifies provider errors into retry
categories using this mapping:

| HTTP status / exception | Error type | Retry behavior |
|---|---|---|
| 429 (Too Many Requests) | `rate_limit` | Use `Retry-After` header if present, else exponential backoff |
| 408 (Request Timeout), `ETIMEDOUT`, `ECONNRESET` | `timeout` | Exponential backoff, same request |
| 500, 502, 503, 504 | `server_error` | Exponential backoff, same request |
| 400, 401, 403, 404, 422 | `client_error` | **Not retried** — propagate immediately |
| Network unreachable, DNS failure | `network_error` | **Not retried** — propagate immediately |
| Provider SDK exception with no HTTP status | Classified by exception type: `RateLimitError` → `rate_limit`, `APITimeoutError` → `timeout`, `APIConnectionError` → `server_error`, all others → `client_error` | Per classification |

`per_model_overrides` in the retry config are **merged** with defaults
(override keys replace default keys; unspecified keys inherit defaults).

**Progress reporting:** Long-running executions report progress via MCP
notifications (server-initiated messages on the MCP transport). The protocol:

1. The host emits `notifications/progress` messages during execution. Each
   message includes `execution_id`, `node_id`, `completed_calls`,
   `total_expected_calls`, `elapsed_seconds`, and optional `message`.
2. Progress interval is configurable per-execution via
   `runtime_options.progress_interval_seconds` (default: 2 seconds).
3. Primitives that process many items (`map-async`, `tree-reduce`) emit
   progress after each completed item or batch.
4. `heartbeat` calls from Racket artifacts also trigger progress notifications.
5. Agents that don't support notifications can poll `get_status(execution_id)`
   for the same information.

**Streaming transport:** Streaming uses the MCP SDK's built-in notification
mechanism: `ctx.session.send_notification()` (Python FastMCP). The server
calls this method to push `notifications/partial_result` messages over the
existing MCP transport (stdio or SSE) — no separate channel is needed.
Clients that don't support notifications (or ignore them) still receive the
final result normally; streaming is best-effort.

**Streaming partial results:** When `stream=true`, the host emits
`notifications/partial_result` messages in addition to progress counts.
For `map-async`, each completed item result is emitted. For `tree-reduce`,
each completed reduction level is emitted. The host controls emission
rate — high-throughput fan-outs may batch multiple items per notification
to avoid flooding the transport.

**LLM result cache:** Before dispatching each LLM call to the provider,
the host checks the content-addressed cache (see section 11.7). On hit,
the cached result is returned immediately — no provider call, no token
consumption, no latency. Cache hits are logged in the trace with
`"source": "cache"`. The host also checks remaining token budget before
each dispatch and activates the template's budget policy when the
threshold is crossed (see section 11.8 for budget-aware degradation).

### 11.7 LLM Result Cache

The server maintains a content-addressed cache of LLM call results that
persists across executions. When the same LLM call (identical instruction,
data, model, temperature, and json_mode) occurs in a later execution, the
cached result is returned without calling the provider.

**Key computation:**

```text
cache_key = sha256(
    canonical_json(instruction) +
    canonical_json(data) +
    model_alias +
    str(temperature) +
    str(json_mode)
)
```

`canonical_json()` produces deterministic JSON: keys sorted lexicographically,
no whitespace, no trailing commas, numbers in their shortest representation
(no trailing zeros), Unicode escaped as `\uXXXX`. This follows the spirit of
RFC 8785 (JCS). Use Python's `json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)` as the reference implementation.

**Cache behavior:**

- Cache entries are immutable — same key always returns same result.
- Temperature = 0 calls are cached by default. Temperature > 0 calls are
  not cached unless the template declares `cacheable: #t`.
- Cache hits are logged in the execution trace with `"source": "cache"`.
- Cache hits do not consume token budget.
- `reset_runtime(scope="cache")` clears all cache entries.

**Cache storage:** Alongside the durable store (filesystem or DB).

**Eviction:** V1 uses manual-only eviction via `reset_runtime(scope="cache")`.
No automatic LRU, TTL, or size-based eviction. This keeps the implementation
simple and predictable — cached results are permanent until explicitly cleared.
Automatic eviction can be added later if storage becomes a concern.

**Dry-run interaction:** When a dry-run instantiates an artifact that
matches a previously-executed one, the dry-run response includes
`cache_hits_expected` with the count of calls that would hit the cache.
Cost estimates are adjusted accordingly.

### 11.8 Template Chaining

A plan can describe a template chain — a linear sequence of template
invocations where each step's output feeds as input context to the next.

**Chain descriptor:** The plan record's `recommended` field has
`"kind": "template_chain"` with a `"steps"` array. Each step is a
standard template invocation. Steps reference the previous step's output
via `"$previous"` in their `slot_values`.

**Execution semantics:**

1. Steps run sequentially.
2. After step N finishes, its output is stored as an intermediate context
   (`ctx_auto_N`) scoped to the execution.
3. Step N+1's `"$previous"` references are resolved to `ctx_auto_N`.
4. Step N+1 is instantiated and executed.
5. The final step's output is the chain's result.

Intermediate contexts are available in the execution trace and via
`get_execution_trace`, but do not appear in the agent's context namespace
(they are not returned by `get_context` unless explicitly requested).

**Failure and retry:** If step N fails, the chain stops. Completed steps
are checkpointed. Retrying the execution resumes from the failed step
using checkpointed intermediate contexts — completed steps are not
re-executed.

**Gates in chains:** A gate at the end of step N suspends the chain
before step N+1 begins. `resume_execution` continues the chain.

**Dry-run:** Each step is instantiated and dry-run independently. The
aggregate response includes per-step estimates and totals. Output schema
compatibility between adjacent steps is validated at dry-run time (see
section 12).

**Scope:** Chains are linear pipelines only. Conditional branching and
parallel fan-out stay with the outer agent. This keeps chain execution
simple and predictable while the agent handles adaptive decisions.

### 11.9 Code Interpreter Pattern

The code interpreter pattern allows LLM calls to generate Python code that
the bridge executes. This is not a new primitive — it is a composition of
existing primitives (`llm-query` → `py-exec` → `with-validation` →
`iterate-until`) elevated to a declared template capability.

**Template declaration:**

```scheme
(define-meta uses-llm-generated-code #t)
(define-meta code-generation-policy
  '((max-code-length 500)
    (allowed-imports (json csv statistics collections re))
    (max-retries 2)
    (sandbox-timeout-seconds 10)))
```

**Policy gating:** Verification checks `uses-llm-generated-code` against
the execution policy. If `policy.allow_llm_generated_code` is `false`
(the default), verification fails with a clear message. The agent must
explicitly opt in by setting `allow_llm_generated_code: true` in the
policy passed to `execute_strategy`.

**Python bridge hardening:** When executing LLM-generated code, the bridge
applies stricter limits than for pre-written template code:

- **Import allowlist** (only modules declared in `code-generation-policy`).
  Default allowlist: `json`, `csv`, `statistics`, `collections`, `re`,
  `math`, `itertools`, `functools`, `datetime`, `decimal`, `fractions`,
  `string`, `textwrap`, `operator`, `copy`, `pprint`. Templates can
  restrict this further via `allowed-imports` in `code-generation-policy`.
  Any `import` or `__import__` of a module not on the allowlist raises
  `ImportError` immediately.
- **Execution timeout:** Default 10 seconds (configurable via
  `sandbox-timeout-seconds` in `code-generation-policy`). Pre-written
  template code uses 30 seconds.
- **Output size limit:** 1 MB stdout capture. Larger output is truncated
  with a warning in the trace.
- No filesystem, network, or subprocess access (same as existing bridge).

**Standard pattern:**

```scheme
(iterate-until
  (lambda (state)
    (let* ((code (syntax-e
                   (llm-query
                     #:instruction (string-append "Write Python: " task "\nPrevious error: " (or (hash-ref state 'error) "none"))
                     #:data data
                     #:model code-model
                     #:json #f)))
           (exec-result (try-fallback
                          (lambda () (py-exec code))
                          (lambda () (hash 'error (current-error-message))))))
      (hash 'code code 'result exec-result 'error #f)))
  (lambda (state) (not (hash-ref state 'error)))
  (hash 'error "no attempt yet")
  #:max-iter max-retries)
```

**Dry-run interaction:** Dry-run reports `"uses_llm_generated_code": true`
and notes that cost estimates are less precise (generated code behavior
is not statically predictable).

---

## 12. Dry-Run And Verification

Dry-run and verification should be artifact-based.

### Dry-Run

Dry-run must simulate structure without real LLM calls:

- use pre-resolved fake futures for async calls. Fake LLM call results are
  deterministic empty values: `""` (empty string) for text calls, `"{}"` for
  JSON-mode calls. This ensures dry-run is fully deterministic and does not
  require random data generation,
- special-case `await-any` so exactly one pending handle completes per call,
- special-case batch await behavior deterministically,
- record fan-out, call counts, model mix, recursive depth, and estimated tokens,
- avoid shared global execution-mode state that can leak across concurrent MCP
  calls — pass dry-run context as a parameter through `send()`, not as
  mutable state on the backend instance.

**Dry-run behavior for concurrent primitives:**

`parallel` is genuinely concurrent in the rewrite. In dry-run mode, `parallel`
should behave as follows:

1. All thunks are invoked. Each returns a pre-resolved fake async handle.
2. `await-all` collects results. Since handles are pre-resolved, this is
   instant but the dry-run context records the concurrency count as
   `len(thunks)`.
3. The dry-run summary reports `max_concurrency` for this node as the thunk
   count, matching real execution behavior.

This works because `parallel` uses `await-all` (wait for all), not
`await-any` (rolling window). The `await-any` special-casing is only needed
for `map-async`'s rolling window and `race`.

`race` in dry-run: all thunks are invoked, all return pre-resolved handles.
`await-any` special-casing picks exactly one deterministically (first in
list). Remaining handles are marked cancelled in the dry-run trace.

The dry-run output should use `recursive_depth`, not `max_depth`, unless true
combinator nesting instrumentation exists.

Tree-reduce estimates should use the recursive formula:

```text
N + ceil(N / B) + ceil(ceil(N / B) / B) + ... + 1
```

Example with `N=100`, `B=5`:

```text
100 + 20 + 4 + 1 = 125 calls
```

**Chain dry-runs:** For template chains, each step is instantiated and
dry-run independently. The aggregate response includes per-step estimates
and total pipeline cost. The dry-run also validates output-input
compatibility between adjacent steps — if step N declares an output schema
and step N+1's slot types expect a different structure, the dry-run reports
a compatibility warning.

**Cache hit prediction:** When a dry-run instantiates an artifact whose
hash matches a previously-executed artifact, the dry-run checks the LLM
result cache for matching call signatures. The response includes
`cache_hits_expected` with the predicted count and adjusts cost estimates
downward accordingly.

### Verification

Verification is more useful than per-call template linting. It should focus on
the filled artifact that will actually run.

**Verification checks:**

| # | Check | Pass condition | Failure message |
|---|---|---|---|
| 1 | `artifact_origin` | Artifact record exists with `source_type: "template_invocation"` | `"Artifact was not created by the instantiator."` |
| 2 | `artifact_hash` | `sha256(artifact_code)` matches `artifact.generated_scheme_ref.hash` | `"Artifact code hash mismatch: expected {expected}, got {actual}."` |
| 3 | `template_version` | Template name+version exists in catalog | `"Unknown template: {name} v{version}."` |
| 4 | `slots_filled` | No `{{slot}}` markers remain in artifact code | `"Unfilled slot markers: {markers}."` |
| 5 | `model_exists` | All model aliases in artifact resolve in registry | `"Unknown model alias: {alias}."` |
| 6 | `model_capabilities` | JSON-mode calls target models with `json_mode: true` | `"Model {alias} does not support JSON mode."` |
| 7 | `image_model` | Image inputs target models with `image: true` | `"Model {alias} does not support image inputs."` |
| 8 | `no_unsafe_forms` | No `eval`, `system`, `shell`, `exec` (non-`py-exec`) in artifact | `"Unsafe form found: {form}."` |
| 9 | `no_raw_import` | No `require`, `load`, `include` outside allowed set | `"Disallowed import: {form}."` |
| 10 | `call_count_limit` | Expected calls <= `policy.max_llm_calls` (default: 1000) | `"Expected {n} calls exceeds limit {limit}."` |
| 11 | `recursive_depth_limit` | Recursive depth <= `policy.max_recursive_depth` (default: 3) | `"Recursive depth {d} exceeds limit {limit}."` |
| 12 | `concurrency_limit` | Max concurrency <= `policy.max_concurrency` (default: 50) | `"Concurrency {c} exceeds limit {limit}."` |
| 13 | `context_exists` | All referenced `context_id` values exist in store | `"Context not found: {id}."` |
| 14 | `output_schema_valid` | If `output-schema` declared, it is structurally valid alist notation | `"Output schema is malformed: {detail}."` |
| 15 | `output_schema_present` | If policy requires output schema, template declares one | `"Output schema required by policy but not declared."` |
| 16 | `dry_run_warnings` | No `error`-level warnings from dry-run | `"Dry-run produced error-level warning: {warning}."` |
| 17 | `code_interpreter_policy` | If `uses-llm-generated-code: #t`, policy has `allow_llm_generated_code: true` | `"Template uses LLM-generated code but policy disallows it."` |
| 18 | `gate_consistency` | Gate names in body match `define-meta gates` declarations | `"Gate '{name}' used in body but not declared in metadata."` |
| 19 | `budget_policy_model` | If `budget-policy` declares a fallback model, it exists in registry | `"Budget fallback model '{alias}' not found in registry."` |
| 20 | `budget_policy_caps` | Fallback model has compatible capabilities (JSON mode, images) | `"Fallback model '{alias}' lacks capability: {cap}."` |
| 21 | `primitive_allowlist` | Only primitives from section 9 are used in artifact | `"Disallowed primitive: {name}."` |
| 22 | `context_window_fit` | Estimated input tokens fit model's context window | `"Estimated {tokens} tokens exceeds {alias} context window of {limit}."` |
| 23 | `temperature_compat` | Temperature and max-token settings are valid for model | `"Invalid temperature {t} for model {alias}."` |

Overall verification decision: `pass` if all checks pass, `warn` if any
produce warnings (non-blocking), `fail` if any check fails (execution
blocked).

Verification can optionally run a cheap semantic model review for high-cost or
high-risk artifacts, but deterministic checks should be the default gate.

---

## 13. Planning And Classification

The planner should classify work before choosing a template.

The planner should use these TaskShape categories:

- Direct,
- Batch,
- Synthesize,
- Search,
- Refine,
- Compare,
- Classify,
- Pipeline,
- Generate,
- Decompose,
- Validate,
- Aggregate,
- Composite.

The planner should accept structured hints:

- `item_count`,
- `independent`,
- `output_type`,
- `operation`,
- `has_second_phase`,
- `sub_operations`,
- `modality`,
- `quality_priority`,
- `latency_priority`,
- `budget_limit`.

Composite tasks must preserve constituent shapes. For example, "extract from
all documents, then synthesize a report" is not just `Composite`; it is:

```text
Batch extract -> Synthesize reduce
```

Planning output should be one of:

- a template invocation with slot values (primary path for single-phase tasks),
- a template chain with sequenced steps (primary path for Composite tasks),
- a short list of alternative template invocations with estimated tradeoffs,
- a `no_template` recommendation describing the needed template for the user
  to create.

For Composite tasks, the planner decomposes the task into constituent atomic
templates and produces a `template_chain` with `$previous` references
connecting steps. This enables combinatorial composition — the planner
assembles pipelines from atomic templates rather than requiring a monolithic
template for every combination.

Planning output must not include raw Scheme. If no template matches the
classified task, the planner returns a structured recommendation for a new
template rather than attempting to generate ad-hoc Scheme code.

---

## 14. Taxonomy Decision Rules

Classification and template selection use a **two-level decision model**:

**Level 1 — Deterministic (code, no LLM):** TaskShape and DataShape
classification from structured hints, plus template selection from
trigger/reject conditions. If all required hints are provided, this level
runs entirely as deterministic code. The decision tree questions in sections
14.1-14.3 below are all Level 1 — they operate on structured fields
(`item_count`, `independent`, `output_type`, `has_second_phase`, etc.),
not on free-text interpretation.

**Level 2 — LLM gap-filling (only when hints are missing):** When the agent
does not provide enough structured hints to answer the decision tree, the
planner makes a single LLM call to fill the missing fields. The LLM answers
structured yes/no or multiple-choice questions. Once fields are filled,
Level 1 runs deterministically on the complete fields.

The LLM never chooses templates directly. It only fills missing structured
fields that the deterministic classifier then uses.

**Trigger condition:** Level 2 fires when any field required by the first
unanswered decision tree question (Q0-Q9 in section 14.1) is missing from
the agent's hints. If the agent provides all fields needed to traverse the
tree to a leaf, Level 2 is skipped entirely.

**Prompt template:**

```text
Given this task description and available metadata, answer each question
with ONLY the specified answer format. Do not explain.

Task: {task_description}
Context metadata: {context_metadata_json}

Questions:
{for each missing field:}
- {field_name}: {question_text} ({answer_format})
{end for}

Respond as JSON:
{
  {for each missing field:}
  "{field_name}": <answer>
  {end for}
}
```

**Question bank** (one per hint field):

| Field | Question | Answer format |
|---|---|---|
| `item_count` | How many input items are there? | integer |
| `independent` | Are the items independent of each other? | `true` or `false` |
| `output_type` | What is the output shape? | `"one"`, `"list"`, or `"per_item"` |
| `operation` | What is the per-item operation? | `"transform"`, `"extract"`, `"label"`, `"check"`, `"grade"`, `"other"` |
| `has_second_phase` | Does the task have a second phase after processing items? | `true` or `false` |
| `sub_operations` | What operations are needed? | array of strings |
| `modality` | What data modalities are present? | array: `"text"`, `"image"`, `"audio"` |
| `ordered` | Does item order matter? | `true` or `false` |

**Response schema:**

```json
{
  "item_count": 100,
  "independent": true,
  "output_type": "one",
  "operation": "extract",
  "has_second_phase": true,
  "sub_operations": ["extract", "synthesize"],
  "modality": ["text"],
  "ordered": false
}
```

The planner validates the LLM response against expected types (integer,
boolean, enum, array) and falls back to conservative defaults if validation
fails: `independent: false`, `output_type: "one"`, `has_second_phase: false`.
The planner model is `quality_text_model` with `json_mode: true` and
`temperature: 0`.

**Level 2 qualitative modifiers** — After template selection, some slot values
require LLM judgment:
- Cost sensitivity → model tier selection
- Quality requirements → validation wrappers, iteration counts
- Error tolerance → error policy selection
- Instruction text → the actual prompts for LLM calls

These are template slot values, not structural decisions. The template's
`slot_schema` constrains them with types, enums, and ranges.

### 14.1 TaskShape

| Shape | Description | Structural family |
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
| `Composite` | Multi-phase task. | instantiated `sequence` of phase templates. |

TaskShape decision tree:

```text
Q0: Is this one small input, one output, one operation, no second phase?
    YES -> Direct
    NO  -> Q1

Q1: Are there many input items?
    YES -> Q2
    NO  -> Q5

Q2: Are items independent?
    YES -> Q3
    NO  -> Q4

Q3: What is the per-item operation?
    Transform/extract -> Batch
    Label/category    -> Classify
    Check/grade/audit -> Validate

Q4: Does information accumulate across ordered items?
    YES -> Synthesize with fold-sequential
    NO  -> Pipeline

Q5: Is the task creating content with no source item list?
    YES -> Generate
    NO  -> Q6

Q6: Is the task improving one artifact?
    YES -> Refine
    NO  -> Q7

Q7: Is the task breaking one input into parts?
    YES -> Decompose
    NO  -> Q8

Q8: Is the task choosing among alternatives?
    YES -> Compare or Search
    NO  -> Synthesize, Aggregate, or Direct depending on output type

Q9: Does the task clearly have multiple phases?
    YES -> Composite, preserving constituent shapes
```

### 14.2 DataShape

| Shape | Description | Important fields |
|---|---|---|
| `FlatList` | Independent or ordered list. | count, item_size, independent. |
| `Hierarchy` | Tree or nested structure. | depth, branching, node_count. |
| `Singular` | One blob that may fit in context. | size, chunkable, boundary. |
| `ChunkedSingular` | Large document split into dependent chunks. | chunk_count, overlap, dependency. |
| `Graph` | Connected entities and edges. | nodes, edges, connectedness. |
| `TimeSeries` | Ordered observations. | length, window_size, causal. |
| `Tabular` | Rows with shared schema. | row_count, columns, grouping keys. |
| `Multimodal` | Text plus images/audio. | modality, count, model requirements. |
| `Paired` | Aligned source/target pairs. | pair_count, alignment key. |
| `KeyValue` | Dictionary/map data. | key_count, preserve_keys. |

DataShape mapping rules:

```text
FlatList { independent: true, count <= 50 }
  -> map-async with max-concurrent = count

FlatList { independent: true, count > 50 }
  -> map-async with max-concurrent = min(count, 20)

FlatList { independent: false }
  -> fold-sequential

Singular { size <= context_limit, one operation }
  -> Direct

Singular { size > context_limit, chunkable: true, chunks independent }
  -> chunk, then FlatList

ChunkedSingular { chunks dependent }
  -> fold-sequential with explicit summary/checkpoint strategy

Hierarchy { depth > 2 }
  -> tree-reduce over matching hierarchy or recursive-spawn

Tabular { row_count > 50, independent_rows: true }
  -> map-async row extraction + python_compute aggregation

Multimodal
  -> require model with image/audio support; include image token estimates

Paired
  -> zip pairs and map-async over pair records

KeyValue
  -> preserve keys in results; aggregate by key
```

### 14.3 Per-Shape Template Selection

Direct:

```text
Q1: Does the input fit in one model context?
    YES -> direct_call
    NO  -> reclassify as Decompose, Batch, or Synthesize

Q2: Is deterministic computation needed before/after the call?
    YES -> python_compute + direct_call, or direct_call + python_compute
    NO  -> direct_call only
```

Batch:

```text
Q1: Return a list or one combined output?
    LIST     -> batch_map
    COMBINED -> batch_extract_reduce

Q2: If combined, is combination order-sensitive?
    YES -> batch_extract_fold
    NO  -> batch_extract_reduce

Q3: Are some items harder or more ambiguous?
    YES -> tiered_review template
    NO  -> one map-async pass

Q4: Are duplicates likely?
    YES -> memoized map phase
```

Synthesize:

```text
Q1: Do all items fit in one context?
    YES -> direct_synthesis
    NO  -> Q2

Q2: Is order important?
    YES -> ordered_synthesis_fold
    NO  -> tree_synthesis

Q3: Is accumulator likely to exceed context?
    YES -> fold_with_summarization
    NO  -> exact fold-sequential
```

Search:

```text
Q1: Is the candidate set finite?
    YES -> compare_candidates
    NO  -> iterative_search

Q2: Is latency more important than quality?
    YES -> race_candidates
    NO  -> evaluate_all_then_select
```

Refine:

```text
Q1: Is there a testable predicate?
    YES -> refine_until_valid
    NO  -> bounded_critique_refine

Q2: Should each iteration be validated?
    YES -> wrap refine step with validation
```

Compare:

```text
Q1: Compare models or strategies?
    MODELS     -> compare_models
    STRATEGIES -> compare_strategies

Q2: Select one or synthesize all?
    SELECT     -> parallel + python_compute/Scheme selection
    SYNTHESIZE -> parallel + llm-query aggregator
```

Classify:

```text
Q1: One item or many?
    ONE  -> direct_classify
    MANY -> batch_classify

Q2: Need distribution/report?
    YES -> python_compute aggregation after labels
    NO  -> return labels

Q3: Ambiguous categories?
    YES -> tiered_review template
```

Pipeline:

```text
Q1: Are stages distinct?
    YES -> sequence
    NO  -> reclassify as Batch

Q2: Can a stage fail?
    YES -> fallback around that stage

Q3: Does a stage need quality gating?
    YES -> validation around that stage
```

Generate:

```text
Q1: Fixed number or until condition?
    FIXED -> map-async over generated index list
    UNTIL -> iterate-until

Q2: Must items be mutually consistent?
    YES -> fold-sequential
    NO  -> map-async

Q3: Must items be unique?
    YES -> python_compute deduplication and regenerate missing count
```

Decompose:

```text
Q1: Known structural boundary?
    YES -> python_compute splitter
    NO  -> llm-query with JSON output

Q2: Is one pass enough?
    YES -> parse parts and return
    NO  -> recursive artifact-aware decomposition

Q3: Process parts afterward?
    YES -> Composite: Decompose -> Batch
```

Validate:

```text
Q1: Same rubric for all items?
    YES -> map-async validation
    NO  -> fold-sequential if criteria evolve

Q2: Need structured assessment?
    YES -> JSON mode plus schema validation

Q3: Which error is costlier?
    FALSE POSITIVE -> expensive review of passes
    FALSE NEGATIVE -> expensive review of failures
```

Aggregate:

```text
Q1: Pure computation after extraction?
    YES -> map-async extraction + python_compute aggregation
    NO  -> map-async extraction + python_compute stats + llm interpretation

Q2: Grouped report?
    YES -> python_compute groupby using extracted schema
```

Composite:

```text
Q1: Identify constituent shapes in order.
Q2: Select an atomic template for each phase.
Q3: Produce a template chain connecting dependent phases with `$previous`.
Q4: If phases are independent, the agent can execute them as separate plans in parallel.
```

---

## 15. Initial Template Catalog

The first implementation should include enough templates to cover common
workflows without asking the planner to invent structure.

| Template | Shapes | Primitive composition |
|---|---|---|
| `direct_call` | Direct | `llm-query`. |
| `direct_json_extract` | Direct, Decompose | `llm-query #:json #t` plus validation. |
| `batch_map` | Batch, Classify, Validate | `map-async`. |
| `batch_extract_reduce` | Batch + Synthesize | `map-async` plus `tree-reduce`. |
| `batch_extract_fold` | Batch + ordered Synthesize | `map-async` plus `fold-sequential`, or direct `fold-sequential` when items are dependent. |
| `ordered_synthesis_fold` | Synthesize | `fold-sequential` with optional checkpointing. |
| `tree_synthesis` | Synthesize | `tree-reduce`. |
| `compare_candidates` | Compare, Search | `parallel` plus selection. |
| `race_candidates` | Search | `race`. |
| `refine_until_valid` | Refine | `iterate-until` plus `with-validation`. |
| `bounded_critique_refine` | Refine | `iterate-until` with critique/refine state. |
| `tiered_review` | Batch, Classify, Validate | cheap `map-async`, uncertainty filter, expensive `map-async`. |
| `tabular_extract_aggregate` | Aggregate, Tabular | `map-async` plus `python_compute`. |
| `decompose_then_batch` | Decompose, Composite | JSON decomposition plus `map-async`. |
| `recursive_decompose` | Decompose, Hierarchy | artifact-aware `recursive-spawn`. |
| `code_interpreter` | Direct, Aggregate | `llm-query` + `py-exec` + `with-validation` + `iterate-until`. Requires `uses-llm-generated-code: #t`. |

Every template should include:

- slot schema,
- expected call formula,
- structural profile,
- model capability requirements,
- verification rules,
- at least one example invocation,
- one instantiation fixture showing the artifact after slot substitution.

**Chaining and monolithic templates:** Several monolithic templates
(`batch_extract_reduce`, `batch_extract_fold`) can also be expressed as
chains of atomic templates (`batch_map` → `tree_synthesis`, `batch_map` →
`ordered_fold`). Monolithic versions are kept for convenience and
performance (one artifact, one instantiation), but chains are the preferred
composition mechanism for new Composite workflows.

Templates should also declare in their `define-meta`:

- `streamable: #t` for templates that produce meaningful intermediate results
  (e.g., `batch_map`, `batch_extract_reduce`, `tabular_extract_aggregate`),
- `cacheable: #t` for templates whose LLM calls are safe to cache across
  executions (most templates at temperature 0).

---

## 16. Implementation File Layout

Suggested greenfield layout:

```text
rlm_scheme/
  __init__.py
  mcp_server.py
  models.py
  ids.py
  store.py
  context_store.py
  template_store.py
  planner.py
  classifier.py
  instantiator.py      # internal library, used by dry_run.py and executor.py
  cache.py            # content-addressed LLM result cache
  chain.py            # template chain execution logic
  gate.py             # gate primitive and resume_execution handler
  dry_run.py           # instantiates + simulates + estimates in one call
  executor.py          # instantiates + verifies + executes in one call
  trace.py
  llm_provider.py
  image_inputs.py
  python_bridge.py
  runtime/
    racket_server.rkt
    primitives.rkt
    sandbox.rkt
    callbacks.rkt
templates/
  direct_call.rkt
  direct_json_extract.rkt
  batch_map.rkt
  batch_extract_reduce.rkt
  batch_extract_fold.rkt
  ordered_synthesis_fold.rkt
  tree_synthesis.rkt
  compare_candidates.rkt
  race_candidates.rkt
  refine_until_valid.rkt
  bounded_critique_refine.rkt
  tiered_review.rkt
  tabular_extract_aggregate.rkt
  decompose_then_batch.rkt
  recursive_decompose.rkt
  code_interpreter.rkt
docs/
  GREENFIELD-REWRITE-PLAN.md
  api-reference.md
  templates.md
  primitives.md
tests/
  test_id_flow.py
  test_mcp_api_schemas.py
  test_template_validation.py
  test_instantiator.py
  test_runtime_primitives.py
  test_dry_run.py      # also covers instantiation and estimation
  test_executor.py     # also covers verification
  test_cache.py
  test_chain.py
  test_gate.py
  test_streaming.py
```

Module responsibilities:

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic/dataclass schemas for all durable records and API payloads. |
| `ids.py` | ID generation and validation for `ctx_`, `plan_`, `art_`, `dry_`, `ver_`, `exec_`, `call_`. |
| `store.py` | Durable JSON or SQLite/PGlite storage abstraction. |
| `context_store.py` | Large context storage, previews, metadata, and path extraction. |
| `template_store.py` | Load, validate, list, and retrieve `.rkt` templates (parse `define-meta` forms + body). |
| `classifier.py` | Deterministic TaskShape/DataShape rules. |
| `planner.py` | Template selection and plan record creation. |
| `instantiator.py` | Internal library: slot validation and safe substitution into template Scheme code. Called by `dry_run.py` and `executor.py`. |
| `dry_run.py` | Instantiates template invocation, simulates execution, computes cost estimates. |
| `executor.py` | Instantiates (or reuses artifact from dry run), verifies against policy, executes in Racket sandbox. |
| `cache.py` | Content-addressed LLM result cache. Key computation, storage, lookup, and `reset_runtime(scope="cache")`. |
| `chain.py` | Template chain execution: step sequencing, intermediate context creation, `$previous` resolution, and chain-level checkpointing. |
| `gate.py` | Gate primitive implementation and `resume_execution` MCP tool handler. Manages gate state in execution records. |
| `trace.py` | Trace event schema and aggregation. |
| `llm_provider.py` | Provider calls, retry, rate limits, token accounting. |
| `image_inputs.py` | Image resolution, MIME sniffing, size limits. |
| `python_bridge.py` | Controlled Python compute subprocess. |

**Racket module responsibilities:**

| Module | Responsibility |
|---|---|
| `racket_server.rkt` | Main entry point: accepts JSON commands from Python host over stdin/stdout, dispatches to sandbox, returns results. |
| `primitives.rkt` | All public primitive bindings (section 9): `llm-query`, `map-async`, `tree-reduce`, `gate`, `finish`, etc. Defines the sandbox namespace. |
| `sandbox.rkt` | Racket sandbox configuration: resource limits, allowed modules, scaffold binding protection, syntax hygiene enforcement. |
| `callbacks.rkt` | Host callback protocol: JSON-RPC-style messages to Python for LLM calls, Python bridge invocations, checkpoint operations, and progress notifications. |

---

## 17. Implementation Phases

### Phase 0: Decisions And Schemas

*Depends on: nothing.*

- Freeze public MCP API names.
- Define ID record schemas.
- Define template `define-meta` schema and slot substitution rules (section 6).
- Define template file format (`.rkt` with `define-meta` forms).
- Decide initial store backend.
- Decide which Python bridge operations are allowed in templates.
- Decide cache key format and eviction strategy.
- Decide streaming notification schema and batching policy.
- Decide gate approval protocol and timeout behavior.

Exit criteria:

- All Pydantic/dataclass schemas in `models.py` compile without errors.
- Example JSON records for all 7 ID types (`ctx_`, `plan_`, `art_`, `dry_`, `ver_`, `exec_`, `call_`) pass schema validation.
- No public MCP tool accepts a `code` or `scheme` string parameter.

### Phase 1: Durable Store And MCP Skeleton

*Depends on: Phase 0.*

- Implement context, plan, artifact, dry-run, verification, execution stores.
- Implement ID generation and parent-child linking.
- Add MCP tools with stubbed behavior.
- Add `get_status`, `cancel_call`, and `reset_runtime` skeletons.

Exit criteria:

- `test_id_flow.py` passes: create ctx → plan → art → dry → ver → exec, verify parent chain via `parent_id` lookups.
- Each MCP tool returns a valid JSON response (even if stubbed).
- `reset_runtime(scope="all")` clears all records; subsequent `get_status()` returns empty.

### Phase 2: Minimal Racket Runtime

*Depends on: Phase 1.*

- Build sandbox lifecycle.
- Implement internal `llm-query`, syntax wrapping, `syntax-e`, `datum->syntax`,
  scope logging, `finish`, and scaffold protection.
- Implement `load-context` runtime command.
- Implement stdout/stderr capture and structured errors.

Exit criteria:

- A test artifact containing `(finish (syntax-e (llm-query ...)))` executes and returns a result string.
- Scope log contains at least one `syntax-e` entry with `preview` and `scope` fields.
- `(set! llm-query 42)` raises an error (scaffold protection).

### Phase 3: Host Callback Loop

*Depends on: Phase 2.*

- Implement real model calls.
- Implement async futures.
- Implement `await`, `await-all`, `await-any`, and cancellation.
- Implement retry and rate-limit tracking.
- Implement progress reporting and heartbeat.

Exit criteria:

- `test_runtime_primitives.py`: 3 concurrent `llm-query-async` calls complete via `await-all`, results list has length 3.
- `cancel_call(call_id=...)` on an in-flight call transitions it to cancelled in the trace.
- `get_status()` shows `tokens_used > 0` and `rate_limits` hash is non-empty after a successful call.

### Phase 4: Primitive Runtime Basis

*Depends on: Phase 3.*

- Add `map-async`, `parallel`, `race`, `tree-reduce`, `fold-sequential`,
  `sequence`, `choose`, `iterate-until`, `recursive-spawn`, `memoized`,
  `with-validation`, and `try-fallback`.
- Add checkpoint/restore and token-budget behavior.

Exit criteria:

- `test_runtime_primitives.py`: `map-async` with 10 items and `max-concurrent: 3` never has more than 3 in-flight calls simultaneously (verified via trace timestamps).
- `tree-reduce` with `N=8, B=2` produces exactly `8 + 4 + 2 + 1 = 15` trace call events.
- `iterate-until` with `max-iter: 5` terminates after at most 5 iterations.
- Racket sandbox exports exactly the primitives listed in section 9 — no `eval`, `system`, `require`.

### Phase 5: Template Catalog And Instantiation Library

*Depends on: Phase 4.*

- Create initial `.rkt` templates for common shapes (see section 15).
- Implement template `define-meta` parsing and validation.
- Implement slot validation and safe substitution as an internal library
  (`instantiator.py`) — no dedicated MCP tool.
- Store instantiated artifacts with hashes.
- Add `output-schema`, `streamable`, `cacheable`, `gates`, `budget-policy`,
  and `uses-llm-generated-code` to the template `define-meta` schema.

Exit criteria:

- `test_instantiator.py`: instantiating `batch_map` with valid slot values produces an artifact; instantiating again with same values produces the same `code_hash`.
- Artifact code contains zero `{{` markers (grep test).
- `template_store.list()` returns >= 16 templates (section 15 catalog).
- `define-meta output-schema`, `streamable`, `cacheable`, `gates`, `budget-policy`, and `uses-llm-generated-code` are parsed and available as metadata fields.

### Phase 6: Planner

*Depends on: Phase 5.*

- Implement deterministic TaskShape/DataShape classification.
- Add structured hints to `plan_strategy`.
- Use template metadata for selection.
- Return alternatives when tradeoffs are meaningful.

Exit criteria:

- `plan_strategy` with full hints (`item_count=100, independent=true, output_type=one, has_second_phase=true`) returns `task_shape: Composite` and a `template_invocation` or `template_chain` — no LLM call made (Level 1 only).
- `plan_strategy` with missing hints triggers exactly 1 LLM call (Level 2 gap-fill).
- `plan_strategy` for an unsupported task returns `status: "no_template"` with `recommendation.needed_template`.

### Phase 7: Dry-Run (with estimation) and Verification

*Depends on: Phase 6.*

- Implement `dry_run_strategy` tool: instantiates internally, computes static
  estimates from artifact profiles, and simulates execution — all in one call.
- Special-case `await-any` and batch await semantics in simulation.
- Implement verification logic as an internal step within `execute_strategy`
  (no separate `verify_strategy` tool).
- Add output schema validation in verification.
- Add chain compatibility validation (output-input schema matching).
- Add cache hit prediction in dry-run response.
- Add `uses-llm-generated-code` policy check in verification.

Exit criteria:

- `dry_run_strategy` for `batch_extract_reduce` with `N=100, B=5` returns `expected_llm_calls: 125` and `critical_path_calls: 4`.
- Dry-run uses no global mutable state (two concurrent dry-runs produce independent results).
- `execute_strategy` with `uses-llm-generated-code: #t` and `allow_llm_generated_code: false` returns verification `decision: "fail"` with check `code_interpreter_policy`.
- Verification of an artifact with `{{unfilled}}` marker returns `decision: "fail"` with check `slots_filled`.

### Phase 8: Execute And Trace

*Depends on: Phase 7.*

- Implement `execute_strategy`.
- Link executions to verification and artifact records.
- Assemble full traces with scope logs, call metrics, stdout, errors, and
  checkpoints.
- Support repeated executions of the same artifact.
- Implement streaming partial results via MCP notifications.
- Implement `gate` primitive and `resume_execution` tool.
- Implement budget threshold monitoring and policy activation.

Exit criteria:

- `execute_strategy` → `get_execution_trace` returns trace with >= 1 `llm_call_completed` event.
- Executing the same `artifact_id` twice creates two distinct `execution_id` values.
- `cancel_call(execution_id=...)` during execution transitions state to `cancelled`.
- `gate` fires → state is `awaiting_gate` → `resume_execution(approve)` → state is `finished`.
- `gate` fires → `resume_execution(reject)` → state is `gate_rejected`, partial work preserved in trace.

### Phase 9: Advanced Features

*Depends on: Phase 8.*

- Add multimodal template support.
- Add controlled Python compute phases.
- Add recursive artifact-aware delegation.
- Add checkpoint recovery workflows.
- Add history-based planner feedback.
- Add cross-execution memoization cache (section 11.7).
- Add plan-level template chaining (section 11.8).
- Add code interpreter template with `uses-llm-generated-code` gating (section 11.9).
- Add budget-aware model switching and checkpoint-and-stop degradation.

Exit criteria:

- `test_cache.py`: execute with `temp=0` → re-execute same inputs → second run has `cache_hits > 0` and makes zero provider calls.
- `test_chain.py`: 2-step chain (`batch_map` → `tree_synthesis`) completes; `chain_step_results` has 2 entries; `ctx_auto_0` is retrievable from trace.
- `test_chain.py`: chain failure at step 1 → retry resumes from step 1 (step 0 not re-executed, verified by call count).
- `test_gate.py`: code interpreter template with `allow_llm_generated_code: false` → verification fails.
- Budget test: set budget to 50% of expected → execution activates budget policy, switches model, completes with partial quality.

### Phase 10: Documentation And Migration

*Depends on: Phase 9.*

- Rewrite README around artifact workflow.
- Replace old raw-code API docs.
- Replace combinator docs with primitive runtime docs and template docs.
- Add examples for each ID stage.
- Keep the old implementation referenced only as historical context.

Exit criteria:

- README contains zero mentions of `execute_scheme` or `dry_run_scheme`.
- README shows the 3-tool happy path: `plan_strategy` → `dry_run_strategy` → `execute_strategy`.
- `docs/primitives.md` lists all primitives from section 9 with signatures.
- `docs/templates.md` documents at least 5 templates with example invocations.

---

## 18. Test Plan

Minimum test coverage:

- schema validation for every ID record,
- parent-child ID flow,
- context metadata classification,
- template validation,
- instantiation determinism,
- generated Scheme hash verification,
- no public `execute_scheme` or `dry_run_scheme` MCP tools,
- only primitive bindings are exported by the runtime,
- syntax hygiene and scope logging,
- async handle validation,
- `await-any` dry-run behavior,
- `map-async` bounded concurrency,
- `parallel` real concurrency,
- cancellation of active and queued calls,
- retry and rate-limit accounting,
- token-budget exhaustion,
- checkpoint/restore,
- JSON-mode validation,
- image validation,
- Python bridge value transfer,
- recursive depth enforcement,
- verification pass/warn/fail behavior (tested through `execute_strategy`),
- instantiation tested through `dry_run_strategy` and `execute_strategy` paths,
- estimation tested through `dry_run_strategy` response,
- execution trace persistence,
- streaming partial results delivery and ordering,
- cross-execution cache hit/miss behavior,
- cache key correctness (same inputs → same key, different inputs → different key),
- temperature > 0 cache bypass (not cached unless `cacheable: #t`),
- template chain execution: 2-step and 3-step chains,
- chain failure at step N: completed steps preserved, retry resumes,
- chain dry-run: per-step estimates and aggregate totals,
- chain output-input schema compatibility validation,
- output schema validation on `finish`,
- output schema mismatch produces structured error,
- gate suspension and `resume_execution` approve/reject,
- gate with `required: #f` and policy override,
- budget threshold detection and model switching,
- budget exhaustion checkpoint-and-stop with partial results,
- LLM-generated Python execution with import allowlist,
- LLM-generated Python policy gating (blocked when policy disallows).

---

## 19. End-to-End Walkthrough

This section shows every MCP call in sequence for a realistic task: "Extract
ACE2 protein mentions from 100 research papers and synthesize a report."

The happy path requires only 3 tool calls after loading context:
`plan_strategy` → `dry_run_strategy` → `execute_strategy`.

### Step 1: Load context

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

### Step 2: Plan strategy

Agent describes the task. Hints provide structured fields so classification
is fully deterministic (Level 1 only, no LLM call needed).

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
        "items_path": "$.items",
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

### Step 3: Dry run

Agent runs a dry run. Internally, this instantiates the template (validates slots,
substitutes markers, hashes, stores artifact), computes cost estimates, and
simulates execution — all in one call. No real LLM calls are made.

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

### Step 4: Execute

Agent runs the strategy. Internally, this instantiates the template invocation
(cache-hits the artifact from the dry run via hash match), runs verification
checks automatically, and executes the instantiated Scheme. Real LLM calls happen
here.

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
        {"name":"primitive_allowlist","status":"pass","message":"Only primitives used."},
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

### Step 5: Inspect trace (optional)

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
        ...
        {"type":"llm_call_started","call_id":"call_101","node_id":"synthesize","model":"quality_text_model","depth":0},
        ...
      ],
      "scope_log": [
        {"op":"syntax-e","preview":"extracted ACE2 mentions...","scope":"sandbox","call_id":"call_001"}
      ],
      "stdout": ""
    }
  }
```

### Summary of ID chain

```text
ctx_7f3a (data) → plan_b2c1 (classification + template) → dry_1a2b (instantiate + simulate + estimate) → exec_5e6f (verify + execute)
```

Internal IDs created along the way: `art_e4d9` (artifact, created by dry-run),
`ver_3c4d` (verification, created by execute). These appear in responses and
durable records for audit/replay but are not passed between tools by the agent.

Each ID is durable and inspectable. The same plan can be re-executed with
different data by creating a new context and updating `context_id` in the
template invocation's slot values.

### Walkthrough 2: Chained Workflow With Streaming And Cache

This walkthrough shows a Composite task using template chaining, streaming,
and cache reuse.

**First run:** Extract then synthesize, with streaming.

```text
→ load_context(data: "[100 papers]", name: "ace2_papers", ...)
← { "context_id": "ctx_7f3a" }

→ plan_strategy(task: "Extract ACE2 mentions and synthesize a report.", context_id: "ctx_7f3a", ...)
← {
    "plan_id": "plan_c1d2",
    "recommended": {
      "kind": "template_chain",
      "steps": [
        {"template_name": "batch_map", "slot_values": {"context_id": "ctx_7f3a", "map_instruction": "Extract ACE2 mentions as JSON.", ...}},
        {"template_name": "tree_synthesis", "slot_values": {"input": "$previous", "reduce_instruction": "Synthesize findings.", ...}}
      ]
    }
  }

→ dry_run_strategy(plan_id: "plan_c1d2")
← {
    "dry_run_id": "dry_e3f4",
    "steps": [
      {"template_name": "batch_map", "estimated_llm_calls": 100, "estimated_cost_usd": {"low": 0.80, "high": 2.00}},
      {"template_name": "tree_synthesis", "estimated_llm_calls": 25, "estimated_cost_usd": {"low": 0.40, "high": 1.50}}
    ],
    "total_estimated_llm_calls": 125,
    "total_estimated_cost_usd": {"low": 1.20, "high": 3.50},
    "cache_hits_expected": 0
  }

→ execute_strategy(plan_id: "plan_c1d2", stream: true, timeout_seconds: 900)

  ... notifications arrive as items complete:
  { "type": "notifications/partial_result", "node_id": "extract", "item_index": 0, "value": "{...}" }
  { "type": "notifications/partial_result", "node_id": "extract", "item_index": 1, "value": "{...}" }
  ...

← {
    "execution_id": "exec_a1b2",
    "result": {"value": "ACE2 findings report..."},
    "execution": {
      "state": "finished",
      "llm_calls": 125,
      "cache_hits": 0,
      "budget_policy_activations": 0,
      "chain_step_results": [
        {"step": 0, "template": "batch_map", "intermediate_context_id": "ctx_auto_0"},
        {"step": 1, "template": "tree_synthesis", "result": "ACE2 findings report..."}
      ]
    }
  }
```

**Second run:** Same extraction, different synthesis instruction. Cache
eliminates all 100 map-phase calls.

```text
→ plan_strategy(task: "Extract ACE2 mentions and write a methods section.", context_id: "ctx_7f3a", ...)
← { "plan_id": "plan_d4e5", "recommended": {"kind": "template_chain", "steps": [
      {"template_name": "batch_map", "slot_values": {"context_id": "ctx_7f3a", "map_instruction": "Extract ACE2 mentions as JSON.", ...}},
      {"template_name": "tree_synthesis", "slot_values": {"input": "$previous", "reduce_instruction": "Write a methods section.", ...}}
    ]} }

→ dry_run_strategy(plan_id: "plan_d4e5")
← {
    "total_estimated_llm_calls": 125,
    "cache_hits_expected": 100,
    "total_estimated_cost_usd": {"low": 0.40, "high": 1.50}
  }

→ execute_strategy(plan_id: "plan_d4e5", timeout_seconds: 900)
← {
    "execution_id": "exec_c3d4",
    "execution": { "llm_calls": 125, "cache_hits": 100 },
    "result": {"value": "Methods section focusing on ACE2..."}
  }
```

The second run costs ~60% less and completes faster because the entire
map phase hits the cache.

---

## 20. Open Design Decisions

These should be decided before implementation begins:

1. **Store backend.** Start with filesystem JSON (decided in section 11.1).
   Migrate to SQLite/PGlite if queryable history or concurrent access becomes
   a bottleneck. *(Partially decided — revisit if needed.)*
2. **Recursive planning.** Decide whether recursive sub-plans are instantiated
   ahead of time or generated at runtime under verification constraints.
3. **History feedback.** Decide which execution metrics influence future
   planning and how to avoid leaking sensitive data into planner prompts.
4. **Gate timeout.** *(Decided: indefinite wait, optional timeout kwarg.)*
   Gates wait indefinitely by default. Templates can pass `#:timeout seconds`
   to the `gate` primitive; if the timeout elapses without a
   `resume_execution` call, the gate auto-rejects with reason
   `"gate_timeout"`. This keeps the default simple while allowing
   time-sensitive workflows to self-terminate.
5. **Cache eviction.** *(Decided: manual-only in V1.)* Cache entries persist
   until `reset_runtime(scope="cache")`. No automatic LRU, TTL, or
   size-based eviction. See section 11.7.
6. **Chain fan-in.** Decide whether future versions should support fan-in
   chains (multiple previous steps feeding one next step) or whether that
   remains the agent's responsibility.

---

## 21. Success Criteria

The rewrite is successful when:

- agents never need to write Scheme,
- agents can still inspect generated Scheme for debugging (via dry-run and
  execute responses),
- all execution goes through instantiated artifacts (created internally),
- the happy-path agent flow is 3 tool calls: plan → dry-run → execute,
- the public MCP API surface is 10 tools,
- instantiation, estimation, and verification are internal — no separate
  agent-facing tools for these steps,
- dry-run and verification happen before expensive calls,
- templates cover common orchestration shapes,
- the runtime exposes only the 10 primitive combinators plus modifiers and
  the `gate` control primitive,
- no unsafe public escape hatches exist,
- large contexts are represented by IDs and metadata,
- recursive workflows remain possible,
- current operational features are preserved: progress, cancel, trace, rate
  limits, token accounting, checkpointing, multimodal input, and controlled
  Python compute,
- streaming delivers partial results during `map-async` and `tree-reduce`,
- cross-execution cache eliminates redundant LLM calls for identical inputs,
- the planner produces template chains for Composite tasks,
- output schemas are declared in templates and validated on `finish`,
- gates suspend execution and `resume_execution` resumes or rejects,
- budget exhaustion produces partial results via checkpoint, not hard failure,
- LLM-generated Python is policy-gated and declared in template metadata.
