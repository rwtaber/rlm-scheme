# RLM-Scheme Greenfield Rewrite Plan

This document is a complete implementation plan for rewriting RLM-Scheme from
scratch. The goal is to rebuild the system around the intended architecture
from the beginning, using the existing codebase only as a feature inventory for
runtime behavior that should be preserved.

No backward compatibility is required.

The rewrite should remove:

- public raw Scheme execution APIs,
- public raw Scheme dry-run APIs,
- planner-generated `code_template` outputs,
- runtime compound combinators,
- public unsafe escape hatches,
- stale docs and model references.

The rewrite should preserve the important capabilities that already exist:

- Racket sandbox execution with resource limits,
- syntax hygiene and provenance logging,
- synchronous and asynchronous LLM callbacks,
- real fan-out concurrency with cancellation,
- progress/status reporting,
- rate-limit and token accounting,
- checkpoint/restore for long workflows,
- multimodal image input support,
- structured JSON-mode support,
- large context loading and named context access,
- controlled Python computation bridge,
- recursive delegation with a hard depth limit,
- execution trace and runtime health inspection.

---

## 1. North Star

The MCP server should expose a structured orchestration system, not a raw code
execution system.

An agent interacts with durable objects:

```text
context_id -> plan_id -> artifact_id -> dry_run_id -> verification_id -> execution_id
```

Each stage has a clear responsibility:

| ID | Meaning |
|---|---|
| `context_id` | Stored input data plus metadata: shape, item count, modality, independence, size estimates, and optional names. |
| `plan_id` | Task intent and planning record: objective, constraints, inferred TaskShape/DataShape, selected template, and rationale. |
| `artifact_id` | Compiled executable strategy: template with filled typed slots (Scheme code after safe substitution), compiler version, and code hash. |
| `dry_run_id` | Structural simulation for an artifact: expected calls, fan-out, recursive depth, model mix, token/cost estimates, warnings, and failure risks. |
| `verification_id` | Verification decision: deterministic checks, dry-run interpretation, optional semantic review, pass/warn/fail status, and reasons. |
| `execution_id` | One real execution attempt: result, stdout, trace, call metrics, token usage, errors, checkpoints, and status history. |

Normal agent flow:

1. `load_context(data, name, metadata)` stores large input and returns a
   `context_id`.
2. `plan_strategy(task, context_id, hints)` classifies the work and returns a
   `plan_id` plus a template invocation.
3. `compile_strategy(plan_id | template_invocation)` validates typed
   slots, substitutes them into the template Scheme, and returns an
   `artifact_id`.
4. `estimate_strategy(artifact_id)` gives a static estimate.
5. `dry_run_strategy(artifact_id)` simulates execution and returns a
   `dry_run_id`.
6. `verify_strategy(artifact_id, dry_run_id)` gates execution and returns a
   `verification_id`.
7. `execute_strategy(artifact_id, verification_id, plan_id, timeout)` executes
   the compiled artifact and returns an `execution_id`.
8. `get_execution_trace(execution_id)`, `get_status`, and `cancel_call` inspect
   or control long-running work.

Scheme is internal compiled code. It may be inspectable through artifact
metadata for debugging, but agents should not submit arbitrary Scheme strings
to the public MCP API.

---

## 2. What Templates Store

Templates are the bridge between high-level planning and executable Scheme.
They should be data, not prompts that ask an LLM to write code.

There are two levels in the compilation pipeline:

```text
Template (Scheme code with {{slot}} markers)
    ↓  compiler validates slots, substitutes values, hashes result
Artifact (executable Scheme run by the Racket sandbox)
```

A **template** is a `.rkt` file containing real Scheme code that uses
primitive runtime bindings directly. Content-specific values are represented
as `{{slot_name}}` markers — typed holes that the compiler fills with
concrete values. For example, a `batch_extract_reduce` template contains
`map-async` and `tree-reduce` calls but uses `{{map_instruction}}`,
`{{map_model}}`, and `{{max_concurrent}}` markers where content-specific
values belong.

Templates are Scheme, not JSON node graphs. There is no intermediate
representation between the template and the executable artifact. The compiler
validates slot values against the template's slot schema, performs safe
substitution of `{{slot}}` markers with concrete values, and hashes the
result. The output is executable Scheme that runs directly in the Racket
sandbox.

A template file contains two parts:

1. **Frontmatter** (structured YAML or JSON in a comment block at the top):
   metadata, supported shapes, trigger/reject conditions, slot schema,
   structural profile, verification rules, and examples.
2. **Body** (Scheme code): the actual computation using primitive runtime
   bindings with `{{slot}}` markers for content-specific values.

The frontmatter stores:

- `name` and `version`,
- supported TaskShape/DataShape combinations,
- trigger conditions and rejection conditions (Scheme predicates evaluated
  against classification hints — e.g., `(> item_count 1)`, `(eq? independent #t)`),
- typed slots with defaults, enums, ranges, required fields, and descriptions,
- model requirements such as JSON mode or image support,
- output shape and schema expectations,
- expected call formulas and structural profiles,
- verification rules and dry-run warnings.

The Scheme body uses only:

- primitive runtime bindings (section 9),
- compiler-owned helper bindings (prefixed with `__`),
- `{{slot_name}}` markers that the compiler substitutes before execution.

The planner reads template frontmatter and fills slots. The compiler
validates slot values, substitutes them into the template body, and stores
the result as an immutable artifact. Agents interact only with templates
(via `plan_strategy` and `compile_strategy`).

This division is important:

- LLMs choose strategy intent and content slots (template selection + slot
  filling).
- Deterministic code validates slots and substitutes them safely
  (compilation). No code generation or IR translation is involved.
- Verification checks the compiled artifact before real model calls happen.

---

## 3. Public MCP API

The greenfield server should start with a small artifact-based MCP surface.

| Tool | Purpose |
|---|---|
| `load_context(data, name=None, metadata=None)` | Store input data and metadata; return `context_id`. |
| `get_context(context_id)` | Inspect metadata and optionally preview stored data. |
| `list_templates(filters=None)` | Show available templates and selection metadata. |
| `get_template(template_name, version=None)` | Return template schema and structural profile. |
| `plan_strategy(task, context_id=None, hints=None)` | Classify task/data and return `plan_id` plus proposed template invocation. |
| `compile_strategy(plan_id=None, template_invocation=None)` | Validate slots, substitute into template Scheme, and return `artifact_id`. |
| `get_artifact(artifact_id)` | Inspect artifact metadata, generated Scheme, hash, and compiler version. |
| `estimate_strategy(artifact_id)` | Static estimate without executing the Racket runtime. |
| `dry_run_strategy(artifact_id)` | Simulate runtime structure without real LLM calls. |
| `verify_strategy(artifact_id, dry_run_id=None, options=None)` | Gate artifact execution. |
| `execute_strategy(artifact_id, verification_id=None, plan_id=None, timeout=None)` | Execute a verified artifact. |
| `get_execution_trace(execution_id)` | Return call hierarchy, data flow, stdout, errors, token usage, and checkpoints. |
| `get_status(execution_id=None)` | Return server/runtime/call status. |
| `cancel_call(call_id=None, execution_id=None)` | Cancel one call or an entire execution. |
| `reset_runtime(scope="session")` | Reset sandbox state without deleting durable artifacts by default. |


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

def list_templates(filters_json: str | None = None) -> str: ...

def get_template(
    template_name: str,
    version: str | None = None,
) -> str: ...

def plan_strategy(
    task: str,
    context_id: str | None = None,
    hints_json: str | None = None,
) -> str: ...

def compile_strategy(
    plan_id: str | None = None,
    template_invocation_json: str | None = None,
    options_json: str | None = None,
) -> str: ...

def get_artifact(
    artifact_id: str,
    include_scheme: bool = False,
) -> str: ...

def estimate_strategy(
    artifact_id: str,
    assumptions_json: str | None = None,
) -> str: ...

def dry_run_strategy(
    artifact_id: str,
    options_json: str | None = None,
) -> str: ...

def verify_strategy(
    artifact_id: str,
    dry_run_id: str | None = None,
    policy_json: str | None = None,
) -> str: ...

async def execute_strategy(
    artifact_id: str,
    verification_id: str | None = None,
    plan_id: str | None = None,
    timeout_seconds: int | None = None,
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
```

`*_json` parameters are JSON strings because MCP clients vary in how reliably
they support nested structured arguments. The server should parse and validate
them into typed internal models immediately.

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

`include_data=true` should be allowed only for small contexts or explicit debug
settings. Large context retrieval should default to previews and metadata.

### 4.3 `list_templates`

Request:

```json
{
  "filters": {
    "task_shape": "Batch",
    "data_shape": "FlatList",
    "requires_multimodal": false,
    "max_expected_calls": 200
  }
}
```

Response:

```json
{
  "status": "ok",
  "templates": [
    {
      "template_name": "batch_extract_reduce",
      "version": "1.0.0",
      "task_shapes": ["Batch", "Synthesize"],
      "data_shapes": ["FlatList", "ChunkedSingular"],
      "summary": "Extract independently, then synthesize with tree reduction.",
      "primitives_used": ["map-async", "tree-reduce"],
      "slot_count": 9
    }
  ]
}
```

### 4.4 `get_template`

Request:

```json
{
  "template_name": "batch_extract_reduce",
  "version": "1.0.0"
}
```

Response:

```json
{
  "status": "ok",
  "template": {
    "template_name": "batch_extract_reduce",
    "version": "1.0.0",
    "summary": "...",
    "slot_schema": {},
    "structural_profile": {},
    "verification_rules": []
  }
}
```

By default, `get_template` returns frontmatter metadata only. The template's
Scheme body (with `{{slot}}` markers) can optionally be included for
debugging but is not needed for normal agent workflows.

### 4.5 `plan_strategy`

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
    "Call compile_strategy(plan_id=plan_01HX...)"
  ]
}
```

Planner output must not include raw Scheme. If no template fits, the planner
should return `status: "no_template"` with a recommendation that the user
create a new template, including the inferred TaskShape/DataShape and a
description of what the template would need.

### 4.6 `compile_strategy`

Purpose: validate slot values, substitute them into a template, and store the
result as an immutable compiled artifact.

Request:

```json
{
  "plan_id": "plan_01HX...",
  "template_invocation": {
    "template_name": "batch_extract_reduce",
    "template_version": "1.0.0",
    "slot_values": {}
  },
  "options": {
    "strict": true
  }
}
```

At least one of `plan_id` or `template_invocation` is required. If `plan_id`
is provided and already contains a recommended template invocation,
`compile_strategy` can use it directly.

Compilation steps:

1. Load the template `.rkt` file and parse its frontmatter.
2. Validate `slot_values` against the template's `slot_schema`.
3. Substitute all `{{slot}}` markers with concrete values (safe substitution).
4. Verify the result contains no remaining `{{...}}` markers.
5. Verify the result uses only allowed primitive bindings.
6. Hash the resulting Scheme code.
7. Store the artifact record.

Response:

```json
{
  "status": "ok",
  "artifact_id": "art_01HX...",
  "plan_id": "plan_01HX...",
  "artifact": {
    "source_type": "template_invocation",
    "template_name": "batch_extract_reduce",
    "template_version": "1.0.0",
    "compiler_version": "0.1.0",
    "code_hash": "sha256:...",
    "primitives_used": ["map-async", "tree-reduce", "llm-query-async", "llm-query"],
    "context_ids": ["ctx_01HX..."],
    "static_profile": {
      "min_calls": 1,
      "expected_calls_formula": "N + ceil(N/B) + ... + 1",
      "max_concurrency": 20,
      "recursive_depth": 0
    }
  },
  "next_actions": [
    "Call estimate_strategy(artifact_id=art_01HX...)",
    "Call dry_run_strategy(artifact_id=art_01HX...)"
  ]
}
```

### 4.7 `get_artifact`

Request:

```json
{
  "artifact_id": "art_01HX...",
  "include_scheme": false
}
```

Response:

```json
{
  "status": "ok",
  "artifact": {
    "artifact_id": "art_01HX...",
    "plan_id": "plan_01HX...",
    "source_type": "template_invocation",
    "template_name": "batch_extract_reduce",
    "slot_values": {},
    "compiler_version": "0.1.0",
    "code_hash": "sha256:...",
    "generated_scheme": null
  }
}
```

`include_scheme=true` is an inspection option, not an execution path. Since
templates are already Scheme, the compiled artifact is human-readable — it
is the template with slots filled in.

### 4.8 `estimate_strategy`

Request:

```json
{
  "artifact_id": "art_01HX...",
  "assumptions": {
    "item_count": 100,
    "avg_input_tokens": 800,
    "avg_output_tokens": 250
  }
}
```

Response:

```json
{
  "status": "ok",
  "artifact_id": "art_01HX...",
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
  "warnings": []
}
```

### 4.9 `dry_run_strategy`

Request:

```json
{
  "artifact_id": "art_01HX...",
  "options": {
    "mock_prefix": "[dry-run]",
    "deterministic_await_any": true,
    "max_simulated_items": 1000
  }
}
```

Response:

```json
{
  "status": "ok",
  "dry_run_id": "dry_01HX...",
  "artifact_id": "art_01HX...",
  "summary": {
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
    "Call verify_strategy(artifact_id=art_01HX..., dry_run_id=dry_01HX...)"
  ]
}
```

### 4.10 `verify_strategy`

Request:

```json
{
  "artifact_id": "art_01HX...",
  "dry_run_id": "dry_01HX...",
  "policy": {
    "max_llm_calls": 500,
    "max_concurrency": 50,
    "max_recursive_depth": 3,
    "allow_python_bridge": true,
    "allow_multimodal": true,
    "semantic_review": "off | cheap | required"
  }
}
```

Response:

```json
{
  "status": "ok",
  "verification_id": "ver_01HX...",
  "decision": "pass",
  "artifact_id": "art_01HX...",
  "dry_run_id": "dry_01HX...",
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
    }
  ],
  "warnings": [],
  "next_actions": [
    "Call execute_strategy(artifact_id=art_01HX..., verification_id=ver_01HX...)"
  ]
}
```

If verification fails, return `status: "error"` and still store a
`verification_id` with `decision: "fail"` so the user can inspect the reasons.

### 4.11 `execute_strategy`

Request:

```json
{
  "artifact_id": "art_01HX...",
  "verification_id": "ver_01HX...",
  "plan_id": "plan_01HX...",
  "timeout_seconds": 900,
  "runtime_options": {
    "progress_interval_seconds": 2,
    "checkpoint_prefix": "ace2-run",
    "max_stdout_chars": 4000
  }
}
```

Response:

```json
{
  "status": "ok",
  "execution_id": "exec_01HX...",
  "artifact_id": "art_01HX...",
  "verification_id": "ver_01HX...",
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

If `verification_id` is omitted, execution should look up the latest passing
verification for the artifact. If no passing verification exists, execution
must fail with a structured error directing the agent to run
`verify_strategy` first.

### 4.12 `get_execution_trace`

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

### 4.13 `get_status`

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

### 4.14 `cancel_call`

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

---

## 5. Durable Record Schemas

The API schemas above are request/response contracts. The server should also
store durable records with explicit schemas so history, verification, and
replay are reliable.

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
  "compiler": {
    "name": "rlm-scheme-template-compiler",
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
      "message": "No removed or unsafe runtime names used."
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
  "state": "queued | running | finished | failed | cancelled",
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

---

## 6. Slot Substitution Model

Templates are Scheme files with `{{slot_name}}` markers. The compiler fills
these markers with concrete values to produce executable artifacts. There is
no intermediate node graph or resolved template representation.

### Slot Markers

Slots use double-brace syntax: `{{slot_name}}`. The compiler substitutes
each marker with the corresponding value from the template invocation's
`slot_values`. Markers can appear anywhere in the Scheme body where a
literal value would be valid:

- String positions: `#:instruction {{map_instruction}}` → `#:instruction "Extract ACE2 mentions..."`
- Numeric positions: `#:max-concurrent {{max_concurrent}}` → `#:max-concurrent 20`
- Boolean positions: `#:json {{json_mode}}` → `#:json #t`
- Identifier positions: `#:model {{map_model}}` → `#:model "fast_text_model"`

### Substitution Rules

1. All `{{slot}}` markers must have corresponding values in `slot_values`.
   Missing required slots are compile errors.
2. Slot values are type-checked against the template's `slot_schema` before
   substitution.
3. String values are escaped and quoted. Numeric and boolean values are
   inserted as Scheme literals. Context IDs are inserted as quoted strings.
4. Substitution is safe — values cannot inject arbitrary Scheme code. The
   compiler rejects slot values that contain unbalanced parentheses, Scheme
   keywords, or other code injection attempts.
5. After substitution, the result must be syntactically valid Scheme that
   uses only primitive runtime bindings and compiler-owned helpers.

### Context References

Templates access loaded context data via the `__context-ref` helper:

```scheme
(define items (__context-ref "{{context_id}}" "{{items_path}}"))
```

`__context-ref` takes a context ID and a JSONPath expression
([RFC 9535](https://www.rfc-editor.org/rfc/rfc9535)) and returns the
extracted data at runtime. The compiler validates that `context_id` slots
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

- removed compound combinator names (`fan-out-aggregate`, `critique-refine`,
  `ensemble`, `vote`, `tiered`, `active-learning`, `fold-summarizing`),
- string `eval`,
- shell commands,
- filesystem access outside declared context/artifact/checkpoint stores,
- `unsafe-interpolate`, `unsafe-overwrite`, `unsafe-exec-sub-output`.

---

## 7. Example Template

Templates are `.rkt` files with structured frontmatter and Scheme code.
Frontmatter is a JSON block in a Racket block comment (`#| ... |#`) at the
top of the file. The Scheme body follows.

File:

```text
templates/batch_extract_reduce.rkt
```

Template:

```scheme
#|
{
  "schema_version": "1",
  "template_name": "batch_extract_reduce",
  "version": "1.0.0",
  "summary": "Run independent extraction over many items, then synthesize results with tree reduction.",
  "task_shapes": ["Batch", "Synthesize", "Composite"],
  "data_shapes": ["FlatList", "ChunkedSingular", "Tabular"],
  "output_shape": "one",
  "trigger_conditions": [
    "(> item_count 1)",
    "(eq? independent #t)",
    "(eq? output_type 'one)",
    "(eq? has_second_phase #t)"
  ],
  "reject_conditions": [
    "(and (eq? ordered #t) (eq? order_sensitive #t))",
    "(eq? requires_pairwise_comparison #t)"
  ],
  "slot_schema": {
    "type": "object",
    "required": [
      "context_id",
      "items_path",
      "map_instruction",
      "reduce_instruction",
      "map_model",
      "reduce_model"
    ],
    "properties": {
      "context_id": {
        "type": "string",
        "pattern": "^ctx_"
      },
      "items_path": {
        "type": "string",
        "default": "$.items"
      },
      "map_instruction": {
        "type": "string",
        "minLength": 10
      },
      "reduce_instruction": {
        "type": "string",
        "minLength": 10
      },
      "map_model": {
        "type": "string",
        "default": "fast_text_model"
      },
      "reduce_model": {
        "type": "string",
        "default": "quality_text_model"
      },
      "max_concurrent": {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": 20
      },
      "branch_factor": {
        "type": "integer",
        "minimum": 2,
        "maximum": 10,
        "default": 5
      },
      "json_mode": {
        "type": "boolean",
        "default": false
      },
      "checkpoint_every": {
        "type": ["integer", "null"],
        "minimum": 1,
        "default": null
      }
    }
  },
  "structural_profile": {
    "expected_calls_formula": "N + ceil(N/B) + ceil(ceil(N/B)/B) + ... + 1",
    "critical_path_formula": "1 + ceil(log_B(N))",
    "max_concurrency_slot": "max_concurrent",
    "recursive_depth": 0,
    "uses_python_bridge": false,
    "uses_multimodal": false
  },
  "verification_rules": [
    "context_id_exists",
    "items_path_resolves_to_list",
    "map_model_supports_json_if_json_mode",
    "expected_calls_within_policy",
    "max_concurrency_within_policy",
    "no_removed_compound_combinators",
    "no_unsafe_forms"
  ],
  "examples": [
    {
      "task": "Extract claims from papers and synthesize a literature review.",
      "slot_values": {
        "items_path": "$.papers",
        "map_instruction": "Extract the core claim, evidence, and uncertainty as JSON.",
        "reduce_instruction": "Synthesize the extracted claims into a literature review."
      }
    }
  ]
}
|#

;; Template body — Scheme code with {{slot}} markers.
;; The compiler substitutes slot values to produce the executable artifact.

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

Example compiled artifact (after slot substitution):

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

The compiled artifact is the template with all `{{slot}}` markers replaced
by concrete values. It uses only primitive runtime bindings and
compiler-owned helper bindings.

---

## 8. Model Registry

Templates should refer to model aliases, not hardcoded provider model names.
The server resolves aliases at compile or execution time through a model
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
      "cost_tier": "high"
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

Provider model names should live in configuration, not in templates or planner
prompts. Documentation examples should use aliases unless they are describing
provider configuration.

---

## 9. Runtime Basis

The Racket runtime should be small and primitive-only. Compound patterns belong
in templates or the compiler.

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
| Delegation | `recursive-spawn` | Nested orchestration with global depth limit. |
| Modifier | `memoized` | Cache by explicit key. |
| Modifier | `with-validation` | Wrap result validation. |
| Modifier | `try-fallback` | Error recovery. |
| State | `checkpoint` / `restore` | Durable partial results. |
| State | `tokens-used` / `rate-limits` | Runtime accounting. |
| State | `heartbeat` | Keep long executions alive. |
| Compute | `py-exec` / `py-eval` / `py-call` / `py-set!` | Controlled Python bridge for parsing, aggregation, and local computation. |

### Primitive Signatures And Semantics

The signatures below are the target public bindings inside compiler-generated
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

Compiler rule: use this for reduce/synthesis/refinement steps where the next
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

Compiler rule: use this inside `map-async`, `parallel`, and `race` bodies.

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
compiler-only `__await-all-syntax`.

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
- propagates per-item errors according to compiler-selected error policy.

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

The compiler should reject `parallel` bodies that call synchronous
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
- used by compiler for multi-phase templates,
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

- expose host accounting to compiler-generated strategies,
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
policy per primitive usage in their frontmatter; the compiler validates the
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
| `await-all` | `fail_fast` | If any handle errored, raises the first error. With `collect`, returns error markers in position. |
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

**Rate-limit and transient errors** are handled by the retry policy (see
section 11.6) before the error policy applies. Only non-retryable errors reach
the per-primitive error handling.

**Checkpoints and partial recovery:** Templates that process large item lists
should declare `checkpoint_every: N`. When `map-async` or `fold-sequential`
checkpoints, a `restore` call on re-execution skips already-completed items.
This interacts with error policies — `fail_fast` with checkpointing means the
failed execution can be retried from the last checkpoint, not from scratch.

### Remove As Runtime Combinators

These should not exist as runtime public names:

| Remove | Compile to |
|---|---|
| `fan-out-aggregate` | `map-async` plus `tree-reduce` or `fold-sequential`. |
| `critique-refine` | `iterate-until` with explicit generate/critique/refine state. |
| `ensemble` | `parallel` plus compiler-generated aggregation. |
| `vote` | `parallel` plus majority/plurality/consensus selection. |
| `tiered` | cheap `map-async`, filter/summarize, expensive review/synthesis. |
| `active-learning` | cheap `map-async`, uncertainty filter, expensive `map-async`. |
| `fold-summarizing` | `fold-sequential` with explicit summarization calls. |

### Remove Unsafe Public Escape Hatches

Do not expose public equivalents of:

- `unsafe-interpolate`,
- `unsafe-overwrite`,
- `unsafe-exec-sub-output`.

If the compiler needs privileged runtime hooks, keep them unbound in user
artifacts or place them behind private host-generated forms that templates
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
thunks that return async handles or compile into equivalent async structure.

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
should declare when Python computation is required, and the compiler should
generate constrained bridge calls.

### Recursive Delegation

Preserve recursive LLM orchestration, but make it artifact-aware:

- recursive calls compile sub-strategies, not arbitrary model-written Scheme,
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
- checkpoints.

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
disallowed operations. The compiler fills `{{slot}}` markers in template
Scheme code with concrete values — there is no intermediate node graph or
resolved template representation.

### 11.3 Template Catalog

See section 7 for the full template schema, and section 15 for the initial
template catalog. Templates live as `.rkt` files with JSON frontmatter:

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

### 11.4 Compiler

The compiler validates slot values and substitutes them into template Scheme
code to produce immutable artifacts.

Responsibilities:

- parse template frontmatter and Scheme body,
- validate slot values against the template's `slot_schema`,
- substitute `{{slot}}` markers with safe, type-appropriate values,
- reject values that could inject arbitrary Scheme code,
- verify all markers are resolved and only allowed primitives are used,
- calculate static structural profiles from template frontmatter,
- hash the resulting Scheme code,
- store artifact metadata.

The compiler should be deterministic: same inputs, same artifact hash.

The compiler does NOT:

- generate Scheme code from a node graph,
- translate between an IR and Scheme,
- produce source maps (the artifact IS the template with filled slots,
  so line numbers correspond directly).

### 11.5 Racket Runtime

The Racket runtime should be a sandboxed execution engine, not the planning
interface.

Responsibilities:

- evaluate compiler-generated Scheme artifacts,
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

---

## 12. Dry-Run And Verification

Dry-run and verification should be artifact-based.

### Dry-Run

Dry-run must simulate structure without real LLM calls:

- use pre-resolved fake futures for async calls,
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

### Verification

Verification is more useful than per-call template linting. It should focus on
the filled artifact that will actually run.

Check:

- artifact was compiler-generated,
- artifact hash matches stored code,
- template version is known,
- all required slots are filled,
- model names and capabilities are valid,
- JSON-mode instructions are compatible,
- image usage targets multimodal-capable models,
- no public unsafe forms are present,
- no raw code import path was used,
- expected call count is within configured limits,
- recursive depth is within configured limits,
- concurrency is within configured limits,
- context references exist,
- output schema is available when required,
- dry-run warnings are acceptable.

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

- a template invocation with slot values (primary path),
- a short list of alternative template invocations with estimated tradeoffs,
- a `no_template` recommendation describing the needed template for the user
  to create.

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
structured yes/no or multiple-choice questions (e.g., "Are items independent?
yes/no", "What is the per-item operation? transform/label/check"). Once
fields are filled, Level 1 runs deterministically on the complete fields.

The LLM never chooses templates directly. It only fills missing structured
fields that the deterministic classifier then uses.

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
| `Composite` | Multi-phase task. | compiled `sequence` of phase templates. |

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
Q2: Compile each phase independently.
Q3: Connect dependent phases with sequence.
Q4: Connect independent phases with parallel.
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

Every template should include:

- slot schema,
- expected call formula,
- structural profile,
- model capability requirements,
- verification rules,
- at least one example invocation,
- one compiler fixture showing the artifact after slot substitution.

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
  compiler.py
  dry_run.py
  verifier.py
  executor.py
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
  batch_extract_reduce.rkt
  batch_map.rkt
  ordered_synthesis_fold.rkt
  ...
docs/
  GREENFIELD-REWRITE-PLAN.md
  api-reference.md
  templates.md
  primitives.md
tests/
  test_id_flow.py
  test_mcp_api_schemas.py
  test_template_validation.py
  test_compiler.py
  test_runtime_primitives.py
  test_dry_run.py
  test_verifier.py
  test_executor.py
```

Module responsibilities:

| Module | Responsibility |
|---|---|
| `models.py` | Pydantic/dataclass schemas for all durable records and API payloads. |
| `ids.py` | ID generation and validation for `ctx_`, `plan_`, `art_`, `dry_`, `ver_`, `exec_`, `call_`. |
| `store.py` | Durable JSON or SQLite/PGlite storage abstraction. |
| `context_store.py` | Large context storage, previews, metadata, and path extraction. |
| `template_store.py` | Load, validate, list, and retrieve `.rkt` templates (parse frontmatter + body). |
| `classifier.py` | Deterministic TaskShape/DataShape rules. |
| `planner.py` | Template selection and plan record creation. |
| `compiler.py` | Slot validation and safe substitution into template Scheme code. |
| `dry_run.py` | Deterministic structural simulation. |
| `verifier.py` | Artifact checks and execution gate. |
| `executor.py` | Racket runtime lifecycle, progress, cancellation, and execution records. |
| `trace.py` | Trace event schema and aggregation. |
| `llm_provider.py` | Provider calls, retry, rate limits, token accounting. |
| `image_inputs.py` | Image resolution, MIME sniffing, size limits. |
| `python_bridge.py` | Controlled Python compute subprocess. |

---

## 17. Implementation Phases

### Phase 0: Decisions And Schemas

- Freeze public MCP API names.
- Define ID record schemas.
- Define template frontmatter schema and slot substitution rules (section 6).
- Define template file format (`.rkt` with JSON frontmatter).
- Decide initial store backend.
- Decide which Python bridge operations are allowed in templates.

Exit criteria:

- schemas checked into the repo,
- example records for all ID types,
- no raw Scheme public API in the design.

### Phase 1: Durable Store And MCP Skeleton

- Implement context, plan, artifact, dry-run, verification, execution stores.
- Implement ID generation and parent-child linking.
- Add MCP tools with stubbed behavior.
- Add `get_status`, `cancel_call`, and `reset_runtime` skeletons.

Exit criteria:

- object lifecycle can be created and inspected,
- parent ID chain is visible,
- tests prove ID flow.

### Phase 2: Minimal Racket Runtime

- Build sandbox lifecycle.
- Implement internal `llm-query`, syntax wrapping, `syntax-e`, `datum->syntax`,
  scope logging, `finish`, and scaffold protection.
- Implement `load-context` runtime command.
- Implement stdout/stderr capture and structured errors.

Exit criteria:

- one compiler-owned artifact can execute,
- syntax provenance appears in traces,
- scaffold overwrite attempts fail.

### Phase 3: Host Callback Loop

- Implement real model calls.
- Implement async futures.
- Implement `await`, `await-all`, `await-any`, and cancellation.
- Implement retry and rate-limit tracking.
- Implement progress reporting and heartbeat.

Exit criteria:

- concurrent fan-out works,
- cancellation works for active calls,
- rate limits and token usage appear in status.

### Phase 4: Primitive Runtime Basis

- Add `map-async`, `parallel`, `race`, `tree-reduce`, `fold-sequential`,
  `sequence`, `choose`, `iterate-until`, `recursive-spawn`, `memoized`,
  `with-validation`, and `try-fallback`.
- Keep compounds out of the runtime.
- Add checkpoint/restore and token-budget behavior.

Exit criteria:

- primitive tests cover success, failure, cancellation, and ordering semantics,
- no compound combinator names are exported.

### Phase 5: Template Catalog And Compiler

- Create initial `.rkt` templates for common shapes (see section 15).
- Implement template frontmatter parsing and validation.
- Implement slot validation and safe substitution.
- Store compiled artifacts with hashes.

Exit criteria:

- planner can select at least one template per common shape,
- compiler output is deterministic (same slots → same hash),
- artifacts are inspectable,
- no `{{slot}}` markers remain in compiled artifacts.

### Phase 6: Planner

- Implement deterministic TaskShape/DataShape classification.
- Add structured hints to `plan_strategy`.
- Use template metadata for selection.
- Return alternatives when tradeoffs are meaningful.

Exit criteria:

- plan output is template invocations only,
- composite classification preserves phases,
- tests cover ambiguous and multi-phase inputs.

### Phase 7: Estimate, Dry-Run, Verify

- Implement static estimates from artifact profiles.
- Implement dry-run execution mode with per-call context.
- Special-case `await-any` and batch await semantics.
- Implement `verify_strategy`.

Exit criteria:

- dry-run has no global mode leak,
- tree-reduce formula is correct,
- failed verification blocks execution by default.

### Phase 8: Execute And Trace

- Implement `execute_strategy`.
- Link executions to verification and artifact records.
- Assemble full traces with scope logs, call metrics, stdout, errors, and
  checkpoints.
- Support repeated executions of the same artifact.

Exit criteria:

- successful and failed executions are inspectable,
- execution IDs remain useful after runtime reset,
- cancellation produces a traceable terminal state.

### Phase 9: Advanced Features

- Add multimodal template support.
- Add controlled Python compute phases.
- Add recursive artifact-aware delegation.
- Add checkpoint recovery workflows.
- Add history-based planner feedback.

Exit criteria:

- large-context workflows can use chunking and recursion,
- Python bridge is used only by trusted templates,
- planner can use execution history without copying raw traces into prompts.

### Phase 10: Documentation And Migration

- Rewrite README around artifact workflow.
- Replace old raw-code API docs.
- Replace combinator docs with primitive runtime docs and template docs.
- Add examples for each ID stage.
- Keep the old implementation referenced only as historical context.

Exit criteria:

- docs do not instruct agents to write raw Scheme,
- docs do not mention removed compound runtime combinators as public API,
- docs show the complete `context_id -> ... -> execution_id` flow.

---

## 18. Test Plan

Minimum test coverage:

- schema validation for every ID record,
- parent-child ID flow,
- context metadata classification,
- template validation,
- compiler determinism,
- generated Scheme hash verification,
- no public `execute_scheme` or `dry_run_scheme` MCP tools,
- no exported compound runtime combinators,
- no public unsafe escape hatches,
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
- verification pass/warn/fail behavior,
- execution trace persistence.

---

## 19. End-to-End Walkthrough

This section shows every MCP call in sequence for a realistic task: "Extract
ACE2 protein mentions from 100 research papers and synthesize a report."

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
    "next_actions": ["Call compile_strategy(plan_id=plan_b2c1)"]
  }
```

### Step 3: Compile strategy

Agent accepts the recommended template. Compilation validates slots,
substitutes them into the template Scheme, and stores an immutable artifact.

```
→ compile_strategy(plan_id: "plan_b2c1")

← {
    "status": "ok",
    "artifact_id": "art_e4d9",
    "plan_id": "plan_b2c1",
    "artifact": {
      "source_type": "template_invocation",
      "template_name": "batch_extract_reduce",
      "template_version": "1.0.0",
      "compiler_version": "0.1.0",
      "code_hash": "sha256:a1b2c3...",
      "primitives_used": ["map-async","tree-reduce","llm-query-async","llm-query"],
      "context_ids": ["ctx_7f3a"],
      "static_profile": {
        "min_calls": 1,
        "expected_calls_formula": "100 + 20 + 4 + 1 = 125",
        "max_concurrency": 20,
        "recursive_depth": 0
      }
    },
    "next_actions": ["Call dry_run_strategy(artifact_id=art_e4d9)"]
  }
```

### Step 4: Dry run

Agent runs a structural simulation. No real LLM calls are made.

```
→ dry_run_strategy(artifact_id: "art_e4d9")

← {
    "status": "ok",
    "dry_run_id": "dry_1a2b",
    "artifact_id": "art_e4d9",
    "summary": {
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
    "next_actions": ["Call verify_strategy(artifact_id=art_e4d9, dry_run_id=dry_1a2b)"]
  }
```

### Step 5: Verify

Agent gates execution. Verification checks the artifact against policy limits.

```
→ verify_strategy(
    artifact_id: "art_e4d9",
    dry_run_id: "dry_1a2b",
    policy_json: "{\"max_llm_calls\":500,\"max_concurrency\":50,\"max_recursive_depth\":3}"
  )

← {
    "status": "ok",
    "verification_id": "ver_3c4d",
    "decision": "pass",
    "artifact_id": "art_e4d9",
    "dry_run_id": "dry_1a2b",
    "checks": [
      {"name":"artifact_hash","status":"pass","message":"Code hash matches."},
      {"name":"primitive_allowlist","status":"pass","message":"Only primitives used."},
      {"name":"call_count_limit","status":"pass","message":"125 <= 500."},
      {"name":"concurrency_limit","status":"pass","message":"20 <= 50."},
      {"name":"model_capabilities","status":"pass","message":"fast_text_model supports json mode."},
      {"name":"context_exists","status":"pass","message":"ctx_7f3a exists."}
    ],
    "warnings": [],
    "next_actions": ["Call execute_strategy(artifact_id=art_e4d9, verification_id=ver_3c4d)"]
  }
```

### Step 6: Execute

Agent runs the verified artifact. Real LLM calls happen here.

```
→ execute_strategy(
    artifact_id: "art_e4d9",
    verification_id: "ver_3c4d",
    plan_id: "plan_b2c1",
    timeout_seconds: 900,
    runtime_options_json: "{\"progress_interval_seconds\":5}"
  )

  ... (progress notifications arrive every 5 seconds) ...

← {
    "status": "ok",
    "execution_id": "exec_5e6f",
    "artifact_id": "art_e4d9",
    "verification_id": "ver_3c4d",
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

### Step 7: Inspect trace (optional)

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
ctx_7f3a (data) → plan_b2c1 (classification + template) → art_e4d9 (compiled Scheme)
  → dry_1a2b (structural simulation) → ver_3c4d (gate) → exec_5e6f (result + trace)
```

Each ID is durable and inspectable. The same artifact can be re-executed with
different data by creating a new context and re-compiling with updated
`context_id` in the slot values.

---

## 20. Open Design Decisions

These should be decided before implementation begins:

1. **Store backend.** Start with filesystem JSON (decided in section 11.1).
   Migrate to SQLite/PGlite if queryable history or concurrent access becomes
   a bottleneck. *(Partially decided — revisit if needed.)*
2. **Artifact mutability.** Prefer immutable artifacts. Edits create new
   artifact IDs. *(Decided.)*
3. **Recursive planning.** Decide whether recursive sub-plans are compiled
   ahead of time or generated at runtime under verification constraints.
4. **History feedback.** Decide which execution metrics influence future
   planning and how to avoid leaking sensitive data into planner prompts.

---

## 21. Success Criteria

The rewrite is successful when:

- agents never need to write Scheme,
- agents can still inspect generated Scheme for debugging,
- all execution goes through compiled artifacts,
- dry-run and verification happen before expensive calls,
- templates cover common orchestration shapes,
- compound combinators are gone from the runtime,
- unsafe public escape hatches are gone,
- large contexts are represented by IDs and metadata,
- recursive workflows remain possible,
- current operational features are preserved: progress, cancel, trace, rate
  limits, token accounting, checkpointing, multimodal input, and controlled
  Python compute.
