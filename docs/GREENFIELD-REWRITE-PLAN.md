# RLM-Scheme Greenfield Rewrite Plan

This is a fork of `IMPROVEMENT-PLAN.md` for a complete rewrite of
RLM-Scheme. The goal is not to modify the current codebase incrementally. The
goal is to rebuild the system around the intended architecture from the
beginning, using the current implementation as a feature inventory and
reference for proven runtime behavior.

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
| `plan_id` | Task intent and planning record: objective, constraints, inferred TaskShape/DataShape, selected template or Strategy Spec, and rationale. |
| `artifact_id` | Compiled executable strategy: template invocation or Strategy Spec, filled typed slots, generated internal Scheme, compiler version, and code hash. |
| `dry_run_id` | Structural simulation for an artifact: expected calls, fan-out, recursive depth, model mix, token/cost estimates, warnings, and failure risks. |
| `verification_id` | Verification decision: deterministic checks, dry-run interpretation, optional semantic review, pass/warn/fail status, and reasons. |
| `execution_id` | One real execution attempt: result, stdout, trace, call metrics, token usage, errors, checkpoints, and status history. |

Normal agent flow:

1. `load_context(data, name, metadata)` stores large input and returns a
   `context_id`.
2. `plan_strategy(task, context_id, hints)` classifies the work and returns a
   `plan_id` plus a template invocation or Strategy Spec.
3. `compile_strategy(plan_id | spec | template_invocation)` validates typed
   slots and returns an `artifact_id`.
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

A template stores:

- `name` and `version`,
- supported TaskShape/DataShape combinations,
- trigger conditions and rejection conditions,
- typed slots with defaults, enums, ranges, required fields, and descriptions,
- model requirements such as JSON mode or image support,
- output shape and schema expectations,
- expected call formulas and structural profiles,
- primitive-only Strategy Spec fragment or compiler-owned Scheme body,
- verification rules and dry-run warnings.

The planner reads template metadata and fills slots. The compiler owns all
Scheme generation.

This division is important:

- LLMs choose strategy intent and content slots.
- Deterministic code chooses syntax, primitive composition, and safety checks.
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
| `plan_strategy(task, context_id=None, hints=None)` | Classify task/data and return `plan_id` plus proposed template/spec. |
| `compile_strategy(plan_id=None, spec=None, template_invocation=None)` | Produce a compiled artifact and return `artifact_id`. |
| `get_artifact(artifact_id)` | Inspect artifact metadata, generated Scheme, hash, and compiler version. |
| `estimate_strategy(artifact_id)` | Static estimate without executing the Racket runtime. |
| `dry_run_strategy(artifact_id)` | Simulate runtime structure without real LLM calls. |
| `verify_strategy(artifact_id, dry_run_id=None, options=None)` | Gate artifact execution. |
| `execute_strategy(artifact_id, verification_id=None, plan_id=None, timeout=None, force=False)` | Execute a verified artifact. |
| `get_execution_trace(execution_id)` | Return call hierarchy, data flow, stdout, errors, token usage, and checkpoints. |
| `get_status(execution_id=None)` | Return server/runtime/call status. |
| `cancel_call(call_id=None, execution_id=None)` | Cancel one call or an entire execution. |
| `reset_runtime(scope="session")` | Reset sandbox state without deleting durable artifacts by default. |
| `get_usage_guide()` | Explain the artifact workflow. |

Do not expose these as public tools:

- `execute_scheme(code, ...)`,
- `dry_run_scheme(code, ...)`,
- arbitrary raw code import,
- public unsafe interpolation/overwrite/eval helpers.

Internal test helpers may still invoke lower-level runtime functions, but the
MCP contract should only expose artifact-based orchestration.

---

## 4. Runtime Basis

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

## 5. Preserved Feature Inventory

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

## 6. Architecture Components

### 6.1 Durable Store

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

### 6.2 Strategy Spec Schema

Define a Strategy Spec that is expressive enough for all templates but still
compiler-friendly.

The spec should represent:

- primitive nodes,
- model calls,
- async fan-out,
- reductions,
- validation,
- fallback,
- Python compute phases,
- recursion,
- context references,
- output schemas,
- cost/quality hints.

The planner can emit Strategy Specs directly for composite tasks, but it should
prefer template invocations when a template fits.

### 6.3 Template Catalog

Templates should live as structured files, for example:

```text
templates/
  batch_extract.yaml
  batch_extract_reduce.yaml
  ordered_synthesis.yaml
  compare_alternatives.yaml
  refine_until_valid.yaml
  validate_items.yaml
  tiered_review.yaml
```

Template validation is a developer/CI concern. Runtime verification assumes
trusted templates are structurally valid, but still verifies filled artifacts.

### 6.4 Compiler

The compiler converts template invocations or Strategy Specs into internal
Scheme artifacts.

Responsibilities:

- validate slots and types,
- reject unsupported shapes,
- select primitive compositions,
- generate Racket code,
- attach source maps from spec nodes to Scheme fragments,
- calculate static structural profiles,
- hash generated code,
- store artifact metadata.

The compiler should be deterministic: same inputs, same artifact hash.

### 6.5 Racket Runtime

The Racket runtime should be a sandboxed execution engine, not the planning
interface.

Responsibilities:

- evaluate compiler-generated Scheme artifacts,
- enforce resource limits,
- preserve syntax hygiene,
- call back to Python host for LLM/Python/checkpoint/rate-limit operations,
- emit stdout, scope logs, and trace events,
- protect scaffold bindings.

### 6.6 Python Host

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

---

## 7. Dry-Run And Verification

Dry-run and verification should be artifact-based.

### Dry-Run

Dry-run must simulate structure without real LLM calls:

- use pre-resolved fake futures for async calls,
- special-case `await-any` so exactly one pending handle completes per call,
- special-case batch await behavior deterministically,
- record fan-out, call counts, model mix, recursive depth, and estimated tokens,
- avoid shared global execution-mode state that can leak across concurrent MCP
  calls.

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
- template/spec version is known,
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

## 8. Planning And Classification

The planner should classify work before choosing a template.

Keep the TaskShape/DataShape model from `IMPROVEMENT-PLAN.md`:

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

- a template invocation with slot values,
- a Strategy Spec,
- a short list of alternatives with estimated tradeoffs.

Planning output should not include raw Scheme.

---

## 9. Implementation Phases

### Phase 0: Decisions And Schemas

- Freeze public MCP API names.
- Define ID record schemas.
- Define Strategy Spec schema.
- Define template schema.
- Decide initial store backend.
- Decide which Python bridge operations are allowed in compiler output.

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

- Create initial templates for common shapes.
- Implement template validation.
- Implement Strategy Spec compiler.
- Store compiled artifacts with source maps and hashes.
- Generate primitive-only Scheme.

Exit criteria:

- planner can select at least one template per common shape,
- compiler output is deterministic,
- artifacts are inspectable.

### Phase 6: Planner

- Implement deterministic TaskShape/DataShape classification.
- Add structured hints to `plan_strategy`.
- Use template metadata for selection.
- Return alternatives when tradeoffs are meaningful.

Exit criteria:

- plan output is template/spec only,
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
- Python bridge is generated only by trusted templates/specs,
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

## 10. Test Plan

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

## 11. Open Design Decisions

These should be decided before implementation begins:

1. **Store backend.** Filesystem JSON is easiest. SQLite/PGlite is better for
   queryable history and concurrent access.
2. **Artifact mutability.** Prefer immutable artifacts. Edits create new
   artifact IDs.
3. **Python bridge policy.** Decide whether only templates can request Python
   phases, or whether Strategy Specs can request them with strict validation.
4. **Recursive planning.** Decide whether recursive sub-plans are compiled
   ahead of time or generated at runtime under verification constraints.
5. **Template language.** Decide whether templates store Strategy Spec
   fragments only, or also compiler-owned Scheme snippets.
6. **History feedback.** Decide which execution metrics influence future
   planning and how to avoid leaking sensitive data into planner prompts.

---

## 12. Success Criteria

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

