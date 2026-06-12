# RLM-Scheme Greenfield Implementation Plan

**Status:** Normative implementation plan.
**Audience:** an implementing agent.
**Design basis:** a practical Racket/Python implementation of the lambda-RLM idea: long-context reasoning should be recursive, typed, and combinator-driven, with LLM calls confined to bounded leaf subproblems.

RLM-Scheme is an MCP server for auditable long-context LLM computation. It treats the input context as an external environment, constructs a typed functional execution plan over that environment, dry-runs the plan to compute cost and resource bounds, verifies the plan against policy, and executes it once. The LLM does not write orchestration code.

The central move is:

```text
open-ended model-authored REPL loop
  -> deterministic typed combinator program with bounded neural leaf calls
```

This design is intentionally narrower than a general agent framework. It is a runtime for structured decomposition, mapping, filtering, reduction, comparison, validation, and synthesis over contexts larger than a model can safely consume in one prompt.

---

## 0. Motivation and Scope

### 0.1 Problem

Large contexts degrade LLM reliability. Feeding all evidence into one call can exceed the model window, force truncation, or create "context rot" where relevant details are present but not used reliably.

Recursive Language Models address this by keeping the prompt outside the model context and letting the model inspect and recurse over pieces. The unsafe version lets the model write arbitrary control code in a REPL. That gives flexibility but creates avoidable failure modes:

- generated code may not parse;
- generated code may crash;
- recursion may not terminate;
- costs and latency are hard to predict;
- intermediate values may be malformed;
- structural control and semantic reasoning are both delegated to the stochastic model.

RLM-Scheme keeps the useful part, prompt-as-environment, and removes the open-ended part. Control flow is symbolic, typed, deterministic, and audited. The model is used as a bounded oracle only at leaves or explicitly declared neural nodes.

### 0.2 Goals

RLM-Scheme **[MUST]** provide:

- deterministic orchestration from a compact combinator library;
- typed inputs, outputs, and chain boundaries;
- no LLM-authored control code;
- dry-run resource estimates before live execution;
- termination and budget checks before live execution;
- traceable artifacts and executions;
- MCP tools usable by coding agents.

RLM-Scheme **[MUST NOT]** attempt to be:

- a general autonomous agent loop;
- a framework for arbitrary generated code execution;
- a hidden prompt-chain DSL with ad hoc string substitution;
- a proof assistant for semantic correctness.

### 0.3 Practical Thesis

The expected capability gain comes from better computation structure, not from assuming the model has become smarter. Decomposition, bounded leaf calls, symbolic map/filter/reduce, typed intermediate values, and verification can make some long-context tasks more reliable than one-shot prompting or model-authored REPL loops.

---

## 1. System Architecture

### 1.1 Two-Process Runtime

```text
Python host
  MCP server
  Store / ContextStore
  TypeRegistry / ModelRegistry
  CombinatorRegistry / TemplateRegistry
  Planner / CostAnalyzer / VerificationEngine
  LLM providers / cache / budget monitor
  GateManager / CheckpointManager / TraceStore
  PyExecSandbox
        |
        | JSON Lines over stdin/stdout
        v
Racket runtime subprocess
  typed combinator evaluator
  prompt environment access
  sandboxed execution
  effect requests to Python
```

Python owns state, policy, model calls, cost accounting, py-exec, persistence, and MCP tools.

Racket owns typed functional control flow. It evaluates a prebuilt combinator program. It does not receive user data by textual substitution. It receives data as JSON values and context bindings.

### 1.2 Prompt as Environment

`load_context` stores the input outside the model window:

```text
ctx_... -> ContextRecord(data, metadata, semantic_type)
```

The runtime can access context slices symbolically through primitives such as `peek`, `slice`, `split`, and `context-items`. Only bounded subprompts are sent to the LLM.

### 1.3 Leaf Oracle

The LLM is modeled as a leaf oracle:

```text
M : BoundedPrompt -> TypedValue
```

Every live LLM call **[MUST]** satisfy:

- prompt estimate <= selected model context window;
- requested output type is known;
- call is counted against budget;
- call is recorded in the trace;
- call is served only after verification passes.

---

## 2. Execution Lifecycle

The user-facing MCP lifecycle is:

```text
load_context       -> ctx_...
plan_strategy      -> plan_...
dry_run_strategy   -> dry_... and art_...
execute_strategy   -> ver_... and exec_...
get_execution_trace
```

Internally, execution follows seven phases.

### Phase 1: Environment Initialization

The host stores input data and metadata as a context record. The Racket runtime receives context data only when evaluating a dry run or live execution.

### Phase 2: Task Detection

The planner classifies:

- task kind;
- structural data shape;
- semantic input type;
- requested output type;
- whether the input fits in the selected model window.

Task detection **[SHOULD]** be deterministic from hints and metadata. If hints are insufficient, the planner **[MAY]** use a single bounded model call to choose from a fixed menu.

### Phase 3: Direct Dispatch

If the input fits the model window and the task does not require symbolic decomposition, the planner may choose a direct typed leaf call:

```text
direct_call : InputType -> OutputType
```

This still goes through dry run and verification.

### Phase 4: Recursive Planning

If the input does not fit, the planner constructs a recursive combinator program:

```text
solve(x):
  if size(x) <= leaf_threshold:
    return leaf_call(x)
  else:
    parts = split(x, split_factor)
    kept = filter(parts)
    partials = map(solve, kept)
    return compose(partials)
```

The planner chooses:

- `split_factor`: number of chunks per recursive split;
- `leaf_threshold_tokens`: maximum prompt size for a leaf call;
- `max_depth`: maximum recursion depth;
- `composition_operator`: how partials are reduced;
- `task_plan`: typed combinator expression.

### Phase 5: Dry Run

Dry run evaluates the exact combinator program in simulate mode. It returns:

- aggregate simulation stats;
- per-call simulated leaf records;
- call graph;
- token estimate;
- cost range;
- max concurrency;
- critical path;
- recursion depth.

No live LLM call occurs during dry run.

### Phase 6: Verification

Verification checks the artifact, types, effects, resource bounds, model aliases, policy, and dry-run freshness. A live execution may start only if verification decision is `pass`.

### Phase 7: Single Live Execution

The runtime executes the prebuilt program once. There is no open-ended loop in which the LLM writes new control code. Recursive calls are internal to the combinator program and bounded by the planned depth.

---

## 3. Typed Value System

The type system is semantic and JSON-backed. It is not domain-specific to academic papers, legal documents, code, or finance.

### 3.1 Type Registry

`config/types.json` defines semantic types:

```json
{
  "Text": {"schema": {"type": "string"}},
  "TextDocument": {
    "schema": {
      "type": "object",
      "required": ["id", "text"],
      "properties": {
        "id": {"type": "string"},
        "text": {"type": "string"},
        "metadata": {"type": "object"}
      }
    }
  },
  "DocumentList": {
    "schema": {
      "type": "array",
      "items": {"$ref": "TextDocument"}
    }
  },
  "Finding": {
    "schema": {
      "type": "object",
      "required": ["finding", "source_id"],
      "properties": {
        "finding": {"type": "string"},
        "source_id": {"type": "string"},
        "evidence": {"type": "array", "items": {"$ref": "EvidenceSpan"}}
      }
    }
  },
  "FindingSet": {
    "schema": {"type": "array", "items": {"$ref": "Finding"}}
  },
  "Report": {
    "schema": {
      "type": "object",
      "required": ["summary"],
      "properties": {"summary": {"type": "string"}}
    }
  }
}
```

Required generic core types:

- `Any`
- `Text`
- `TextDocument`
- `DocumentList`
- `Record`
- `RecordList`
- `Table`
- `TableSet`
- `Finding`
- `FindingSet`
- `Claim`
- `ClaimSet`
- `EvidenceSpan`
- `Conflict`
- `ConflictSet`
- `Candidate`
- `CandidateSet`
- `RankedCandidateSet`
- `ValidationResult`
- `Report`

Domain packs **[MAY]** add subtypes such as `AcademicPaper`, `Contract`, `SourceFile`, or `TransactionRecord`, but core planning **[MUST]** work without them.

### 3.2 Type Compatibility

Template and combinator composition uses:

```text
producer.output_type <= consumer.input_type
```

Compatibility is valid when:

- names are equal; or
- the type registry declares a subtype relation; or
- the consumer accepts `Any`.

There is no implicit compatibility between arbitrary JSON schemas.

### 3.3 Guarantees

Templates and combinator plans may declare runtime-checkable guarantees:

- `schema-valid`
- `preserves-source-id`
- `evidence-required`
- `one-output-per-input`
- `order-preserving`
- `deduplicated`
- `conflicts-preserved`

Guarantees become verification obligations when declared.

Example:

```racket
(guarantees
  (schema-valid)
  (preserves-source-id)
  (evidence-required))
```

### 3.4 Output Validation

Every leaf call and every chain step **[MUST]** validate its output against the declared semantic output type before the value can flow to the next step.

Invalid output follows the active error policy:

- `fail_fast`: fail the execution;
- `skip_and_log`: record error and use `null`;
- `retry_then_skip`: retry provider policy, then skip.

---

## 4. Combinator Runtime

### 4.1 Core Combinators

The runtime exposes a compact typed library:

| Combinator | Type Shape | Meaning |
|---|---|---|
| `split` | `A -> List[A]` | Partition a value into bounded parts. |
| `peek` | `ContextRef x Range -> Text` | Read a bounded slice from context. |
| `map` | `(A -> B) x List[A] -> List[B]` | Apply a function to each item. |
| `filter` | `(A -> Bool) x List[A] -> List[A]` | Keep selected items. |
| `reduce` | `(B x B -> B) x List[B] -> B` | Combine values. |
| `concat` | `List[Text] -> Text` | Join text values. |
| `cross` | `List[A] x List[B] -> List[Pair[A,B]]` | Cartesian product. |
| `leaf-call` | `BoundedPrompt -> TypedValue` | Invoke the LLM oracle. |
| `validate` | `Type x Value -> ValidationResult` | Validate against semantic type. |
| `fix` | `(F -> F) -> F` | Tie bounded recursive programs. |

The library may also provide pragmatic extensions:

- `map-async`
- `tree-reduce`
- `fold-sequential`
- `race`
- `checkpoint`
- `gate`
- `partial-result`
- `py-exec`

These extensions **[MUST]** be declared as effects where applicable.

### 4.2 Effects

Effects are explicit:

- `LLM`
- `PY_EXEC`
- `CHECKPOINT`
- `GATE`
- `PARTIAL_RESULT`

Every template or plan has an inferred effect set from its body. Verification checks:

```text
required_effects <= policy.allowed_effects
```

`py-eval` is syntax sugar for `py-exec` and requires `PY_EXEC`.

### 4.3 Totality and Determinism

All symbolic combinators **[MUST]** be total and deterministic over valid typed inputs. Partial operations must return structured errors, not crash the runtime.

The LLM is the only nondeterministic semantic oracle unless `py-exec` is explicitly allowed.

---

## 5. Recursive Executor

### 5.1 Fixed Recursive Shape

Recursive plans are represented as a fixed combinator term, not as model-authored code.

Abstract form:

```text
fix solver.
  lambda input.
    if size(input) <= leaf_threshold:
      leaf-call(format_leaf(input))
    else:
      reduce compose
        (map solver
          (filter keep?
            (split input split_factor)))
```

The implementation may encode this as Racket S-expressions, but the recursive shape **[MUST]** be constructed by the deterministic planner.

### 5.2 Termination Conditions

A recursive plan is valid only if:

- `split_factor >= 2`;
- `leaf_threshold_tokens > 0`;
- every split child has strictly smaller estimated size than its parent when parent size exceeds threshold;
- `max_depth` is finite;
- simulated `recursive_depth <= policy.max_recursion_depth`.

If the size-decrease check cannot be proven from the splitter, verification fails.

### 5.3 Call Count Bound

For a balanced recursive plan with branching factor `b`, input size `n`, and leaf threshold `tau`:

```text
depth = ceil(log_b(ceil(n / tau)))
leaf_calls <= b^depth
internal_nodes <= (b^depth - 1) / (b - 1)
```

The dry run measures exact counts for the instantiated plan. The analyzer records both:

- closed-form upper bound;
- simulated exact count.

Verification fails if simulated counts exceed closed-form bounds.

### 5.4 Cost Bound

Leaf call cost:

```text
prompt_tokens(call) = ceil((instruction_chars + input_chars) / 4)
completion_tokens(call) = max_tokens or model.default_completion_estimate
cost(call) = prompt_tokens * input_rate + completion_tokens * output_rate
```

Total dry-run cost is the sum over simulated leaf calls. High estimate caps completion at `model.max_output_tokens`.

### 5.5 Partition Planning

The planner chooses `split_factor` and `leaf_threshold_tokens` to satisfy:

- every leaf fits the selected model window with margin;
- estimated call count <= policy;
- estimated cost <= policy;
- estimated critical path <= policy;
- recursion depth <= policy.

The initial implementation **MAY** use bounded search over candidate split factors rather than a closed-form optimum. Candidate values:

```text
split_factor in {2, 3, 4, 5, 8, 10}
leaf_threshold_tokens in {0.4W, 0.6W, 0.8W}
```

where `W` is the selected model context window.

---

## 6. Templates and Plan Artifacts

### 6.1 Role of Templates

Templates are audited combinator programs. They are not arbitrary scripts and not generated by the LLM.

One template may define:

- semantic input type;
- semantic output type;
- task kinds;
- required slots;
- guarantees;
- effect set;
- resource law;
- Racket combinator body.

### 6.2 Template File Format

One template is one `.rkt` file under `templates/`. The first top-level form is `define-meta`.

Example:

```racket
(define-meta
  (name "recursive_document_summarize")
  (version "1.0.0")
  (task-kinds (summarise aggregate))
  (input-type "DocumentList")
  (output-type "Report")
  (slots
    (context_id (type context-ref) (required #t))
    (leaf_instruction (type string) (required #t))
    (compose_instruction (type string) (required #t))
    (model (type model-alias) (required #f) (default "quality_text_model"))
    (split_factor (type integer) (required #f) (default 4) (min 2) (max 10))
    (leaf_threshold_tokens (type integer) (required #f) (default 64000)))
  (guarantees (schema-valid))
  (effects (LLM))
  (resource-law
    (leaf-calls "b^ceil(log_b(ceil(n/tau)))")
    (critical-path "ceil(log_b(ceil(n/tau))) + 1")))
```

Every remaining top-level form is the body.

### 6.3 No Textual Substitution

Slots are data:

```racket
(slot 'leaf_instruction)
```

There are no `{{slot}}` markers. Slot content is never parsed as code.

### 6.4 Artifact Hash

Artifact hash includes:

- template name;
- template version;
- body hash;
- metadata hash;
- canonical slot values;
- semantic input/output types;
- declared guarantees;
- inferred effects.

`artifact_id = "art_" + artifact_hash[:16]`.

---

## 7. Planner

### 7.1 Inputs

Planner input:

- `context_id`;
- task description;
- hints;
- context metadata;
- requested output type;
- model registry;
- type registry;
- execution policy.

### 7.2 Task Kinds

Core task kinds:

- `direct`
- `search`
- `classify`
- `aggregate`
- `summarise`
- `pairwise`
- `multi_hop`
- `compare`
- `validate`
- `refine`
- `decompose`
- `generate`

Task kind selection is deterministic when hints are complete. If hints are incomplete, one bounded menu-selection LLM call is allowed before planning, and the result is recorded.

### 7.3 Plan Selection

The planner chooses:

- direct leaf call when input fits;
- recursive split-map-reduce for long-context summarise/aggregate;
- symbolic filter before leaf calls for search;
- symbolic cross product plus bounded classification for pairwise tasks;
- iterative validation plan for refine/validate;
- explicit chain when multiple semantic transformations are needed.

### 7.4 Chain Typing

For every adjacent pair:

```text
step_i.output_type <= step_{i+1}.input_type
```

Verification fails if no compatibility relation exists.

### 7.5 Strategy Alternatives

`plan_strategy` should return:

- recommended plan;
- alternatives with cost/latency/quality tradeoffs;
- type flow;
- estimated resource shape before full dry run when available.

---

## 8. Dry Run and Static Analysis

### 8.1 Dry Run Output

Dry run returns:

```python
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
    node_id: str | None
    model: str
    input_type: str
    output_type: str
    instruction_chars: int
    input_chars: int
    max_tokens: int | None
    json_mode: bool
    depth: int
```

The terminal runtime message in simulate mode includes both:

```json
{"stats": "...", "calls": ["..."]}
```

Python computes call graph and cost from `calls`.

### 8.2 Simulation Semantics

Simulation is deterministic:

- `leaf-call` records a `SimulatedCall` and returns a synthetic value of the declared output type;
- `split` creates deterministic partitions;
- `map` executes every item;
- `filter` uses symbolic predicates when possible, otherwise keeps all items as an upper bound;
- `reduce` executes all required composition steps;
- `race` starts all branches and chooses the first branch in source order;
- `iterate` runs max iterations;
- `memoized` counts repeated canonical calls once;
- `py-exec` is counted but not run.

### 8.3 Static Checks

Before dry run, the registry checks:

- template metadata is valid;
- body parses as S-expressions;
- body references only declared slots;
- body uses only allowed combinators and pure forms;
- inferred effects match declared effects;
- semantic types exist;
- resource-law syntax is valid.

S-expression parsing and body analysis **MUST NOT** use regular expressions.

---

## 9. Verification

Verification runs all checks and never short-circuits.

Exactly 25 checks:

| # | Name | Severity | Rule |
|---|---|---|---|
| 1 | `artifact_exists` | fail | artifact is stored |
| 2 | `artifact_hash` | fail | recomputed hash matches |
| 3 | `template_known` | fail | template name/version exists |
| 4 | `slot_schema` | fail | slots validate |
| 5 | `types_exist` | fail | input/output semantic types exist |
| 6 | `input_type_compatible` | fail | context value matches template input type |
| 7 | `chain_type_compatible` | fail | adjacent chain types compose |
| 8 | `guarantees_known` | fail | declared guarantees are known |
| 9 | `effects_allowed` | fail | required effects allowed by policy |
| 10 | `context_exists` | fail | context-ref slots resolve |
| 11 | `model_aliases_resolve` | fail | model aliases exist |
| 12 | `primitive_allowlist` | fail | body uses only allowed bindings |
| 13 | `termination_bound` | fail | recursive plan has finite decreasing bound |
| 14 | `call_count_limit` | fail | simulated calls <= policy |
| 15 | `critical_path_limit` | warn | critical path <= policy |
| 16 | `concurrency_limit` | fail | max concurrency <= policy |
| 17 | `token_budget` | fail | estimated tokens <= policy |
| 18 | `cost_budget` | fail | estimated high cost <= policy |
| 19 | `recursion_depth_limit` | fail | simulated depth <= policy |
| 20 | `dry_run_fresh` | fail | dry run artifact matches plan |
| 21 | `output_schema_valid` | fail | output schema/type schema is valid |
| 22 | `py_exec_policy` | fail | py-exec requires policy allow |
| 23 | `llm_generated_code_absent` | fail | no generated control code path exists |
| 24 | `checkpoint_writable` | warn | checkpoints namespace writable if used |
| 25 | `resource_law_respected` | fail | simulated stats do not exceed declared law |

Policy defaults:

```python
max_llm_calls = 500
max_concurrency = 50
max_critical_path = 25
max_tokens = 2_000_000
max_cost_usd = 10.00
max_recursion_depth = 5
allowed_effects = {"LLM", "CHECKPOINT", "PARTIAL_RESULT", "GATE"}
max_timeout_seconds = 3600
```

---

## 10. Runtime Protocol

JSON Lines over stdin/stdout.

### 10.1 Startup

```json
{"type":"ready","protocol":"1.0"}
```

### 10.2 Run

```json
{
  "type": "run",
  "mode": "simulate",
  "artifact_id": "art_a1b2c3d4e5f60718",
  "program": "(combinator-program ...)",
  "slot_values": {},
  "contexts": {"ctx_9f8e7d6c5b4a3210": []},
  "types": {},
  "limits": {"max_recursion_depth": 5}
}
```

### 10.3 Effects

Live effect requests:

```json
{"type":"llm_call","id":"call_001a2b3c4d5e6f70","node_id":"leaf","input_type":"Text","output_type":"FindingSet","instruction":"...","input":"...","model":"fast_text_model","temperature":0,"json":true,"max_tokens":null}
{"type":"py_exec","id":"call_111a2b3c4d5e6f70","code":"...","input":{},"allowed_imports":["json"],"timeout_seconds":30}
{"type":"checkpoint","id":"ckpt_77a8b9c0d1e2f300","node_id":"reduce","data":{}}
{"type":"gate","id":"gate_77a8b9c0d1e2f300","label":"review","payload":{}}
{"type":"partial_result","node_id":"map","index":7,"value":{}}
```

Python replies to `llm_call`, `py_exec`, `checkpoint`, and `gate`. `partial_result` is fire-and-forget.

### 10.4 Terminal

```json
{"type":"done","value":{},"stats":{},"calls":[]}
{"type":"error","error_code":"runtime_error","message":"...","trace":"..."}
```

---

## 11. MCP Tools

Exactly 10 tools:

| Tool | Purpose |
|---|---|
| `load_context` | Store input and metadata. |
| `plan_strategy` | Build typed combinator plan. |
| `dry_run_strategy` | Simulate plan and estimate resources. |
| `execute_strategy` | Verify and execute live. |
| `resume_execution` | Resume a live gate. |
| `get_execution_trace` | Fetch execution trace. |
| `list_templates` | List audited templates. |
| `describe_template` | Describe one template. |
| `get_record` | Fetch stored record by ID. |
| `reset` | Clear state by scope. |

Envelope:

```json
{"status":"ok", "...":"..."}
{"status":"error","error_code":"...","message":"..."}
{"status":"suspended","execution_id":"exec_...","gate":{}}
```

---

## 12. Identifiers and Store

ID grammar:

```text
^(ctx|plan|dry|exec|art|ver|call|ckpt|gate)_[0-9a-f]{16}$
```

Store namespaces:

- `contexts`
- `plans`
- `artifacts`
- `dry_runs`
- `verifications`
- `executions`
- `cache`
- `checkpoints`
- `traces`
- `types`

Gates are in-memory only and are not resumable after process death.

---

## 13. Python Interfaces

Core records:

```python
class ContextRecord(BaseModel):
    context_id: str
    data: Any
    data_shape: str
    semantic_type: str
    metadata: dict[str, Any]
    created_at: float

class SemanticType(BaseModel):
    name: str
    schema: dict[str, Any]
    extends: list[str] = []

class TemplateMeta(BaseModel):
    name: str
    version: str
    task_kinds: list[str]
    input_type: str
    output_type: str
    slots: dict[str, Any]
    guarantees: list[str] = []
    effects: list[str] = []
    resource_law: dict[str, str] = {}

class PlanRecord(BaseModel):
    plan_id: str
    context_id: str
    task: str
    input_type: str
    output_type: str
    program_ref: str
    type_flow: list[tuple[str, str]]
    planner_parameters: dict[str, Any]
    alternatives: list[dict[str, Any]] = []
    created_at: float

class ArtifactRecord(BaseModel):
    artifact_id: str
    plan_id: str
    program_hash: str
    metadata_hash: str
    slot_values: dict[str, Any]
    input_type: str
    output_type: str
    effects: list[str]
    guarantees: list[str]

class DryRunRecord(BaseModel):
    dry_run_id: str
    plan_id: str
    artifact_id: str
    stats: SimulationStats
    calls: list[SimulatedCall]
    estimate: CostEstimate
    call_graph: list[dict[str, Any]]
    closed_form_bounds: dict[str, Any]
    warnings: list[str] = []

class VerificationRecord(BaseModel):
    verification_id: str
    artifact_id: str
    dry_run_id: str
    decision: Literal["pass", "fail"]
    checks: list[dict[str, Any]]

class ExecutionRecord(BaseModel):
    execution_id: str
    plan_id: str
    artifact_id: str
    verification_id: str
    state: str
    result: Any | None = None
    stats: dict[str, Any] = {}
    created_at: float
    completed_at: float | None = None
```

Components:

```python
Store(root: Path)
TypeRegistry(config_path: Path)
ModelRegistry(config_path: Path)
ContextStore(store: Store, types: TypeRegistry)
CombinatorRegistry(runtime_dir: Path)
TemplateRegistry(template_dir: Path, types: TypeRegistry, models: ModelRegistry)
Planner(types: TypeRegistry, templates: TemplateRegistry, models: ModelRegistry)
CostAnalyzer(models: ModelRegistry)
VerificationEngine(store: Store, types: TypeRegistry, templates: TemplateRegistry)
RacketRuntime(runtime_dir: Path)
DryRunner(store: Store, runtime: RacketRuntime, cost: CostAnalyzer)
Executor(...)
```

All are built in `rlm_scheme/app.py::build_app(root)`.

---

## 14. Build Batches

### Batch 0: Foundations

Files:

- `rlm_scheme/ids.py`
- `rlm_scheme/models.py`
- `rlm_scheme/store.py`
- `config/models.json`
- `config/types.json`

Acceptance:

- ID grammar tests;
- store namespace tests;
- type registry loads core types;
- model registry loads required aliases.

### Batch 1: S-Expressions and Template Registry

Files:

- `rlm_scheme/sexpr.py`
- `rlm_scheme/template_store.py`
- `templates/*.rkt`

Acceptance:

- no regex for S-expression parsing or body analysis;
- metadata parse tests;
- type/effect/resource metadata validation;
- template body allowlist tests.

### Batch 2: Type System and Contexts

Files:

- `rlm_scheme/types.py`
- `rlm_scheme/context_store.py`

Acceptance:

- semantic type validation;
- subtype compatibility;
- context loading with semantic type;
- `DocumentList`, `RecordList`, `FindingSet`, and `Report` fixtures.

### Batch 3: Planner

Files:

- `rlm_scheme/planner.py`
- `rlm_scheme/cost.py`

Acceptance:

- direct plan when input fits;
- recursive plan when input exceeds window;
- split-factor bounded search;
- type-flow construction;
- alternatives include tradeoffs.

### Batch 4: Racket Runtime

Files:

- `runtime/main.rkt`
- `runtime/combinators.rkt`
- `runtime/sandbox.rkt`
- `runtime/wire.rkt`
- `rlm_scheme/runtime.py`

Acceptance:

- handshake;
- combinator evaluation;
- recursive executor terminates under bounds;
- sandbox escape tests fail;
- simulate mode returns `stats` and `calls`.

### Batch 5: Providers, Cache, and Effects

Files:

- `rlm_scheme/llm_provider.py`
- `rlm_scheme/cache.py`
- `rlm_scheme/budget.py`
- `rlm_scheme/python_bridge.py`
- `rlm_scheme/trace.py`

Acceptance:

- deterministic mock provider;
- cache key tests;
- budget rejection;
- py-exec process isolation;
- trace records effects.

### Batch 6: Dry Run and Verification

Files:

- `rlm_scheme/dry_run.py`
- `rlm_scheme/verification.py`

Acceptance:

- dry-run cost from simulated calls;
- closed-form bound checks;
- all 25 verification checks;
- no live call before pass verification.

### Batch 7: Executor and Chains

Files:

- `rlm_scheme/executor.py`
- `rlm_scheme/chain.py`
- `rlm_scheme/gate.py`
- `rlm_scheme/checkpoint.py`

Acceptance:

- execute verified artifact;
- validate every intermediate type;
- gate suspend/resume;
- checkpoints persist;
- chain type compatibility.

### Batch 8: MCP Server

Files:

- `rlm_scheme/mcp_server.py`
- `rlm_scheme/app.py`

Acceptance:

- exactly 10 tools;
- response envelope;
- happy path through MCP;
- long-context recursive plan through MCP.

### Batch 9: Docs and CI

Files:

- `README.md`
- `examples/*`
- CI config

Acceptance:

- README explains lambda-RLM-inspired design;
- Racket requirement documented;
- py-exec limitation documented;
- full test suite green.

---

## 15. Definition of Done

The implementation is done when:

1. All batch tests pass.
2. Core semantic types load and validate fixtures.
3. A direct plan executes end to end.
4. A recursive long-context plan executes end to end.
5. Dry-run measured calls match live host-counted calls.
6. Closed-form bounds are recorded and enforced for recursive plans.
7. Chain type compatibility is enforced.
8. No live LLM call occurs before passing verification.
9. Racket sandbox escape tests fail.
10. No S-expression parsing or body analysis uses regex.
11. MCP exposes exactly 10 tools.
12. `SPEC-DEVIATIONS.md` is empty or reviewed.

