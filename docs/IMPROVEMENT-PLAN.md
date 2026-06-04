# RLM-Scheme Improvement Plan

This document defines the plan for turning RLM-Scheme from a prompt-driven
Scheme code generator into a structured orchestration system:

1. classify the task and data shape,
2. select a strategy from deterministic rules,
3. compile a typed strategy specification into Scheme,
4. dry-run and verify the structure,
5. execute in the Racket sandbox.

There is no requirement for backward compatibility. Compound combinators can be
removed from the runtime once equivalent primitive/template paths exist.

---

## North Star Architecture

The end state is a structured planning and execution pipeline, not an MCP
server that asks agents to pass raw Scheme strings around.

An agent should interact with the MCP server at the level of plans, templates,
strategy specs, compiled artifacts, verification records, and executions:

1. **Plan.** The agent calls `plan_strategy(...)` with the task and any known
   structured hints. The planner classifies TaskShape/DataShape, selects an
   appropriate template or emits a Strategy Spec, and returns a `plan_id`.

2. **Compile.** The agent calls `compile_strategy(...)` with either a Strategy
   Spec or a template invocation. The compiler validates inputs, fills typed
   slots, emits Scheme internally, stores a compiled artifact, and returns an
   `artifact_id`. Raw Scheme remains inspectable, but it is not the primary API
   contract.

3. **Estimate and dry-run.** The agent can call `estimate_strategy(artifact_id)`
   for a static cost/call estimate, then `dry_run_strategy(artifact_id)` to
   simulate structure without real LLM calls. The server stores the dry-run
   result and returns a `dry_run_id`.

4. **Lint and verify.** The agent calls `lint_strategy(artifact_id,
   dry_run_id, ...)` for deterministic no-token checks, then optionally
   `verify_strategy(artifact_id, dry_run_id, ...)` for a cheap semantic model
   check. Verification reuses the dry-run result and returns a
   `verification_id`.

5. **Execute.** The agent calls `execute_strategy(artifact_id,
   verification_id, timeout, plan_id)`. By default, execution refuses failed
   verification unless `force=true`.

6. **Inspect and learn.** The agent calls `get_execution_trace(execution_id)` or
   status/cancel tools as needed. History links `plan_id -> artifact_id ->
   dry_run_id -> verification_id -> execution_id`.

Templates are the bridge between high-level planner intent and executable
Scheme. A template generally stores:

- its name, TaskShape/DataShape fit, trigger conditions, and produced output
  shape,
- typed slots with defaults, enums, ranges, and descriptions,
- a primitive-only Scheme body or equivalent Strategy Spec fragment,
- expected call formulas and structural profiles for validation,
- constraints such as required model capabilities or JSON output requirements.

The planner reads template metadata and slot schemas; it should not edit the
Scheme body directly. The Python compiler owns Scheme generation. This makes
the LLM responsible for structured choices and content slots, while deterministic
code is responsible for syntax, primitive composition, and runtime safety.

Low-level escape hatches can remain for debugging:

- `execute_scheme(code, ...)` can execute raw Scheme directly.
- `dry_run_scheme(code, ...)` can simulate raw Scheme directly.
- reference tools can expose runtime docs.

Those are not the normal path. The normal path is artifact-based and auditable.

---

## 1. Problem Statement

The current planner gives an LLM all 17 combinators in a long prompt and asks
it to be creative. That is the wrong interface for reliable code generation.
LLMs perform best when structural choices are constrained and mechanical, not
when they must search a large composition space and produce valid nested
Scheme at the same time.

There are five structural problems:

1. **Flat combinator list with no decision structure.** All 17 combinators are
   presented as if equally relevant. For most tasks, only a few are sensible.

2. **`data_characteristics` is free text.** The choice of strategy depends on
   data shape: independent list, ordered stream, singular blob, table,
   multimodal record, graph, and so on. The system currently asks the LLM to
   infer those properties from prose.

3. **No feedback loop.** Planning starts from zero every time. The runtime
   records execution statistics, but the planner does not learn from prior
   successful strategies.

4. **No structural verification.** A bad strategy costs real model calls. The
   user cannot cheaply inspect call counts, fan-out, recursive depth, model
   mix, or likely latency before execution.

5. **Decomposition is conflated with code generation.** The LLM must decide
   how to decompose the work and also write correct Scheme. Those are separate
   tasks with different failure modes.

The target architecture separates these concerns:

- **LLM-facing layer:** structured JSON/YAML strategy specs and slot values.
- **Planner layer:** deterministic TaskShape/DataShape decision rules.
- **Compiler layer:** typed template/spec compiler that emits Scheme.
- **Runtime layer:** a small Racket primitive set plus sandboxed execution.
- **Verification layer:** dry-run and linting before real model calls.

---

## 2. Analysis

### 2.1 Combinator Orthogonality

Combinator libraries work best when the primitive set is small, orthogonal, and
composable. The current library has useful names, but too many are compound
patterns masquerading as primitives.

The runtime should be reduced to a minimal primitive basis. Templates and the
strategy compiler can still expose convenient patterns, but those patterns
should compile to primitives.

#### Primitive Runtime Basis

The retained runtime basis should be:

| Group | Primitive | Purpose |
|---|---|---|
| Parallel | `map-async` | Apply an async function to items with concurrency control. |
| Parallel | `parallel` | Run thunks concurrently and return their results. |
| Parallel | `race` | Run async thunks and return the first completed result. |
| Reduction | `tree-reduce` | Hierarchical associative reduction. |
| Reduction | `fold-sequential` | Ordered accumulation. |
| Control | `sequence` | Function composition pipeline. |
| Control | `choose` | Conditional dispatch. |
| Control | `iterate-until` | Loop until predicate or max iterations. |
| Delegation | `recursive-spawn` | Delegate to a nested sandbox. |
| Modifier | `memoized` | Cache function results. |
| Modifier | `with-validation` | Validate output of a function. |
| Modifier | `try-fallback` | Recover from errors with fallback function. |

This is 9 primitives plus 3 modifiers. The exact count matters less than the
rule: every retained runtime combinator should be hard to express cleanly as a
short composition of the others.

#### Compound Combinators To Remove

Because backward compatibility is not required, compound combinators should be
removed from `racket_server.rkt` after templates/spec compilation cover their
uses.

| Remove | Compile to |
|---|---|
| `fan-out-aggregate` | `map-async` followed by `tree-reduce` or `fold-sequential`. |
| `critique-refine` | `iterate-until` with explicit generate, critique, refine state. |
| `ensemble` | `parallel` plus custom aggregation. |
| `vote` | `parallel` plus majority/plurality/consensus selection in Scheme or `py-exec`. |
| `tiered` | `map-async` cheap phase plus expensive synthesis or selective review. |
| `active-learning` | `map-async` cheap phase, `py-exec` uncertainty filter, `map-async` expensive phase. |
| `fold-summarizing` | Do not add as a primitive; compile to `fold-sequential` with explicit summarization calls. |

Templates must use only the primitive basis. The planner should never recommend
the removed compound names.

### 2.2 Runtime Concurrency Semantics

The current runtime names overpromise concurrency:

- `map-async` is genuinely concurrent.
- `race` is genuinely concurrent because it uses async handles and `await-any`.
- `parallel` currently calls thunks synchronously.
- `vote` and `ensemble` currently call thunks synchronously.
- `tiered` and `active-learning` currently use synchronous `map`.

This must be corrected before planner templates rely on these names.

Required runtime behavior:

- `parallel` must run thunks concurrently under `#:max-concurrent`.
- `vote` and `ensemble` should be removed as runtime compounds, not fixed.
- `tiered` and `active-learning` should be removed as runtime compounds, not
  fixed.
- templates that need these patterns should use `map-async`, `parallel`,
  `py-exec`, and `tree-reduce` explicitly.

### 2.3 Composition Depth

Theoretical composition is unbounded, but practical orchestration should stay
shallow. Each nesting level adds critical-path latency, debugging complexity,
and more places where partial failures can propagate.

Recommended rule:

- Level 0: direct `llm-query`.
- Level 1: one primitive, e.g. `map-async` or `iterate-until`.
- Level 2: primitive plus reducer/modifier.
- Level 3: two primitives plus modifier.
- Beyond level 3: prefer `recursive-spawn` or a separate compiled phase.

Dry-run can report recursive sandbox depth immediately. True combinator nesting
depth requires Racket-side instrumentation and is not part of the first dry-run
phase.

### 2.4 Dead Code And Inconsistencies

These should be fixed early because they distort the planner interface:

1. **`recursive-spawn #:depth` is dead.** The keyword is accepted in Racket but
   not used. Actual enforcement is Python-side `MAX_RECURSION_DEPTH = 3`.
   Since compatibility is not required, remove the keyword from the public API
   or wire it through correctly. Prefer removing it and documenting the global
   recursive depth limit.

2. **Duplicate stale MCP decorator on `execute_scheme`.** There is an extra
   `@mcp.tool(description="Analyze task and identify clarifying questions...")`
   stacked above the real `execute_scheme` decorator. Remove it.

3. **Async parameter extraction should be audited.** Confirm `temperature`,
   `max_tokens`, `json_mode`, and `images` are passed through consistently for
   `llm-query-async`. Remove any unused `llm_kwargs` remnants.

4. **Docs mention deprecated model names.** Replace stale names such as
   `curie`, `gpt-3.5`, and repeated `gpt-4` entries with current supported
   model names used by the planner/templates.

5. **Type docs must match runtime behavior.** In particular, document which
   primitives await internally and which return async handles or plain values.

### 2.5 Literature References

The plan is informed by a long pattern in combinator library design:

- Schönfinkel (1924), "On the building blocks of mathematical logic" — minimal
  combinator bases.
- Hughes (1989), "Why Functional Programming Matters" — composition as glue.
- Hutton & Meijer (1996), "Monadic Parser Combinators" — small primitive sets
  compose into rich grammars.
- Dean & Ghemawat (2004), "MapReduce" — map/reduce as a practical distributed
  computation basis.
- Apache Beam programming model — a small number of transforms plus modifiers.
- DSPy — a small set of prompting modules plus optimizers.
- AGORA (2025), "From Agent Loops to Structured Graphs" — simpler structured
  orchestration often beats complex agent loops.

The engineering conclusion is conservative: keep the runtime primitive basis
small, compile higher-level patterns into it, and verify structure before
execution.

---

## 3. Taxonomy

### 3.1 TaskShape

TaskShape describes the operation the user wants performed.

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
| `Decompose` | Break one input into structured parts. | `llm-query` JSON, `py-exec`, or `recursive-spawn`. |
| `Validate` | Produce pass/fail/score assessments. | `map-async`, validation, aggregation. |
| `Aggregate` | Extract metrics and compute report. | `map-async` plus `py-exec`. |
| `Composite` | Multi-phase task. | compiled `sequence` of phase specs. |

#### TaskShape Decision Tree

Use deterministic rules whenever structured fields are available. Use an LLM
only to fill missing fields, not to choose the final shape directly.

```
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

`plan_strategy` should accept structured fields:

- `item_count: int | None`
- `independent: bool | None`
- `output_type: "one" | "list" | None`
- `operation: str | None`
- `has_second_phase: bool | None`
- `sub_operations: list[str] | None`

If `has_second_phase` is true, classification must preserve constituent shapes,
not just return `Composite`.

### 3.2 DataShape

DataShape describes how data is structured and drives parameters.

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

#### DataShape Mapping Rules

```
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
  -> map-async row extraction + py-exec aggregation

Multimodal
  -> require model with image/audio support; include image token estimates

Paired
  -> zip pairs and map-async over pair records

KeyValue
  -> preserve keys in results; aggregate by key
```

### 3.3 Per-Shape Decision Trees

The planner should not map each shape to a single combinator. It should select
from primitives based on output shape, order sensitivity, scale, and quality
constraints.

#### Direct

```
Q1: Does the input fit in one model context?
    YES -> single llm-query
    NO  -> reclassify as Decompose, Batch, or Synthesize

Q2: Is deterministic computation needed before/after the call?
    YES -> py-exec + llm-query or llm-query + py-exec
    NO  -> llm-query only
```

#### Batch

```
Q1: Return a list or one combined output?
    LIST     -> map-async
    COMBINED -> map-async then reduce

Q2: If combined, is combination order-sensitive?
    YES -> fold-sequential
    NO  -> tree-reduce

Q3: Are some items harder or more ambiguous?
    YES -> map-async cheap pass, py-exec uncertainty filter,
           map-async expensive pass for uncertain items
    NO  -> one map-async pass

Q4: Are duplicates likely?
    YES -> wrap map function with memoized
```

Templates must spell this out with primitives. They must not use
`fan-out-aggregate`, `tiered`, or `active-learning`.

#### Synthesize

```
Q1: Do all items fit in one context?
    YES -> Direct synthesis call
    NO  -> Q2

Q2: Is order important?
    YES -> fold-sequential
    NO  -> tree-reduce

Q3: Is accumulator likely to exceed context?
    YES -> compile explicit summarization step inside fold-sequential
    NO  -> exact fold-sequential
```

#### Search

```
Q1: Is the candidate set finite?
    YES -> parallel candidate evaluations + selection
    NO  -> iterate-until

Q2: Is latency more important than quality?
    YES -> race
    NO  -> parallel all candidates, then compare
```

#### Refine

```
Q1: Is there a testable predicate?
    YES -> iterate-until with deterministic/LLM predicate
    NO  -> bounded iterate-until with critique/refine state

Q2: Should each iteration be validated?
    YES -> with-validation around refine step
```

Templates must compile critique/refine to `iterate-until`; they must not use a
runtime `critique-refine` combinator.

#### Compare

```
Q1: Compare models or strategies?
    MODELS     -> parallel thunks with model-specific llm-query
    STRATEGIES -> parallel strategy thunks

Q2: Select one or synthesize all?
    SELECT     -> py-exec or Scheme majority/plurality logic
    SYNTHESIZE -> llm-query aggregator over parallel results
```

Templates must not use `vote` or `ensemble`.

#### Classify

```
Q1: One item or many?
    ONE  -> Direct
    MANY -> map-async

Q2: Need distribution/report?
    YES -> py-exec aggregation after labels
    NO  -> return labels

Q3: Ambiguous categories?
    YES -> cheap map-async, uncertainty filter, expensive map-async
```

#### Pipeline

```
Q1: Are stages distinct?
    YES -> sequence
    NO  -> reclassify as Batch

Q2: Can a stage fail?
    YES -> try-fallback around that stage

Q3: Does a stage need quality gating?
    YES -> with-validation around that stage
```

#### Generate

```
Q1: Fixed number or until condition?
    FIXED -> map-async over generated index list
    UNTIL -> iterate-until

Q2: Must items be mutually consistent?
    YES -> fold-sequential, each item sees prior outputs
    NO  -> map-async

Q3: Must items be unique?
    YES -> py-exec deduplication and regenerate missing count
```

#### Decompose

```
Q1: Known structural boundary?
    YES -> py-exec splitter
    NO  -> llm-query #:json #t to identify parts

Q2: Is one pass enough?
    YES -> parse parts and return
    NO  -> recursive-spawn

Q3: Process parts afterward?
    YES -> Composite: Decompose -> Batch
```

#### Validate

```
Q1: Same rubric for all items?
    YES -> map-async validation
    NO  -> fold-sequential if criteria evolve

Q2: Need structured assessment?
    YES -> #:json #t plus with-validation for schema

Q3: Which error is costlier?
    FALSE POSITIVE -> expensive review of passes
    FALSE NEGATIVE -> expensive review of failures
```

#### Aggregate

```
Q1: Pure computation after extraction?
    YES -> map-async extraction + py-exec aggregation
    NO  -> map-async extraction + py-exec stats + llm-query interpretation

Q2: Grouped report?
    YES -> py-exec groupby using extracted schema
```

#### Composite

```
Q1: Identify constituent shapes in order.
Q2: Compile each phase independently.
Q3: Connect dependent phases with sequence.
Q4: Connect independent phases with parallel.
```

Common composites:

- Batch -> Synthesize
- Decompose -> Batch
- Batch -> Aggregate
- Generate -> Validate
- Classify -> Aggregate
- Batch -> Refine

### 3.4 Two-Level Selection Model

Combinator selection has two levels:

1. **Structural selection:** TaskShape + DataShape select primitives and
   structural parameters. This is deterministic.

2. **Content selection:** prompts, model choices, rubric wording, uncertainty
   thresholds, and fallback behavior. The LLM can help here, but it should
   fill structured slots rather than write Scheme.

---

## 4. Improvements

### 4.1 Docs And Runtime Consistency

**Goal:** Make public docs match the runtime and remove confusing dead API
surface before building new planner behavior.

**Changes:**

- Fix `docs/api-reference.md` model names.
- Add precise type signatures to `docs/combinators.md`.
- Document which primitives await internally.
- Remove or wire `recursive-spawn #:depth`; prefer removal.
- Remove the stale duplicate `@mcp.tool` decorator above `execute_scheme`.
- Audit async parameter pass-through in `mcp_server.py`.
- Add `Direct` to the taxonomy and docs.

**Type signature vocabulary:**

```
SyntaxObject = wrapped LLM response from llm-query
AsyncHandle  = handle from llm-query-async
String       = plain unwrapped string
[A]          = list of A
Fn<A,B>      = function from A to B
Thunk<A>     = zero-arg function returning A
```

**Primitive signatures after cleanup:**

```
map-async       : (Fn<Item, AsyncHandle>, [Item], #:max-concurrent Int) -> [String]
parallel        : ([Thunk<A>], #:max-concurrent Int) -> [A]
race            : [Thunk<AsyncHandle>] -> String
tree-reduce     : (Fn<Item..., Item>, [Item], #:branch-factor Int) -> Item
fold-sequential : (Fn<Acc, Item, Acc>, Acc, [Item]) -> Acc
sequence        : (Fn<A,B>, Fn<B,C>, ...) -> Fn<A, ...>
choose          : (Fn<A,Bool>, Fn<A,B>, Fn<A,B>) -> Fn<A,B>
iterate-until   : (Fn<A,A>, Fn<A,Bool>, A, #:max-iter Int) -> A
recursive-spawn : (Thunk<String>) -> Fn<Item, SyntaxObject>
memoized        : Fn<A,B> -> Fn<A,B>
with-validation : (Fn<A,B>, Fn<B,Bool>) -> Fn<A,B>
try-fallback    : (Fn<A,B>, Fn<A,B>) -> Fn<A,B>
```

**Tests:**

- Existing combinator tests updated for removed compound combinators.
- Tool surface test verifies no stale duplicate tool registration.
- Docs test checks removed compound names are not presented as runtime
  primitives.

### 4.2 Runtime Primitive Cleanup And Concurrency

**Goal:** Make the runtime primitive set small and make concurrency semantics
honest.

**Changes to `racket_server.rkt`:**

- Remove runtime definitions for:
  - `fan-out-aggregate`
  - `critique-refine`
  - `ensemble`
  - `vote`
  - `tiered`
  - `active-learning`
  - any proposed `fold-summarizing`
- Remove these names from `scaffold-names`.
- Reimplement `parallel` so it actually runs thunks concurrently.

**Design for concurrent `parallel`:**

The cleanest semantics are:

- A strategy thunk may return an `AsyncHandle`, in which case `parallel` awaits
  it.
- A strategy thunk may return a plain value, in which case `parallel` returns
  it directly.
- For actual LLM concurrency, templates should use thunks that call
  `llm-query-async`.

If mixed return types become awkward, restrict `parallel` to async thunks and
make that explicit:

```
parallel : [Thunk<AsyncHandle>] -> [String]
```

Prefer the restricted form for simplicity. It aligns with the core use case and
avoids pretending synchronous `llm-query` can be made concurrent after the
fact.

**Tests:**

- `parallel` with async thunks returns all results.
- `parallel #:max-concurrent 2` preserves result order.
- Removed compound names are unavailable in the sandbox.
- Templates compile without using removed names.

### 4.3 Dry Run Mode

**Goal:** Simulate Scheme orchestration without real LLM calls and return
structural metrics: total calls, sync calls, async calls, max fan-out,
recursive depth, models used, and rough cost/latency estimates.

Estimation and dry-run should exist at two levels:

- `estimate_strategy(artifact_id)` is a static estimate from the stored
  template/spec, expected item count, and call formulas. It does not execute
  Scheme.

- `dry_run_strategy(artifact_id)` is the normal artifact-based API. It retrieves
  compiled Scheme from the artifact store, runs the dry-run, stores the result,
  and returns a `dry_run_id`.
- `dry_run_scheme(code, ...)` remains a low-level debugging escape hatch for
  raw Scheme.

**Important design constraints:**

- Dry-run must not mutate real call accounting.
- Dry-run must not set execution mode on the shared backend singleton.
- Dry-run must preserve `await-any` rolling-window behavior.
- `recursive_depth` is not combinator nesting depth.

**Python design:**

Pass execution context into `RacketREPL.send()`:

```python
class ExecutionMode(Enum):
    REAL = "real"
    DRY_RUN = "dry_run"

class DryRunContext:
    def __init__(self):
        self.pending = {}          # pending_id -> mock_result
        self.calls = []
        self.sync_calls = 0
        self.async_calls = 0
        self.current_pending = 0
        self.max_pending = 0
        self.models = {}

    def record_sync(self, model, instruction, recursive_depth):
        ...

    def record_async(self, pending_id, model, instruction, recursive_depth, result):
        self.pending[pending_id] = result
        self.current_pending += 1
        self.max_pending = max(self.max_pending, self.current_pending)
        ...

    def consume(self, pending_id):
        result = self.pending.pop(pending_id)
        self.current_pending = max(0, self.current_pending - 1)
        return result
```

`send()` accepts:

```python
def send(self, cmd: dict, timeout: float = 300,
         mode: ExecutionMode = ExecutionMode.REAL,
         dry_run: DryRunContext | None = None) -> dict:
```

Do not store `_execution_mode` or `_dry_run_ctx` on the backend instance.

**Dry-run callback behavior:**

- `llm-query`: return deterministic mock response immediately.
- `llm-query-async`: record mock result in `DryRunContext.pending`; no real
  future required.
- `await`: consume exactly that pending id.
- `await-batch`: consume all requested ids in order.
- `await-any`: choose exactly one requested pending id deterministically
  (first id is fine), consume it, return all other requested ids as
  `remaining_ids`, and leave them pending.

This special-casing fixes the bug where pre-resolved futures cause
`wait(FIRST_COMPLETED)` to return all futures and break rolling-window
`map-async`.

**Output field names:**

Use `recursive_depth`, not `max_depth`, because it comes from
`self._current_depth`. True combinator nesting depth needs separate
Racket-side instrumentation and is not included in this phase.

**Cost estimate:**

Dry-run can estimate per-call model count. Token cost requires prompt/data size
estimation and should be marked approximate.

**Tests:**

- No OpenAI client calls during dry-run.
- No `_call_registry` mutation during dry-run.
- `map-async` with `items > max-concurrent` reports max fan-out equal to
  `max-concurrent`, not item count.
- `await-any` leaves remaining ids pending.
- Dry-run can be run concurrently with a real execution without mode leakage.
- `execute_scheme` results are unchanged after a dry-run.
- `estimate_strategy(artifact_id)` returns a cheap static estimate without
  invoking Racket.
- `dry_run_strategy(artifact_id)` stores and returns a reusable `dry_run_id`.

### 4.4 Self-Verification Tool

**Goal:** Catch structural and semantic errors before real execution.

Verification is split into two layers:

- `lint_strategy` is deterministic and token-free. It operates on compiled
  artifacts and dry-run results.
- `verify_strategy` optionally uses a cheap model for semantic checks after
  linting.

Both operate on compiled artifacts. Both should reuse an existing dry-run result
when a `dry_run_id` is provided instead of rerunning dry-run.

```python
async def lint_strategy(
    artifact_id: str,
    task_description: str,
    dry_run_id: str | None = None,
    expected_items: int | None = None,
    ctx: Context = None,
) -> str:
    ...
```

```python
async def verify_strategy(
    artifact_id: str,
    task_description: str,
    dry_run_id: str | None = None,
    expected_items: int | None = None,
    ctx: Context = None,
) -> str:
    ...
```

The tool stores its result and returns a `verification_id`. Raw-code
verification can remain available as a debugging helper, but the normal path is
artifact-based.

**Deterministic lints:**

- zero LLM calls for a non-Direct strategy,
- async calls much lower than expected item count,
- max fan-out above configured safe limit,
- too many sync calls in a supposedly parallel strategy,
- use of removed compound combinator names,
- JSON mode without "json" in instruction,
- `llm-query` result used in string operations without `syntax-e` where
  detectable,
- Direct shape wrongly using orchestration.

**Semantic check:**

Use a cheap model only after deterministic checks pass or produce warnings.
The semantic check should answer structured questions:

- does the output shape match the task?
- are all items processed?
- are failure modes handled?
- does the model mix match quality/cost constraints?

**Tests:**

- fails removed compound names,
- warns on undercounted item processing,
- accepts direct single-call plans,
- reports dry-run structure in verification output.
- reuses `dry_run_id` without rerunning dry-run.
- returns deterministic lint results without calling an LLM.
- returns `verification_id`.

### 4.5 Machine-Readable Templates

**Goal:** Stop asking the LLM to write Scheme for common strategies. The LLM
should fill typed slots or emit a strategy spec. Python compiles to Scheme.

Templates are JSON manifests in `docs/templates/*.json`.

Each template contains:

- `name`
- `shape`
- `trigger`
- `produces`
- `slots` with types, defaults, ranges, enums
- `code` using primitive combinators only
- `expected_calls_formula`
- `structural_profile`
- optional `spec_equivalent` for future Strategy Spec compiler alignment

Templates are consumed by the planner and compiler:

- the planner reads metadata, triggers, slot schemas, output shape, and expected
  structural profile;
- the compiler reads the primitive-only Scheme body or `spec_equivalent`;
- agents and LLMs fill `slot_values`, not Scheme code.

**Rules:**

- Templates must not reference removed compound combinators.
- Template filling must validate types and ranges.
- Scheme strings must be escaped with `json.dumps()`.
- Template output should prefer `template_name + slot_values`; raw
  `code_template` is fallback only.

**Tree-reduce formula:**

For `N` mapped items and branch factor `B`, reduce calls are:

```
level = N
reduce_calls = 0
while level > 1:
    level = ceil(level / B)
    reduce_calls += level
total_calls = N + reduce_calls
```

Examples:

- `N=10, B=5`: `10 + 2 + 1 = 13`
- `N=100, B=5`: `100 + 20 + 4 + 1 = 125`
- `N=500, B=10`: `500 + 50 + 5 + 1 = 556`

**Initial templates:**

| File | Shape | Primitive composition |
|---|---|---|
| `direct-single-call.json` | Direct | `llm-query` |
| `batch-extract-only.json` | Batch | `map-async` |
| `batch-extract-synthesize.json` | Batch -> Synthesize | `map-async` + `tree-reduce` |
| `batch-extract-ordered-synthesis.json` | Batch -> Synthesize | `map-async` + `fold-sequential` |
| `refine-iterate.json` | Refine | `iterate-until` |
| `compare-parallel-select.json` | Compare | `parallel` + `py-exec`/Scheme selection |
| `classify-aggregate.json` | Classify -> Aggregate | `map-async` + `py-exec` |
| `generate-n-items.json` | Generate | `map-async` over index list |
| `validate-all.json` | Validate | `map-async` + `with-validation` |
| `decompose-and-process.json` | Composite | `llm-query #:json` + `py-exec` + `map-async` |

**Tests:**

- every template validates,
- every template compiles,
- no template contains removed compound names,
- expected call formulas match dry-run for fixed sample inputs,
- string slots escape quotes/newlines safely.
- template invocations compile to stored artifacts with `artifact_id`.

### 4.6 Task Classification And Progressive Disclosure

**Goal:** Replace the monolithic creative planner prompt with deterministic
classification plus shape-specific prompts.

**Classification design:**

```python
def plan_strategy(
    task_description: str,
    data_characteristics: str | None = None,
    constraints: str | None = None,
    priority: str = "balanced",
    scale: str = "medium",
    min_outputs: int | None = None,
    coverage_target: str | None = None,
    task_shape: str | None = None,
    item_count: int | None = None,
    independent: bool | None = None,
    output_type: str | None = None,
    operation: str | None = None,
    has_second_phase: bool | None = None,
    sub_operations: list[str] | None = None,
) -> str:
    ...
```

Process:

1. Use caller-provided structured fields.
2. Use a cheap model only to fill missing fields.
3. Run deterministic classification.
4. Select a shape-specific prompt.
5. Return `plan_id`, shape metadata, and preferably a template invocation or
   Strategy Spec.

**Composite preservation:**

If `has_second_phase` is true or `sub_operations` are provided, the result must
include constituent shapes:

```json
{
  "shape": "composite",
  "sub_shapes": ["batch", "synthesize"],
  "reasoning": "extract from documents, then synthesize"
}
```

**Forward compatibility:**

Shape prompts should ask for:

- `template_name`,
- `slot_values`,
- optional `strategy_spec`,
- raw `code_template` only as fallback.

This aligns with the Strategy Spec compiler end state.

The planner should include `clarifying_questions` instead of guessing when
classification fields are missing or contradictory enough to affect structural
choice.

**Tests:**

- structured fields classify without LLM call,
- missing fields call model once,
- Direct is detected before Synthesize,
- Composite preserves sub-shapes,
- `task_shape` override skips classification,
- low-confidence/unknown falls back safely.
- missing critical fields can produce clarifying questions.

### 4.7 Strategy Replay Database

**Goal:** Record execution outcomes and feed relevant successful prior
strategies into future planning.

Use explicit IDs, not global mutable "last plan" state.

`plan_strategy` returns:

```json
{
  "_meta": {
    "plan_id": "uuid",
    "task_shape": "batch",
    "sub_shapes": ["batch", "synthesize"]
  }
}
```

`execute_scheme` keeps `timeout` and adds `plan_id`:

```python
async def execute_scheme(
    code: str,
    timeout: int | None = None,
    plan_id: str | None = None,
    ctx: Context = None,
) -> str:
    ...
```

History file:

`~/.rlm-scheme/strategy-history.jsonl`

Entry shape:

```json
{
  "timestamp": "2026-06-03T10:00:00Z",
  "plan_id": "...",
  "task_shape": "batch",
  "sub_shapes": ["batch", "synthesize"],
  "strategy_name": "batch-extract-synthesize",
  "template_used": "batch-extract-synthesize",
  "code_hash": "a1b2c3d4",
  "outcome": "success",
  "metrics": {
    "total_calls": 125,
    "elapsed_seconds": 272,
    "total_tokens": 511000,
    "models": {"gpt-4o-mini": 100, "gpt-4o": 25}
  },
  "scale": "large",
  "item_count": 100
}
```

Only include short history summaries in prompts. Do not inject full prior code
unless explicitly requested.

History should link the full artifact chain:

```
plan_id -> artifact_id -> dry_run_id -> verification_id -> execution_id
```

This prevents accidental execution of stale code and makes post-run analysis
auditable.

**Tests:**

- history links to `plan_id`,
- execution without `plan_id` records unknown context,
- history rotation keeps file bounded,
- relevant history filters by shape/sub-shape.
- history records artifact, dry-run, verification, and execution IDs when
  present.

### 4.8 Error Semantics And Checkpoint-Aware Retry

**Goal:** Define how failures propagate and make recovery declarative in
compiled templates.

This should be template/compiler behavior first, not new runtime compounds.

Required semantics:

- `map-async` item failure returns structured error record when wrapped in
  `try-fallback`.
- Batch templates can checkpoint partial results every N completed items.
- Retry policy is explicit in the template/spec:
  - retryable: rate limit, timeout, transient API error,
  - non-retryable: validation failure, bad JSON after max retries.
- Reduce phase should know whether to include, skip, or summarize failures.

Spec fragment:

```yaml
retry:
  max_attempts: 3
  retry_on: [rate_limit, timeout]
checkpoint:
  every_items: 50
  key_prefix: extract
on_item_error: include_error_record
```

**Tests:**

- generated code checkpoints at configured intervals,
- retry wrapper handles simulated item failures,
- reduce phase receives explicit error records.

### 4.9 Explicit Summarization In Folds

**Goal:** Address memory pressure in long ordered folds without silently
changing `fold-sequential`.

Do not add `fold-summarizing` as a runtime primitive. Compile explicit
summarization into the fold body when the template/spec requests it.

Compiled shape:

```scheme
(fold-sequential
  (lambda (acc item)
    (define next (step acc item))
    (if (> (string-length next) (* horizon 4))
        (syntax-e (llm-query
          #:instruction summary-instruction
          #:data next
          #:model summary-model))
        next))
  ""
  items)
```

**Documentation must state:**

- this is lossy,
- accumulator must be string-like,
- cost increases by each summary call,
- exact accumulation requires plain `fold-sequential`.

**Tests:**

- compiled fold includes explicit summary call,
- no runtime `fold-summarizing` binding exists,
- dry-run reports summary calls when threshold is crossed in test data.

### 4.10 Model Routing

**Goal:** Centralize model choice without hiding it from execution traces.

Model routing belongs in the strategy spec/compiler layer first. Avoid adding
implicit runtime magic until there is a clear need.

Spec example:

```yaml
models:
  extract: gpt-4.1-nano
  synthesize: gpt-4o-mini
  critique: gpt-4o
```

Compiled Scheme should still include explicit `#:model` arguments so dry-run
and traces remain transparent.

Runtime `model-router` can be considered later, but it is not required for the
first implementation.

### 4.11 Reasoning Annotations

**Goal:** Improve auditability by recording why a decomposition was chosen.

Add a small primitive or host callback:

```scheme
(with-reasoning "Split by chapter because chapters are independent"
  body)
```

or compile to:

```scheme
(record-reasoning "Split by chapter because chapters are independent")
body
```

The annotation should appear in `get_execution_trace()` alongside call
structure. It must not affect control flow.

**Tests:**

- reasoning appears in scope log,
- nested reasoning preserves order,
- no user data is accidentally logged beyond preview limits.

### 4.12 Strategy Spec Compiler

**Goal:** Make structured strategy specs the primary LLM-facing interface. The
LLM should not write combinator compositions from scratch.

Input:

```yaml
phases:
  - name: extract
    type: parallel_map
    input: context
    model: gpt-4o-mini
    instruction: "Extract key points from this document as JSON."
    concurrency: 20
  - name: synthesize
    type: hierarchical_reduce
    input: extract.output
    model: gpt-4o
    instruction: "Combine these findings."
    branch_factor: 5
```

Compiler:

```python
def spec_to_scheme(spec: dict) -> str:
    """Validate and compile a Strategy Spec to executable Scheme."""
```

Supported phase types initially:

- `direct_call`
- `parallel_map`
- `hierarchical_reduce`
- `ordered_fold`
- `python_compute`
- `iterate`
- `parallel_compare`
- `recursive_decompose`

`compile_strategy` MCP tool:

```python
@mcp.tool(description="Compile a strategy specification to executable Scheme code.")
def compile_strategy(
    spec: str | None = None,
    template_name: str | None = None,
    slot_values: dict | None = None,
    plan_id: str | None = None,
) -> str:
    ...
```

Output:

```json
{
  "artifact_id": "...",
  "templates_used": ["batch-extract-synthesize"],
  "can_dry_run": true,
  "next_step": "Run dry_run_strategy, then verify_strategy, then execute_strategy."
}
```

Compiled artifacts are stored server-side:

```json
{
  "artifact_id": "...",
  "plan_id": "...",
  "source_type": "strategy_spec | template_invocation | raw_scheme",
  "strategy_spec": {},
  "template_name": "...",
  "slot_values": {},
  "scheme_code": "...",
  "created_at": "...",
  "code_hash": "..."
}
```

Additional artifact tools:

- `get_artifact(artifact_id)` returns metadata and optionally compiled Scheme.
- `estimate_strategy(artifact_id)` returns a static call/cost estimate from
  template/spec metadata without executing Scheme.
- `lint_strategy(artifact_id, dry_run_id, ...)` runs deterministic no-token
  checks.
- `execute_strategy(artifact_id, verification_id, timeout, plan_id,
  force=false)` runs verified artifacts. It refuses failed verification unless
  forced.
- `execute_scheme(code, ...)` remains a low-level escape hatch.

**Tests:**

- validates required fields,
- rejects unknown phase types,
- rejects removed compound combinator names,
- compiles each initial template equivalent,
- dry-run of compiled specs matches expected call profiles.
- stores compiled artifacts and returns stable `artifact_id`.
- static estimates are available before dry-run.
- lint results can be reused by verification.
- `execute_strategy` refuses failed verification unless `force=true`.

---

## 5. Implementation Roadmap

Phase numbers match improvement numbers.

| Phase | Improvement | Dependency | Main files |
|---|---|---|---|
| 4.1 | Docs and runtime consistency | none | docs, `mcp_server.py`, `racket_server.rkt` |
| 4.2 | Runtime primitive cleanup and concurrency | 4.1 | `racket_server.rkt`, tests |
| 4.3 | Dry run mode | 4.1 | `mcp_server.py`, tool docs, tests |
| 4.4 | Self-verification tool | 4.3 | `mcp_server.py`, tool docs, tests |
| 4.5 | Machine-readable templates | 4.1, 4.2 | `docs/templates`, compiler helpers |
| 4.6 | Task classification and progressive disclosure | 4.5 | `mcp_server.py`, shape prompts |
| 4.7 | Strategy replay database | 4.3, 4.6 | `mcp_server.py` |
| 4.8 | Error semantics and checkpoint-aware retry | 4.5 | templates/compiler |
| 4.9 | Explicit summarization in folds | 4.5 | templates/compiler |
| 4.10 | Model routing | 4.5 | strategy spec/compiler |
| 4.11 | Reasoning annotations | none | `racket_server.rkt`, trace docs |
| 4.12 | Strategy spec compiler and artifact store | 4.5, 4.6 | `mcp_server.py`, tests |

Recommended execution order:

1. 4.1 and 4.2 first, so the runtime surface is honest.
2. 4.3 next, because dry-run enables safe verification.
3. 4.5 before 4.6, because classification should select templates/specs.
4. 4.12 is the end-state interface; 4.4, 4.6, and 4.7 should be designed to
   accept specs and template slots, not raw Scheme, except as fallback.

---

## 6. Files Modified And Created

### Modified

| File | Changes |
|---|---|
| `docs/IMPROVEMENT-PLAN.md` | This rewritten plan. |
| `docs/api-reference.md` | Correct model names and primitive docs. |
| `docs/combinators.md` | Primitive-only type signatures and common errors. |
| `docs/planner-prompt.md` | Kept as fallback or rewritten to prefer specs/slots. |
| `mcp_server.py` | Dry-run, verification, classification, templates, history, compiler. |
| `racket_server.rkt` | Remove compounds, fix `parallel`, remove dead API, reasoning annotation. |

### Created

| File | Purpose |
|---|---|
| `docs/task-shapes.md` | TaskShape/DataShape taxonomy and decision rules. |
| `docs/shape-prompts/*.md` | Shape-specific planner prompts. |
| `docs/templates/*.json` | Typed primitive-only templates. |
| `docs/tool-descriptions/dry_run_scheme.md` | Dry-run tool docs. |
| `docs/tool-descriptions/estimate_strategy.md` | Static strategy estimate docs. |
| `docs/tool-descriptions/dry_run_strategy.md` | Artifact dry-run tool docs. |
| `docs/tool-descriptions/lint_strategy.md` | Deterministic lint docs. |
| `docs/tool-descriptions/verify_strategy.md` | Verification tool docs. |
| `docs/tool-descriptions/compile_strategy.md` | Strategy compiler tool docs. |
| `docs/tool-descriptions/execute_strategy.md` | Verified artifact execution docs. |
| `docs/tool-descriptions/get_artifact.md` | Compiled artifact inspection docs. |
| `tests/test_runtime_primitives.py` | Primitive and removed-compound behavior. |
| `tests/test_dry_run.py` | Dry-run behavior and isolation. |
| `tests/test_templates.py` | Template validation and compilation. |
| `tests/test_task_classification.py` | Deterministic classification and LLM field fill. |
| `tests/test_strategy_history.py` | History recording and lookup. |
| `tests/test_verify_strategy.py` | Verification tool. |
| `tests/test_spec_compiler.py` | Strategy Spec compiler. |
| `tests/test_artifacts.py` | Artifact storage, retrieval, and execution gating. |

---

## 7. Verification Checklist For This Plan

- The problem statement says five problems and lists five.
- There is no standalone section for prior-review responses.
- Critique fixes are incorporated into the relevant sections.
- Phase numbers in the roadmap match improvement numbers.
- Dead code items are listed in section 2.4.
- Runtime concurrency correction is section 4.2.
- Dry-run `await-any` is special-cased; no delayed-future scheduler is used.
- Dry-run uses per-call context passed into `send()`, not singleton backend mode.
- Dry-run reports `recursive_depth`, not false combinator nesting depth.
- Tree-reduce call formula is recursive and includes all reduce levels.
- Classification includes `has_second_phase` and `sub_operations`.
- `execute_scheme` retains `timeout` and adds `plan_id`.
- Templates use primitives only.
- Compound combinators are slated for runtime removal.
- Strategy Spec compiler is the end-state LLM-facing interface.
- Normal execution uses artifact IDs instead of passing raw Scheme between MCP
  tools.
