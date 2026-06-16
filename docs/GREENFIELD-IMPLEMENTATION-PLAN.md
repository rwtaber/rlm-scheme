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

## Orientation (Plain-Language, Non-Normative)

This section is a conceptual map for a reader with a general computer-science background and no LLM-specific experience. Sections 0–15 are the normative specification; this one only builds intuition and may be skipped by an implementer.

### The one-line idea

RLM-Scheme is **a query planner for LLM calls.** Instead of handing a giant prompt to a language model and hoping, it compiles a task into a typed, deterministic dataflow program, proves resource bounds on it, and only then runs it — invoking the model only at the leaves.

### The problem, in CS terms

A language model has a fixed **context window**: the maximum input it can read at once — a hard `MAX_INPUT` on a function `model : String -> String`. Two failure modes follow. First, data larger than the window is truncated. Second, and subtler, accuracy decays *before* the window is full — "**context rot**." The function is defined up to `W` tokens but only *reliable* up to some smaller `K`. `K`, not `W`, is the real budget.

The popular fix ("recursive language models") lets the model write and run its own control code in a REPL to chop the data up. That is `eval()` on stochastic output: it may not parse, may not terminate, may cost anything — properties no compiler author would accept.

### The core move

Keep the safe half, delete the dangerous half.

- **Safe half — "prompt as environment":** the data lives *outside* the model, in a store the host owns. The program *inspects* it through bounded reads (`peek`, `slice`, `split`) rather than swallowing it whole — an external data source behind a cursor, not an in-memory blob.
- **Deleted half — model-authored control flow.** Orchestration is a **fixed combinator program**: `map`, `filter`, `reduce`, `concat`, `cross`, `split`, and a recursion operator `fix`. The model appears as exactly one primitive:

  ```text
  M : BoundedPrompt -> TypedValue        -- the "leaf oracle"
  ```

So the system is **an interpreter whose only impure, nondeterministic primitive is the model**, and that primitive is only ever called on inputs small enough to be reliable. Everything else is total and deterministic.

### The lifecycle: EXPLAIN before you run

Like a database that shows a query plan and cost estimate before executing:

```text
load_context -> plan -> dry_run -> verify -> execute
```

- **plan** — a *deterministic* planner turns the task into a combinator AST. A strong model may make *one* bounded call to pick a strategy from a fixed menu, but it never writes the code.
- **dry_run** — the program runs in **simulate mode**: every `leaf-call` returns a synthetic typed value instead of hitting the model. This yields exact call counts, a cost range, recursion depth, and a wall-clock estimate with *zero* live calls. It is symbolic execution for resource accounting.
- **verify** — 24 static checks (termination, budgets, schema compatibility, "no model-generated code exists," …). **No live call is permitted until verification passes** — the type-checker, linter, and budget gate combined.
- **execute** — run the verified plan exactly once.

### Recursion as divide-and-conquer with a cost algebra

The recursive skeleton is the one you would write for merge sort:

```text
solve(x) = leaf_call(x)                              if size(x) <= tau
         = compose(map(solve, filter(split(x, k))))  otherwise
```

Because the shape is fixed, the plan has **closed-form bounds** — depth `ceil(log_k(n/tau))`, leaf calls `k^depth` — checked against simulation, like applying the master theorem and then verifying it empirically. Two parameters carry the insight: **`tau`** (leaf threshold) is sized to the reliable budget **`K`, not the window `W`**, keeping every leaf in the trustworthy regime; **`k`** (split factor) is chosen against the critical path, not a magic constant.

### The principle that does the heavy lifting

**Symbolic-first composition:** never ask the model to do what ordinary code can do. Joining results is `concat`, not a model call; all-pairs comparison is a `cross` product, not O(n²) model calls. When composition is symbolic, total model calls collapse to just the leaf count — the difference between an O(n) plan and an O(n log n) plan multiplied by a very expensive constant.

### Types as schema validation at boundaries

No grand ontology — two things only. Combinators carry type *shapes* (you cannot `reduce` a non-list), and every value crossing a boundary is validated against a small JSON-Schema subset. A malformed model output is caught *at the edge* and handled by an error policy (fail / skip / retry / escalate) before it corrupts downstream state. Structural typing, decided on shape; no nominal subtyping.

### Architecture and the multi-tier model

Two processes: a **Python host** (owns state, money, providers, persistence, the MCP tool API) and a **Racket subprocess** (the pure, sandboxed combinator evaluator, fed only bounded slices over JSON-Lines). The Racket side never sees the full data and never performs effects directly — it *requests* them and the host decides. Capability-style isolation.

Execution is explicitly **heterogeneous and multi-tier**:

- **Tier 0** — a frontier model (e.g. a Happy/Claude session) authors the plan through the tool API.
- **Tier 1** — the deterministic combinator engine; no model at all.
- **Tier 2** — cheap **local models** doing the grunt-work leaves.

The cost model is two-currency: Tier-0 calls cost *dollars* but are fast and parallel; Tier-2 calls on a single GPU cost ~nothing but are *serial*, so fan-out is a queue and **wall-clock**, not money, binds. The model registry encodes each model's `K`, throughput, concurrency, `device_group` (GPU residency, so swaps are penalized), and a `fallback_alias` so a weak local leaf that fails validation can **escalate** to the strong model rather than routing every leaf there.

### In one breath

Take the reliability and cost-predictability of a compiled, type-checked, resource-bounded dataflow program, and use a language model only as a sandboxed leaf primitive on inputs small enough to trust.

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
  ModelRegistry
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
ctx_... -> ContextRecord(data, metadata, schema)
```

Python remains the sole owner of context bytes. The Racket runtime never receives the full context. It reads bounded slices on demand through context-read primitives (`peek`, `slice`, `context-items`), which are issued as effect requests to Python (§10.3) and answered with bounded slices only. `split` then partitions already-materialized bounded values. Only bounded subprompts are sent to the LLM, and only bounded slices ever cross the pipe to Racket.

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

### 1.4 Orchestration Tiers

RLM-Scheme is built to run across heterogeneous models without embedding any of them as the orchestrator. Three tiers cooperate:

- **Tier 0 — caller (a coding agent over MCP, e.g. a Happy/Claude session).** A strong model authors intent: it calls `plan_strategy` (filling slot values such as `leaf_instruction` and `compose_instruction`), inspects dry-run estimates and traces, and decides whether to execute. It never writes control code; it parameterizes audited templates. This is the only place a frontier model is required.
- **Tier 1 — combinator engine (no model).** Deterministic split/filter/map/reduce/cross plus bounds. This is where orchestration actually happens.
- **Tier 2 — leaf models.** Bounded `leaf-call` oracles, which **[MAY]** be cheap models on constrained local hardware (e.g. a single 24 GB GPU). Multiple leaf models and an orchestrator-tier model can coexist; each is a `ModelRegistryEntry` (§13) with its own window `W`, reliability knee `K`, throughput, concurrency, device group, and fallback.

A model's tier is not hardcoded; it follows from how the planner assigns aliases to nodes. The same alias machinery lets the planner route grunt-work leaves to a local model while reserving an orchestrator-tier model for plan authoring, neural reduce that genuinely needs reasoning, and escalations (§3.3). The single-device economics that make Tier 2 behave very differently from a remote API are specified in §5.7.

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

The host stores input data and metadata as a context record. The Racket runtime never receives the full context; during a dry run or live execution it requests bounded slices through context-read effects (§10.3), and Python answers from the `ContextStore`.

### Phase 2: Task Detection

The planner classifies:

- task kind;
- structural data shape;
- requested output schema;
- whether the input fits the selected model's **reliable-input budget** `K` (not its raw context window `W`).

The planner reads structure cheaply through `peek` (§5.6) rather than ingesting the whole context, so task detection stays cheap even on inputs far larger than the window.

Task detection **[SHOULD]** be deterministic from hints and metadata. If hints are insufficient, the planner **[MAY]** use at most one bounded model call to choose from a fixed menu, and the call is recorded.

### Phase 3: Direct Dispatch

If the input fits the reliable-input budget `K` and the task does not require symbolic decomposition, the planner may choose a direct typed leaf call:

```text
direct_call : InputType -> OutputType
```

This still goes through dry run and verification.

### Phase 4: Recursive Planning

If the input does not fit, the planner constructs a recursive combinator program:

```text
solve(x):
  if size(x) <= leaf_threshold:
    return leaf_call(x)            # neural work only at bounded leaves
  else:
    parts = split(x, split_factor)
    kept = filter(parts)           # symbolic narrowing before any neural call
    partials = map(solve, kept)
    return compose(partials)       # symbolic composition by default
```

The planner chooses:

- `split_factor` (`k`): chunks per split. The planner picks `k` to keep the composition tree shallow enough to satisfy the critical-path policy while keeping leaf parallelism wide; it does not default to a single magic number (§5.5).
- `leaf_threshold_tokens` (`tau`): maximum prompt size for a leaf call, sized to the model's reliable-input budget `K`, not its raw window `W` (§5.5).
- `max_depth`: maximum recursion depth;
- `composition_operator` (`compose`): how partials are reduced. The planner **[MUST]** prefer a symbolic operator (concat, merge, sum) and use a neural reduce only when composition genuinely requires understanding (§4.4);
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
- wall-clock estimate (single-device-aware, §5.7);
- recursion depth.

No live LLM call occurs during dry run.

### Phase 6: Verification

Verification checks the artifact, boundary schemas, effects, resource bounds, model aliases, policy, and dry-run freshness. A live execution may start only if verification decision is `pass`.

### Phase 7: Single Live Execution

The runtime executes the prebuilt program once. There is no open-ended loop in which the LLM writes new control code. Recursive calls are internal to the combinator program and bounded by the planned depth.

---

## 3. Typed Boundaries

Typing in RLM-Scheme is structural, not a domain ontology. Following λ-RLM, "typed" means two things only:

1. **Parametric combinator typing.** Combinators carry type shapes (`Map : (A -> B) x List[A] -> List[B]`) so the planner can compose them without nonsense like reducing a non-list.
2. **Per-boundary schema validation.** Every leaf call output and every chain-step output is validated against a declared JSON-schema before it may flow downstream.

There is no required registry of named semantic types, no subtype lattice, and no declared-guarantee layer. Those were domain conveniences the paper does not need; they live in an optional domain pack (§3.4).

### 3.1 Output Schema Subset

A boundary schema is a JSON string containing a restricted JSON Schema object. Allowed keywords:

- `type`: one of `string`, `number`, `integer`, `boolean`, `object`, `array`, `null`;
- `properties`: object mapping property names to schemas;
- `required`: array of property-name strings;
- `items`: schema for array items;
- `enum`: array of scalar values;
- `additionalProperties`: boolean.

No other keywords are supported. A schema using an unsupported keyword fails `output_schema_valid`. The absence of a schema means the boundary is unconstrained (`Any`).

### 3.2 Boundary Compatibility

For a chain, adjacent steps compose when the producer's output schema is structurally compatible with the consumer's input schema:

- the consumer input schema is absent (`Any`); or
- the two schemas are structurally equal (same `type`, and for objects the producer's `required` properties cover the consumer's, and for arrays the `items` are compatible).

There is no name-based subtyping. Compatibility is decided on schema shape alone (`chain_type_compatible`).

### 3.3 Output Validation

Every leaf call and every chain step **[MUST]** validate its output against its declared output schema before the value can flow to the next step. A boundary with no schema is not validated.

Invalid output follows the active error policy:

- `fail_fast`: fail the execution;
- `skip_and_log`: record error and use `null`;
- `retry_then_skip`: retry provider policy, then skip;
- `retry_then_escalate`: retry on the same model, then re-issue the call on that model's `fallback_alias` (§13) — typically a stronger orchestrator-tier model — then skip if still invalid. Escalation calls are counted and recorded like any other leaf call; a model with no `fallback_alias` degrades this policy to `retry_then_skip`.

`retry_then_escalate` is the intended policy for cheap local leaf models: weak models fail schema validation far more often than frontier models, and a one-shot escalation recovers the leaf without sending every leaf to the expensive tier.

### 3.4 Optional Domain Packs

A deployment **[MAY]** ship a domain pack mapping named types (for example `TextDocument`, `Finding`, `Report`) to boundary schemas, purely as authoring shorthand: a slot or boundary may name a type and the loader expands it to the §3.1 schema. Core planning, dry run, verification, and execution **[MUST]** work with no domain pack present. Domain packs add no new verification checks and no new store namespace.

---

## 4. Combinator Runtime

### 4.1 Core Combinators

The runtime exposes a compact typed library:

| Combinator | Type Shape | Meaning |
|---|---|---|
| `split` | `A -> List[A]` | Partition an in-memory value into bounded parts (symbolic). |
| `peek` | `ContextRef x Range -> Text` | Sample a bounded slice (size, head, structural sample) for inspection (effect: `CONTEXT_READ`). |
| `slice` | `ContextRef x Range -> Text` | Extract a bounded contiguous range from context for processing (effect: `CONTEXT_READ`). |
| `context-items` | `ContextRef -> List[ItemRef]` | Enumerate context items for `map`/`filter` (effect: `CONTEXT_READ`). |
| `map` | `(A -> B) x List[A] -> List[B]` | Apply a function to each item. |
| `filter` | `(A -> Bool) x List[A] -> List[A]` | Keep selected items. |
| `reduce` | `(B x B -> B) x List[B] -> B` | Combine values. |
| `concat` | `List[Text] -> Text` | Join text values. |
| `cross` | `List[A] x List[B] -> List[Pair[A,B]]` | Cartesian product. |
| `leaf-call` | `BoundedPrompt -> TypedValue` | Invoke the LLM oracle. |
| `validate` | `Schema x Value -> ValidationResult` | Validate against a boundary schema. |
| `fix` | `(F -> F) -> F` | Tie bounded recursive programs. |

The library may also provide pragmatic extensions:

- `map-async`
- `tree-reduce`
- `fold-sequential`
- `iterate` (bounded refine/validate loop; runs at most a declared max iterations)
- `memoized` (canonical-call deduplication; repeated identical calls count once)
- `race`
- `checkpoint`
- `gate`
- `partial-result`
- `py-exec`

These extensions **[MUST]** be declared as effects where applicable.

### 4.2 Effects

Effects are explicit:

- `LLM`
- `CONTEXT_READ`
- `PY_EXEC`
- `CHECKPOINT`
- `GATE`
- `PARTIAL_RESULT`

`CONTEXT_READ` is the bounded context-slice effect (`peek`, `slice`, `context-items`); it reads from Python's `ContextStore` and returns only bounded slices, never the whole context.

Every template or plan has an inferred effect set from its body. Verification checks:

```text
required_effects <= policy.allowed_effects
```

`py-eval` is syntax sugar for `py-exec` and requires `PY_EXEC`.

### 4.3 Totality and Determinism

All symbolic combinators **[MUST]** be total and deterministic over valid typed inputs. Partial operations must return structured errors, not crash the runtime.

The LLM is the only nondeterministic semantic oracle unless `py-exec` is explicitly allowed.

### 4.4 Symbolic-First Composition

This is the core efficiency principle of λ-RLM and the source of its largest measured wins: structural work that can be done symbolically **[MUST NOT]** be delegated to the model.

- The planner **[MUST]** use deterministic combinators (`split`, `filter`, `concat`, `cross`, symbolic `reduce`) wherever the task permits, and emit `leaf-call` only at bounded leaves or where composition genuinely requires understanding.
- `compose` **[MUST]** default to a symbolic operator. A neural reduce (an `llm-query` inside `tree-reduce`/`fold-sequential`) is used only when the task's composition step itself needs the model.
- For search and retrieval, the planner **[MUST]** apply a symbolic `filter` to narrow candidates before any `leaf-call`. Filtering **[MAY]** use embeddings or lexical signals when available; both are optional and degrade to "keep all" as an upper bound.

The practical effect: when composition is symbolic, total LLM calls collapse to the leaf count (§5.3), and quadratic structural tasks (for example all-pairs comparison via `cross`) cost no extra neural calls.

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
- simulated `recursive_depth <= policy.max_recursion_depth`;
- every `iterate` node declares a finite `max_iterations`, and its simulated iteration count does not exceed it.

If the size-decrease check cannot be proven from the splitter, verification fails. `iterate` loops terminate by their declared iteration cap rather than a size-decrease argument; `termination_bound` (check 11) enforces both the `fix`/split-decrease conditions above and the presence of a finite `max_iterations` on every `iterate` node.

### 5.3 Call Count Bound

For a balanced recursive plan with split factor `k`, input size `n`, and leaf threshold `tau`:

```text
depth      = ceil(log_k(ceil(n / tau)))
leaf_calls = k^depth
reduce_nodes = (k^depth - 1) / (k - 1)
```

The number of LLM calls depends on whether composition is symbolic (§4.4):

```text
llm_calls <= leaf_calls                         when compose is symbolic
llm_calls <= leaf_calls + reduce_nodes          when compose is a neural reduce
```

λ-RLM's clean `N(n) = k^depth + 1` bound is the symbolic-composition case (the `+1` is the final reduce). A neural reduce adds one LLM call per internal node, which also makes `k` a latency lever: larger `k` gives a shallower, faster reduce tree (§5.5).

The dry run measures exact counts for the instantiated plan. The analyzer records both the closed-form upper bound and the simulated exact count, split into leaf and reduce calls. Verification fails if simulated counts exceed the closed-form bound for the chosen composition kind.

### 5.4 Cost Bound

Leaf call cost:

```text
prompt_tokens(call) = ceil((instruction_chars + input_chars) / 4)
completion_tokens(call) = max_tokens or model.default_completion_estimate
cost(call) = prompt_tokens * input_rate + completion_tokens * output_rate
```

Total dry-run cost is the sum over simulated leaf calls. The **low** estimate assumes no escalation. The **high** estimate caps completion at `model.max_output_tokens` and adds the `retry_then_escalate` worst case: one fallback call per leaf using that policy, priced at the leaf model's `fallback_alias` rates (§3.3). The high estimate is therefore a true upper bound — `cost_budget` (check 16) and `call_count_limit` (check 12) are enforced against it, so escalation to the orchestrator tier cannot silently blow a passing budget at runtime.

### 5.5 Partition Planning

The planner chooses `leaf_threshold_tokens` (`tau`) and `split_factor` (`k`) to satisfy:

- every leaf fits the model's **reliable-input budget** `K` with margin;
- estimated call count <= policy;
- estimated cost <= policy;
- estimated critical path <= policy;
- recursion depth <= policy.

**Sizing `tau` to the reliability knee, not the window.** Context rot means accuracy decays well before the raw window `W` is full (λ-RLM, Definition 4). Each `ModelRegistryEntry` declares `reliable_input_tokens` (`K`), the input size beyond which the model's reliability degrades materially; `K <= context_window_tokens`. The planner sizes `tau` against `K`, not `W`:

```text
leaf_threshold_tokens (tau) in {0.5K, 0.7K, 0.9K}
```

Sizing leaves to `K` keeps each leaf in the model's reliable regime; sizing them to `0.8W` would minimize call count at the cost of pushing every leaf into the rot zone. A template-declared `leaf_threshold_tokens` default (for example the `64000` in §6.2) is only an upper fallback; the planner **[MUST]** clamp the effective `tau` to the selected model's `K`, so the same template behaves correctly whether the assigned leaf model is a 200k-window frontier model or a quantized local model with a few-thousand-token reliable budget.

**Choosing `k`.** With symbolic composition, `k` barely affects total LLM calls (they equal the leaf count regardless), so the planner prefers a larger `k` to keep the tree shallow. With a neural reduce, `k` is a genuine tradeoff: larger `k` means a shallower, faster, but wider reduce tree, while smaller `k` means more, smaller reduce calls. The planner picks the largest `k` whose simulated critical path stays within `policy.max_critical_path`. The implementation **[MAY]** use bounded search over candidate split factors:

```text
split_factor (k) in {2, 3, 4, 5, 8, 10}
```

`k = 2` is cost-optimal only when composition is symbolic and latency is unconstrained (λ-RLM, Theorem 4); it is not a default for neural-reduce plans, where it maximizes reduce-call count and depth.

### 5.6 Prompt-as-Environment Access

The context lives outside the model window in Python's `ContextStore` and is inspected, not ingested. There are two readers, sharing the same bounded-slice semantics but operating in different processes:

- **Planner probe (Python).** During planning (Phase 2), the planner inspects the `ContextStore` directly — record metadata plus bounded `peek` reads — to estimate input size `n` and structure. This needs no Racket run.
- **Runtime context-read effects (Racket).** During dry run and live execution, the combinator program issues `peek` / `slice` / `context-items` as `CONTEXT_READ` effect requests (§10.3); Python answers each from the `ContextStore` with a bounded slice. The full context is never serialized into the run message or held in Racket memory.

`split` then partitions the bounded values returned by these reads. The planner **[MUST]** estimate `n` and structure via the planner probe rather than by loading the full context into a prompt. Only the bounded leaves produced by `split` are ever sent to the model, and only bounded slices ever cross the pipe to Racket.

### 5.7 Heterogeneous Models and Single-Device Execution

The §5.3 call-count bound assumes every LLM call is equivalent. That is true for cost accounting against a remote API with effectively unbounded parallelism, but false when leaves run on a constrained local device (the Tier-2 case, §1.4). A single 24 GB GPU serves leaves *serially* (or in a small fixed batch), so "fan out" produces a queue, not parallelism, and wall-clock — not dollars — becomes the binding constraint.

**Per-model concurrency.** Each `ModelRegistryEntry` declares `max_concurrency` (the effective parallel in-flight calls for that model: a single-GPU local model is typically `1`, a small server batch `2`–`8`, a remote API large). The effective concurrency of a node is `min(node_model.max_concurrency, policy.max_concurrency)`. The simulator and executor schedule per model, not globally.

**Wall-clock estimate.** For each model `m`, with throughput `R_m = throughput_tokens_per_sec`:

```text
per_call_seconds(call, m)  = (prompt_tokens(call) + completion_tokens(call)) / R_m
stage_seconds(level, m)    = ceil(calls_on_m(level) / m.max_concurrency)
                             * mean(per_call_seconds(., m))
wall_clock_estimate        = sum over critical-path levels of max_m stage_seconds(level, m)
                             + device-swap penalties
```

**Device-swap penalty.** Models that share a `device_group` (one physical GPU holds one resident model at a time) cannot both be hot. When consecutive stages on the same `device_group` use different models, add `load_latency_seconds` for the model being swapped in. A plan that alternates between two local leaf models on one card pays this repeatedly; the planner **[SHOULD]** prefer assigning one leaf model per `device_group` per plan, or batch all calls to a model before switching.

**Two-tier objective.** Tier-0 (orchestrator) calls are expensive in dollars but fast and parallel; Tier-2 (local leaf) calls are near-zero dollars but throughput-bound and serial. The planner's objective is therefore to **minimize orchestrator-tier dollar cost and total wall-clock**, not raw call count. Concretely, the planner **[SHOULD]**:

- push grunt-work leaves to a local leaf model (cheap, slow) and reserve the orchestrator-tier model for plan authoring, genuinely-neural reduce, and `retry_then_escalate` fallbacks;
- prefer symbolic composition (§4.4) so a slow local tier is never multiplied by neural reduce nodes;
- choose `k` against the wall-clock model above rather than the unconstrained-latency Theorem-4 optimum, because on a serial device a wider split does not add parallelism — it just enlarges each stage's queue.

These quantities are simulated in dry run (§8) and bounded in verification (`concurrency_limit`, `wall_clock_limit`; §9).

---

## 6. Templates and Plan Artifacts

### 6.1 Role of Templates

Templates are audited combinator programs. They are not arbitrary scripts and not generated by the LLM.

One template may define:

- input boundary schema (or optional named shorthand);
- output boundary schema (or optional named shorthand);
- task kinds;
- required slots;
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
  (effects (LLM))
  (resource-law
    (leaf-calls "k^ceil(log_k(ceil(n/tau)))")
    (critical-path "ceil(log_k(ceil(n/tau))) + 1")))
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
- input/output boundary schemas;
- inferred effects.

`artifact_id = "art_" + artifact_hash[:16]`.

---

## 7. Planner

### 7.0 How the Planner Works (Orientation, Non-Normative)

The planner is the system's **query optimizer**: it turns a task plus statistics about the data into a costed, typed execution plan. It is itself deterministic and offline — it emits a plan and runs nothing live.

**It reasons over statistics, not data.** Like a database optimizer that plans from table cardinalities and histograms rather than the rows themselves, the planner never holds the context. Its entire working state is:

- a context *handle* plus metadata (`n` = estimated size, data shape, item count), obtained by the planner probe (§5.6) — bounded `peek` reads, never the whole input;
- the task description and any hints;
- the requested output schema;
- the model registry — per alias: window `W`, reliable budget `K`, throughput, concurrency, device group, and price;
- the policy budgets (calls, cost, recursion depth, wall-clock).

From that state it builds a plan in four moves:

**1. Classify — pick the algorithm.** Map the task to one of a fixed catalog of *task kinds* (§7.2), the way an optimizer picks hash- vs merge-join from a fixed set. Deterministic when hints suffice; at most one bounded model call breaks ties, and that call is recorded.

**2. Select a skeleton — pick the plan shape.** Each task kind maps to a combinator skeleton (§7.3): a direct leaf for small inputs, recursive split-map-reduce for long-context aggregation, filter-before-leaf for search, `cross` + bounded classify for all-pairs, or an explicit typed chain when several transforms compose. This is plan enumeration over a small audited template catalog — not free-form code generation.

**3. Cost and tune — the optimization.** This is the heart of the planner, and it is pure arithmetic over the statistics above:

- size the leaf threshold `tau` to the leaf model's reliable budget `K`, not its window `W` (§5.5);
- choose the split factor `k` by bounded search, costing each candidate's depth, call count, and — decisively — its single-device wall-clock and dollar cost (§5.3, §5.7);
- decide symbolic-vs-neural for every composition step, defaulting to symbolic (§4.4);
- assign each node a model alias — grunt-work leaves to the cheap local tier, plan authoring / neural reduce / escalation to the strong tier (§1.4).

The objective is not "fewest calls" but "least dollar cost and wall-clock within policy." Candidates that bust a budget are discarded here, before any simulation.

**4. Thread the types — check the dataflow.** Walk the chain and require each step's output schema to be structurally compatible with the next step's input schema (§3.2, §7.4) — ordinary type-checking of the pipeline. The resulting `schema_flow` is recorded.

**The output is an EXPLAIN, not a result.** `plan_strategy` returns a recommended `PlanRecord` plus *alternatives* annotated with cost/latency/quality tradeoffs (§7.5), so the Tier-0 caller can choose. The chosen parameters (`k`, `tau`, `compose`, per-node model) live in `planner_parameters`; the record references an immutable program but holds no data and triggers no live call. Because the planner is deterministic and its artifact is content-addressed (§6.4), identical inputs always yield an identical plan — the property that makes dry-run estimates trustworthy and executions reproducible.

What the planner deliberately does **not** hold: any context bytes, any intermediate results, any live model output. Its state is candidate plans and the statistics it costed them with — which is exactly why it can plan over inputs far larger than any window.

### 7.1 Inputs

Planner input:

- `context_id`;
- task description;
- hints;
- context metadata;
- requested output schema;
- model registry;
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

The planner chooses, preferring symbolic combinators over neural calls wherever the task permits (§4.4):

- direct leaf call when input fits;
- recursive split-map-reduce for long-context summarise/aggregate;
- symbolic filter before leaf calls for search;
- symbolic cross product plus bounded classification for pairwise tasks;
- iterative validation plan for refine/validate;
- explicit chain when multiple semantic transformations are needed.

### 7.4 Chain Typing

For every adjacent pair, the producer's output schema must be structurally compatible with the consumer's input schema (§3.2):

```text
compatible(step_i.output_schema, step_{i+1}.input_schema)
```

Verification fails if no compatibility relation exists.

### 7.5 Strategy Alternatives

`plan_strategy` should return:

- recommended plan;
- alternatives with cost/latency/quality tradeoffs;
- schema flow;
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
    context_reads: int
    calls_by_model: dict[str, int]
    concurrency_by_model: dict[str, int]      # effective per-model parallelism (§5.7)
    wall_clock_seconds_estimate: float        # single-device-aware latency (§5.7)
    escalation_upper_bound_calls: int         # worst-case retry_then_escalate fallbacks (§8.2)

class SimulatedCall(BaseModel):
    call_id: str
    node_id: str | None
    model: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    instruction_chars: int
    input_chars: int
    max_tokens: int | None
    json_mode: bool
    depth: int

class CostEstimate(BaseModel):
    prompt_tokens: int
    completion_tokens_low: int
    completion_tokens_high: int
    cost_usd_low: float                        # no escalation
    cost_usd_high: float                       # includes escalation worst case (§5.4)
    escalation_calls_high: int                 # == SimulationStats.escalation_upper_bound_calls
    calls_high: int                            # SimulationStats.llm_calls + escalation_calls_high
    calls_by_model: dict[str, int]
```

`CostEstimate` is computed by `CostAnalyzer` from the simulated `calls` (§5.4) and stored on the `DryRunRecord` (§13). The low figures assume no escalation; the high figures fold in the one-fallback-per-leaf `retry_then_escalate` worst case, priced at each leaf model's `fallback_alias` rates.

The terminal runtime message in simulate mode includes both:

```json
{"stats": "...", "calls": ["..."]}
```

Python computes call graph and cost from `calls`.

### 8.2 Simulation Semantics

Simulation is deterministic:

- `leaf-call` records a `SimulatedCall` and returns a synthetic value of the declared output schema;
- `peek` / `slice` / `context-items` are simulated from context metadata: they return bounded slice sizes (and item counts) without a live read, carry no model cost, and are counted as `CONTEXT_READ` effect uses;
- `split` creates deterministic partitions;
- `map` executes every item;
- `filter` uses symbolic predicates when possible, otherwise keeps all items as an upper bound;
- `reduce` executes all required composition steps;
- `race` starts all branches and chooses the first branch in source order;
- `iterate` runs max iterations;
- `memoized` counts repeated canonical calls once;
- `py-exec` is counted but not run;
- leaf calls are scheduled per model under each model's `max_concurrency`, and the simulator estimates `wall_clock_seconds_estimate` by serializing each tree level through its model's concurrency and throughput, adding a `load_latency_seconds` device-swap penalty when consecutive levels on one `device_group` change models (§5.7);
- escalation is not simulated inline (validity failures are unknowable in dry run); instead the analyzer computes `escalation_upper_bound_calls` — one fallback call per leaf using `retry_then_escalate`, attributed to each leaf model's `fallback_alias` — and folds it into the high estimate (`calls_high`, `cost_usd_high`; §5.4, §8.1). The low estimate excludes it. `call_count_limit` (check 12) and `cost_budget` (check 16) are enforced against the high figures, so escalation cannot exceed a passing budget at runtime.

### 8.3 Static Checks

Before dry run, the registry checks:

- template metadata is valid;
- body parses as S-expressions;
- body references only declared slots;
- body uses only allowed combinators and pure forms;
- inferred effects match declared effects;
- declared boundary schemas are valid (§3.1);
- resource-law syntax is valid.

S-expression parsing and body analysis **MUST NOT** use regular expressions.

---

## 9. Verification

Verification runs all checks and never short-circuits.

`decision` is `fail` if and only if at least one `fail`-severity check fails. `warn`-severity checks (13 `critical_path_limit`, 22 `checkpoint_writable`, 24 `wall_clock_limit`) are always recorded in `checks` but never flip `decision`; a plan may pass with outstanding warnings. Live execution may start only when `decision` is `pass`.

Exactly 24 checks:

| # | Name | Severity | Rule |
|---|---|---|---|
| 1 | `artifact_exists` | fail | artifact is stored |
| 2 | `artifact_hash` | fail | recomputed hash matches |
| 3 | `template_known` | fail | template name/version exists |
| 4 | `slot_schema` | fail | slots validate |
| 5 | `input_type_compatible` | fail | context value matches template input schema (structural) |
| 6 | `chain_type_compatible` | fail | adjacent chain output/input schemas compose (structural) |
| 7 | `effects_allowed` | fail | required effects allowed by policy |
| 8 | `context_exists` | fail | context-ref slots resolve |
| 9 | `model_aliases_resolve` | fail | model aliases exist |
| 10 | `primitive_allowlist` | fail | body uses only allowed bindings |
| 11 | `termination_bound` | fail | recursive plan has finite decreasing bound |
| 12 | `call_count_limit` | fail | `calls_high` (simulated calls + escalation worst case) <= policy |
| 13 | `critical_path_limit` | warn | critical path <= policy |
| 14 | `concurrency_limit` | fail | per-model and global max concurrency <= limits (§5.7) |
| 15 | `token_budget` | fail | estimated tokens <= policy |
| 16 | `cost_budget` | fail | `cost_usd_high` (incl. escalation worst case) <= policy |
| 17 | `recursion_depth_limit` | fail | simulated depth <= policy |
| 18 | `dry_run_fresh` | fail | dry run artifact matches plan |
| 19 | `output_schema_valid` | fail | declared boundary schema is valid |
| 20 | `py_exec_policy` | fail | py-exec requires policy allow |
| 21 | `llm_generated_code_absent` | fail | no generated control code path exists |
| 22 | `checkpoint_writable` | warn | checkpoints namespace writable if used |
| 23 | `resource_law_respected` | fail | simulated stats do not exceed declared law |
| 24 | `wall_clock_limit` | warn | estimated wall clock <= policy (§5.7) |

Policy defaults:

```python
max_llm_calls = 500
max_concurrency = 50
max_critical_path = 25
max_tokens = 2_000_000
max_cost_usd = 10.00
max_recursion_depth = 5
allowed_effects = {"LLM", "CONTEXT_READ", "CHECKPOINT", "PARTIAL_RESULT", "GATE"}
max_timeout_seconds = 3600
max_wall_clock_seconds = 1800            # single-device-aware latency budget (§5.7)
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
  "contexts": {"ctx_9f8e7d6c5b4a3210": {"data_shape": "list", "size": 1280000, "item_count": 4200}},
  "limits": {"max_recursion_depth": 5}
}
```

The `contexts` map carries only handles and metadata (shape, size, item count) — never the context bytes. The runtime obtains content lazily through `CONTEXT_READ` effects (§10.3).

### 10.3 Effects

Live effect requests:

```json
{"type":"context_read","id":"call_0009a8b7c6d5e4f3","context_id":"ctx_9f8e7d6c5b4a3210","op":"peek","range":{"start":0,"length":4000}}
{"type":"llm_call","id":"call_001a2b3c4d5e6f70","node_id":"leaf","output_schema":{"type":"object"},"instruction":"...","input":"...","model":"fast_text_model","temperature":0,"json":true,"max_tokens":null}
{"type":"py_exec","id":"call_111a2b3c4d5e6f70","code":"...","input":{},"allowed_imports":["json"],"timeout_seconds":30}
{"type":"checkpoint","id":"ckpt_77a8b9c0d1e2f300","node_id":"reduce","data":{}}
{"type":"gate","id":"gate_77a8b9c0d1e2f300","label":"review","payload":{}}
{"type":"partial_result","node_id":"map","index":7,"value":{}}
```

Python replies to `context_read`, `llm_call`, `py_exec`, `checkpoint`, and `gate`. A `context_read` reply returns only the requested bounded slice (`op` is one of `peek`, `slice`, `context-items`; `slice`/`peek` take a `range`, `context-items` returns item handles). `partial_result` is fire-and-forget.

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

Gates are in-memory only and are not resumable after process death.

---

## 13. Python Interfaces

Core records:

```python
class ModelRegistryEntry(BaseModel):
    alias: str                          # planner-facing name, e.g. "quality_text_model"
    provider: str
    model_id: str
    context_window_tokens: int          # raw window W
    reliable_input_tokens: int          # reliability knee K; K <= W (§5.5)
    input_rate: float                   # USD per prompt token
    output_rate: float                  # USD per completion token
    max_output_tokens: int              # caps the high cost estimate (§5.4)
    default_completion_estimate: int    # used when a leaf sets no max_tokens (§5.4)
    max_concurrency: int = 1            # effective parallel in-flight calls; 1 for a single-GPU local model (§5.7)
    throughput_tokens_per_sec: float    # for the wall-clock estimate (§5.7)
    load_latency_seconds: float = 0.0   # cost to swap this model onto its device (§5.7)
    device_group: str | None = None     # models sharing a value contend for one device; None = remote/unconstrained
    fallback_alias: str | None = None   # escalation target for retry_then_escalate (§3.3)

class ContextRecord(BaseModel):
    context_id: str
    data: Any
    data_shape: str
    schema: dict[str, Any] = {}
    metadata: dict[str, Any]
    created_at: float

class TemplateMeta(BaseModel):
    name: str
    version: str
    task_kinds: list[str]
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    slots: dict[str, Any]
    effects: list[str] = []
    resource_law: dict[str, str] = {}

class PlanRecord(BaseModel):
    plan_id: str
    context_id: str
    task: str
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    program_ref: str
    schema_flow: list[tuple[dict[str, Any], dict[str, Any]]]
    planner_parameters: dict[str, Any]
    alternatives: list[dict[str, Any]] = []
    created_at: float

class ArtifactRecord(BaseModel):
    artifact_id: str
    plan_id: str
    program_hash: str
    metadata_hash: str
    slot_values: dict[str, Any]
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}
    effects: list[str]

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
ModelRegistry(config_path: Path)
ContextStore(store: Store)
CombinatorRegistry(runtime_dir: Path)
TemplateRegistry(template_dir: Path, models: ModelRegistry)
Planner(templates: TemplateRegistry, models: ModelRegistry)
CostAnalyzer(models: ModelRegistry)
VerificationEngine(store: Store, templates: TemplateRegistry)
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

Acceptance:

- ID grammar tests;
- store namespace tests;
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

### Batch 2: Boundary Schemas and Contexts

Files:

- `rlm_scheme/schema.py`
- `rlm_scheme/context_store.py`

Acceptance:

- boundary schema validation (§3.1);
- structural schema compatibility (§3.2);
- context loading with boundary schema;
- restricted JSON Schema fixtures (object, array, enum).

### Batch 3: Planner

Files:

- `rlm_scheme/planner.py`
- `rlm_scheme/cost.py`

Acceptance:

- direct plan when input fits;
- recursive plan when input exceeds window;
- split-factor bounded search;
- schema-flow construction;
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
- per-model wall-clock estimate with device-swap penalty (§5.7);
- all 24 verification checks;
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
2. Boundary schemas validate fixtures (§3.1, §3.2).
3. A direct plan executes end to end.
4. A recursive long-context plan executes end to end.
5. Dry-run measured calls match live host-counted calls.
6. Closed-form bounds are recorded and enforced for recursive plans.
7. Chain schema compatibility is enforced.
8. No live LLM call occurs before passing verification.
9. Racket sandbox escape tests fail.
10. No S-expression parsing or body analysis uses regex.
11. MCP exposes exactly 10 tools.
12. Verification runs exactly 24 checks and never short-circuits.
13. Symbolic-first composition is the planner default; symbolic-composed plans collapse LLM calls to the leaf count.
14. Leaf threshold `tau` is sized to the model's reliable-input budget `K`, not its raw window `W`, and is clamped to `K` regardless of any template default.
15. Context bytes are never serialized into the run message or held by the Racket runtime; the runtime obtains content only as bounded slices via `CONTEXT_READ` effects.
16. Per-model concurrency and a single-device-aware `wall_clock_seconds_estimate` are simulated and bounded; a single-GPU leaf model is modeled as serial (`max_concurrency = 1`) with a device-swap penalty across `device_group`.
17. Heterogeneous routing works: a plan can run grunt-work leaves on a local model while reserving an orchestrator-tier model for authoring and `retry_then_escalate` fallbacks.
18. `SPEC-DEVIATIONS.md` is empty or reviewed.

