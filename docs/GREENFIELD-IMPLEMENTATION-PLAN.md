# RLM-Scheme Greenfield Implementation Plan

**Status:** Normative implementation plan.
**Audience:** an implementing agent.
**Design basis:** a practical Racket/Python implementation of the lambda-RLM idea: long-context reasoning should be recursive, typed, and combinator-driven, with LLM calls confined to bounded leaf subproblems.

RLM-Scheme is an MCP server for auditable long-context LLM computation. It treats the input context as an external environment, constructs a typed functional execution plan over that environment, dry-runs the plan to compute cost and resource bounds, verifies the plan against policy, and executes it once. The LLM does not write orchestration code.

The central move is:

```text
model-authored code run immediately in an open-ended REPL loop
  -> model-authored program in a restricted total language,
     verified and resource-bounded before it runs, with bounded neural leaf calls
```

This design is intentionally narrower than a general agent framework. It is a runtime for structured decomposition, mapping, filtering, reduction, comparison, validation, and synthesis over contexts larger than a model can safely consume in one prompt.

---

## Orientation (Plain-Language, Non-Normative)

This section is a conceptual map for a reader with a general computer-science background and no LLM-specific experience. Sections 0–15 are the normative specification; this one only builds intuition and may be skipped by an implementer.

### The one-line idea

RLM-Scheme **runs an expensive cloud model's reasoning over a context far larger than it can afford to read, by decomposing the task so the bulk of the work runs on small models on hardware you already own.** A frontier model authors the strategy as a program in a restricted, total combinator language through the tool API; the host type-checks it, fixes its resource bounds, and verifies it before it runs; the verified program then drives cheap local models at the leaves, escalating back to the cloud model only when a local leaf fails. Seen another way, it is **a query planner for LLM calls** — it proves resource bounds before it runs and invokes a model only at the leaves — but the economic payoff is that the costly model's footprint collapses to plan-authoring plus the occasional rescue, while everything else runs locally.

### The problem, in CS terms

A language model has a fixed **context window**: the maximum input it can read at once — a hard `MAX_INPUT` on a function `model : String -> String`. Two failure modes follow. First, data larger than the window is truncated. Second, and subtler, accuracy decays *before* the window is full — "**context rot**." The function is defined up to `W` tokens but only *reliable* up to some smaller `K`. `K`, not `W`, is the real budget.

The popular fix ("recursive language models") lets the model write and run its own control code in a REPL to chop the data up. That is `eval()` on stochastic output: it may not parse, may not terminate, may cost anything — properties no compiler author would accept.

### The core move

The dangerous part of the recursive-LLM idea is not that the *model* writes the control flow — it is that the control flow is **arbitrary code, run immediately, unchecked**. Those are two separable properties. Take away the second and you can keep the first.

- **Kept — "prompt as environment":** the data lives *outside* the model, in a store the host owns. The program *inspects* it through bounded reads (`peek`, `slice`, `split`) rather than swallowing it whole — an external data source behind a cursor, not an in-memory blob.
- **Kept, but caged — model-authored control flow.** The model *does* write the orchestration, but only as a program in a **restricted, total combinator language**: `map`, `filter`, `reduce`, `concat`, `cross`, `split`, and a recursion operator `fix` whose every use must prove its input shrinks. The model appears twice — as the *author* of this program, and as exactly one primitive *inside* it:

  ```text
  M : BoundedPrompt -> TypedValue        -- the "leaf oracle"
  ```

- **Removed — unchecked execution.** No program the model writes ever runs on faith. Every program is parsed, type-checked at each boundary, simulated for exact cost, proven to terminate, and budget-gated *before a single live call* (the EXPLAIN lifecycle below). A program that fails is rejected and handed back to the model to repair.

So the system is **an interpreter for a total combinator language whose only nondeterministic primitive is the model** — the model both writes the program and is called at its leaves, but the program is admitted only once it is proven bounded. The language is deliberately **not Turing-complete**: expressive enough for decompose / map / filter / reduce / search / compare, restricted just enough that termination and cost are provable before execution. That static provability is the whole point — it is what a model-written REPL loop can never offer.

### The lifecycle: EXPLAIN before you run

Like a database that shows a query plan and cost estimate before executing:

```text
load_context -> plan -> dry_run -> verify -> execute
```

- **plan** — the model authors a combinator program for the task (optionally starting from a host-suggested skeleton, §6) and submits it through the tool API. The host type-checks it and fills in the *numeric* safety parameters deterministically — leaf size `tau`, split factor `k`, per-node model — so the model owns the control-flow shape while the host owns the resource bounds. The model writes the program, but only in the checked total language; it never runs arbitrary code.
- **dry_run** — the program runs in **simulate mode**: every `leaf-call` returns a synthetic typed value instead of hitting the model. This yields exact call counts, a cost range, recursion depth, and a wall-clock estimate with *zero* live calls. It is symbolic execution for resource accounting.
- **verify** — 24 static checks (termination, budgets, schema compatibility, "the program uses only the allowed total primitives — no escape to arbitrary code," …). This is the **safety boundary for the model-authored program**. **No live call is permitted until verification passes** — type-checker, linter, and budget gate combined. On failure the model receives structured errors and revises, a bounded repair loop (§9); execution only ever runs an admitted program.
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

Take the reliability and cost-predictability of a compiled, type-checked, resource-bounded dataflow program; use a frontier model only to author the plan and rescue failed leaves, and run the bulk of the work as bounded leaf calls on small local models — invoking any model only on inputs small enough to trust.

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

RLM-Scheme keeps *both* useful parts — prompt-as-environment **and** model-authored control flow — and removes only the unchecked-execution part. The model writes the orchestration, but in a restricted total combinator language, and the program is parsed, typed, simulated, proven to terminate, and budget-gated before it runs. Every failure mode above is caught statically: code that does not parse, does not type, or cannot be shown to terminate or stay within budget is rejected before any live call, and the model repairs it. The model is used as a bounded oracle at leaves (and at explicitly declared neural nodes), and as the author of a program the host verifies.

### 0.2 Goals

RLM-Scheme **[MUST]** provide:

- a compact, total combinator library expressive enough to author orchestration in;
- typed inputs, outputs, and chain boundaries;
- all orchestration as a model-authored program in that restricted total language — parsed, typed, proven-terminating, and budget-gated before any live call, never run as arbitrary code;
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

Two gains, neither of which assumes the model has become smarter.

First, **structure**: decomposition, bounded leaf calls, symbolic map/filter/reduce, typed intermediate values, and verification can make some long-context tasks more reliable than one-shot prompting or model-authored REPL loops.

Second, **economics**: because orchestration is verified and resource-bounded before it runs and every leaf is bounded, the expensive cloud model is needed only to author the plan and to rescue the occasional failed leaf (§3.3). The bulk of the work — every grunt-work leaf — runs on small models on local hardware (§1.4, §5.7). The frontier model's footprint shrinks toward a near-constant per task, and the marginal cost of scaling the work moves from dollars-per-call on a metered API to wall-clock on a GPU you already own. This is the primary motivation for the heterogeneous, multi-tier execution model; it is not an afterthought of the structural design but a co-equal goal.

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

Racket owns typed functional control flow. It evaluates a pre-verified combinator program (authored by the model, admitted by the host before this point). It does not receive user data by textual substitution. It receives data as JSON values and context bindings.

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

- prompt estimate <= selected model context window `W` (the hard floor); the planner sizes leaves to the tighter reliable-input budget `K <= W` (§5.5), so in practice prompts sit well under `W`;
- requested output type is known;
- call is counted against budget;
- call is recorded in the trace;
- call is served only after verification passes.

### 1.4 Orchestration Tiers

RLM-Scheme is built to run across heterogeneous models without embedding any of them as the orchestrator. Three tiers cooperate:

- **Tier 0 — caller (a coding agent over MCP, e.g. a Happy/Claude session).** A strong model authors the orchestration: it writes a combinator program for the task (optionally starting from a host-suggested skeleton, §6), supplies the leaf/compose instructions, submits it through `plan_strategy`, inspects dry-run estimates and traces, and decides whether to execute. It writes the control-flow *shape* in the restricted total language; the host deterministically fills the numeric safety parameters and verifies the program before it runs. This is the only place a frontier model is required.
- **Tier 1 — combinator engine (no model).** Deterministic split/filter/map/reduce/cross plus the cost/termination bounds and verification. It does not author the program; it *checks and executes* the one the model wrote.
- **Tier 2 — leaf models.** Bounded `leaf-call` oracles, which **[MAY]** be cheap models on constrained local hardware (e.g. a single 24 GB GPU), served through a local backend declared per entry — `openai_compatible`, `ollama`, `vllm`, or `llamacpp` (§13, Batch 5). Multiple leaf models and an orchestrator-tier model can coexist; each is a `ModelRegistryEntry` (§13) with its own backend/endpoint, window `W`, reliability knee `K`, prefill/decode rates, concurrency, device group, and fallback.

A model's tier is not hardcoded; it follows from how nodes are assigned model aliases — proposed by the author and defaulted/clamped by the host. The same alias machinery routes grunt-work leaves to a local model while reserving an orchestrator-tier model for plan authoring, neural reduce that genuinely needs reasoning, and escalations (§3.3). The single-device economics that make Tier 2 behave very differently from a remote API are specified in §5.7.

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

### Phase 2: Statistics and Skeleton Suggestion

The host gathers the statistics the author needs and **[MAY]** suggest a skeleton:

- task kind (from hints, or proposed by the host);
- structural data shape;
- requested output schema;
- whether the input fits the selected model's **reliable-input budget** `K` (not its raw context window `W`).

The host reads structure cheaply through `peek` (§5.6) rather than ingesting the whole context, so this stays cheap even on inputs far larger than the window. From these statistics the host **[MAY]** offer a matching skeleton from the catalog (§7.3) as a starting point; the authoring model accepts, edits, or ignores it. The host never commits the model to a skeleton — it only scaffolds.

### Phase 3: Direct Dispatch

If the input fits the reliable-input budget `K` and the task does not require decomposition, the authored program **[MAY]** be a single direct typed leaf call:

```text
direct_call : InputType -> OutputType
```

This still goes through dry run and verification like any other program.

### Phase 4: Program Authoring

If the input does not fit, the model authors a recursive combinator program. The canonical shape (which a host skeleton may supply, §5.1) is:

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

Authoring divides cleanly into a *shape* the model chooses and *numeric safety parameters* the host fills deterministically:

- **Model chooses (control-flow shape):** which combinators, how they compose, where `leaf-call` appears, and `composition_operator` (`compose`) — and the model **[MUST]** prefer a symbolic operator (concat, merge, sum), using a neural reduce only when composition genuinely requires understanding (§4.4).
- **Host fills and clamps (resource parameters):** `leaf_threshold_tokens` (`tau`), sized to the model's reliable-input budget `K`, not its window `W`, and clamped to `K` regardless of any author-supplied value (§5.5); `split_factor` (`k`), chosen by bounded search against the critical-path and wall-clock policy rather than a magic number (§5.5, §5.7); and `max_depth`. Letting the host own these is what preserves the static bounds even though the model authored the shape — the model cannot set a `tau` that pushes leaves past `K` or a `k` that busts the latency budget.

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

The runtime executes the verified program once. There is no open-ended REPL in which the LLM writes and runs fresh code mid-execution: the program was authored, verified, and frozen *before* this phase, and execution only interprets it. Recursive calls are internal to the combinator program and bounded by the verified depth. Any change to the orchestration means authoring a new program and re-running plan → dry-run → verify.

---

## 3. Typed Boundaries

Typing in RLM-Scheme is structural, not a domain ontology. Following λ-RLM, "typed" means two things only:

1. **Parametric combinator typing.** Combinators carry type shapes (`Map : (A -> B) x List[A] -> List[B]`) so an authored program composes them without nonsense like reducing a non-list, and the verifier rejects it when it does.
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

### 4.0 Program Grammar and Termination Discipline

Because programs are now model-authored (§7), the language the model writes in and the rule that makes it total must be **exact** — the verifier accepts or rejects against this definition, and the author must target it. This subsection is normative; checks 10 (`primitive_allowlist`), 11 (`termination_bound`), and 21 (`total_language_closed`) and the §8.3 static checks all refer to it.

**Grammar.** A program is a single expression in this restricted S-expression language. No other forms are legal:

```text
program     ::= expr
expr        ::= literal
              | var
              | slot-ref
              | if
              | let
              | lambda
              | application
literal     ::= string | number | boolean | "null"
              | "(quote" datum ")"          ; quote of DATA only — never code
var         ::= identifier                  ; must be bound by let / lambda / fix
slot-ref    ::= "(slot" symbol ")"          ; a named input binding (§6.3); data, not code
if          ::= "(if" expr expr expr ")"
let         ::= "(let ((" var expr ")" ...) expr ")"
lambda      ::= "(lambda (" var ... ")" expr ")"
application ::= "(" operator expr ... ")"
operator    ::= combinator-name             ; ONLY a name on the §4.1 allow-list
              | var                          ; a locally-bound function value (e.g. the recursive self)
fix         ::= "(fix (lambda (" var ")" expr "))"   ; see termination discipline
```

Constraints enforced by verification:

- **Closed under the allow-list (`total_language_closed`, check 21).** The operator position of an `application` **[MUST]** be either a §4.1 combinator name or a locally-bound `var`. It **[MUST NOT]** be a computed value, a `quote`d symbol resolved dynamically, or any host identifier outside the allow-list. There is no `eval`, no macro, no dynamic operator synthesis. `quote` may wrap only literal data.
- **No free variables (`primitive_allowlist`, check 10, with §8.3).** Every `var` **[MUST]** be bound by an enclosing `let`, `lambda`, or `fix`, or resolve to a `slot-ref`/declared input. Unbound identifiers fail.
- **`py-exec` is the only door to arbitrary code**, and it is a policy-gated effect (check 20); absent that effect, the program is pure but for `leaf-call` and the `CONTEXT_READ` reads.

**Termination discipline (`termination_bound`, check 11).** `fix` is the only recursion form, and it is admitted only when the verifier can prove, *syntactically*, that every recursive call descends on a structurally smaller value:

1. A `fix` body **[MUST]** contain an `if` whose then/else split a **base case** (no recursive call — a `leaf-call` or symbolic value) from a **recursive case**, and whose guard is a size test against the leaf threshold (`size(x) <= tau` or equivalent on a `context-items` count).
2. In the recursive case, every application of the recursive `self` **[MUST]** take as its argument a value that traces back to the bound parameter `x` **through at least one `split` or `context-items`**, composed only with size-non-increasing combinators (`filter`, `map` of a non-growing function, element selection). Equivalently: the recursive argument is a *proper sub-part* of `x`, never `x` itself nor a value not derived by descent from `x`.
3. `split` **[MUST]** satisfy §5.2 (factor `>= 2`, strictly smaller children); this is what makes the descent in (2) a strict decrease.
4. Non-`fix` bounded loops use `iterate` with a declared finite `max_iterations` (§5.2); they terminate by the cap, not by descent.

A program whose recursion the verifier cannot place in this shape is **rejected with a structured error** (§9 repair loop), not executed. This syntactic discipline — not a halting oracle — is exactly why the language is total-by-construction and why the dry-run depth/call bounds (§5.3) are guarantees rather than guesses.

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

The program is authored by the model (a nondeterministic step), but authoring is a **separate, pre-execution phase**: once a program is verified and frozen, *executing* it is deterministic except for the `leaf-call` oracle. So the LLM is the only nondeterministic semantic oracle at run time (unless `py-exec` is explicitly allowed), and re-running a verified program reproduces the same call structure and cost.

### 4.4 Symbolic-First Composition

This is the core efficiency principle of λ-RLM and the source of its largest measured wins: structural work that can be done symbolically **[MUST NOT]** be delegated to the model. It is an obligation on the authored program, which the host's suggested skeletons follow by default and which verification and the dry-run cost estimate make visible (a program that routes structural work through the model simply costs more and is the author's choice to repair).

- The authored program **[MUST]** use deterministic combinators (`split`, `filter`, `concat`, `cross`, symbolic `reduce`) wherever the task permits, and emit `leaf-call` only at bounded leaves or where composition genuinely requires understanding.
- `compose` **[MUST]** default to a symbolic operator. A neural reduce (an `llm-query` inside `tree-reduce`/`fold-sequential`) is used only when the task's composition step itself needs the model.
- For search and retrieval, the program **[MUST]** apply a symbolic `filter` to narrow candidates before any `leaf-call`. Filtering **[MAY]** use embeddings or lexical signals when available; both are optional and degrade to "keep all" as an upper bound.

The practical effect: when composition is symbolic, total LLM calls collapse to the leaf count (§5.3). For all-pairs tasks, the `cross` product that enumerates the pairs is itself symbolic and free; what costs neural calls is the per-pair judgment — one `leaf-call` per pair, i.e. O(n²) leaves for a pairwise classify. The win is that the *structural* join adds no calls on top of those leaves, not that all-pairs work is free. Where the pairwise judgment can also be made symbolic (a comparator, a key match), it should be, eliminating the leaves entirely.

---

## 5. Recursive Executor

### 5.1 Fixed Recursive Shape

Recursive plans are represented as a combinator term in the total language, authored by the model and verified by the host before execution.

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

The implementation may encode this as Racket S-expressions. The recursive shape is authored by the model, but its `fix`/split structure **[MUST]** satisfy the §5.2 termination conditions, which the host verifies before any live call; a shape whose recursion cannot be shown to shrink is rejected and returned for repair. This is what makes "model-authored" compatible with "statically bounded": the author proposes the shape, the verifier admits it only if it provably terminates.

### 5.2 Termination Conditions

A recursive plan is valid only if:

- `split_factor >= 2`;
- `leaf_threshold_tokens > 0`;
- every split child has strictly smaller estimated size than its parent when parent size exceeds threshold;
- `max_depth` is finite;
- simulated `recursive_depth <= policy.max_recursion_depth`;
- every `iterate` node declares a finite `max_iterations`, and its simulated iteration count does not exceed it.

Throughout the recursive shape, `size` is measured in **estimated tokens** (`ceil(chars / 4)`, §5.4) — the same unit as `leaf_threshold_tokens` (`tau`) and the model budgets `W`/`K` — so the threshold test `size(input) <= tau` and the strict-decrease test are comparing like with like. If the size-decrease check cannot be proven from the splitter, verification fails. `iterate` loops terminate by their declared iteration cap rather than a size-decrease argument; `termination_bound` (check 11) enforces both the `fix`/split-decrease conditions above and the presence of a finite `max_iterations` on every `iterate` node.

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

The numeric partition parameters are **host-owned**, not author-owned (§7.0): the model authors the program shape and **[MAY]** propose values, but the host fixes and clamps `leaf_threshold_tokens` (`tau`) and `split_factor` (`k`). "The planner" in this section means that host parameter-sizing role. It chooses `tau` and `k` to satisfy:

- every leaf fits the model's **reliable-input budget** `K` with margin;
- estimated call count <= policy;
- estimated cost <= policy;
- estimated critical path <= policy;
- recursion depth <= policy.

**Sizing `tau` to the reliability knee, not the window.** Context rot means accuracy decays well before the raw window `W` is full (λ-RLM, Definition 4). Each `ModelRegistryEntry` declares `reliable_input_tokens` (`K`), the input size beyond which the model's reliability degrades materially; `K <= context_window_tokens`. The planner sizes `tau` against `K`, not `W`:

```text
leaf_threshold_tokens (tau) in {0.5K, 0.7K, 0.9K}
```

`K` is a per-deployment tunable, not a published constant, and is ideally measured rather than guessed: sweep input length on a needle-in-a-haystack or task-representative probe and take the length at which accuracy drops materially below the small-input baseline. When no measurement exists, hand-set `K` conservatively as a fraction of `W` (a quantized local model's reliable budget is often a small fraction of its advertised window). Sizing leaves to `K` keeps each leaf in the model's reliable regime; sizing them to `0.8W` would minimize call count at the cost of pushing every leaf into the rot zone. The `{0.5K, 0.7K, 0.9K}` candidate set *is* the margin referred to in the "fits `K` with margin" criterion above — there is no separate margin term; choosing a fraction below `1.0K` is exactly how the planner leaves headroom, with smaller fractions trading more calls for more reliability. A template-declared `leaf_threshold_tokens` default (for example the `64000` in §6.2) is only an upper fallback; the planner **[MUST]** clamp the effective `tau` to the selected model's `K`, so the same template behaves correctly whether the assigned leaf model is a 200k-window frontier model or a quantized local model with a few-thousand-token reliable budget.

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

**Wall-clock estimate.** Local inference has two very different rates: prompt **prefill** (ingesting the prompt, effectively parallel over prompt tokens) and **decode** (generating completion tokens one at a time). For decomposition leaves — long prompt, short output — these differ by an order of magnitude, so a single throughput figure mis-estimates the constraint that actually binds on local hardware. Each `ModelRegistryEntry` therefore declares `prefill_tokens_per_sec` and `decode_tokens_per_sec` separately; a remote model whose API exposes only an aggregate rate may set both to that figure. For each model `m`:

```text
per_call_seconds(call, m)  = prompt_tokens(call) / m.prefill_tokens_per_sec
                             + completion_tokens(call) / m.decode_tokens_per_sec
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

Templates are an **optional standard library** of audited combinator programs and skeletons. They are not mandatory and not the only thing that can run: the model **[MAY]** author a program directly from the primitives. Their purpose is leverage and safety-by-reuse —

- a skeleton the host **[MAY]** *suggest* for a recognized task kind (§7.3), which the authoring model accepts, edits, or ignores;
- a reusable sub-program the model can reference by name instead of re-deriving;
- a worked example of idiomatic, verifiable primitive use.

A template earns no trust the verifier does not re-establish: whether a program is authored fresh or built from a template, it goes through the identical parse / type-check / dry-run / verify path (§8, §9). Templates are a convenience for the author and a head start for the host's suggestions, not a privileged execution path.

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

There are no `{{slot}}` markers. Slot content is never parsed as code. This holds identically for a freshly authored, template-less program: instruction text (e.g. `leaf_instruction`, `compose_instruction`) is supplied either as a string literal in the program body or as a `slot-ref` bound through `slot_values` (§7.1) — in both cases it is data passed to `leaf-call`, never an operator or control form.

### 6.4 Artifact Hash

Artifact hash includes:

- template name (empty for a freshly authored program);
- template version (empty for a freshly authored program);
- body hash (the hash of the authored S-expression source — the primary identity for a template-less program);
- metadata hash;
- canonical slot values;
- input/output boundary schemas;
- inferred effects.

`artifact_id = "art_" + artifact_hash[:16]`. The artifact is content-addressed: identical program source + slot values + schemas yield the same `artifact_id`, which is what makes a verified plan re-runnable and the dry-run estimate bindable to execution (§7.0).

---

## 7. Program Authoring and Validation

### 7.0 How a Program Gets Built and Checked (Orientation, Non-Normative)

The orchestration is **authored by the Tier-0 model** and **validated by the host**. There is no deterministic planner that emits the program; instead the host plays the role of a query *validator and optimizer* over a program the model wrote — it type-checks the dataflow, fills the numeric safety parameters, costs the result, and either admits it or returns errors for repair. Think SQL: the model writes the query, the host plans, bounds, and may reject it before it runs.

> **Terminology.** Where this spec says "the planner" — and in the component name `Planner`, the file `planner.py`, and the field `planner_parameters` — it means this host validation/costing/parameter-sizing role. It **does not author** the program; the model does. The name is retained because the host's job (type-check, cost, optimize numeric parameters, admit/reject) is exactly a query planner's, minus code generation.

The split of responsibility is the heart of the design:

- **The model owns the control-flow *shape*** — which combinators, how they nest, where `leaf-call` sits, whether a composition is symbolic or neural.
- **The host owns the *numeric* safety parameters and the verdict** — `tau` (clamped to `K`), `k` (bounded search against wall-clock), per-node model defaults, and the pass/fail gate. The model cannot widen these past policy; that is what keeps a model-authored program statically bounded.

**The host reasons over statistics, not data.** Like a database optimizer that plans from cardinalities and histograms rather than the rows, the host never holds the context while validating. Its entire working state is:

- a context *handle* plus metadata (`n` = estimated size, data shape, item count), obtained by the planner probe (§5.6) — bounded `peek` reads, never the whole input;
- the task description and any hints;
- the requested output schema;
- the model registry — per alias: window `W`, reliable budget `K`, throughput, concurrency, device group, and price;
- the policy budgets (calls, cost, recursion depth, wall-clock).

A program goes from idea to admitted in four moves — the model leads moves 1–2, the host leads moves 3–4:

**1. Author the shape (model).** The model writes a combinator program for the task, optionally starting from a host-suggested skeleton (§7.3) keyed to a recognized task kind (§7.2) — direct leaf for small inputs, recursive split-map-reduce for long-context aggregation, filter-before-leaf for search, `cross` + bounded classify for all-pairs, an explicit typed chain when several transforms compose. The model **[MAY]** deviate from any skeleton, so long as the result stays inside the total language. This is authoring in a restricted DSL, not free-form code generation: there is no `eval`, no arbitrary Python (except the policy-gated `py-exec` effect), and no recursion the verifier cannot bound.

**2. Parameterize the leaves (model proposes, host fixes).** The model marks where `leaf-call` appears and whether each composition is symbolic or neural, preferring symbolic (§4.4). It **[MAY]** propose `k`/`tau`/per-node models, but those are advisory.

**3. Cost, tune, and clamp (host).** The host turns the authored shape into a costed plan by pure arithmetic over the statistics above, and it **owns** the resource parameters:

- size the leaf threshold `tau` to the leaf model's reliable budget `K`, not its window `W`, clamping any author-proposed value to `K` (§5.5);
- choose (or accept-and-validate) the split factor `k` by bounded search, costing each candidate's depth, call count, and — decisively — its single-device wall-clock and dollar cost (§5.3, §5.7);
- default each node's model alias — grunt-work leaves to the cheap local tier, neural reduce / escalation to the strong tier (§1.4) — honoring author hints only where they stay within policy.

The objective is not "fewest calls" but "least dollar cost and wall-clock within policy." A shape whose costed form busts a budget is **not silently rewritten** — it is reported back so the author can change the shape (e.g. a coarser split, a symbolic compose). The host tunes numbers; it does not redesign the model's algorithm.

**4. Thread the types and admit-or-repair (host).** Walk the chain and require each step's output schema to be structurally compatible with the next step's input schema (§3.2, §7.4) — ordinary type-checking of the pipeline; the resulting `schema_flow` is recorded. If type-checking, the total-language check, or the costed budgets fail, the host returns **structured errors** and the model revises and resubmits — the **repair loop** (§9), bounded by a policy attempt cap. No live call happens at any point in this loop.

**The output is an EXPLAIN, not a result.** `plan_strategy` accepts the model-authored program and returns a typed `PlanRecord` (plus any host-suggested *alternatives* with cost/latency/quality tradeoffs, §7.5) so the Tier-0 caller can inspect and choose. The host-fixed parameters (`k`, `tau`, `compose`, per-node model) live in `planner_parameters`; the record references the content-addressed program (§6.4) but holds no data and triggers no live call.

**On reproducibility:** the program is authored by a nondeterministic model, so "same task → same program" does **not** hold. What does hold is that a *given admitted program* is content-addressed and its execution is deterministic except for leaf-calls (§4.3): re-running an artifact reproduces its call structure and cost, and the authored program plus the model exchange that produced it are logged, so every run is auditable and replayable even though it is not re-derivable. A deployment that needs bit-identical plans across sessions should pin and reuse the artifact rather than re-author.

What the host deliberately does **not** hold while validating: any context bytes, any intermediate results, any live model output. Its state is the authored program and the statistics it costed it with — which is exactly why it can validate plans over inputs far larger than any window.

### 7.1 Inputs

`plan_strategy` input (the model-authored program plus the context the host needs to validate and cost it):

- the **authored combinator program** — a single S-expression string in the §4.0 grammar (the `program` argument), or `{ "template": name@version, "edits": ... }` to start from a library skeleton (§6.1);
- `slot_values` — bindings for the program's `slot-ref`s, including instruction strings such as `leaf_instruction` / `compose_instruction` (data, never code; §6.3);
- `context_id`;
- task description;
- hints;
- context metadata;
- requested output schema;
- model registry;
- execution policy.

The host parses the `program` against the §4.0 grammar, type-checks its boundaries (§7.4), fills the numeric safety parameters (§5.5), and either stores it (a `programs` entry referenced by `PlanRecord.program_ref`, §12–13) and returns a `PlanRecord`, or returns structured errors for the repair loop (§9). The program is never `eval`'d or string-substituted — it is parsed into an AST and interpreted only by the verified runtime.

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

Task kinds are the index into the skeleton catalog (§7.3): the host uses the task kind (from hints, or inferred) to pick which skeleton to *suggest*. The authoring model is not bound by it — it may author any program in the total language regardless of the suggested kind.

### 7.3 Skeleton Catalog

For each task kind the host can suggest a starting skeleton, preferring symbolic combinators over neural calls wherever the task permits (§4.4). These are starting points the model edits or replaces, not mandatory shapes:

- direct leaf call when input fits;
- recursive split-map-reduce for long-context summarise/aggregate;
- symbolic filter before leaf calls for search;
- symbolic cross product plus bounded classification for pairwise tasks;
- iterative validation plan for refine/validate;
- explicit chain when multiple semantic transformations are needed.

Whatever the model authors — skeleton-derived or fresh — is admitted only through the same §8/§9 checks.

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
- `filter` uses symbolic predicates when possible, otherwise keeps all items as an upper bound (see the conservatism note below);
- `reduce` executes all required composition steps;
- `race` starts all branches and chooses the first branch in source order;
- `iterate` runs max iterations;
- `memoized` counts repeated canonical calls once;
- `py-exec` is counted but not run;
- leaf calls are scheduled per model under each model's `max_concurrency`, and the simulator estimates `wall_clock_seconds_estimate` by serializing each tree level through its model's concurrency and its separate prefill/decode rates (§5.7), adding a `load_latency_seconds` device-swap penalty when consecutive levels on one `device_group` change models (§5.7);
- escalation is not simulated inline (validity failures are unknowable in dry run); instead the analyzer computes `escalation_upper_bound_calls` — one fallback call per leaf using `retry_then_escalate`, attributed to each leaf model's `fallback_alias` — and folds it into the high estimate (`calls_high`, `cost_usd_high`; §5.4, §8.1). The low estimate excludes it. `call_count_limit` (check 12) and `cost_budget` (check 16) are enforced against the high figures, so escalation cannot exceed a passing budget at runtime.

**Filter conservatism.** Because a `filter` whose selectivity cannot be evaluated statically is costed as keep-all, the simulated call count is an upper bound that can be far above the live count. A plan whose real filter would discard most candidates may therefore fail `call_count_limit` or `cost_budget` on a count it would never actually reach at runtime. This is deliberate — verification must bound the worst case — but it means an author whose plan is rejected on a filter-dominated estimate should narrow it so the cost is provable: use a statically-evaluable predicate (lexical/embedding threshold the simulator can apply), pre-filter the candidate set before the plan, or raise the policy budget knowingly. The conservatism is on the safe side: it never under-counts.

### 8.3 Static Checks

Before dry run, the host statically checks the authored program (whether template-derived or fresh):

- if the program is built from a template, its template metadata is valid;
- body parses as S-expressions and conforms to the §4.0 grammar;
- every `var` is bound by an enclosing `let`/`lambda`/`fix` or resolves to a declared `slot-ref`/input — no free variables;
- body uses only the §4.0 forms and the §4.1 allow-listed operators (no operator outside the allow-list, no `eval`/dynamic operator);
- recursion satisfies the §4.0 termination discipline;
- inferred effects match declared/allowed effects;
- declared boundary schemas are valid (§3.1);
- resource-law syntax is valid (when a template declares one).

S-expression parsing and body analysis **MUST NOT** use regular expressions.

---

## 9. Verification

Verification is the **safety boundary for the model-authored program**: because the orchestration is written by a nondeterministic model, every guarantee the system makes is established here, not assumed from who wrote the code. It admits a program only if it is well-typed at every boundary, closed under the total combinator language (no escape to arbitrary code), provably terminating, and within every budget. A frontier model and a local 1B model are held to the identical bar.

Verification runs all checks and never short-circuits.

`decision` is `fail` if and only if at least one `fail`-severity check fails. `warn`-severity checks (13 `critical_path_limit`, 22 `checkpoint_writable`, 24 `wall_clock_limit`) are always recorded in `checks` but never flip `decision`; a plan may pass with outstanding warnings. Live execution may start only when `decision` is `pass`.

**Repair loop.** When `decision` is `fail`, the host returns the failing checks as structured errors (check name, offending node, expected-vs-actual) to the Tier-0 author, which revises the program and resubmits through `plan_strategy` → `dry_run_strategy` → `execute_strategy`. This loop is bounded by `max_repair_attempts` (policy default below); exceeding it returns a terminal error rather than looping forever. The attempt counter is tracked per repair session, keyed by `(context_id, task)` and recorded on the `PlanRecord` (`repair_attempt`, §13); each rejected resubmission increments it, and a fresh task resets it. **No live LLM call occurs anywhere in the repair loop** — only authoring, type-checking, simulation, and verification, all of which are free of live calls. Repair is how model-authored flexibility is reconciled with strict admission: the model may try any shape, but only an admitted one ever runs.

Exactly 24 checks:

| # | Name | Severity | Rule |
|---|---|---|---|
| 1 | `artifact_exists` | fail | artifact is stored |
| 2 | `artifact_hash` | fail | recomputed hash matches |
| 3 | `template_known` | fail | if the program references a template, that name/version exists (vacuously passes for a freshly authored program) |
| 4 | `slot_schema` | fail | declared slots validate |
| 5 | `input_type_compatible` | fail | context value matches template input schema (structural) |
| 6 | `chain_type_compatible` | fail | adjacent chain output/input schemas compose (structural) |
| 7 | `effects_allowed` | fail | required effects allowed by policy |
| 8 | `context_exists` | fail | context-ref slots resolve |
| 9 | `model_aliases_resolve` | fail | model aliases exist |
| 10 | `primitive_allowlist` | fail | body uses only §4.0 grammar forms and §4.1 allow-listed operators; no free variables |
| 11 | `termination_bound` | fail | recursion satisfies the §4.0 termination discipline (structural descent via `split`/`context-items`); `iterate` nodes have a finite `max_iterations` (§5.2) |
| 12 | `call_count_limit` | fail | `calls_high` (simulated calls + escalation worst case) <= policy |
| 13 | `critical_path_limit` | warn | critical path <= policy |
| 14 | `concurrency_limit` | fail | per-model and global max concurrency <= limits (§5.7) |
| 15 | `token_budget` | fail | estimated tokens <= policy |
| 16 | `cost_budget` | fail | `cost_usd_high` (incl. escalation worst case) <= policy |
| 17 | `recursion_depth_limit` | fail | simulated depth <= policy |
| 18 | `dry_run_fresh` | fail | dry run artifact matches plan |
| 19 | `output_schema_valid` | fail | declared boundary schema is valid |
| 20 | `py_exec_policy` | fail | py-exec requires policy allow |
| 21 | `total_language_closed` | fail | program is closed under the §4.0 total language: only allow-listed operators, no `eval`/quote-of-code/dynamic-operator or other arbitrary-code construct; `py-exec` only as the policy-gated effect (check 20) |
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
max_repair_attempts = 5                  # bounded verify -> repair -> re-verify loop (§9)
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
| `plan_strategy` | Accept a model-authored combinator program (or a skeleton name plus edits), type-check it, fill the numeric safety parameters, and return a typed `PlanRecord` or structured errors. |
| `dry_run_strategy` | Simulate plan and estimate resources. |
| `execute_strategy` | Verify and execute live. |
| `resume_execution` | Resume a live gate. |
| `get_execution_trace` | Fetch execution trace. |
| `list_templates` | List the optional skeleton / standard-library templates. |
| `describe_template` | Describe one template. |
| `get_record` | Fetch stored record by ID. |
| `reset` | Clear state by scope. |

The repair loop (§9) needs no extra tool: a failing `plan_strategy`, `dry_run_strategy`, or `execute_strategy` returns `{"status":"error", ...}` with the offending checks, and the Tier-0 author resubmits a revised program through the same three tools, up to `max_repair_attempts`.

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
^(ctx|prog|plan|dry|exec|art|ver|call|ckpt|gate)_[0-9a-f]{16}$
```

`prog_...` identifies a `programs` entry (the authored S-expression source); `PlanRecord.program_ref` holds a `prog_...` id.

Store namespaces:

- `contexts`
- `programs` — the authored combinator source (S-expression) referenced by `PlanRecord.program_ref`; the program text the runtime loads into the `run` message (§10.2)
- `plans`
- `artifacts`
- `dry_runs`
- `verifications`
- `executions`
- `cache`
- `checkpoints`
- `traces`

The authored program is persisted to `programs` at `plan_strategy` time and referenced by `program_ref`; the content-addressed `ArtifactRecord` (program hash + slot values + boundary schemas + effects) is created at `dry_run_strategy` time (§6.4) and is what verification and execution bind to.

Gates are in-memory only and are not resumable after process death: `resume_execution` (§11) succeeds only while the original host process and its suspended execution are still alive. `checkpoints` (persisted) are the durable recovery mechanism across process restarts.

---

## 13. Python Interfaces

Core records:

```python
class ModelRegistryEntry(BaseModel):
    alias: str                          # author/host-facing name, e.g. "quality_text_model"
    provider: str
    model_id: str
    backend: str = "remote"             # "remote" | "openai_compatible" | "ollama" | "vllm" | "llamacpp" (§1.4, Batch 5)
    endpoint: str | None = None         # base URL for a locally-served model; None = provider default
    context_window_tokens: int          # raw window W
    reliable_input_tokens: int          # reliability knee K; K <= W (§5.5)
    input_rate: float                   # USD per prompt token (0.0 for a local model)
    output_rate: float                  # USD per completion token (0.0 for a local model)
    max_output_tokens: int              # caps the high cost estimate (§5.4)
    default_completion_estimate: int    # used when a leaf sets no max_tokens (§5.4)
    max_concurrency: int = 1            # effective parallel in-flight calls; 1 for a single-GPU local model (§5.7)
    prefill_tokens_per_sec: float       # prompt-ingest rate for the wall-clock estimate (§5.7)
    decode_tokens_per_sec: float        # completion-generation rate for the wall-clock estimate (§5.7)
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
    program_ref: str                    # -> `programs` namespace: the authored S-expression source (§12)
    schema_flow: list[tuple[dict[str, Any], dict[str, Any]]]
    planner_parameters: dict[str, Any]  # host-fixed numeric params (k, tau, per-node model); see §7.0 note on "planner"
    repair_attempt: int = 0             # repair-loop counter for (context_id, task); capped at max_repair_attempts (§9)
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

### Batch 1: S-Expressions, Program Grammar, and Template Registry

Files:

- `rlm_scheme/sexpr.py`
- `rlm_scheme/grammar.py`
- `rlm_scheme/template_store.py`
- `templates/*.rkt`

Acceptance:

- no regex for S-expression parsing or body analysis;
- §4.0 grammar acceptance/rejection tests (legal forms parse; `eval`/computed-operator/free-variable programs are rejected);
- termination-discipline tests: a `fix` descending on a `split`/`context-items` sub-part is accepted; recursion on `x` itself or on a non-derived value is rejected;
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

### Batch 3: Program Validation and Costing

Files:

- `rlm_scheme/planner.py`
- `rlm_scheme/cost.py`

Acceptance:

- accept a model-authored combinator program and type-check its boundaries;
- `tau` clamped to the leaf model's `K` regardless of any author-supplied value;
- split-factor bounded search (host-chosen or author-proposed-and-validated);
- schema-flow construction;
- host suggests a skeleton from task kind and statistics;
- a failing program returns structured per-check errors suitable for a repair attempt;
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
- combinator evaluation of §4.0-grammar programs (interpreted as an AST; no host-`eval` path);
- recursive executor terminates under verified bounds;
- sandbox escape tests fail (no arbitrary code outside policy-gated `py-exec`);
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
- local backend provider path (an OpenAI-compatible / Ollama endpoint from `ModelRegistryEntry.backend`+`endpoint`) selectable per model alias;
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
- all 24 verification checks, including `total_language_closed` (21) rejecting an arbitrary-code escape and `termination_bound` (11) rejecting non-decreasing recursion;
- bounded repair loop: a failing program returns structured per-check errors and increments `repair_attempt`; exceeding `max_repair_attempts` returns a terminal error;
- no live call anywhere before pass verification (including during repair).

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
13. Symbolic-first composition is the authoring default and the host's suggested skeletons reflect it; symbolic-composed plans collapse LLM calls to the leaf count.
14. Leaf threshold `tau` is sized to the model's reliable-input budget `K`, not its raw window `W`, and is clamped to `K` by the host regardless of any author-supplied or template value.
15. Context bytes are never serialized into the run message or held by the Racket runtime; the runtime obtains content only as bounded slices via `CONTEXT_READ` effects.
16. Per-model concurrency and a single-device-aware `wall_clock_seconds_estimate` are simulated and bounded; a single-GPU leaf model is modeled as serial (`max_concurrency = 1`) with a device-swap penalty across `device_group`.
17. Heterogeneous routing works: a plan can run grunt-work leaves on a local-backend model (`openai_compatible`/`ollama`/`vllm`/`llamacpp`, §13) while reserving an orchestrator-tier cloud model for authoring and `retry_then_escalate` fallbacks — the core economic goal of §0.3.
18. Control flow is model-authored but admitted only through verification: a program in the total combinator language passes §9 before any live call, the `total_language_closed` check (21) rejects any escape to arbitrary code, and a failing program drives a bounded `max_repair_attempts` repair loop with zero live calls.
19. `SPEC-DEVIATIONS.md` is empty or reviewed.

