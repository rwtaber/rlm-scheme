# RLM-Scheme Improvement Plan: Better LLM Reasoning for Orchestration

## Problem Statement

The current RLM-Scheme planner gives an LLM all 17 combinators in a 275-line prompt and says "be creative" (docs/planner-prompt.md, line 6: "Don't follow templates or stick to safe patterns"). This is the wrong interface for LLM code generation. LLMs perform best under tight constraints with clear decision procedures, not open-ended creative mandates.

Three structural problems:

1. **Flat combinator list with no decision structure.** All 17 combinators are presented as equally relevant. For any given task, only 3-5 are sensible. The LLM searches a combinatorial composition space when it should be navigating a decision tree.

2. **"Data characteristics" is free text.** The choice of combinator depends almost entirely on the *shape* of the input data (flat list of independent items vs. hierarchical document vs. single large blob), but this is captured as an unstructured string the planner must interpret.

3. **No feedback loop.** Every planning call starts from zero. The system never learns that `fan-out-aggregate` with `gpt-4.1-nano` at `#:max-concurrent 20` is the right call for "process N independent documents" -- it re-derives this every time.

4. **No structural verification.** Every strategy mistake costs real tokens. There's no way to verify a strategy's structure (call counts, fan-out, depth) before committing to execution.

5. **Decomposition conflated with code generation.** The LLM must simultaneously figure out *how* to break a problem apart AND write *correct nested Scheme* implementing that decomposition. These are different cognitive tasks that fail for different reasons.

---

## Literature: Combinator Upper Bounds and Minimal Basis Sets

### Is there an upper bound on combinators?

Theoretically, no — combinators compose infinitely. Practically, yes — there are strong design constraints from four decades of combinator library research.

**Theoretical minimum (2-3 primitives):**
The SKI combinator calculus (Schönfinkel 1924, Curry 1930) proves that just S, K, and I are computationally universal — any computable function can be expressed. For data processing specifically, Dean & Ghemawat's MapReduce (2004) showed that `map` + `reduce` covers the vast majority of distributed computation. These are existence proofs that very small primitive sets suffice.

**Practical sweet spot (5-8 core + 3-5 modifiers):**
Every successful combinator library converges on a similar size:

| System | Core primitives | Modifiers/wrappers | Total | Domain |
|--------|-----------------|-------------------|-------|--------|
| Apache Beam | ParDo, GroupByKey, CoGroupByKey, Flatten, Partition | Windowing, Triggers | ~7 | Data pipelines |
| Hutton & Meijer parser combinators (1996) | item, return, bind, zero, (++) | many, some, satisfy | ~8 | Parsing |
| DSPy (2024-2026) | Predict, ChainOfThought, ReAct, ProgramOfThought | MultiChainComparison, Retry | ~6 | LLM prompting |
| MapReduce | Map, Reduce | Combine, Partition | ~4 | Distributed compute |
| Haskell Prelude list combinators | map, fold, filter, zip, concat | take, drop, iterate | ~8 | List processing |

**Key design principle from Hughes (1989), "Why Functional Programming Matters":**
> "The ways in which one can divide up the original problem depend directly on the ways in which one can glue solutions together."

The glue functions (higher-order functions + lazy evaluation) are what make a small set of primitives powerful. Hughes argues that **the primitives should be orthogonal** (no two do the same thing) and **composable** (output of one fits input of another). The combinatorial explosion comes from *composition*, not from adding more primitives.

**RLM-Scheme currently has 17 combinators. This is too many.**

Applying the orthogonality test:

| Combinator | Truly primitive? | Or expressible as composition? |
|------------|-----------------|-------------------------------|
| `parallel` | Yes — batch thunk execution | Core |
| `race` | Yes — first-to-complete | Core |
| `map-async` | Yes — parallel map with concurrency control | Core |
| `fan-out-aggregate` | **No** — `map-async` + reduce-fn | Compound: `(reduce-fn (map-async map-fn items))` |
| `tree-reduce` | Yes — hierarchical associative reduction | Core |
| `fold-sequential` | Yes — ordered accumulation | Core |
| `sequence` | Yes — function composition | Core |
| `iterate-until` | Yes — conditional loop | Core |
| `critique-refine` | **No** — `iterate-until` + generate/critique/refine steps | Compound: specialized iterate-until |
| `vote` | **Borderline** — parallel + majority selection | Could be `parallel` + py-exec majority |
| `ensemble` | **No** — `parallel` + custom aggregator | Compound: `(agg (parallel strategies))` |
| `tiered` | **No** — sequential map + expensive synthesis | Compound: `(expensive-fn (map cheap-fn items))` |
| `active-learning` | **Borderline** — map + filter + selective re-map | Specialized pattern |
| `memoized` | Yes — caching wrapper | Modifier |
| `with-validation` | Yes — assertion wrapper | Modifier |
| `try-fallback` | Yes — error recovery wrapper | Modifier |
| `choose` | Yes — conditional dispatch | Core |
| `recursive-spawn` | Yes — nested sandbox delegation | Core |

**Minimal core: 10 primitives**
- `parallel`, `race`, `map-async` (parallel execution)
- `tree-reduce`, `fold-sequential` (reduction)
- `sequence`, `choose` (control flow)
- `iterate-until` (looping)
- `recursive-spawn` (delegation)
- `memoized`, `with-validation`, `try-fallback` (modifiers)

**Derived/compound: 7 combinators that should be templates, not primitives**
- `fan-out-aggregate` = `map-async` + reduction function
- `critique-refine` = specialized `iterate-until`
- `ensemble` = `parallel` + aggregation
- `vote` = `parallel` + majority selection
- `tiered` = sequential map + expensive synthesis
- `active-learning` = map + filter + selective expensive re-map

This doesn't mean removing the 7 compound combinators from the library — they're useful shorthands. But it means the LLM should understand them as **named compositions of primitives**, not as 17 independent concepts to choose from. The planner prompt should present the 10 primitives and show how the 7 compounds are built from them.

**Composition depth bound:**
No formal bound exists in theory. In practice, the literature suggests:
- **Parser combinators**: typically 3-5 levels of nesting (Hutton & Meijer 1996)
- **Apache Beam**: pipelines are typically 4-8 transforms deep (Google Dataflow docs)
- **MapReduce**: canonically 1 level (map then reduce), with chained jobs for multi-stage

For LLM orchestration specifically, each nesting level adds:
- One sequential dependency in the critical path
- One more level of error propagation to handle
- One more level of indirection for debugging

**Practical recommendation: cap at 3 levels of combinator nesting.**
- Level 0: single `llm-query` (Direct shape)
- Level 1: one combinator (e.g., `map-async` or `iterate-until`)
- Level 2: combinator + modifier (e.g., `fan-out-aggregate` with `try-fallback` wrapped map-fn)
- Level 3: two combinators + modifier (e.g., `sequence(fan-out-aggregate, critique-refine)`)

Beyond level 3, use `recursive-spawn` to delegate to a sub-sandbox — that's the principled way to add depth without nesting.

### Relevant references

- Schönfinkel (1924), "On the building blocks of mathematical logic" — SKI combinator basis
- Hughes (1989), ["Why Functional Programming Matters"](https://www.cs.auckland.ac.nz/~j-hamer/360/why-fp-matters.html) — orthogonal primitives + composition as glue
- Hutton & Meijer (1996), ["Monadic Parser Combinators"](https://people.cs.nott.ac.uk/pszgmh/monparsing.pdf) — ~8 primitives compose into arbitrary parsers
- Dean & Ghemawat (2004), ["MapReduce"](https://research.google.com/archive/mapreduce-osdi04.pdf) — map + reduce covers most distributed computation
- Khattab et al. (2024-2026), [DSPy](https://dspy.ai/) — ~6 LLM prompting primitives (Predict, ChainOfThought, ReAct, ProgramOfThought)
- Willis et al. (2021), ["Design Patterns for Parser Combinators"](https://dl.acm.org/doi/10.1145/3471874.3472984) — formalizes combinator design patterns
- Apache Beam, ["Programming Model"](https://cloud.google.com/dataflow/docs/concepts/beam-programming-model) — ~5-7 primitive transforms for data pipelines
- Gibbons (2006), "Design Patterns as Higher-Order Datatype-Generic Programs" — design patterns are combinator compositions
- [AGORA (2025)](https://arxiv.org/html/2604.11378v1), "From Agent Loops to Structured Graphs" — graph-based LLM agent orchestration, shows simpler methods often outperform complex graphs

---

## Critique Response and Additional Gaps

An external review of this plan identified several issues. Each is addressed below with the planned fix.

### 1. Dry-run semantics: pre-resolved futures break await-any (VALID)

**Critique:** Pre-resolved futures change control flow in `await-any`'s rolling-window execution. When all futures are already done, `await-any` returns the first and the remaining handles are reconstructed from `not_done` — but `not_done` is empty if everything resolved instantly. This undercounts calls in pipelined map-async executions where `items > max-concurrent`.

**Verification:** Confirmed. `map-async` (racket_server.rkt lines 857-912) uses `await-any` for rolling windows. The Python-side `await-any` (mcp_server.py lines 717-820) uses `concurrent.futures.wait(..., FIRST_COMPLETED)`. If all futures are pre-resolved, `wait()` returns all of them in `done`, not just one. The rolling window scheduler then thinks all slots are free simultaneously, launching all remaining items at once instead of one-at-a-time.

**Fix:** Don't use pre-resolved futures. Use a **simulated scheduler** that resolves futures with controlled timing:

```python
class DryRunScheduler:
    """Simulates async execution timing for structural analysis."""
    def __init__(self):
        self._resolve_delay = 0.001  # 1ms simulated latency
        self._pending_count = 0
        self._max_concurrent_seen = 0
    
    def create_future(self, mock_result: dict) -> concurrent.futures.Future:
        """Create a future that resolves after a small delay."""
        future = concurrent.futures.Future()
        self._pending_count += 1
        self._max_concurrent_seen = max(self._max_concurrent_seen, self._pending_count)
        
        def resolve():
            time.sleep(self._resolve_delay)
            future.set_result(mock_result)
            self._pending_count -= 1
        
        threading.Thread(target=resolve, daemon=True).start()
        return future
```

This preserves the scheduling order of `await-any` while keeping dry-run fast (~1ms per call vs ~2-5s per real call).

### 2. Separate dry-run metrics from real execution accounting (VALID)

**Critique:** The plan registers dry-run calls through `_register_call()` / `_complete_call()`, which contaminates the real `_call_registry`, cumulative token counters, and execution history.

**Fix:** Add an `ExecutionMode` enum and a `DryRunContext` that collects metrics independently:

```python
class ExecutionMode(Enum):
    REAL = "real"
    DRY_RUN = "dry_run"

class DryRunContext:
    """Isolated metrics collector for dry-run execution."""
    def __init__(self):
        self.calls = []
        self.total_sync = 0
        self.total_async = 0
        self.max_concurrent = 0
        self.models = {}
    
    def record_call(self, call_type, model, instruction, depth):
        self.calls.append({...})
        # Update counters without touching _call_registry
```

Pass `ExecutionMode` through `send()`. In dry-run mode, skip `_register_call` / `_complete_call` entirely.

### 3. Type signatures must distinguish return types more precisely (VALID)

**Critique:** The type system uses broad `Item` type but doesn't distinguish between: returns async handle, returns already-awaited string, returns syntax object, returns plain value. Whether a combinator awaits internally or just calls thunks matters.

**Verification:** Confirmed from racket_server.rkt:
- `parallel` calls thunks, does NOT await → returns whatever thunks return
- `race` calls thunks expecting async handles, calls `await-any` → returns unwrapped string
- `vote`/`ensemble` call thunks synchronously → returns whatever thunks return
- `tiered` applies cheap-fn synchronously via `map` (NOT `map-async`) → returns plain value
- `fan-out-aggregate` uses `map-async` (awaits internally) → returns reduce-fn's return value

**Fix:** Revise type signatures to use:
```
SyntaxObject    = opaque wrapped LLM response (from llm-query)
AsyncHandle     = opaque handle (from llm-query-async, MUST be awaited)
String          = plain unwrapped string
[A]             = list of A
Fn<A,B>         = function A -> B
Thunk<A>        = () -> A

;; Combinators that AWAIT internally (return unwrapped values):
map-async         : (Fn<Item, AsyncHandle>, [Item]) -> [String]
fan-out-aggregate : (Fn<Item, AsyncHandle>, Fn<[String], B>, [Item]) -> B
race              : [Thunk<AsyncHandle>] -> String

;; Combinators that DON'T await (return whatever thunks return):
parallel          : [Thunk<A>] -> [A]
vote              : [Thunk<A>] -> A
ensemble          : [Thunk<A>] -> B

;; Combinators that are synchronous (no async at all):
tiered            : (Fn<Item, A>, Fn<[A], B>, [Item]) -> B       ;; NOTE: sequential map, not parallel
fold-sequential   : (Fn<Acc, Item, Acc>, Acc, [Item]) -> Acc
tree-reduce       : (Fn<Item..., Item>, [Item]) -> Item
sequence          : (Fn<A,B>, Fn<B,C>) -> Fn<A,C>
iterate-until     : (Fn<A,A>, Fn<A,Bool>, A) -> A
critique-refine   : (Thunk<A>, Fn<A,A>, Fn<A,A,A>) -> A

;; Wrappers (preserve inner type):
memoized          : Fn<A,B> -> Fn<A,B>
with-validation   : (Fn<A,B>, Fn<B,Bool>) -> Fn<A,B>
try-fallback      : (Fn<A,B>, Fn<A,B>) -> Fn<A,B>
choose            : (Fn<A,Bool>, Fn<A,B>, Fn<A,B>) -> Fn<A,B>
```

Critical documentation note: **`tiered` is sequential** (`map`, not `map-async`). This is a performance-significant distinction that the current plan's decision trees must account for. For parallel tiered processing, use `fan-out-aggregate` with a cheap model in the map-fn.

### 4. Add Direct/SingleCall shape (VALID)

**Critique:** The TaskShape taxonomy says trivial tasks should "just use llm-query directly" but this isn't a first-class planner output. The planner forces trivial tasks into Synthesize, Refine, or Pipeline.

**Fix:** Add `Direct` as shape #0 in the taxonomy. It explicitly recommends **no combinator** — just a single `llm-query` or `llm-query` + `py-exec`. The classifier should detect this first (before checking other shapes) based on: single input, simple task, no iteration needed.

Decision rule:
```
Is input small enough for one context window (<32K tokens)?
  AND is the task a single operation (not multi-step)?
  AND is there only one input (not a list of items)?
    → Direct. No orchestration needed. Single llm-query.
```

### 5. Don't replace the decision tree with another soft classifier (VALID)

**Critique:** The plan describes a deterministic decision tree but implements a single LLM classification prompt. That's still a soft classifier.

**Fix:** Implement the actual decision tree as deterministic code. Use LLM classification ONLY to fill fields that can't be parsed from structured input:

```python
def _classify_task(task_description, data_characteristics, structured_fields=None):
    """Classify task using deterministic rules first, LLM only for gaps."""
    
    # Phase 1: Parse structured fields if provided
    if structured_fields:
        item_count = structured_fields.get("item_count")
        independent = structured_fields.get("independent")
        output_type = structured_fields.get("output_type")  # "one" or "list"
        # ... deterministic tree from these fields ...
    
    # Phase 2: Use LLM only to fill missing fields
    missing = [f for f in ["item_count", "independent", "output_type"] if f not in (structured_fields or {})]
    if missing:
        llm_result = _fill_missing_fields(task_description, data_characteristics, missing)
        structured_fields = {**(structured_fields or {}), **llm_result}
    
    # Phase 3: Deterministic tree on complete fields
    return _deterministic_classify(structured_fields)
```

This means: `plan_strategy` gains optional structured parameters (`item_count`, `independent`, `output_type`, `modality`). If provided, classification is pure code. If missing, the LLM fills gaps, then classification is still pure code.

### 6. Machine-readable templates (VALID)

**Critique:** Markdown-first templates with string-based `_fill_template()` is fragile. `f'"{value}"'` quoting is unsafe. Should use proper Scheme string escaping and validate slot types.

**Fix:** Store templates as JSON manifest + Scheme body:

```json
{
  "name": "batch-extract-synthesize",
  "shape": "batch",
  "slots": {
    "EXTRACTION_INSTRUCTION": {"type": "string", "required": true},
    "SYNTHESIS_INSTRUCTION": {"type": "string", "required": true},
    "EXTRACT_MODEL": {"type": "model", "default": "gpt-4o-mini", "enum": ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4.5"]},
    "MAX_CONCURRENT": {"type": "int", "default": 20, "min": 1, "max": 50},
    "BRANCH_FACTOR": {"type": "int", "default": 5, "min": 2, "max": 15}
  },
  "code": "(define results\n  (fan-out-aggregate\n    (lambda (item)\n      (llm-query-async #:instruction <<EXTRACTION_INSTRUCTION>>\n                       #:data item\n                       #:model <<EXTRACT_MODEL>>))\n    ...))\n(finish results)",
  "expected_calls_formula": "N + ceil(N / BRANCH_FACTOR)"
}
```

Slot filling uses JSON string encoding for Scheme string literals, validates enum/range constraints, and rejects invalid values before template instantiation.

### 7. History: use plan_id, not global state (VALID)

**Critique:** `_last_plan_context` global is fragile under concurrent MCP tool calls. Use an explicit `plan_id` returned by `plan_strategy` and passed into `execute_scheme`.

**Fix:** `plan_strategy` returns a `plan_id` (UUID) in its `_meta` response. `execute_scheme` gains an optional `plan_id` parameter. History is keyed by `plan_id`, not by global mutable state.

### 8. Progressive summarization: make it an explicit combinator (VALID)

**Critique:** Silently changing `fold-sequential` behavior based on accumulator length is dangerous. It adds LLM calls, changes semantics, and assumes accumulator is a string.

**Fix:** Create `fold-summarizing` as a new combinator (or wrapper):

```scheme
(fold-summarizing fn init items
  #:horizon 8000
  #:summary-model "gpt-4o-mini"
  #:summary-instruction "Compress this running summary, preserving all key facts.")
```

This is explicit, opt-in, and doesn't modify `fold-sequential` behavior.

### 9. Additional gaps identified

**Gap: `recursive-spawn` has dead `max-depth` parameter.**
racket_server.rkt lines 1103-1108: the `#:depth` keyword is accepted but never used. Actual depth enforcement happens globally in Python (`MAX_RECURSION_DEPTH = 3`). Either wire the parameter through or remove it to avoid confusion.

**Gap: No composition depth limit for combinators.**
Only `recursive-spawn` is depth-limited. Arbitrary combinator nesting (e.g., `fan-out-aggregate` inside `critique-refine` inside `sequence` inside `ensemble`) has no enforcement. Recommendation: add a soft warning (not hard limit) when nesting exceeds 3 levels, surfaced in dry-run output.

**Gap: `tiered` is sequential, decision trees assume parallel.**
The per-shape decision tree for Batch recommends `tiered` for cost optimization, but `tiered` uses synchronous `map` (not `map-async`). For 500+ items, this is dramatically slower. Either: (a) document this clearly, (b) add a `tiered-async` variant that uses `map-async` for the cheap phase, or (c) recommend `fan-out-aggregate` with cheap model in map-fn as the "tiered parallel" pattern.

**Gap: Error propagation model.**
No discussion of what happens when a sub-call fails in a deeply nested composition. `try-fallback` handles single-function errors, but what about: a rate limit in the middle of `fan-out-aggregate`? A timeout in one branch of `parallel`? Do partial results survive? The plan should specify error semantics for each combinator.

### Revised implementation roadmap (per critique)

The critique's suggested reorder is better than the original:

1. **Fix docs/code consistency** — correct model names, type signatures, async/sync distinction, dead parameters
2. **Dry-run with correct async scheduler** — simulated scheduler, isolated DryRunContext
3. **verify_strategy using dry-run + deterministic lints** — structural checks, no LLM needed for basic verification
4. **Machine-readable templates** for top 4 cases: Direct, Batch extract, Batch extract+synthesize, Refine
5. **TaskShape/DataShape classification** with deterministic rules + LLM gap-filling
6. **Strategy replay** only after execution metadata is reliable

---

## Taxonomy: TaskShape

Tasks the framework handles, each mapping to a canonical set of combinators.

### Current shapes (7 + 1 missing core)

| Shape | Description | Primary Combinators |
|-------|-------------|-------------------|
| **Direct** | Single operation on single input. No orchestration needed. | llm-query (no combinator) |
| **Batch** | Apply same operation to many independent items | fan-out-aggregate, map-async, tiered, active-learning, memoized |
| **Synthesize** | Combine/summarize many inputs into one output | tree-reduce, fold-sequential, ensemble |
| **Search** | Explore solution space, find best answer | race, iterate-until, vote |
| **Refine** | Iteratively improve a single artifact | critique-refine, iterate-until, with-validation |
| **Compare** | Evaluate alternatives against criteria | vote, ensemble, parallel |
| **Classify** | Categorize/label items | tiered, active-learning, choose, fan-out-aggregate |
| **Pipeline** | Multi-stage sequential transformation | sequence, with-validation, try-fallback |

### Per-shape combinator selection (internal decision trees)

The TaskShape determines the *candidate* combinators. These internal decision trees determine the *specific* combinator within a shape. Each tree produces a single concrete strategy — no LLM judgment needed for structural decisions.

#### Batch

```
Q1: Do results need to be COMBINED into one output, or returned as a LIST?
    COMBINED → Q2
    LIST     → Q3

Q2 (combined output):
    Is combination order-sensitive? (e.g., chronological narrative)
        YES → fan-out-aggregate with fold-sequential as reduce-fn
        NO  → fan-out-aggregate with tree-reduce as reduce-fn
              (set branch-factor = 5 for <=500 items, 10 for >500)

Q3 (list output):
    Is quality uniform across items, or are some items harder?
        UNIFORM  → map-async (plain parallel, all same model)
        VARIABLE → active-learning
                   (cheap model on all, expensive model re-processes items
                    where cheap model's confidence is low)

Q4 (always, modifier): Is cost a primary concern AND items > 100?
    YES → Two options depending on latency tolerance:
          LATENCY OK → tiered: cheap-fn applied SEQUENTIALLY to all items,
                       then expensive-fn for synthesis. Simple but slow for large N.
          LATENCY CRITICAL → fan-out-aggregate with cheap model in map-fn,
                             expensive model in reduce-fn. Parallel but same cost profile.
    NO  → use single model tier

Q5 (always, modifier): Could items be duplicated or near-duplicated?
    YES → wrap inner fn with memoized (hash on first 200 chars)
    NO  → skip

Result combinator:    fan-out-aggregate | map-async | active-learning
Result modifiers:     + tiered wrapper | + memoized wrapper
Result reduce-fn:     tree-reduce (parallel) | fold-sequential (ordered)
```

**Concrete examples:**
- "Extract entities from 500 papers, synthesize findings" → fan-out-aggregate + tree-reduce, branch-factor=10
- "Translate 200 documents, return list" → map-async
- "Grade 300 essays, some are ambiguous" → active-learning (cheap pass, expensive on uncertain)
- "Extract from 2000 docs, cost matters" → tiered (nano extract, gpt-4o-mini synth) + memoized

#### Synthesize

```
Q1: How many source items?
    2-5 items   → single llm-query with all items concatenated in data
                   (no combinator needed, fits in one context window)
    6-50 items  → tree-reduce, branch-factor = min(items, 5)
    50+ items   → tree-reduce, branch-factor = 5-10

Q2: Is synthesis order-sensitive? (e.g., maintaining narrative arc)
    YES → fold-sequential (items processed one at a time, accumulating)
    NO  → tree-reduce (parallel, log-depth, faster)

Q3: Do you want multiple PERSPECTIVES on the same sources?
    YES → ensemble (run 2-3 models, aggregate their syntheses)
    NO  → single-model tree-reduce or fold-sequential

Result combinator:    tree-reduce | fold-sequential | ensemble
Result parameter:     branch-factor (for tree-reduce) | model-list (for ensemble)
```

#### Search

```
Q1: Is the solution space ENUMERABLE (finite candidates) or OPEN-ENDED?
    ENUMERABLE → Q2
    OPEN-ENDED → Q3

Q2 (finite candidates):
    How many candidates?
        2-5   → parallel + vote (run all, pick winner)
        5-20  → parallel + vote with plurality
        20+   → tournament bracket: tree-reduce with pairwise comparison

Q3 (open-ended search):
    Is there a quality signal you can check programmatically?
        YES → iterate-until (generate, check, repeat until passing)
        NO  → iterate-until with LLM-as-judge predicate
              (generate, critique, check if critique says "acceptable")

Q4: Is latency critical? (need answer ASAP, quality secondary)
    YES → race (launch multiple approaches, first to finish wins)
    NO  → parallel + vote (launch all, pick best)

Result combinator:    vote | iterate-until | race
```

#### Refine

```
Q1: Do you have EXPLICIT quality criteria (rubric, checklist)?
    YES → critique-refine
          (critique-fn checks against criteria, refine-fn addresses gaps)
          Set max-iter based on criteria count: 2 for simple, 3-4 for complex
    NO  → Q2

Q2: Is the refinement goal CONVERGENT (approaching a known standard)
    or EXPLORATORY (trying to make it "better" without clear target)?
    CONVERGENT → iterate-until with a testable predicate
                 (e.g., "contains all required sections", JSON validates)
    EXPLORATORY → critique-refine with open-ended critique
                  max-iter = 2-3 (diminishing returns beyond this)

Q3 (always, modifier): Should each iteration be validated before continuing?
    YES → wrap refine step with with-validation
          (if validation fails, retry the refine step, not the whole loop)
    NO  → skip

Result combinator:    critique-refine | iterate-until
Result modifier:      + with-validation wrapper
Result parameter:     max-iter (2-4)
```

#### Compare

```
Q1: Are you comparing STRATEGIES (different approaches to same task)
    or MODELS (same approach, different models)?
    STRATEGIES → parallel + vote
                 (run each strategy, vote on results)
    MODELS     → ensemble
                 (same prompt to multiple models, aggregate)

Q2: How should the winner be determined?
    MAJORITY (>50% agree)   → vote #:method 'majority (need odd number, 3+)
    PLURALITY (most votes)  → vote #:method 'plurality (any number)
    CONSENSUS (all agree)   → vote #:method 'consensus (strict, may fail)
    SYNTHESIS (combine all) → ensemble with custom aggregator
                              (LLM synthesizes all responses into one)

Q3: How many alternatives?
    2   → simple parallel, no voting needed (just compare and pick)
    3-5 → parallel + vote
    5+  → parallel (capped at max-concurrent) + tournament or weighted vote

Result combinator:    vote | ensemble | parallel
Result parameter:     voting method | aggregator function | model list
```

#### Classify

```
Q1: How many items to classify?
    1 item     → single llm-query (no combinator needed)
    2-50 items → map-async (parallel, uniform)
    50+ items  → Q2

Q2: Are categories clear-cut or ambiguous?
    CLEAR    → fan-out-aggregate with cheap model
               (nano/mini can handle clear classification)
    AMBIGUOUS → active-learning
                (cheap model on all, expensive re-classifies uncertain ones)

Q3: Do you need per-item labels or aggregate distribution?
    PER-ITEM LABELS      → map-async (return list)
    AGGREGATE DISTRIBUTION → fan-out-aggregate + py-exec
                             (extract per-item, then Python counts/aggregates)

Q4 (modifier): Is misclassification costly?
    YES → add with-validation wrapper or vote (classify with 2-3 models, majority)
    NO  → single-model classification

Result combinator:    fan-out-aggregate | map-async | active-learning
Result modifier:      + vote wrapper (if misclassification costly)
```

#### Pipeline

```
Q1: How many stages?
    2 stages → sequence(stage1, stage2) — simple chain
    3+ stages → sequence(stage1, stage2, stage3, ...)

Q2 (per stage): Could this stage fail?
    YES → wrap stage with try-fallback
          (primary approach + simpler fallback)
    NO  → unwrapped stage

Q3 (per stage): Does this stage need quality assurance?
    YES → wrap stage with with-validation
          (check output before passing to next stage)
    NO  → unwrapped stage

Q4: Are stages the same operation on different data, or distinct operations?
    SAME OPERATION → this is actually Batch, not Pipeline. Re-classify.
    DISTINCT OPS   → true Pipeline, sequence is correct.

Result combinator:    sequence
Result modifiers:     + try-fallback per stage | + with-validation per stage
```

#### Generate

```
Q1: Do you need a FIXED NUMBER of outputs or generate UNTIL a condition?
    FIXED NUMBER → fan-out-aggregate over (range 1 N)
                   (create a dummy index list, map generation over it)
    UNTIL CONDITION → iterate-until
                      (generate batch, check condition, repeat)

Q2 (fixed number): Do generated items need to be UNIQUE/diverse?
    YES → parallel generation + deduplication via memoized or py-exec
    NO  → plain fan-out-aggregate over index list

Q3 (fixed number): Do items need to be CONSISTENT with each other?
    YES → fold-sequential (each generation sees prior items as context)
    NO  → parallel fan-out-aggregate

Result combinator:    fan-out-aggregate (over index list) | iterate-until | fold-sequential
```

#### Decompose

```
Q1: Is the decomposition structure KNOWN in advance (e.g., chapters, sections)?
    YES → py-exec to split (regex/structural parsing), return list
          (no LLM needed for decomposition itself)
    NO  → Q2

Q2: Can the LLM identify the structure in ONE pass?
    YES → single llm-query with #:json #t
          (ask LLM to return structured breakdown as JSON)
          then py-exec to parse and separate
    NO  → recursive-spawn
          (LLM breaks into coarse parts, then recurses on each part)

Q3: After decomposition, do parts need individual processing?
    YES → Decompose becomes first stage of a Composite:
          Decompose → Batch (process each part)
    NO  → return the decomposed parts directly

Result combinator:    py-exec (structural) | llm-query + py-exec | recursive-spawn
Result composition:   often Decompose → Batch (Composite pattern)
```

#### Validate/Audit

```
Q1: How many items to validate?
    1 item   → single llm-query (structured assessment prompt)
    2+ items → Q2

Q2: Is validation criteria FIXED (same rubric for all items)?
    YES → fan-out-aggregate
          map-fn: validate each item against rubric
          reduce-fn: aggregate pass/fail counts + summarize failures
    NO  → fold-sequential (criteria evolve based on what you've seen)

Q3: Is false-positive or false-negative more costly?
    FALSE POSITIVE costly → tiered: cheap screen first, expensive review on "pass"
    FALSE NEGATIVE costly → tiered: cheap screen first, expensive review on "fail"
    EQUAL                 → uniform model, no tiering

Q4 (modifier): Do you need structured output (JSON with score, reasoning)?
    YES → add #:json #t to llm-query, with-validation to verify JSON structure
    NO  → prose assessment

Result combinator:    fan-out-aggregate | fold-sequential
Result modifier:      + tiered | + with-validation (for JSON output)
```

#### Aggregate/Report

```
Q1: Is aggregation purely COMPUTATIONAL (counts, averages, distributions)?
    YES → fan-out-aggregate for LLM extraction + py-exec for computation
          map-fn: LLM extracts structured data from each item (#:json #t)
          reduce-fn: py-exec computes statistics, not LLM synthesis
    NO  → Q2

Q2: Does aggregation require LLM INTERPRETATION (trends, insights)?
    YES → Two-phase:
          Phase 1: fan-out-aggregate to extract structured data
          Phase 2: py-exec to compute stats
          Phase 3: single llm-query to interpret stats
    NO  → fan-out-aggregate + py-exec only

Q3: Are there GROUPING dimensions (aggregate per-category)?
    YES → py-exec handles groupby after extraction
    NO  → single aggregation pass

Result combinator:    fan-out-aggregate + py-exec
Result composition:   often a Pipeline: extract → compute → interpret
```

#### Composite

```
Q1: Identify the constituent shapes in order.
    Common patterns:
    - Batch → Synthesize (extract from many, combine into one)
    - Decompose → Batch (break apart, process each piece)
    - Batch → Aggregate (extract from many, compute statistics)
    - Generate → Validate (create items, check each one)
    - Classify → Aggregate (label items, count distribution)
    - Batch → Refine (extract, then iteratively improve the combined result)

Q2: Connect phases with sequence.
    Each phase uses its own shape's decision tree to select combinators.
    The output of phase N becomes the input of phase N+1.

Q3: Are phases INDEPENDENT (can run in parallel) or DEPENDENT (sequential)?
    INDEPENDENT → parallel([phase1, phase2]) then combine
    DEPENDENT   → sequence(phase1, phase2) — most common

Result combinator:    sequence(shape1_combinator, shape2_combinator)
```

### Combinator selection is a TWO-LEVEL decision

To summarize: combinator selection is NOT a single mapping from shape to combinator. It's a two-level process:

**Level 1: TaskShape + DataShape → Structural combinator**
- TaskShape determines the *kind* of operation (map, reduce, iterate, compare)
- DataShape determines the *structural parameters* (concurrency, branch-factor, sequential vs parallel)
- This level is **deterministic** — can be implemented as code, no LLM needed

**Level 2: Task properties → Combinator modifiers and parameters**
- Cost sensitivity → tiered / active-learning wrappers
- Quality requirements → with-validation / vote / critique-refine wrappers
- Output format → #:json, py-exec aggregation
- Error tolerance → try-fallback wrappers
- This level requires **LLM judgment** for some decisions (how costly is misclassification? is combination order-sensitive?)

The system should handle Level 1 mechanically. The LLM should only make Level 2 decisions, which are qualitative judgments expressible as simple yes/no or multiple-choice answers — not code generation.

### Missing shapes

| Shape | Description | Why missing matters | Primary Combinators |
|-------|-------------|-------------------|-------------------|
| **Generate** | Create new content from scratch (not transforming existing data). "Write 50 product descriptions," "Generate test cases for this API." | Currently forced into Batch, but there's no input list to map over -- the LLM must create items, not transform them. Needs a different template: generate-N-items loop, not map-over-existing. | iterate-until, parallel, fan-out-aggregate (with index list) |
| **Decompose** | Break one complex thing into structured parts. "Parse this document into sections," "Extract a schema from unstructured text." Inverse of Synthesize. | Currently forced into Refine or Pipeline, but the goal is producing *multiple* outputs from *one* input, not improving a single artifact. | llm-query + py-exec (structural parsing), recursive-spawn |
| **Validate/Audit** | Check items against criteria, produce pass/fail + explanation. "Check these 100 configs for security issues," "Grade these 50 essays." | Overlaps with Classify but the output shape is fundamentally different: not a category label but a structured assessment (pass/fail, score, explanation, evidence). | fan-out-aggregate, with-validation, tiered |
| **Aggregate/Report** | Gather structured metrics across items, not prose synthesis. "Count entity frequencies," "Compute average sentiment per category." | Synthesize assumes prose output. Aggregate/Report needs structured/numeric output, often requiring Python computation between LLM calls. | fan-out-aggregate + py-exec, fold-sequential + py-exec |
| **Composite** | Multi-shape tasks: Batch->Synthesize, Classify->Aggregate, Generate->Validate. Most real tasks are compositions. | The planner currently picks ONE shape. Many tasks are "extract from each document THEN synthesize," which is Batch->Synthesize. Needs first-class support for shape composition. | Depends on constituent shapes |

### TaskShape decision tree

Instead of the LLM choosing from a flat list of 12 shapes, walk a binary decision tree:

```
Q1: Do you have MANY input items to process?
    YES -> Q2: Are items independent (order doesn't matter)?
               YES -> Q3: What's the per-item operation?
                          Transform/Extract -> Batch
                          Label/Categorize  -> Classify
                          Check/Grade       -> Validate
               NO  -> Q4: Does information accumulate across items?
                          YES -> Fold (fold-sequential)
                          NO  -> Pipeline (sequence)
    NO  -> Q5: Do you have ONE input to work with?
               YES -> Q6: Goal is to improve it or break it apart?
                          Improve    -> Refine
                          Break apart -> Decompose
                          Summarize  -> (single-item, just use llm-query directly)
               NO  -> Q7: Are you creating content from nothing?
                          YES -> Generate
                          NO  -> Q8: Are you choosing between approaches?
                                     YES -> Compare / Search
                                     NO  -> Synthesize (combining existing sources)

Q9 (always): Does this task have a SECOND phase?
    YES -> Composite (re-run tree for phase 2, connect with sequence)
    NO  -> Single-shape strategy
```

This is ~9 binary questions. LLMs answer yes/no with near-perfect accuracy. The system assembles the strategy.

---

## Taxonomy: DataShape

The shape of input data determines combinator selection mechanically.

### Current shapes (5)

| Shape | Description | Key Properties |
|-------|-------------|---------------|
| **FlatList** | List of independent items | count, item_size, independent: bool |
| **Hierarchy** | Tree-structured data | depth, branching, node_count |
| **Singular** | One large blob of text/data | size, chunkable: bool, chunk_boundary |
| **Graph** | Connected entities with relationships | nodes, edges, connected: bool |
| **TimeSeries** | Sequentially ordered data points | length, window_size, causal: bool |

### Missing shapes

| Shape | Description | Why it matters for combinator selection |
|-------|-------------|---------------------------------------|
| **Tabular** | CSV, database rows, structured records with schema/columns. | Very common. Different from FlatList because items share a schema, enabling column-level operations (extract column X from all rows) and aggregation (group by column Y). Affects prompt construction. |
| **Multimodal** | Images + text, or audio + text. | The framework supports `#:images` but DataShape doesn't reflect this. Multimodal data changes model selection (needs vision-capable models), cost estimates (image tokens are expensive), and chunking strategy. |
| **Paired/Aligned** | Two parallel lists where items correspond (source/target for translation, before/after for comparison). | Different from FlatList because each map operation needs TWO inputs. Affects lambda signature in fan-out-aggregate. |
| **Key-Value/Map** | Dictionary-like data accessed by key, not index. | Affects how data is loaded, how results are stored (preserve keys), and how the LLM references specific items. |
| **ChunkedSingular** | A large document that must be chunked where chunks are NOT independent (context flows between them). | Different from both Singular (too big for one call) and FlatList (chunks aren't independent). Requires fold-sequential or sliding-window, NOT fan-out-aggregate. Critical distinction -- using parallel processing on contextually-dependent chunks loses information. |

### DataShape -> Combinator mapping rules

These should be deterministic (the system selects, not the LLM):

```
FlatList { independent: true, count > 50 }
  -> fan-out-aggregate, max-concurrent = min(count, 20)

FlatList { independent: true, count <= 50 }
  -> map-async, max-concurrent = count

FlatList { independent: false }
  -> fold-sequential (order matters)

Singular { size > 32K, chunkable: true }
  -> chunk at boundary, then treat as FlatList or ChunkedSingular

Singular { size <= 32K }
  -> single llm-query (no orchestration needed)

ChunkedSingular
  -> fold-sequential with sliding window (NOT fan-out-aggregate)

Hierarchy { depth > 2 }
  -> recursive-spawn or tree-reduce matching the hierarchy

Tabular { row_count > 50, independent_rows: true }
  -> fan-out-aggregate (row-level), py-exec for column aggregation

Multimodal
  -> model must support vision (#:images parameter); cost estimates use image token pricing

Paired { count > 50 }
  -> fan-out-aggregate with (lambda (pair) ...) over zipped list
```

---

## Improvement 1: Dry Run Mode

**Priority:** Highest. Enables all other structural verification. No dependencies.

**Goal:** Execute orchestration code structurally without real LLM calls. Return call counts, max depth, fan-out, estimated cost/latency. Free (<1s, no API calls).

**Architecture:** Intercept in Python's `RacketREPL.send()` callback loop (mcp_server.py lines 503-583), NOT in Racket. The sandbox sees no difference -- it sends `llm-query` callbacks as usual, but the Python side returns instant mock responses instead of calling OpenAI.

### Changes to mcp_server.py

**New classes (add near line 210):**

The dry-run system uses two key classes identified by critique review:

1. **`DryRunContext`** — isolated metrics collector that does NOT touch `_register_call` / `_complete_call` or any real execution accounting:

```python
class ExecutionMode(Enum):
    REAL = "real"
    DRY_RUN = "dry_run"

class DryRunContext:
    """Isolated metrics collector for dry-run execution.
    
    Critical: does NOT call _register_call/_complete_call, which would
    contaminate the real _call_registry, cumulative token counters, and
    execution history.
    """
    def __init__(self):
        self.calls = []
        self.total_sync = 0
        self.total_async = 0
        self.max_concurrent = 0
        self.models = {}  # model -> count
        self._current_pending = 0
    
    def record_call(self, call_type: str, model: str, instruction: str, depth: int):
        self.calls.append({
            "instruction_preview": instruction[:80],
            "model": model,
            "type": call_type,
            "depth": depth,
        })
        self.models[model] = self.models.get(model, 0) + 1
        if call_type == "sync":
            self.total_sync += 1
        else:
            self.total_async += 1
            self._current_pending += 1
            self.max_concurrent = max(self.max_concurrent, self._current_pending)
    
    def complete_async(self):
        self._current_pending = max(0, self._current_pending - 1)
    
    @property
    def total_calls(self):
        return self.total_sync + self.total_async
```

2. **`DryRunScheduler`** — creates futures with controlled timing instead of pre-resolved futures. Pre-resolved futures break `await-any`'s rolling window (see Critique Response #1):

```python
class DryRunScheduler:
    """Simulates async execution timing for structural analysis.
    
    Why not pre-resolved futures: map-async (racket_server.rkt lines 857-912)
    uses await-any for rolling windows. The Python-side await-any
    (mcp_server.py lines 717-820) uses concurrent.futures.wait(..., FIRST_COMPLETED).
    If all futures are pre-resolved, wait() returns ALL of them in `done`, not
    just one. The rolling window scheduler then thinks all slots are free
    simultaneously, launching all remaining items at once instead of one-at-a-time.
    This undercounts calls in pipelined executions where items > max-concurrent.
    """
    def __init__(self, ctx: DryRunContext):
        self._resolve_delay = 0.001  # 1ms simulated latency
        self._ctx = ctx
    
    def create_future(self, mock_result: dict) -> concurrent.futures.Future:
        """Create a future that resolves after a small delay."""
        future = concurrent.futures.Future()
        
        def resolve():
            time.sleep(self._resolve_delay)
            future.set_result(mock_result)
            self._ctx.complete_async()
        
        threading.Thread(target=resolve, daemon=True).start()
        return future
```

**RacketREPL.__init__ (line 221):** Add instance variables:
```python
self._execution_mode = ExecutionMode.REAL
self._dry_run_ctx = None       # DryRunContext, set only during dry-run
self._dry_run_scheduler = None  # DryRunScheduler, set only during dry-run
```

**RacketREPL.send(), `op == "llm-query"` branch (line 510):** Insert dry-run bypass at the top of the branch, before the existing `_call_llm()` call:
```python
if op == "llm-query":
    if self._execution_mode == ExecutionMode.DRY_RUN:
        instruction = msg.get("instruction", "")
        data = msg.get("data", "")
        model = msg.get("model", "") or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
        mock_hash = hashlib.md5((instruction + data[:100]).encode()).hexdigest()[:8]
        mock_text = f"[DRY-RUN:{mock_hash}] Mock response for: {instruction[:60]}"
        
        # Record in isolated DryRunContext (NOT _register_call/_complete_call)
        self._dry_run_ctx.record_call("sync", model, instruction, self._current_depth)
        
        # Write mock response back to Racket stdin
        self.proc.stdin.write(json.dumps({
            "result": mock_text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }) + "\n")
        self.proc.stdin.flush()
        continue  # skip the real _call_llm()
    
    # ... existing real call code ...
```

**RacketREPL.send(), `op == "llm-query-async"` branch (line 559):** Use DryRunScheduler instead of pre-resolved futures:
```python
elif op == "llm-query-async":
    if self._execution_mode == ExecutionMode.DRY_RUN:
        instruction = msg.get("instruction", "")
        data = msg.get("data", "")
        model = msg.get("model", "") or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
        mock_hash = hashlib.md5((instruction + data[:100]).encode()).hexdigest()[:8]
        mock_text = f"[DRY-RUN:{mock_hash}] Mock async for: {instruction[:60]}"
        
        # Record in isolated DryRunContext
        self._dry_run_ctx.record_call("async", model, instruction, self._current_depth)
        
        # Use DryRunScheduler to create delayed future (NOT pre-resolved).
        # This preserves await-any's rolling window scheduling behavior.
        mock_result = {"text": mock_text, "prompt_tokens": 0, "completion_tokens": 0}
        future = self._dry_run_scheduler.create_future(mock_result)
        self._pending[msg["id"]] = future
        
        # No response to Racket (same as real async -- Racket continues immediately)
        continue
    
    # ... existing real async code ...
```

**New MCP tool `dry_run_scheme` (after line 1471):**
```python
@mcp.tool(description="Simulate orchestration without real LLM calls. Returns structural analysis.")
async def dry_run_scheme(code: str, ctx: Context = None) -> str:
    backend = get_backend()
    
    # Set up isolated dry-run state
    dry_ctx = DryRunContext()
    scheduler = DryRunScheduler(dry_ctx)
    backend._execution_mode = ExecutionMode.DRY_RUN
    backend._dry_run_ctx = dry_ctx
    backend._dry_run_scheduler = scheduler
    # NOTE: do NOT call backend.reset_call_stats() or _call_registry.reset_stats()
    # — the whole point of DryRunContext is to leave real accounting untouched.
    
    loop = asyncio.get_event_loop()
    t_start = time.monotonic()
    
    try:
        resp = await loop.run_in_executor(
            None, lambda: backend.send({"op": "eval", "code": code}, timeout=30)
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        # Always reset to real mode, even on error
        backend._execution_mode = ExecutionMode.REAL
        backend._dry_run_ctx = None
        backend._dry_run_scheduler = None
    
    elapsed = round(time.monotonic() - t_start, 3)
    
    # Estimate cost from model usage
    MODEL_COSTS = {
        "gpt-4.1-nano": 0.001, "gpt-4o-mini": 0.005,
        "gpt-4o": 0.01, "gpt-4.5": 0.03,
    }
    estimated_cost = sum(
        count * MODEL_COSTS.get(model, 0.01)
        for model, count in dry_ctx.models.items()
    )
    
    # Compute max depth from call tree
    max_depth = max((c["depth"] for c in dry_ctx.calls), default=0)
    
    # Nesting depth warning (see Literature section: cap at 3 levels)
    warnings = []
    if max_depth > 3:
        warnings.append(
            f"Nesting depth {max_depth} exceeds recommended maximum of 3. "
            f"Consider using recursive-spawn for additional depth."
        )
    
    return json.dumps({
        "status": resp["status"],
        "value": resp.get("result", ""),
        "dry_run_elapsed": elapsed,
        "structure": {
            "total_calls": dry_ctx.total_calls,
            "sync_calls": dry_ctx.total_sync,
            "async_calls": dry_ctx.total_async,
            "max_fan_out": dry_ctx.max_concurrent,
            "max_depth": max_depth,
            "models_used": dry_ctx.models,
        },
        "estimates": {
            "cost": f"${estimated_cost:.2f}",
            "latency_range": f"{dry_ctx.total_sync * 2}s - {dry_ctx.total_sync * 5}s (sync calls dominate)",
        },
        "warnings": warnings,
        "call_tree": dry_ctx.calls[:50],  # truncate for readability
    }, indent=2)
```

**New file: docs/tool-descriptions/dry_run_scheme.md**

**New file: tests/test_dry_run.py** -- tests:
- Dry-run returns in <1s
- No real API calls made (mock OpenAI client, assert never called)
- Correct call counts for known fan-out-aggregate(10 items) -> expect 10 async + ceil(10/5) reduce
- `_dry_run` flag reset even on error (finally block)
- Existing `execute_scheme` unaffected (no dry-run behavior leaks)

**Known edge case:** `iterate-until` / `critique-refine` predicates check LLM output content. With mocks, predicates may always be false, running max iterations. This is acceptable -- it shows worst-case call count. Document this in dry_run_scheme.md.

---

## Improvement 2: Combinator Type Signatures

**Priority:** High. Documentation-only change. Enables self-checking.

**Goal:** Add input/output types to combinator docs so LLMs can type-check compositions before generating Scheme.

### Changes to docs/combinators.md

Add new section before the combinator listings:

```markdown
## Type System

Types used in combinator signatures:

    SyntaxObject  = opaque wrapped LLM response (from llm-query). MUST unwrap with (syntax-e result) before string operations.
    AsyncHandle   = opaque handle from llm-query-async. NOT a string — must be awaited by a combinator (map-async, race, etc.)
    String        = plain unwrapped string (result of syntax-e, or auto-unwrapped by map-async/fan-out-aggregate)
    [A]           = list of A
    Fn<A, B>      = function from A to B
    Fn<A, B, C>   = function from (A, B) to C
    Thunk<A>      = () -> A (zero-argument function)
    Acc           = accumulator type (typically String)

Key rules:
- llm-query returns SyntaxObject → must unwrap with (syntax-e result)
- llm-query-async returns AsyncHandle → must be awaited by a combinator
- map-async/fan-out-aggregate auto-unwrap: results are plain Strings (no syntax-e needed)
- Whether a combinator awaits internally or just calls thunks matters for type correctness
```

Add type signature to each combinator's `**Signature:**` line. Signatures are grouped by async behavior (verified against racket_server.rkt):

```
;; Combinators that AWAIT internally (return unwrapped values):
map-async         : (Fn<Item, AsyncHandle>, [Item], #:max-concurrent Int) -> [String]
fan-out-aggregate : (Fn<Item, AsyncHandle>, Fn<[String], B>, [Item], #:max-concurrent Int) -> B
race              : [Thunk<AsyncHandle>] -> String

;; Combinators that DON'T await (return whatever thunks return):
parallel          : ([Thunk<A>], #:max-concurrent Int) -> [A]
vote              : ([Thunk<A>], #:method Symbol) -> A
ensemble          : ([Thunk<A>], #:aggregator Fn<[A], B>) -> B

;; Combinators that are synchronous (no async involvement):
tiered            : (Fn<Item, A>, Fn<[A], B>, [Item]) -> B       ;; NOTE: uses sequential map, NOT map-async
fold-sequential   : (Fn<Acc, Item, Acc>, Acc, [Item]) -> Acc
tree-reduce       : (Fn<Item..., Item>, [Item], #:branch-factor Int) -> Item
sequence          : (Fn<A,B>, Fn<B,C>, ...) -> Fn<A, ...last output>
iterate-until     : (Fn<A, A>, Fn<A, Bool>, A, #:max-iter Int) -> A
critique-refine   : (Thunk<Item>, Fn<Item, Item>, Fn<Item, Item, Item>, #:max-iter Int) -> Item

;; Wrappers (preserve inner type):
memoized          : Fn<A, B> -> Fn<A, B>
with-validation   : (Fn<A, B>, Fn<B, Bool>) -> Fn<A, B>
try-fallback      : (Fn<A, B>, Fn<A, B>) -> Fn<A, B>
choose            : (Fn<A, Bool>, Fn<A, B>, Fn<A, B>) -> Fn<A, B>

;; Delegation:
recursive-spawn   : (Thunk<String>, #:depth Int) -> Fn<Item, Item>
;;   NOTE: #:depth keyword is accepted but currently dead (racket_server.rkt lines 1103-1108).
;;   Actual depth enforcement is global in Python (MAX_RECURSION_DEPTH = 3).
```

**Critical distinction: `tiered` is sequential.** It uses `map` (not `map-async`) to apply the cheap function. For 500+ items this is dramatically slower than parallel alternatives. For parallel tiered processing, use `fan-out-aggregate` with a cheap model in the map-fn and an expensive model in the reduce-fn.

Add "Common Type Errors" section:

```markdown
## Common Type Errors

### Using llm-query where llm-query-async is needed
fan-out-aggregate's map-fn must return AsyncHandle, not SyntaxObject.
WRONG:  (lambda (doc) (llm-query #:instruction "..." #:data doc))
RIGHT:  (lambda (doc) (llm-query-async #:instruction "..." #:data doc))

### Forgetting syntax-e on llm-query results
llm-query returns SyntaxObject. String operations (string-append, etc.) fail on SyntaxObject.
WRONG:  (string-append (llm-query #:instruction "..." #:data "...") " more text")
RIGHT:  (string-append (syntax-e (llm-query #:instruction "..." #:data "...")) " more text")

### Passing thunk where one-arg function expected
tree-reduce expects Fn<Item..., Item> (takes multiple args), not Thunk<Item>.
WRONG:  (tree-reduce (lambda () (llm-query ...)) items)
RIGHT:  (tree-reduce (lambda args (llm-query #:data (string-join args))) items)

### race returns Item, not [Item]
race produces one result (the first to complete). Feeding it into tree-reduce (expects [Item]) is a type error.
WRONG:  (tree-reduce combine-fn (race strategies))
RIGHT:  (tree-reduce combine-fn (parallel strategies))  ; parallel returns [Item]

### Composition type-checking rule
For (sequence f g): the output type of f must match the input type of g.
(sequence fan-out-aggregate critique-refine) is WRONG because:
  fan-out-aggregate returns Item (single result)
  critique-refine expects Thunk<Item> (zero-arg function), not Item
Fix: wrap in a thunk: (sequence fan-out-aggregate (lambda (result) (critique-refine (lambda () result) ...)))
```

---

## Improvement 3: Task Classification + Progressive Disclosure

**Priority:** High. Reduces search space by ~70%.

**Goal:** Replace monolithic planner prompt with two-phase flow: cheap classification (gpt-4o-mini, ~$0.001) then focused planning with only the relevant 3-5 combinators shown.

### New file: docs/task-shapes.md

Defines the full TaskShape taxonomy (12 shapes including the 5 new ones), the decision tree, and the DataShape taxonomy with mapping rules.

### New directory: docs/shape-prompts/

One markdown file per shape, each ~80-100 lines (vs 275 for the monolithic prompt). Each includes:
1. Shape description and when it applies
2. Only the 3-5 relevant combinators with type signatures
3. 2-3 focused examples specific to that shape
4. Scale validation checklist
5. Output format (same JSON structure as current)

Files:
- `direct.md` -- no combinator needed; single llm-query or llm-query + py-exec
- `batch.md` -- fan-out-aggregate, map-async, tiered, active-learning, memoized
- `synthesize.md` -- tree-reduce, fold-sequential, ensemble
- `search.md` -- race, vote, iterate-until
- `refine.md` -- critique-refine, iterate-until, with-validation
- `compare.md` -- vote, ensemble, parallel
- `classify.md` -- tiered, active-learning, choose, fan-out-aggregate
- `pipeline.md` -- sequence, with-validation, try-fallback
- `generate.md` -- iterate-until, parallel, fan-out-aggregate (with index list)
- `decompose.md` -- llm-query + py-exec, recursive-spawn
- `validate.md` -- fan-out-aggregate, with-validation, tiered
- `aggregate.md` -- fan-out-aggregate + py-exec, fold-sequential + py-exec
- `composite.md` -- instructions for chaining shapes with sequence

### Changes to mcp_server.py

**New internal function `_classify_task()` (add near line 1285):**

The classification is a three-phase process (see Critique Response #5). The decision tree is implemented as deterministic code, NOT as a soft LLM classifier. The LLM is used ONLY to fill missing structured fields when they can't be parsed from the caller's input.

```python
def _deterministic_classify(fields: dict) -> dict:
    """Pure-code decision tree. No LLM calls.
    
    fields must contain: item_count (int|None), independent (bool|None),
    output_type ("one"|"list"|None), operation (str|None), has_second_phase (bool).
    """
    item_count = fields.get("item_count")
    independent = fields.get("independent")
    output_type = fields.get("output_type")
    operation = fields.get("operation", "")
    has_second_phase = fields.get("has_second_phase", False)
    
    # Q0: Is this trivially simple? (Direct shape)
    if (item_count is not None and item_count <= 1
            and output_type == "one"
            and not has_second_phase):
        return {"shape": "direct", "confidence": 1.0, "reasoning": "single input, single output, no orchestration needed"}
    
    # Q1: Many input items?
    if item_count is not None and item_count > 1:
        if independent:
            # Q3: What's the per-item operation?
            if operation in ("label", "categorize", "classify"):
                shape = "classify"
            elif operation in ("check", "grade", "validate", "audit"):
                shape = "validate"
            else:
                shape = "batch"  # default for independent items: transform/extract
        else:
            # Q4: Does information accumulate?
            if operation in ("accumulate", "fold", "running"):
                shape = "synthesize"  # fold-sequential
            else:
                shape = "pipeline"
    elif item_count == 1 or item_count is None:
        if operation in ("improve", "refine", "edit", "polish"):
            shape = "refine"
        elif operation in ("decompose", "break", "split", "parse"):
            shape = "decompose"
        elif operation in ("generate", "create", "write", "produce"):
            shape = "generate"
        elif operation in ("compare", "choose", "evaluate", "rank"):
            if output_type == "one":
                shape = "search"
            else:
                shape = "compare"
        elif operation in ("aggregate", "count", "statistics", "metric"):
            shape = "aggregate"
        elif output_type == "one":
            shape = "synthesize"
        else:
            shape = "direct"  # fallback for unknown single-input tasks
    else:
        shape = "direct"
    
    # Q9: Second phase?
    if has_second_phase:
        return {"shape": "composite", "confidence": 0.9,
                "reasoning": f"multi-phase task, primary shape: {shape}",
                "primary_shape": shape}
    
    return {"shape": shape, "confidence": 0.9, "reasoning": f"deterministic classification from structured fields"}


_FIELD_EXTRACTION_PROMPT = """Extract structured fields from this task description.

Task: {task_description}
Data: {data_characteristics}

Return ONLY JSON with these fields:
{{
  "item_count": <integer or null if unknown>,
  "independent": <true if items can be processed independently, false if order matters, null if unclear>,
  "output_type": "one" or "list" (does the task produce a single result or a list?),
  "operation": <one word: "extract", "label", "check", "improve", "decompose", "generate", "compare", "aggregate", or other>,
  "has_second_phase": <true if task clearly has two phases like "extract THEN synthesize">
}}"""


def _classify_task(task_description: str, data_characteristics: str,
                   structured_fields: dict | None = None) -> dict:
    """Classify task using deterministic rules first, LLM only for gaps.
    
    Three phases:
    1. Use caller-provided structured_fields if available
    2. Use LLM to fill any missing fields (NOT to classify directly)
    3. Run deterministic decision tree on complete fields
    """
    required_fields = ["item_count", "independent", "output_type", "operation", "has_second_phase"]
    fields = dict(structured_fields or {})
    
    # Phase 1: Check which fields are already provided
    missing = [f for f in required_fields if f not in fields or fields[f] is None]
    
    # Phase 2: Use LLM only to fill missing fields
    if missing:
        backend = get_backend()
        result = backend._call_llm(
            instruction=_FIELD_EXTRACTION_PROMPT.format(
                task_description=task_description,
                data_characteristics=data_characteristics or "Not specified",
            ),
            data="",
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=200,
        )
        try:
            text = result["text"].strip()
            json_match = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
            if json_match:
                text = json_match.group(1).strip()
            llm_fields = json.loads(text)
            # Only fill gaps — caller-provided fields take precedence
            for f in missing:
                if f in llm_fields:
                    fields[f] = llm_fields[f]
        except (json.JSONDecodeError, KeyError):
            pass  # proceed with whatever fields we have
    
    # Phase 3: Deterministic classification
    return _deterministic_classify(fields)
```

This means: `plan_strategy` gains optional structured parameters (`item_count`, `independent`, `output_type`, `operation`). If provided, classification is pure code with zero LLM calls. If missing, the LLM fills gaps (~$0.001), then classification is still pure code.

**Load shape prompts at module level (add near line 210):**
```python
_SHAPE_PROMPTS = {}
_SHAPE_PROMPT_DIR = os.path.join(_DOCS_DIR, "shape-prompts")
if os.path.isdir(_SHAPE_PROMPT_DIR):
    for shape_file in os.listdir(_SHAPE_PROMPT_DIR):
        if shape_file.endswith(".md"):
            shape_name = shape_file[:-3]  # strip .md
            with open(os.path.join(_SHAPE_PROMPT_DIR, shape_file), "r", encoding="utf-8") as f:
                _SHAPE_PROMPTS[shape_name] = f.read()
```

**Modify `plan_strategy` (line 1285):** Add `task_shape` and structured field parameters, two-phase flow with plan_id:
```python
@mcp.tool(description="Design an optimal orchestration strategy for your task.")
def plan_strategy(
    task_description: str,
    data_characteristics: str | None = None,
    constraints: str | None = None,
    priority: str = "balanced",
    scale: str = "medium",
    min_outputs: int | None = None,
    coverage_target: str | None = None,
    task_shape: str | None = None,  # override classification
    # Structured fields for deterministic classification (Critique Response #5):
    item_count: int | None = None,
    independent: bool | None = None,
    output_type: str | None = None,  # "one" or "list"
    operation: str | None = None,    # "extract", "label", "check", etc.
) -> str:
    # Generate unique plan_id for history keying (Critique Response #7)
    plan_id = str(uuid.uuid4())
    
    # Phase 1: Classify task shape
    if task_shape:
        shape = task_shape.lower()
        classification = {"shape": shape, "confidence": 1.0, "reasoning": "user-specified"}
    else:
        # Build structured_fields from caller-provided parameters
        structured_fields = {}
        if item_count is not None: structured_fields["item_count"] = item_count
        if independent is not None: structured_fields["independent"] = independent
        if output_type is not None: structured_fields["output_type"] = output_type
        if operation is not None: structured_fields["operation"] = operation
        
        classification = _classify_task(
            task_description, data_characteristics,
            structured_fields=structured_fields if structured_fields else None,
        )
        shape = classification.get("shape", "unknown")
    
    # Phase 2: Select prompt
    if classification["confidence"] >= 0.6 and shape in _SHAPE_PROMPTS:
        prompt_template = _SHAPE_PROMPTS[shape]
    else:
        # Fall back to monolithic prompt
        prompt_template = _PLANNER_PROMPT_TEMPLATE
    
    # Inject history if available (Improvement 6, keyed by plan_id)
    history_section = _load_relevant_history(task_description, shape)
    
    # Format prompt
    prompt = prompt_template.format(
        task_description=task_description,
        data_characteristics=data_characteristics or "Not specified",
        constraints=constraints or "None specified",
        priority=priority,
        scale=scale,
        min_outputs=min_outputs if min_outputs is not None else "Not specified",
        coverage_target=coverage_target or "Not specified",
        history=history_section,  # empty string if no history
    )
    
    # ... rest of existing plan_strategy (model call, JSON parsing, metadata) ...
    # Add to _meta (plan_id replaces _last_plan_context global):
    parsed["_meta"]["plan_id"] = plan_id
    parsed["_meta"]["task_shape"] = shape
    parsed["_meta"]["classification_confidence"] = classification["confidence"]
    parsed["_meta"]["classification_reasoning"] = classification.get("reasoning", "")
    # Store plan context keyed by plan_id for history recording
    _plan_contexts[plan_id] = {
        "task_description": task_description,
        "task_shape": shape,
        "strategy_name": parsed.get("recommended", {}).get("strategy_name", ""),
        "template_used": parsed.get("recommended", {}).get("template_name", ""),
        "scale": scale,
    }
```

### New file: tests/test_task_classification.py

Tests:
- "Extract key points from 500 papers" -> Batch
- "Iteratively improve this essay" -> Refine
- "Find the best algorithm for this problem" -> Search
- "Extract from 100 docs then write a synthesis" -> Composite (Batch, Synthesize)
- Confidence < 0.6 falls back to monolithic prompt
- Unknown shape falls back to monolithic prompt
- User-provided task_shape skips classification

---

## Improvement 4: Slot-Based Strategy Templates

**Priority:** Medium-high. Eliminates from-scratch code generation for common tasks.

**Goal:** For each TaskShape, provide 2-3 pre-validated Scheme templates with typed slots. The LLM fills slots (prompts, model names, parameters) instead of generating nested Scheme from scratch.

### Template format

Templates are stored as **JSON manifests** in `docs/templates/` (see Critique Response #6). This replaces the earlier markdown-based template proposal. JSON manifests enable:
- Typed slot validation (string, model enum, int with min/max)
- Proper Scheme string escaping via `json.dumps()` instead of fragile `f'"{value}"'`
- Machine-readable expected_calls_formula for dry-run cross-checking

Example: `docs/templates/batch-extract-synthesize.json`:

```json
{
  "name": "batch-extract-synthesize",
  "shape": "batch",
  "trigger": "Processing many items with extraction then combining results",
  "produces": "Single synthesized output from all items",
  "slots": {
    "EXTRACTION_INSTRUCTION": {
      "type": "string",
      "required": true,
      "description": "Prompt for extracting from each item"
    },
    "SYNTHESIS_INSTRUCTION": {
      "type": "string",
      "required": true,
      "description": "Prompt for combining extractions"
    },
    "EXTRACT_MODEL": {
      "type": "model",
      "default": "gpt-4o-mini",
      "enum": ["gpt-4.1-nano", "gpt-4o-mini", "gpt-4o", "gpt-4.5"],
      "description": "Model for per-item extraction"
    },
    "SYNTH_MODEL": {
      "type": "model",
      "default": "gpt-4o",
      "enum": ["gpt-4o-mini", "gpt-4o", "gpt-4.5"],
      "description": "Model for synthesis"
    },
    "MAX_CONCURRENT": {
      "type": "int",
      "default": 20,
      "min": 1,
      "max": 50,
      "description": "Concurrent extraction limit"
    },
    "BRANCH_FACTOR": {
      "type": "int",
      "default": 5,
      "min": 2,
      "max": 15,
      "description": "Tree-reduce branching factor"
    }
  },
  "code": "(define results\n  (fan-out-aggregate\n    (lambda (item)\n      (llm-query-async #:instruction <<EXTRACTION_INSTRUCTION>>\n                       #:data item\n                       #:model <<EXTRACT_MODEL>>))\n    (lambda (extractions)\n      (tree-reduce\n        (lambda args\n          (syntax-e (llm-query #:instruction <<SYNTHESIS_INSTRUCTION>>\n                               #:data (string-join args \"\\n---\\n\")\n                               #:model <<SYNTH_MODEL>>)))\n        extractions\n        #:branch-factor <<BRANCH_FACTOR>>))\n    context\n    #:max-concurrent <<MAX_CONCURRENT>>))\n(finish results)",
  "expected_calls_formula": "N + ceil(N / BRANCH_FACTOR)",
  "structural_profile": {
    "100_items": "100 async + 25 reduce = 125 total calls",
    "500_items": "500 async + 125 reduce = 625 total calls"
  }
}
```

### Templates to create

| File | Shape | Description |
|------|-------|-------------|
| `direct-single-call.json` | Direct | Single llm-query, no combinator. Simplest possible strategy. |
| `batch-extract-synthesize.json` | Batch | fan-out-aggregate + tree-reduce. The workhorse. |
| `batch-extract-only.json` | Batch | map-async, return list of results. No synthesis phase. |
| `batch-tiered.json` | Batch/Classify | Cheap model on all, expensive on uncertain. active-learning pattern. |
| `iterative-refinement.json` | Refine | critique-refine loop with configurable max iterations. |
| `multi-model-vote.json` | Compare | Vote across 3 models, majority/plurality selection. |
| `sequential-pipeline.json` | Pipeline | sequence of 2-4 stages with validation gates. |
| `hierarchical-synthesis.json` | Synthesize | tree-reduce only (items already extracted, just need combining). |
| `generate-n-items.json` | Generate | Parallel generation with deduplication. |
| `decompose-and-process.json` | Decompose | Break input into parts, then process each. |
| `validate-all.json` | Validate | fan-out-aggregate with structured pass/fail output. |

### Changes to mcp_server.py

**Add `_fill_template()` helper (with proper validation and Scheme string escaping):**
```python
def _fill_template(template_manifest: dict, slot_values: dict) -> str:
    """Fill a JSON-manifest template with validated slot values.
    
    Uses json.dumps() for Scheme string encoding instead of fragile f-string
    quoting (see Critique Response #6).
    """
    slots_spec = template_manifest["slots"]
    code = template_manifest["code"]
    
    # Validate all required slots are provided
    for slot_name, spec in slots_spec.items():
        if spec.get("required") and slot_name not in slot_values:
            if "default" not in spec:
                raise ValueError(f"Required slot missing: {slot_name}")
    
    # Merge defaults
    merged = {}
    for slot_name, spec in slots_spec.items():
        if slot_name in slot_values:
            merged[slot_name] = slot_values[slot_name]
        elif "default" in spec:
            merged[slot_name] = spec["default"]
    
    # Validate types and constraints
    for slot_name, value in merged.items():
        spec = slots_spec.get(slot_name, {})
        slot_type = spec.get("type", "string")
        
        if slot_type == "int":
            value = int(value)
            if "min" in spec and value < spec["min"]:
                raise ValueError(f"Slot {slot_name}: {value} < min {spec['min']}")
            if "max" in spec and value > spec["max"]:
                raise ValueError(f"Slot {slot_name}: {value} > max {spec['max']}")
            merged[slot_name] = value
        
        if slot_type == "model" and "enum" in spec:
            if value not in spec["enum"]:
                raise ValueError(f"Slot {slot_name}: '{value}' not in {spec['enum']}")
    
    # Replace markers in code
    result = code
    for slot_name, value in merged.items():
        marker = f"<<{slot_name}>>"
        if marker not in result:
            raise ValueError(f"Unknown slot: {slot_name}")
        
        spec = slots_spec.get(slot_name, {})
        slot_type = spec.get("type", "string")
        
        if slot_type == "string" or slot_type == "model":
            # Use json.dumps for safe Scheme string encoding
            # json.dumps("hello \"world\"") -> '"hello \\"world\\""'
            # This is valid Scheme string syntax.
            replacement = json.dumps(str(value))
        elif slot_type == "int":
            replacement = str(int(value))
        else:
            replacement = json.dumps(str(value))
        
        result = result.replace(marker, replacement)
    
    # Check for unfilled slots
    remaining = re.findall(r'<<(\w+)>>', result)
    if remaining:
        raise ValueError(f"Unfilled slots: {remaining}")
    
    return result
```

**Include templates in shape-specific planner prompts:** Each shape prompt (from Improvement 3) includes its 2-3 relevant templates with slot descriptions. The planner output can include either raw `code_template` (as today) or `template_name` + `slot_values` (new, preferred). The MCP client can use either.

---

## Improvement 5: Execution Simulation (Decision Tree Interrogation)

**Priority:** Medium. Alternative to open-ended planning for users who want guided decomposition.

**Goal:** Instead of one-shot planning, walk the LLM through structured yes/no questions that uniquely determine a combinator composition.

### New MCP tool: `guided_plan`

```python
@mcp.tool(description="Interactively design a strategy via structured questions.")
def guided_plan(task_description: str, data_characteristics: str | None = None) -> str:
```

This tool runs the TaskShape decision tree (from the taxonomy section above) and returns:
1. The determined TaskShape
2. The DataShape (from data_characteristics)
3. The recommended template with pre-filled structural parameters
4. A list of slots the user/LLM still needs to fill (the actual prompts and model choices)

The key insight: **structural decisions are deterministic** (which combinators, what fan-out, what depth). Only **content decisions** require LLM judgment (what prompts to write, which model is appropriate).

This tool separates those concerns. The system handles structure; the LLM handles content.

### Implementation

```python
def guided_plan(task_description: str, data_characteristics: str | None = None) -> str:
    # Step 1: Classify task shape
    classification = _classify_task(task_description, data_characteristics)
    shape = classification["shape"]
    
    # Step 2: Classify data shape
    data_shape = _classify_data_shape(data_characteristics or "")
    
    # Step 3: Select template based on shape + data_shape
    template_name = _select_template(shape, data_shape)
    
    # Step 4: Pre-fill structural slots from data_shape
    structural_slots = _compute_structural_slots(data_shape, template_name)
    
    # Step 5: Return template + pre-filled structural slots + empty content slots
    return json.dumps({
        "task_shape": shape,
        "data_shape": data_shape,
        "template": template_name,
        "structural_slots_filled": structural_slots,
        "content_slots_needed": _get_unfilled_slots(template_name, structural_slots),
        "next_step": "Fill the content slots (prompts, model choices) and call execute_scheme with the filled template.",
    }, indent=2)
```

---

## Improvement 6: Strategy Replay Database

**Priority:** Medium. Enables learning across sessions.

**Goal:** After each execution, record task shape + strategy + outcome. Feed the 5 most relevant past executions into future planner prompts.

### Storage

File: `~/.rlm-scheme/strategy-history.jsonl` (one JSON object per line).

```json
{
  "timestamp": "2026-06-03T10:00:00Z",
  "task_shape": "batch",
  "task_description_preview": "Extract key points from research papers...",
  "strategy_name": "Parallel Fan-Out with Tree Reduction",
  "template_used": "batch-extract-synthesize",
  "code_hash": "a1b2c3d4",
  "outcome": "success",
  "metrics": {
    "total_calls": 625,
    "elapsed_seconds": 272,
    "total_tokens": 511000,
    "total_cost": "$1.52",
    "models": {"gpt-4o-mini": 500, "gpt-4o": 125}
  },
  "scale": "comprehensive",
  "item_count": 500
}
```

### Changes to mcp_server.py

**Module-level state (near line 210):**
```python
import uuid

# Plan contexts keyed by plan_id (UUID), not a single global.
# This is safe under concurrent MCP tool calls (Critique Response #7).
_plan_contexts = {}  # {plan_id: {task_description, task_shape, strategy_name, ...}}
_HISTORY_DIR = os.path.expanduser("~/.rlm-scheme")
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "strategy-history.jsonl")
```

**In `plan_strategy`, after successful JSON parse:** (already updated in Improvement 3 above — `plan_strategy` now generates `plan_id = str(uuid.uuid4())` and stores context in `_plan_contexts[plan_id]`).

**New helper `_record_strategy_history()` (add after execute_scheme):**
```python
def _record_strategy_history(code: str, outcome: str, execution_summary: dict,
                             plan_id: str | None = None):
    """Record execution history, keyed by plan_id instead of global state.
    
    plan_id is the UUID returned by plan_strategy in _meta.plan_id.
    If not provided (e.g., execute_scheme called without prior plan_strategy),
    records with "unknown" context.
    """
    try:
        os.makedirs(_HISTORY_DIR, exist_ok=True)
        
        # Look up plan context by plan_id (not from a mutable global)
        ctx = _plan_contexts.get(plan_id, {}) if plan_id else {}
        
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "plan_id": plan_id,
            "task_shape": ctx.get("task_shape", "unknown"),
            "task_description_preview": ctx.get("task_description", "")[:100],
            "strategy_name": ctx.get("strategy_name", ""),
            "template_used": ctx.get("template_used", ""),
            "code_hash": hashlib.md5(code.encode()).hexdigest()[:8],
            "outcome": outcome,
            "metrics": {
                "total_calls": execution_summary.get("llm_calls", 0),
                "elapsed_seconds": execution_summary.get("elapsed", 0),
                "total_tokens": execution_summary.get("tokens", 0),
                "total_cost": execution_summary.get("total_cost_estimate", ""),
            },
            "scale": ctx.get("scale", ""),
        }
        with open(_HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        
        # Clean up plan context after recording (avoid unbounded growth)
        if plan_id and plan_id in _plan_contexts:
            del _plan_contexts[plan_id]
    except Exception:
        pass  # Never fail the execution
```

**Modify `execute_scheme` signature (line 1364):** Add optional `plan_id` parameter:
```python
@mcp.tool(description="Execute Scheme orchestration code in the sandbox.")
async def execute_scheme(code: str, plan_id: str | None = None, ctx: Context = None) -> str:
    # ... existing execution logic ...
```

**In `execute_scheme`, before the return (line 1469):**
```python
_record_strategy_history(
    code,
    "success" if resp["status"] == "finished" else "error",
    exec_summary,
    plan_id=plan_id,  # links execution to the plan that produced it
)
```

**New helper `_load_relevant_history()`:**
```python
def _load_relevant_history(task_description: str, shape: str) -> str:
    try:
        if not os.path.exists(_HISTORY_FILE):
            return ""
        entries = []
        with open(_HISTORY_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        
        # Rotate if too large
        if len(entries) > 1000:
            entries = entries[-500:]
            with open(_HISTORY_FILE, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
        
        # Filter to matching shape, take top 5 by recency
        relevant = [e for e in entries if e.get("task_shape") == shape]
        relevant = relevant[-5:]
        
        if not relevant:
            return ""
        
        lines = ["## Past Executions (Similar Tasks)", ""]
        for i, e in enumerate(relevant, 1):
            lines.append(
                f"{i}. **{e.get('strategy_name', 'unnamed')}** | "
                f"Shape: {e.get('task_shape')} | "
                f"Outcome: {e.get('outcome')} | "
                f"{e['metrics'].get('total_calls', '?')} calls, "
                f"{e['metrics'].get('elapsed_seconds', '?')}s, "
                f"{e['metrics'].get('total_cost', '?')}"
            )
        return "\n".join(lines) + "\n"
    except Exception:
        return ""
```

---

## Improvement 7: Self-Verification Tool

**Priority:** Medium. Catches errors before execution.

**Goal:** After generating a strategy, verify it structurally (dry-run) and semantically (LLM self-check) before spending real tokens.

### New MCP tool: `verify_strategy`

```python
@mcp.tool(description="Verify a strategy before execution. Runs dry-run + semantic check.")
async def verify_strategy(code: str, task_description: str, expected_items: int | None = None) -> str:
```

### Implementation

```python
async def verify_strategy(code: str, task_description: str, expected_items: int | None = None, ctx: Context = None) -> str:
    # Step 1: Dry run for structural metrics
    dry_run_result = json.loads(await dry_run_scheme(code))
    
    if dry_run_result["status"] == "error":
        return json.dumps({
            "verified": False,
            "errors": [f"Code failed dry-run: {dry_run_result.get('message', 'unknown')}"],
        }, indent=2)
    
    structure = dry_run_result["structure"]
    warnings = []
    errors = []
    
    # Step 2: Structural checks (deterministic, no LLM needed)
    if structure["total_calls"] == 0:
        errors.append("Strategy makes 0 LLM calls. Did you forget llm-query?")
    
    if expected_items and structure["async_calls"] > 0:
        if structure["async_calls"] < expected_items * 0.5:
            warnings.append(
                f"Expected ~{expected_items} items but only {structure['async_calls']} async calls. "
                f"Strategy may not process all items."
            )
    
    if structure["max_fan_out"] > 50:
        warnings.append(
            f"Max fan-out is {structure['max_fan_out']}. May hit rate limits. "
            f"Consider reducing #:max-concurrent."
        )
    
    if structure["sync_calls"] > 20:
        warnings.append(
            f"{structure['sync_calls']} sequential LLM calls. This will be slow. "
            f"Consider using async calls where possible."
        )
    
    # Step 3: Semantic check (cheap LLM call)
    verification_prompt = f"""Analyze this orchestration code against the task requirements.

Task: {task_description}
Expected items: {expected_items or 'not specified'}

Code:
```scheme
{code}
```

Dry-run metrics:
- Total LLM calls: {structure['total_calls']}
- Async (parallel): {structure['async_calls']}
- Sync (sequential): {structure['sync_calls']}
- Max fan-out: {structure['max_fan_out']}
- Max depth: {structure['max_depth']}

Answer ONLY these questions as JSON:
{{
  "processes_all_items": true/false,
  "output_matches_task": true/false,
  "has_error_handling": true/false,
  "potential_issues": ["issue1", "issue2"],
  "recommendation": "proceed" or "fix X before running"
}}"""

    backend = get_backend()
    result = backend._call_llm(
        instruction=verification_prompt,
        data="",
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=500,
    )
    
    try:
        text = result["text"].strip()
        json_match = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        semantic_check = json.loads(text)
    except (json.JSONDecodeError, KeyError):
        semantic_check = {"error": "semantic check failed to parse"}
    
    if semantic_check.get("processes_all_items") == False:
        warnings.append("Semantic check: strategy may not process all items")
    if semantic_check.get("potential_issues"):
        warnings.extend(semantic_check["potential_issues"])
    
    return json.dumps({
        "verified": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "structure": structure,
        "estimates": dry_run_result["estimates"],
        "semantic_check": semantic_check,
        "recommendation": semantic_check.get("recommendation", "review warnings"),
    }, indent=2)
```

---

## Improvement 8: Progressive Summarization via `fold-summarizing`

**Priority:** Medium-low. Addresses memory pressure in long fold-sequential chains.

**Goal:** Provide a way to automatically summarize accumulating context when it exceeds a token threshold, without silently changing `fold-sequential`'s semantics.

**Design decision (Critique Response #8):** The original plan silently modified `fold-sequential` behavior based on accumulator length. This is dangerous because it:
- Adds LLM calls that change cost and latency characteristics unpredictably
- Changes semantics (the accumulator is now an LLM-compressed approximation, not the actual fold result)
- Assumes the accumulator is always a string (breaks for structured data)

Instead, create `fold-summarizing` as an **explicit, separate combinator** that users opt into when they know they're folding text.

### Changes to racket_server.rkt

**Add new combinator `fold-summarizing`** (do NOT modify `fold-sequential`):

```scheme
(define (fold-summarizing fn init items
          #:horizon [horizon 8000]
          #:summary-model [summary-model "gpt-4o-mini"]
          #:summary-instruction [summary-instruction
            "Compress this running summary, preserving all key facts and findings."])
  "Like fold-sequential, but compresses the accumulator when it exceeds #:horizon tokens.
   Use this when folding large numbers of items where context would exceed the model's window.
   WARNING: The accumulator is periodically summarized by an LLM, so information may be lost.
   For exact accumulation, use fold-sequential instead."
  (foldl
    (lambda (item acc)
      (let ([new-acc (fn acc item)])
        (if (> (string-length new-acc) (* horizon 4))  ; approx tokens as chars/4
            ;; Explicit summarization step
            (syntax-e (llm-query
              #:instruction summary-instruction
              #:data new-acc
              #:model summary-model))
            new-acc)))
    init
    items))
```

**Type signature:**
```
fold-summarizing : (Fn<Acc, Item, String>, String, [Item],
                    #:horizon Int, #:summary-model String, #:summary-instruction String) -> String
```

Note: Unlike `fold-sequential` which is polymorphic over accumulator type, `fold-summarizing` requires `String` accumulators because it feeds them to an LLM for compression.

### Changes to docs/combinators.md

Add `fold-summarizing` to the Sequential section, clearly marked as a variant of `fold-sequential`:

```markdown
### fold-summarizing

Like fold-sequential, but automatically compresses the running accumulator when it exceeds
a token threshold. Use for processing many items where context would grow beyond model limits.

**Trade-off:** Periodic LLM summarization prevents context overflow but may lose detail.
For exact accumulation (e.g., building a data structure), use fold-sequential instead.

**Signature:** `(Fn<String, Item, String>, String, [Item], #:horizon Int, #:summary-model String) -> String`
```

### Impact on decision trees

In the Synthesize and Batch decision trees, add `fold-summarizing` as an option:

```
Q2: Is synthesis order-sensitive?
    YES → Is the item list large (50+ items)?
        YES → fold-summarizing (prevents context overflow)
        NO  → fold-sequential (exact accumulation)
    NO  → tree-reduce
```

---

## Improvement 9: Checkpoint-Aware Retry

**Priority:** Medium-low. Reduces boilerplate in error-prone pipelines.

**Goal:** Make checkpointing and retry declarative parameters on combinators rather than manual boilerplate.

### Changes to racket_server.rkt

Add keyword arguments to `fan-out-aggregate`:
```scheme
(fan-out-aggregate map-fn reduce-fn items
  #:max-concurrent 20
  #:checkpoint-every 50      ; save progress every N items
  #:retry-on '(rate-limit timeout)
  #:max-retries 3)
```

Implementation: wrap the map phase in a loop that checkpoints partial results and retries failed items.

---

## Improvement 10: Multi-Model Routing

**Priority:** Low. Prevents cross-context breakage from the scope hygiene paper.

**Goal:** Scope-aware model selection based on task annotations, preventing prompts tuned for one model from being sent to another.

### Changes to racket_server.rkt

Add `model-router` parameter:
```scheme
(parameterize ([model-router
                (make-router
                  #:extract "gpt-4.1-nano"
                  #:synthesize "gpt-4o"
                  #:critique "gpt-4o-mini"
                  #:default "gpt-4o-mini")])
  (fan-out-aggregate ...))
```

When `llm-query` is called without an explicit `#:model`, the router selects based on a `#:task-type` annotation:
```scheme
(llm-query #:instruction "..." #:data "..." #:task-type 'extract)
;; Router selects gpt-4.1-nano based on the task-type annotation
```

---

## Improvement 11: Reasoning Annotations

**Priority:** Low. Improves audit trail and debugging.

**Goal:** Let the LLM annotate its decomposition rationale as first-class metadata in the execution trace.

### Changes to racket_server.rkt

Add `with-reasoning` form:
```scheme
(with-reasoning "Split by chapter because each is self-contained"
  (fan-out-aggregate extract-fn reduce-fn chapters))
```

This writes the reasoning string to the scope log alongside the combinator execution, making the audit trail semantic (why this decomposition?) rather than just structural (who called whom?).

---

## Improvement 12: Separate Decomposition from Code Generation

**Priority:** Medium. The most architecturally significant change.

**Goal:** The LLM should NEVER write combinator compositions from scratch. It should describe decomposition strategy in structured YAML/JSON, and the system compiles that to Scheme.

### New intermediate representation: Strategy Spec

```yaml
phases:
  - name: extract
    type: parallel_map
    input: all_items
    model: gpt-4o-mini
    instruction: "Extract key points from this document"
    concurrency: 20
  - name: synthesize
    type: hierarchical_reduce
    input: extract.output
    model: gpt-4o
    instruction: "Combine these findings into a coherent summary"
    branch_factor: 5
quality:
  type: critique_refine
  target: synthesize.output
  max_iterations: 2
  critique_model: gpt-4o-mini
  refine_model: gpt-4o
```

### Compiler: spec_to_scheme()

A Python function that deterministically translates the spec to valid Scheme:

```python
def spec_to_scheme(spec: dict) -> str:
    """Compile a strategy spec to executable Scheme code."""
    # ... deterministic template filling based on spec structure ...
```

This eliminates an entire class of errors: the LLM writes structured data (which it's good at), and the system generates correct Scheme (which is deterministic). The LLM never needs to know Scheme syntax.

### New MCP tool: `compile_strategy`

```python
@mcp.tool(description="Compile a strategy specification to executable Scheme code.")
def compile_strategy(spec: str) -> str:
    """Takes a YAML/JSON strategy spec and returns executable Scheme code."""
    parsed = yaml.safe_load(spec) if spec.strip().startswith("phases") else json.loads(spec)
    code = spec_to_scheme(parsed)
    return json.dumps({
        "scheme_code": code,
        "can_dry_run": True,
        "next_step": "Run dry_run_scheme to verify structure, then execute_scheme to run.",
    }, indent=2)
```

---

## Implementation Order

Revised per critique review. The key reordering: fix docs/code consistency first (cheap, reduces downstream errors), then build dry-run with correct async semantics, then verification, then templates, then classification, then history last (only after execution metadata is reliable).

```
Phase 1: Docs/Code Consistency      [no dependencies, zero risk]
          - Correct model names in docs (deprecated names → current)
          - Add type signatures with async/sync distinction (Improvement 2)
          - Document dead recursive-spawn #:depth parameter
          - Document tiered's sequential behavior
          - Add Direct shape to taxonomy

Phase 2: Dry Run Mode               [no dependencies, highest leverage]
          - DryRunScheduler with delayed futures (not pre-resolved)
          - DryRunContext for isolated metrics (not _register_call)
          - Nesting depth warning at >3 levels
          - Improvement 1

Phase 3: Self-Verification Tool     [depends on Phase 2 dry-run]
          - Structural checks (deterministic lints, no LLM needed)
          - Semantic check (cheap LLM call for deeper analysis)
          - Improvement 7

Phase 4: Machine-Readable Templates [no dependency on classification]
          - JSON manifests with typed/validated slots
          - Top 4 templates: Direct, Batch extract, Batch extract+synthesize, Refine
          - _fill_template with json.dumps string encoding
          - Improvement 4

Phase 5: Task Classification        [benefits from Phase 1 types, Phase 4 templates]
          - Deterministic decision tree (_deterministic_classify)
          - LLM only fills missing structured fields
          - structured_fields parameter on plan_strategy
          - Shape-specific planner prompts (13 shapes incl. Direct)
          - Improvement 3

Phase 6: Strategy Replay Database   [depends on Phases 2+5 for reliable metadata]
          - plan_id (UUID) keying, not _last_plan_context global
          - execute_scheme gains plan_id parameter
          - History rotation at 1000 entries
          - Improvement 6

Phase 7: fold-summarizing           [independent, touches racket_server.rkt]
          - New explicit combinator (not silent fold-sequential mod)
          - Improvement 8

Phase 8: Checkpoint-Aware Retry     [independent, touches racket_server.rkt]
          - Improvement 9

Phase 9: Multi-Model Routing        [independent, touches racket_server.rkt]
          - Improvement 10

Phase 10: Reasoning Annotations     [independent, touches racket_server.rkt]
           - Improvement 11

Phase 11: Guided Plan Tool          [depends on Phases 4-5 for shapes + templates]
           - Improvement 5 (Execution Simulation)

Phase 12: Strategy Spec Compiler    [depends on Phases 4-5 for templates]
           - Improvement 12
```

Phases 1-6 are Python-side changes to mcp_server.py and documentation.
Phases 7-10 are Racket-side changes to racket_server.rkt.
Phases 11-12 are the culmination: the LLM writes YAML specs, not Scheme.

---

## Files Modified

| File | Phases | Nature of change |
|------|--------|-----------------|
| `mcp_server.py` | 1, 3, 4, 5, 6, 11, 12 | New tools, two-phase planning, history, spec compiler |
| `racket_server.rkt` | 7, 8, 9, 10 | New parameters, combinator enhancements |
| `docs/combinators.md` | 2 | Type signatures, common errors |
| `docs/planner-prompt.md` | -- | Kept as fallback (no changes) |

## Files Created

| File | Phase | Purpose |
|------|-------|---------|
| `docs/task-shapes.md` | 5 | Full taxonomy of TaskShape + DataShape + decision tree |
| `docs/shape-prompts/*.md` (13 files incl. direct.md) | 5 | Per-shape planner prompts |
| `docs/templates/*.json` (11 files incl. direct-single-call.json) | 4 | JSON manifest strategy templates |
| `docs/tool-descriptions/dry_run_scheme.md` | 1 | Dry-run tool docs |
| `docs/tool-descriptions/verify_strategy.md` | 6 | Verification tool docs |
| `docs/tool-descriptions/guided_plan.md` | 12 | Guided planning tool docs |
| `docs/tool-descriptions/compile_strategy.md` | 11 | Strategy compiler tool docs |
| `tests/test_dry_run.py` | 1 | Dry-run tests |
| `tests/test_task_classification.py` | 3 | Classification tests |
| `tests/test_strategy_history.py` | 5 | History recording/loading tests |
| `tests/test_verify_strategy.py` | 6 | Verification tests |
| `tests/test_spec_compiler.py` | 11 | Spec-to-Scheme compiler tests |
