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

## Taxonomy: TaskShape

Tasks the framework handles, each mapping to a canonical set of combinators.

### Current shapes (7)

| Shape | Description | Primary Combinators |
|-------|-------------|-------------------|
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
    YES → wrap with tiered: cheap model for per-item extraction,
          expensive model ONLY for synthesis/aggregation phase
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

**RacketREPL.__init__ (line 221):** Add instance variables:
```python
self._dry_run = False
self._dry_run_metrics = {
    "total_calls": 0,
    "sync_calls": 0,
    "async_calls": 0,
    "max_concurrent_pending": 0,
    "call_tree": [],  # [{instruction_preview, model, type, depth}]
    "models_used": {},  # model -> count
}
```

**RacketREPL.send(), `op == "llm-query"` branch (line 510):** Insert dry-run bypass at the top of the branch, before the existing `_call_llm()` call:
```python
if op == "llm-query":
    if self._dry_run:
        # Generate deterministic mock response
        instruction = msg.get("instruction", "")
        data = msg.get("data", "")
        model = msg.get("model", "") or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
        mock_hash = hashlib.md5((instruction + data[:100]).encode()).hexdigest()[:8]
        mock_text = f"[DRY-RUN:{mock_hash}] Mock response for: {instruction[:60]}"
        
        # Track metrics
        self._dry_run_metrics["total_calls"] += 1
        self._dry_run_metrics["sync_calls"] += 1
        self._dry_run_metrics["models_used"][model] = self._dry_run_metrics["models_used"].get(model, 0) + 1
        self._dry_run_metrics["call_tree"].append({
            "instruction_preview": instruction[:80],
            "model": model,
            "type": "sync",
            "depth": self._current_depth,
        })
        
        # Still register in call_registry for stats
        call_id = self._next_call_id()
        self._register_call(call_id, "sync-dry", model, instruction, depth=self._current_depth)
        self._complete_call(call_id, 0, 0.0)
        
        # Write mock response back to Racket stdin
        self.proc.stdin.write(json.dumps({
            "result": mock_text,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }) + "\n")
        self.proc.stdin.flush()
        continue  # or equivalent control flow to skip the real call
    
    # ... existing real call code ...
```

**RacketREPL.send(), `op == "llm-query-async"` branch (line 559):** Similar bypass:
```python
elif op == "llm-query-async":
    if self._dry_run:
        instruction = msg.get("instruction", "")
        data = msg.get("data", "")
        model = msg.get("model", "") or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
        mock_hash = hashlib.md5((instruction + data[:100]).encode()).hexdigest()[:8]
        mock_text = f"[DRY-RUN:{mock_hash}] Mock async for: {instruction[:60]}"
        
        self._dry_run_metrics["total_calls"] += 1
        self._dry_run_metrics["async_calls"] += 1
        self._dry_run_metrics["models_used"][model] = self._dry_run_metrics["models_used"].get(model, 0) + 1
        self._dry_run_metrics["call_tree"].append({
            "instruction_preview": instruction[:80],
            "model": model,
            "type": "async",
            "depth": self._current_depth,
        })
        
        # Create pre-resolved future
        future = concurrent.futures.Future()
        future.set_result({"text": mock_text, "prompt_tokens": 0, "completion_tokens": 0})
        self._pending[msg["id"]] = future
        
        # Track max concurrent pending
        current_pending = len(self._pending)
        if current_pending > self._dry_run_metrics["max_concurrent_pending"]:
            self._dry_run_metrics["max_concurrent_pending"] = current_pending
        
        # No response to Racket (same as real async -- Racket continues immediately)
        continue
    
    # ... existing real async code ...
```

**New MCP tool `dry_run_scheme` (after line 1471):**
```python
@mcp.tool(description="Simulate orchestration without real LLM calls. Returns structural analysis.")
async def dry_run_scheme(code: str, ctx: Context = None) -> str:
    backend = get_backend()
    backend._dry_run = True
    backend._dry_run_metrics = {
        "total_calls": 0, "sync_calls": 0, "async_calls": 0,
        "max_concurrent_pending": 0, "call_tree": [], "models_used": {},
    }
    backend.reset_call_stats()
    _call_registry.reset_stats()
    loop = asyncio.get_event_loop()
    t_start = time.monotonic()
    
    try:
        resp = await loop.run_in_executor(
            None, lambda: backend.send({"op": "eval", "code": code}, timeout=30)
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        backend._dry_run = False
    
    elapsed = round(time.monotonic() - t_start, 3)
    metrics = backend._dry_run_metrics
    
    # Estimate cost from model usage
    MODEL_COSTS = {
        "gpt-4.1-nano": 0.001, "gpt-4o-mini": 0.005,
        "gpt-4o": 0.01, "gpt-4.5": 0.03,
    }
    estimated_cost = sum(
        count * MODEL_COSTS.get(model, 0.01)
        for model, count in metrics["models_used"].items()
    )
    
    # Compute max depth from call tree
    max_depth = max((c["depth"] for c in metrics["call_tree"]), default=0)
    
    return json.dumps({
        "status": resp["status"],
        "value": resp.get("result", ""),
        "dry_run_elapsed": elapsed,
        "structure": {
            "total_calls": metrics["total_calls"],
            "sync_calls": metrics["sync_calls"],
            "async_calls": metrics["async_calls"],
            "max_fan_out": metrics["max_concurrent_pending"],
            "max_depth": max_depth,
            "models_used": metrics["models_used"],
        },
        "estimates": {
            "cost": f"${estimated_cost:.2f}",
            "latency_range": f"{metrics['sync_calls'] * 2}s - {metrics['sync_calls'] * 5}s (sync calls dominate)",
        },
        "call_tree": metrics["call_tree"][:50],  # truncate for readability
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

    Item          = a single string (LLM response text, document text, etc.)
    [Item]        = list of items
    Fn<A, B>      = function from A to B
    Fn<A, B, C>   = function from (A, B) to C
    AsyncHandle   = opaque handle from llm-query-async (NOT a string -- must be awaited)
    Thunk<A>      = () -> A (zero-argument function)
    SyntaxObject  = wrapped LLM response (must unwrap with syntax-e before string operations)

Key rules:
- llm-query returns SyntaxObject, must unwrap with (syntax-e result)
- llm-query-async returns AsyncHandle, awaited by map-async/fan-out-aggregate
- map-async/fan-out-aggregate auto-unwrap: results are plain strings (no syntax-e needed)
```

Add type signature to each combinator's `**Signature:**` line:

```
parallel          : ([Thunk<A>], #:max-concurrent Int) -> [A]
race              : [Thunk<AsyncHandle>] -> Item
sequence          : (Fn<A,B>, Fn<B,C>, ...) -> Fn<A, ...last output>
fold-sequential   : (Fn<Acc, Item, Acc>, Acc, [Item]) -> Acc
tree-reduce       : (Fn<Item..., Item>, [Item], #:branch-factor Int) -> Item
fan-out-aggregate : (Fn<Item, AsyncHandle>, Fn<[Item], B>, [Item]) -> B
recursive-spawn   : (Thunk<String>, #:depth Int) -> Fn<Item, Item>
iterate-until     : (Fn<A, A>, Fn<A, Bool>, A, #:max-iter Int) -> A
critique-refine   : (Thunk<Item>, Fn<Item, Item>, Fn<Item, Item, Item>) -> Item
with-validation   : (Fn<A, B>, Fn<B, Bool>) -> Fn<A, B>
vote              : ([Thunk<A>], #:method Symbol) -> A
ensemble          : ([Thunk<A>], #:aggregator Fn<[A], B>) -> B
tiered            : (Fn<Item, A>, Fn<[A], B>, [Item]) -> B
active-learning   : (Fn<Item, A>, Fn<Item, A>, Fn<A, Float>, [Item]) -> [A]
memoized          : (Fn<A, B>, #:key-fn Fn<A, String>) -> Fn<A, B>
choose            : (Fn<A, Bool>, Fn<A, B>, Fn<A, B>) -> Fn<A, B>
try-fallback      : (Fn<A, B>, Fn<A, B>) -> Fn<A, B>
```

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
```python
_CLASSIFICATION_PROMPT = """Classify this task into exactly one TaskShape.

Shapes:
- Batch: Apply same operation to many independent items
- Synthesize: Combine/summarize many inputs into one output
- Search: Explore solution space, find best answer
- Refine: Iteratively improve a single artifact
- Compare: Evaluate alternatives against criteria
- Classify: Categorize/label items
- Pipeline: Multi-stage sequential transformation
- Generate: Create new content from scratch (no input items to transform)
- Decompose: Break one complex thing into structured parts
- Validate: Check items against criteria, produce pass/fail assessments
- Aggregate: Gather structured/numeric metrics across items
- Composite: Task clearly has multiple phases (e.g., extract THEN synthesize)

If Composite, identify the 2-3 constituent shapes in order.

Task: {task_description}
Data: {data_characteristics}

Return ONLY JSON: {{"shape": "...", "confidence": 0.0-1.0, "reasoning": "one sentence", "sub_shapes": ["shape1", "shape2"] (only if Composite)}}"""

def _classify_task(task_description: str, data_characteristics: str) -> dict:
    backend = get_backend()
    result = backend._call_llm(
        instruction=_CLASSIFICATION_PROMPT.format(
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
        # Extract JSON from markdown fences if present
        json_match = re.search(r'```(?:json)?\s*\n(.*?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
        return json.loads(text)
    except (json.JSONDecodeError, KeyError):
        return {"shape": "unknown", "confidence": 0.0, "reasoning": "classification failed"}
```

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

**Modify `plan_strategy` (line 1285):** Add `task_shape` parameter, two-phase flow:
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
    task_shape: str | None = None,  # NEW: override classification
) -> str:
    # Phase 1: Classify task shape
    classification = None
    if task_shape:
        shape = task_shape.lower()
        classification = {"shape": shape, "confidence": 1.0, "reasoning": "user-specified"}
    else:
        classification = _classify_task(task_description, data_characteristics)
        shape = classification.get("shape", "unknown")
    
    # Phase 2: Select prompt
    if classification["confidence"] >= 0.6 and shape in _SHAPE_PROMPTS:
        prompt_template = _SHAPE_PROMPTS[shape]
    else:
        # Fall back to monolithic prompt
        prompt_template = _PLANNER_PROMPT_TEMPLATE
    
    # Inject history if available (Improvement 5)
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
    # Add to _meta:
    parsed["_meta"]["task_shape"] = shape
    parsed["_meta"]["classification_confidence"] = classification["confidence"]
    parsed["_meta"]["classification_reasoning"] = classification.get("reasoning", "")
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

Each template is a markdown file in `docs/templates/` with:

```markdown
# Template: Batch Extract and Synthesize

**Shape:** Batch (with synthesis phase)
**Trigger:** Processing many items with extraction then combining results
**Produces:** Single synthesized output from all items

## Slots

| Slot | Type | Default | Description |
|------|------|---------|-------------|
| EXTRACTION_INSTRUCTION | string | (required) | Prompt for extracting from each item |
| SYNTHESIS_INSTRUCTION | string | (required) | Prompt for combining extractions |
| EXTRACT_MODEL | model | gpt-4o-mini | Model for per-item extraction |
| SYNTH_MODEL | model | gpt-4o | Model for synthesis |
| MAX_CONCURRENT | int (5-50) | 20 | Concurrent extraction limit |
| BRANCH_FACTOR | int (3-10) | 5 | Tree-reduce branching factor |

## Code

    (define results
      (fan-out-aggregate
        (lambda (item)
          (llm-query-async #:instruction <<EXTRACTION_INSTRUCTION>>
                           #:data item
                           #:model <<EXTRACT_MODEL>>))
        (lambda (extractions)
          (tree-reduce
            (lambda args
              (syntax-e (llm-query #:instruction <<SYNTHESIS_INSTRUCTION>>
                                   #:data (string-join args "\n---\n")
                                   #:model <<SYNTH_MODEL>>)))
            extractions
            #:branch-factor <<BRANCH_FACTOR>>))
        context
        #:max-concurrent <<MAX_CONCURRENT>>))
    (finish results)

## Structural Profile (from dry-run)
- For 100 items: 100 async + 25 reduce = 125 total calls
- For 500 items: 500 async + 125 reduce = 625 total calls
- Cost formula: (N * extract_cost) + (N/branch_factor * synth_cost)
```

### Templates to create

| File | Shape | Description |
|------|-------|-------------|
| `batch-extract-synthesize.md` | Batch | fan-out-aggregate + tree-reduce. The workhorse. |
| `batch-extract-only.md` | Batch | map-async, return list of results. No synthesis phase. |
| `batch-tiered.md` | Batch/Classify | Cheap model on all, expensive on uncertain. active-learning pattern. |
| `iterative-refinement.md` | Refine | critique-refine loop with configurable max iterations. |
| `multi-model-vote.md` | Compare | Vote across 3 models, majority/plurality selection. |
| `sequential-pipeline.md` | Pipeline | sequence of 2-4 stages with validation gates. |
| `hierarchical-synthesis.md` | Synthesize | tree-reduce only (items already extracted, just need combining). |
| `generate-n-items.md` | Generate | Parallel generation with deduplication. |
| `decompose-and-process.md` | Decompose | Break input into parts, then process each. |
| `validate-all.md` | Validate | fan-out-aggregate with structured pass/fail output. |

### Changes to mcp_server.py

**Add `_fill_template()` helper:**
```python
def _fill_template(template_code: str, slots: dict[str, str]) -> str:
    """Replace <<SLOT_NAME>> markers with provided values."""
    result = template_code
    for slot_name, value in slots.items():
        marker = f"<<{slot_name}>>"
        if marker not in result:
            raise ValueError(f"Unknown slot: {slot_name}")
        # Quote string values for Scheme
        if isinstance(value, str) and not value.startswith('"'):
            value = f'"{value}"'
        result = result.replace(marker, value)
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
_last_plan_context = {}  # Set by plan_strategy, read by execute_scheme
_HISTORY_DIR = os.path.expanduser("~/.rlm-scheme")
_HISTORY_FILE = os.path.join(_HISTORY_DIR, "strategy-history.jsonl")
```

**In `plan_strategy`, after successful JSON parse (line 1344):**
```python
_last_plan_context.update({
    "task_description": task_description,
    "task_shape": shape,
    "strategy_name": parsed.get("recommended", {}).get("strategy_name", ""),
    "template_used": parsed.get("recommended", {}).get("template_name", ""),
    "scale": scale,
})
```

**New helper `_record_strategy_history()` (add after execute_scheme):**
```python
def _record_strategy_history(code: str, outcome: str, execution_summary: dict):
    try:
        os.makedirs(_HISTORY_DIR, exist_ok=True)
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "task_shape": _last_plan_context.get("task_shape", "unknown"),
            "task_description_preview": _last_plan_context.get("task_description", "")[:100],
            "strategy_name": _last_plan_context.get("strategy_name", ""),
            "template_used": _last_plan_context.get("template_used", ""),
            "code_hash": hashlib.md5(code.encode()).hexdigest()[:8],
            "outcome": outcome,
            "metrics": {
                "total_calls": execution_summary.get("llm_calls", 0),
                "elapsed_seconds": execution_summary.get("elapsed", 0),
                "total_tokens": execution_summary.get("tokens", 0),
                "total_cost": execution_summary.get("total_cost_estimate", ""),
            },
            "scale": _last_plan_context.get("scale", ""),
        }
        with open(_HISTORY_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass  # Never fail the execution
```

**In `execute_scheme`, before the return (line 1469):**
```python
_record_strategy_history(
    code,
    "success" if resp["status"] == "finished" else "error",
    exec_summary,
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

## Improvement 8: Progressive Summarization

**Priority:** Medium-low. Addresses memory pressure in long fold-sequential chains.

**Goal:** When intermediate context in a fold-sequential exceeds a token threshold, automatically summarize before the next step using a cheap model.

### Changes to racket_server.rkt

Add `summary-horizon` parameter (default: disabled):
```scheme
(define summary-horizon (make-parameter #f))  ; #f = disabled, number = token threshold
```

In `fold-sequential`, after each accumulator update, check if the accumulator string length exceeds the horizon (approximating tokens as chars/4):
```scheme
(define (fold-sequential fn init items)
  (foldl
    (lambda (item acc)
      (let ([new-acc (fn acc item)])
        (if (and (summary-horizon)
                 (> (string-length new-acc) (* (summary-horizon) 4)))
            ;; Auto-summarize
            (syntax-e (llm-query
              #:instruction "Compress this running summary, preserving all key facts and findings."
              #:data new-acc
              #:model "gpt-4o-mini"))
            new-acc)))
    init
    items))
```

This is opt-in via `(parameterize ([summary-horizon 8000]) ...)`.

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

```
Phase 1: Dry Run Mode              [no dependencies, highest leverage]
Phase 2: Combinator Type Signatures [no dependencies, documentation only]
Phase 3: Task Classification        [benefits from Phase 2 types in prompts]
Phase 4: Slot-Based Templates       [depends on Phase 3 for shape-specific prompts]
Phase 5: Strategy Replay Database   [depends on Phase 1 for structural data, Phase 3 for shapes]
Phase 6: Self-Verification Tool     [depends on Phase 1 dry-run]
Phase 7: Progressive Summarization  [independent, touches racket_server.rkt]
Phase 8: Checkpoint-Aware Retry     [independent, touches racket_server.rkt]
Phase 9: Multi-Model Routing        [independent, touches racket_server.rkt]
Phase 10: Reasoning Annotations     [independent, touches racket_server.rkt]
Phase 11: Strategy Spec Compiler    [depends on Phases 3-4 for templates]
Phase 12: Guided Plan Tool          [depends on Phases 3-4 for shapes + templates]
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
| `docs/task-shapes.md` | 3 | Full taxonomy of TaskShape + DataShape + decision tree |
| `docs/shape-prompts/*.md` (12 files) | 3 | Per-shape planner prompts |
| `docs/templates/*.md` (10 files) | 4 | Slot-based strategy templates |
| `docs/tool-descriptions/dry_run_scheme.md` | 1 | Dry-run tool docs |
| `docs/tool-descriptions/verify_strategy.md` | 6 | Verification tool docs |
| `docs/tool-descriptions/guided_plan.md` | 12 | Guided planning tool docs |
| `docs/tool-descriptions/compile_strategy.md` | 11 | Strategy compiler tool docs |
| `tests/test_dry_run.py` | 1 | Dry-run tests |
| `tests/test_task_classification.py` | 3 | Classification tests |
| `tests/test_strategy_history.py` | 5 | History recording/loading tests |
| `tests/test_verify_strategy.py` | 6 | Verification tests |
| `tests/test_spec_compiler.py` | 11 | Spec-to-Scheme compiler tests |
