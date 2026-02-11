# RLM-Scheme: LLM Orchestration Guide

**A Scheme sandbox for safe, scalable LLM orchestration with 16 composable patterns.**

Use **plan_strategy()** to get strategy recommendations for your task. This guide provides pattern overview and primitive reference.

---

## Quick Start

```scheme
;; 1. Load data (optional)
(load-context "your data here")

;; 2. Basic LLM call
(define result (syntax-e (llm-query
  #:instruction "Summarize this document"
  #:data context
  #:model "gpt-3.5-turbo")))

;; 3. Parallel processing (10× faster)
(define results (map-async
  (lambda (item) (llm-query-async #:instruction "Process this" #:data item))
  items))

;; 4. Return result
(finish result)
```

**Workflow:**
1. Call `plan_strategy(task_description)` to get recommended orchestration strategy
2. Implement with `execute_scheme(code)`
3. Use `reset()` between unrelated tasks

---

## State Persistence Between Calls

**IMPORTANT:** All `execute_scheme()` calls share persistent state until `reset()` is called. Understanding this is critical for multi-call pipelines.

### What Persists

- **Scheme variables** (`define` creates variables that survive across calls)
- **Python globals** (variables in `py-exec` persist in Python subprocess)
- **Disk checkpoints** (saved via `checkpoint`, stored in `.rlm-scheme-checkpoints/`)
- **Scope log** (audit trail of all sub-model calls)

### What Doesn't Persist

- Python subprocess state after timeout (>300s default)
- In-progress async handles after errors
- Progress/status from previous calls (call registry resets each execution)

### Design Patterns

**Pattern A: Multi-Call Pipeline (Recommended)**

Split long-running work across multiple `execute_scheme` calls, each under timeout limit:

```scheme
;; Call 1: Load and prepare data (saves to disk)
(py-exec "
import json, os
all_files = [f for f in os.listdir('.') if f.endswith('.py')]
with open('files.json', 'w') as f:
    json.dump(all_files, f)
print(f'Saved {len(all_files)} files')
")
(finish "Data prepared")

;; Call 2: Process data (reads from disk, references Call 1 state)
(define files (py-exec "
with open('files.json') as f:
    print(json.load(f))
"))
(define analyses (map-async
  (lambda (f) (llm-query-async #:instruction "Analyze" #:data f))
  files))
(py-exec "
with open('analyses.json', 'w') as f:
    json.dump(analyses, f)
")
(finish "Analysis complete")

;; Call 3: Synthesize (reads Call 2 results)
(define synthesis (llm-query
  #:instruction "Synthesize all analyses"
  #:data (py-exec "
with open('analyses.json') as f:
    print(json.dumps(json.load(f)))
")))
(finish synthesis)
```

**Pattern B: Checkpoints for Crash Recovery**

Use `checkpoint` to save intermediate results that survive timeouts:

```scheme
;; Phase 1: Expensive computation
(define results (map-async expensive-analysis items))
(checkpoint "phase1_results" results)
(finish "Phase 1 complete")

;; Phase 2: If this times out, Phase 1 results are safe
(define phase1 (restore "phase1_results"))
(if phase1
    (finish (process phase1))
    (finish-error "Phase 1 results not found - run Phase 1 first"))
```

**Pattern C: Inspect State Between Calls**

Use `get_sandbox_state()` MCP tool to debug what persists:

```python
# After running Scheme code
state = get_sandbox_state()
print(state)
# {
#   "scheme_variables": ["context", "result", "summaries"],
#   "python_available": true,
#   "checkpoints": ["phase1_data"],
#   "scope_log_entries": 12
# }
```

### Anti-Patterns (Avoid These)

**❌ Relying on Implicit State**

```scheme
;; Call 1
(define data (py-exec "print(expensive_load())"))

;; Call 2 (hours later) - BUG!
(finish (process data))  ;; ERROR: 'data' may not exist if Call 1 timed out
```

**✅ Fix: Save to disk explicitly**

```scheme
;; Call 1
(define data (py-exec "..."))
(py-exec "
with open('data.json', 'w') as f:
    json.dump(data, f)
")

;; Call 2: Load from disk
(define data (py-exec "
with open('data.json') as f:
    print(json.load(f))
"))
```

**❌ Not Calling reset() Between Unrelated Tasks**

```scheme
;; Task A
(define temp (compute-something))
(finish temp)

;; Task B (unrelated) - BUG!
(define result (process temp))  ;; Accidentally uses Task A's 'temp'
```

**✅ Fix: Call reset() between tasks**

```python
execute_scheme("...")  # Task A
reset()  # Clear state
execute_scheme("...")  # Task B starts fresh
```

### Best Practices

1. **Save to disk for long pipelines** - Use `py-exec` to write JSON files after each major phase
2. **Call `reset()` between unrelated tasks** - Avoid state leakage
3. **Use `checkpoint` for crash recovery** - Persists to disk, survives timeouts
4. **Don't assume Python globals survive timeouts** - They live in a subprocess that may restart
5. **Use `get_sandbox_state()` to debug** - Verify what variables exist before using them

---

## The 16 Patterns

### Speed (Latency Optimization)
1. **Parallel Fan-Out** - Process N items in parallel, synthesize with expensive model
7. **Speculative Execution** - Launch multiple strategies, use first to complete
15. **Stream Processing** - Process data incrementally for constant memory

### Quality (Output Optimization)
4. **Critique-Refine Loop** - Generate, critique with cheap model, refine iteratively
8. **Ensemble Voting** - Run multiple models/prompts, vote on best answer
11. **Consensus Protocol** - Byzantine fault tolerance with model voting

### Cost (Budget Optimization)
9. **Active Learning** - Cheap model on all, expensive on uncertain cases only
14. **Memoization** - Content-addressed caching for repeated queries
16. **Multi-Armed Bandit** - Adaptive model selection based on performance

### Structure (Problem Decomposition)
2. **Code Generation** - LLM generates analysis code, execute in sandbox
3. **Recursive Delegation** - Hierarchical task decomposition with sub-agents
6. **Meta-Orchestration** - LLM designs the orchestration strategy
10. **Tree Aggregation** - Hierarchical reduction for large fan-out results

### Specialized
5. **Cumulative Fold** - Sequential processing with accumulating context
12. **Backtracking Search** - Explore strategy space, backtrack on failure
13. **Anytime Algorithms** - Progressive refinement with quality checkpoints

**Pattern details:** See `docs/patterns/pattern-*.md` files or call `plan_strategy()` for recommendations.

---

## Model Selection

| Task | Model | Cost/1K | When to Use |
|------|-------|---------|-------------|
| **Fan-out** | gpt-3.5-turbo | $0.002 | Parallel processing, extraction (default for bulk work) |
| **Filtering** | ada | $0.0004 | Classification, simple filtering (50× cheaper than GPT-4) |
| **Critique** | curie | $0.002-0.03 | Summarization, comparison, validation (10× cheaper than GPT-4) |
| **Synthesis** | gpt-4 | $0.03-0.06 | Final output, complex reasoning (use sparingly) |
| **Code** | code-davinci-002 | $0.02-0.05 | Code generation, technical tasks |

**Golden rule:** Use gpt-3.5-turbo or ada for 80% of work, gpt-4 only for final synthesis or complex reasoning.

---

## Key Primitives

### LLM Calls
```scheme
;; Synchronous call (blocks until complete)
(llm-query #:instruction "..." #:data "..." #:model "gpt-3.5-turbo")
;; Returns syntax object - use (syntax-e ...) to unwrap

;; Async call (returns immediately)
(llm-query-async #:instruction "..." #:data "..." #:model "gpt-3.5-turbo")
;; Returns future - use (await future) to get result

;; Await multiple futures
(define results (await-all (list future1 future2 future3)))

;; Await first to complete (race)
(define-values (winner rest) (await-any (list f1 f2 f3)))
```

### Parallel Processing
```scheme
;; Map over items in parallel (auto-batching, heartbeat for timeout prevention)
(map-async
  (lambda (item) (llm-query-async #:instruction "..." #:data item))
  items
  #:max-concurrent 10)  ;; Optional: control concurrency
```

### Python Bridge
```scheme
;; Execute Python (prints become return value)
(py-exec "print('hello')")  ;; Returns: "hello"

;; Evaluate Python expression
(py-eval "[1, 2, 3]")  ;; Returns: (list 1 2 3)

;; Transfer data to Python
(py-set! "var_name" scheme-value)

;; Access context in Python
(py-exec "print(context)")  ;; context is loaded data
```

### Context Management
```scheme
;; Access loaded context
context  ;; String of loaded data

;; Get named context slot
(get-context "slot-name")

;; Finish and return
(finish result-value)
```

### Token Budgets
```scheme
;; Set token budget for execution (throws error if exceeded)
(parameterize ([token-budget 1000])
  (llm-query ...))

;; Check remaining budget
(tokens-remaining)
```

### Checkpointing
```scheme
;; Save state to disk (survives timeouts)
(checkpoint! "key-name" value)

;; Restore state
(restore "key-name")  ;; Returns value or #f if not found
```

---

## Common Patterns

### Pattern: Fan-Out + Synthesis
```scheme
;; Process 100 items in parallel, synthesize
(define analyses (map-async
  (lambda (item) (llm-query-async #:instruction "Analyze" #:data item #:model "gpt-3.5-turbo"))
  items))

(define synthesis (syntax-e (llm-query
  #:instruction "Synthesize all analyses"
  #:data (string-join (await-all analyses) "\n")
  #:model "gpt-4")))

(finish synthesis)
```

### Pattern: Critique-Refine
```scheme
;; Generate initial draft
(define draft (syntax-e (llm-query #:instruction "Write draft" #:data context)))

;; Critique with cheap model
(define critique (syntax-e (llm-query
  #:instruction "What are the weaknesses?"
  #:data draft
  #:model "curie")))

;; Refine based on critique
(define refined (syntax-e (llm-query
  #:instruction (string-append "Improve based on: " critique)
  #:data draft
  #:model "gpt-4")))

(finish refined)
```

### Pattern: Active Learning (Cost Optimization)
```scheme
;; Phase 1: Cheap model on all items
(define cheap-results (map-async
  (lambda (item) (llm-query-async #:instruction "Analyze + confidence score" #:data item #:model "gpt-3.5-turbo"))
  items))

;; Phase 2: Identify uncertain cases (via Python)
(py-set! "results" (await-all cheap-results))
(define uncertain-indices (py-eval "
[i for i, r in enumerate(results) if 'confidence: low' in r.lower()]
"))

;; Phase 3: Expensive model only on uncertain
(define refined (map-async
  (lambda (idx) (llm-query-async #:instruction "Deep analysis" #:data (list-ref items idx) #:model "gpt-4"))
  uncertain-indices))
```

### Pattern: Strategy Exploration
```scheme
;; Launch 3 different strategies in parallel
(define strategy-a (llm-query-async #:instruction "Strategy A: ..." #:data context))
(define strategy-b (llm-query-async #:instruction "Strategy B: ..." #:data context))
(define strategy-c (llm-query-async #:instruction "Strategy C: ..." #:data context))

;; Cheap model compares and selects winner
(define comparison (syntax-e (llm-query
  #:instruction "Compare these 3 results. Return JSON: {winner: 'a'|'b'|'c', reasoning: str}"
  #:data (string-append "A: " (await strategy-a) "\n\nB: " (await strategy-b) "\n\nC: " (await strategy-c))
  #:json #t
  #:model "curie")))

;; Use winner approach for full dataset
```

---

## Tool Reference

### Available MCP Tools
- **get_usage_guide()** - Returns this guide
- **plan_strategy(task, data_chars, constraints, priority)** - Get strategy recommendations
- **execute_scheme(code, timeout)** - Execute Scheme orchestration code
- **load_context(data, name)** - Load data into sandbox
- **get_code_generation_api_reference()** - API reference for code-gen patterns
- **reset()** - Clear sandbox state
- **get_status()** - View active calls, token usage, rate limits
- **get_scope_log()** - View scope tracking audit trail
- **cancel_call(call_id)** - Cancel in-flight LLM call

---

## Strategy Design Workflow

**Recommended approach:**

1. **Plan First** - Call `plan_strategy()` with your task description:
   ```python
   plan_strategy(
       task_description="Analyze 200 research papers for AMR genes",
       data_characteristics="~5KB per paper, 1MB total",
       priority="balanced"
   )
   ```

2. **Review Recommendations** - Planner returns:
   - Primary strategy (pattern composition, model assignments, cost/latency estimates)
   - 2 alternatives with tradeoffs
   - 1-2 experimental options

3. **Implement** - Use provided code template with `execute_scheme()`

4. **Iterate** - If unsatisfied, try alternatives or creative options

---

## Best Practices

### Cost Optimization
- Use gpt-3.5-turbo (not gpt-4) for fan-out operations
- Use ada for simple filtering/classification (50× cheaper)
- Reserve gpt-4 for final synthesis only
- Test strategies on 10-20% sample before scaling

### Latency Optimization
- Use `map-async` for parallel processing (10× faster than sequential)
- Use `#:max-concurrent` to control parallelism vs rate limits
- Consider speculative execution for critical-path operations

### Quality Optimization
- Add critique-refine loop for high-stakes outputs
- Use ensemble voting when single model output is uncertain
- Validate with cheap model before expensive synthesis

### Memory Management
- Use stream processing for large datasets (>100MB)
- Checkpoint intermediate results for long-running jobs
- Clear context with `reset()` between unrelated tasks

---

## Examples by Use Case

### Document Analysis (100+ documents)
**Pattern:** Fan-Out + Tree Aggregation
```scheme
(define extractions (map-async extract-fn documents #:model "gpt-3.5-turbo"))
(define synthesis (tree-reduce aggregate-fn extractions #:model "gpt-4"))
```

### Code Generation (multiple files)
**Pattern:** Code Generation + Critique-Refine
```scheme
(define code (generate-code-via-llm spec))
(define critique (critique-code code))
(define refined (refine-based-on-critique code critique))
```

### Unknown Data Structure
**Pattern:** Meta-Orchestration
```scheme
(define strategy-plan (llm-query #:instruction "Analyze data, recommend strategy" #:data sample))
(execute-recommended-strategy strategy-plan)
```

### Budget-Constrained (<$1)
**Pattern:** Active Learning + Memoization
```scheme
(define cheap-pass (map-async cheap-fn items))
(define uncertain (identify-uncertain cheap-pass))
(define refined (map-async expensive-fn uncertain))
```

---

## Debugging

### View Active Calls
```python
get_status()  # Returns JSON with active calls, token usage, rate limits
```

### Cancel Long-Running Call
```python
# Get call ID from get_status()
cancel_call("call_123")
```

### View Scope Tracking
```python
get_scope_log()  # Returns audit trail of data movement
```

### Common Issues
- **Timeout:** Increase timeout with `execute_scheme(code, timeout=600)`
- **Rate limits:** Reduce `#:max-concurrent` in map-async
- **Out of memory:** Use stream processing or checkpointing
- **High cost:** Profile with get_status(), switch to cheaper models

---

## Further Reading

- **Pattern details:** `docs/patterns/pattern-*.md` (16 files with full examples)
- **API reference:** Call `get_code_generation_api_reference()`
- **Best practices:** `docs/best-practices.md`
- **Primitive reference:** `docs/primitive-reference.md`

**Start here:** Call `plan_strategy()` with your task to get personalized recommendations.
