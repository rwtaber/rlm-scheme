# Combinator Library Reference

## Overview

Combinators are higher-order functions for composing orchestration strategies. Instead of manually coding patterns, you compose ~17 core combinators to create custom strategies.

### Why Combinators?

**Before (Pattern Catalog):**
- 16 enumerated patterns
- Users pick from list or add new pattern
- `plan_strategy` returns pattern IDs: `[1, 4, 10]`

**After (Combinator Library):**
- ~17 core combinators
- Users compose custom strategies
- `plan_strategy` returns executable compositions
- 16 patterns become "named compositions" (reference implementations)
- Infinite strategy space via composition

### Benefits

1. ✅ **Finite primitive set** (~17 combinators vs growing pattern list)
2. ✅ **Infinite compositional space** (no need to enumerate all possibilities)
3. ✅ **Clear semantics** (combinator algebra vs ad-hoc patterns)
4. ✅ **Better tooling** (`plan_strategy` generates executable code)
5. ✅ **Easier learning** (learn combinators + composition rules vs 16+ patterns)

---

## Parallel Combinators

### `parallel`

**Signature:** `(parallel strategies #:max-concurrent N)`

**Description:** Execute strategies concurrently, return all results as list.

**Parameters:**
- `strategies`: List of thunks (zero-arg functions)
- `#:max-concurrent`: Max concurrent executions (default: no limit)

**Returns:** List of results (order preserved)

**Example:**
```scheme
(define results (parallel
  (list
    (lambda () (llm-query-async #:instruction "Approach A" #:data context))
    (lambda () (llm-query-async #:instruction "Approach B" #:data context))
    (lambda () (llm-query-async #:instruction "Approach C" #:data context)))
  #:max-concurrent 3))

;; Returns: list of 3 async handles
```

**Composition:** Composes with `vote` for ensemble voting

**Performance:** O(N/k) where N=strategies, k=max-concurrent

**Cost:** Sum of all strategy costs

---

### `race`

**Signature:** `(race strategies)`

**Description:** First to complete wins, cancel others.

**Parameters:**
- `strategies`: List of thunks that return async handles

**Returns:** Result of first completed strategy

**Example:**
```scheme
(define fastest (race
  (list
    (lambda () (llm-query-async #:instruction "Fast approach" #:model "gpt-4.1-nano"))
    (lambda () (llm-query-async #:instruction "Slow approach" #:model "gpt-4o"))
    (lambda () (llm-query-async #:instruction "Medium approach" #:model "gpt-4o-mini")))))

;; Returns: result from whichever completes first
```

**Composition:** Use with `try-fallback` for timeout handling

**Performance:** O(1) - returns as soon as first completes

**Cost:** Cost of fastest strategy

---

## Sequential Combinators

### `sequence`

**Signature:** `(sequence fn1 fn2 fn3 ...)`

**Description:** Chain functions left-to-right: fn1 → fn2 → fn3

**Parameters:**
- `fn1`, `fn2`, ... : Functions to chain

**Returns:** Function that applies all transformations in sequence

**Example:**
```scheme
(define pipeline (sequence
  extract-fn
  summarize-fn
  synthesize-fn))

(define result (pipeline data))
;; Equivalent to: (synthesize-fn (summarize-fn (extract-fn data)))
```

**Composition:** Composes with any combinator

**Performance:** O(N) where N=number of functions

**Algebraic Property:** Associative

---

### `fold-sequential`

**Signature:** `(fold-sequential fn init items)`

**Description:** Sequential fold with accumulator.

**Parameters:**
- `fn`: Function `(accumulator item -> new-accumulator)`
- `init`: Initial accumulator value
- `items`: List of items to process

**Returns:** Final accumulator

**Example:**
```scheme
(define summary (fold-sequential
  (lambda (acc doc)
    (string-append acc "\n" (syntax-e (llm-query #:instruction "Summarize" #:data doc))))
  "Summary:"
  documents))
```

**Composition:** Use with `tree-reduce` for parallel-then-sequential patterns

**Performance:** O(N) sequential operations

---

## Hierarchical Combinators

### `tree-reduce`

**Signature:** `(tree-reduce fn items #:branch-factor N #:leaf-fn leaf)`

**Description:** Recursive tree aggregation with branching factor.

**Parameters:**
- `fn`: Function to combine results at each level
- `items`: List of items to reduce
- `#:branch-factor`: Max items per group (default: 5)
- `#:leaf-fn`: Optional transformation applied to leaves (default: identity)

**Returns:** Single reduced value

**Example:**
```scheme
(define summary (tree-reduce
  (lambda (left right)
    (syntax-e (llm-query #:instruction "Combine summaries"
                         #:data (string-append left "\n" right))))
  extractions
  #:branch-factor 5))
```

**Composition:** Composes with `fan-out-aggregate` for map-reduce patterns

**Performance:** O(log_k(N)) depth where k=branch-factor

**Cost:** O(N) total LLM calls (each item processed once)

---

### `recursive-spawn`

**Signature:** `(recursive-spawn strategy #:depth N)`

**Description:** Delegate to sub-sandbox with recursion.

**Parameters:**
- `strategy`: Thunk that returns a strategy description
- `#:depth`: Maximum recursion depth (default: 1)

**Returns:** Function that executes strategy recursively

**Example:**
```scheme
(define recursive-analyzer (recursive-spawn
  (lambda () "Analyze this data and break it into sub-problems if complex")
  #:depth 2))

(define result (recursive-analyzer complex-data))
```

**Composition:** Use sparingly - increases latency

**Performance:** O(depth) nested sandbox launches

---

### `fan-out-aggregate`

**Signature:** `(fan-out-aggregate map-fn reduce-fn items #:max-concurrent N)`

**Description:** Parallel map + hierarchical reduce in one combinator.

**Parameters:**
- `map-fn`: Function applied to each item (should return async handle or value)
- `reduce-fn`: Function to combine mapped results
- `items`: List of items to process
- `#:max-concurrent`: Max concurrent map operations (default: 20)

**Returns:** Aggregated result

**Example:**
```scheme
(define summary (fan-out-aggregate
  ;; Map phase: extract from each document
  (lambda (doc)
    (llm-query-async #:instruction "Extract key points"
                     #:data doc
                     #:model "gpt-4.1-nano"))
  ;; Reduce phase: combine extractions
  (lambda (extractions)
    (tree-reduce
      (lambda (left right)
        (syntax-e (llm-query #:instruction "Combine"
                             #:data (string-append left "\n" right))))
      extractions
      #:branch-factor 5))
  documents))
```

**Composition:** The workhorse combinator - composes with everything

**Performance:** O(N/k) parallel + O(log(N)) reduce where k=max-concurrent

**Cost:** O(N) map calls + O(N) reduce calls

---

## Iterative Combinators

### `iterate-until`

**Signature:** `(iterate-until fn pred init #:max-iter N)`

**Description:** Repeat until predicate or max iterations.

**Parameters:**
- `fn`: Function to apply iteratively `(value -> new-value)`
- `pred`: Predicate to test termination `(value -> boolean)`
- `init`: Initial value
- `#:max-iter`: Maximum iterations (default: 10)

**Returns:** Final value

**Example:**
```scheme
(define refined (iterate-until
  (lambda (draft)
    (syntax-e (llm-query #:instruction "Improve this draft" #:data draft)))
  (lambda (draft)
    (string-contains? (string-downcase draft) "final"))
  initial-draft
  #:max-iter 5))
```

**Composition:** Use with `with-validation` for quality gates

**Performance:** O(iterations) - early termination possible

---

### `critique-refine`

**Signature:** `(critique-refine generate-fn critique-fn refine-fn #:max-iter N #:quality-threshold T)`

**Description:** Generate, critique, refine loop for iterative quality improvement.

**Parameters:**
- `generate-fn`: Thunk that generates initial draft
- `critique-fn`: Function `(draft -> critique-text)`
- `refine-fn`: Function `(draft critique -> refined-draft)`
- `#:max-iter`: Maximum refinement iterations (default: 3)
- `#:quality-threshold`: Optional quality threshold (not fully implemented)

**Returns:** Final refined result

**Example:**
```scheme
(define final-report (critique-refine
  (lambda () (syntax-e (llm-query #:instruction "Write report" #:data context)))
  (lambda (draft)
    (syntax-e (llm-query #:instruction "Critique this report"
                         #:data draft
                         #:model "gpt-4o-mini")))
  (lambda (draft critique)
    (syntax-e (llm-query #:instruction "Refine based on critique"
                         #:data (string-append "Draft:\n" draft "\n\nCritique:\n" critique)
                         #:model "gpt-4o")))
  #:max-iter 3))
```

**Composition:** Composes with `fan-out-aggregate` for multi-document refinement

**Performance:** O(max-iter) iterations

**Cost:** (generate cost) + (max-iter × (critique cost + refine cost))

---

## Quality Combinators

### `with-validation`

**Signature:** `(with-validation fn validator)`

**Description:** Wrap function with validation step.

**Parameters:**
- `fn`: Function to execute
- `validator`: Function `(result -> boolean)` that returns true if valid

**Returns:** Wrapped function that validates before returning

**Example:**
```scheme
(define validated-extract (with-validation
  (lambda (doc)
    (syntax-e (llm-query #:instruction "Extract entities" #:data doc #:json #t)))
  (lambda (result)
    (string-contains? result "{"))))  ; Validate JSON output

(define result (validated-extract document))
```

**Composition:** Use in pipelines for quality gates

**Performance:** O(1) validation overhead

---

### `vote`

**Signature:** `(vote strategies #:method majority)`

**Description:** Multi-strategy voting (majority, plurality, consensus).

**Parameters:**
- `strategies`: List of thunks that return results
- `#:method`: Voting method - `'majority`, `'plurality`, or `'consensus` (default: `'majority`)

**Returns:** Winning result based on voting method

**Voting Methods:**
- **`majority`**: Requires >50% agreement (errors if no majority)
- **`plurality`**: Most votes wins (ties go to first)
- **`consensus`**: All strategies must agree (errors if disagreement)

**Example:**
```scheme
(define classification (vote
  (list
    (lambda () (syntax-e (llm-query #:instruction "Classify" #:data doc #:model "gpt-4o")))
    (lambda () (syntax-e (llm-query #:instruction "Classify" #:data doc #:model "claude-3.5-sonnet")))
    (lambda () (syntax-e (llm-query #:instruction "Classify" #:data doc #:model "gpt-4o"))))
  #:method 'majority))
```

**Composition:** Use with `parallel` for concurrent execution before voting

**Performance:** O(N) where N=number of strategies

**Cost:** Sum of all strategy costs

---

### `ensemble`

**Signature:** `(ensemble strategies #:aggregator fn)`

**Description:** Multi-model ensemble with custom aggregation.

**Parameters:**
- `strategies`: List of thunks that return results
- `#:aggregator`: Optional custom aggregation function (default: concatenate with labels)

**Returns:** Aggregated result

**Example:**
```scheme
(define multi-model-summary (ensemble
  (list
    (lambda () (syntax-e (llm-query #:instruction "Summarize" #:data doc #:model "gpt-4o")))
    (lambda () (syntax-e (llm-query #:instruction "Summarize" #:data doc #:model "claude-3.5-sonnet"))))
  #:aggregator (lambda (results)
                (syntax-e (llm-query #:instruction "Synthesize these summaries"
                                     #:data (string-join results "\n---\n"))))))
```

**Composition:** Use for quality-critical tasks

**Performance:** O(N) strategies + O(1) aggregation

---

## Cost Combinators

### `tiered`

**Signature:** `(tiered cheap-fn expensive-fn items)`

**Description:** Cheap function on all items, expensive function for synthesis.

**Parameters:**
- `cheap-fn`: Fast/cheap function applied to all items
- `expensive-fn`: Slow/expensive function for final synthesis
- `items`: List of items

**Returns:** Result of expensive function applied to cheap results

**Example:**
```scheme
(define summary (tiered
  ;; Cheap: extract with nano model
  (lambda (doc)
    (syntax-e (llm-query-async #:instruction "Extract key points"
                                #:data doc
                                #:model "gpt-4.1-nano")))
  ;; Expensive: synthesize with gpt-4o
  (lambda (extractions)
    (syntax-e (llm-query #:instruction "Synthesize summary"
                         #:data (string-join extractions "\n")
                         #:model "gpt-4o")))
  documents))
```

**Composition:** Foundational pattern - composes with everything

**Performance:** O(N) cheap + O(1) expensive

**Cost:** (N × cheap) + (1 × expensive)

---

### `active-learning`

**Signature:** `(active-learning cheap-fn expensive-fn uncertainty-fn items #:threshold T)`

**Description:** Cheap model on all items, expensive model only on uncertain cases for cost optimization.

**Parameters:**
- `cheap-fn`: Fast function applied to all items
- `expensive-fn`: Expensive function for uncertain items
- `uncertainty-fn`: Function to assess uncertainty `(result -> score 0-1)`
- `items`: List of items
- `#:threshold`: Uncertainty threshold (default: 0.7)

**Returns:** List of results (cheap where certain, expensive where uncertain)

**Example:**
```scheme
(define extractions (active-learning
  ;; Cheap pass
  (lambda (doc)
    (syntax-e (llm-query-async #:instruction "Extract entities"
                                #:data doc
                                #:model "gpt-4o-mini")))
  ;; Expensive refinement
  (lambda (doc)
    (syntax-e (llm-query #:instruction "Extract entities carefully"
                         #:data doc
                         #:model "gpt-4o")))
  ;; Uncertainty measure
  (lambda (result)
    (/ (length (string-split result ",")) 10.0))  ; Simple: fewer entities = more uncertain
  documents
  #:threshold 0.7))
```

**Composition:** Use when cost vs quality trade-off is critical

**Performance:** O(N) cheap + O(U) expensive where U=uncertain count

**Cost:** (N × cheap) + (U × expensive)

---

### `memoized`

**Signature:** `(memoized fn #:key-fn hash-fn)`

**Description:** Cache results by content hash.

**Parameters:**
- `fn`: Function to memoize
- `#:key-fn`: Optional key extraction function (default: `identity`)

**Returns:** Memoized version of fn

**Example:**
```scheme
(define extract-memo (memoized
  (lambda (doc)
    (syntax-e (llm-query #:instruction "Extract" #:data doc)))
  #:key-fn (lambda (args)
            ;; Hash on first 100 chars to avoid exact duplicates
            (substring (car args) 0 (min 100 (string-length (car args)))))))

(define result1 (extract-memo doc1))  ; Calls LLM
(define result2 (extract-memo doc1))  ; Uses cache
```

**Composition:** Use with `map-async` for deduplication

**Performance:** O(1) cache lookup

**Cost:** Only pays for unique inputs

---

## Control Flow Combinators

### `choose`

**Signature:** `(choose pred then-fn else-fn)`

**Description:** Conditional execution based on predicate.

**Parameters:**
- `pred`: Predicate function or boolean
- `then-fn`: Function to execute if pred is true
- `else-fn`: Function to execute if pred is false

**Returns:** Wrapped function that chooses branch

**Example:**
```scheme
(define smart-process (choose
  (lambda (data) (< (string-length data) 1000))  ; Small dataset?
  (lambda (data) (syntax-e (llm-query #:instruction "Simple" #:data data #:model "gpt-4o-mini")))
  (lambda (data) (syntax-e (llm-query #:instruction "Complex" #:data data #:model "gpt-4o")))))

(define result (smart-process data))
```

**Composition:** Use for adaptive strategies

**Performance:** O(predicate) + O(chosen branch)

---

### `try-fallback`

**Signature:** `(try-fallback primary-fn fallback-fn)`

**Description:** Try primary, use fallback on error.

**Parameters:**
- `primary-fn`: Primary function to try
- `fallback-fn`: Fallback function if primary fails

**Returns:** Wrapped function with error handling

**Example:**
```scheme
(define robust-parse (try-fallback
  (lambda (text)
    (syntax-e (llm-query #:instruction "Parse as JSON"
                         #:data text
                         #:json #t)))
  (lambda (text)
    (syntax-e (llm-query #:instruction "Extract key-value pairs loosely"
                         #:data text)))))

(define result (robust-parse messy-data))
```

**Composition:** Use for robustness in pipelines

**Performance:** O(primary) on success, O(primary + fallback) on failure

---

## Composition Rules

### Valid Compositions

✅ **fan-out-aggregate → critique-refine**: Refine aggregated result
```scheme
(critique-refine
  (lambda () (fan-out-aggregate extract-fn reduce-fn items))
  critique-fn
  refine-fn)
```

✅ **parallel → vote**: Parallel strategies, vote on result
```scheme
(vote (parallel strategies) #:method 'majority)
```

✅ **tiered → tree-reduce**: Cheap extraction, hierarchical reduction
```scheme
(tiered extract-fn
        (lambda (results) (tree-reduce combine results))
        items)
```

✅ **sequence → with-validation**: Pipeline with quality gates
```scheme
(sequence
  (with-validation step1 validate1)
  (with-validation step2 validate2))
```

### Invalid Compositions

❌ **race → tree-reduce**: Race produces 1 result, tree-reduce needs many

❌ **recursive-spawn → recursive-spawn**: Exceeds depth limit

### Composition Algebra

**Associativity:**
```scheme
(sequence a (sequence b c))  ≡  (sequence (sequence a b) c)  ≡  (sequence a b c)
```

**Identity:**
```scheme
(sequence identity f)  ≡  f  ≡  (sequence f identity)
```

**Distributivity (parallel over sequence):**
```scheme
(sequence (parallel [a b]) f)  ≡  (parallel [(sequence a f) (sequence b f)])
```

---

## Usage Examples

### Example 1: Multi-Phase Validated Pipeline

```scheme
(define result
  (sequence
    ;; Phase 1: Extract with validation
    (with-validation
      (lambda (docs)
        (map-async
          (lambda (doc) (llm-query-async #:instruction "Extract" #:data doc))
          docs))
      (lambda (results) (> (length results) 0)))

    ;; Phase 2: Refine with critique
    (lambda (extractions)
      (critique-refine
        (lambda () (tree-reduce combine extractions))
        critique-fn
        refine-fn))

    ;; Phase 3: Final validation
    (with-validation
      identity
      (lambda (result) (string-contains? result "conclusion")))))

(finish ((result) documents))
```

### Example 2: Ensemble with Fallback

```scheme
(define robust-ensemble (try-fallback
  (lambda (doc)
    (ensemble
      (list
        (lambda () (llm-query #:model "gpt-4o" #:instruction "Analyze" #:data doc))
        (lambda () (llm-query #:model "claude-3.5-sonnet" #:instruction "Analyze" #:data doc)))
      #:aggregator (lambda (results)
                    (llm-query #:instruction "Synthesize" #:data (string-join results "\n")))))
  (lambda (doc)
    ;; Fallback: single strong model
    (llm-query #:model "gpt-4o" #:instruction "Analyze deeply" #:data doc))))

(finish (robust-ensemble document))
```

### Example 3: Adaptive Cost Strategy

```scheme
(define smart-extract (choose
  (lambda (items) (< (length items) 100))
  ;; Small dataset: use expensive model directly
  (lambda (items)
    (map-async
      (lambda (item) (llm-query-async #:model "gpt-4o" #:instruction "Extract" #:data item))
      items))
  ;; Large dataset: use active learning
  (lambda (items)
    (active-learning
      (lambda (item) (llm-query-async #:model "gpt-4o-mini" #:instruction "Extract" #:data item))
      (lambda (item) (llm-query-async #:model "gpt-4o" #:instruction "Extract carefully" #:data item))
      uncertainty-fn
      items))))

(finish (smart-extract documents))
```

---

## Next Steps

- **Learn composition techniques:** See `composition-guide.md` for in-depth combinator composition patterns
- **Get started:** See `getting-started.md` for quick start guide
- **See examples:** See `examples.md` for working code examples
- **Get recommendations:** Use `plan_strategy()` tool to get combinator strategies for your specific task
- **Understand execution:** See `execution-model.md` for sandbox state and async operations
