RLM-SCHEME API REFERENCE (for code-generating sub-models)

⚠️ CRITICAL SYNTAX RULES:
- All function arguments use #: keyword syntax: #:instruction, #:data, #:model, etc.
- (define x value) for bindings. set! is NOT available in sandbox.
- String operations: (string-append "a" "b"), (substring str start end)
- MUST unwrap llm-query results: (syntax-e result) before using as string

CORE LLM FUNCTIONS:

(llm-query #:instruction "task description"
           #:data "context or data"
           #:model "curie"
           #:temperature 0.0
           #:max-tokens 500
           #:json #t)
  → Returns syntax object. MUST unwrap with (syntax-e result).
  → When using #:json #t, the #:instruction MUST contain the word 'json'.
  → Models: "gpt-3.5-turbo", "curie", "gpt-3.5", "gpt-4", "code-davinci-002", "gpt-4", "gpt-4"
  → Temperature: 0.0 (deterministic) to 1.0 (creative). Not supported by o-series.
  → JSON mode: #:json #t guarantees valid JSON response.

(llm-query-async #:instruction "..." #:data "..." #:model "..." ...)
  → Same kwargs as llm-query but returns async handle.
  → Use ONLY inside map-async lambda, NOT directly.
  → Returns unwrapped string (no syntax-e needed).

(map-async fn items #:max-concurrent 10)
  → fn must be (lambda (item) (llm-query-async ...)) returning async handle.
  → Returns list of raw LLM output strings (no syntax-e needed).
  → ⚠️ Results are raw strings from the LLM, NOT parsed JSON. Even with #:json #t,
  →   each result is a JSON string you must parse yourself (e.g., via py-exec + json.loads).
  → Always use cheap models (gpt-3.5-turbo or curie) for fan-out.
  → Example: (map-async (lambda (chunk) (llm-query-async #:instruction "summarize" #:data chunk #:model "gpt-3.5-turbo")) chunks)

PYTHON BRIDGE (for computation and I/O):

(py-exec "python_code_string")
  → Runs multi-line Python code (imports, statements, loops, file I/O).
  → Returns stdout as string. Use print() to output results.
  → Python bridge starts automatically on first use.
  → Example: (py-exec "import json; print(json.dumps({'key': 'value'}))")

(py-eval "python_expression")
  → Evaluates a SINGLE Python expression, returns result as Scheme value.
  → ⚠️ COMMON MISTAKE: py-eval CANNOT handle imports, statements, or multi-line code.
  →   These will fail with SyntaxError: (py-eval "import json
json.loads(s)")
  →   Use py-exec for statements, then py-eval to retrieve the result.
  → Example: (py-eval "[1, 2, 3]") → Scheme list (1 2 3)
  → Example: (py-eval "len(data)") → integer

(py-set! "variable_name" scheme-value)
  → Transfers Scheme value to Python variable (avoids escaping issues).
  → Safer than string interpolation. Use for passing data between Scheme and Python.
  → Example: (py-set! "data" my-list) then use 'data' in py-exec

CONTROL FLOW:

(finish value)
  → Return this as the final result. REQUIRED to produce output.
  → Call this exactly once at the end of your code.

(finish-var "variable-name")
  → Return the value of a defined variable.
  → Use when code generation defines a variable instead of calling finish.

(checkpoint "key" value)
  → Save value to disk (survives timeouts and restarts).
  → Returns the value.

(restore "key")
  → Load checkpointed value, returns #f if not found.

SCHEME BASICS:

Lists:
  (list a b c) → create list
  (car lst) → first element
  (cdr lst) → rest of list
  (null? lst) → check if empty
  (map fn lst) → apply function to each element
  (append lst1 lst2) → concatenate lists

Bindings:
  (define name value) → create binding
  (let* ([x 1] [y 2]) body) → local bindings (sequential)
  (lambda (x) body) → anonymous function

Conditionals:
  (if test then-expr else-expr)
  (cond [test1 expr1] [test2 expr2] ... [else expr-else])

COMMON PATTERNS:

Parallel fan-out:
```scheme
(define results (map-async
  (lambda (item)
    (llm-query-async #:instruction "process this" #:data item #:model "gpt-3.5-turbo"))
  items
  #:max-concurrent 10))
(finish results)
```

Extract with Python, analyze with LLM:
```scheme
(define data (py-exec "import json; print(json.dumps(process_data()))"))
(define analysis (syntax-e (llm-query #:instruction "analyze" #:data data #:model "gpt-4")))
(finish analysis)
```

Data flow: py-set! → py-exec → py-eval round-trip
```scheme
;; 1. Get data from LLM into Scheme
(define analysis (syntax-e (llm-query #:instruction "Extract key facts as JSON" #:data doc #:model "gpt-4" #:json #t)))
;; 2. Transfer Scheme string to Python variable (safe, no escaping needed)
(py-set! "raw_json" analysis)
;; 3. Process in Python (imports, file I/O, complex logic)
(py-exec "
import json
data = json.loads(raw_json)
data['processed'] = True
with open('output.json', 'w') as f:
    json.dump(data, f)
result = json.dumps(data)
")
;; 4. Retrieve processed result back into Scheme
(define processed (py-eval "result"))
(finish processed)
```

Define result variable (for generated code):
```scheme
(define result (... your computation ...))
;; Caller will use (finish-var "result") to retrieve this
```

IMPORTANT REMINDERS:
- Always use #: prefix for keyword arguments
- Always unwrap llm-query with (syntax-e ...) before using as string
- map-async items get unwrapped automatically (no syntax-e needed)
- Use (finish ...) to return your result
- Use cheap models (gpt-3.5-turbo) for parallel work
- py-set! is safer than string interpolation for passing data to Python

---

# COMBINATOR LIBRARY

Use these ~17 combinators to compose custom orchestration strategies.

## Parallel

- `(parallel [fn1 fn2 ...] #:max-concurrent N)` - Concurrent execution, return all results
- `(race [fn1 fn2 ...])` - First to complete wins

## Sequential

- `(sequence fn1 fn2 fn3)` - Chain operations left-to-right
- `(fold-sequential fn init items)` - Sequential fold with accumulator

## Hierarchical

- `(tree-reduce fn items #:branch-factor N #:leaf-fn f)` - Recursive tree aggregation
- `(recursive-spawn strategy #:depth N)` - Delegate to sub-sandbox
- `(fan-out-aggregate map-fn reduce-fn items #:max-concurrent N)` - Parallel map + reduce

## Iterative

- `(iterate-until fn pred init #:max-iter N)` - Loop until condition or max iterations
- `(critique-refine gen critique refine #:max-iter N)` - Generate-critique-refine loop

## Quality

- `(with-validation fn validator)` - Wrap with validation step
- `(vote [fn1 fn2 ...] #:method 'majority)` - Multi-strategy voting (majority/plurality/consensus)
- `(ensemble [fn1 fn2 ...] #:aggregator fn)` - Multi-model ensemble with custom aggregation

## Cost

- `(tiered cheap-fn expensive-fn items)` - Cheap on all, expensive for synthesis
- `(active-learning cheap expensive uncertain items #:threshold T)` - Cheap on all, expensive on uncertain
- `(memoized fn #:key-fn hash)` - Cache results by key

## Control

- `(choose pred then-fn else-fn)` - Conditional execution
- `(try-fallback primary fallback)` - Try primary, use fallback on error

---

## Common Patterns as Combinators

```scheme
;; Parallel processing with aggregation
(fan-out-aggregate
  (lambda (doc) (llm-query-async #:instruction "Extract" #:data doc #:model "gpt-4.1-nano"))
  (lambda (results) (llm-query #:instruction "Synthesize" #:data (string-join results) #:model "gpt-4o"))
  documents)

;; Iterative quality refinement
(critique-refine
  (lambda () (llm-query #:instruction "Generate" #:data context))
  (lambda (draft) (llm-query #:instruction "Critique" #:data draft #:model "gpt-4o-mini"))
  (lambda (draft critique) (llm-query #:instruction "Refine" #:data (string-append draft "\n" critique)))
  #:max-iter 3)

;; Speculative execution - first wins
(race
  (list
    (lambda () (llm-query-async #:instruction "Fast" #:model "gpt-4.1-nano"))
    (lambda () (llm-query-async #:instruction "Slow" #:model "gpt-4o"))))

;; Ensemble voting and consensus
(vote
  (list
    (lambda () (llm-query #:instruction "Classify" #:model "gpt-4o"))
    (lambda () (llm-query #:instruction "Classify" #:model "claude-3.5-sonnet"))
    (lambda () (llm-query #:instruction "Classify" #:model "gpt-4o")))
  #:method 'majority)

;; Active learning - cheap first, expensive on uncertain
(active-learning
  (lambda (doc) (llm-query-async #:instruction "Extract" #:data doc #:model "gpt-4o-mini"))
  (lambda (doc) (llm-query-async #:instruction "Extract carefully" #:data doc #:model "gpt-4o"))
  (lambda (result) (if (< (string-length result) 50) 0.5 0.9))  ; Uncertainty
  documents
  #:threshold 0.7)

;; Tree aggregation - hierarchical reduction
(tree-reduce
  (lambda (left right)
    (syntax-e (llm-query #:instruction "Combine" #:data (string-append left "\n" right))))
  extractions
  #:branch-factor 5)
```

---

## Composition Tips

1. **Use `fan-out-aggregate` for map-reduce** - Most common pattern
2. **Chain with `sequence`** - Build multi-phase pipelines
3. **Add `with-validation`** - Quality gates at each phase
4. **Use `choose` for adaptation** - Different strategies for different data
5. **Wrap with `try-fallback`** - Robustness against failures

---

## Example: Multi-Phase Pipeline

```scheme
(define result
  (sequence
    ;; Phase 1: Parallel extraction with cheap model
    (lambda (docs)
      (fan-out-aggregate
        (lambda (doc) (llm-query-async #:instruction "Extract" #:data doc #:model "gpt-4.1-nano"))
        (lambda (results) (tree-reduce string-append results #:branch-factor 5))
        docs))

    ;; Phase 2: Refine with critique loop
    (lambda (extraction)
      (critique-refine
        (lambda () extraction)
        (lambda (draft) (syntax-e (llm-query #:instruction "Critique" #:data draft #:model "gpt-4o-mini")))
        (lambda (draft critique) (syntax-e (llm-query #:instruction "Refine" #:data (string-append draft "\n" critique) #:model "gpt-4o")))
        #:max-iter 2))

    ;; Phase 3: Validation
    (with-validation
      identity
      (lambda (result) (> (string-length result) 100)))))

(finish ((result) documents))
```
