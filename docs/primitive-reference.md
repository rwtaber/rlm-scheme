## Part IV: Complete Primitive Reference

This section documents all 27 scaffolding primitives available in rlm-scheme, organized by capability.

---

### 4.1 Core Primitives (4)

#### `(finish result-string)`
**Purpose:** Exit sandbox with final result
**Why it exists:** Every orchestration must produce output. This is the only way to return results to the host.
**Example:**
```scheme
(finish "Analysis complete: 95% accuracy achieved")
```
**Use when:** Ready to return final result
**Common mistake:** Calling finish in sub-sandbox (forbidden). Only host-level sandbox can finish.
**Seen in:** All patterns

---

#### `(finish-var variable-name)`
**Purpose:** Return Python variable as result
**Why it exists:** Sub-sandbox return mechanism for code generation patterns
**Example:**
```scheme
;; In sub-sandbox generated code:
(py-exec "final_report = 'Analysis complete'")
(finish-var "final_report")
```
**Use when:** Code generation patterns, sub-sandbox results
**Common mistake:** Using finish instead of finish-var in sub-sandbox
**Seen in:** Patterns 2, 6 (meta-orchestration)

---

#### `context`
**Purpose:** Access input data loaded via load_context
**Why it exists:** Data must be accessible to orchestration code without manual passing
**Example:**
```scheme
(define papers context)  ;; Access loaded papers
(llm-query #:data context ...)
```
**Use when:** Need to access input data
**Common mistake:** Assuming context is always a string (may be any data type)
**Seen in:** Patterns 1-16 (all)

---

#### `(get-context name)`
**Purpose:** Retrieve named context slot
**Why it exists:** Multiple datasets can be loaded with different names
**Example:**
```scheme
;; After load_context with name parameter:
(define gwas-data (get-context "gwas-data"))
(define control-data (get-context "control-data"))
```
**Use when:** Working with multiple named datasets
**Common mistake:** Forgetting to load context with a name first
**Seen in:** Multi-dataset patterns

---

### 4.2 State Management (2)

#### `(checkpoint key value)`
**Purpose:** Persist state for fault tolerance and caching
**Why it exists:** Long-running orchestrations need recovery. Repeated queries need caching.
**Example:**
```scheme
(checkpoint "phase1_results" results)
(checkpoint (hash-key input) output)  ;; Caching
```
**Use when:** After expensive operations, before risky operations, for caching
**Common mistake:** Not checkpointing before multi-hour operations
**Seen in:** Patterns 4 (refinement), 13 (anytime), 14 (memoization), 15 (streaming)

---

#### `(restore key)`
**Purpose:** Retrieve checkpointed state
**Why it exists:** Recovery from failures, cache lookups
**Example:**
```scheme
(define cached (restore key))
(if cached cached (compute-expensive))
```
**Use when:** Recovery, caching, resumable orchestrations
**Common mistake:** Not checking if restore returns #f (key not found)
**Seen in:** Patterns 13 (anytime), 14 (memoization)

---

### 4.3 LLM Primitives (9)

#### `(llm-query #:instruction str #:data str #:model str ...)`
**Purpose:** Synchronous LLM call. Blocks until complete.
**Why it exists:** Sequential orchestration where next step depends on result
**Example:**
```scheme
(define result (syntax-e (llm-query
  #:instruction "Analyze this text"
  #:data input
  #:model "gpt-4"
  #:temperature 0.0
  #:max-tokens 500)))
```
**Use when:** Sequential flow, need result before continuing
**Common mistake:** Using llm-query when llm-query-async would parallelize
**Seen in:** Patterns 4 (critique), 5 (fold), 6 (meta-orchestration)

---

#### `(llm-query-async #:instruction str #:data str #:model str ...)`
**Purpose:** Asynchronous LLM call. Returns handle immediately.
**Why it exists:** Parallelism. Launch multiple calls, await later.
**Example:**
```scheme
(define handle1 (llm-query-async #:instruction "Task 1" #:model "gpt-4"))
(define handle2 (llm-query-async #:instruction "Task 2" #:model "gpt-4"))
(define results (await-all (list handle1 handle2)))
```
**Use when:** Parallel independent calls
**Common mistake:** Forgetting to await (handle is not result)
**Seen in:** Patterns 1 (fan-out), 7 (hedging), 8 (ensemble), 9 (active learning)

---

#### `(unsafe-raw-query #:instruction str #:model str ...)`
**Purpose:** Get raw LLM output without wrapping in syntax object
**Why it exists:** Code generation needs raw text, not syntax object
**Example:**
```scheme
(define code (unsafe-raw-query
  #:instruction "Write Scheme code"
  #:model "code-davinci-002"))
;; Returns raw string, not syntax object
```
**Use when:** Code generation, meta-programming
**Common mistake:** Using for normal queries (syntax-e is safer)
**Seen in:** Patterns 2, 6 (code generation)

---

#### `(map-async function list #:max-concurrent N)`
**Purpose:** Parallel map with concurrency control
**Why it exists:** Core parallelization primitive. Process N items concurrently.
**Example:**
```scheme
(define results (map-async
  (lambda (item) (llm-query-async #:instruction "..." #:data item #:model "gpt-4"))
  items
  #:max-concurrent 10))
```
**Use when:** Process multiple items in parallel
**Common mistake:** Too high concurrency (rate limits), too low (underutilized)
**Seen in:** Patterns 1, 7, 8, 9, 10 (most patterns)

---

#### `(await handle)`
**Purpose:** Block until specific async call completes
**Why it exists:** Get result from single async call
**Example:**
```scheme
(define handle (llm-query-async ...))
(define result (syntax-e (await handle)))
```
**Use when:** Need single async result
**Common mistake:** Sequential await in loop (use await-all instead)
**Seen in:** Basic async patterns

---

#### `(await-all handles)`
**Purpose:** Block until ALL async calls complete
**Why it exists:** Synchronization point for parallel work
**Example:**
```scheme
(define handles (list h1 h2 h3))
(define results (await-all handles))  ;; Wait for all 3
```
**Use when:** Need all results before continuing
**Common mistake:** Sequential await in loop (use await-all instead)
**Seen in:** Patterns 1 (fan-out synthesis), 8 (ensemble), 10 (tree)

---

#### `(await-all-syntax handles)`
**Purpose:** Like await-all but returns syntax objects (not unwrapped strings)
**Why it exists:** Sometimes need wrapped results for further processing
**Example:**
```scheme
(define syntax-results (await-all-syntax handles))
;; Results are still wrapped, need syntax-e to extract strings
```
**Use when:** Need syntax objects instead of strings
**Seen in:** Rare, advanced patterns

---

#### `(await-any handles)`
**Purpose:** Block until FIRST async call completes
**Why it exists:** Hedging, speculative execution
**Example:**
```scheme
(define-values (winner remaining) (await-any (list h1 h2 h3)))
;; winner = first result, remaining = [h2 h3] handles
```
**Use when:** Race conditions, hedging for latency
**Common mistake:** Not canceling remaining (cost waste)
**Seen in:** Pattern 7 (hedging)

---

#### `#:recursive #t` (llm-query parameter)
**Purpose:** Give sub-model its own sandbox (can use scaffolding)
**Why it exists:** Hierarchical delegation. Sub-model can orchestrate.
**Example:**
```scheme
(llm-query
  #:instruction "Analyze section. You can use map-async to delegate."
  #:data section
  #:recursive #t
  #:model "gpt-4")
```
**Use when:** Hierarchical problems, each level needs autonomy
**Common mistake:** Using with async (not supported), exceeding depth limit (max 3)
**Seen in:** Pattern 3 (recursive delegation)

---

### 4.4 Scope Primitives (2)

#### `(syntax-e syntax-object)`
**Purpose:** Extract string from wrapped LLM response
**Why it exists:** llm-query returns syntax object, need string
**Example:**
```scheme
(define wrapped (llm-query ...))  ;; Returns syntax
(define str (syntax-e wrapped))   ;; Extract string
```
**Use when:** After synchronous llm-query
**Common mistake:** Forgetting syntax-e (get syntax object instead of string)
**Seen in:** All patterns using llm-query

---

#### `(datum->syntax #f code-string)`
**Purpose:** Convert string to executable syntax
**Why it exists:** Code generation. Execute LLM-generated code.
**Example:**
```scheme
(define code-str "(+ 1 2)")
(define syntax-obj (datum->syntax #f code-str))
(eval syntax-obj)  ;; Executes code
```
**Use when:** Meta-programming, code generation
**Common mistake:** Not validating code before executing (security risk)
**Seen in:** Patterns 2 (code gen), 6 (meta-orchestration)

---

### 4.5 Python Bridge Primitives (4)

#### `(py-exec code-string)`
**Purpose:** Execute Python code, return printed output
**Why it exists:** Python for data processing (numpy, pandas, etc.)
**Example:**
```scheme
(py-exec "
import json
data = json.loads(context)
print(data['key'])
")
```
**Use when:** Data processing, libraries not in Scheme
**Common mistake:** Not printing output (py-exec returns printed text)
**Seen in:** Patterns 1 (chunking), 9 (filtering), 10 (pairing)

---

#### `(py-eval expression-string)`
**Purpose:** Evaluate Python expression, return result
**Why it exists:** Quick calculations, data access
**Example:**
```scheme
(define count (py-eval "len(items)"))
(define subset (py-eval "items[:10]"))
```
**Use when:** Simple expressions, data access
**Common mistake:** Using py-eval for multi-line code (use py-exec)
**Seen in:** Patterns 1, 5, 9, 10 (data manipulation)

---

#### `(py-set! variable-name value)`
**Purpose:** Transfer data from Scheme to Python
**Why it exists:** LLM results need to be processed in Python
**Example:**
```scheme
(define results (map-async ...))
(py-set! "results" results)
(py-exec "
import json
for r in results:
    obj = json.loads(r)
    print(obj)
")
```
**Use when:** Passing Scheme data to Python
**Common mistake:** Not setting before using in py-exec
**Seen in:** Patterns 1, 4, 5, 8, 9, 10 (most patterns)

---

#### `(py-call function-name args)`
**Purpose:** Call Python function with arguments
**Why it exists:** Cleaner than py-eval for function calls
**Example:**
```scheme
(py-exec "def add(a, b): return a + b")
(define result (py-call "add" (list 3 5)))  ;; Returns 8
```
**Use when:** Calling defined Python functions
**Seen in:** Advanced Python integration

---

### 4.6 Resource Primitives (3)

#### `(tokens-used)`
**Purpose:** Get token consumption report
**Why it exists:** Cost monitoring, budget enforcement
**Example:**
```scheme
(define report (tokens-used))
;; Returns: "Input: 10K, Output: 5K, Cost: $0.15"
```
**Use when:** End of orchestration, cost tracking
**Common mistake:** Calling too frequently (overhead)
**Seen in:** Cost-sensitive patterns

---

#### `(rate-limits)`
**Purpose:** Check current rate limit status
**Why it exists:** Avoid hitting rate limits, adaptive throttling
**Example:**
```scheme
(define limits (rate-limits))
;; Returns: "Requests: 450/500, Tokens: 80K/100K"
```
**Use when:** Before large fan-out, adaptive concurrency
**Common mistake:** Not checking before 100+ concurrent calls
**Seen in:** Pattern 1 (high concurrency)

---

#### `(token-budget remaining-tokens)`
**Purpose:** Set token budget limit
**Why it exists:** Hard cost cap, prevent runaway spending
**Example:**
```scheme
(token-budget 50000)  ;; Stop after 50K tokens
```
**Use when:** Cost-constrained orchestrations
**Common mistake:** Setting too low (premature termination)
**Seen in:** Budget-constrained patterns

---

### 4.7 Escape Hatch Primitives (3)

These bypass safety checks. Use with caution.

#### `(unsafe-interpolate syntax-object)`
**Purpose:** Strip syntax wrapper without logging
**Why it exists:** Performance when logging not needed
**Example:**
```scheme
(define val (unsafe-interpolate stx))
```
**Use when:** Performance-critical paths
**Common mistake:** Using instead of syntax-e (loses auditability)

---

#### `(unsafe-overwrite name value)`
**Purpose:** Overwrite any variable binding via set!
**Why it exists:** Emergency override of protected bindings
**Example:**
```scheme
(unsafe-overwrite 'my-var new-value)
```
**Use when:** Debugging, emergency fixes
**Common mistake:** Breaking scaffold by overwriting protected names

---

#### `(unsafe-exec-sub-output syntax)`
**Purpose:** Execute code in sub-sandbox
**Why it exists:** Isolation for generated/untrusted code
**Example:**
```scheme
(unsafe-exec-sub-output (datum->syntax #f generated-code))
```
**Use when:** Executing untrusted/generated code
**Common mistake:** Not constraining sub-sandbox (infinite loops)
**Seen in:** Patterns 2, 6 (code generation)

---

### 4.8 Error Handling (1)

#### `(try expression on-error handler)`
**Purpose:** Graceful error handling for sub-model calls
**Why it exists:** Continue map-async even if some items fail
**Example:**
```scheme
(try
  (llm-query #:instruction "..." #:data item)
  on-error
  (lambda (err) "FAILED"))
```
**Use when:** Fault-tolerant processing
**Common mistake:** Not handling errors in map-async (whole batch fails)
**Seen in:** Robust production patterns

---

### 4.9 Primitive Selection Guide

**When to use what:**

| Task | Primitive |
|------|-----------|
| Synchronous LLM call | llm-query + syntax-e |
| Parallel LLM calls | llm-query-async + map-async + await-all |
| First-wins race | llm-query-async + await-any |
| Structured output | #:json #t |
| Deterministic output | #:temperature 0.0 |
| Cost cap | #:max-tokens |
| Hierarchical delegation | #:recursive #t |
| Data processing | py-exec, py-eval, py-set! |
| Fault tolerance | checkpoint, restore, try/on-error |
| Return result | finish, finish-var |
| Code generation | unsafe-raw-query, datum->syntax |
| Cost tracking | tokens-used |
| Rate limit check | rate-limits |

---

