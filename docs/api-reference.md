RLM-SCHEME API REFERENCE (for code-generating sub-models)

⚠️ CRITICAL SYNTAX RULES:
- All function arguments use #: keyword syntax: #:instruction, #:data, #:model, etc.
- (define x value) for bindings. set! is NOT available in sandbox.
- String operations: (string-append "a" "b"), (substring str start end)
- MUST unwrap llm-query results: (syntax-e result) before using as string

CORE LLM FUNCTIONS:

(llm-query #:instruction "task description"
           #:data "context or data"
           #:model "gpt-4o-mini"
           #:temperature 0.0
           #:max-tokens 500
           #:json #t)
  → Returns syntax object. MUST unwrap with (syntax-e result).
  → When using #:json #t, the #:instruction MUST contain the word 'json'.
  → Models: "gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "o3-mini", "o4-mini"
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
  → Always use cheap models (gpt-4.1-nano or gpt-4o-mini) for fan-out.
  → Example: (map-async (lambda (chunk) (llm-query-async #:instruction "summarize" #:data chunk #:model "gpt-4.1-nano")) chunks)

PYTHON BRIDGE (for computation and I/O):

(py-exec "python_code_string")
  → Runs multi-line Python code (imports, statements, loops, file I/O).
  → Returns stdout as string. Use print() to output results.
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

Pattern 1: Parallel fan-out
```scheme
(define results (map-async
  (lambda (item)
    (llm-query-async #:instruction "process this" #:data item #:model "gpt-4.1-nano"))
  items
  #:max-concurrent 10))
(finish results)
```

Pattern 2: Extract with Python, analyze with LLM
```scheme
(define data (py-exec "import json; print(json.dumps(process_data()))"))
(define analysis (syntax-e (llm-query #:instruction "analyze" #:data data #:model "gpt-4o")))
(finish analysis)
```

Pattern 3: py-set! → py-exec → py-eval round-trip (most common data flow)
```scheme
;; 1. Get data from LLM into Scheme
(define analysis (syntax-e (llm-query #:instruction "Extract key facts as JSON" #:data doc #:model "gpt-4o" #:json #t)))
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

Pattern 4: Define result variable (for generated code)
```scheme
(define result (... your computation ...))
;; Caller will use (finish-var "result") to retrieve this
```

IMPORTANT REMINDERS:
- Always use #: prefix for keyword arguments
- Always unwrap llm-query with (syntax-e ...) before using as string
- map-async items get unwrapped automatically (no syntax-e needed)
- Use (finish ...) to return your result
- Use cheap models (gpt-4.1-nano) for parallel work
- py-set! is safer than string interpolation for passing data to Python
