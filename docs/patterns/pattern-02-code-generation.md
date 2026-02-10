## Pattern 2: Code Generation (Meta-Programming)

### Problem Statement
Data structure is unknown until runtime (e.g., new API format, unfamiliar schema). You don't know the optimal analysis strategy beforehand. Need adaptive approach that adjusts to data characteristics.

### Why This Pattern Exists
**Problem it solves:** Unknown structure, data-dependent strategies
**Alternative approaches fail because:**
- Hardcoded analysis: Breaks on schema changes
- Manual inspection: Takes 30+ minutes human time
- Generic approach: Suboptimal for specific data

### When to Use This Pattern
✅ **Use when:**
- Data schema unknown at design time
- Optimal strategy depends on data characteristics
- Want model to discover its own approach

❌ **Don't use when:**
- Structure is well-known (use specific pattern)
- Model-generated code too risky (security concerns)
- Need deterministic behavior (code gen is probabilistic)

### How It Works
**Conceptual flow:**
1. Show model a sample of the data
2. Model writes Scheme code that analyzes it
3. Execute the generated code (deliberate scope crossing)
4. Return the defined result variable

**Key primitives used:**
- `unsafe-raw-query` - Generate code (returns string, not syntax)
- `datum->syntax` - Wrap code string for eval
- `unsafe-exec-sub-output` - Execute generated code
- `finish-var` - Return variable defined by generated code
- `py-exec` - Data inspection in Python

### Complete Example
```scheme
;; Problem: Analyze Stripe payment data with unknown schema
;; Why this pattern: Never seen this API format before, don't know optimal approach

;; Step 1: Inspect data sample
(define sample (py-exec "
import json
data = json.loads(context) if isinstance(context, str) else context
sample = data[:2] if isinstance(data, list) else [data]
print(json.dumps(sample, indent=2))
"))

(display "Data sample:\n")
(display (py-exec "print(sample[:500])"))
(display "\n\n")

;; Step 2: Get API reference for sub-model
;; (In real use: Call get_code_generation_api_reference MCP tool)
(define api-ref "
CRITICAL: Use #: for keyword args (#:instruction, #:data, #:model)
MUST unwrap with (syntax-e result) before using as string

Available functions:
- (llm-query #:instruction \"...\" #:data \"...\" #:model \"...\")
- (py-exec \"code\") - returns stdout
- (py-eval \"expr\") - returns value
- (py-set! \"var\" value) - safe transfer
- (map-async fn items) - parallel processing
- (string-append ...) - concatenate strings
- (finish result) - return final answer
")

;; Step 3: Model writes custom analysis code
(define analysis-code (unsafe-raw-query
  #:instruction (string-append
    "You are writing Scheme code for rlm-scheme sandbox.

TASK: Analyze this Stripe payment data. Write code that:
1. Uses py-exec to compute: total revenue, average transaction, top 5 customers
2. Uses llm-query to identify anomalies/fraud patterns
3. Stores result in variable `final_report` (string)
4. Do NOT call (finish) - just define the variable

DATA SAMPLE:
" sample "

API REFERENCE:
" api-ref "

Return ONLY Scheme code, no explanations.")
  #:model "code-davinci-002"  ;; Strong code generation model
  #:temperature 0.0
  #:max-tokens 2000))

(display "Generated analysis code:\n")
(display analysis-code)
(display "\n\n=== EXECUTING ===\n")

;; Step 4: Execute the generated strategy
(unsafe-exec-sub-output (datum->syntax #f analysis-code))

;; Step 5: Return the variable it defined
(finish-var "final_report")
```

### Quantified Improvements
**vs Manual approach:**
- **Time:** 30min manual inspection → 60s automated
- **Adaptability:** 100% (works on any schema)
- **Reusability:** Same prompt works for Shopify, AWS, GitHub APIs

**vs Hardcoded:**
- **Brittleness:** 0% (adapts to schema changes)
- **Development time:** 0min (no coding needed)

**Complexity:**
- Time: O(M + P) where M=meta-model inference, P=generated program execution
- Space: O(1)
- API calls: 1 (code gen) + P (program's sub-calls)

### Optimization Tips
1. **Use code-davinci-002 for code generation** (best at code, better than gpt-4)
2. **Include API reference** (sub-model doesn't know rlm-scheme syntax)
3. **Validate generated code** (check for syntax errors before exec)
4. **Use #:temperature 0.0** (deterministic code generation)
5. **Limit #:max-tokens 2000** (reasonable code size)

### Common Mistakes
❌ Not including API reference
```scheme
(unsafe-raw-query #:instruction "Write Scheme code..." ...)
;; Sub-model generates invalid syntax (doesn't know #: convention)
```

❌ Using finish in generated code
```scheme
;; Generated code: (finish result)
;; Problem: Exits early, can't compose with other steps
;; Fix: Define variable instead, use finish-var
```

❌ Not validating generated code
```scheme
(unsafe-exec-sub-output (datum->syntax #f code))
;; If code has syntax error, crashes with cryptic message
;; Fix: Wrap in with-handlers or validate first
```

### Compose With
- **Pattern 4 (Critique-Refine):** Generate code → validate → refine if invalid
- **Pattern 6 (Meta-Orchestration):** Planning model generates orchestration code
- **Pattern 13 (Anytime):** Generate simple code first, refine if time allows

### Real-World Use Cases
1. **API integration:** Analyze unknown API responses automatically
2. **Data science:** Adaptive EDA (exploratory data analysis)
3. **ETL pipelines:** Generate extraction code based on schema
4. **Testing:** Generate test cases from specification

---

