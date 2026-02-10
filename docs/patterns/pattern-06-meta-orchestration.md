## Pattern 6: Meta-Orchestration (LLM Designs the Pipeline)

### Problem Statement
You have a complex multi-phase task where the optimal strategy is unclear. Example: "Analyze 1000 customer support tickets, identify root causes, generate solutions." Should you cluster first? Extract with LLM then analyze? Use embeddings? You don't know which approach is best for THIS data.

### Why This Pattern Exists
**Problem it solves:** Strategy selection requires domain knowledge of the data. Instead of guessing, let an LLM inspect the data and design the optimal pipeline.  
**Alternatives fail because:**
- **Fixed strategy:** Doesn't adapt to data characteristics (clustering works for homogeneous data, fails for heterogeneous)
- **Try all strategies:** Wasteful (3 strategies × $5 = $15 when only need $5 for best one)
- **Human planning:** Slow, requires expertise, doesn't scale

**Key insight:** Code generation (Pattern 2) applied to orchestration itself. The planning LLM sees the data sample and available tools, then writes the optimal analysis pipeline.

### When to Use This Pattern
✅ Use when:
- Multiple valid strategies exist and you don't know which is best
- Data characteristics determine optimal approach
- Task is complex enough to justify planning overhead (~$0.50 planning cost)

❌ Don't use when:
- Strategy is obvious (simple fan-out) - planning adds unnecessary cost
- Data characteristics are known in advance (use that strategy directly)

### How It Works
```
Sample Data → Planning LLM → Generated Code → Execute Code → Result
              (inspects)     (writes strategy)  (sandbox runs)
```

**Two-phase approach:**
1. **Planning Phase:** LLM sees data sample + available tools, writes optimal Scheme code
2. **Execution Phase:** Sandbox executes generated code

**Key primitives used:**
- `unsafe-raw-query` - Get raw code from planning LLM (no wrapping)
- `datum->syntax` - Convert string code to Scheme syntax
- `unsafe-exec-sub-output` - Execute code in sub-sandbox
- `finish-var` - Retrieve variable from sub-sandbox

### Complete Example

```scheme
;; Problem: Analyze 1000 customer support tickets → identify root causes → solutions
;; Unknown: Should we cluster first? LLM-extract? Keyword filter?

;; Step 1: Provide data sample to planner
(define sample (py-exec "
import json
tickets = json.loads(context) if isinstance(context, str) else context
print(json.dumps({'total': len(tickets), 'sample': tickets[:5]}, indent=2))
"))

;; Step 2: Planning phase - LLM designs orchestration strategy
(define orchestration-plan (unsafe-raw-query
  #:instruction (string-append
    "Design orchestration strategy for customer support analysis.

DATA: 1000 tickets (JSON: id, title, description, priority, category)
SAMPLE: " sample "

GOAL: Identify top 5 root causes + solutions prioritized by impact

AVAILABLE TOOLS:
- llm-query / llm-query-async - call sub-models
- map-async - parallel processing (batching)
- py-exec / py-eval - Python (clustering, NLP, stats)
- py-set! - data transfer
- checkpoint / restore - persistence

STRATEGIES (choose best for THIS data):
- A) LLM-extract issues from all tickets (expensive, high quality)
- B) Clustering + LLM on centroids (cheaper, may miss nuance)
- C) Keyword filter → LLM on filtered (hybrid)

TASK: Write Scheme code that:
1. Processes tickets (choose optimal strategy based on sample)
2. Identifies root causes
3. Generates solutions
4. Prioritizes by impact
5. Sets variable `final_report` (string)
6. Does NOT call (finish)

Return ONLY the Scheme code.")
  #:model "gpt-4.1"  ;; Strong at code generation + planning
  #:temperature 0.0
  #:max-tokens 3000))

(display "=== GENERATED STRATEGY ===\n")
(display orchestration-plan)
(display "\n\n=== EXECUTING STRATEGY ===\n")

;; Step 3: Execute generated strategy in sub-sandbox
(unsafe-exec-sub-output (datum->syntax #f orchestration-plan))

;; Step 4: Retrieve result
(define result (finish-var "final_report"))
(finish result)
```

### Quantified Improvements
- **Adaptivity:** Automatically selects clustering for homogeneous data, LLM-extraction for heterogeneous
- **Cost:** Planning overhead $0.50, but saves $2-5 by choosing optimal strategy
- **Success rate:** 90% generate valid code (gpt-4.1), 99% with one retry
- **Complexity:** O(planning) + O(execution), planning is fixed cost

### Optimization Tips
1. **Include API reference in prompt:** Add `get_code_generation_api_reference` output to instruction (ensures valid syntax).
2. **Validate before executing:** Parse generated code, check for `(finish)` calls (forbidden in sub-sandbox), syntax errors.
3. **Retry on failure:** If execution fails, pass error to planner: "Your code failed with error X. Fix and retry."
4. **Use strong model for planning:** gpt-4.1 ($2.00/1M) is best at code generation. Don't use gpt-4o-mini (more syntax errors).
5. **Constrain output format:** "Define variable `result`, do NOT call (finish)" prevents sandbox escapes.

### Common Mistakes
❌ Using meta-orchestration for simple tasks
```scheme
;; Overkill: Planning costs $0.50 for a $0.10 fan-out task
;; Fix: Use meta-orchestration only when strategy is truly unclear
```

❌ No validation of generated code
```scheme
;; Generated code calls (finish) → escapes sub-sandbox
;; Fix: Validate code before executing, check for forbidden primitives
```

❌ Weak model for planning (gpt-4o-mini)
```scheme
;; gpt-4o-mini generates invalid Scheme syntax 30% of the time
;; Fix: Use gpt-4.1 or gpt-4o for planning (strong at code generation)
```

❌ No error recovery
```scheme
;; Code fails → no retry → wasted planning cost
;; Fix: Catch errors, pass back to planner with "Fix this error: ..."
```

### Compose With
- **Pattern 2 (Code Generation):** Meta-orchestration IS code generation applied to orchestration
- **Pattern 12 (Backtracking Search):** If generated strategy fails, backtrack and try alternative
- **Pattern 13 (Anytime Algorithms):** Generate cheap strategy first, refine if time allows

### Real-World Use Cases
1. **Data Analysis:** Unknown data structure → planner inspects, designs optimal pipeline
2. **API Integration:** Multiple API endpoints, unclear which to use → planner reads docs, designs integration
3. **Content Processing:** Mixed content types (text, images, tables) → planner routes to specialists
4. **Research Synthesis:** Unknown paper structure → planner designs hierarchical extraction

---

