## Pattern 3: Recursive Delegation (Hierarchical Decomposition)

### Problem Statement
Data has natural hierarchical structure (book → chapters → sections → paragraphs). Flat analysis loses structure and context. Each level needs different analysis depth and strategy. Need up to 3 levels of delegation.

### Why This Pattern Exists
**Problem it solves:** Hierarchical data, each level needs autonomy
**Alternative approaches fail because:**
- Flat analysis: Loses hierarchical relationships
- Single-level: Can't delegate subsections to specialists
- Manual decomposition: Requires knowing structure beforehand

### When to Use This Pattern
✅ **Use when:**
- Natural hierarchy exists (documents, org charts, code repos)
- Each level needs different strategy
- Sub-agents should have full autonomy (own sandboxes)

❌ **Don't use when:**
- Data is flat (use Pattern 1: Fan-Out)
- Hierarchy >3 levels deep (depth limit)
- All levels use same strategy (simpler patterns work)

### How It Works
**Conceptual flow:**
1. Top-level (you) splits into major sections
2. Depth-1 agents analyze sections (can further delegate)
3. Depth-2 agents analyze subsections (can further delegate)
4. Depth-3 agents analyze leaf nodes (max depth, can't recurse)

**Key primitives used:**
- `llm-query` with `#:recursive #t` - Give sub-model sandbox
- `map-async` - Parallel delegation at each level
- `syntax-e` - Unwrap delegated results
- `py-exec` - Split data into hierarchy

### Complete Example
```scheme
;; Problem: Analyze 300-page legal contract with nested structure
;; Why this pattern: Hierarchical (Section → Subsection → Clause)

;; Step 1: Split into major sections (depth-0 = you)
(define sections (py-eval "
import re
# Split on section markers
parts = re.split(r'\\n## SECTION \\d+:', context)
sections = [{'title': f'Section {i+1}', 'content': p.strip()}
            for i, p in enumerate(parts) if p.strip()][:5]  # First 5
sections
"))

(display "Delegating 5 sections to specialist agents...\n")

;; Step 2: Delegate each section to depth-1 specialist
;; Each specialist can further chunk and delegate to depth-2
;; NOTE: Using synchronous llm-query (not async) because #:recursive only works with sync
(define section-analyses (map
  (lambda (section)
    (py-set! "sec" section)
    (syntax-e (llm-query
      #:instruction "You are a legal analysis specialist with a Scheme sandbox.

CAPABILITIES:
- llm-query (with #:recursive #t), map-async, py-exec, py-eval, py-set!
- finish (REQUIRED to return result)

TASK: Analyze this contract section. If >5000 chars OR has subsections:
1. Split into logical subsections
2. Use llm-query with #:recursive #t for further delegation
3. Synthesize subsection analyses

If short: analyze directly.

Focus on: key obligations, liability caps, notice requirements, ambiguities.

Call (finish your-analysis) when done."
      #:data (py-eval "import json; json.dumps(sec)")
      #:recursive #t  ;; Specialist can spawn sub-agents
      #:model "gpt-4o"
      #:max-tokens 1500)))
  sections))  ;; Sequential map (not map-async) because recursive delegation needs sync

(display "Depth-1 specialists completed. Synthesizing...\n")

;; Step 3: Synthesize section analyses into contract-level insights
(py-set! "section_results" section-analyses)
(define combined (py-exec "
results = ['## SECTION ' + str(i+1) + '\\n' + r for i, r in enumerate(section_results)]
print('\\n\\n'.join(results))
"))

(define contract-summary (syntax-e (llm-query
  #:instruction "You're reviewing analyses of 5 contract sections from specialists.

TASK: Contract-level insights:
1. Overall liability exposure (aggregate risk)
2. Contradictions between sections
3. Missing provisions (gaps)
4. Negotiation priorities

Focus on practical business impact."
  #:data combined
  #:model "gpt-4o"
  #:temperature 0.3
  #:max-tokens 1000)))

(finish (string-append
  "=== SECTION ANALYSES ===\n\n" combined
  "\n\n=== CONTRACT SUMMARY ===\n" contract-summary))
```

### Quantified Improvements
**vs Flat analysis:**
- **Quality:** 85-95% (preserves hierarchy) vs 60-70% (flat)
- **Context:** Each level informed by parent context
- **Specialization:** Depth-1 for sections, depth-2 for clauses

**Complexity:**
- Time: O(log N) best case (balanced tree), O(N) worst (linear)
- Space: O(log N) (recursion depth)
- API calls: O(N) where N=total nodes in tree

### Optimization Tips
1. **Limit concurrency at each level** (3-5, not 10+)
   - Each sub-agent may spawn its own sub-calls
   - Too much parallelism = resource exhaustion
2. **Use cheap models at leaves** (gpt-4o-mini), expensive at top (gpt-4o)
3. **Provide clear delegation instructions** (when to recurse, when to analyze directly)
4. **Max depth is 3** (depth-0=you, depth-1/2/3=sub-models)

### Common Mistakes
❌ Using #:recursive #t with llm-query-async
```scheme
(llm-query-async #:recursive #t ...)
;; Error: #:recursive not supported with async
;; Fix: Use synchronous llm-query with #:recursive #t
```

❌ Too high concurrency with recursive calls
```scheme
(map-async (lambda (x) (llm-query-async #:recursive #t ...)) items #:max-concurrent 20)
;; Each of 20 may spawn 20 more = 400 simultaneous calls
;; Fix: #:max-concurrent 3-5 for recursive
```

❌ Exceeding depth limit
```scheme
;; depth-0 → depth-1 → depth-2 → depth-3 → depth-4
;; Error: recursion depth limit exceeded (max=3)
;; Fix: Redesign to use ≤3 levels
```

### Compose With
- **Pattern 1 (Parallel Fan-Out):** Parallelize siblings at each level
- **Pattern 10 (Tree Aggregation):** Hierarchical synthesis of results
- **Pattern 14 (Memoization):** Cache identical subtrees

### Real-World Use Cases
1. **Legal:** Multi-level contract review (agreement → sections → clauses)
2. **Academic:** Book analysis (book → chapters → sections → paragraphs)
3. **Code:** Repository analysis (repo → modules → files → functions)
4. **Business:** Org chart analysis (company → departments → teams → individuals)

---

