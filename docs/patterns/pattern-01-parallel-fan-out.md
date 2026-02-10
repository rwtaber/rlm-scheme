## Pattern 1: Parallel Fan-Out (MapReduce)

### Problem Statement
Data exceeds single model's context window (e.g., 50 research papers, each 20 pages). Processing sequentially takes too long (25+ minutes). Need fast, cost-effective parallel processing.

### Why This Pattern Exists
**Problem it solves:** Data > context window, independent sub-problems
**Alternative approaches fail because:**
- Single call: Context overflow (50 papers × 20 pages >> 128K tokens)
- Sequential: 50 papers × 30s = 25 minutes (too slow)
- Expensive model on all: 50 × $0.05 = $2.50 (wasteful)

### When to Use This Pattern
✅ **Use when:**
- Independent chunks (no dependencies between items)
- Need speed (latency matters)
- Can use cheap model for bulk work

❌ **Don't use when:**
- Chunks have dependencies (use Pattern 5: Cumulative Fold)
- <10 items (overhead not worth it)
- Need deep reasoning per item (use single expensive model)

### How It Works
**Conceptual flow:**
1. Chunk data into N pieces (via Python)
2. Process chunks in parallel with CHEAP model (map-async)
3. Synthesize results with EXPENSIVE model (single call)

**Key primitives used:**
- `map-async` - Parallel processing with batching
- `llm-query-async` - Async sub-calls (inside lambda)
- `py-eval` - Chunking in Python
- `py-set!` - Safe data transfer to Python
- `syntax-e` - Unwrap final synthesis

### Complete Example
```scheme
;; Problem: Analyze 50 scientific papers for mentions of "ACE2 protein"
;; Why this pattern: 50 papers too large for single call, independent analysis

;; Step 1: Load and chunk papers
(define papers (py-eval "
import json
papers = json.loads(context)  # Assume context is JSON array
[p.get('content', '')[:15000] for p in papers[:50]]  # First 15K chars each
"))

;; Step 2: Parallel fan-out with CHEAP model (gpt-3.5-turbo)
(display "Analyzing 50 papers in parallel...\n")
(define analyses (map-async
  (lambda (paper)
    (llm-query-async
      #:instruction "Find all mentions of ACE2 protein. Return JSON: [{mention: string, context: string, page: int}]"
      #:data paper
      #:model "gpt-3.5-turbo"  ;; CRITICAL: Use cheapest model for fan-out
      #:json #t
      #:temperature 0.0
      #:max-tokens 500))
  papers
  #:max-concurrent 10))  ;; 10 papers at a time

;; Step 3: Combine results in Python
(py-set! "all_analyses" analyses)
(define combined-json (py-exec "
import json
all_mentions = []
for i, analysis_str in enumerate(all_analyses):
    analysis = json.loads(analysis_str)
    for mention in analysis:
        mention['paper_id'] = i + 1
        all_mentions.append(mention)
print(json.dumps(all_mentions, indent=2))
"))

;; Step 4: Synthesize with EXPENSIVE model (gpt-4)
(define synthesis (syntax-e (llm-query
  #:instruction "Synthesize ACE2 protein findings across 50 papers:
1. Most common functions (with frequency)
2. Novel findings
3. Research gaps"
  #:data combined-json
  #:model "gpt-4"  ;; Expensive model for synthesis only
  #:temperature 0.3
  #:max-tokens 800)))

(finish (string-append
  "=== RAW MENTIONS ===\n" combined-json "\n\n"
  "=== SYNTHESIS ===\n" synthesis))
```

### Quantified Improvements
**vs Naive approach (sequential with gpt-4):**
- **Latency:** 10× faster (25min → 2.5min)
  - Naive: 50 × 30s = 25 minutes
  - Fan-out: max(50/10 batches × 30s, synthesis 10s) ≈ 2.5 minutes
- **Cost:** 7× cheaper ($2.50 → $0.35)
  - Naive: 50 × $0.05 (gpt-4) = $2.50
  - Fan-out: 50 × $0.001 (nano) + 1 × $0.10 (synthesis) = $0.35
- **Quality:** Comparable (nano good enough for extraction)

**Complexity:**
- Time: O(N/k) where N=items, k=parallelism
- Space: O(N) (store all results)
- API calls: N (fan-out) + 1 (synthesis)

### Optimization Tips
1. **Always use gpt-3.5-turbo for fan-out** (not gpt-4) - 25× cheaper
2. **Batch size 10-20 optimal** (#:max-concurrent 10)
3. **Checkpoint after batches** for fault tolerance on large workloads
4. **Use #:max-tokens** to cap response length (save cost)
5. **Set #:temperature 0.0** for deterministic extraction

### Common Mistakes
❌ Using expensive model for fan-out
```scheme
(map-async (lambda (x) (llm-query-async ... #:model "gpt-4")) items)
;; 50× more expensive than necessary
```

❌ Not using map-async (sequential instead)
```scheme
(map (lambda (x) (syntax-e (llm-query ...))) items)
;; 10× slower (sequential)
```

❌ Synthesizing with cheap model
```scheme
(llm-query #:data combined #:model "gpt-3.5-turbo" ...)
;; Synthesis needs reasoning power (use gpt-4)
```

### Compose With
- **Pattern 10 (Tree Aggregation):** When >20 chunks (hierarchical synthesis)
- **Pattern 14 (Memoization):** Cache fan-out results if repeated queries
- **Pattern 9 (Active Learning):** Fan-out with cheap, re-process uncertain with expensive

### Real-World Use Cases
1. **Academic research:** Analyze 100+ papers for literature review
2. **Legal:** Review 50 contracts for specific clauses
3. **Business intelligence:** Extract insights from 1000 customer reviews
4. **Content moderation:** Screen 500 social media posts in parallel

---

