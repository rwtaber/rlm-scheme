## Pattern 14: Memoization (Content-Addressed Caching)

### Problem Statement
Repeated queries cost money. "Analyze customer complaints about billing" asked 100 times = $50. 80% are identical queries.

### Why This Pattern Exists
**Problem it solves:** Redundant computation. Same input = same output (if deterministic).
**Alternatives fail because:**
- **No caching:** Pay for every query
- **Simple key-value:** Hard to detect semantic similarity
- **TTL caching:** Doesn't leverage content similarity

**Key insight:** Content-hash (instruction + data + model). Identical content = cache hit. With temperature=0.0, output deterministic.

### When to Use This Pattern
Use when:
- Repeated queries likely
- Deterministic output (temperature=0.0)
- Storage available

Don't use when:
- All queries unique
- Non-deterministic (temperature>0)
- Cache invalidation complex

### How It Works
```
Query -> Hash(instruction + data + model) -> Cache lookup
  Hit: Return cached (0.1s, $0)
  Miss: LLM query -> Cache result -> Return (5s, $0.50)
```

**Key primitives:** checkpoint/restore, py-exec (hashing), #:temperature 0.0

### Complete Example

```scheme
(define (cached-query instruction data model)
  ;; Compute content hash
  (py-set! "inst" instruction)
  (py-set! "dat" data)
  (py-set! "mod" model)
  (define key (py-exec "
import hashlib
key = hashlib.sha256(f'{inst}||{dat}||{mod}'.encode()).hexdigest()
print(key)
"))

  ;; Try restore
  (define cached (restore key))
  (if cached
      (begin
        (display "CACHE HIT\n")
        cached)
      (begin
        (display "CACHE MISS\n")
        ;; Query LLM
        (define result (syntax-e (llm-query
          #:instruction instruction
          #:data data
          #:model model
          #:temperature 0.0)))  ;; Deterministic
        ;; Cache for future
        (checkpoint key result)
        result)))

;; Usage
(define analysis1 (cached-query "Analyze complaints" context "gpt-4o"))
;; Second call instant
(define analysis2 (cached-query "Analyze complaints" context "gpt-4o"))  ;; CACHE HIT
```

### Quantified Improvements
- Cache hit rate: 30-80% (depends on query patterns)
- Cost savings: 50%+ with 50% hit rate
- Latency: 0.1s vs 5s (50x faster on hit)

### Optimization Tips
1. Temperature 0.0 (deterministic)
2. Normalize data (strip whitespace before hashing)
3. TTL for cache (expire after 7 days)
4. Semantic similarity: Use embeddings for fuzzy matching

### Common Mistakes
- Temperature > 0 (non-deterministic, cache useless)
- No normalization (whitespace diffs = cache miss)
- Unbounded cache (memory leak)

### Compose With
- Pattern 7 (Cache + hedge: check cache, if miss hedge)
- Pattern 9 (Cache Phase 1 results)

### Real-World Use Cases
1. FAQ answering (same questions repeatedly)
2. Code analysis (same codebases)
3. Document classification (duplicate docs)
4. API endpoints (repeated requests)

---

