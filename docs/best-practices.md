## Part V: Best Practices & Cost Optimization

### 5.1 Cost Optimization Principles

#### Rule 1: Use Cheapest Model That Works
```scheme
;; DON'T: Expensive for simple task
(map-async (lambda (x) (llm-query-async ... #:model "gpt-4o")) items)

;; DO: Cheap for fan-out
(map-async (lambda (x) (llm-query-async ... #:model "gpt-4.1-nano")) items)
```

**Cost comparison:**
- gpt-4.1-nano: $0.10/1M tokens
- gpt-4o-mini: $0.15/1M tokens
- gpt-4o: $2.50/1M tokens (25x more expensive)
- gpt-4.1: $2.00/1M tokens

**Rule of thumb:** Fan-out = nano, Synthesis = gpt-4o

---

#### Rule 2: Cache with temperature=0.0
```scheme
;; Deterministic = cacheable
(cached-query inst data model)  ;; Pattern 14

;; Non-deterministic = not cacheable
(llm-query #:temperature 0.7 ...)  ;; Every call different
```

**Impact:** 50%+ cost savings with 50% cache hit rate

---

#### Rule 3: Use Active Learning (Pattern 9)
```scheme
;; DON'T: Expensive on all
(map-async (lambda (x) (llm-query-async #:model "gpt-4o" ...)) all-items)

;; DO: Cheap on easy, expensive on hard
;; Phase 1: Cheap with confidence
;; Phase 2: Expensive on low-confidence only
```

**Impact:** 5x cost reduction at same accuracy

---

#### Rule 4: Cost Pyramid in Tree Aggregation
```scheme
(define model (cond
  [(<= level 2) "gpt-4.1-nano"]   ;; Cheap at leaves (90% of calls)
  [(<= level 4) "gpt-4o-mini"]    ;; Mid-tier
  [else "gpt-4o"]))               ;; Expensive at top (1 call)
```

**Impact:** 10x cheaper than gpt-4o everywhere

---

### 5.2 Quality Optimization Principles

#### Rule 1: Ensemble for High Stakes (Pattern 8)
```scheme
;; Single model: 82% accuracy
;; Ensemble (5 models): 92-95% accuracy
```

**Trade-off:** 5x cost for +10-13% accuracy

---

#### Rule 2: Critique-Refine Loop (Pattern 4)
```scheme
;; Single-shot: 65% quality
;; 3 iterations: 85-90% quality
```

**Best practice:** Cheap critic ($0.15/1M), expensive generator ($2.50/1M)

---

#### Rule 3: Consensus for Safety (Pattern 11)
```scheme
;; 5 models, 2 rounds, supermajority
;; Error rate: <1% vs 10% single model
```

**Use when:** Mission-critical (medical, legal, safety)

---

### 5.3 Latency Optimization Principles

#### Rule 1: Parallel Fan-Out (Pattern 1)
```scheme
;; DON'T: Sequential
(map (lambda (x) (syntax-e (llm-query ...))) items)  ;; 50 items × 2s = 100s

;; DO: Parallel
(map-async (lambda (x) (llm-query-async ...)) items #:max-concurrent 10)  ;; 10s
```

**Impact:** 10x faster with concurrency=10

---

#### Rule 2: Hedging for P99 (Pattern 7)
```scheme
;; 3 parallel approaches, first wins
;; P99: 45s -> 4s (10x improvement)
;; Cost: 2x
```

**Use when:** User-facing, latency-sensitive

---

#### Rule 3: Anytime Algorithms (Pattern 13)
```scheme
;; Level 1: 2s, 70% quality (checkpoint)
;; Level 2: 7s, 85% quality (checkpoint)
;; Level 3: 22s, 95% quality
```

**Use when:** Variable latency tolerance

---

### 5.4 Anti-Patterns (Common Mistakes)

#### Anti-Pattern 1: Sequential Instead of Parallel
```scheme
;; BAD: Sequential
(define r1 (syntax-e (llm-query ...)))
(define r2 (syntax-e (llm-query ...)))
(define r3 (syntax-e (llm-query ...)))

;; GOOD: Parallel
(define handles (list
  (llm-query-async ...)
  (llm-query-async ...)
  (llm-query-async ...)))
(define results (await-all handles))
```

**Cost:** 3x slower

---

#### Anti-Pattern 2: Expensive Model on Fan-Out
```scheme
;; BAD: gpt-4o on all
(map-async (lambda (x) (llm-query-async #:model "gpt-4o" ...)) 100-items)
;; Cost: $2.50

;; GOOD: nano on fan-out
(map-async (lambda (x) (llm-query-async #:model "gpt-4.1-nano" ...)) 100-items)
;; Cost: $0.10 (25x cheaper)
```

---

#### Anti-Pattern 3: No Caching on Repeated Queries
```scheme
;; BAD: No cache
(llm-query #:instruction same #:data same #:model same)  ;; Called 100 times

;; GOOD: Cache
(cached-query inst data model)  ;; Pattern 14
```

**Cost:** 50%+ savings

---

#### Anti-Pattern 4: Flat Aggregation of Large Dataset
```scheme
;; BAD: Concat 100 chunks
(define all (py-exec "print('\\n'.join(chunks))"))
(llm-query #:instruction "Summarize" #:data all)  ;; Context overflow

;; GOOD: Tree aggregation
(tree-reduce chunks 1)  ;; Pattern 10
```

---

#### Anti-Pattern 5: No Error Handling
```scheme
;; BAD: No checkpoints
(expensive-computation-10-hours)
;; Crash at hour 9 = lose everything

;; GOOD: Checkpoint
(checkpoint "progress" state)
```

---

### 5.5 Cost Reference Tables

#### Model Costs (per 1M tokens)
| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| gpt-4.1-nano | $0.10 | $0.40 | Fan-out, filtering |
| gpt-4o-mini | $0.15 | $0.60 | Mid-tier synthesis |
| gpt-4.1-mini | $0.30 | $1.20 | Code generation |
| gpt-4o | $2.50 | $10.00 | Complex reasoning |
| gpt-4.1 | $2.00 | $8.00 | Code + reasoning |
| o3-mini | $1.10 | $4.40 | Math/logic |
| o4-mini | $1.10 | $4.40 | Advanced reasoning |

---

#### Pattern Cost Profiles
| Pattern | Cost Multiplier | Use Case |
|---------|----------------|----------|
| 1. Parallel Fan-Out | 1x (baseline) | Standard processing |
| 7. Hedging | 2x | P99 latency critical |
| 8. Ensemble | 5x | High accuracy required |
| 9. Active Learning | 0.2x | Budget-constrained |
| 11. Consensus | 10x | Mission-critical |
| 14. Memoization | 0.5x | Repeated queries |

---

#### Optimization Targets
| Metric | Target | How to Achieve |
|--------|--------|----------------|
| P99 latency | <5s | Pattern 7 (Hedging) |
| Accuracy | >95% | Pattern 8 (Ensemble) |
| Cost reduction | >50% | Pattern 14 (Caching) + Pattern 9 (Active) |
| Throughput | >100/min | Pattern 1 (Fan-out, concurrency=50) |
| Cache hit rate | >30% | Pattern 14, temperature=0.0 |

---

### 5.6 Debugging Checklist

Before deploying:
- [ ] Using gpt-4.1-nano for fan-out? (not gpt-4o)
- [ ] temperature=0.0 for caching/classification?
- [ ] max-tokens set to cap response length?
- [ ] Using await-all not sequential await?
- [ ] Checkpointing intermediate results?
- [ ] Tree reduce for >20 chunks? (not flat concat)
- [ ] Checking tokens-used() to monitor costs?
- [ ] Using py-set! for LLM->Python transfer?
- [ ] Concurrency appropriate (#:max-concurrent 10-50)?
- [ ] Error handling for LLM failures?

---

### 5.7 Quick Decision Guide

**"My task is..."**

| Task Description | Pattern | Why |
|-----------------|---------|-----|
| Process 500 documents | 1 (Parallel Fan-Out) | Speed + scale |
| Need 95%+ accuracy | 8 (Ensemble) | Quality |
| Budget = $5, need quality | 9 (Active Learning) | Cost + quality |
| P99 latency = 45s (bad) | 7 (Hedging) | Latency |
| Synthesize 100 chunks | 10 (Tree Aggregation) | Quality at scale |
| Unknown data structure | 2 (Code Generation) | Adaptivity |
| Mission-critical | 11 (Consensus) | Safety |
| Repeated queries | 14 (Memoization) | Cost |
| Uncertain deadline | 13 (Anytime) | Flexibility |
| Hierarchical data | 3 (Recursive Delegation) | Structure |

---

## Summary

This guide covered:
- **Part I:** When to use rlm-scheme, pattern overview
- **Part II:** Decision framework for pattern selection
- **Part III:** 16 patterns with complete examples
- **Part IV:** 27 primitives reference
- **Part V:** Best practices, cost optimization, anti-patterns

**Key Takeaway:** Use the cheapest model that works, parallelize everything, cache repeats, and compose patterns to win.

**Getting Started:**
1. Call `get_usage_guide` to see this guide
2. Choose pattern based on your constraints (latency/quality/cost/structure/scale)
3. Copy example code, adapt to your data
4. Monitor with `tokens-used()`, optimize iteratively

**Need Help?** Call `get_code_generation_api_reference` for condensed API when generating code.

---

*End of Usage Guide*
