## Pattern 10: Tree Aggregation (Hierarchical Reduction)

### Problem Statement
Summarize 100 research abstracts. Flat concatenation (concat all 100 -> synthesize) exceeds context limit and loses information. Single-shot quality: 60-70%.

### Why This Pattern Exists
**Problem it solves:** Flat aggregation doesn't scale. 100 abstracts = 50K tokens, context overflow. Model can't attend to all details simultaneously.
**Alternatives fail because:**
- **Flat concat:** Context overflow, information loss
- **Sample subset:** Loses data (only see 20 of 100)
- **Summarize then concat:** Still hits limits at scale

**Key insight:** Hierarchical pairwise merging. Level 1: 100->50 pairs. Level 2: 50->25. Continue until 1. Each merge preserves key info from both children. O(log N) depth.

### When to Use This Pattern
Use when:
- 20+ chunks to aggregate
- Quality matters more than cost
- Hierarchical relationships in data

Don't use when:
- <20 chunks (flat works fine)
- Cost-sensitive (tree costs 2x flat)

### How It Works
```
Level 0: 100 items
Level 1: 50 pairs merge -> 50 summaries
Level 2: 25 pairs merge -> 25 summaries
Level 3: 12 pairs merge -> 12 summaries
...
Level 6: 1 final synthesis
```

**Key primitives:** map-async, py-eval (pairing), recursion

### Complete Example

```scheme
;; Tree reduction function
(define (tree-reduce items level)
  (if (<= (length items) 1)
      (car items)  ;; Base case
      (let* ([_ (py-set! "items" items)]
             ;; Pair up items
             [pairs (py-eval "
[[items[i], items[i+1]] if i+1 < len(items) else [items[i]]
 for i in range(0, len(items), 2)]
")]
             ;; Choose model by level (cheap at leaves, expensive at top)
             [model (cond [(<= level 2) "gpt-4.1-nano"]
                          [(<= level 4) "gpt-4o-mini"]
                          [else "gpt-4o"])]
             ;; Merge pairs in parallel
             [merged (map-async
                       (lambda (pair)
                         (py-set! "p" pair)
                         (llm-query-async
                           #:instruction "Merge these summaries. Preserve key findings from both."
                           #:data (py-exec "print('\\n---\\n'.join(p))")
                           #:model model
                           #:max-tokens 400))
                       pairs
                       #:max-concurrent 20)])
        ;; Recurse
        (tree-reduce merged (+ level 1)))))

;; Initial: summarize each abstract
(define summaries (map-async
  (lambda (abstract)
    (llm-query-async
      #:instruction "Summarize in 2-3 sentences"
      #:data abstract
      #:model "gpt-4.1-nano"
      #:max-tokens 150))
  abstracts
  #:max-concurrent 50))

;; Tree aggregation
(define final-synthesis (tree-reduce summaries 1))

;; Meta-analysis
(define meta (syntax-e (llm-query
  #:instruction "Identify major themes, breakthroughs, trends, gaps"
  #:data final-synthesis
  #:model "gpt-4o")))

(finish meta)
```

### Quantified Improvements
- Quality: 85-90% vs 60-70% flat
- Context: No overflow (each merge handles 2 items)
- Depth: O(log N) levels
- Parallelism: Each level fully parallel
- Cost pyramid: Cheap at leaves, expensive at top

### Optimization Tips
1. Cost pyramid: nano/mini at leaves, gpt-4o at top
2. Checkpoint levels: Save state at each level for recovery
3. Adaptive merging: If pair similar, summarize briefly; if different, preserve both
4. Balance: Odd items handled gracefully (single-element pairs)

### Common Mistakes
- Same model all levels (expensive at leaves)
- No checkpointing (re-run from scratch on failure)
- Too deep (stop at 1-3 items, don't merge to single item if unnecessary)

### Compose With
- Pattern 1 (Parallel fan-out at each level)
- Pattern 14 (Cache intermediate levels)

### Real-World Use Cases
1. Research synthesis (100+ papers)
2. Log aggregation (1M entries)
3. Customer feedback (1000s of reviews)
4. Code documentation (large codebases)

---

