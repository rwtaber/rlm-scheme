## Pattern 5: Cumulative Fold (Sequential Synthesis)

### Problem Statement
You need to synthesize multiple perspectives where later items should be aware of earlier items. Example: 10 expert reviews of a research paper, where Review 5 should reference concerns from Reviews 1-4. Parallel aggregation loses cross-item dialogue and consensus tracking.

### Why This Pattern Exists
**Problem it solves:** Context accumulation across sequential items. Parallel map-reduce loses dialogue between items.  
**Alternatives fail because:**
- **Parallel fan-out:** Each review analyzed independently, misses "Review 3 addresses Review 1's concern"
- **Single-shot on all:** LLM can't track 10 perspectives simultaneously, loses nuance
- **Naive concatenation:** Exceeds context window, loses structure

**Key insight:** Human experts process sequentially (read Review 1, internalize, then read Review 2 with Review 1 in mind). Mimic this with fold.

### When to Use This Pattern
✅ Use when:
- Items have temporal/logical order (reviews, time-series, narrative)
- Later items should reference/build on earlier items
- Need consensus tracking (agreement vs disagreement across items)

❌ Don't use when:
- Items are independent (use Pattern 1 parallel fan-out instead - 10× faster)
- Order doesn't matter (fold adds no value over parallel)

### How It Works
```
Initial → Item1 → Synthesis1 → Item2 → Synthesis2 → ... → Final Synthesis
   ↓         ↓         ↓          ↓         ↓
  empty   review1   context   review2  context
                   (includes       (includes
                    review1)     reviews 1-2)
```

**Key primitives used:**
- Recursion - Natural fit for fold (Scheme's `cdr` + recursive call)
- `py-set!` + `py-eval` - Transfer list items one at a time
- `string-append` - Build instruction with current synthesis as context
- `#:temperature 0.3` - Low variance for consistent synthesis style

### Complete Example

```scheme
;; Problem: Synthesize 10 expert peer reviews with consensus tracking
;; Reviews should inform each other (cross-review dialogue)

;; Load reviews
(define reviews (py-eval "
import json
reviews = json.loads(context) if isinstance(context, str) else context
reviews[:10]
"))

;; Cumulative fold function
(define (fold-reviews review-list current-synthesis)
  (if (null? review-list)
      current-synthesis  ;; Base case: all reviews processed
      (let* ([review (car review-list)]
             [_ (py-set! "rev" review)]
             [review-json (py-eval "import json; json.dumps(rev)")]
             
             ;; Synthesize this review INTO existing synthesis
             [updated-synthesis (syntax-e (llm-query
                #:instruction (string-append
                  "Synthesizing peer reviews. Update synthesis with new review.

CURRENT SYNTHESIS:
" current-synthesis "

NEW REVIEW:
" review-json "

UPDATE SYNTHESIS:
1. If new review AGREES with existing concerns → strengthen (note consensus)
2. If new review CONTRADICTS → note disagreement explicitly
3. If new review raises NOVEL issues → add them
4. Track: Consensus strengths, Consensus weaknesses, Contentious points

Return updated synthesis.")
                #:model "gpt-4o"
                #:temperature 0.3
                #:max-tokens 800))])
        
        ;; Recurse with remaining reviews
        (fold-reviews (cdr review-list) updated-synthesis))))

;; Execute fold
(define initial "No reviews processed yet.")
(define final-synthesis (fold-reviews (py-eval "reviews") initial))

;; Final recommendation
(define recommendation (syntax-e (llm-query
  #:instruction "Based on synthesis, provide:
1. Recommendation (Accept/Minor Revisions/Major Revisions/Reject)
2. Justification (consensus points)
3. Priority actions for authors"
  #:data final-synthesis
  #:model "gpt-4o"
  #:max-tokens 400)))

(finish (string-append "=== SYNTHESIS ===\n" final-synthesis
                       "\n\n=== RECOMMENDATION ===\n" recommendation))
```

### Quantified Improvements
- **Context richness:** Review 10 has context of all 9 prior reviews (vs 0 in parallel)
- **Consensus detection:** Explicitly tracks agreement/disagreement (parallel can't do this)
- **Latency:** 10× slower than parallel (10 sequential calls vs 1 parallel batch)
- **Cost:** Same as parallel (10 calls either way), but synthesis is higher quality
- **Complexity:** O(n) sequential calls, each with O(accumulated_context)

### Optimization Tips
1. **Checkpoint intermediate synthesis:** After every 3 reviews, `(checkpoint "fold_state_3" current-synthesis)`. Recover from failures.
2. **Max tokens per synthesis:** Cap at 800-1000 to prevent runaway context growth. Force summarization.
3. **Early consensus detection:** If first 5 reviews all agree, remaining 5 add diminishing value. Consider stopping early.
4. **Use cheaper model:** If reviews are short and clear, gpt-4o-mini may suffice ($0.15/1M vs $2.50/1M).

### Common Mistakes
❌ Using fold when parallel would work (unnecessary slowdown)
```scheme
;; If reviews are independent, use Pattern 1 parallel fan-out instead
;; Fold is 10× slower when order doesn't matter
```

❌ No context size management (synthesis grows unbounded)
```scheme
;; After 50 reviews, synthesis is 100KB
;; Fix: Force summarization or chunking
(if (> (string-length current-synthesis) 50000)
    (summarize-synthesis current-synthesis)
    current-synthesis)
```

❌ Too high temperature (synthesis style changes over time)
```scheme
;; temperature 0.7 → synthesis becomes more verbose/creative over time
;; Fix: temperature 0.2-0.3 for consistent style
```

### Compose With
- **Pattern 10 (Tree Aggregation):** Fold small batches, then tree-reduce batches
- **Pattern 14 (Memoization):** Cache synthesis at checkpoints
- **Pattern 4 (Critique-Refine):** Fold with critique after every N items

### Real-World Use Cases
1. **Peer Review:** Academic paper reviews with consensus tracking
2. **Time-Series Analysis:** Financial data where each day contextualizes next
3. **Narrative Construction:** Story generation where each chapter builds on previous
4. **Code Review:** Sequential PR comments where later comments reference earlier

---

