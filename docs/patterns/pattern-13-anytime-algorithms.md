## Pattern 13: Anytime Algorithms (Progressive Refinement)

### Problem Statement
Uncertain deadline. User may interrupt after 2s or wait 30s. Need OK result fast, better result if time allows. All-or-nothing approaches fail.

### Why This Pattern Exists
**Problem it solves:** Variable latency requirements. Better to have 70% quality in 2s than nothing.
**Alternatives fail because:**
- **Single expensive call:** 30s latency, no intermediate result
- **Timeout expensive call:** Wasted cost if timeout
- **Cheap only:** Low quality even if time available

**Key insight:** Cascade of models. Nano (2s) -> Mini (5s) -> GPT-4o (15s). Each improves previous. Checkpoint each level. Interrupt anytime = use best available.

### When to Use This Pattern
Use when:
- Variable latency tolerance
- Prefer OK now to perfect later
- Can measure quality progression

Don't use when:
- Fixed deadline (use appropriate model)
- Binary quality (works or doesn't)

### How It Works
```
Level 1: Nano -> 70% quality, 2s (checkpoint)
Level 2: Mini improves -> 85% quality, 7s total (checkpoint)
Level 3: GPT-4o refines -> 95% quality, 22s total
```

**Key primitives:** llm-query (cascade), checkpoint (intermediate results), restore (on interrupt)

### Complete Example

```scheme
;; Level 1: Fast draft
(define draft-nano (syntax-e (llm-query
  #:instruction "Quick analysis. 2-3 paragraphs."
  #:data context
  #:model "gpt-3.5-turbo"
  #:max-tokens 500)))

(checkpoint "level1" draft-nano)
(display "Level 1 complete (70% quality, 2s)\n")

;; Level 2: Improvement
(define draft-mini (syntax-e (llm-query
  #:instruction (string-append "Improve this analysis:\n" draft-nano)
  #:data context
  #:model "curie"
  #:max-tokens 800)))

(checkpoint "level2" draft-mini)
(display "Level 2 complete (85% quality, 7s total)\n")

;; Level 3: Expert refinement
(define final (syntax-e (llm-query
  #:instruction (string-append "Expert refinement:\n" draft-mini)
  #:data context
  #:model "gpt-4"
  #:max-tokens 1200)))

(checkpoint "level3" final)
(display "Level 3 complete (95% quality, 22s total)\n")

(finish final)

;; If interrupted: (restore "level2") or (restore "level1")
```

### Quantified Improvements
- 2s: 70% quality
- 7s: 85% quality
- 22s: 95% quality
- Graceful degradation: Always have result

### Optimization Tips
1. Exponential quality/time: Each level 2-3x time, +10-15% quality
2. Checkpoint aggressively
3. User feedback: Show progress bar
4. Adaptive: Skip levels if previous good enough

### Common Mistakes
- Linear progression (5 levels of 2s each = no benefit)
- No checkpoints (lose work on interrupt)
- Fixed levels (should adapt to quality)

### Compose With
- Pattern 4 (Critique-refine at each level)
- Pattern 14 (Cache levels)

### Real-World Use Cases
1. Search results (instant preview, detailed on demand)
2. Code completion (fast suggestion, detailed on tab)
3. Report generation (summary fast, full report later)
4. Translation (rough fast, polished later)

---

