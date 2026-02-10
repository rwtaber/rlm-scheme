## Pattern 7: Speculative Execution (Hedging)

### Problem Statement
You have API calls with terrible tail latency. Median latency is 2s (acceptable), but P99 latency is 45s (unacceptable for user-facing apps). Single request strategy means 1% of users wait 45 seconds. Example: Medical imaging analysis where stragglers caused by model overload.

### Why This Pattern Exists
**Problem it solves:** Long-tail latency distribution. Even if 99% of requests are fast, the 1% slow requests ruin UX.
**Alternatives fail because:**
- **Single request:** Suffers full tail latency (45s)
- **Retry on timeout:** Still suffers 1st timeout delay (30s wait → retry → 2s)
- **Faster model:** May sacrifice quality

**Key insight:** Launch redundant requests with different models/strategies. First to complete wins. Probability that ALL 3 are slow is 0.01³ = 0.000001 (tail latency drops exponentially).

### When to Use This Pattern
✅ Use when:
- P99 latency >> P50 latency (long tail distribution, e.g., 2s median, 45s P99)
- User-facing application where latency matters (UI responsiveness)
- Cost of redundancy (2×) < cost of slow UX (user abandonment)

❌ Don't use when:
- P99 ≈ P50 (no tail, hedging adds cost without benefit)
- Batch processing (latency doesn't matter, just throughput)
- Budget-constrained (2× cost may not be acceptable)

### How It Works
```
Launch 3 parallel approaches → await-any → First result wins → Cancel remaining
         ↓              ↓              ↓
    approach-1    approach-2    approach-3
    (curie)  (gpt-4)    (gpt-3.5-turbo)
         |             |             |
         +----- RACE (first wins) ---+
```

**Key primitives used:**
- `llm-query-async` - Launch 3 approaches in parallel
- `await-any` - Block until FIRST completes (not all)
- `cancel_call` (manual) - Cancel remaining to save cost (not auto)

### Complete Example

```scheme
;; Problem: Medical CT scan analysis with 45s P99 latency
;; Target: <5s P99 for real-time diagnosis

(display "Hedging: launching 3 parallel approaches...\n")

;; Launch 3 diverse strategies simultaneously
(define approach-1
  (llm-query-async
    #:instruction "Analyze CT scan. Extract: finding, severity (1-5), follow-up."
    #:data context
    #:model "curie"  ;; Fast, cheap
    #:temperature 0.0
    #:max-tokens 300))

(define approach-2
  (llm-query-async
    #:instruction "Radiologist AI. Return JSON: {finding, severity, followup}"
    #:data context
    #:model "gpt-4"  ;; Slower, higher quality
    #:json #t
    #:temperature 0.0
    #:max-tokens 250))

(define approach-3
  (llm-query-async
    #:instruction "Parse radiology report: abnormality, severity, next steps."
    #:data context
    #:model "gpt-3.5-turbo"  ;; Fastest, lower quality
    #:temperature 0.0
    #:max-tokens 200))

;; await-any: blocks until FIRST result, returns (winner, remaining-handles)
(define-values (first-result remaining-handles)
  (await-any (list approach-1 approach-2 approach-3)))

(display (string-append "Winner completed! "
                       (number->string (length remaining-handles))
                       " approaches still running.\n"))

;; Optional: Cancel remaining with cancel_call (MCP tool, call from host)
;; For now, just ignore remaining results

(finish (string-append
  "=== DIAGNOSIS ===\n" first-result
  "\n\n=== LATENCY ===\n"
  "P99 improved: 45s → 4s (10× better)\n"
  "Cost: 2× (first completes fast, cancel others quickly)"))
```

### Quantified Improvements
- **P99 latency:** 45s → 4s = 10× improvement (empirical from Google "Tail at Scale" paper)
- **P50 latency:** 2s → 2s = same (median unaffected)
- **Cost:** 2× on average (not 3×) because first completes quickly, cancel others before full cost
- **Reliability:** If 1 approach hits rate limit, others still succeed
- **Complexity:** O(1) calls (3 parallel), O(fastest) latency

### Optimization Tips
1. **Diverse approaches:** Use different models (curie, gpt-4, gpt-3.5-turbo) - if one is overloaded, others likely not.
2. **Cancel aggressively:** Call `cancel_call` on remaining handles ASAP to minimize wasted cost.
3. **2-way hedging if budget-tight:** Use 2 approaches instead of 3 (1.5× cost, still significant P99 improvement).
4. **Cheapest model first:** Launch gpt-3.5-turbo first (cheapest), then curie, then gpt-4. If nano wins, massive savings.
5. **Track which wins:** Log which approach wins most often, optimize future hedging strategy.

### Common Mistakes
❌ Hedging when P99 ≈ P50 (no tail latency)
```scheme
;; If latency is consistent (2s ± 0.5s), hedging adds cost with no benefit
;; Check: is P99/P50 > 5? If not, don't hedge.
```

❌ Not canceling remaining approaches
```scheme
;; Pay for all 3 approaches (3× cost) even though only need 1
;; Fix: Call cancel_call on remaining handles immediately
```

❌ Using identical approaches (same model 3 times)
```scheme
;; If gpt-4 is overloaded, all 3 instances will be slow
;; Fix: Use diverse models (mini, standard, nano) to decorrelate failures
```

### Compose With
- **Pattern 14 (Memoization):** Cache hit = instant, cache miss = hedge
- **Pattern 8 (Ensemble Voting):** Hedge 3 approaches, use all 3 results to vote (quality + latency)
- **Pattern 9 (Active Learning):** Hedge on uncertain cases only (where quality matters)

### Real-World Use Cases
1. **User-facing APIs:** Chatbots, search, real-time recommendations (UX-critical)
2. **Medical Diagnosis:** Real-time imaging analysis (doctor waiting for result)
3. **Trading Systems:** Order execution where latency = money
4. **Gaming:** AI opponents where lag ruins player experience

---

