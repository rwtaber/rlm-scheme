## Pattern 12: Backtracking Search (Strategy Exploration)

### Problem Statement
Solve complex optimization problem. Multiple strategies possible (linear programming, greedy, dynamic programming, branch-and-bound). Picking one = 60% success. Need 90%+.

### Why This Pattern Exists
**Problem it solves:** Unknown which strategy fits problem. Trial-and-error with verification.
**Alternatives fail because:**
- **Single strategy:** May not fit problem structure (60% success)
- **Try all parallel:** Wasteful (pay for 5 strategies when need 1)
- **Human selection:** Requires expertise

**Key insight:** Try strategies sequentially. Cheap verifier checks correctness. Backtrack on failure. Early termination on success.

### When to Use This Pattern
Use when:
- Multiple valid approaches
- Can verify solutions cheaply
- Failure recovery critical

Don't use when:
- Single obvious strategy
- Can't verify correctness
- All strategies likely to succeed (use parallel)

### How It Works
```
Strategy 1 -> Generate -> Verify -> Valid? Yes: DONE
                                  -> No: Backtrack
Strategy 2 -> Generate -> Verify -> Valid? Yes: DONE
                                  -> No: Backtrack
...
```

**Key primitives:** Recursion (backtracking), llm-query (generate + verify), py-eval (validation)

### Complete Example

```scheme
;; Cheap verifier
(define (verify-solution solution)
  (syntax-e (llm-query
    #:instruction "Verify solution correctness. Return JSON: {valid: bool, errors: [str]}"
    #:data (string-append "PROBLEM: " context "\nSOLUTION: " solution)
    #:model "gpt-4.1-nano"
    #:json #t
    #:temperature 0.0)))

;; Backtracking search
(define (search-strategies strategies)
  (if (null? strategies)
      "NO SOLUTION FOUND"
      (let* ([strategy (car strategies)]
             ;; Generate solution
             [candidate (syntax-e (llm-query
                          #:instruction (string-append "Solve using: " strategy)
                          #:data context
                          #:model "gpt-4o"
                          #:temperature 0.3))]
             ;; Verify
             [verification (verify-solution candidate)]
             [_ (py-set! "ver" verification)]
             [is-valid (py-eval "import json; json.loads(ver)['valid']")])
        (if is-valid
            ;; Success!
            (string-append "SOLUTION: " candidate "\nStrategy: " strategy)
            ;; Backtrack
            (begin
              (display (string-append "Strategy failed: " strategy "\n"))
              (search-strategies (cdr strategies)))))))

;; Strategy list
(define strategies (list
  "Linear Programming"
  "Greedy Algorithm"
  "Dynamic Programming"
  "Branch and Bound"
  "Simulated Annealing"))

(define result (search-strategies strategies))
(finish result)
```

### Quantified Improvements
- Success rate: 90-95% vs 60% single strategy
- Cost: 1-3 strategies on average (vs 5 if parallel)
- Provability: Can prove "no solution exists"
- Early termination: Stops on first success

### Optimization Tips
1. Order strategies by likelihood (try best first)
2. Cheap verifier (nano, fast rules)
3. Checkpoint candidates (recovery)
4. Parallel verification if multiple candidates

### Common Mistakes
- Expensive verifier (defeats purpose)
- No strategy ordering (waste time on unlikely)
- Infinite strategies (need termination)

### Compose With
- Pattern 6 (Meta-orchestration generates strategy list)
- Pattern 14 (Cache verified solutions)

### Real-World Use Cases
1. Optimization problems
2. Code generation (try patterns until compiles)
3. Proof search (multiple proof strategies)
4. Configuration search (try configs until valid)

---

