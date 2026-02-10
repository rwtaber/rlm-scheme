# RLM-Scheme: Creative LLM Orchestration

**Think of this as a vocabulary for expressing orchestration strategies, not a cookbook of recipes.**

The 16 patterns below are **primitives** you compose to solve problems. The most interesting solutions combine 3-4 patterns in ways that aren't documented here. Your job: understand the vocabulary, then speak creatively.

---

# Quick Start: Think Composition First

## The Core Idea

You have a powerful primitive: **ask a cheap model to compare/critique/select between options**. This unlocks:

```scheme
;; Don't just run one strategy - test 3 and let a model choose
(define approach-a (llm-query-async #:instruction "Strategy A..." ...))
(define approach-b (llm-query-async #:instruction "Strategy B..." ...))
(define approach-c (llm-query-async #:instruction "Strategy C..." ...))

;; All run in parallel (2s total, not 6s)
(define results (await-all (list approach-a approach-b approach-c)))

;; Cheap model picks winner based on your criteria
(define winner (syntax-e (llm-query
  #:instruction "Compare these 3 on [your criteria]. Return JSON: {winner: 'a'|'b'|'c', reasoning: str}"
  #:data (string-join results "\n\n---\n\n")
  #:json #t #:model "curie")))  ;; $0.15/1M vs $2.50/1M

;; Parse and use winner for full dataset
```

**This pattern (strategy exploration) isn't one of the 16 patterns - it's what you create by composing them.**

---

## Immediate Creative Techniques

### 1. Mix Cheap and Expensive Models Aggressively

The cost difference between models is **50-100×**:
- `ada`: $0.0004/1K tokens (simplest - classification, filtering)
- `gpt-3.5-turbo`: $0.002/1K tokens (extraction, bulk processing)
- `curie`: $0.002-0.03/1K tokens (critique, summarization)
- `gpt-4`: $0.03-0.06/1K tokens (synthesis, complex reasoning - expensive!)

**Creative opportunity:** Use ada/gpt-3.5-turbo for 80% of work, gpt-4 for 20%. Test which 20% needs expensive.

```scheme
;; Phase 1: gpt-3.5-turbo extracts from 100 documents (parallel, 10s, ~$0.20)
(define extractions (map-async turbo-extract documents))

;; Phase 2: ada or curie identifies uncertain (2s, ~$0.04)
(define uncertain-indices (identify-low-confidence extractions))

;; Phase 3: gpt-4 only on uncertain 20% (5s, ~$1.20 not $6.00)
(define refined (map-async expensive-refine (select extractions uncertain-indices)))

;; Total: ~$1.44 vs $6.00 (76% savings), same quality
```

### 2. Parallelize Everything by Default

Parallel calls cost the **same** as sequential but are 10-50× faster:

```scheme
;; Sequential: 50 × 2s = 100s
(map (lambda (doc) (syntax-e (llm-query ...))) documents)

;; Parallel: 50/10 concurrent × 2s = 10s (same cost!)
(map-async (lambda (doc) (llm-query-async ...)) documents)
```

**Creative opportunity:** Launch multiple strategies in parallel, use first to complete (hedging) or use all results (ensemble).

### 3. Let Models Design the Strategy

Don't choose patterns yourself - let a cheap model inspect your data and recommend:

```scheme
(define sample (take-first-5 documents))

(define recommendation (syntax-e (llm-query
  #:instruction "Analyze these samples. Recommend strategy:
- Uniform quality? → Fast fan-out with nano
- Variable quality? → Active learning (nano on all, expensive on uncertain)
- Hierarchical structure? → Recursive delegation
- Unknown structure? → Generate custom analysis code

Return JSON: {strategy: str, reasoning: str, estimated_cost: float}"
  #:data sample #:json #t #:model "curie")))

;; Execute recommended strategy (dispatch based on JSON response)
```

### 4. A/B Test Pattern Choices

When uncertain, test 2-3 approaches on 10-20% of data and compare empirically:

```scheme
;; Test both tree aggregation and flat fan-out on 20 papers
(define test-set (take-first 20 papers))

(define flat-result (flat-fanout-then-synthesize test-set))
(define tree-result (tree-aggregate test-set))

;; Cheap model compares quality
(define comparison (syntax-e (llm-query
  #:instruction "Which preserves more detail? Return winner + scores"
  #:data (string-append "FLAT:\n" flat-result "\n\nTREE:\n" tree-result)
  #:json #t)))

;; Use winner on remaining 80 papers
```

**Insight:** Testing on 20% costs $0.10-0.20. Making wrong choice costs $2-5. Always test when uncertain.

---

# The 16 Patterns: Your Orchestration Vocabulary

## Speed Patterns (Latency-Driven)

### Pattern 1: Parallel Fan-Out (MapReduce)
**The fundamental pattern. Master this first.**

Process N independent items in parallel with cheap model, synthesize with expensive model.

```scheme
;; 100 papers: analyze in parallel (10s with 10 concurrency)
(define summaries (map-async (lambda (paper)
  (llm-query-async
    #:instruction "Extract key finding in 1 sentence"
    #:data paper
    #:model "gpt-3.5-turbo"))  ;; $0.10/1M
  papers))

;; Synthesize with expensive model (5s)
(define overview (syntax-e (llm-query
  #:instruction "Synthesize these 100 findings into overview"
  #:data (string-join (await-all summaries) "\n")
  #:model "gpt-4")))  ;; $2.50/1M for synthesis only

(finish overview)
```

**Metrics:** 10× faster (25min → 2.5min), 7× cheaper ($2.50 → $0.35)

**When to use:** 10+ independent items, speed matters
**Compose with:** Tree aggregation (>20 chunks), memoization (repeated), active learning (uncertain)

---

### Pattern 7: Speculative Execution (Hedging)
**Eliminate tail latency by running 3 approaches in parallel, taking first result.**

Problem: Median latency 2s but P99 is 45s (stragglers hurt UX).
Solution: Launch 3 diverse strategies, `await-any` takes first, cancel rest.

```scheme
;; 3 different models/strategies in parallel
(define approach-a (llm-query-async #:model "curie" ...))
(define approach-b (llm-query-async #:model "gpt-4" ...))
(define approach-c (llm-query-async #:model "gpt-3.5-turbo" ...))

;; First to complete wins (typically 2-4s, never 45s)
(define result (await-any (list approach-a approach-b approach-c)))
;; Cancel the others to save cost
```

**Metrics:** P99: 45s → 4s (10× improvement), 2× cost (not 3× - cancel quickly)

**Creative twist:** Combine with ensemble - use all 3 results to vote instead of taking first.

---

### Pattern 15: Stream Processing
**Process unbounded data with constant memory.**

When data doesn't fit in memory (1M log entries), maintain running state and process incrementally:

```scheme
(define (process-stream chunks initial-state)
  (if (null? chunks) initial-state
    (let* ([chunk (car chunks)]
           [updated-state (syntax-e (llm-query
             #:instruction "Update state based on this chunk. Return JSON: {error_count: int, patterns: [str], ...}"
             #:data (string-append "STATE:\n" initial-state "\n\nNEW CHUNK:\n" chunk)
             #:json #t))])
      ;; Checkpoint every 100 chunks
      (when (zero? (modulo (length chunks) 100))
        (checkpoint "stream-state" updated-state))
      ;; Discard chunk, continue with updated state
      (process-stream (cdr chunks) updated-state))))
```

**Metrics:** O(1) memory vs O(N), unlimited dataset size

**Compose with:** Tree aggregation (aggregate chunks periodically), memoization (cache chunk results)

---

## Quality Patterns (Accuracy-Driven)

### Pattern 4: Critique-Refine Loop
**Iteratively improve via cheap critic + expensive generator.**

Single-shot quality ~70%. Iterate to 90%:

```scheme
(define (refine draft iteration)
  (if (>= iteration 3) draft  ;; Max 3 iterations
    (let* ([critique (syntax-e (llm-query
             #:instruction "Identify weaknesses. Return JSON: {issues: [{type: str, severity: 1-3, fix: str}]}"
             #:data draft
             #:json #t
             #:model "curie"))]  ;; Cheap critic
           [avg-severity (calculate-average-severity critique)])
      (if (< avg-severity 1.5) draft  ;; Early stopping
        (let ([improved (syntax-e (llm-query
               #:instruction (string-append "Fix these issues:\n" critique)
               #:data draft
               #:model "gpt-4"))])  ;; Expensive fixer
          (refine improved (+ iteration 1)))))))
```

**Metrics:** 70% → 90% quality (+20 points), 3× cost, converges in 2-3 iterations

**Beyond refinement - critique as comparison operator:**
```scheme
;; Compare two methods
(define winner (critique-compare method-a method-b "Which is more efficient?"))

;; A/B test patterns
(define better-pattern (critique-compare
  (run-pattern-1 sample-data)
  (run-pattern-10 sample-data)
  "Which preserves more detail?"))

;; Adaptive routing
(define model-choice (critique-sample sample-output "Is this high-quality? If no, recommend expensive model."))
```

---

### Pattern 8: Ensemble Voting
**5 models vote, majority wins. 82% → 95% accuracy.**

If errors are uncorrelated, ensemble of 5 at 82% each → 92-95% via majority:

```scheme
;; 5 models vote in parallel (same latency as 1!)
(define votes (map-async (lambda (model)
  (llm-query-async
    #:instruction "Classify: benign or malignant? Return JSON: {diagnosis: str, confidence: float}"
    #:data medical-scan
    #:model model
    #:json #t
    #:temperature 0.0))  ;; Deterministic for voting
  (list "gpt-4" "code-davinci-002" "curie" "claude-3-5-sonnet" "gpt-3.5-turbo")))

;; Aggregate votes
(define consensus (syntax-e (llm-query
  #:instruction "Vote on diagnosis. If 3+ agree = high confidence. Return JSON: {diagnosis: str, confidence: str, vote_breakdown: {...}}"
  #:data (string-join (await-all votes) "\n---\n")
  #:json #t)))
```

**Metrics:** 82% single → 95% ensemble (+13 points), 5× cost

**Cost optimization:** 3× nano + 2× expensive = 2× cost but 90% accuracy

**Compose with:** Active learning (ensemble only on uncertain), hedging (ensemble + speed)

---

### Pattern 11: Consensus Protocol
**Byzantine fault-tolerant for mission-critical decisions (<1% error).**

Two rounds: (1) 5 models propose independently. (2) Each reviews ALL proposals and votes. Need 3/5 supermajority.

```scheme
;; Round 1: Independent proposals
(define proposals (map-async (lambda (model)
  (llm-query-async #:instruction "Propose diagnosis independently" #:model model ...))
  models))

;; Round 2: Cross-review and vote
(define votes (map-async (lambda (model)
  (llm-query-async
    #:instruction "Review ALL proposals. Vote for best. Return JSON: {vote_for: int, reasoning: str}"
    #:data (string-append "PROPOSALS:\n" (string-join proposals "\n---\n"))
    #:model model
    #:json #t))
  models))

;; Check supermajority (3/5)
(define result (check-supermajority votes))
(if (not result)
  (finish "NO CONSENSUS - escalate to human")  ;; Safe failure
  (finish result))
```

**Metrics:** <1% error, tolerates 2/5 faulty models, 10× cost

**Only use for:** Medical, legal, financial decisions where errors are catastrophic.

---

## Cost Patterns (Budget-Driven)

### Pattern 9: Active Learning
**Cheap model on all, expensive only on uncertain 20%. 3-5× savings.**

Not all examples are equally hard. 80% obvious, 20% ambiguous:

```scheme
;; Phase 1: Cheap model on ALL with confidence scores
(define cheap-results (map-async (lambda (item)
  (llm-query-async
    #:instruction "Classify. Return JSON: {label: str, confidence: 0-1}"
    #:data item
    #:json #t
    #:model "gpt-3.5-turbo"))
  items))

;; Phase 2: Identify low-confidence items
(py-exec "
uncertain = [i for i, r in enumerate(results) if json.loads(r)['confidence'] < 0.7]
")
(define uncertain-indices (py-eval "uncertain"))

;; Phase 3: Expensive model ONLY on uncertain
(define expensive-results (map-async (lambda (idx)
  (llm-query-async #:data (list-ref items idx) #:model "gpt-4" ...))
  uncertain-indices))

;; Merge: keep high-confidence cheap, replace uncertain with expensive
```

**Metrics:** $3.50 vs $25 (86% savings), 92% accuracy vs 65% cheap-only

**Compose with:** Ensemble on uncertain (active learning + ensemble = 95% at 2× cost, not 5×)

---

### Pattern 14: Memoization
**Cache query results. 30-80% hit rate = 50%+ savings.**

With temperature=0.0, output is deterministic → same input = same output:

```scheme
(define (cached-query instruction data model)
  (define cache-key (hash (string-append instruction data model)))
  (define cached (restore cache-key))
  (if cached cached
    (let ([result (syntax-e (llm-query
                     #:instruction instruction
                     #:data data
                     #:model model
                     #:temperature 0.0))])  ;; Deterministic!
      (checkpoint cache-key result)
      result)))
```

**Metrics:** 30-80% hit rate, 0.1s on hit vs 5s on miss (50× faster), 50%+ cost savings

**Normalization:** Strip whitespace before hashing to increase hit rate.

**Compose with:** Hedging (check cache, if miss then hedge 3 approaches)

---

### Pattern 16: Multi-Armed Bandit
**Learn optimal model over time via explore-exploit. 20-40% savings.**

Unknown which model is best for this task? Let UCB (Upper Confidence Bound) learn:

```scheme
(define (ucb-select-model models trials successes total)
  (define scores (map (lambda (model i)
    (define n-trials (list-ref trials i))
    (define n-success (list-ref successes i))
    (if (= n-trials 0)
      +inf.0  ;; Untried models have infinite score (explore)
      (+ (/ n-success n-trials)  ;; Average success (exploit)
         (sqrt (/ (* 2 (log total)) n-trials)))))  ;; Exploration bonus
    models (range (length models))))
  ;; Select model with highest UCB score
  (list-ref models (argmax scores)))

;; Use for 100+ queries, learns optimal allocation
```

**Metrics:** Converges in ~100 trials, 20-40% savings vs uniform

**Compose with:** Active learning (bandit for model selection in Phase 1)

---

## Structure Patterns (Data-Driven)

### Pattern 2: Code Generation (Meta-Programming)
**Model writes custom Scheme code to analyze data.**

When data structure is unknown, let model inspect and write analysis code:

```scheme
(define sample (take-first-5 documents))

;; Model writes Scheme code
(define generated-code (syntax-e (llm-query
  #:instruction "Write Scheme code to analyze this data. Use (llm-query ...), (py-exec ...), etc. End with (finish result)."
  #:data (string-append "SAMPLE:\n" sample "\n\nAPI Reference:\n" api-ref)
  #:model "gpt-4")))

;; Execute generated code in sandbox
(define result (unsafe-exec-sub-output generated-code))
```

**Metrics:** 30min manual → 60s automated, 100% adaptable to any schema

**Compose with:** Critique-refine (validate generated code), anytime (generate simple first, refine if time)

---

### Pattern 3: Recursive Delegation
**Hierarchical tree of autonomous sub-agents.**

Book → chapters → sections → paragraphs, each level with its own specialist:

```scheme
(define (analyze-hierarchically data depth max-depth)
  (if (or (> depth max-depth) (small-enough? data))
    ;; Base case: direct analysis
    (syntax-e (llm-query #:instruction "Analyze this section" #:data data ...))
    ;; Recursive case: delegate to sub-agent with its own sandbox
    (syntax-e (llm-query
      #:instruction "Break this into subsections. For each, recursively analyze. Combine results."
      #:data data
      #:recursive #t  ;; Sub-model gets its own sandbox!
      #:model "gpt-4"))))
```

**Metrics:** 85-95% quality (preserves hierarchy), O(log N) depth, max 3 levels

**Critical:** Must use synchronous `llm-query` (not async) for recursive delegation.

**Compose with:** Fan-out (parallelize siblings at each level), tree aggregation

---

### Pattern 6: Meta-Orchestration
**Planning LLM designs the optimal pipeline.**

When multiple strategies exist and you don't know which is best, let a model decide:

```scheme
(define sample (take-first-5 documents))

;; Planning model generates orchestration code
(define pipeline-code (syntax-e (llm-query
  #:instruction "Analyze this data sample. Write Scheme code for optimal pipeline.
Consider:
- Uniform? → fan-out with nano
- Variable difficulty? → active learning
- Hierarchical? → recursive delegation
- Large dataset? → tree aggregation

Write complete executable code using available primitives."
  #:data (string-append "SAMPLE:\n" sample "\n\nAPI:\n" api-reference)
  #:model "gpt-4")))

;; Execute generated pipeline
(define result (unsafe-exec-sub-output pipeline-code))
```

**Metrics:** Adapts strategy automatically, planning costs $0.50 but saves $2-5

**This IS Pattern 2 (code generation) applied to orchestration itself.**

---

### Pattern 10: Tree Aggregation
**Hierarchical pairwise reduction for 20+ chunks.**

Flat concatenation hits context limits. Tree aggregation preserves information:

```scheme
(define (tree-reduce items merge-fn)
  (if (<= (length items) 1) (car items)
    (let* ([pairs (chunk items 2)]  ;; [[a,b], [c,d], [e,f], ...]
           [merged (map-async (lambda (pair)
                      (llm-query-async
                        #:instruction "Merge these two summaries"
                        #:data (string-join pair "\n\n")
                        #:model "curie"))  ;; Cheap at lower levels
                    pairs)])
      ;; Recurse on merged results (log N depth)
      (tree-reduce (await-all merged) merge-fn))))

;; Cost pyramid: nano at leaves, expensive at top
(define (tiered-tree-reduce items)
  (tree-reduce-with-models items
    (list "gpt-3.5-turbo" "curie" "gpt-4")))  ;; Escalate model quality at each level
```

**Metrics:** 85-90% quality vs 60-70% flat, no context overflow, O(log N) depth

**Compose with:** Fan-out (parallel at each level), memoization (cache intermediate levels)

---

## Specialized Patterns

### Pattern 5: Cumulative Fold
**Sequential synthesis where later items reference earlier.**

When order matters (literature review where Review 5 references Reviews 1-4):

```scheme
(define (cumulative-fold items initial-state)
  (if (null? items) initial-state
    (let ([updated-state (syntax-e (llm-query
            #:instruction "Integrate this new item into accumulated knowledge. Later items should reference earlier patterns."
            #:data (string-append "ACCUMULATED:\n" initial-state "\n\nNEW ITEM:\n" (car items))
            #:model "gpt-4"))])
      (cumulative-fold (cdr items) updated-state))))
```

**Metrics:** 10× slower than parallel but tracks cross-item dialogue

**When NOT to use:** If order doesn't matter, fold is wasteful - use parallel fan-out instead.

**Compose with:** Tree aggregation (fold small batches, tree-reduce batches)

---

### Pattern 12: Backtracking Search
**Try strategies sequentially, backtrack on failure.**

When multiple approaches exist and you can verify correctness:

```scheme
(define (backtracking-search strategies data)
  (if (null? strategies) #f  ;; No solution
    (let* ([strategy (car strategies)]
           [solution (syntax-e (llm-query
                       #:instruction (string-append "Solve using: " strategy)
                       #:data data
                       #:model "gpt-4"))]
           [valid? (verify-solution solution)])  ;; Cheap verifier
      (if valid? solution
        ;; Backtrack: try next strategy
        (backtracking-search (cdr strategies) data)))))

;; Order strategies by likelihood (try best first)
(define strategies '("dynamic-programming" "greedy" "branch-and-bound" "brute-force"))
```

**Metrics:** 90-95% success rate, 1-3 strategies executed on average (not all 5)

**Compose with:** Meta-orchestration (model generates strategy list), critique (validate instead of verify)

---

### Pattern 13: Anytime Algorithms
**Progressive refinement with graceful degradation.**

When deadline is uncertain, produce increasingly better results:

```scheme
(define (anytime-refine data deadline)
  ;; Level 1: Fast (2s, 70% quality)
  (define quick (syntax-e (llm-query #:data data #:model "gpt-3.5-turbo" ...)))
  (checkpoint "anytime-level-1" quick)

  ;; Level 2: Better (7s, 85% quality)
  (when (> (time-remaining deadline) 10)
    (define better (syntax-e (llm-query #:data data #:model "curie" ...)))
    (checkpoint "anytime-level-2" better)

    ;; Level 3: Best (22s, 95% quality)
    (when (> (time-remaining deadline) 25)
      (define best (syntax-e (llm-query #:data data #:model "gpt-4" ...)))
      (checkpoint "anytime-level-3" best))))

;; On interrupt, restore best checkpoint available
```

**Metrics:** 2s=70%, 7s=85%, 22s=95% quality. Always have a result.

**Compose with:** Critique at each level, meta-orchestration (generate cheap strategy first)

---

# Creative Composition Principles

## 1. Strategy Exploration Pattern (Not One of the 16!)

**The meta-pattern you'll use most: test multiple approaches, let cheap model choose winner.**

```scheme
;; General template for strategy exploration
(define (explore-strategies strategies data evaluation-criteria)
  ;; Run all strategies in parallel on sample
  (define results (map-async (lambda (strategy)
    (llm-query-async #:instruction (strategy-instruction strategy) #:data (sample data) ...))
    strategies))

  ;; Cheap model evaluates all results
  (define evaluation (syntax-e (llm-query
    #:instruction (string-append "Compare these approaches on: " evaluation-criteria ". Return JSON: {winner: int, reasoning: str, scores: [...]}")
    #:data (string-join (await-all results) "\n\n===\n\n")
    #:json #t
    #:model "curie")))  ;; $0.15/1M for comparison!

  ;; Apply winning strategy to full data
  (define winner-idx (parse-winner evaluation))
  (apply-strategy (list-ref strategies winner-idx) data))
```

**Use this whenever uncertain!** Testing on 10-20% costs $0.10-0.20, making wrong choice costs $2-5.

## 2. Speculative Ensemble

**Combine hedging (Pattern 7) + voting (Pattern 8) = speed + quality.**

```scheme
;; Launch 3-5 approaches in parallel
(define results (map-async diverse-strategy models-and-strategies data))

;; Option A: Take first (hedging) - 2-4s latency
(define fast-result (await-any results))

;; Option B: Use all to vote (ensemble) - same latency as A, higher quality
(define voted-result (vote-on (await-all results)))
```

**When to use:** High-stakes decisions where both speed and quality matter.

## 3. Critique-Driven Backtracking

**Combine backtracking (Pattern 12) + critique (Pattern 4) = systematic search with cheap validation.**

```scheme
(define (backtrack-with-critique strategies data)
  (if (null? strategies) #f
    (let* ([solution (generate-solution (car strategies) data)]
           [critique (syntax-e (llm-query
                       #:instruction "Validate. Return JSON: {valid: bool, issues: [str]}"
                       #:data solution
                       #:json #t
                       #:model "curie"))]  ;; Cheap validator
           [valid? (parse-validity critique)])
      (if valid? solution
        ;; Backtrack with feedback
        (backtrack-with-critique (cdr strategies)
          (string-append data "\n\nPrevious failed: " (extract-issues critique)))))))
```

**Key insight:** Cheap critique ($0.15/1M) prunes bad branches, saves expensive retries ($2.50/1M).

## 4. Active Ensemble

**Combine active learning (Pattern 9) + ensemble (Pattern 8) = 95% accuracy at 2× cost (not 5×).**

```scheme
;; Phase 1: Cheap model on all
(define cheap-results (map-async nano-classify items))

;; Phase 2: Identify uncertain 20%
(define uncertain (filter-low-confidence cheap-results))

;; Phase 3: Ensemble vote ONLY on uncertain
(define ensemble-results (map-async (lambda (item)
  (ensemble-vote models item))
  uncertain))

;; Merge: keep confident cheap results, replace uncertain with ensemble
```

**Metrics:** 95% accuracy at 2× cost vs 5× for ensemble on all.

## 5. Memoized Hedging

**Combine caching (Pattern 14) + hedging (Pattern 7) = 0.1s on hit, 4s on miss.**

```scheme
(define (cached-or-hedged query)
  ;; Check cache first
  (define cached (restore (hash query)))
  (if cached cached  ;; Cache hit: 0.1s
    ;; Cache miss: hedge 3 approaches, cache winner
    (let ([result (await-any (list
                     (approach-a query)
                     (approach-b query)
                     (approach-c query)))])
      (checkpoint (hash query) result)  ;; Cache for next time
      result)))
```

**Metrics:** 30-80% hit rate = 0.1s, 20-70% miss rate = 4s (hedged), never 45s

---

# When to Experiment vs Keep It Simple

## ✅ Experiment when:
- **Multiple approaches seem viable** - Test 2-3 on sample, let model choose
- **Uncertain about pattern choice** - A/B test on 10-20% of data
- **High stakes** (quality critical) - Combine patterns (ensemble + critique)
- **Tight constraints** (cost/latency) - Active learning, hedging, tiered models
- **Unknown data characteristics** - Let model inspect and recommend strategy

## ❌ Keep it simple when:
- **Single pattern obviously fits** - Just use it, don't over-engineer
- **Data is uniform** - No need for active learning, tiered approaches
- **Cost negligible** (<$1 total) - Fancy orchestration costs more than brute force
- **Quality already 95%+** - Don't critique-refine perfection
- **Small dataset** (<10 items) - Overhead of parallelism outweighs benefit

**Rule of thumb:** If testing costs <20% of full execution, always test. If constraints are tight, always experiment. Otherwise, start simple and iterate.

---

# Thinking in Orchestration

## The Mental Model

**Think of patterns as verbs in a language:**
- **Fan-out** = do work in parallel
- **Critique** = compare/validate/select
- **Fold** = accumulate context sequentially
- **Hedge** = try multiple, take first
- **Ensemble** = try multiple, vote
- **Tree-reduce** = aggregate hierarchically
- **Recursive** = decompose hierarchically
- **Generate-code** = meta-program
- **Memoize** = cache
- **Stream** = process incrementally

**Compose these verbs to express your strategy:**
- "Fan-out with nano, critique to find uncertain, ensemble on uncertain, tree-reduce results"
- "Generate code to inspect data, execute generated analysis, critique output, refine if needed"
- "Hedge 3 approaches on sample, choose fastest, memoize for repeated queries"

## Ask These Questions

1. **Can I test multiple strategies cheaply?** → Strategy exploration
2. **Are items independent?** → Fan-out. **Ordered?** → Fold.
3. **Is quality insufficient?** → Critique-refine, ensemble, consensus
4. **Is cost tight?** → Active learning, tiered models, memoization
5. **Is latency a problem?** → Hedging, parallelism, streaming
6. **Is structure unknown?** → Code generation, meta-orchestration
7. **Can I use cheap models for 80% of work?** → Always yes - find where

## The Creative Process

```
1. Understand constraints (cost? latency? quality?)
2. Brainstorm 2-3 strategies that could work
3. Test each on 10-20% sample (parallel, 10-30s)
4. Let cheap model compare results (2s, $0.01)
5. Apply winner to full data
6. (Optional) Add validation layer if high-stakes
```

**This process itself is an orchestration strategy** - and it's not one of the 16 patterns!

---

# Quick Reference

## Core Workflow

```
1. load_context("data")              # Optional: load input
2. execute_scheme("(finish ...)")    # Run orchestration
3. Response: {"status": "finished", "value": "...", "execution": {"calls": N, "elapsed": s, "tokens": N}}
4. reset()                           # Between unrelated tasks
```

State persists across calls until `reset()`.

## Model Selection

| Task | Model | Cost (per 1K tokens) | Use For |
|------|-------|---------------------|---------|
| **Fan-out** | gpt-3.5-turbo | $0.002 prompt + completion | ALWAYS for parallel work - cheapest for bulk processing |
| **Simple tasks** | ada | $0.0004 / $0.005 | Classification, filtering, basic extraction (50× cheaper than GPT-4) |
| **Mid-tier** | curie | $0.002 / $0.03 | Summarization, Q&A, critique (10× cheaper than GPT-4) |
| **Quality synthesis** | gpt-4 | $0.03 / $0.06 | Final output, complex reasoning, creative writing |
| **Long context** | gpt-4-32k | $0.06 / $0.12 | 32K token context (expensive - chunk and fan-out instead) |
| **Code generation** | code-davinci-002 | $0.02 / $0.05 | Writing code, technical implementations |
| **Embeddings** | text-embedding-ada-002 | $0.0004 | Semantic search, similarity, clustering |
| **Audio** | whisper-1 | $0.006/min | Speech-to-text transcription |

**Key rule:** Use ada/curie for 80% of work (fan-out, filtering, critique), gpt-4 only for final synthesis. The cost difference is 50-100×.

**Model hierarchy by cost:**
1. **ada** ($0.0004) - Cheapest, simple tasks
2. **gpt-3.5-turbo** ($0.002) - Best value for general tasks
3. **curie** ($0.002) - Good for critique/summarization
4. **gpt-4** ($0.03) - Expensive, complex reasoning only
5. **gpt-4-32k** ($0.06) - Very expensive, avoid if possible (use chunking instead)

## Common Patterns

```scheme
;; Strategy exploration (test 3, choose winner)
(define results (await-all (list (approach-a ...) (approach-b ...) (approach-c ...))))
(define winner (critique-compare results "Which is best on [criteria]?"))

;; Fan-out + tree-reduce (>20 items)
(define summaries (map-async nano-extract items))
(define final (tree-reduce summaries merge-fn))

;; Active learning (cheap on all, expensive on uncertain)
(define cheap-results (map-async nano-classify items))
(define uncertain (filter-low-confidence cheap-results))
(define expensive-results (map-async gpt4o-classify uncertain))

;; Hedge + ensemble (speed + quality)
(define results (await-all (list (model-a ...) (model-b ...) (model-c ...))))
(define voted (vote results))

;; Memoized query (cache first)
(define cached (restore (hash query)))
(if cached cached (let ([r (query ...)]) (checkpoint (hash query) r) r))
```

## Pattern Composition Reference

| If you need... | Primary pattern | Compose with | Why |
|---|---|---|---|
| Speed + Quality | Hedging (7) | Ensemble (8) | Parallel approaches + vote |
| Quality on budget | Active Learning (9) | Ensemble (8) | Expensive only on uncertain |
| Large dataset | Fan-Out (1) | Tree Aggregation (10) | Parallel + hierarchical reduce |
| Unknown strategy | Meta-Orchestration (6) | Backtracking (12) | Model designs, retries on fail |
| Repeated queries | Memoization (14) | Hedging (7) | Cache hit fast, miss hedged |
| Any pattern | Critique (4) | Any | Compare/validate/select |

---

## Primitives Reference

### LLM Calls
```scheme
(llm-query #:instruction "..." #:data "..." #:model "gpt-4")  ;; Synchronous
(llm-query-async #:instruction "..." #:data "..." #:model "gpt-3.5-turbo")  ;; Async
(await-all (list future1 future2 ...))  ;; Wait for all
(await-any (list future1 future2 ...))  ;; Wait for first (returns {id, result, remaining-ids})

;; Options
#:model "gpt-4" | "curie" | "gpt-3.5-turbo" | "code-davinci-002" | "gpt-4" | "gpt-4"
#:temperature 0.0-1.0  ;; 0.0=deterministic (for caching), 0.4-0.6=creative
#:max-tokens 1000  ;; Default 4096
#:json #t  ;; Force JSON output (MUST include "json" in instruction)
#:images (list "path/to/image.png" ...)  ;; Vision models only
#:recursive #t  ;; Sub-model gets own sandbox (Pattern 3) - MUST use sync llm-query

;; Extract text from syntax object
(syntax-e <syntax-object>)
```

### Python Integration
```scheme
(py-exec "import json\ndata = json.load(open('file.json'))")  ;; Multi-line statements
(py-eval "data['key']")  ;; Single expression, returns value
(py-set! "varname" scheme-value)  ;; Set Python variable from Scheme
```

### Checkpointing
```scheme
(checkpoint "key" value)  ;; Save to disk (.rlm-scheme-checkpoints/)
(restore "key")  ;; Load from disk, returns #f if not found
```

### Parallelism
```scheme
(map-async fn list)  ;; Map in parallel, returns list of futures
(await-all futures)  ;; Wait for all, returns list of results
(await-any futures)  ;; Wait for first, returns {id, result, remaining-ids}
```

### Code Generation (Patterns 2, 6)
```scheme
(unsafe-raw-query "Write Scheme code..." context)  ;; Returns raw string (not syntax)
(datum->syntax <raw-code>)  ;; Convert string to syntax object
(unsafe-exec-sub-output <code-string>)  ;; Execute in sub-sandbox, return output
```

### Context
```scheme
context  ;; Global variable with loaded data (from load_context MCP tool)
(get-context "name")  ;; Retrieve named context slot
```

### Utilities
```scheme
(display "message")  ;; Print to stdout (appears in response)
(eprintf "message")  ;; Print to stderr (appears in server log)
(finish value)  ;; Return value from execute_scheme
```

---

## Pattern Summaries

1. **Parallel Fan-Out** - Parallel processing with cheap model, synthesis with expensive
2. **Code Generation** - Model writes custom Scheme code for unknown structure
3. **Recursive Delegation** - Hierarchical tree of autonomous sub-agents (max 3 levels)
4. **Critique-Refine** - Iterative improvement via cheap critic + expensive generator
5. **Cumulative Fold** - Sequential synthesis where later items reference earlier
6. **Meta-Orchestration** - Planning LLM designs optimal pipeline
7. **Speculative Execution** - Hedge 3 approaches, take first result (P99 improvement)
8. **Ensemble Voting** - 5 models vote in parallel, majority wins (82% → 95%)
9. **Active Learning** - Cheap on all, expensive on uncertain 20% (3-5× savings)
10. **Tree Aggregation** - Hierarchical pairwise reduction for 20+ chunks
11. **Consensus Protocol** - Byzantine fault tolerance for mission-critical (<1% error, 10× cost)
12. **Backtracking Search** - Try strategies sequentially, verify, backtrack on failure
13. **Anytime Algorithms** - Progressive refinement with graceful degradation
14. **Memoization** - Content-addressed caching (30-80% hit rate, 50%+ savings)
15. **Stream Processing** - Constant memory for unbounded data
16. **Multi-Armed Bandit** - Learn optimal model over time (20-40% savings after 100 trials)

**For complete implementations with code examples**, call `get_pattern_details([pattern_ids])`.

---

**Remember: These are primitives, not prescriptions. The most interesting solutions combine 3-4 patterns in ways not documented here. Test multiple strategies, let cheap models compare, compose creatively.**
