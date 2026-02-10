## RLM-Scheme: LLM Orchestration Patterns

A persistent Scheme REPL for orchestrating LLM sub-calls safely and efficiently.

# PART I: QUICK REFERENCE

## When to Use RLM-Scheme

Use rlm-scheme instead of other tools when:

**1. Data exceeds context window**
- Single document too large (>100K tokens)
- Need to process many documents (50+ papers, 1000+ items)
- Solution: Chunk + parallel processing + synthesis

**2. Multi-step pipeline needed**
- Extract → Transform → Validate → Synthesize
- Each step uses different models/strategies
- Results from step N inform step N+1

**3. Cost control critical**
- Budget limited, need strategic allocation
- Use cheap models (gpt-4.1-nano) for bulk, expensive (gpt-4o) for synthesis
- Active learning: expensive model only on hard cases

**4. Quality requirements high**
- Single model accuracy insufficient (need 95%+)
- Ensemble voting (3-5 models) or validation loops
- Critical decisions (medical, legal, financial)

**5. Unknown data structure**
- Schema unknown until runtime
- Let model write its own analysis strategy
- Adaptive to data characteristics

**6. Latency critical**
- P99 latency hurts UX (stragglers)
- Hedge with multiple approaches (speculative execution)
- Real-time processing (streaming)

---

## Core Workflow

```
1. load_context("data")              # Optional: Load input data
2. execute_scheme("(finish ...)")    # Run orchestration code
3. Response: {"status": "finished", "value": "..."}
4. get_scope_log()                   # Optional: Audit trail
5. reset()                           # Between unrelated tasks
```

## Sandbox State Persistence

**All state persists across `execute_scheme` calls** until `reset()` is called:

- **Scheme bindings:** `(define x 42)` in call N is available in call N+1
- **Python variables:** Variables set via `py-exec` or `py-set!` persist across calls
- **Loaded contexts:** `load_context()` data remains available

This enables **multi-step pipelines** where each `execute_scheme` call builds on previous state:
```
Call 1: (py-exec "import json; data = json.load(open('input.json'))")
Call 2: (define analysis (syntax-e (llm-query #:instruction "analyze" #:data (py-eval "str(data)") ...)))
Call 3: (py-set! "analysis" analysis) (py-exec "save(analysis)")
```
Call `reset()` between **unrelated tasks** to clear all state and start fresh.

## Progress Monitoring

During long-running `execute_scheme` calls, progress is reported automatically:

- **MCP progress notifications** update every 2-5s showing: completed/dispatched calls, active models, elapsed time (e.g., "3/7 done — 2 active: gpt-4.1-nano×2 (longest: 45s)")
- **Execution summary** is always included in the response: `{"execution": {"calls": 7, "elapsed": 12.3, "tokens": 5000}}`
- **`(display ...)` output** is captured and returned in the response `stdout` field (available after completion, not streamed)
- **Timeout warnings** are emitted at 80% of the timeout limit

For custom progress in pipelines, use `(eprintf "Phase ~a/~a complete\n" i n)` — this writes to stderr and appears in the server log.

---

## 16 Patterns by Optimization Goal

| Goal | # | Pattern | Key Metric |
|------|---|---------|------------|
| **Speed** | 1 | Parallel Fan-Out | 10-50× faster |
| | 7 | Speculative Execution | 10× P99 |
| | 15 | Stream Processing | Real-time (<1s) |
| **Quality** | 4 | Critique-Refine Loop | 60% → 90% |
| | 8 | Ensemble Voting | 82% → 95% |
| | 11 | Consensus Protocol | <1% error |
| **Cost** | 9 | Active Learning | 3-5× savings |
| | 14 | Memoization | 50%+ savings |
| | 16 | Multi-Armed Bandit | 20-40% savings |
| **Adaptive** | 2 | Code Generation | Unknown schema |
| | 6 | Meta-Orchestration | Unknown strategy |
| **Structure** | 3 | Recursive Delegation | Hierarchical data |
| | 10 | Tree Aggregation | 20+ chunks |
| **Specialized** | 5 | Cumulative Fold | Order matters |
| | 12 | Backtracking Search | Verifiable solutions |
| | 13 | Anytime Algorithms | Graceful degradation |

See Part III for pattern descriptions. Call `get_pattern_details(N)` for full code examples.

---

## Model Selection Quick Guide

| Task Type | Model | Cost (per 1M in) | Use When |
|-----------|-------|------------------|----------|
| **Fan-out (map-async)** | gpt-4.1-nano | $0.10 | ALWAYS for parallel work (10-25× cheaper) |
| **Classification** | gpt-4.1-nano | $0.10 | Simple categories, extraction, yes/no |
| **Summarization** | gpt-4o-mini | $0.15 | Factual condensation, straightforward |
| **Complex reasoning** | gpt-4o | $2.50 | Multi-step logic, synthesis, nuance |
| **Code generation** | gpt-4.1 | $2.00 | Writing code, technical accuracy |
| **Math/Logic (o-series)** | o3-mini, o4-mini | $1.10 | Proofs, equations, formal reasoning |
| **Vision** | gpt-4o, gpt-4.1 | $2.50-$2.00 | Image analysis (with #:image) |
| **JSON output** | Any + #:json #t | - | Structured data (MUST include 'json' in instruction) |

**Key rule:** Default to **gpt-4.1-nano** for fan-out, **gpt-4o** for single complex tasks.

**Note:** o-series models (o3-mini, o4-mini) do not support `#:temperature` parameter.

---

## Most Common Mistakes

❌ **Using gpt-4o for fan-out**
```scheme
(map-async (lambda (x) (llm-query-async ... #:model "gpt-4o")) items)
;; Cost: 100 items × $0.05 = $5.00
```
✅ **Fix: Use gpt-4.1-nano**
```scheme
(map-async (lambda (x) (llm-query-async ... #:model "gpt-4.1-nano")) items)
;; Cost: 100 items × $0.001 = $0.10 (50× cheaper)
```

❌ **Sequential processing of independent items**
```scheme
(map (lambda (x) (syntax-e (llm-query ...))) items)  ;; N × 2s sequential
```
✅ **Fix: Use map-async**
```scheme
(map-async (lambda (x) (llm-query-async ...)) items)  ;; N/10 × 2s parallel
```

❌ **Flat aggregation of 50+ chunks**
```scheme
(define all (string-join chunks))
(llm-query #:data all ...)  ;; Context overflow, loses info
```
✅ **Fix: Use tree aggregation (Pattern 10)**
```scheme
(tree-reduce chunks)  ;; Hierarchical pairwise merging
```

❌ **No caching on repeated queries**
```scheme
(llm-query #:instruction "How do I authenticate?" ...)  ;; Called 1000 times
```
✅ **Fix: Use memoization (Pattern 14)**
```scheme
(cached-query "How do I authenticate?" ...)  ;; 60% hit rate = 60% savings
```

❌ **Single model on critical decision**
```scheme
(llm-query #:instruction "Diagnose patient" ...)  ;; 85% accuracy insufficient
```
✅ **Fix: Use ensemble (Pattern 8)**
```scheme
(ensemble-vote models question)  ;; 95% accuracy (3-5 models voting)
```

❌ **Using py-eval for multi-line code or imports**
```scheme
(py-eval "import json\nwith open('f.json') as f:\n    json.load(f)")  ;; SyntaxError!
```
✅ **Fix: Use py-exec for statements, py-eval to retrieve**
```scheme
(py-exec "import json\nwith open('f.json') as f:\n    result = json.load(f)")
(define data (py-eval "result"))
```

---

# PART II: PROBLEM → PATTERN DECISION FRAMEWORK

## How to Choose a Pattern

### Step 1: Identify Your Primary Constraint

Ask yourself: **"What's my PRIMARY bottleneck?"**

```
├─ LATENCY (users waiting, need speed)
│  └─ Go to Section 2.1
│
├─ QUALITY (accuracy/correctness critical)
│  └─ Go to Section 2.2
│
├─ COST (budget limited, need efficiency)
│  └─ Go to Section 2.3
│
├─ STRUCTURE (data format unknown or complex)
│  └─ Go to Section 2.4
│
└─ SCALE (large dataset, 1000+ items)
   └─ Go to Section 2.5
```

---

### 2.1 LATENCY-Constrained Problems

**Symptom:** Users complaining about slow responses, P99 latency hurts UX

**Q1: Are tasks independent and parallelizable?**
├─ YES → **Pattern 1: Parallel Fan-Out**
│   - Example: Analyze 50 documents simultaneously
│   - Why: Amortizes latency across workers
│   - Improvement: 10-50× faster (25min → 2.5min)
│   - Code: map-async with #:max-concurrent 10
│
└─ NO, but some requests are stragglers (P99 >> P50)
    └─ **Pattern 7: Speculative Execution (Hedging)**
       - Example: Medical scan analysis (P99: 45s → 4s)
       - Why: Hedges against tail latency
       - Improvement: 10× P99 improvement, 2× cost
       - Code: await-any with 3 approaches

**Q2: Is data arriving continuously (streaming)?**
└─ YES → **Pattern 15: Stream Processing**
   - Example: Real-time log monitoring
   - Why: Incremental updates (no batch delay)
   - Improvement: Real-time (<1s) vs batch (30min avg)
   - Code: Fold over stream with checkpoint

**Quantified improvements:**
- Parallel Fan-Out: 10-50× latency reduction
- Hedging: 5-10× P99 improvement (2× cost)
- Streaming: Real-time vs batch delay

---

### 2.2 QUALITY-Constrained Problems

**Symptom:** Single model accuracy insufficient, errors are costly

**Q1: Is this a classification/yes-no task?**
├─ YES → **Pattern 8: Ensemble Voting**
│   - Example: Medical diagnosis (82% → 95% accuracy)
│   - Why: Errors are uncorrelated (Condorcet's theorem)
│   - Improvement: +10-15% accuracy
│   - Cost: 3-5 models = 3-5× API calls
│
└─ NO, it's generation/writing
    └─ **Pattern 4: Critique-Refine Loop**
       - Example: Technical writing (60% → 90% quality)
       - Why: Iterative improvement with validation
       - Improvement: +20-30% quality
       - Cost: 3× (generate + critique + refine)

**Q2: Is this a life-critical decision?**
└─ YES → **Pattern 11: Consensus Protocol**
   - Example: High-stakes medical/legal decision
   - Why: Byzantine fault tolerance (provable safety)
   - Improvement: <1% error rate
   - Cost: 5 models × 2 rounds = 10× (mission-critical only)

**Q3: Can you verify correctness quickly?**
└─ YES → **Pattern 12: Backtracking Search**
   - Example: Math problem solving
   - Why: Fast verification catches errors early
   - Improvement: Tries multiple strategies until valid
   - Cost: Variable (stops on first valid solution)

**Quantified improvements:**
- Ensemble: +10-15% accuracy (3× cost)
- Critique-Refine: +20-30% quality (3× cost)
- Consensus: <1% error (10× cost, critical only)
- Backtracking: 90-95% success rate (2-5× cost)

---

### 2.3 COST-Constrained Problems

**Symptom:** Budget limited, need to minimize API spend

**Q1: Do you have repeated/duplicate queries?**
├─ YES (30%+ repeats) → **Pattern 14: Memoization**
│   - Example: API docs Q&A (60% repeat rate)
│   - Why: Cache hits are free
│   - Improvement: 30-70% cost reduction
│   - Implementation: checkpoint/restore with content hash
│
└─ NO repeats

**Q2: Are items variable difficulty (some easy, some hard)?**
└─ YES → **Pattern 9: Active Learning**
   - Example: 5000 legal docs ($25 → $5)
   - Why: Cheap model on easy 90%, expensive on hard 10%
   - Improvement: 3-5× cost reduction
   - Strategy: Confidence-based routing

**Q3: Is this long-running with many similar tasks?**
└─ YES → **Pattern 16: Multi-Armed Bandit**
   - Example: Email classification (learns optimal model)
   - Why: Automatically discovers cheapest effective model
   - Improvement: 20-40% savings over 1000+ tasks
   - Algorithm: UCB (explore-exploit tradeoff)

**Quantified savings:**
- Memoization: 30-70% (depends on repeat rate)
- Active Learning: 60-80% (3-5× reduction)
- Multi-Armed Bandit: 20-40% (over time)

---

### 2.4 STRUCTURE-Constrained Problems

**Symptom:** Data structure unknown until runtime, or approach unclear

**Q1: Do you know the optimal analysis strategy?**
├─ NO → **Pattern 2: Code Generation**
│   - Example: Unknown API response format
│   - Why: Model adapts to data structure automatically
│   - When: Schema unknown, data-dependent strategy
│   - Implementation: Model writes Scheme code, you execute it
│
└─ NO, and task is complex multi-phase
    └─ **Pattern 6: Meta-Orchestration**
       - Example: Customer support ticket analysis
       - Why: Planning model chooses optimal pipeline
       - When: Complex task, unclear best approach
       - Implementation: Planner writes code, executor runs

**Q2: Is data hierarchical (book → chapters → sections)?**
└─ YES → **Pattern 3: Recursive Delegation**
   - Example: 300-page contract analysis
   - Why: Each level specializes (3-level depth)
   - When: Natural hierarchy, autonomy per level
   - Implementation: #:recursive #t (sub-models get sandboxes)

**Why these work:**
- Code Generation: Meta-programming (adapts to any structure)
- Meta-Orchestration: Discovers novel strategies
- Recursive: Matches hierarchical problem structure

---

### 2.5 SCALE-Constrained Problems

**Symptom:** Large dataset (1000+ items), need efficient processing

**Q1: How many chunks to aggregate?**
├─ <20 chunks → **Pattern 1: Parallel Fan-Out**
│   - Simple: map-async + flat synthesis
│   - Cost: N × cheap + 1 × expensive
│
└─ 20-100+ chunks → **Pattern 10: Tree Aggregation**
    - Hierarchical: Pairwise reduction (log N depth)
    - Why: Preserves information (flat loses detail)
    - Cost: Cost pyramid (cheap at leaves, expensive at top)

**Q2: Can you process incrementally as data arrives?**
└─ YES → **Pattern 15: Stream Processing**
   - Example: Log monitoring (bounded memory)
   - Why: Don't need to store full dataset
   - Memory: O(1) vs O(N) batch

**Q3: Do you have overlapping sub-problems?**
└─ YES → **Pattern 14: Memoization**
   - Cache sub-results, avoid redundant computation
   - Example: Recursive tree with shared subtrees

**Scalability:**
- Parallel Fan-Out: Linear scaling with workers
- Tree Aggregation: O(log N) depth vs O(1) flat
- Streaming: O(1) memory vs O(N) batch
- Memoization: O(unique queries) vs O(total queries)

---

### 2.6 Common Scenarios → Pattern Mapping

**"Analyze 500 scientific papers"**
→ Pattern 1 (Parallel Fan-Out) for speed
→ Pattern 10 (Tree Aggregation) for synthesis quality
→ Composition: Fan-out leaves + Tree reduce aggregation

**"Medical image classification: 10K images, $500 budget, need 99% accuracy"**
→ Pattern 9 (Active Learning): Cheap on all, find uncertain
→ Pattern 8 (Ensemble): 5 models vote on uncertain 10% only
→ Composition: $10 (cheap on all) + $250 (ensemble on 500 uncertain) = $260 ✓

**"Real-time chat moderation, need <1s latency"**
→ Pattern 13 (Anytime): Quick scan (nano, <1s) → Detailed if time allows
→ Graceful degradation under time pressure

**"API documentation Q&A bot: 1000 queries/day, 60% repeats"**
→ Pattern 14 (Memoization): Check cache → If miss, query + cache
→ 60% hit rate = 60% cost savings ($100/day → $40/day)

**"Customer support tickets, unknown optimal strategy"**
→ Pattern 6 (Meta-Orchestration): Planning model designs pipeline
→ Adapts to data characteristics automatically

**"P99 latency is 30s (median: 2s), need responsive UI"**
→ Pattern 7 (Hedging): Launch 3 approaches, await-any, cancel losers
→ P99: 30s → 4s (7.5× improvement), Cost: 2× (worth it)

---

### 2.7 Pattern Combination Recipes

**Recipe 1: Budget-Constrained Quality**
= Pattern 9 (Active Learning) + Pattern 8 (Ensemble)
```
Phase 1: Cheap model on ALL items (gpt-4.1-nano)
Phase 2: Identify low-confidence 10% (confidence < 0.7)
Phase 3: Ensemble (5 models) on uncertain only
Result: 95%+ accuracy at 1/3 cost of full ensemble
```

**Recipe 2: Ultra-Low Latency with Caching**
= Pattern 14 (Memoization) + Pattern 7 (Hedging)
```
Step 1: Check cache (content hash)
Step 2: If miss, launch 3 approaches in parallel
Step 3: await-any, take first result, cancel others
Step 4: Cache winner
Result: 0.1s on hit, 4s on miss (vs 30s naive)
```

**Recipe 3: Hierarchical MapReduce**
= Pattern 1 (Parallel Fan-Out) + Pattern 10 (Tree Aggregation)
```
Level 0: Analyze 500 papers in parallel (gpt-4.1-nano)
Level 1-5: Tree reduce pairwise (better models at top)
Result: Quality preservation + cost optimization
```

**Recipe 4: Adaptive Long-Running System**
= Pattern 16 (Bandit) + Pattern 9 (Active Learning)
```
Bandit learns: Which model is best for this task type?
Active learning: Allocate budget (cheap/expensive split)
Result: Self-optimizing (cheaper + better over time)
```

**Recipe 5: Production-Grade Pipeline**
= Pattern 13 (Anytime) + Pattern 14 (Cache) + Pattern 4 (Critique)
```
Step 1: Check cache (instant if hit)
Step 2: If miss, quick result (anytime level 1)
Step 3: Refine if time allows (anytime level 2-3)
Step 4: Validate with critique before returning
Step 5: Checkpoint intermediate results
Result: Robust, fault-tolerant, quality-assured
```

---

---

# PART III: PATTERN SUMMARIES

The 16 patterns below provide brief overviews. **For complete code examples, quantified
improvements, optimization tips, and detailed implementations, call the get_pattern_details()
tool with the pattern number(s) you need.**

Example: `get_pattern_details([1, 4])` to get full details for Patterns 1 and 4.

## Pattern Summaries

**1. Parallel Fan-Out (MapReduce)** - Process independent items in parallel with cheap model, synthesize with expensive model. 10× faster, 7× cheaper. Use for 10+ independent chunks.

**2. Code Generation (Meta-Programming)** - Model writes Scheme code adapted to your data structure. Use when schema is unknown or strategy unclear. 100% adaptable.

**3. Recursive Delegation** - Hierarchical decomposition with #:recursive #t. Each level delegates to specialist sub-agents. Use for naturally hierarchical data (max 3 levels).

**4. Critique-Refine Loop** - Generate → critique (cheap) → refine (expensive) → repeat. Quality improves from 60% to 90%+. Use when single-shot quality insufficient.

**5. Cumulative Fold** - Sequential synthesis where each item builds on previous context. Use when order matters and later items should reference earlier ones.

**6. Meta-Orchestration** - Planning LLM inspects data and writes optimal pipeline code. Use when multiple strategies exist and you don't know which is best.

**7. Speculative Execution (Hedging)** - Launch 3 parallel approaches, take first to complete. P99 latency improves 10×. Use when tail latency hurts UX.

**8. Ensemble Voting** - 3-5 models vote, majority wins. Accuracy improves from 82% to 95%. Use for classification tasks where errors are costly.

**9. Active Learning** - Cheap model on all items, expensive model on uncertain 10%. 60-80% cost savings. Use when items have variable difficulty.

**10. Tree Aggregation** - Hierarchical pairwise reduction for 20+ chunks. Preserves information better than flat aggregation. Use when fan-out produces many results.

**11. Consensus Protocol** - Byzantine fault-tolerant voting with 5 models over 2 rounds. <1% error rate. Use only for mission-critical decisions (10× cost).

**12. Backtracking Search** - Try strategy → verify → if invalid, backtrack and try another. Use when you can verify correctness quickly.

**13. Anytime Algorithms** - Progressive refinement with time budget. Quick result first, refine if time allows. Use for real-time systems with graceful degradation.

**14. Memoization** - Cache results with checkpoint/restore and content hashing. 30-70% cost savings on repeated queries. Use when 30%+ queries repeat.

**15. Stream Processing** - Process data incrementally as it arrives. O(1) memory vs O(N) batch. Use for real-time monitoring or unbounded streams.

**16. Multi-Armed Bandit** - Learn optimal model over time via explore-exploit. 20-40% savings over 1000+ tasks. Use for long-running systems.

---

**Next Steps:**
1. Use the decision framework above (Part II) to identify which pattern(s) fit your problem
2. Call `get_pattern_details([pattern_ids])` to get complete implementations with code examples
3. Refer to `get_code_generation_api_reference()` for condensed API docs when writing code
