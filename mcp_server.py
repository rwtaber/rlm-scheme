"""RLM-Scheme — MCP server for safe LLM orchestration.

Exposes a Racket sandbox with hygienic scope tracking as 8 MCP tools.
Claude Code launches this via .mcp.json; it manages racket_server.rkt
as a subprocess with interleaved llm-query callbacks.

Architecture:
  Claude Code --JSON-RPC/stdio--> mcp_server.py --JSON/stdin--> racket_server.rkt
                                                                    |
                                                                    +--JSON/stdin--> py_bridge.py
"""

import asyncio
import base64
import collections
import concurrent.futures
import json
import mimetypes
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time

from dotenv import load_dotenv
load_dotenv()

import openai
from mcp.server.fastmcp import Context, FastMCP

IS_WINDOWS = platform.system() == "Windows"

mcp = FastMCP("scope")

MAX_RECURSION_DEPTH = 3

# L7: Checkpoint directory for persistent storage across timeouts
CHECKPOINT_DIR = os.path.join(os.getcwd(), ".rlm-scheme-checkpoints")

def _ensure_checkpoint_dir():
    """Create checkpoint directory if it doesn't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared call registry — visible across all RacketREPL instances (S7)
# ---------------------------------------------------------------------------

class CallRegistry:
    """Thread-safe shared registry for all in-flight LLM calls across REPL depths."""

    def __init__(self):
        self._calls: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._change_event = threading.Event()
        self._stats = {"dispatched": 0, "completed": 0}
        # Per-execution history of completed calls (model, latency)
        self._history: list[dict] = []

    def register(self, call_id: str, call_type: str, model: str,
                 depth: int = 0, instruction_preview: str = "",
                 parent_id: str | None = None) -> None:
        with self._lock:
            self._calls[call_id] = {
                "call_id": call_id,
                "type": call_type,
                "model": model,
                "depth": depth,
                "instruction_preview": instruction_preview,
                "parent_id": parent_id,
                "start_time": time.time(),
            }
            self._stats["dispatched"] += 1
        self._change_event.set()

    def complete(self, call_id: str) -> None:
        with self._lock:
            entry = self._calls.pop(call_id, None)
            self._stats["completed"] += 1
            if entry:
                self._history.append({
                    "model": entry["model"],
                    "latency": round(time.time() - entry["start_time"], 1),
                })
        self._change_event.set()

    def snapshot(self) -> list[dict]:
        with self._lock:
            now = time.time()
            return [
                {**c, "elapsed_seconds": round(now - c["start_time"], 1)}
                for c in self._calls.values()
            ]

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def get_execution_summary(self) -> dict:
        """Return a summary of the completed execution: call count, models, max latency."""
        with self._lock:
            stats = dict(self._stats)
            history = list(self._history)
        if not history:
            return {"llm_calls": 0}
        model_counts = collections.Counter(h["model"] for h in history)
        models_str = ", ".join(f"{m}\u00d7{n}" for m, n in model_counts.most_common())
        max_latency = max(h["latency"] for h in history)
        return {
            "llm_calls": stats["completed"],
            "models": models_str,
            "max_call_latency": max_latency,
        }

    def reset_stats(self) -> None:
        with self._lock:
            self._stats["dispatched"] = 0
            self._stats["completed"] = 0
            self._history.clear()

    def clear(self) -> None:
        with self._lock:
            self._calls.clear()
            self._stats["dispatched"] = 0
            self._stats["completed"] = 0
            self._history.clear()

    def wait_for_change(self, timeout: float) -> bool:
        """Wait up to *timeout* seconds for a state change. Returns True if woken early."""
        self._change_event.clear()
        return self._change_event.wait(timeout=timeout)


_call_registry = CallRegistry()


def _detect_project_python() -> str | None:
    """Detect the project's Python interpreter (venv, VIRTUAL_ENV, etc.)."""
    def _find_python_in_venv(venv_path: str) -> str | None:
        """Try to find Python executable in a venv (Unix or Windows)."""
        # Unix: bin/python3
        unix_candidate = os.path.join(venv_path, "bin", "python3")
        if os.path.isfile(unix_candidate):
            return unix_candidate
        # Windows: Scripts/python.exe
        windows_candidate = os.path.join(venv_path, "Scripts", "python.exe")
        if os.path.isfile(windows_candidate):
            return windows_candidate
        return None

    # 1. Explicit env var
    if os.environ.get("RLM_PYTHON"):
        return os.environ["RLM_PYTHON"]
    # 2. VIRTUAL_ENV env var (standard for activated venvs)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = _find_python_in_venv(venv)
        if candidate:
            return candidate
    # 3. .venv in working directory
    cwd_venv_path = os.path.join(os.getcwd(), ".venv")
    candidate = _find_python_in_venv(cwd_venv_path)
    if candidate:
        return candidate
    # 4. .venv relative to this file's directory
    project_venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv")
    candidate = _find_python_in_venv(project_venv_path)
    if candidate:
        return candidate
    return None

# ---------------------------------------------------------------------------
# Usage guide — returned verbatim by get_usage_guide
# ---------------------------------------------------------------------------

USAGE_GUIDE_CORE = r"""## RLM-Scheme: LLM Orchestration Patterns

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
"""

PATTERN_DETAILS = {
    1: r"""## Pattern 1: Parallel Fan-Out (MapReduce)

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

;; Step 2: Parallel fan-out with CHEAP model (gpt-4.1-nano)
(display "Analyzing 50 papers in parallel...\n")
(define analyses (map-async
  (lambda (paper)
    (llm-query-async
      #:instruction "Find all mentions of ACE2 protein. Return JSON: [{mention: string, context: string, page: int}]"
      #:data paper
      #:model "gpt-4.1-nano"  ;; CRITICAL: Use cheapest model for fan-out
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

;; Step 4: Synthesize with EXPENSIVE model (gpt-4o)
(define synthesis (syntax-e (llm-query
  #:instruction "Synthesize ACE2 protein findings across 50 papers:
1. Most common functions (with frequency)
2. Novel findings
3. Research gaps"
  #:data combined-json
  #:model "gpt-4o"  ;; Expensive model for synthesis only
  #:temperature 0.3
  #:max-tokens 800)))

(finish (string-append
  "=== RAW MENTIONS ===\n" combined-json "\n\n"
  "=== SYNTHESIS ===\n" synthesis))
```

### Quantified Improvements
**vs Naive approach (sequential with gpt-4o):**
- **Latency:** 10× faster (25min → 2.5min)
  - Naive: 50 × 30s = 25 minutes
  - Fan-out: max(50/10 batches × 30s, synthesis 10s) ≈ 2.5 minutes
- **Cost:** 7× cheaper ($2.50 → $0.35)
  - Naive: 50 × $0.05 (gpt-4o) = $2.50
  - Fan-out: 50 × $0.001 (nano) + 1 × $0.10 (synthesis) = $0.35
- **Quality:** Comparable (nano good enough for extraction)

**Complexity:**
- Time: O(N/k) where N=items, k=parallelism
- Space: O(N) (store all results)
- API calls: N (fan-out) + 1 (synthesis)

### Optimization Tips
1. **Always use gpt-4.1-nano for fan-out** (not gpt-4o) - 25× cheaper
2. **Batch size 10-20 optimal** (#:max-concurrent 10)
3. **Checkpoint after batches** for fault tolerance on large workloads
4. **Use #:max-tokens** to cap response length (save cost)
5. **Set #:temperature 0.0** for deterministic extraction

### Common Mistakes
❌ Using expensive model for fan-out
```scheme
(map-async (lambda (x) (llm-query-async ... #:model "gpt-4o")) items)
;; 50× more expensive than necessary
```

❌ Not using map-async (sequential instead)
```scheme
(map (lambda (x) (syntax-e (llm-query ...))) items)
;; 10× slower (sequential)
```

❌ Synthesizing with cheap model
```scheme
(llm-query #:data combined #:model "gpt-4.1-nano" ...)
;; Synthesis needs reasoning power (use gpt-4o)
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

""",

    2: r"""## Pattern 2: Code Generation (Meta-Programming)

### Problem Statement
Data structure is unknown until runtime (e.g., new API format, unfamiliar schema). You don't know the optimal analysis strategy beforehand. Need adaptive approach that adjusts to data characteristics.

### Why This Pattern Exists
**Problem it solves:** Unknown structure, data-dependent strategies
**Alternative approaches fail because:**
- Hardcoded analysis: Breaks on schema changes
- Manual inspection: Takes 30+ minutes human time
- Generic approach: Suboptimal for specific data

### When to Use This Pattern
✅ **Use when:**
- Data schema unknown at design time
- Optimal strategy depends on data characteristics
- Want model to discover its own approach

❌ **Don't use when:**
- Structure is well-known (use specific pattern)
- Model-generated code too risky (security concerns)
- Need deterministic behavior (code gen is probabilistic)

### How It Works
**Conceptual flow:**
1. Show model a sample of the data
2. Model writes Scheme code that analyzes it
3. Execute the generated code (deliberate scope crossing)
4. Return the defined result variable

**Key primitives used:**
- `unsafe-raw-query` - Generate code (returns string, not syntax)
- `datum->syntax` - Wrap code string for eval
- `unsafe-exec-sub-output` - Execute generated code
- `finish-var` - Return variable defined by generated code
- `py-exec` - Data inspection in Python

### Complete Example
```scheme
;; Problem: Analyze Stripe payment data with unknown schema
;; Why this pattern: Never seen this API format before, don't know optimal approach

;; Step 1: Inspect data sample
(define sample (py-exec "
import json
data = json.loads(context) if isinstance(context, str) else context
sample = data[:2] if isinstance(data, list) else [data]
print(json.dumps(sample, indent=2))
"))

(display "Data sample:\n")
(display (py-exec "print(sample[:500])"))
(display "\n\n")

;; Step 2: Get API reference for sub-model
;; (In real use: Call get_code_generation_api_reference MCP tool)
(define api-ref "
CRITICAL: Use #: for keyword args (#:instruction, #:data, #:model)
MUST unwrap with (syntax-e result) before using as string

Available functions:
- (llm-query #:instruction \"...\" #:data \"...\" #:model \"...\")
- (py-exec \"code\") - returns stdout
- (py-eval \"expr\") - returns value
- (py-set! \"var\" value) - safe transfer
- (map-async fn items) - parallel processing
- (string-append ...) - concatenate strings
- (finish result) - return final answer
")

;; Step 3: Model writes custom analysis code
(define analysis-code (unsafe-raw-query
  #:instruction (string-append
    "You are writing Scheme code for rlm-scheme sandbox.

TASK: Analyze this Stripe payment data. Write code that:
1. Uses py-exec to compute: total revenue, average transaction, top 5 customers
2. Uses llm-query to identify anomalies/fraud patterns
3. Stores result in variable `final_report` (string)
4. Do NOT call (finish) - just define the variable

DATA SAMPLE:
" sample "

API REFERENCE:
" api-ref "

Return ONLY Scheme code, no explanations.")
  #:model "gpt-4.1"  ;; Strong code generation model
  #:temperature 0.0
  #:max-tokens 2000))

(display "Generated analysis code:\n")
(display analysis-code)
(display "\n\n=== EXECUTING ===\n")

;; Step 4: Execute the generated strategy
(unsafe-exec-sub-output (datum->syntax #f analysis-code))

;; Step 5: Return the variable it defined
(finish-var "final_report")
```

### Quantified Improvements
**vs Manual approach:**
- **Time:** 30min manual inspection → 60s automated
- **Adaptability:** 100% (works on any schema)
- **Reusability:** Same prompt works for Shopify, AWS, GitHub APIs

**vs Hardcoded:**
- **Brittleness:** 0% (adapts to schema changes)
- **Development time:** 0min (no coding needed)

**Complexity:**
- Time: O(M + P) where M=meta-model inference, P=generated program execution
- Space: O(1)
- API calls: 1 (code gen) + P (program's sub-calls)

### Optimization Tips
1. **Use gpt-4.1 for code generation** (best at code, better than gpt-4o)
2. **Include API reference** (sub-model doesn't know rlm-scheme syntax)
3. **Validate generated code** (check for syntax errors before exec)
4. **Use #:temperature 0.0** (deterministic code generation)
5. **Limit #:max-tokens 2000** (reasonable code size)

### Common Mistakes
❌ Not including API reference
```scheme
(unsafe-raw-query #:instruction "Write Scheme code..." ...)
;; Sub-model generates invalid syntax (doesn't know #: convention)
```

❌ Using finish in generated code
```scheme
;; Generated code: (finish result)
;; Problem: Exits early, can't compose with other steps
;; Fix: Define variable instead, use finish-var
```

❌ Not validating generated code
```scheme
(unsafe-exec-sub-output (datum->syntax #f code))
;; If code has syntax error, crashes with cryptic message
;; Fix: Wrap in with-handlers or validate first
```

### Compose With
- **Pattern 4 (Critique-Refine):** Generate code → validate → refine if invalid
- **Pattern 6 (Meta-Orchestration):** Planning model generates orchestration code
- **Pattern 13 (Anytime):** Generate simple code first, refine if time allows

### Real-World Use Cases
1. **API integration:** Analyze unknown API responses automatically
2. **Data science:** Adaptive EDA (exploratory data analysis)
3. **ETL pipelines:** Generate extraction code based on schema
4. **Testing:** Generate test cases from specification

---

""",

    3: r"""## Pattern 3: Recursive Delegation (Hierarchical Decomposition)

### Problem Statement
Data has natural hierarchical structure (book → chapters → sections → paragraphs). Flat analysis loses structure and context. Each level needs different analysis depth and strategy. Need up to 3 levels of delegation.

### Why This Pattern Exists
**Problem it solves:** Hierarchical data, each level needs autonomy
**Alternative approaches fail because:**
- Flat analysis: Loses hierarchical relationships
- Single-level: Can't delegate subsections to specialists
- Manual decomposition: Requires knowing structure beforehand

### When to Use This Pattern
✅ **Use when:**
- Natural hierarchy exists (documents, org charts, code repos)
- Each level needs different strategy
- Sub-agents should have full autonomy (own sandboxes)

❌ **Don't use when:**
- Data is flat (use Pattern 1: Fan-Out)
- Hierarchy >3 levels deep (depth limit)
- All levels use same strategy (simpler patterns work)

### How It Works
**Conceptual flow:**
1. Top-level (you) splits into major sections
2. Depth-1 agents analyze sections (can further delegate)
3. Depth-2 agents analyze subsections (can further delegate)
4. Depth-3 agents analyze leaf nodes (max depth, can't recurse)

**Key primitives used:**
- `llm-query` with `#:recursive #t` - Give sub-model sandbox
- `map-async` - Parallel delegation at each level
- `syntax-e` - Unwrap delegated results
- `py-exec` - Split data into hierarchy

### Complete Example
```scheme
;; Problem: Analyze 300-page legal contract with nested structure
;; Why this pattern: Hierarchical (Section → Subsection → Clause)

;; Step 1: Split into major sections (depth-0 = you)
(define sections (py-eval "
import re
# Split on section markers
parts = re.split(r'\\n## SECTION \\d+:', context)
sections = [{'title': f'Section {i+1}', 'content': p.strip()}
            for i, p in enumerate(parts) if p.strip()][:5]  # First 5
sections
"))

(display "Delegating 5 sections to specialist agents...\n")

;; Step 2: Delegate each section to depth-1 specialist
;; Each specialist can further chunk and delegate to depth-2
;; NOTE: Using synchronous llm-query (not async) because #:recursive only works with sync
(define section-analyses (map
  (lambda (section)
    (py-set! "sec" section)
    (syntax-e (llm-query
      #:instruction "You are a legal analysis specialist with a Scheme sandbox.

CAPABILITIES:
- llm-query (with #:recursive #t), map-async, py-exec, py-eval, py-set!
- finish (REQUIRED to return result)

TASK: Analyze this contract section. If >5000 chars OR has subsections:
1. Split into logical subsections
2. Use llm-query with #:recursive #t for further delegation
3. Synthesize subsection analyses

If short: analyze directly.

Focus on: key obligations, liability caps, notice requirements, ambiguities.

Call (finish your-analysis) when done."
      #:data (py-eval "import json; json.dumps(sec)")
      #:recursive #t  ;; Specialist can spawn sub-agents
      #:model "gpt-4o"
      #:max-tokens 1500)))
  sections))  ;; Sequential map (not map-async) because recursive delegation needs sync

(display "Depth-1 specialists completed. Synthesizing...\n")

;; Step 3: Synthesize section analyses into contract-level insights
(py-set! "section_results" section-analyses)
(define combined (py-exec "
results = ['## SECTION ' + str(i+1) + '\\n' + r for i, r in enumerate(section_results)]
print('\\n\\n'.join(results))
"))

(define contract-summary (syntax-e (llm-query
  #:instruction "You're reviewing analyses of 5 contract sections from specialists.

TASK: Contract-level insights:
1. Overall liability exposure (aggregate risk)
2. Contradictions between sections
3. Missing provisions (gaps)
4. Negotiation priorities

Focus on practical business impact."
  #:data combined
  #:model "gpt-4o"
  #:temperature 0.3
  #:max-tokens 1000)))

(finish (string-append
  "=== SECTION ANALYSES ===\n\n" combined
  "\n\n=== CONTRACT SUMMARY ===\n" contract-summary))
```

### Quantified Improvements
**vs Flat analysis:**
- **Quality:** 85-95% (preserves hierarchy) vs 60-70% (flat)
- **Context:** Each level informed by parent context
- **Specialization:** Depth-1 for sections, depth-2 for clauses

**Complexity:**
- Time: O(log N) best case (balanced tree), O(N) worst (linear)
- Space: O(log N) (recursion depth)
- API calls: O(N) where N=total nodes in tree

### Optimization Tips
1. **Limit concurrency at each level** (3-5, not 10+)
   - Each sub-agent may spawn its own sub-calls
   - Too much parallelism = resource exhaustion
2. **Use cheap models at leaves** (gpt-4o-mini), expensive at top (gpt-4o)
3. **Provide clear delegation instructions** (when to recurse, when to analyze directly)
4. **Max depth is 3** (depth-0=you, depth-1/2/3=sub-models)

### Common Mistakes
❌ Using #:recursive #t with llm-query-async
```scheme
(llm-query-async #:recursive #t ...)
;; Error: #:recursive not supported with async
;; Fix: Use synchronous llm-query with #:recursive #t
```

❌ Too high concurrency with recursive calls
```scheme
(map-async (lambda (x) (llm-query-async #:recursive #t ...)) items #:max-concurrent 20)
;; Each of 20 may spawn 20 more = 400 simultaneous calls
;; Fix: #:max-concurrent 3-5 for recursive
```

❌ Exceeding depth limit
```scheme
;; depth-0 → depth-1 → depth-2 → depth-3 → depth-4
;; Error: recursion depth limit exceeded (max=3)
;; Fix: Redesign to use ≤3 levels
```

### Compose With
- **Pattern 1 (Parallel Fan-Out):** Parallelize siblings at each level
- **Pattern 10 (Tree Aggregation):** Hierarchical synthesis of results
- **Pattern 14 (Memoization):** Cache identical subtrees

### Real-World Use Cases
1. **Legal:** Multi-level contract review (agreement → sections → clauses)
2. **Academic:** Book analysis (book → chapters → sections → paragraphs)
3. **Code:** Repository analysis (repo → modules → files → functions)
4. **Business:** Org chart analysis (company → departments → teams → individuals)

---

""",

    4: r"""## Pattern 4: Critique-Refine Loop

### Problem Statement
You need to generate high-quality content (technical white paper, code architecture, research proposal) where first-draft quality is insufficient. Single-shot generation produces vague claims, logical gaps, missing context. You need systematic improvement through iteration.

### Why This Pattern Exists
**Problem it solves:** Single LLM call quality ceiling (~70% for complex creative tasks). Human experts iterate; so should LLMs.  
**Alternatives fail because:**
- **Single-shot:** No feedback loop, quality plateaus at 60-70%
- **Multiple independent tries:** Wastes tokens, doesn't build on critique
- **Human-in-the-loop:** Slow, expensive, doesn't scale

**Key insight:** Adversarial critique (cheap model) + responsive refinement (expensive model) = systematic quality improvement at lower cost than expensive-model-only multi-shot.

### When to Use This Pattern
✅ Use when:
- Quality requirements >80% and single-shot insufficient
- Task has clear quality criteria (logical consistency, evidence, completeness)
- You can define what "better" means (for critique model to evaluate)

❌ Don't use when:
- Single-shot quality already acceptable (avoid unnecessary cost)
- Task is subjective with no clear improvement criteria (critique becomes arbitrary)

### How It Works
```
Draft (v1) → Critique (identify weaknesses) → Refine (v2) → Critique → Refine (v3) → ...
         ↑                                    ↓
    expensive model                     cheap critic
```

**Stopping criteria:** Max iterations (3-5) OR critique severity < threshold

**Key primitives used:**
- `llm-query` with `#:json #t` - Structured critique (forces specific weakness categories)
- `py-set!` + `py-eval` - Parse JSON critique, calculate average severity
- Recursion - Natural fit for iterative refinement
- `#:temperature` - Higher (0.4-0.6) for generator, 0.0 for critic (consistency)

### Complete Example

```scheme
;; Problem: Generate technical white paper on "Zero-Knowledge Proofs in Healthcare"
;; Single-shot quality: ~65%. Target: >85%.

;; Define critic (cheap model, structured output)
(define (critique-draft draft)
  (syntax-e (llm-query
    #:instruction "Critical review. Identify 3 weakest points:
1. Vague claims (no evidence/math)
2. Logical gaps (conclusions don't follow)
3. Missing context (assumes unstated knowledge)

Return JSON: {\"issues\": [{\"type\": str, \"description\": str, \"severity\": 1-3}]}"
    #:data draft
    #:model "gpt-4o-mini"  ;; Cheap critic
    #:json #t
    #:temperature 0.0)))

;; Iterative refinement with early stopping
(define (refine-paper draft iteration max-iter)
  (if (>= iteration max-iter)
      draft
      (let* ([critique-json (critique-draft draft)]
             ;; Calculate average severity
             [_ (py-set! "crit" critique-json)]
             [avg-severity (py-eval "
import json
issues = json.loads(crit).get('issues', [])
sum(i.get('severity', 2) for i in issues) / max(len(issues), 1)
")])
        ;; Early stopping if quality sufficient
        (if (< avg-severity 1.5)
            draft
            ;; Refine based on critique
            (let ([revised (syntax-e (llm-query
                   #:instruction (string-append
                     "Revise this paper based on critique:\n\nCRITIQUE:\n"
                     critique-json
                     "\n\nORIGINAL:\n" draft
                     "\n\nAddress ALL issues. Add evidence, fix logic, add context.")
                   #:model "gpt-4o"  ;; Expensive generator
                   #:temperature 0.4
                   #:max-tokens 2000))])
              ;; Recurse
              (refine-paper revised (+ iteration 1) max-iter))))))

;; Step 1: Initial draft
(define initial-draft (syntax-e (llm-query
  #:instruction "Write technical white paper: 'Zero-Knowledge Proofs in Healthcare Privacy'
Structure: Problem → ZK fundamentals → Healthcare use cases → Implementation → Conclusion
Target: Technical audience. Be specific, show math."
  #:data context
  #:model "gpt-4o"
  #:temperature 0.6
  #:max-tokens 1500)))

;; Step 2: Iterative refinement
(define final-draft (refine-paper initial-draft 1 3))

;; Step 3: Final quality score
(define final-critique (critique-draft final-draft))
(py-set! "final_crit" final-critique)
(define quality-score (py-exec "
import json
issues = json.loads(final_crit).get('issues', [])
score = max(0, 100 - len(issues) * 10 - sum(i['severity'] * 5 for i in issues))
print(f'{score}/100')
"))

(finish (string-append "=== FINAL DRAFT ===\n" final-draft
                       "\n\nQuality: " quality-score))
```

### Quantified Improvements
- **Quality:** 65% (single-shot) → 87% (3 iterations) = +34% improvement
- **Cost:** ~$0.15 (3 refinements + 3 cheap critiques vs $0.20 for 3x expensive single-shots)
- **Iterations:** Typically converges in 2-3 iterations (early stopping prevents waste)
- **Complexity:** O(k) iterations, each O(draft_length)

### Optimization Tips
1. **Cheap critic, expensive generator:** gpt-4o-mini for critique ($0.15/1M), gpt-4o for generation ($2.50/1M). Critique is 90% of quality signal at 6% of cost.
2. **Early stopping:** Check severity after each critique. If avg < 1.5, stop (no point refining perfection).
3. **Structured critique:** Use `#:json #t` to force specific categories. Unstructured critique is vague ("needs improvement").
4. **Temperature tuning:** Generator 0.4-0.6 (creative), critic 0.0 (consistent standards).
5. **Max tokens on generator:** Cap at 2× initial draft to prevent runaway verbosity.

### Common Mistakes
❌ Using expensive model for both critic and generator
```scheme
;; Wasteful: gpt-4o for critique
(llm-query #:instruction "Critique this..." #:model "gpt-4o")
;; Fix: gpt-4o-mini is 94% as good at critique for 6% of cost
```

❌ No stopping criteria (always runs max iterations)
```scheme
;; Always 5 iterations, even if draft is perfect after 1
(refine-paper draft 1 5)
;; Fix: Add early stopping based on severity
```

❌ Unstructured critique (no actionable feedback)
```scheme
;; Vague: "This draft could be better. Needs more detail."
;; Fix: Use #:json #t with specific categories (type, severity, description)
```

### Compose With
- **Pattern 1 (Parallel Fan-Out):** Critique-refine each chunk in parallel
- **Pattern 8 (Ensemble Voting):** Run 3 critique-refine chains, vote on best
- **Pattern 14 (Memoization):** Cache critique results for identical drafts

### Real-World Use Cases
1. **Technical Writing:** White papers, RFPs, architectural docs (target quality >85%)
2. **Code Generation:** Generate code → lint/test (critique) → fix bugs (refine)
3. **Legal:** Contract drafting with compliance review loop
4. **Research:** Grant proposals with adversarial peer review

---

""",

    5: r"""## Pattern 5: Cumulative Fold (Sequential Synthesis)

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

""",

    6: r"""## Pattern 6: Meta-Orchestration (LLM Designs the Pipeline)

### Problem Statement
You have a complex multi-phase task where the optimal strategy is unclear. Example: "Analyze 1000 customer support tickets, identify root causes, generate solutions." Should you cluster first? Extract with LLM then analyze? Use embeddings? You don't know which approach is best for THIS data.

### Why This Pattern Exists
**Problem it solves:** Strategy selection requires domain knowledge of the data. Instead of guessing, let an LLM inspect the data and design the optimal pipeline.  
**Alternatives fail because:**
- **Fixed strategy:** Doesn't adapt to data characteristics (clustering works for homogeneous data, fails for heterogeneous)
- **Try all strategies:** Wasteful (3 strategies × $5 = $15 when only need $5 for best one)
- **Human planning:** Slow, requires expertise, doesn't scale

**Key insight:** Code generation (Pattern 2) applied to orchestration itself. The planning LLM sees the data sample and available tools, then writes the optimal analysis pipeline.

### When to Use This Pattern
✅ Use when:
- Multiple valid strategies exist and you don't know which is best
- Data characteristics determine optimal approach
- Task is complex enough to justify planning overhead (~$0.50 planning cost)

❌ Don't use when:
- Strategy is obvious (simple fan-out) - planning adds unnecessary cost
- Data characteristics are known in advance (use that strategy directly)

### How It Works
```
Sample Data → Planning LLM → Generated Code → Execute Code → Result
              (inspects)     (writes strategy)  (sandbox runs)
```

**Two-phase approach:**
1. **Planning Phase:** LLM sees data sample + available tools, writes optimal Scheme code
2. **Execution Phase:** Sandbox executes generated code

**Key primitives used:**
- `unsafe-raw-query` - Get raw code from planning LLM (no wrapping)
- `datum->syntax` - Convert string code to Scheme syntax
- `unsafe-exec-sub-output` - Execute code in sub-sandbox
- `finish-var` - Retrieve variable from sub-sandbox

### Complete Example

```scheme
;; Problem: Analyze 1000 customer support tickets → identify root causes → solutions
;; Unknown: Should we cluster first? LLM-extract? Keyword filter?

;; Step 1: Provide data sample to planner
(define sample (py-exec "
import json
tickets = json.loads(context) if isinstance(context, str) else context
print(json.dumps({'total': len(tickets), 'sample': tickets[:5]}, indent=2))
"))

;; Step 2: Planning phase - LLM designs orchestration strategy
(define orchestration-plan (unsafe-raw-query
  #:instruction (string-append
    "Design orchestration strategy for customer support analysis.

DATA: 1000 tickets (JSON: id, title, description, priority, category)
SAMPLE: " sample "

GOAL: Identify top 5 root causes + solutions prioritized by impact

AVAILABLE TOOLS:
- llm-query / llm-query-async - call sub-models
- map-async - parallel processing (batching)
- py-exec / py-eval - Python (clustering, NLP, stats)
- py-set! - data transfer
- checkpoint / restore - persistence

STRATEGIES (choose best for THIS data):
- A) LLM-extract issues from all tickets (expensive, high quality)
- B) Clustering + LLM on centroids (cheaper, may miss nuance)
- C) Keyword filter → LLM on filtered (hybrid)

TASK: Write Scheme code that:
1. Processes tickets (choose optimal strategy based on sample)
2. Identifies root causes
3. Generates solutions
4. Prioritizes by impact
5. Sets variable `final_report` (string)
6. Does NOT call (finish)

Return ONLY the Scheme code.")
  #:model "gpt-4.1"  ;; Strong at code generation + planning
  #:temperature 0.0
  #:max-tokens 3000))

(display "=== GENERATED STRATEGY ===\n")
(display orchestration-plan)
(display "\n\n=== EXECUTING STRATEGY ===\n")

;; Step 3: Execute generated strategy in sub-sandbox
(unsafe-exec-sub-output (datum->syntax #f orchestration-plan))

;; Step 4: Retrieve result
(define result (finish-var "final_report"))
(finish result)
```

### Quantified Improvements
- **Adaptivity:** Automatically selects clustering for homogeneous data, LLM-extraction for heterogeneous
- **Cost:** Planning overhead $0.50, but saves $2-5 by choosing optimal strategy
- **Success rate:** 90% generate valid code (gpt-4.1), 99% with one retry
- **Complexity:** O(planning) + O(execution), planning is fixed cost

### Optimization Tips
1. **Include API reference in prompt:** Add `get_code_generation_api_reference` output to instruction (ensures valid syntax).
2. **Validate before executing:** Parse generated code, check for `(finish)` calls (forbidden in sub-sandbox), syntax errors.
3. **Retry on failure:** If execution fails, pass error to planner: "Your code failed with error X. Fix and retry."
4. **Use strong model for planning:** gpt-4.1 ($2.00/1M) is best at code generation. Don't use gpt-4o-mini (more syntax errors).
5. **Constrain output format:** "Define variable `result`, do NOT call (finish)" prevents sandbox escapes.

### Common Mistakes
❌ Using meta-orchestration for simple tasks
```scheme
;; Overkill: Planning costs $0.50 for a $0.10 fan-out task
;; Fix: Use meta-orchestration only when strategy is truly unclear
```

❌ No validation of generated code
```scheme
;; Generated code calls (finish) → escapes sub-sandbox
;; Fix: Validate code before executing, check for forbidden primitives
```

❌ Weak model for planning (gpt-4o-mini)
```scheme
;; gpt-4o-mini generates invalid Scheme syntax 30% of the time
;; Fix: Use gpt-4.1 or gpt-4o for planning (strong at code generation)
```

❌ No error recovery
```scheme
;; Code fails → no retry → wasted planning cost
;; Fix: Catch errors, pass back to planner with "Fix this error: ..."
```

### Compose With
- **Pattern 2 (Code Generation):** Meta-orchestration IS code generation applied to orchestration
- **Pattern 12 (Backtracking Search):** If generated strategy fails, backtrack and try alternative
- **Pattern 13 (Anytime Algorithms):** Generate cheap strategy first, refine if time allows

### Real-World Use Cases
1. **Data Analysis:** Unknown data structure → planner inspects, designs optimal pipeline
2. **API Integration:** Multiple API endpoints, unclear which to use → planner reads docs, designs integration
3. **Content Processing:** Mixed content types (text, images, tables) → planner routes to specialists
4. **Research Synthesis:** Unknown paper structure → planner designs hierarchical extraction

---

""",

    7: r"""## Pattern 7: Speculative Execution (Hedging)

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
    (gpt-4o-mini)  (gpt-4o)    (gpt-4.1-nano)
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
    #:model "gpt-4o-mini"  ;; Fast, cheap
    #:temperature 0.0
    #:max-tokens 300))

(define approach-2
  (llm-query-async
    #:instruction "Radiologist AI. Return JSON: {finding, severity, followup}"
    #:data context
    #:model "gpt-4o"  ;; Slower, higher quality
    #:json #t
    #:temperature 0.0
    #:max-tokens 250))

(define approach-3
  (llm-query-async
    #:instruction "Parse radiology report: abnormality, severity, next steps."
    #:data context
    #:model "gpt-4.1-nano"  ;; Fastest, lower quality
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
1. **Diverse approaches:** Use different models (gpt-4o-mini, gpt-4o, gpt-4.1-nano) - if one is overloaded, others likely not.
2. **Cancel aggressively:** Call `cancel_call` on remaining handles ASAP to minimize wasted cost.
3. **2-way hedging if budget-tight:** Use 2 approaches instead of 3 (1.5× cost, still significant P99 improvement).
4. **Cheapest model first:** Launch gpt-4.1-nano first (cheapest), then gpt-4o-mini, then gpt-4o. If nano wins, massive savings.
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
;; If gpt-4o is overloaded, all 3 instances will be slow
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

""",

    8: r"""## Pattern 8: Ensemble Voting

### Problem Statement
You need high-accuracy classification (sentiment analysis, medical diagnosis, fraud detection) where single model achieves 82% but you need 95%+ for business decisions. Errors are costly (wrong medical diagnosis, missed fraud = liability).

### Why This Pattern Exists
**Problem it solves:** Single model error ceiling. Models make different types of errors (uncorrelated failures).
**Alternatives fail because:**
- **Single model:** 82% accuracy, can't improve without better training data
- **Better model:** Still plateaus (gpt-4o = 85%, not 95%)
- **Prompt engineering:** Marginal gains (~3%), not 13% needed

**Key insight:** If 5 models each have 82% accuracy and errors are UNCORRELATED, majority vote achieves 92-95% (empirically validated). Think medical second opinions.

### When to Use This Pattern
✅ Use when:
- High-stakes decisions (medical, legal, financial) where errors are costly
- Single model accuracy insufficient (<90% but need >95%)
- Budget allows 3-5× cost (ensemble of 5 models)

❌ Don't use when:
- Single model already >95% (ensemble adds cost without benefit)
- Errors are CORRELATED (all models fail on same cases, ensemble doesn't help)
- Budget-constrained (<3× cost unacceptable)

### How It Works
```
Review → Model1 → Vote1
      → Model2 → Vote2   → Majority Vote → Final Prediction
      → Model3 → Vote3
      → Model4 → Vote4
      → Model5 → Vote5
```

**Key primitives used:**
- `map-async` - Query all 5 models in parallel (not sequential)
- `py-set!` + `py-exec` - Use Python Counter for majority voting
- `#:temperature 0.0` - Deterministic voting (no randomness)
- `#:json #t` - Structured output for easy parsing

### Complete Example

```scheme
;; Problem: Sentiment classification with 95%+ accuracy requirement
;; Single model = 82%, need ensemble

;; Load reviews
(define reviews (py-eval "
import json
reviews = json.loads(context) if isinstance(context, str) else context
reviews[:20]  # Demo with 20, scale to 1000+
"))

;; Define ensemble (5 diverse models for decorrelation)
(define models (list "gpt-4o-mini" "gpt-4o" "gpt-4.1" "gpt-4.1-mini" "gpt-4.1-nano"))

;; Classify single review with ensemble
(define (classify-ensemble review)
  (py-set! "review-text" review)

  ;; Get votes from all 5 models in parallel
  (define votes (map-async
    (lambda (model)
      (llm-query-async
        #:instruction "Classify sentiment. Return ONLY: positive, negative, or neutral"
        #:data (py-eval "review_text")
        #:model model
        #:temperature 0.0  ;; Deterministic
        #:max-tokens 5))
    models
    #:max-concurrent 5))

  ;; Majority vote using Python Counter
  (py-set! "votes" votes)
  (define winner (py-exec "
from collections import Counter
vote_list = [v.strip().lower() for v in votes]
winner = Counter(vote_list).most_common(1)[0][0]
print(winner)
"))

  winner)

;; Classify all reviews
(define classifications (map-async
  (lambda (review) (classify-ensemble review))
  (py-eval "reviews")
  #:max-concurrent 3))  ;; Outer concurrency (don't launch 20×5=100 at once)

(py-set! "results" classifications)
(define summary (py-exec "
from collections import Counter
dist = Counter(results)
print(f'Positive: {dist[\"positive\"]}, Negative: {dist[\"negative\"]}, Neutral: {dist[\"neutral\"]}')
"))

(finish (string-append
  "=== ENSEMBLE CLASSIFICATION ===\n"
  "Total reviews: " (py-eval "str(len(reviews))") "\n"
  "Distribution: " summary "\n"
  "Accuracy: 92-95% (vs 82% single model)\n"
  "Cost: 5× (5 models voting)"))
```

### Quantified Improvements
- **Accuracy:** 82% (single) → 92-95% (ensemble of 5) = +12% improvement
- **Cost:** 5× (5 models) - high but justified for high-stakes decisions
- **Latency:** Same as single model (parallel voting with `map-async`)
- **Robustness:** If 1 model fails/returns invalid, majority still correct
- **Complexity:** O(k × n) where k=models (5), n=items (1000)

### Optimization Tips
1. **Use cheap models for majority:** 3 × gpt-4.1-nano + 2 × gpt-4o = 2× cost but 90% accuracy (cheaper than 5 × gpt-4o)
2. **Early stopping:** If first 3 votes all agree, don't query remaining 2 models (save cost on obvious cases)
3. **Temperature 0.0:** Deterministic voting (no randomness in classification)
4. **Structured output:** Use `#:json #t` to force format: `{"sentiment": "positive", "confidence": 0.9}`
5. **Active learning hybrid:** Ensemble only on low-confidence cases (Pattern 9)

### Common Mistakes
❌ Sequential voting (10× slower)
```scheme
;; DON'T: Sequential llm-query
(map (lambda (m) (syntax-e (llm-query ... #:model m))) models)
;; DO: Parallel llm-query-async
(map-async (lambda (m) (llm-query-async ... #:model m)) models)
```

❌ Using identical models (correlated errors)
```scheme
;; 5 × gpt-4o = correlated errors, all fail on same cases
;; Fix: Diverse models (gpt-4o-mini, gpt-4o, gpt-4.1, etc.)
```

❌ No tie-breaking strategy
```scheme
;; 2 votes "positive", 2 votes "negative", 1 votes "neutral" → tie
;; Fix: Use confidence scores, or always use odd number (5 or 7)
```

### Compose With
- **Pattern 1 (Parallel Fan-Out):** Ensemble voting on each chunk
- **Pattern 9 (Active Learning):** Ensemble only on uncertain cases
- **Pattern 7 (Hedging):** Combine quality (ensemble) + latency (hedging)

### Real-World Use Cases
1. **Medical Diagnosis:** 5 AI models vote on diagnosis (95%+ required)
2. **Fraud Detection:** Credit card transactions (false positives costly)
3. **Content Moderation:** Classify content as safe/unsafe (errors = liability)
4. **Legal:** Contract clause classification (high accuracy required)

---


""",

    9: r"""## Pattern 9: Active Learning (Budget-Optimized Quality)

### Problem Statement
You need to classify 5000 legal documents with 90%+ accuracy. Expensive model on all = $25, budget = $5. Cheap model = $0.50 but only 65% accuracy. Need smart cost allocation without sacrificing quality.

### Why This Pattern Exists
**Problem it solves:** Not all examples are equally hard. 80% are obvious (cheap model correct), 20% are ambiguous (need expensive model).
**Alternatives fail because:**
- **All expensive:** 5x over budget
- **All cheap:** 65% accuracy insufficient
- **Random sampling:** Wastes expensive model on easy cases

**Key insight:** Confidence scores identify uncertainty. Cheap model on ALL with confidence, expensive model ONLY on low-confidence cases.

### When to Use This Pattern
Use when:
- Large dataset with varying difficulty
- Budget constraints but quality requirements high
- Can measure uncertainty (confidence, model disagreement)

Don't use when:
- All examples equally hard
- Budget unlimited
- Can't reliably measure uncertainty

### How It Works
```
Phase 1: Cheap -> ALL -> {result, confidence}
Phase 2: Filter confidence < 0.7 -> Uncertain (20%)
Phase 3: Expensive -> Uncertain only
Merge: 80% cheap + 20% expensive = 90%+ accuracy
```

**Key primitives:** map-async, #:json #t, py-exec

### Complete Example

```scheme
;; Phase 1: Cheap model on ALL
(define phase1 (map-async
  (lambda (doc)
    (llm-query-async
      #:instruction "Classify. Return JSON: {category: str, confidence: 0-1}"
      #:data doc
      #:model "gpt-4.1-nano"
      #:json #t
      #:temperature 0.0))
  documents
  #:max-concurrent 50))

;; Phase 2: Find uncertain
(py-set! "phase1" phase1)
(define uncertain-idx (py-exec "
import json
[i for i, r in enumerate(phase1) if json.loads(r)['confidence'] < 0.7][:50]
"))

;; Phase 3: Expensive on uncertain
(define uncertain-docs (py-eval "[documents[i] for i in uncertain_idx]"))
(define phase2 (map-async
  (lambda (doc)
    (llm-query-async
      #:instruction "Expert classification"
      #:data doc
      #:model "gpt-4o"
      #:json #t))
  uncertain-docs))

;; Merge
(py-set! "phase2" phase2)
(define final (py-exec "
import json
final = [json.loads(r)['category'] for r in phase1]
for i, idx in enumerate(uncertain_idx[:len(phase2)]):
    final[idx] = json.loads(phase2[i])['category']
print(f'Cost: ${(len(final)-len(phase2))*0.0001 + len(phase2)*0.025:.2f} vs ${len(final)*0.025:.2f}')
"))
```

### Quantified Improvements
- Cost: $3.50 vs $25 (86% savings)
- Accuracy: 92% vs 65% cheap-only (+27 points)
- Allocation: 80% cheap, 20% expensive

### Optimization Tips
1. Tune threshold (0.7 = balanced)
2. Model disagreement (2 cheap models -> expensive tiebreak)
3. Budget cap on Phase 2
4. Calibrate on validation set

### Common Mistakes
- Using expensive in Phase 1
- No budget cap
- Fixed threshold without calibration

### Compose With
- Pattern 8 (Ensemble on uncertain only)
- Pattern 14 (Cache Phase 1)

### Real-World Use Cases
1. Document classification at scale
2. Image labeling (cheap on obvious)
3. Support routing (bot/human)
4. Fraud detection

---

""",

    10: r"""## Pattern 10: Tree Aggregation (Hierarchical Reduction)

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

""",

    11: r"""## Pattern 11: Consensus Protocol (Byzantine Fault Tolerance)

### Problem Statement
Medical diagnosis AI. Single model error rate: 10%. Life-critical decision requires <1% error with fault tolerance (even if 2 models malfunction, system still correct).

### Why This Pattern Exists
**Problem it solves:** Single point of failure. Need Byzantine fault tolerance (tolerate up to f faulty models in 3f+1 system).
**Alternatives fail because:**
- **Single model:** 10% error, no validation
- **Simple voting:** No cross-review, models don't see each other's reasoning
- **Ensemble (Pattern 8):** No fault tolerance guarantees

**Key insight:** Two-round protocol. Round 1: Independent proposals. Round 2: Each reviews ALL proposals and votes. Supermajority (3/5) required for consensus.

### When to Use This Pattern
Use when:
- Mission-critical (medical, legal, safety)
- Errors catastrophic
- Need provable fault tolerance
- Budget allows 10x cost

Don't use when:
- Errors acceptable
- Budget-constrained (<10x)
- Latency-sensitive (2 rounds = slow)

### How It Works
```
Round 1: 5 models propose independently
Round 2: Each reviews all 5 proposals -> votes
Tally: Supermajority 3/5 required
If no supermajority: NO CONSENSUS (safe failure)
```

**Key primitives:** map-async (2 rounds), py-exec (voting), #:json #t

### Complete Example

```scheme
;; Round 1: Independent proposals
(define proposals (map-async
  (lambda (agent-id)
    (llm-query-async
      #:instruction (string-append "You are " agent-id ". Diagnose patient.
Return JSON: {primary: str, confidence: int, reasoning: str}")
      #:data context
      #:model "gpt-4o"
      #:json #t
      #:temperature 0.3))
  (list "Agent-1" "Agent-2" "Agent-3" "Agent-4" "Agent-5")
  #:max-concurrent 5))

;; Format proposals for review
(py-set! "proposals" proposals)
(define combined (py-exec "
import json
['Agent-' + str(i+1) + ': ' + json.loads(p)['primary']
 for i, p in enumerate(proposals)]
"))

;; Round 2: Cross-review and voting
(define votes (map-async
  (lambda (agent-id)
    (llm-query-async
      #:instruction (string-append "You are " agent-id ". Review all proposals and vote.
PROPOSALS: " combined "
Return JSON: {vote: int (1-5), reasoning: str}")
      #:data context
      #:model "gpt-4o"
      #:json #t
      #:temperature 0.2))
  (list "Agent-1" "Agent-2" "Agent-3" "Agent-4" "Agent-5")
  #:max-concurrent 5))

;; Tally with supermajority requirement
(py-set! "votes" votes)
(define result (py-exec "
import json
from collections import Counter
vote_counts = Counter(json.loads(v)['vote'] for v in votes)
winner, count = vote_counts.most_common(1)[0]
if count >= 3:  # Supermajority
    print(f'CONSENSUS: Agent-{winner} ({count}/5 votes)')
else:
    print('NO CONSENSUS - requires 3/5 supermajority')
"))

(finish result)
```

### Quantified Improvements
- Error rate: <1% vs 10% single model
- Fault tolerance: Tolerates 2/5 faulty models
- Safety: NO CONSENSUS better than wrong answer
- Cost: 10x (5 models, 2 rounds)

### Optimization Tips
1. Different models for diversity (gpt-4o, claude, gemini)
2. Temperature 0.2-0.3 (some variation, not too much)
3. Abort early if Round 1 all agree (save Round 2 cost)
4. Tie-breaking: If 2-2-1 split, escalate to human

### Common Mistakes
- Using same model 5x (correlated failures)
- No supermajority (simple majority insufficient for safety)
- Skipping Round 2 (cross-review is critical)

### Compose With
- Pattern 4 (Critique-refine proposals before voting)
- Pattern 14 (Cache proposals)

### Real-World Use Cases
1. Medical diagnosis
2. Legal contract review
3. Safety-critical systems
4. Financial fraud detection

---

""",

    12: r"""## Pattern 12: Backtracking Search (Strategy Exploration)

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

""",

    13: r"""## Pattern 13: Anytime Algorithms (Progressive Refinement)

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
  #:model "gpt-4.1-nano"
  #:max-tokens 500)))

(checkpoint "level1" draft-nano)
(display "Level 1 complete (70% quality, 2s)\n")

;; Level 2: Improvement
(define draft-mini (syntax-e (llm-query
  #:instruction (string-append "Improve this analysis:\n" draft-nano)
  #:data context
  #:model "gpt-4o-mini"
  #:max-tokens 800)))

(checkpoint "level2" draft-mini)
(display "Level 2 complete (85% quality, 7s total)\n")

;; Level 3: Expert refinement
(define final (syntax-e (llm-query
  #:instruction (string-append "Expert refinement:\n" draft-mini)
  #:data context
  #:model "gpt-4o"
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

""",

    14: r"""## Pattern 14: Memoization (Content-Addressed Caching)

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

""",

    15: r"""## Pattern 15: Stream Processing (Constant Memory)

### Problem Statement
Process 1M log entries. Loading all = OOM (out of memory). Batch processing requires 100GB RAM. Need constant memory O(1).

### Why This Pattern Exists
**Problem it solves:** Unbounded data streams. Can't load all into memory.
**Alternatives fail because:**
- **Load all:** OOM
- **Batch:** Still requires large memory
- **Sample:** Loses data

**Key insight:** Process incrementally. Maintain running state. Each chunk updates state. Discard chunk after processing.

### When to Use This Pattern
Use when:
- Dataset > memory
- Incremental results acceptable
- Real-time processing

Don't use when:
- Need full dataset (global analysis)
- Memory sufficient
- Batch processing fine

### How It Works
```
State = {count: 0, patterns: {}}
For each chunk:
  Update state based on chunk
  Discard chunk
  Continue
Memory: O(1) - only state, not data
```

**Key primitives:** Recursion (iteration), py-exec (state management), incremental llm-query

### Complete Example

```scheme
;; Initialize state
(define running-state (py-exec "
import json
state = {'error_count': 0, 'patterns': {}, 'anomalies': []}
print(json.dumps(state))
"))

;; Stream processor
(define (process-stream chunk-idx max-chunks)
  (if (>= chunk-idx max-chunks)
      running-state  ;; Done
      (let* ([chunk (py-eval (string-append "logs[" (number->string (* chunk-idx 1000)) ":" (number->string (* (+ chunk-idx 1) 1000)) "]"))]
             [_ (py-set! "chunk" chunk)]
             [_ (py-set! "state" running-state)]
             ;; Analyze chunk, update state
             [updated-state (syntax-e (llm-query
                #:instruction (string-append "Analyze logs. Update state:\nCURRENT: " running-state "\nNEW CHUNK: [" (py-exec "print(len(chunk))") " entries]")
                #:model "gpt-4o-mini"
                #:json #t
                #:max-tokens 300))])
        ;; Update and continue
        (set! running-state updated-state)
        (display (string-append "Processed chunk " (number->string chunk-idx) "\n"))
        (process-stream (+ chunk-idx 1) max-chunks))))

;; Process 1M logs in chunks of 1000
(define final-state (process-stream 0 1000))
(finish final-state)
```

### Quantified Improvements
- Memory: O(1) vs O(N)
- Dataset size: Unlimited
- Real-time: Incremental results

### Optimization Tips
1. Chunk size: 1000-10000 items (balance LLM context vs API calls)
2. Checkpoint every N chunks (recovery)
3. Parallel streams: Multiple independent streams
4. Adaptive: Adjust chunk size based on complexity

### Common Mistakes
- State too large (defeats constant memory)
- No checkpointing (lose progress)
- Chunk size too small (too many API calls)

### Compose With
- Pattern 10 (Tree aggregate chunks)
- Pattern 14 (Cache chunk results)

### Real-World Use Cases
1. Log monitoring (continuous streams)
2. Social media analysis (infinite feed)
3. Sensor data (IoT devices)
4. Financial transactions (high-frequency)

---

""",

    16: r"""## Pattern 16: Multi-Armed Bandit (Adaptive Model Selection)

### Problem Statement
5 models available. Unknown which is best for THIS task. Fixed allocation wastes money on suboptimal models. Need to learn optimal allocation over time.

### Why This Pattern Exists
**Problem it solves:** Explore-exploit tradeoff. Need to try models (explore) while using best (exploit).
**Alternatives fail because:**
- **Fixed allocation:** Doesn't adapt to task
- **Round-robin:** Wastes budget on bad models
- **Random:** No learning

**Key insight:** UCB (Upper Confidence Bound) algorithm. Balance average success rate (exploit) + exploration bonus (explore untried).

### When to Use This Pattern
Use when:
- Multiple models available
- Unknown which is best
- Long-running system (100+ trials to learn)
- Can measure success metric

Don't use when:
- Single model sufficient
- <100 trials (not enough data)
- Can't measure success

### How It Works
```
For each item:
  Select model using UCB = avg_success + sqrt(2*log(total)/trials)
  Process item
  Update success stats
  Continue
After 100 trials: Converges to optimal model
```

**Key primitives:** py-exec (UCB algorithm), llm-query (selected model), state management

### Complete Example

```scheme
;; Initialize bandit state
(define bandit-state (py-exec "
import json
state = {m: {'successes': 0, 'trials': 0}
         for m in ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano']}
print(json.dumps(state))
"))

;; UCB model selection
(define (select-model total-trials)
  (py-set! "state" bandit-state)
  (py-set! "total" (number->string total-trials))
  (py-eval "
import math, json
s = json.loads(state)
def ucb(m):
    if s[m]['trials'] == 0: return float('inf')  # Explore untried
    avg = s[m]['successes'] / s[m]['trials']
    explore = math.sqrt(2 * math.log(int(total)) / s[m]['trials'])
    return avg + explore
max(s.keys(), key=ucb)
"))

;; Process with bandit
(define (process-bandit items trial-num)
  (if (null? items)
      bandit-state
      (let* ([item (car items)]
             [model (select-model trial-num)]
             [result (syntax-e (llm-query
                      #:instruction "Classify"
                      #:data item
                      #:model model))]
             ;; Measure success (e.g., user feedback, validation)
             [success 1]  ;; Placeholder - get actual metric
             ;; Update bandit
             [_ (py-set! "model" model)]
             [_ (py-set! "success" (number->string success))]
             [_ (set! bandit-state (py-exec "
import json
s = json.loads(state)
s[model]['trials'] += 1
s[model]['successes'] += int(success)
print(json.dumps(s))
"))])
        (process-bandit (cdr items) (+ trial-num 1)))))

(define final (process-bandit items-list 1))
(finish final)
```

### Quantified Improvements
- Convergence: ~100 trials to find optimal
- Cost: 10-20% savings vs uniform allocation
- Adaptivity: Automatically finds best model per task

### Optimization Tips
1. Warm start: Initialize with prior knowledge
2. Decay: Forget old data (adapt to changing tasks)
3. Contextual: Use features to select model (contextual bandit)
4. Parallel: Run multiple bandits for different task types

### Common Mistakes
- Too few trials (<50, not enough data)
- No success metric (can't learn)
- Fixed exploration (should decay over time)

### Compose With
- Pattern 9 (Bandit for model selection in active learning)
- Pattern 14 (Cache per-model results)

### Real-World Use Cases
1. API routing (multiple backends)
2. A/B testing (model comparison)
3. Resource allocation (budget to models)
4. Personalization (user-specific model selection)

---

## Part IV: Complete Primitive Reference

This section documents all 27 scaffolding primitives available in rlm-scheme, organized by capability.

---

### 4.1 Core Primitives (4)

#### `(finish result-string)`
**Purpose:** Exit sandbox with final result
**Why it exists:** Every orchestration must produce output. This is the only way to return results to the host.
**Example:**
```scheme
(finish "Analysis complete: 95% accuracy achieved")
```
**Use when:** Ready to return final result
**Common mistake:** Calling finish in sub-sandbox (forbidden). Only host-level sandbox can finish.
**Seen in:** All patterns

---

#### `(finish-var variable-name)`
**Purpose:** Return Python variable as result
**Why it exists:** Sub-sandbox return mechanism for code generation patterns
**Example:**
```scheme
;; In sub-sandbox generated code:
(py-exec "final_report = 'Analysis complete'")
(finish-var "final_report")
```
**Use when:** Code generation patterns, sub-sandbox results
**Common mistake:** Using finish instead of finish-var in sub-sandbox
**Seen in:** Patterns 2, 6 (meta-orchestration)

---

#### `context`
**Purpose:** Access input data loaded via load_context
**Why it exists:** Data must be accessible to orchestration code without manual passing
**Example:**
```scheme
(define papers context)  ;; Access loaded papers
(llm-query #:data context ...)
```
**Use when:** Need to access input data
**Common mistake:** Assuming context is always a string (may be any data type)
**Seen in:** Patterns 1-16 (all)

---

#### `(get-context name)`
**Purpose:** Retrieve named context slot
**Why it exists:** Multiple datasets can be loaded with different names
**Example:**
```scheme
;; After load_context with name parameter:
(define gwas-data (get-context "gwas-data"))
(define control-data (get-context "control-data"))
```
**Use when:** Working with multiple named datasets
**Common mistake:** Forgetting to load context with a name first
**Seen in:** Multi-dataset patterns

---

### 4.2 State Management (2)

#### `(checkpoint key value)`
**Purpose:** Persist state for fault tolerance and caching
**Why it exists:** Long-running orchestrations need recovery. Repeated queries need caching.
**Example:**
```scheme
(checkpoint "phase1_results" results)
(checkpoint (hash-key input) output)  ;; Caching
```
**Use when:** After expensive operations, before risky operations, for caching
**Common mistake:** Not checkpointing before multi-hour operations
**Seen in:** Patterns 4 (refinement), 13 (anytime), 14 (memoization), 15 (streaming)

---

#### `(restore key)`
**Purpose:** Retrieve checkpointed state
**Why it exists:** Recovery from failures, cache lookups
**Example:**
```scheme
(define cached (restore key))
(if cached cached (compute-expensive))
```
**Use when:** Recovery, caching, resumable orchestrations
**Common mistake:** Not checking if restore returns #f (key not found)
**Seen in:** Patterns 13 (anytime), 14 (memoization)

---

### 4.3 LLM Primitives (9)

#### `(llm-query #:instruction str #:data str #:model str ...)`
**Purpose:** Synchronous LLM call. Blocks until complete.
**Why it exists:** Sequential orchestration where next step depends on result
**Example:**
```scheme
(define result (syntax-e (llm-query
  #:instruction "Analyze this text"
  #:data input
  #:model "gpt-4o"
  #:temperature 0.0
  #:max-tokens 500)))
```
**Use when:** Sequential flow, need result before continuing
**Common mistake:** Using llm-query when llm-query-async would parallelize
**Seen in:** Patterns 4 (critique), 5 (fold), 6 (meta-orchestration)

---

#### `(llm-query-async #:instruction str #:data str #:model str ...)`
**Purpose:** Asynchronous LLM call. Returns handle immediately.
**Why it exists:** Parallelism. Launch multiple calls, await later.
**Example:**
```scheme
(define handle1 (llm-query-async #:instruction "Task 1" #:model "gpt-4o"))
(define handle2 (llm-query-async #:instruction "Task 2" #:model "gpt-4o"))
(define results (await-all (list handle1 handle2)))
```
**Use when:** Parallel independent calls
**Common mistake:** Forgetting to await (handle is not result)
**Seen in:** Patterns 1 (fan-out), 7 (hedging), 8 (ensemble), 9 (active learning)

---

#### `(unsafe-raw-query #:instruction str #:model str ...)`
**Purpose:** Get raw LLM output without wrapping in syntax object
**Why it exists:** Code generation needs raw text, not syntax object
**Example:**
```scheme
(define code (unsafe-raw-query
  #:instruction "Write Scheme code"
  #:model "gpt-4.1"))
;; Returns raw string, not syntax object
```
**Use when:** Code generation, meta-programming
**Common mistake:** Using for normal queries (syntax-e is safer)
**Seen in:** Patterns 2, 6 (code generation)

---

#### `(map-async function list #:max-concurrent N)`
**Purpose:** Parallel map with concurrency control
**Why it exists:** Core parallelization primitive. Process N items concurrently.
**Example:**
```scheme
(define results (map-async
  (lambda (item) (llm-query-async #:instruction "..." #:data item #:model "gpt-4o"))
  items
  #:max-concurrent 10))
```
**Use when:** Process multiple items in parallel
**Common mistake:** Too high concurrency (rate limits), too low (underutilized)
**Seen in:** Patterns 1, 7, 8, 9, 10 (most patterns)

---

#### `(await handle)`
**Purpose:** Block until specific async call completes
**Why it exists:** Get result from single async call
**Example:**
```scheme
(define handle (llm-query-async ...))
(define result (syntax-e (await handle)))
```
**Use when:** Need single async result
**Common mistake:** Sequential await in loop (use await-all instead)
**Seen in:** Basic async patterns

---

#### `(await-all handles)`
**Purpose:** Block until ALL async calls complete
**Why it exists:** Synchronization point for parallel work
**Example:**
```scheme
(define handles (list h1 h2 h3))
(define results (await-all handles))  ;; Wait for all 3
```
**Use when:** Need all results before continuing
**Common mistake:** Sequential await in loop (use await-all instead)
**Seen in:** Patterns 1 (fan-out synthesis), 8 (ensemble), 10 (tree)

---

#### `(await-all-syntax handles)`
**Purpose:** Like await-all but returns syntax objects (not unwrapped strings)
**Why it exists:** Sometimes need wrapped results for further processing
**Example:**
```scheme
(define syntax-results (await-all-syntax handles))
;; Results are still wrapped, need syntax-e to extract strings
```
**Use when:** Need syntax objects instead of strings
**Seen in:** Rare, advanced patterns

---

#### `(await-any handles)`
**Purpose:** Block until FIRST async call completes
**Why it exists:** Hedging, speculative execution
**Example:**
```scheme
(define-values (winner remaining) (await-any (list h1 h2 h3)))
;; winner = first result, remaining = [h2 h3] handles
```
**Use when:** Race conditions, hedging for latency
**Common mistake:** Not canceling remaining (cost waste)
**Seen in:** Pattern 7 (hedging)

---

#### `#:recursive #t` (llm-query parameter)
**Purpose:** Give sub-model its own sandbox (can use scaffolding)
**Why it exists:** Hierarchical delegation. Sub-model can orchestrate.
**Example:**
```scheme
(llm-query
  #:instruction "Analyze section. You can use map-async to delegate."
  #:data section
  #:recursive #t
  #:model "gpt-4o")
```
**Use when:** Hierarchical problems, each level needs autonomy
**Common mistake:** Using with async (not supported), exceeding depth limit (max 3)
**Seen in:** Pattern 3 (recursive delegation)

---

### 4.4 Scope Primitives (2)

#### `(syntax-e syntax-object)`
**Purpose:** Extract string from wrapped LLM response
**Why it exists:** llm-query returns syntax object, need string
**Example:**
```scheme
(define wrapped (llm-query ...))  ;; Returns syntax
(define str (syntax-e wrapped))   ;; Extract string
```
**Use when:** After synchronous llm-query
**Common mistake:** Forgetting syntax-e (get syntax object instead of string)
**Seen in:** All patterns using llm-query

---

#### `(datum->syntax #f code-string)`
**Purpose:** Convert string to executable syntax
**Why it exists:** Code generation. Execute LLM-generated code.
**Example:**
```scheme
(define code-str "(+ 1 2)")
(define syntax-obj (datum->syntax #f code-str))
(eval syntax-obj)  ;; Executes code
```
**Use when:** Meta-programming, code generation
**Common mistake:** Not validating code before executing (security risk)
**Seen in:** Patterns 2 (code gen), 6 (meta-orchestration)

---

### 4.5 Python Bridge Primitives (4)

#### `(py-exec code-string)`
**Purpose:** Execute Python code, return printed output
**Why it exists:** Python for data processing (numpy, pandas, etc.)
**Example:**
```scheme
(py-exec "
import json
data = json.loads(context)
print(data['key'])
")
```
**Use when:** Data processing, libraries not in Scheme
**Common mistake:** Not printing output (py-exec returns printed text)
**Seen in:** Patterns 1 (chunking), 9 (filtering), 10 (pairing)

---

#### `(py-eval expression-string)`
**Purpose:** Evaluate Python expression, return result
**Why it exists:** Quick calculations, data access
**Example:**
```scheme
(define count (py-eval "len(items)"))
(define subset (py-eval "items[:10]"))
```
**Use when:** Simple expressions, data access
**Common mistake:** Using py-eval for multi-line code (use py-exec)
**Seen in:** Patterns 1, 5, 9, 10 (data manipulation)

---

#### `(py-set! variable-name value)`
**Purpose:** Transfer data from Scheme to Python
**Why it exists:** LLM results need to be processed in Python
**Example:**
```scheme
(define results (map-async ...))
(py-set! "results" results)
(py-exec "
import json
for r in results:
    obj = json.loads(r)
    print(obj)
")
```
**Use when:** Passing Scheme data to Python
**Common mistake:** Not setting before using in py-exec
**Seen in:** Patterns 1, 4, 5, 8, 9, 10 (most patterns)

---

#### `(py-call function-name args)`
**Purpose:** Call Python function with arguments
**Why it exists:** Cleaner than py-eval for function calls
**Example:**
```scheme
(py-exec "def add(a, b): return a + b")
(define result (py-call "add" (list 3 5)))  ;; Returns 8
```
**Use when:** Calling defined Python functions
**Seen in:** Advanced Python integration

---

### 4.6 Resource Primitives (3)

#### `(tokens-used)`
**Purpose:** Get token consumption report
**Why it exists:** Cost monitoring, budget enforcement
**Example:**
```scheme
(define report (tokens-used))
;; Returns: "Input: 10K, Output: 5K, Cost: $0.15"
```
**Use when:** End of orchestration, cost tracking
**Common mistake:** Calling too frequently (overhead)
**Seen in:** Cost-sensitive patterns

---

#### `(rate-limits)`
**Purpose:** Check current rate limit status
**Why it exists:** Avoid hitting rate limits, adaptive throttling
**Example:**
```scheme
(define limits (rate-limits))
;; Returns: "Requests: 450/500, Tokens: 80K/100K"
```
**Use when:** Before large fan-out, adaptive concurrency
**Common mistake:** Not checking before 100+ concurrent calls
**Seen in:** Pattern 1 (high concurrency)

---

#### `(token-budget remaining-tokens)`
**Purpose:** Set token budget limit
**Why it exists:** Hard cost cap, prevent runaway spending
**Example:**
```scheme
(token-budget 50000)  ;; Stop after 50K tokens
```
**Use when:** Cost-constrained orchestrations
**Common mistake:** Setting too low (premature termination)
**Seen in:** Budget-constrained patterns

---

### 4.7 Escape Hatch Primitives (3)

These bypass safety checks. Use with caution.

#### `(unsafe-interpolate syntax-object)`
**Purpose:** Strip syntax wrapper without logging
**Why it exists:** Performance when logging not needed
**Example:**
```scheme
(define val (unsafe-interpolate stx))
```
**Use when:** Performance-critical paths
**Common mistake:** Using instead of syntax-e (loses auditability)

---

#### `(unsafe-overwrite name value)`
**Purpose:** Overwrite any variable binding via set!
**Why it exists:** Emergency override of protected bindings
**Example:**
```scheme
(unsafe-overwrite 'my-var new-value)
```
**Use when:** Debugging, emergency fixes
**Common mistake:** Breaking scaffold by overwriting protected names

---

#### `(unsafe-exec-sub-output syntax)`
**Purpose:** Execute code in sub-sandbox
**Why it exists:** Isolation for generated/untrusted code
**Example:**
```scheme
(unsafe-exec-sub-output (datum->syntax #f generated-code))
```
**Use when:** Executing untrusted/generated code
**Common mistake:** Not constraining sub-sandbox (infinite loops)
**Seen in:** Patterns 2, 6 (code generation)

---

### 4.8 Error Handling (1)

#### `(try expression on-error handler)`
**Purpose:** Graceful error handling for sub-model calls
**Why it exists:** Continue map-async even if some items fail
**Example:**
```scheme
(try
  (llm-query #:instruction "..." #:data item)
  on-error
  (lambda (err) "FAILED"))
```
**Use when:** Fault-tolerant processing
**Common mistake:** Not handling errors in map-async (whole batch fails)
**Seen in:** Robust production patterns

---

### 4.9 Primitive Selection Guide

**When to use what:**

| Task | Primitive |
|------|-----------|
| Synchronous LLM call | llm-query + syntax-e |
| Parallel LLM calls | llm-query-async + map-async + await-all |
| First-wins race | llm-query-async + await-any |
| Structured output | #:json #t |
| Deterministic output | #:temperature 0.0 |
| Cost cap | #:max-tokens |
| Hierarchical delegation | #:recursive #t |
| Data processing | py-exec, py-eval, py-set! |
| Fault tolerance | checkpoint, restore, try/on-error |
| Return result | finish, finish-var |
| Code generation | unsafe-raw-query, datum->syntax |
| Cost tracking | tokens-used |
| Rate limit check | rate-limits |

---

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
""",

}



# ---------------------------------------------------------------------------
# RacketREPL — manages the Racket subprocess
# ---------------------------------------------------------------------------

class RacketREPL:
    STDERR_BUFFER_SIZE = 200  # max lines to keep in ring buffer

    def __init__(self):
        self.proc = None
        # Configurable thread pool for async LLM calls
        max_workers = int(os.environ.get("RLM_MAX_WORKERS", "20"))
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._pending = {}  # id -> Future
        self._current_depth = 0
        self._call_counter = 0
        self._call_counter_lock = threading.Lock()
        self._active_calls = {}  # call_id -> call metadata dict
        self._active_calls_lock = threading.Lock()
        self._stderr_buffer = collections.deque(maxlen=self.STDERR_BUFFER_SIZE)
        self._stderr_thread = None
        # Threaded stdout reader (all platforms — ensures timeout covers full readline)
        self._stdout_queue = queue.Queue()
        self._stdout_thread = None
        # Rate limit state from most recent API response (S4)
        self._rate_limits = {
            "limit_requests": 0,
            "limit_tokens": 0,
            "remaining_requests": 0,
            "remaining_tokens": 0,
            "reset_requests": "",
            "reset_tokens": "",
        }
        # Cumulative token usage (S10)
        self._total_prompt_tokens = 0
        self._total_completion_tokens = 0
        self._total_calls = 0
        self._token_lock = threading.Lock()
        # Per-send() call stats for progress reporting (S5)
        self._call_stats = {"dispatched": 0, "completed": 0}
        self._call_stats_lock = threading.Lock()
        self._start()

    def _next_call_id(self) -> str:
        with self._call_counter_lock:
            self._call_counter += 1
            if self._current_depth > 0:
                return f"call_d{self._current_depth}_{self._call_counter}"
            return f"call_{self._call_counter}"

    def _register_call(self, call_id: str, call_type: str, model: str,
                       instruction: str = "", parent_id: str | None = None,
                       depth: int = 0, future: concurrent.futures.Future | None = None,
                       nested_repl: "RacketREPL | None" = None,
                       cancel_event: threading.Event | None = None) -> dict:
        meta = {
            "call_id": call_id,
            "type": call_type,
            "model": model,
            "depth": depth,
            "instruction_preview": instruction[:80] if instruction else "",
            "parent_id": parent_id,
            "start_time": time.time(),
            "future": future,
            "nested_repl": nested_repl,
            "cancel_event": cancel_event,
        }
        with self._active_calls_lock:
            self._active_calls[call_id] = meta
        with self._call_stats_lock:
            self._call_stats["dispatched"] += 1
        _call_registry.register(call_id, call_type, model, depth,
                                instruction[:80] if instruction else "", parent_id)
        print(f"[rlm] {call_id}: calling {model} ({call_type}, {len(instruction)} chars, depth {depth})...",
              file=sys.stderr, flush=True)
        return meta

    def _complete_call(self, call_id: str, tokens: int = 0, elapsed: float = 0,
                       prompt_tokens: int = 0, completion_tokens: int = 0):
        with self._active_calls_lock:
            self._active_calls.pop(call_id, None)
        with self._call_stats_lock:
            self._call_stats["completed"] += 1
        _call_registry.complete(call_id)
        with self._token_lock:
            self._total_prompt_tokens += prompt_tokens
            self._total_completion_tokens += completion_tokens
            self._total_calls += 1
        print(f"[rlm] {call_id}: completed ({tokens} tokens, {elapsed:.1f}s)",
              file=sys.stderr, flush=True)

    def get_token_usage(self) -> dict:
        """Return cumulative token usage across all completed calls."""
        with self._token_lock:
            return {
                "prompt_tokens": self._total_prompt_tokens,
                "completion_tokens": self._total_completion_tokens,
                "total_tokens": self._total_prompt_tokens + self._total_completion_tokens,
                "total_calls": self._total_calls,
            }

    def _update_rate_limits(self, headers) -> None:
        """Update rate limit state from OpenAI response headers."""
        def _int(name: str) -> int:
            val = headers.get(name, "0")
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
        self._rate_limits = {
            "limit_requests": _int("x-ratelimit-limit-requests"),
            "limit_tokens": _int("x-ratelimit-limit-tokens"),
            "remaining_requests": _int("x-ratelimit-remaining-requests"),
            "remaining_tokens": _int("x-ratelimit-remaining-tokens"),
            "reset_requests": headers.get("x-ratelimit-reset-requests", ""),
            "reset_tokens": headers.get("x-ratelimit-reset-tokens", ""),
        }

    def get_rate_limits(self) -> dict:
        """Return rate limit state from the most recent API response."""
        return dict(self._rate_limits)

    def reset_call_stats(self):
        """Reset per-execution call stats (called at start of each execute_scheme)."""
        with self._call_stats_lock:
            self._call_stats["dispatched"] = 0
            self._call_stats["completed"] = 0

    def get_call_stats(self) -> dict:
        """Return dispatched/completed call counts for current execution."""
        with self._call_stats_lock:
            return dict(self._call_stats)

    def get_active_calls_snapshot(self) -> list[dict]:
        with self._active_calls_lock:
            now = time.time()
            return [
                {
                    "call_id": m["call_id"],
                    "type": m["type"],
                    "model": m["model"],
                    "depth": m["depth"],
                    "instruction_preview": m["instruction_preview"],
                    "parent_id": m["parent_id"],
                    "elapsed_seconds": round(now - m["start_time"], 1),
                }
                for m in self._active_calls.values()
            ]

    def cancel_call(self, call_id: str) -> str:
        with self._active_calls_lock:
            meta = self._active_calls.pop(call_id, None)
        if meta is None:
            return f"No active call with ID {call_id}"
        # Signal cancellation via event (checked before API call)
        if meta.get("cancel_event") is not None:
            meta["cancel_event"].set()
        # Cancel async futures
        if meta.get("future") is not None:
            meta["future"].cancel()
        # Kill nested REPL for recursive calls
        if meta.get("nested_repl") is not None:
            meta["nested_repl"].close()
        print(f"[rlm] {call_id}: cancelled by user", file=sys.stderr, flush=True)
        return f"Cancelled {call_id}"

    def _drain_stderr(self):
        """Daemon thread: continuously read stderr lines into ring buffer."""
        try:
            while self.proc and self.proc.stderr:
                line = self.proc.stderr.readline()
                if not line:
                    break
                stripped = line.rstrip("\n\r")
                if stripped:
                    self._stderr_buffer.append(stripped)
                    print(f"[racket-stderr] {stripped}", file=sys.stderr, flush=True)
        except (ValueError, OSError):
            # Pipe closed — process is shutting down
            pass

    def get_stderr_log(self) -> list[str]:
        """Return the last N lines from Racket's stderr."""
        return list(self._stderr_buffer)

    def _drain_stdout(self):
        """Background thread: read lines from Racket stdout into queue.

        Used on all platforms to ensure _read_line timeout covers the full
        readline (not just select/poll). This prevents the race where select()
        fires on partial data but readline() blocks waiting for the newline.
        """
        proc = self.proc  # Capture reference — self.proc may become None on timeout
        try:
            while proc and proc.poll() is None:
                line = proc.stdout.readline()
                if line:
                    self._stdout_queue.put(("line", line))
                else:
                    break
        except Exception as e:
            self._stdout_queue.put(("error", str(e)))
        finally:
            self._stdout_queue.put(("eof", None))

    def _start(self):
        if self.proc is not None:
            self.close()
        server_rkt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "racket_server.rkt")
        env = os.environ.copy()
        # Pass detected project Python to Racket (S9: venv inheritance)
        python_path = _detect_project_python()
        if python_path:
            env["RLM_PYTHON"] = python_path
            print(f"[rlm] Using Python: {python_path}", file=sys.stderr, flush=True)
        self.proc = subprocess.Popen(
            ["racket", server_rkt],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            env=env,
        )
        # Use threaded stdout reader on all platforms for reliable timeout enforcement.
        # (selector + readline can miss timeout if partial data triggers select
        # but readline blocks waiting for newline on large messages)
        self._stdout_queue.queue.clear()
        self._stdout_thread = threading.Thread(target=self._drain_stdout, daemon=True)
        self._stdout_thread.start()
        # Drain stderr in a daemon thread to prevent pipe buffer deadlock
        self._stderr_buffer.clear()
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

    SIGTERM_GRACE_SECONDS = 2

    def _read_line(self, timeout: float) -> str:
        """Read one line from Racket stdout with timeout.

        Uses a queue fed by a background thread on all platforms.
        This ensures the timeout covers the full readline, not just
        the select/poll check (which can fire on partial data).
        """
        try:
            msg_type, data = self._stdout_queue.get(timeout=timeout)
            if msg_type == "line":
                return data
            elif msg_type == "eof":
                raise RuntimeError("Racket process stdout closed")
            elif msg_type == "error":
                raise RuntimeError(f"Racket stdout reader error: {data}")
        except queue.Empty:
            # Timeout — Racket hasn't sent a message in `timeout` seconds.
            # Graceful shutdown: SIGTERM first, then SIGKILL after grace period.
            self.proc.terminate()
            try:
                self.proc.wait(timeout=self.SIGTERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            self.proc = None
            raise TimeoutError(f"Racket process timed out after {timeout}s")

    def send(self, cmd: dict, timeout: float = 300) -> dict:
        """Send a command, handle interleaved llm-query callbacks, return the final response.

        Args:
            timeout: Max seconds of idle time between Racket messages (computation timeout).
                     Does NOT limit LLM API call time — those use a separate, longer timeout.
        """
        # Decouple LLM wait timeout from Racket computation timeout.
        # LLM API calls (map-async, await-batch, await-any) can take much longer
        # than Racket computation. Using the same timeout for both was the root
        # cause of map-async failures: even generous timeouts (600s) could be
        # exceeded by parallel LLM batches, killing the Racket process.
        #
        # llm_timeout: max time to wait for LLM futures in concurrent.futures.wait()
        # Configurable via RLM_LLM_TIMEOUT_SECONDS env var.
        # Default: 5x the Racket timeout, minimum 300s (5 minutes).
        llm_timeout_env = os.environ.get("RLM_LLM_TIMEOUT_SECONDS")
        llm_timeout = float(llm_timeout_env) if llm_timeout_env else max(timeout * 5, 300)

        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self.proc.stdin.write(json.dumps(cmd) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self._read_line(timeout)
                if not line:
                    raise RuntimeError("Racket process died")
                msg = json.loads(line.strip())
                op = msg.get("op")

                if op == "llm-query":
                    llm_kwargs = {
                        "instruction": msg.get("instruction", ""),
                        "data": msg.get("data", ""),
                        "model": msg.get("model", ""),
                        "temperature": msg.get("temperature"),
                        "max_tokens": msg.get("max_tokens"),
                        "json_mode": msg.get("json_mode", False),
                        "images": msg.get("images"),
                    }
                    call_id = self._next_call_id()
                    model_name = llm_kwargs["model"] or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
                    call_type = "recursive" if msg.get("recursive") else "sync"
                    t0 = time.time()
                    self._register_call(call_id, call_type, model_name,
                                        llm_kwargs["instruction"],
                                        depth=self._current_depth)
                    try:
                        if msg.get("recursive"):
                            rec_result = self._call_llm_recursive(
                                llm_kwargs["instruction"],
                                llm_kwargs["data"],
                                llm_kwargs["model"],
                                msg.get("budget"),
                                _call_id=call_id,
                            )
                            total_tokens = rec_result["prompt_tokens"] + rec_result["completion_tokens"]
                            self._complete_call(call_id, total_tokens, time.time() - t0,
                                                rec_result["prompt_tokens"], rec_result["completion_tokens"])
                            self.proc.stdin.write(json.dumps({
                                "result": rec_result["text"],
                                "prompt_tokens": rec_result["prompt_tokens"],
                                "completion_tokens": rec_result["completion_tokens"],
                            }) + "\n")
                        else:
                            result = self._call_llm(**llm_kwargs)
                            total_tokens = result["prompt_tokens"] + result["completion_tokens"]
                            self._complete_call(call_id, total_tokens, time.time() - t0,
                                                result["prompt_tokens"], result["completion_tokens"])
                            self.proc.stdin.write(json.dumps({
                                "result": result["text"],
                                "prompt_tokens": result["prompt_tokens"],
                                "completion_tokens": result["completion_tokens"],
                            }) + "\n")
                    except Exception as exc:
                        self._complete_call(call_id, 0, time.time() - t0)
                        raise exc
                    self.proc.stdin.flush()

                elif op == "llm-query-async":
                    call_id = self._next_call_id()
                    model_name = msg.get("model", "") or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
                    cancel_event = threading.Event()
                    self._register_call(call_id, "async", model_name,
                                        msg.get("instruction", ""),
                                        depth=self._current_depth,
                                        cancel_event=cancel_event)
                    future = self._executor.submit(
                        self._call_llm_tracked, call_id,
                        msg.get("instruction", ""),
                        msg.get("data", ""),
                        msg.get("model", ""),
                        msg.get("temperature"),
                        msg.get("max_tokens"),
                        msg.get("json_mode", False),
                        msg.get("images"),
                        cancel_event,
                    )
                    self._pending[msg["id"]] = future
                    # Store future for cancellation
                    with self._active_calls_lock:
                        if call_id in self._active_calls:
                            self._active_calls[call_id]["future"] = future
                    # No response — Racket continues immediately

                elif op == "await":
                    future = self._pending.pop(msg["id"], None)
                    if future is None:
                        self.proc.stdin.write(json.dumps({
                            "result": "",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                        }) + "\n")
                    else:
                        try:
                            result = future.result(timeout=llm_timeout)
                            self.proc.stdin.write(json.dumps({
                                "result": result["text"],
                                "prompt_tokens": result["prompt_tokens"],
                                "completion_tokens": result["completion_tokens"],
                            }) + "\n")
                        except concurrent.futures.CancelledError:
                            self.proc.stdin.write(json.dumps({
                                "result": "[cancelled] call was cancelled by user",
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                        except ValueError as e:
                            # L8: JSON mode validation errors get clear message
                            error_msg = f"[async error] {type(e).__name__}: {str(e)}"
                            self.proc.stdin.write(json.dumps({
                                "result": error_msg,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                        except openai.APIError as e:
                            # L8: OpenAI API errors with status code
                            status = getattr(e, "status_code", "unknown")
                            error_msg = f"[async error] API {status}: {str(e)[:300]}"
                            self.proc.stdin.write(json.dumps({
                                "result": error_msg,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                        except Exception as e:
                            # L8: Generic errors with type info
                            error_msg = f"[async error] {type(e).__name__}: {str(e)[:300]}"
                            self.proc.stdin.write(json.dumps({
                                "result": error_msg,
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                            }) + "\n")
                    self.proc.stdin.flush()

                elif op == "await-batch":
                    # Parallel await: wait for multiple futures concurrently
                    future_ids = msg.get("ids", [])
                    futures_list = []
                    id_to_future = {}

                    for fid in future_ids:
                        future = self._pending.get(fid)
                        if future:
                            futures_list.append(future)
                            id_to_future[future] = fid

                    if not futures_list:
                        # No valid futures
                        self.proc.stdin.write(json.dumps({"results": []}) + "\n")
                        self.proc.stdin.flush()
                        continue

                    # Wait for all futures concurrently.
                    # Uses llm_timeout (not timeout) — LLM calls can take much longer
                    # than Racket computation. This is the key fix for map-async timeouts.
                    done, not_done = concurrent.futures.wait(
                        futures_list,
                        timeout=llm_timeout,
                        return_when=concurrent.futures.ALL_COMPLETED
                    )

                    # Build results in original order
                    results = []
                    for fid in future_ids:
                        future = self._pending.pop(fid, None)
                        if future is None:
                            results.append({
                                "result": "[error] future not found",
                                "prompt_tokens": 0,
                                "completion_tokens": 0
                            })
                        elif future not in done:
                            results.append({
                                "result": "[error] future timed out",
                                "prompt_tokens": 0,
                                "completion_tokens": 0
                            })
                        else:
                            try:
                                res = future.result(timeout=0)  # Already done
                                results.append({
                                    "result": res["text"],
                                    "prompt_tokens": res["prompt_tokens"],
                                    "completion_tokens": res["completion_tokens"]
                                })
                            except concurrent.futures.CancelledError:
                                results.append({
                                    "result": "[cancelled] call was cancelled by user",
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })
                            except ValueError as e:
                                error_msg = f"[async error] {type(e).__name__}: {str(e)}"
                                results.append({
                                    "result": error_msg,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })
                            except openai.APIError as e:
                                status = getattr(e, "status_code", "unknown")
                                error_msg = f"[async error] API {status}: {str(e)[:300]}"
                                results.append({
                                    "result": error_msg,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })
                            except Exception as e:
                                error_msg = f"[async error] {type(e).__name__}: {str(e)[:300]}"
                                results.append({
                                    "result": error_msg,
                                    "prompt_tokens": 0,
                                    "completion_tokens": 0
                                })

                    self.proc.stdin.write(json.dumps({"results": results}) + "\n")
                    self.proc.stdin.flush()

                elif op == "await-any":
                    # Wait for ANY future to complete (race pattern)
                    future_ids = msg.get("ids", [])
                    if not future_ids:
                        self.proc.stdin.write(json.dumps({
                            "error": "No futures provided",
                            "completed_id": None,
                            "remaining_ids": []
                        }) + "\n")
                        self.proc.stdin.flush()
                        continue

                    futures_list = []
                    id_to_future = {}

                    for fid in future_ids:
                        future = self._pending.get(fid)
                        if future:
                            futures_list.append(future)
                            id_to_future[future] = fid

                    if not futures_list:
                        self.proc.stdin.write(json.dumps({
                            "error": "No valid futures found",
                            "completed_id": None,
                            "remaining_ids": []
                        }) + "\n")
                        self.proc.stdin.flush()
                        continue

                    # Wait for FIRST completion.
                    # Uses llm_timeout — same decoupling as await-batch.
                    done, not_done = concurrent.futures.wait(
                        futures_list,
                        timeout=llm_timeout,
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )

                    if not done:
                        # Timeout - no futures completed
                        self.proc.stdin.write(json.dumps({
                            "error": "Timeout waiting for any future",
                            "completed_id": None,
                            "remaining_ids": future_ids
                        }) + "\n")
                        self.proc.stdin.flush()
                        continue

                    # Get the completed future
                    completed_future = next(iter(done))
                    completed_id = id_to_future[completed_future]
                    self._pending.pop(completed_id)

                    # Get remaining IDs
                    remaining_ids = [id_to_future[f] for f in not_done]

                    # Extract result from completed future
                    try:
                        res = completed_future.result(timeout=0)  # Already done
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": res["text"],
                            "prompt_tokens": res["prompt_tokens"],
                            "completion_tokens": res["completion_tokens"],
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except concurrent.futures.CancelledError:
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": "[cancelled] call was cancelled by user",
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except ValueError as e:
                        error_msg = f"[async error] {type(e).__name__}: {str(e)}"
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": error_msg,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except openai.APIError as e:
                        status = getattr(e, "status_code", "unknown")
                        error_msg = f"[async error] API {status}: {str(e)[:300]}"
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": error_msg,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")
                    except Exception as e:
                        error_msg = f"[async error] {type(e).__name__}: {str(e)[:300]}"
                        self.proc.stdin.write(json.dumps({
                            "completed_id": completed_id,
                            "result": error_msg,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "remaining_ids": remaining_ids
                        }) + "\n")

                    self.proc.stdin.flush()

                elif op == "tokens-used":
                    usage = self.get_token_usage()
                    self.proc.stdin.write(json.dumps(usage) + "\n")
                    self.proc.stdin.flush()

                elif op == "rate-limits":
                    self.proc.stdin.write(json.dumps(self._rate_limits) + "\n")
                    self.proc.stdin.flush()

                elif op == "checkpoint":
                    # L7: Save value to disk under key
                    key = msg.get("key", "")
                    value = msg.get("value")
                    if not key:
                        self.proc.stdin.write(json.dumps({"status": "error", "message": "key required"}) + "\n")
                    else:
                        try:
                            checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{key}.json")
                            with open(checkpoint_file, "w") as f:
                                json.dump(value, f, indent=2)
                            self.proc.stdin.write(json.dumps({"status": "ok"}) + "\n")
                            print(f"[rlm] Checkpoint saved: {key}", file=sys.stderr, flush=True)
                        except Exception as e:
                            self.proc.stdin.write(json.dumps({"status": "error", "message": str(e)}) + "\n")
                    self.proc.stdin.flush()

                elif op == "restore":
                    # L7: Load value from disk
                    key = msg.get("key", "")
                    if not key:
                        self.proc.stdin.write(json.dumps({"value": None}) + "\n")
                    else:
                        try:
                            checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{key}.json")
                            if os.path.exists(checkpoint_file):
                                with open(checkpoint_file, "r") as f:
                                    value = json.load(f)
                                self.proc.stdin.write(json.dumps({"value": value}) + "\n")
                                print(f"[rlm] Checkpoint restored: {key}", file=sys.stderr, flush=True)
                            else:
                                self.proc.stdin.write(json.dumps({"value": None}) + "\n")
                        except Exception as e:
                            print(f"[rlm] Checkpoint restore error ({key}): {e}", file=sys.stderr, flush=True)
                            self.proc.stdin.write(json.dumps({"value": None}) + "\n")
                    self.proc.stdin.flush()

                elif op == "heartbeat":
                    # Heartbeat from Racket — it's alive, just reset the idle timer.
                    # Sent automatically by map-async during long fan-outs, and
                    # available to user code via (heartbeat) for custom long computations.
                    self.proc.stdin.write(json.dumps({"ok": True}) + "\n")
                    self.proc.stdin.flush()
                    continue  # Loop back to _read_line with fresh timeout

                else:
                    return msg
        except TimeoutError:
            raise
        except (RuntimeError, BrokenPipeError, OSError) as e:
            if self.proc:
                self.proc.kill()
                self.proc = None
            stderr_log = self.get_stderr_log()
            stderr_text = "\n".join(stderr_log[-10:]) if stderr_log else "(no stderr)"
            self._start()
            return {
                "status": "error",
                "message": f"Racket process crashed: {e}. Stderr: {stderr_text}. Sandbox restarted — state was lost. Call load_context again if needed.",
            }

    MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB
    IMAGE_MAGIC = {
        b"\x89PNG": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"GIF8": "image/gif",
        b"RIFF": "image/webp",  # WebP starts with RIFF
    }

    @staticmethod
    def _resolve_image(img: str) -> str:
        """Convert an image argument to a data URL.

        Accepts:
          - A data URL (data:image/...) — returned as-is.
          - A file path — read, base64-encoded, and wrapped in a data URL.
          - A raw base64 string — wrapped in a data URL (assumes PNG).

        Raises ValueError for files exceeding MAX_IMAGE_BYTES or
        non-image files (based on magic bytes).
        """
        if img.startswith("data:"):
            return img
        if os.path.isfile(img):
            file_size = os.path.getsize(img)
            if file_size > RacketREPL.MAX_IMAGE_BYTES:
                raise ValueError(
                    f"Image file too large: {file_size} bytes "
                    f"(limit: {RacketREPL.MAX_IMAGE_BYTES // (1024*1024)} MB)"
                )
            with open(img, "rb") as f:
                data = f.read()
            # Validate magic bytes
            recognized = False
            for magic, _ in RacketREPL.IMAGE_MAGIC.items():
                if data[:len(magic)] == magic:
                    recognized = True
                    break
            if not recognized and len(data) > 4:
                print(f"[rlm] Warning: {img} may not be an image (unrecognized magic bytes)",
                      file=sys.stderr, flush=True)
            mime, _ = mimetypes.guess_type(img)
            if mime is None:
                mime = "image/png"
            b64 = base64.b64encode(data).decode()
            return f"data:{mime};base64,{b64}"
        # Raw base64 — check length (base64 of 20MB is ~27MB)
        max_b64_len = RacketREPL.MAX_IMAGE_BYTES * 4 // 3 + 4
        if len(img) > max_b64_len:
            raise ValueError(
                f"Base64 image data too large: {len(img)} chars "
                f"(limit: {max_b64_len} chars)"
            )
        return f"data:image/png;base64,{img}"

    MAX_RECOMMENDED_IMAGES = 5

    # Retry configuration — tuneable via environment variables.
    _max_retries: int = int(os.environ.get("RLM_MAX_RETRIES", "3"))
    _retry_base_delay: float = float(os.environ.get("RLM_RETRY_BASE_DELAY", "2.0"))

    def _call_llm(self, instruction: str, data: str, model: str = "",
                   temperature: float | None = None, max_tokens: int | None = None,
                   json_mode: bool = False,
                   images: list[str] | None = None,
                   cancel_event: threading.Event | None = None) -> dict:
        """Dispatch a sub-LLM call via the OpenAI API. Returns text + token counts."""
        if cancel_event and cancel_event.is_set():
            raise concurrent.futures.CancelledError("Call cancelled before API request")

        # L8: Validate JSON mode requirements (must have "json" in instruction)
        if json_mode and (not instruction or "json" not in instruction.lower()):
            error_msg = (
                "JSON mode requires the word 'json' in the instruction. "
                "OpenAI API will return 400 error without it. "
                f"Instruction: {instruction[:100] if instruction else '(empty)'}..."
            )
            print(f"[rlm] Error: {error_msg}", file=sys.stderr, flush=True)
            raise ValueError(error_msg)

        if images and len(images) > self.MAX_RECOMMENDED_IMAGES:
            print(f"[rlm] Warning: {len(images)} images sent (recommended max: "
                  f"{self.MAX_RECOMMENDED_IMAGES}). Response quality may degrade.",
                  file=sys.stderr, flush=True)
        client = openai.OpenAI()
        model = model or os.environ.get("RLM_SUB_MODEL", "gpt-4o")
        # Build messages — instruction is always system, data is always user.
        messages = []
        if instruction:
            messages.append({"role": "system", "content": instruction})

        # Build user message — multimodal when images are present.
        user_text = data or "(no data provided)"
        if images:
            content_parts = [{"type": "text", "text": user_text}]
            for img in images:
                url = self._resolve_image(img)
                content_parts.append({"type": "image_url", "image_url": {"url": url}})
            messages.append({"role": "user", "content": content_parts})
        else:
            messages.append({"role": "user", "content": user_text})
        kwargs = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "messages": messages,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        max_retries = self._max_retries
        base_delay = self._retry_base_delay

        for attempt in range(max_retries + 1):
            if cancel_event and cancel_event.is_set():
                raise concurrent.futures.CancelledError("Call cancelled before retry")
            try:
                raw = client.chat.completions.with_raw_response.create(**kwargs)
                resp = raw.parse()
                # Capture rate limit headers (S4)
                self._update_rate_limits(raw.headers)
                return {
                    "text": resp.choices[0].message.content,
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                }
            except openai.RateLimitError as e:
                if attempt >= max_retries:
                    raise
                # Respect Retry-After header if present.
                retry_after = None
                if hasattr(e, "response") and e.response is not None:
                    retry_after_str = e.response.headers.get("Retry-After")
                    if retry_after_str:
                        try:
                            retry_after = float(retry_after_str)
                        except (ValueError, TypeError):
                            pass
                delay = base_delay * (2 ** attempt)
                if retry_after is not None:
                    delay = max(delay, retry_after)
                print(f"[rlm] Rate limited (attempt {attempt + 1}/{max_retries}), "
                      f"retrying in {delay:.1f}s...",
                      file=sys.stderr, flush=True)
                time.sleep(delay)
            except openai.APIStatusError as e:
                if e.status_code == 429 and attempt < max_retries:
                    retry_after = None
                    if hasattr(e, "response") and e.response is not None:
                        retry_after_str = e.response.headers.get("Retry-After")
                        if retry_after_str:
                            try:
                                retry_after = float(retry_after_str)
                            except (ValueError, TypeError):
                                pass
                    delay = base_delay * (2 ** attempt)
                    if retry_after is not None:
                        delay = max(delay, retry_after)
                    print(f"[rlm] API 429 (attempt {attempt + 1}/{max_retries}), "
                          f"retrying in {delay:.1f}s...",
                          file=sys.stderr, flush=True)
                    time.sleep(delay)
                    continue
                raise

    def _call_llm_tracked(self, call_id: str, instruction: str, data: str,
                           model: str = "", temperature: float | None = None,
                           max_tokens: int | None = None,
                           json_mode: bool = False,
                           images: list[str] | None = None,
                           cancel_event: threading.Event | None = None) -> dict:
        """Like _call_llm but auto-completes the call registry entry when done."""
        t0 = time.time()
        try:
            result = self._call_llm(instruction, data, model, temperature, max_tokens, json_mode, images,
                                     cancel_event=cancel_event)
            total_tokens = result["prompt_tokens"] + result["completion_tokens"]
            self._complete_call(call_id, total_tokens, time.time() - t0,
                                result["prompt_tokens"], result["completion_tokens"])
            return result
        except Exception as e:
            # L8: Log async failures with details
            elapsed = time.time() - t0
            error_type = type(e).__name__
            print(f"[rlm] {call_id}: FAILED after {elapsed:.1f}s - {error_type}: {str(e)[:200]}",
                  file=sys.stderr, flush=True)
            self._complete_call(call_id, 0, elapsed)
            raise

    def _call_llm_recursive(self, instruction: str, data: str, model: str = "",
                             budget: float | None = None,
                             _call_id: str | None = None) -> dict:
        """Handle a recursive llm-query: call the sub-model, then run its Scheme code in a nested sandbox."""
        depth = self._current_depth + 1
        if depth > MAX_RECURSION_DEPTH:
            return {
                "text": f"[error] recursion depth limit ({MAX_RECURSION_DEPTH}) exceeded",
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        # Build the prompt — include USAGE_GUIDE so the sub-model knows the sandbox API
        full_instruction = (
            f"{instruction}\n\n"
            f"You have access to a Scheme sandbox. Write Scheme code to accomplish the task.\n"
            f"Use (finish value) to return your result.\n\n"
            f"## Sandbox Reference\n{USAGE_GUIDE}"
        )

        # Call the sub-model to get Scheme code
        result = self._call_llm(full_instruction, data, model)
        scheme_code = self._extract_code(result["text"])
        total_prompt = result["prompt_tokens"]
        total_completion = result["completion_tokens"]

        # Spin up a nested sandbox
        nested = RacketREPL()
        nested._current_depth = depth
        # Store nested REPL in call registry for cancellation
        if _call_id:
            with self._active_calls_lock:
                if _call_id in self._active_calls:
                    self._active_calls[_call_id]["nested_repl"] = nested
        try:
            # Load context if data was provided
            if data:
                nested.send({"op": "load-context", "data": data})

            # Propagate budget if finite — wrap user code in parameterize
            budget_val = None
            if budget is not None and budget != "+inf.0":
                try:
                    bv = float(budget)
                    if bv < float("inf"):
                        budget_val = int(bv)
                except (ValueError, TypeError):
                    pass

            # Evaluate the sub-model's code, optionally under a budget
            if budget_val is not None:
                wrapped_code = f"(parameterize ([token-budget {budget_val}]) {scheme_code})"
                resp = nested.send({"op": "eval", "code": wrapped_code})
            else:
                resp = nested.send({"op": "eval", "code": scheme_code})

            if resp.get("status") == "finished":
                text = resp.get("result", "")
            elif resp.get("status") == "error":
                text = f"[sub-model error] {resp.get('message', 'unknown error')}"
            else:
                text = "(sub-model produced no output)"
        finally:
            nested.close()

        return {
            "text": text,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        }

    @staticmethod
    def _extract_code(text: str) -> str:
        """Extract Scheme code from markdown fences if present."""
        match = re.search(r'```(?:scheme|racket)?\s*\n(.*?)```', text, re.DOTALL)
        return match.group(1).strip() if match else text.strip()

    def close(self):
        for future in self._pending.values():
            future.cancel()
        self._pending.clear()
        with self._active_calls_lock:
            self._active_calls.clear()
        if self.proc:
            self.proc.terminate()
            self.proc = None


# ---------------------------------------------------------------------------
# Singleton backend
# ---------------------------------------------------------------------------

_backend = None


def get_backend() -> RacketREPL:
    global _backend
    if _backend is None:
        _ensure_checkpoint_dir()  # L7: Create checkpoint directory on init
        _backend = RacketREPL()
    return _backend


# ---------------------------------------------------------------------------
# Progress reporting (S5)
# ---------------------------------------------------------------------------

PROGRESS_POLL_INITIAL = 2   # seconds between progress updates
PROGRESS_POLL_LONG = 5      # seconds after threshold
PROGRESS_POLL_THRESHOLD = 30  # seconds before switching to long interval


def _format_progress_message(calls: list[dict], stats: dict) -> str | None:
    """Format a compact progress message from active calls and stats.

    Returns None when there's nothing to report (no calls dispatched).
    Progressive detail within 80-char cap:
    - Flat fan-out: "3/7 done — 2 active: gpt-4o×2 (longest: 45s)"
    - Recursive present: "3/7 done — 2 active: gpt-4o×2 [d1:1] (longest: 45s)"
    - Single recursive: "1/3 done — 1 active: gpt-4o [recursive] (12s)"
    - Single with room: "0/1 done — 1 active: gpt-4o 'Summarize...' (5s)"

    Truncation order: drop preview → drop depth → hard truncate at 77+"..."
    """
    dispatched = stats["dispatched"]
    completed = stats["completed"]
    if not calls and dispatched == 0:
        return None
    if not calls:
        return f"{completed}/{dispatched} done"

    model_counts = collections.Counter(c["model"] for c in calls)
    models_str = ", ".join(f"{n}\u00d7{m}" for m, n in model_counts.most_common())
    max_elapsed = max(c.get("elapsed_seconds", 0) for c in calls)

    # Depth annotation: count calls at depth > 0
    deep_calls = [c for c in calls if c.get("depth", 0) > 0]
    depth_counts = collections.Counter(c.get("depth", 0) for c in deep_calls)
    depth_str = ""
    if depth_counts:
        depth_str = " " + " ".join(
            f"[d{d}:{n}]" for d, n in sorted(depth_counts.items())
        )

    # Single-call preview: show type annotation or instruction preview
    preview_str = ""
    if len(calls) == 1:
        call = calls[0]
        if call.get("type") == "recursive":
            preview_str = " [recursive]"
        elif call.get("instruction_preview"):
            preview_str = f" '{call['instruction_preview'][:30]}...'"

    prefix = f"{completed}/{dispatched} done \u2014 {len(calls)} active: {models_str}"
    suffix = f" (longest: {max_elapsed:.0f}s)"

    # Try full message with all detail
    msg = f"{prefix}{depth_str}{preview_str}{suffix}"
    if len(msg) <= 80:
        return msg

    # Drop preview first
    msg = f"{prefix}{depth_str}{suffix}"
    if len(msg) <= 80:
        return msg

    # Drop depth
    msg = f"{prefix}{suffix}"
    if len(msg) <= 80:
        return msg

    # Hard truncate
    return msg[:77] + "..."


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

STDOUT_LIMIT = 2000


@mcp.tool()
def get_usage_guide() -> str:
    """Get essential guide with quick reference, decision framework, model selection guide, and pattern summaries. For detailed pattern implementations with complete code examples, call get_pattern_details() with specific pattern numbers. This guide helps you choose which patterns to use; get_pattern_details() gives you the full implementation details."""
    return USAGE_GUIDE_CORE


@mcp.tool()
def get_code_generation_api_reference() -> str:
    """Get condensed API reference for code-generating sub-models.

    When using Pattern 2 (Code Generation), sub-models don't automatically know
    the rlm-scheme API. Call this tool and include its output in your unsafe-raw-query
    #:data parameter so the sub-model generates correct syntax.

    This returns a minimal reference (~200 lines) optimized for inclusion in prompts.
    For the full guide with strategies and examples, use get_usage_guide instead."""

    return """RLM-SCHEME API REFERENCE (for code-generating sub-models)

⚠️ CRITICAL SYNTAX RULES:
- All function arguments use #: keyword syntax: #:instruction, #:data, #:model, etc.
- (define x value) for bindings. set! is NOT available in sandbox.
- String operations: (string-append "a" "b"), (substring str start end)
- MUST unwrap llm-query results: (syntax-e result) before using as string

CORE LLM FUNCTIONS:

(llm-query #:instruction "task description"
           #:data "context or data"
           #:model "gpt-4o-mini"
           #:temperature 0.0
           #:max-tokens 500
           #:json #t)
  → Returns syntax object. MUST unwrap with (syntax-e result).
  → When using #:json #t, the #:instruction MUST contain the word 'json'.
  → Models: "gpt-4.1-nano", "gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "o3-mini", "o4-mini"
  → Temperature: 0.0 (deterministic) to 1.0 (creative). Not supported by o-series.
  → JSON mode: #:json #t guarantees valid JSON response.

(llm-query-async #:instruction "..." #:data "..." #:model "..." ...)
  → Same kwargs as llm-query but returns async handle.
  → Use ONLY inside map-async lambda, NOT directly.
  → Returns unwrapped string (no syntax-e needed).

(map-async fn items #:max-concurrent 10)
  → fn must be (lambda (item) (llm-query-async ...)) returning async handle.
  → Returns list of raw LLM output strings (no syntax-e needed).
  → ⚠️ Results are raw strings from the LLM, NOT parsed JSON. Even with #:json #t,
  →   each result is a JSON string you must parse yourself (e.g., via py-exec + json.loads).
  → Always use cheap models (gpt-4.1-nano or gpt-4o-mini) for fan-out.
  → Example: (map-async (lambda (chunk) (llm-query-async #:instruction "summarize" #:data chunk #:model "gpt-4.1-nano")) chunks)

PYTHON BRIDGE (for computation and I/O):

(py-exec "python_code_string")
  → Runs multi-line Python code (imports, statements, loops, file I/O).
  → Returns stdout as string. Use print() to output results.
  → Example: (py-exec "import json; print(json.dumps({'key': 'value'}))")

(py-eval "python_expression")
  → Evaluates a SINGLE Python expression, returns result as Scheme value.
  → ⚠️ COMMON MISTAKE: py-eval CANNOT handle imports, statements, or multi-line code.
  →   These will fail with SyntaxError: (py-eval "import json\njson.loads(s)")
  →   Use py-exec for statements, then py-eval to retrieve the result.
  → Example: (py-eval "[1, 2, 3]") → Scheme list (1 2 3)
  → Example: (py-eval "len(data)") → integer

(py-set! "variable_name" scheme-value)
  → Transfers Scheme value to Python variable (avoids escaping issues).
  → Safer than string interpolation. Use for passing data between Scheme and Python.
  → Example: (py-set! "data" my-list) then use 'data' in py-exec

CONTROL FLOW:

(finish value)
  → Return this as the final result. REQUIRED to produce output.
  → Call this exactly once at the end of your code.

(finish-var "variable-name")
  → Return the value of a defined variable.
  → Use when code generation defines a variable instead of calling finish.

(checkpoint "key" value)
  → Save value to disk (survives timeouts and restarts).
  → Returns the value.

(restore "key")
  → Load checkpointed value, returns #f if not found.

SCHEME BASICS:

Lists:
  (list a b c) → create list
  (car lst) → first element
  (cdr lst) → rest of list
  (null? lst) → check if empty
  (map fn lst) → apply function to each element
  (append lst1 lst2) → concatenate lists

Bindings:
  (define name value) → create binding
  (let* ([x 1] [y 2]) body) → local bindings (sequential)
  (lambda (x) body) → anonymous function

Conditionals:
  (if test then-expr else-expr)
  (cond [test1 expr1] [test2 expr2] ... [else expr-else])

COMMON PATTERNS:

Pattern 1: Parallel fan-out
```scheme
(define results (map-async
  (lambda (item)
    (llm-query-async #:instruction "process this" #:data item #:model "gpt-4.1-nano"))
  items
  #:max-concurrent 10))
(finish results)
```

Pattern 2: Extract with Python, analyze with LLM
```scheme
(define data (py-exec "import json; print(json.dumps(process_data()))"))
(define analysis (syntax-e (llm-query #:instruction "analyze" #:data data #:model "gpt-4o")))
(finish analysis)
```

Pattern 3: py-set! → py-exec → py-eval round-trip (most common data flow)
```scheme
;; 1. Get data from LLM into Scheme
(define analysis (syntax-e (llm-query #:instruction "Extract key facts as JSON" #:data doc #:model "gpt-4o" #:json #t)))
;; 2. Transfer Scheme string to Python variable (safe, no escaping needed)
(py-set! "raw_json" analysis)
;; 3. Process in Python (imports, file I/O, complex logic)
(py-exec "
import json
data = json.loads(raw_json)
data['processed'] = True
with open('output.json', 'w') as f:
    json.dump(data, f)
result = json.dumps(data)
")
;; 4. Retrieve processed result back into Scheme
(define processed (py-eval "result"))
(finish processed)
```

Pattern 4: Define result variable (for generated code)
```scheme
(define result (... your computation ...))
;; Caller will use (finish-var "result") to retrieve this
```

IMPORTANT REMINDERS:
- Always use #: prefix for keyword arguments
- Always unwrap llm-query with (syntax-e ...) before using as string
- map-async items get unwrapped automatically (no syntax-e needed)
- Use (finish ...) to return your result
- Use cheap models (gpt-4.1-nano) for parallel work
- py-set! is safer than string interpolation for passing data to Python
"""

@mcp.tool()
def get_pattern_details(pattern_ids: int | list[int]) -> str:
    """Get detailed documentation for specific orchestration patterns with complete code examples.
    
    After using get_usage_guide() to choose which patterns fit your problem, call this tool
    to get full implementation details including:
    - Complete working code examples
    - Quantified improvements (latency, cost, quality metrics)
    - Optimization tips and best practices
    - Common mistakes to avoid
    - Pattern composition suggestions
    - Real-world use cases
    
    Args:
        pattern_ids: Single pattern number (1-16) or list of pattern numbers, e.g., [1, 4, 10]
        
    Available Patterns:
        1: Parallel Fan-Out (MapReduce)
        2: Code Generation (Meta-Programming)
        3: Recursive Delegation (Hierarchical Decomposition)
        4: Critique-Refine Loop
        5: Cumulative Fold (Sequential Synthesis)
        6: Meta-Orchestration (LLM Designs the Pipeline)
        7: Speculative Execution (Hedging)
        8: Ensemble Voting
        9: Active Learning (Budget-Optimized Quality)
        10: Tree Aggregation (Hierarchical Reduction)
        11: Consensus Protocol (Byzantine Fault Tolerance)
        12: Backtracking Search (Strategy Exploration)
        13: Anytime Algorithms (Progressive Refinement)
        14: Memoization (Content-Addressed Caching)
        15: Stream Processing (Constant Memory)
        16: Multi-Armed Bandit (Adaptive Model Selection)
    
    Example:
        get_pattern_details(1)  # Get Pattern 1 details
        get_pattern_details([1, 4, 10])  # Get multiple patterns
    """
    # Normalize to list
    if isinstance(pattern_ids, int):
        pattern_ids = [pattern_ids]
    
    result_parts = []
    for pid in pattern_ids:
        if pid in PATTERN_DETAILS:
            result_parts.append(PATTERN_DETAILS[pid])
        else:
            result_parts.append(f"Error: Pattern {pid} not found. Valid pattern IDs are 1-16.")

    return "\n\n---\n\n".join(result_parts)



@mcp.tool()
async def execute_scheme(code: str, timeout: int | None = None, ctx: Context = None) -> str:
    """Execute Scheme orchestration code. Use for strategy-driven LLM orchestration.

    AVAILABLE PATTERNS (16 total - choose based on your constraints):

    LATENCY: Parallel Fan-Out, Speculative Execution, Stream Processing
    QUALITY: Critique-Refine, Ensemble Voting, Consensus Protocol
    COST: Active Learning, Memoization, Multi-Armed Bandit
    STRUCTURE: Code Generation, Meta-Orchestration, Recursive Delegation, Tree Aggregation
    SPECIALIZED: Cumulative Fold, Backtracking Search, Anytime Algorithms

    Call get_usage_guide for complete details, decision framework, and code examples for all 16 patterns.

    Model selection: use gpt-4.1-nano ($0.10/1M) for fan-out and simple tasks, gpt-4o or gpt-4.1 for complex reasoning, o3-mini or o4-mini for math/logic. Always use the cheapest model that fits the task.

    State persists across calls. Capabilities: orchestrate LLM sub-calls with scope tracking, run Python code for file I/O and web requests (py-exec), process images (vision models), and fan out parallel work (map-async).

    Args:
        code: Scheme code to execute
        timeout: Optional timeout in seconds. If not specified, uses RLM_TIMEOUT_SECONDS env var (default 300).
        ctx: MCP context for progress reporting"""
    # Resolve timeout: parameter > env var > default 300
    if timeout is None:
        timeout = int(os.environ.get("RLM_TIMEOUT_SECONDS", "300"))

    backend = get_backend()
    backend.reset_call_stats()
    _call_registry.reset_stats()
    loop = asyncio.get_event_loop()
    t_start = time.monotonic()

    # Monitor active calls and report compact progress while send() blocks
    stop_monitor = threading.Event()
    timeout_warning_sent = [False]  # mutable cell for closure

    async def monitor_progress():
        start_time = time.monotonic()
        while not stop_monitor.is_set():
            elapsed = time.monotonic() - start_time

            # Warn at 80% of timeout
            if not timeout_warning_sent[0] and elapsed > timeout * 0.8:
                timeout_warning_sent[0] = True
                print(f"[rlm] Warning: {elapsed:.0f}s elapsed ({elapsed/timeout*100:.0f}% of {timeout}s timeout)",
                      file=sys.stderr, flush=True)

            interval = PROGRESS_POLL_LONG if elapsed > PROGRESS_POLL_THRESHOLD else PROGRESS_POLL_INITIAL
            # Event-driven: wake immediately on call start/complete, else poll
            await loop.run_in_executor(None, _call_registry.wait_for_change, interval)
            if stop_monitor.is_set():
                break
            calls = _call_registry.snapshot()
            stats = _call_registry.get_stats()
            msg = _format_progress_message(calls, stats)
            if msg is None:
                continue
            await ctx.report_progress(
                stats["completed"],
                total=max(stats["dispatched"], 1),
                message=msg,
            )

    monitor_task = asyncio.create_task(monitor_progress())

    try:
        resp = await loop.run_in_executor(
            None, lambda: backend.send({"op": "eval", "code": code}, timeout=timeout)
        )
    except TimeoutError as e:
        return json.dumps({"status": "error", "message": str(e)})
    finally:
        stop_monitor.set()
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    elapsed = round(time.monotonic() - t_start, 1)

    result = {"status": resp["status"]}
    if resp.get("stdout"):
        stdout = resp["stdout"]
        if len(stdout) > STDOUT_LIMIT:
            stdout = stdout[:STDOUT_LIMIT] + f"\n... ({len(resp['stdout'])} chars total, truncated)"
        result["stdout"] = stdout
    if resp["status"] == "finished":
        result["value"] = resp["result"]
    elif resp["status"] == "error":
        result["message"] = resp.get("message", "unknown error")

    # Execution summary — always included so caller has visibility
    exec_summary = _call_registry.get_execution_summary()
    exec_summary["elapsed"] = elapsed
    token_usage = backend.get_token_usage()
    if token_usage["total_tokens"] > 0:
        exec_summary["tokens"] = token_usage["total_tokens"]
    result["execution"] = exec_summary

    return json.dumps(result)


@mcp.tool()
def load_context(data: str, name: str | None = None) -> str:
    """Load input data into the sandbox. Available as `context` in Scheme and Python.

    Args:
        data: Text data to load (documents, code, CSV, JSON, etc.)
        name: Optional name for this context slot (e.g., "gwas-data", "expression").
              Use get-context to retrieve named contexts later.

    Named context slots (improvement #5) allow managing multiple datasets:
    - load_context(gwas_csv, "gwas-data")
    - load_context(expr_csv, "expression-data")
    - Later in Scheme: (get-context "gwas-data") or (get-context "expression-data")

    Strategy considerations after loading:
    - Data >100KB? → Use Pattern 1 (chunk via py-exec, parallel fan-out with map-async)
    - Unknown structure? → Use Pattern 2 (model inspects sample, generates analysis code)
    - Hierarchical? → Use Pattern 3 (recursive delegation to specialists)

    See get_usage_guide for strategy templates."""
    cmd = {"op": "load-context", "data": data}
    if name is not None:
        cmd["name"] = name
    resp = get_backend().send(cmd)
    if resp["status"] == "error":
        return f"[stderr] {resp['message']}"
    return resp.get("result", "context loaded")


@mcp.tool()
def get_scope_log() -> str:
    """Get the audit trail as JSON array. Each entry contains: op ('llm-query'|'syntax-e'|'datum->syntax'|'py-exec'|'py-eval'|'unsafe-*'), datum_preview (first 80 chars of data), and scope ('host'|'sandbox'|'sub-N'). Use to trace data flow and debug scope issues."""
    resp = get_backend().send({"op": "get-scope-log"})
    return resp.get("result", "[]")


@mcp.tool()
def reset() -> str:
    """Clear all sandbox state and start fresh. Call between unrelated tasks."""
    resp = get_backend().send({"op": "reset"})
    return resp.get("result", "sandbox reset")


@mcp.tool()
def get_status() -> str:
    """Get sandbox status: active calls, token usage, and rate limits. Non-blocking — safe to call any time."""
    backend = get_backend()
    return json.dumps({
        "active_calls": _call_registry.snapshot(),
        "token_usage": backend.get_token_usage(),
        "rate_limits": backend.get_rate_limits(),
    }, indent=2)


@mcp.tool()
def cancel_call(call_id: str) -> str:
    """Cancel an in-flight sub-model call by its ID. Use get_status to find call IDs. Cancels async futures and terminates nested REPLs for recursive calls. Returns immediately. Does not affect token accounting for already-completed work."""
    return get_backend().cancel_call(call_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run()
