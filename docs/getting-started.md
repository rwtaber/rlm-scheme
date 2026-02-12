# RLM-Scheme Usage Guide

**Composable combinators for LLM orchestration**

---

## Philosophy: Playbook, Not Cookbook

RLM-Scheme provides **building blocks**, not recipes. The ~17 combinators can be composed in infinite ways.

**Don't just follow patterns - experiment!**

**Why experimentation works:**
- Cost of testing 3 approaches: **$0.01-0.05**
- Cost of choosing wrong strategy: **$1-5**
- **Conclusion:** Real experiments beat theoretical planning

**What this means:**
- Try creative compositions
- Test multiple approaches
- Break patterns when it makes sense
- Learn what works through iteration

---

## What is RLM-Scheme?

A Scheme sandbox for composing LLM orchestration strategies using **combinators** - reusable building blocks that handle:
- Parallelization and concurrency control
- Hierarchical aggregation (avoiding context limits)
- Cost optimization and model selection
- Quality control and validation
- Iterative refinement
- Error handling and fallbacks

**Core idea:** Learn ~17 combinators, compose them into custom strategies for your needs.

---

## Quick Start

### Option 1: Get Strategy Recommendations (Easiest)

```python
# 1. Ask the planner for strategies
plan = plan_strategy(
    task_description="Process 100 documents and synthesize final report",
    data_characteristics="~5KB per document, ~500KB total",
    priority="balanced"  # speed/cost/quality/balanced
)

# 2. Load your data
load_context(your_documents)

# 3. Try the recommended strategy
result = execute_scheme(plan["recommended"]["code_template"])

# 4. Or try alternatives/creative options
result = execute_scheme(plan["alternatives"][0]["code_template"])
result = execute_scheme(plan["creative_options"][0]["code_template"])
```

### Option 2: Compose Manually

```python
# 1. Learn combinators
get_combinator_reference()

# 2. Design your strategy
code = '''
(define docs (py-eval "context.split('\\\\n\\\\n')"))

(define result (fan-out-aggregate
  (lambda (doc)
    (llm-query-async
      #:instruction "Extract key points"
      #:data doc
      #:model "gpt-4o-mini"))
  (lambda (results)
    (tree-reduce
      (lambda args (llm-query #:instruction "Combine" #:data (string-join args)))
      results
      #:branch-factor 5))
  docs))

(finish result)
'''

# 3. Execute
load_context(your_data)
execute_scheme(code)
```

---

## Core Concepts

### 1. Combinators - The Building Blocks (~17 total)

**Parallel Execution:** `parallel`, `race`
**Sequential Processing:** `sequence`, `fold-sequential`
**Hierarchical Aggregation:** `tree-reduce`, `fan-out-aggregate`, `recursive-spawn`
**Iterative Refinement:** `iterate-until`, `critique-refine`
**Quality Control:** `with-validation`, `vote`, `ensemble`
**Cost Optimization:** `tiered`, `active-learning`, `memoized`
**Control Flow:** `choose`, `try-fallback`

**For full details:** `get_combinator_reference()` - comprehensive documentation with examples

### 2. Execution Model

**State Persistence:**
- Scheme variables, Python globals, and disk checkpoints persist across `execute_scheme()` calls
- Call `reset()` to clear state between unrelated tasks
- Use `checkpoint("key", value)` for critical data that must survive timeouts

**Async Operations:**
- `llm-query-async` returns handle immediately (non-blocking)
- `await` or `await-all` to collect results
- `map-async` for efficient batched parallel execution with concurrency control

**Timeouts:**
- Default: 300s computation timeout (configurable via `execute_scheme(code, timeout=600)`)
- Use checkpoints to save expensive intermediate results

### 3. Model Selection

| Model | Cost/1K | Characteristics |
|-------|---------|----------------|
| gpt-4.1-nano | $0.0001 | Simple extraction, classification |
| gpt-4o-mini | $0.0005 | General tasks, bulk processing |
| gpt-4o | $0.01 | Complex reasoning, synthesis |
| gpt-4.5 | $0.03 | Highest quality |

**No rules - just tradeoffs.** Experiment to find what works for your task.

---

## Example Strategies

### Standard: Parallel Processing with Tree Reduction

```scheme
(define docs (py-eval "context.split('\\n\\n')"))

(define result (fan-out-aggregate
  ;; Map: Extract from each doc in parallel
  (lambda (doc)
    (llm-query-async
      #:instruction "Extract key insights as bullet points"
      #:data doc
      #:model "gpt-4o-mini"))

  ;; Reduce: Hierarchical synthesis
  (lambda (extracts)
    (tree-reduce
      (lambda args
        (syntax-e (llm-query
          #:instruction "Combine these summaries"
          #:data (string-join (map syntax-e args) "\n\n")
          #:model "gpt-4o")))
      extracts
      #:branch-factor 5))

  docs))

(finish result)
```

**Cost:** ~$0.05-0.20 for 100 docs | **Time:** ~5-10s | **Quality:** High

---

### Creative: Multi-Stage Refinement with Voting

```scheme
;; Stage 1: Parallel extraction
(define raw-extracts (map-async
  (lambda (doc) (llm-query-async #:instruction "Extract" #:data doc #:model "gpt-4o-mini"))
  documents))

;; Stage 2: Tree reduction with voting at each node
(define synthesis (tree-reduce
  (lambda args
    (vote
      (list
        (lambda () (llm-query #:instruction "Analytical merge" #:data (string-join args)))
        (lambda () (llm-query #:instruction "Narrative merge" #:data (string-join args)))
        (lambda () (llm-query #:instruction "Technical merge" #:data (string-join args))))
      #:method 'plurality))
  raw-extracts
  #:branch-factor 3))

;; Stage 3: Critique-refine final output
(define final (critique-refine
  (lambda () synthesis)
  (lambda (draft) (llm-query #:instruction "Critique" #:data draft #:model "gpt-4o-mini"))
  (lambda (draft critique) (llm-query #:instruction "Refine" #:data (string-append draft "\n" critique) #:model "gpt-4o"))
  #:max-iter 2))

(finish final)
```

**Why creative:** Voting at tree nodes for robust aggregation, multi-stage pipeline
**Cost:** Higher (~$0.50-1.00) | **Quality:** Very high

---

### Experimental: Adaptive Tiering with Memoization

```scheme
;; Deduplicate similar documents first
(define unique-extracts (memoized
  (lambda (doc) (llm-query-async #:instruction "Extract" #:data doc #:model "gpt-4.1-nano"))
  #:key-fn (lambda (args) (substring (car args) 0 (min 100 (string-length (car args)))))))

;; Apply to all docs (automatically deduplicates)
(define extracts (map unique-extracts documents))

;; Adaptive processing based on complexity
(define processed (active-learning
  (lambda (extract) (llm-query-async #:instruction "Simple analysis" #:data extract #:model "gpt-4o-mini"))
  (lambda (extract) (llm-query-async #:instruction "Deep analysis" #:data extract #:model "gpt-4o"))
  (lambda (result) (if (< (string-length result) 100) 0.9 0.5))  ; Short = uncertain
  (await-all extracts)
  #:threshold 0.7))

(finish (tree-reduce combine-fn processed))
```

**Why experimental:** Content-based deduplication + adaptive routing based on output length
**Risk:** Complexity | **Upside:** Significant cost savings on redundant data

---

## What's Possible

**Arbitrary composition:**
- Nest combinators as deep as needed
- Mix sync and async operations
- Combine multiple combinators in a single strategy

**Creative control flow:**
- Conditional routing with `choose`
- Feedback loops with `iterate-until`
- Robust execution with `try-fallback`
- Multi-model consensus with `vote`/`ensemble`

**Performance tuning:**
- Parallel execution: `parallel`, `race`, `map-async`
- Hierarchical aggregation: `tree-reduce`, `fan-out-aggregate`
- Cost optimization: `tiered`, `active-learning`, `memoized`

**Quality optimization:**
- Iterative refinement: `critique-refine`, `iterate-until`
- Validation: `with-validation`
- Consensus: `vote`, `ensemble`
- Recursive delegation: `recursive-spawn`

**There are no "correct" patterns - only tradeoffs.** Experiment to find what works.

---

## Execution Model Details

### State Persistence

**What persists:**
1. Scheme variables: `(define x ...)` stays across executions
2. Python globals: `(py-exec "x = 42")` persists
3. Checkpoints: `(checkpoint "key" value)` saved to disk

**What doesn't persist:**
- State cleared when you call `reset()`
- Python subprocess may restart after timeout (use checkpoints!)

**Best practice:** Call `reset()` between unrelated tasks

### Async Operations

```scheme
;; Manual async control
(define f1 (llm-query-async #:instruction "Task 1" #:data data1))
(define f2 (llm-query-async #:instruction "Task 2" #:data data2))
(define results (await-all (list f1 f2)))

;; Or use map-async for batches
(define results (map-async
  (lambda (item) (llm-query-async #:instruction "Process" #:data item))
  items
  #:max-concurrent 20))
```

### Data Transfer: Moving Data Between Scheme, Python, and Files

RLM-Scheme provides three mechanisms for data transfer and persistence. Understanding when to use each saves time and prevents errors.

| Mechanism | Purpose | Persistence | Size Limit | Best For |
|-----------|---------|-------------|------------|----------|
| **py-set!** | Scheme → Python transfer | Subprocess lifetime | ~1MB | Passing data to Python for processing |
| **checkpoint** | Save to disk | Survives timeouts/restarts | Disk limit | Expensive results, recovery points |
| **File I/O** | Write to filesystem | Permanent | Disk limit | Final outputs, multiple files |

#### Quick Decision Guide

```
Is this final output that needs to persist?
  YES → File I/O (permanent, multiple files OK)
  NO  → Is it >1MB or might timeout?
    YES → checkpoint (recovery + persistence)
    NO  → Need to process in Python?
      YES → py-set! (fast transfer)
      NO  → Keep in Scheme (just use define)
```

---

#### 1. py-set! - Scheme to Python Transfer

**Purpose:** Transfer Scheme values to Python variables safely, without worrying about quote escaping or special characters.

**How it works:** Serializes value to JSON, sends to Python subprocess, deserializes into Python variable.

**Basic workflow:**
```scheme
;; Step 1: Get LLM result (returns syntax object)
(define result (llm-query #:instruction "Extract data" #:data context))

;; Step 2: Unwrap syntax object (IMPORTANT!)
(define text (syntax-e result))

;; Step 3: Transfer to Python
(py-set! "llm_output" text)

;; Step 4: Process in Python
(py-exec "
import json
data = json.loads(llm_output)
processed = transform(data)
output = json.dumps(processed)
")

;; Step 5: Read result back to Scheme
(define processed (py-eval "output"))
(finish processed)
```

**Why use py-set! instead of string-append?**

❌ **This breaks with quotes/newlines:**
```scheme
(define content "Has \"quotes\" and \n newlines")
(py-exec (string-append "x = \"" content "\""))  ; JSON error!
```

✅ **This always works:**
```scheme
(define content "Has \"quotes\" and \n newlines")
(py-set! "x" content)  ; Safe - proper JSON serialization
(py-exec "print(x)")   ; Works perfectly
```

**⚠️ Common Mistakes:**

1. **Forgetting to unwrap syntax objects:**
```scheme
;; ❌ Wrong - syntax object not JSON-serializable
(define result (llm-query #:instruction "task" #:data context))
(py-set! "x" result)  ; ERROR!

;; ✅ Right - unwrap first
(define result (syntax-e (llm-query #:instruction "task" #:data context)))
(py-set! "x" result)  ; OK
```

2. **Too many small transfers:**
```scheme
;; ❌ Slow - 100 round-trips = 100-500ms overhead
(for ([item items])
  (py-set! "current" item)
  (py-exec "process(current)"))

;; ✅ Fast - 1 round-trip = 1-5ms overhead
(py-set! "all_items" items)
(py-exec "
for item in all_items:
    process(item)
")
```

**✅ Best practices:**
- Unwrap syntax objects with `syntax-e` before transfer
- Batch multiple values into a list or hash
- Use for <1MB data transfers
- Keep Python processing in py-exec (avoid multiple round-trips)

---

#### 2. checkpoint - Disk Persistence

**Purpose:** Save expensive computation results to disk so they survive timeouts, crashes, and can be reused across multiple `execute_scheme()` calls.

**How it works:** Serializes value to JSON and writes to `.rlm-scheme-checkpoints/{key}.json` file.

**Basic usage:**
```scheme
;; Save checkpoint (returns the value for chaining)
(define extractions (map-async expensive-extract documents))
(checkpoint "doc-extractions" extractions)  ; Saved to disk
(finish extractions)
```

**Restore in next execution:**
```scheme
;; Try to restore, recompute if missing
(define extractions
  (or (restore "doc-extractions")
      (begin
        (displayln "Checkpoint not found, recomputing...")
        (map-async expensive-extract documents))))

;; Continue from checkpoint
(define synthesis (expensive-synthesis extractions))
(finish synthesis)
```

**Multi-stage pipeline pattern:**
```scheme
;; Stage 1: Extraction (5 minutes, expensive)
(define extractions
  (or (restore "stage1-extractions")
      (begin
        (define result (map-async extract documents))
        (checkpoint "stage1-extractions" result)
        result)))

;; Stage 2: Synthesis (2 minutes)
(define synthesis
  (or (restore "stage2-synthesis")
      (begin
        (define result (synthesize extractions))
        (checkpoint "stage2-synthesis" result)
        result)))

;; Stage 3: Refinement (3 minutes)
(define final
  (or (restore "stage3-final")
      (begin
        (define result (refine synthesis))
        (checkpoint "stage3-final" result)
        result)))

(finish final)
```

**When checkpoints survive:**
- ✅ Timeouts (Racket process killed)
- ✅ Errors in subsequent code
- ✅ Across multiple `execute_scheme()` calls
- ✅ After `reset()` is called
- ❌ Manual deletion of checkpoint files
- ❌ Disk full or I/O errors

**⚠️ Limitations:**
- Same JSON serialization constraints as py-set!
- Must unwrap syntax objects first
- No automatic cleanup (files persist forever)
- Large checkpoints (>10MB) are slow to save/restore
- No versioning (newer checkpoint overwrites old one)

**✅ Use checkpoints for:**
- Expensive computations (>1 minute runtime)
- Multi-stage pipelines with recovery points
- Data that must survive timeouts
- Results you want to inspect across multiple runs

---

#### 3. File I/O - Write Permanent Outputs

**Purpose:** Write final results to permanent files on disk.

**How it works:** Use Python bridge to execute file I/O operations (Scheme sandbox has no direct file access for security).

**Single file:**
```scheme
;; Generate content
(define readme (syntax-e (llm-query
  #:instruction "Generate comprehensive README"
  #:data context
  #:model "gpt-4o")))

;; Write to file via Python
(py-set! "content" readme)
(py-exec "
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(content)
print(f'Wrote {len(content)} chars to README.md')
")

(finish "README.md written")
```

**Multiple files (efficient batch pattern):**
```scheme
;; Generate all documentation in parallel
(define sections (list
  (hash "name" "intro" "topic" "Introduction")
  (hash "name" "usage" "topic" "Usage Guide")
  (hash "name" "api" "topic" "API Reference")))

(define docs (map-async
  (lambda (section)
    (llm-query-async
      #:instruction (format "Generate ~a documentation" (hash-ref section "topic"))
      #:data context
      #:model "gpt-4o"))
  sections))

;; Prepare file specifications
(define file-specs
  (map (lambda (section doc)
         (hash "filename" (format "docs/~a.md" (hash-ref section "name"))
               "content" doc))
       sections
       docs))

;; Write all files in single Python batch (efficient!)
(py-set! "files" file-specs)
(py-exec "
import os
success = 0
for spec in files:
    path = spec['filename']
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(spec['content'])
    success += 1
print(f'Wrote {success}/{len(files)} files')
")
```

**File path handling:**
```scheme
;; Relative paths (relative to where MCP server started)
(py-exec "with open('output.txt', 'w') as f: f.write('data')")
;; → /current/working/directory/output.txt

;; Subdirectories (auto-created with os.makedirs)
(py-exec "
import os
os.makedirs('docs/api', exist_ok=True)
with open('docs/api/reference.md', 'w') as f:
    f.write(content)
")

;; Absolute paths
(py-exec "with open('/tmp/output.txt', 'w') as f: f.write('data')")
```

**⚠️ Common mistakes:**

1. **Forgetting to create parent directories:**
```scheme
;; ❌ Fails if docs/ doesn't exist
(py-exec "with open('docs/api.md', 'w') as f: f.write(content)")

;; ✅ Create parent directory first
(py-exec "
import os
os.makedirs('docs', exist_ok=True)
with open('docs/api.md', 'w') as f:
    f.write(content)
")
```

2. **Not unwrapping syntax objects:**
```scheme
;; ❌ Syntax object in file
(define doc (llm-query ...))
(py-set! "content" doc)  ; ERROR: syntax object not serializable

;; ✅ Unwrap first
(define doc (syntax-e (llm-query ...)))
(py-set! "content" doc)  ; OK: string
```

---

#### Complete End-to-End Example

**Scenario:** Generate 50 documentation files, handle timeouts, write to disk

```scheme
;; Load sections (assume loaded via load_context)
(define sections (py-eval "
import json
sections = json.loads(context)
json.dumps(sections[:50])  # Limit to 50 for this example
"))

;; Stage 1: Generate all docs in parallel (might take 5-10 minutes)
(define docs
  (or (restore "generated-docs")
      (begin
        (displayln "Generating documentation...")
        (define result (map-async
          (lambda (section-json)
            (llm-query-async
              #:instruction (format "Generate documentation for ~a" section-json)
              #:data context
              #:model "gpt-4o"
              #:max-tokens 4000))
          sections
          #:max-concurrent 20))
        ;; Checkpoint BEFORE any file operations
        (checkpoint "generated-docs" result)
        (displayln "Documentation generated and checkpointed")
        result)))

;; Stage 2: Write all files (fast, but checkpoint just in case)
(define write-result
  (or (restore "files-written")
      (begin
        (py-set! "sections" sections)
        (py-set! "docs" docs)
        (define output (py-exec "
import os
import json

# Parse sections
section_data = json.loads(sections)

# Write each file
written = []
for i, (section, doc) in enumerate(zip(section_data, docs)):
    filename = f'output/doc_{i:03d}_{section.get(\"name\", \"unnamed\")}.md'
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(doc)

    written.append({
        'file': filename,
        'size': len(doc)
    })

# Write manifest
manifest = {
    'total_files': len(written),
    'total_size': sum(f['size'] for f in written),
    'files': written
}

with open('output/manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f'Wrote {len(written)} files')
result = json.dumps(manifest)
"))
        (checkpoint "files-written" output)
        output)))

;; Stage 3: Return summary
(finish write-result)
```

**What this demonstrates:**
- ✅ Checkpoints before expensive operations (stage 1)
- ✅ Checkpoints after critical operations (stage 2)
- ✅ Batch file writes in Python (efficient)
- ✅ Proper error handling with or pattern
- ✅ Progress tracking with displayln
- ✅ Manifest file for tracking outputs

---

#### The `context` Variable

`load_context(data)` automatically makes data available in both Scheme and Python:

```python
# In Python
load_context('{"users": [{"name": "Alice"}, {"name": "Bob"}]}')
```

```scheme
;; Access in Scheme
(define data context)  ; Available immediately

;; Access in Python
(py-exec "
import json
data = json.loads(context)
print(f'Found {len(data[\"users\"])} users')
")  ; Prints: Found 2 users
```

---

#### Quick Reference

| Task | Code | Notes |
|------|------|-------|
| **Transfer to Python** | `(py-set! "var" value)` | Unwrap syntax objects first |
| **Run Python code** | `(py-exec "code")` | Returns stdout as string |
| **Get Python value** | `(py-eval "expression")` | Single expression only |
| **Save checkpoint** | `(checkpoint "key" value)` | Survives timeouts |
| **Restore checkpoint** | `(restore "key")` | Returns #f if not found |
| **Write single file** | `(py-exec "with open...")` | Transfer data with py-set! first |
| **Write multiple files** | Batch with py-set! + loop | More efficient than individual writes |

---

#### When to Use Each Mechanism

**Use py-set! when:**
- Passing LLM results to Python for processing
- Data is <1MB
- Need Python libraries (pandas, json, etc.)
- Avoiding Scheme string escaping issues

**Use checkpoint when:**
- Computation takes >1 minute
- Risk of timeout
- Multi-stage pipeline (save between stages)
- Want to inspect/reuse results later
- Need recovery points

**Use File I/O when:**
- Final output files needed
- Multiple output files (docs, reports, etc.)
- Results >1MB
- Permanent storage required
- Sharing results with other tools

**📖 Full details:** See [Data Transfer Patterns](data-transfer-patterns.md) for comprehensive guide with performance optimization, error handling strategies, and advanced patterns.

---

### Debugging Tools

```python
# Inspect sandbox state
get_sandbox_state()  # Variables, checkpoints, Python status

# Monitor active operations
get_status()  # Active calls, token usage, rate limits

# View execution history
get_scope_log()  # Audit trail of all LLM calls

# Cancel long-running call
cancel_call("call_123")
```

---

## Available Tools

**Planning & Reference:**
- `plan_strategy(task, data, priority)` - Get custom strategy recommendations
- `get_usage_guide()` - This guide
- `get_combinator_reference()` - Full docs for all 17 combinators
- `get_code_generation_api_reference()` - API reference

**Execution:**
- `load_context(data, name)` - Load data into sandbox
- `execute_scheme(code, timeout)` - Run Scheme code
- `reset()` - Clear sandbox state

**Monitoring:**
- `get_sandbox_state()` - Inspect current state
- `get_status()` - Active calls, token usage
- `get_scope_log()` - Execution audit trail
- `cancel_call(call_id)` - Cancel active call

---

## Tips for Success

1. **Start small** - Test on 10-20 items before scaling to 1000
2. **Experiment freely** - Testing costs pennies, wrong choices cost dollars
3. **Use async** - `map-async` for independent operations
4. **Tree-reduce for scale** - Hierarchical aggregation avoids context limits
5. **Checkpoint expensive work** - Save intermediate results to disk
6. **Monitor costs** - Check `get_status()` for token usage
7. **Try creative approaches** - Novel compositions often outperform safe patterns

---

## Workflow for New Users

1. **Get recommendations:**
   ```python
   plan = plan_strategy("Your task here", priority="balanced")
   ```

2. **Review options:**
   - Check `plan["recommended"]` for primary strategy
   - Review `plan["alternatives"]` for different tradeoffs
   - Consider `plan["creative_options"]` for experimental approaches

3. **Test strategies:**
   ```python
   load_context(your_data)
   result1 = execute_scheme(plan["recommended"]["code_template"])
   result2 = execute_scheme(plan["creative_options"][0]["code_template"])
   ```

4. **Iterate:**
   - If unsatisfied, try other options
   - Read `get_combinator_reference()` for deeper understanding
   - Compose your own strategy based on what you learned

5. **Reset between tasks:**
   ```python
   reset()  # Clear state before new work
   ```

---

## Next Steps

**Quick start:** `plan_strategy("Your task")` → execute recommended strategy

**Deep dive:** `get_combinator_reference()` → compose manually

**Experiment:** Try multiple strategies, compare results, learn what works

---

**Ready to build?** Start with `plan_strategy()` or dive into `get_combinator_reference()`. Remember: experimentation is cheap, so try bold ideas!
