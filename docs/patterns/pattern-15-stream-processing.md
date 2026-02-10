## Pattern 15: Stream Processing (Constant Memory)

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

