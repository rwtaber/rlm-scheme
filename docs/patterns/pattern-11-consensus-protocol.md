## Pattern 11: Consensus Protocol (Byzantine Fault Tolerance)

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

