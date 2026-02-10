## Pattern 9: Active Learning (Budget-Optimized Quality)

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
      #:model "gpt-3.5-turbo"
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
      #:model "gpt-4"
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

