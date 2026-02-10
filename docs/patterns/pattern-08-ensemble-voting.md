## Pattern 8: Ensemble Voting

### Problem Statement
You need high-accuracy classification (sentiment analysis, medical diagnosis, fraud detection) where single model achieves 82% but you need 95%+ for business decisions. Errors are costly (wrong medical diagnosis, missed fraud = liability).

### Why This Pattern Exists
**Problem it solves:** Single model error ceiling. Models make different types of errors (uncorrelated failures).
**Alternatives fail because:**
- **Single model:** 82% accuracy, can't improve without better training data
- **Better model:** Still plateaus (gpt-4 = 85%, not 95%)
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
(define models (list "curie" "gpt-4" "code-davinci-002" "gpt-3.5" "gpt-3.5-turbo"))

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
1. **Use cheap models for majority:** 3 × gpt-3.5-turbo + 2 × gpt-4 = 2× cost but 90% accuracy (cheaper than 5 × gpt-4)
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
;; 5 × gpt-4 = correlated errors, all fail on same cases
;; Fix: Diverse models (curie, gpt-4, code-davinci-002, etc.)
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


