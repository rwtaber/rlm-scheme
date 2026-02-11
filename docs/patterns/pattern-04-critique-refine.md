## Pattern 4: Critique-Refine Loop

### Problem Statement
You need to generate high-quality content (technical white paper, code architecture, research proposal) where first-draft quality is insufficient. Single-shot generation produces vague claims, logical gaps, missing context. You need systematic improvement through iteration.

### Why This Pattern Exists
**Problem it solves:** Single LLM call quality ceiling (~70% for complex creative tasks). Human experts iterate; so should LLMs.  
**Alternatives fail because:**
- **Single-shot:** No feedback loop, quality plateaus at 60-70%
- **Multiple independent tries:** Wastes tokens, doesn't build on critique
- **Human-in-the-loop:** Slow, expensive, doesn't scale

**Key insight:** Adversarial critique (cheap model) + responsive refinement (expensive model) = systematic quality improvement at lower cost than expensive-model-only multi-shot.

### Limitations: What Critique-Refine Can and Cannot Fix

**✅ What Critique-Refine CAN fix:**
- **Structural issues:** Missing sections, poor organization, unclear flow
- **Vagueness:** Generic statements → specific claims with evidence
- **Style problems:** Tone, formatting, clarity, readability
- **Logical consistency:** Internal contradictions, weak arguments
- **Completeness:** Missing context, unexplained assumptions

**❌ What Critique-Refine CANNOT fix:**
- **Factual hallucinations:** Invented APIs, wrong data, fabricated references
- **Missing external context:** Information not in the original input
- **Fundamental approach errors:** Wrong strategy, impossible requirements
- **Domain knowledge gaps:** Tasks requiring expertise the model lacks

**Why the limitation exists:**

The critique model has no access to ground truth. It can only evaluate based on:
- Internal consistency (does it contradict itself?)
- Structural completeness (are all sections present?)
- Stylistic quality (is it well-written?)

It CANNOT verify:
- Whether class `FooBarFactory` actually exists in your codebase
- Whether the cited paper's methodology matches the description
- Whether the SQL query will actually return correct results

**Real example (from GraphRAG evaluation):**
```scheme
;; Generated draft documents invented package "graphrag-community"
(define draft (llm-query #:instruction "Document this package" ...))

;; Critique checked structure, not facts
(define critique (llm-query
  #:instruction "Review this documentation"
  #:data draft))
;; Result: "Good structure, add more details about CommunityDetector class"

;; Refine made it MORE confidently wrong
(define refined (llm-query
  #:instruction "Improve based on critique"
  #:data draft))
;; Result: Expanded description of non-existent CommunityDetector
```

The critique-refine loop made the hallucination *more elaborate*, not more accurate.

### When to Add Verification

For tasks requiring factual accuracy (code documentation, data analysis, technical reference), add a **verification step** after refinement:

```scheme
;; Standard critique-refine
(define draft (llm-query #:instruction "Generate API docs" #:data code))
(define critique (llm-query #:instruction "Critique" #:data draft #:model "curie"))
(define refined (llm-query #:instruction "Refine" #:data (string-append draft critique)))

;; + Verification step (for factual tasks)
(define verified (py-exec "
# Check every documented class actually exists
import re
import subprocess

documented_classes = re.findall(r'class (\\w+)', refined)
errors = []

for cls in documented_classes:
    # Grep the codebase
    result = subprocess.run(['grep', '-r', f'class {cls}', 'src/'], capture_output=True)
    if result.returncode != 0:
        errors.append(f'HALLUCINATION: Class {cls} not found in codebase')

print('PASS' if not errors else f'FAIL: {chr(10).join(errors)}')
"))

(if (string-contains? verified "FAIL")
    (finish-error (string-append "Documentation contains hallucinations:\\n" verified))
    (finish refined))
```

**Better approach for factual tasks:** Use **Pattern 17 (Hybrid Extraction)**
- Extract facts deterministically (AST, JSON parsing, grep)
- LLM generates prose from verified facts only
- Verification catches remaining errors

**Summary:**
- Critique-refine for: Style, structure, completeness, clarity
- Verification for: Factual accuracy, existence checks
- Hybrid extraction for: Code documentation, structured data analysis

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
    #:model "curie"  ;; Cheap critic
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
                   #:model "gpt-4"  ;; Expensive generator
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
  #:model "gpt-4"
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
1. **Cheap critic, expensive generator:** curie for critique ($0.15/1M), gpt-4 for generation ($2.50/1M). Critique is 90% of quality signal at 6% of cost.
2. **Early stopping:** Check severity after each critique. If avg < 1.5, stop (no point refining perfection).
3. **Structured critique:** Use `#:json #t` to force specific categories. Unstructured critique is vague ("needs improvement").
4. **Temperature tuning:** Generator 0.4-0.6 (creative), critic 0.0 (consistent standards).
5. **Max tokens on generator:** Cap at 2× initial draft to prevent runaway verbosity.

### Common Mistakes
❌ Using expensive model for both critic and generator
```scheme
;; Wasteful: gpt-4 for critique
(llm-query #:instruction "Critique this..." #:model "gpt-4")
;; Fix: curie is 94% as good at critique for 6% of cost
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

### Beyond Refinement: Use Critique for Strategy Comparison

**The critique primitive is more powerful than just refinement loops.** Use it to compare different approaches:

#### Example: Choosing Between Two Methods
```scheme
;; Generate solutions using two different strategies
(define method-a (syntax-e (llm-query
  #:instruction "Solve using dynamic programming"
  #:data problem-description
  #:model "gpt-4")))

(define method-b (syntax-e (llm-query
  #:instruction "Solve using greedy algorithm"
  #:data problem-description
  #:model "gpt-4")))

;; Use critique to compare them
(define comparison (syntax-e (llm-query
  #:instruction "Compare these two algorithmic approaches on:
1. Correctness (does it handle all edge cases?)
2. Efficiency (time/space complexity)
3. Clarity (maintainability)
Return JSON: {\"winner\": \"a\"|\"b\", \"reasoning\": str, \"scores\": {...}}"
  #:data (string-append "METHOD A:\n" method-a "\n\nMETHOD B:\n" method-b)
  #:json #t
  #:model "gpt-4")))

;; Parse winner
(py-set! "cmp" comparison)
(define winner (py-eval "json.loads(cmp)['winner']"))

(finish (if (string=? winner "a") method-a method-b))
```

**Key insight:** Critique isn't just for iterative refinement — it's a **general-purpose comparison operator** for evaluating alternative strategies.

**More creative uses:**
- **A/B testing patterns**: Compare fan-out vs tree-aggregation on sample data
- **Adaptive routing**: Critique sample output to decide which model to use for remaining data
- **Search validation**: Critique each backtracking branch to prune bad paths early

See `get_creative_orchestration_guide()` for more examples.

---

