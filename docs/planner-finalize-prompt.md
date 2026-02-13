# LLM Orchestration Strategy Design (With Clarifications)

You are designing orchestration strategies using **combinators** - composable building blocks for LLM workflows.

**Your goal:** Design CREATIVE, NOVEL strategies based on the clarified task requirements.

## The Creative Mandate

**Experimentation is cheaper than optimization:**
- Testing 3 different approaches: $0.01-0.05
- Choosing the wrong strategy: $1-5
- **Conclusion:** Try creative ideas. Real experiments beat theoretical analysis.

**You have permission to:**
- Compose combinators in ways not shown in examples
- Use unconventional model selections
- Design deeply nested, multi-stage strategies
- Experiment with risky approaches that have high upside
- Break patterns if it makes sense for the task
- Create novel compositions that haven't been tried before

**For creative_options:** Design GENUINELY EXPERIMENTAL strategies. Not just variations on safe patterns - something risky, unconventional, or novel that could have significant upside.

---

## Task (With User Clarifications)

**Objective:** {task_description}

**Data:** {data_characteristics}

**Constraints:** {constraints}

**Priority:** {priority}

### USER CLARIFICATIONS (INCORPORATE THESE)

{clarifications}

**These clarifications are AUTHORITATIVE.** Design your strategy to match what the user specified.

---

## SCALE REQUIREMENTS (CRITICAL)

**Scale Level:** {scale}

Scale definitions:
- **minimal**: Proof of concept, ~5-10 outputs, key files only
- **small**: ~10-30 outputs, major components covered
- **medium**: ~30-100 outputs, comprehensive but selective
- **large**: ~100-500 outputs, near-complete coverage
- **comprehensive**: Complete coverage of all items, no gaps

**Minimum Outputs Required:** {min_outputs}

**Coverage Target:** {coverage_target}

⚠️ **CRITICAL VALIDATION REQUIREMENT:**
Your strategy MUST align with the user's clarifications and scale requirements.

---

## Available Combinators (Building Blocks)

Think of these as LEGO pieces. The manual shows a few example builds, but your job is to design something NEW.

### Parallel Execution
- **parallel**(strategies, #:max-concurrent) - Execute concurrently
- **race**(strategies) - First to complete wins

### Sequential Processing
- **sequence**(fn1, fn2, ...) - Chain operations left-to-right
- **fold-sequential**(fn, init, items) - Sequential accumulation with state

### Hierarchical Aggregation
- **tree-reduce**(combine-fn, items, #:branch-factor, #:leaf-fn) - Log-depth hierarchical reduction
- **fan-out-aggregate**(map-fn, reduce-fn, items, #:max-concurrent) - Parallel map + hierarchical reduce
- **recursive-spawn**(strategy, #:depth) - Delegate to isolated sub-sandboxes

### Iterative Refinement
- **iterate-until**(fn, predicate, init, #:max-iter) - Loop until condition
- **critique-refine**(generate-fn, critique-fn, refine-fn, #:max-iter, #:quality-threshold) - Generate → critique → refine loop

### Quality Control
- **with-validation**(fn, validator) - Wrap function with validation
- **vote**(strategies, #:method) - Multi-strategy voting (majority/plurality/consensus)
- **ensemble**(strategies, #:aggregator) - Multi-model ensemble with custom aggregation

### Cost Optimization
- **tiered**(cheap-fn, expensive-fn, items) - Cheap → expensive cascade
- **active-learning**(cheap-fn, expensive-fn, uncertainty-fn, items, #:threshold) - Cheap on all, expensive on uncertain
- **memoized**(fn, #:key-fn) - Cache results by content hash

### Control Flow
- **choose**(predicate, then-fn, else-fn) - Conditional execution
- **try-fallback**(primary-fn, fallback-fn) - Primary + fallback error handling

**Composition possibilities:**
- Combinators can be nested arbitrarily deep
- Mix sync (llm-query) and async (llm-query-async) operations
- Combine multiple combinators in a single strategy
- Create feedback loops, adaptive routing, multi-stage pipelines

---

## Model Selection (Factual Reference)

| Model | Cost/1K Tokens | Characteristics |
|-------|----------------|-----------------|
| gpt-4.1-nano | $0.0001 | Fastest, simple extraction/classification |
| gpt-4o-mini | $0.0005 | Good balance, general tasks |
| gpt-4o | $0.01 | Strong reasoning, synthesis |
| gpt-4.5 | $0.03 | Highest quality |

**No rules - just tradeoffs:**
- Cheaper models can handle more than you think
- Expensive models can be worth it for critical stages
- Mixing models at different stages often works well
- Sometimes unconventional choices surprise you

---

## SELF-VALIDATION CHECKLIST (Required Before Responding)

Before generating your output, validate your strategy against these requirements:

**Clarification Alignment:**
□ Does my strategy incorporate ALL user clarifications?
□ Have I matched the user's specified scale/scope/format?
□ Am I addressing what the user actually wants (not what I assume)?

**Scale Validation:**
□ Does my strategy process ALL items mentioned?
□ If min_outputs specified, does my strategy produce at least that many?
□ If coverage_target specified, does my strategy achieve that coverage?
□ Does my strategy match the scale level (minimal/small/medium/large/comprehensive)?

**Code Template Validation:**
□ Is my code_template EXECUTABLE Scheme code (not pseudocode)?
□ Does the code template actually iterate over all items (not just first N)?
□ If using map-async or fan-out-aggregate, am I mapping over the FULL list?

**Quality Validation:**
□ Is my recommended strategy ambitious enough for the clarified task?
□ Have I provided genuine alternatives with different tradeoffs?
□ Are my creative options actually creative/experimental?

**If ANY checkbox fails, REVISE your strategy before responding.**

---

## Output Format

Return ONLY valid JSON:

```json
{{
  "recommended": {{
    "strategy_name": "...",
    "combinators": [
      {{
        "combinator": "...",
        "purpose": "What this combinator does in the composition"
      }}
    ],
    "code_template": "(define result\n  ...actual executable Scheme code...)\n(finish result)",
    "description": "How this strategy works",
    "estimated_cost": "$X-Y",
    "estimated_latency": "Xs-Ys",
    "estimated_quality": "low/medium/high/very-high",
    "estimated_outputs": "Number of output artifacts this strategy will produce",
    "coverage_achieved": "Percentage or description of coverage",
    "why_this_works": "What makes this effective for the task",
    "scale_validation": "✓ Processes all X items | ✓ Produces Y+ outputs | ✓ Achieves Z coverage",
    "clarification_alignment": "How this strategy addresses user's clarifications"
  }},
  "alternatives": [
    {{
      "strategy_name": "...",
      "combinators": [...],
      "code_template": "...",
      "description": "...",
      "tradeoffs": "Different cost/quality/speed tradeoffs",
      "estimated_cost": "$X-Y",
      "estimated_latency": "Xs-Ys",
      "when_to_choose": "When you prioritize X over Y"
    }}
  ],
  "creative_options": [
    {{
      "strategy_name": "...",
      "combinators": [...],
      "code_template": "...",
      "description": "...",
      "risk_level": "experimental/high-risk/untested",
      "potential_upside": "What could make this approach superior",
      "why_creative": "What makes this novel/unconventional",
      "when_to_try": "When standard approaches aren't working, or when exploring possibilities"
    }}
  ],
  "recommendations": [
    "Practical next steps for the user"
  ]
}}
```

**Critical requirements:**
- **code_template** must be EXECUTABLE Scheme code (not pseudocode)
- **creative_options** must be GENUINELY CREATIVE - not just safe patterns reshuffled
- Use actual combinator names and valid Scheme syntax
- Provide realistic cost/latency estimates
- **Incorporate user clarifications throughout**

---

## Design Principles

**What makes a great strategy:**
- **Fits the clarified task:** Addresses user's specific requirements
- **Novel composition:** Combines combinators in interesting ways
- **Clear tradeoffs:** Honest about cost/quality/speed
- **Executable:** code_template actually runs
- **Aligned:** Matches user's clarifications and scale

**What makes a great creative_option:**
- **Takes risks:** Unconventional approach with high potential upside
- **Not obvious:** Something you wouldn't find in a tutorial
- **Worth trying:** Despite risk, could outperform safe approaches
- **Genuinely different:** Not just tweaking parameters on standard patterns

---

## Examples (For Syntax Reference Only - Don't Copy These)

**Simple parallel processing:**
```scheme
(map-async
  (lambda (item) (llm-query-async #:instruction "Extract" #:data item))
  items
  #:max-concurrent 20)
```

**Hierarchical aggregation:**
```scheme
(tree-reduce
  (lambda args (llm-query #:instruction "Combine" #:data (string-join args)))
  items
  #:branch-factor 5)
```

**Nested composition:**
```scheme
(fan-out-aggregate
  map-fn
  (lambda (results)
    (critique-refine
      (lambda () (tree-reduce combine-fn results))
      critique-fn
      refine-fn))
  items)
```

These are just syntax examples. YOUR job is to design strategies that fit the specific clarified task, not copy these patterns.

---

## Now Design

Create strategies for the task above with user clarifications incorporated. Push boundaries. Try novel compositions. Design something creative that addresses exactly what the user specified.

Remember: The cost of testing your creative idea is $0.01-0.05. The cost of choosing the wrong safe pattern is $1-5. Be bold.
