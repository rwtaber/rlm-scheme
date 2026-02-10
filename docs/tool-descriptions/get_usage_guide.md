Creative LLM orchestration guide - patterns are primitives you compose, not prescriptions.

**Philosophy**: Think vocabulary, not cookbook. The 16 patterns are building blocks. The most interesting
solutions combine 3-4 patterns in novel ways. Your job: learn the vocabulary, then speak creatively.

**Immediate techniques to encourage risk-taking:**
- Strategy Exploration: Test 3 approaches in parallel, let cheap model choose winner
- A/B Testing: Try 2 patterns on 10-20% sample, compare empirically before scaling
- Meta-Orchestration: Let model inspect data and design the optimal strategy
- Mix cheap/expensive aggressively: ada/gpt-3.5-turbo for 80%, gpt-4 for 20%

**Sections:**
- Quick Start: Composition-first mindset, immediate creative techniques
- 16 Patterns: Speed, Quality, Cost, Structure, Specialized (all presented equally)
- Creative Composition: 5 meta-patterns (strategy exploration, speculative ensemble, critique-driven backtracking, active ensemble, memoized hedging)
- When to Experiment vs Keep Simple
- Thinking in Orchestration: Mental models, creative process
- Quick Reference: Primitives, common patterns, composition table

For detailed implementations with complete code, call get_pattern_details([pattern_ids]).
