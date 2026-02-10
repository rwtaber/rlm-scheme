Execute Scheme orchestration code. Think vocabulary, not cookbook - compose patterns creatively.

TAKE RISKS - Experiment with strategies:
- **Test 3 approaches, let cheap model choose**: Launch parallel strategies, critique to compare (testing on 10-20% costs $0.01-0.05, wrong choice costs $1-5)
- **Mix cheap/expensive aggressively**: Use nano ($0.10/1M) for 80% of work, expensive ($2.50/1M) for 20%
- **Compose patterns freely**: Ensemble + Critique, Hedging + Voting, Fan-out + Tree-reduce
- **Let models guide**: Inspect sample data, recommend strategy dynamically
- **A/B test when uncertain**: Compare patterns empirically on samples before scaling

16 PATTERNS (primitives you compose - all presented equally):

Speed: Parallel Fan-Out, Speculative Execution, Stream Processing
Quality: Critique-Refine, Ensemble Voting, Consensus Protocol
Cost: Active Learning, Memoization, Multi-Armed Bandit
Structure: Code Generation, Meta-Orchestration, Recursive Delegation, Tree Aggregation
Specialized: Cumulative Fold, Backtracking Search, Anytime Algorithms

CREATIVE COMPOSITIONS (not in the 16 patterns - you create these):
- Strategy Exploration: Test 3 approaches in parallel → cheap model compares → use winner
- Speculative Ensemble: Hedge (parallel, take first) + Ensemble (use all, vote) = speed + quality
- Critique-Driven Backtracking: Try strategy → cheap critique validates → backtrack with feedback
- Active Ensemble: Cheap on all → ensemble only on uncertain 20% → 95% at 2× (not 5×)
- Memoized Hedging: Check cache → if miss, hedge 3 approaches → cache winner

MODEL COSTS (exploit the 50-100× difference):
- ada: $0.0004/1K (simplest - classification, filtering)
- gpt-3.5-turbo: $0.002/1K (fan-out, extraction, bulk processing)
- curie: $0.002-0.03/1K (critique, comparison, summarization)
- gpt-4: $0.03-0.06/1K (synthesis, complex reasoning - expensive!)
- code-davinci-002: $0.02-0.05/1K (code generation)

Call get_usage_guide() for pattern overview and primitive reference.
Call plan_strategy(task_description) for strategy recommendations tailored to your task.

State persists across execute_scheme calls until reset(). Capabilities: LLM orchestration with scope tracking, Python integration (py-exec, file I/O), image processing (#:images), web requests (Python urllib/requests), parallel work (map-async).

Args:
    code: Scheme code to execute
    timeout: Optional timeout in seconds (default: 300s from RLM_TIMEOUT_SECONDS env)
    ctx: MCP context for progress reporting
