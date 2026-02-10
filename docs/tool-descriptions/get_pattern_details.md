Get detailed documentation for specific orchestration patterns with complete code examples.

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
