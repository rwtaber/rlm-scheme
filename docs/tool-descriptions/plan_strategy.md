Design an optimal orchestration strategy for your task by analyzing requirements and recommending pattern compositions.

**When to use:**
- Starting a new orchestration task
- Unsure which patterns to compose
- Want to see cost/latency/quality tradeoffs
- Need creative alternatives

**Philosophy:**
The planner encourages creative risk-taking:
- Shows 3 tiers: recommended, alternatives, experimental
- Testing costs $0.01-0.05; wrong choices cost $1-5
- Real experiments beat theoretical analysis
- Combines 1-4 patterns for optimal results

**Input:**
- task_description: What you want to accomplish (required)
- data_characteristics: Optional data details (size, structure, format)
- constraints: Optional constraints (latency/cost/quality requirements)
- priority: "speed", "cost", "quality", or "balanced" (default)

**Output:**
JSON with:
- **recommended**: Primary strategy with pattern composition, models, cost estimates, implementation steps, code template
- **alternatives**: 2 alternative approaches with different tradeoffs
- **creative_options**: 1-2 experimental approaches with risk/upside analysis
- **recommendations**: Practical next steps

**Cost:** $0.01-0.10 per planning call (curie or gpt-4)
**ROI:** Typically saves 10-200× planning cost by choosing optimal strategy

**Examples:**
```python
# Example 1: Document analysis
plan_strategy(
    "Analyze 200 research papers and extract key findings",
    data_characteristics="~5KB per paper, ~1MB total",
    priority="balanced"
)

# Example 2: Code generation
plan_strategy(
    "Generate test cases for REST API with 20 endpoints",
    constraints="Highest quality needed",
    priority="quality"
)

# Example 3: Large dataset processing
plan_strategy(
    "Process 500KB CSV of genomic data for AMR analysis",
    data_characteristics="CSV, 10K rows × 50 columns",
    priority="speed"
)
```

**Workflow:**
1. Call plan_strategy() to get recommendations
2. Review recommended strategy and alternatives
3. Implement with execute_scheme() using provided code template
4. If unsatisfied, try creative_options or alternatives
5. For pattern details, see get_usage_guide()
