Design an optimal orchestration strategy for your task by analyzing requirements and recommending combinator compositions.

**When to use:**
- Starting a new orchestration task
- Unsure which combinators to compose
- Want to see cost/latency/quality tradeoffs
- Need creative alternatives

**Philosophy - Encourages Creative Risk-Taking:**
The planner is designed to generate **creative, novel strategies** - not just safe templates:
- Shows 3 tiers: recommended, alternatives, **genuinely experimental** options
- Testing 3 approaches costs $0.01-0.05; wrong choices cost $1-5
- **Real experiments beat theoretical analysis**
- Creative compositions often outperform safe patterns
- No single "right" approach - encourages exploration

**Input:**
- task_description: What you want to accomplish (required)
- data_characteristics: Optional data details (size, structure, format)
- constraints: Optional constraints (latency/cost/quality requirements)
- priority: "speed", "cost", "quality", or "balanced" (default)

**Output:**
JSON with:
- **recommended**: Primary strategy with combinator composition, models, cost estimates, executable code template
- **alternatives**: 2 alternative approaches with different tradeoffs
- **creative_options**: 1-2 genuinely experimental approaches - unconventional, risky, novel compositions with high potential upside
- **recommendations**: Practical next steps

**Creative Options Are Genuinely Creative:**
The creative_options field contains EXPERIMENTAL strategies - not just variations on standard patterns:
- Novel combinator compositions not shown in docs
- Unconventional approaches worth trying despite risk
- High-upside ideas that could outperform safe patterns
- Strategies designed for exploration, not production

**Cost:** $0.01-0.10 per planning call (uses gpt-4o-mini or gpt-4o)
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
2. Review recommended, alternatives, AND creative_options
3. **Test multiple strategies** - experimentation is cheap
4. Execute with execute_scheme() using provided code templates
5. For combinator details, call get_combinator_reference()

**Key Difference from Traditional Planners:**
This planner is designed to **push boundaries and encourage experimentation**, not just recommend safe patterns. The creative_options are genuinely risky/novel - try them!
