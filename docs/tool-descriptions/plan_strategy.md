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
- **scale**: "minimal", "small", "medium" (default), "large", or "comprehensive" (Phase 1)
- **min_outputs**: Minimum number of output artifacts required (Phase 1)
- **coverage_target**: Coverage requirement, e.g., "all files", "100%", "public APIs only" (Phase 1)

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

**Cost:** $0.10-0.40 per planning call (uses gpt-4o with 16K max tokens)
**ROI:** Typically saves 10-200× planning cost by choosing optimal strategy
**Why max tokens:** Thoroughness matters more than token cost - better strategies save far more than a few extra tokens

**Phase 1 Improvements (Scale Parameters):**
- Explicit **scale** parameter prevents under-scoping
- **min_outputs** ensures strategy produces enough artifacts
- **coverage_target** clarifies what "comprehensive" means
- Increased token limits (15K-20K) allow thorough planning
- Upgraded default model to gpt-4o for better quality

**Iterative Refinement:**
For ambiguous or complex tasks, call plan_strategy() multiple times with refined inputs:
1. Start with initial task description
2. Review the recommendations
3. If the strategy doesn't match expectations, call again with:
   - More specific task_description
   - Adjusted scale/min_outputs/coverage_target parameters
   - Additional constraints or data_characteristics
4. Each call provides fresh perspective with ~$0.10-0.40 cost

**Example iterative workflow:**
```python
# First call with initial understanding
result1 = plan_strategy("Document the repository", scale="medium")

# After reviewing output, refine with specifics
result2 = plan_strategy(
    "Document all Python modules in repository",
    data_characteristics="500 Python files, ~50KB each",
    scale="comprehensive",
    min_outputs=500,
    coverage_target="all files"
)
```

**Examples:**

**Example 1: Document analysis with explicit scale (Phase 1)**
```python
plan_strategy(
    "Analyze 200 research papers and extract key findings",
    data_characteristics="~5KB per paper, ~1MB total",
    scale="large",  # NEW
    min_outputs=200,  # NEW: Ensure 200 analyses
    coverage_target="all papers",  # NEW
    priority="balanced"
)

# Example 2: Code generation with coverage
plan_strategy(
    "Generate test cases for REST API with 20 endpoints",
    scale="medium",
    min_outputs=20,  # At least one test per endpoint
    coverage_target="all endpoints",
    priority="quality"
)

# Example 3: Comprehensive documentation
plan_strategy(
    "Document large repository",
    data_characteristics="500 Python files",
    scale="comprehensive",  # Process ALL files
    min_outputs=500,  # One doc per file
    coverage_target="all files",
    priority="balanced"
)

# Example 4: Quick overview (minimal scale)
plan_strategy(
    "Summarize key modules in codebase",
    scale="minimal",  # Just highlights
    min_outputs=5,  # Top 5 modules
    priority="speed"
)
```

**Scale Guidelines:**
- **minimal**: Proof of concept, ~5-10 outputs
- **small**: ~10-30 outputs, major components
- **medium**: ~30-100 outputs, comprehensive but selective (default)
- **large**: ~100-500 outputs, near-complete coverage
- **comprehensive**: Complete coverage, no gaps

**Workflow:**
1. Call plan_strategy() to get recommendations
2. Review recommended, alternatives, AND creative_options
3. **Test multiple strategies** - experimentation is cheap
4. Execute with execute_scheme() using provided code templates
5. For combinator details, call get_combinator_reference()

**Key Difference from Traditional Planners:**
This planner is designed to **push boundaries and encourage experimentation**, not just recommend safe patterns. The creative_options are genuinely risky/novel - try them!
