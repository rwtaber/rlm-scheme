Generate final orchestration strategy with user clarifications incorporated (Phase 2: Multi-turn planning).

**When to use:**
- After `plan_strategy_clarify()` identified ambiguities
- User has provided answers to clarifying questions
- Ready to generate final strategy with resolved ambiguities

**When NOT to use:**
- Task was clear from the start (use `plan_strategy()` instead)
- Haven't called `plan_strategy_clarify()` yet
- User hasn't provided clarifications

**Workflow:**
1. Called `plan_strategy_clarify()` first
2. Received `recommended_clarifications` and asked user
3. User provided answers
4. Format answers as clear text/JSON string
5. Call `plan_strategy_finalize()` with clarifications

**Input:**
- task_description: Original task description (required)
- clarifications: User's answers to clarifying questions (required, string)
- data_characteristics: Optional data details
- constraints: Optional constraints
- priority: "speed", "cost", "quality", or "balanced" (default)
- scale: "minimal", "small", "medium", "large", "comprehensive" (Phase 1)
- min_outputs: Minimum number of output artifacts
- coverage_target: Coverage requirement (e.g., "100%", "all files")

**Output:**
JSON with (same structure as `plan_strategy`):
- **recommended**: Primary strategy with combinator composition, code template, cost estimates
- **alternatives**: 2 alternative approaches with different tradeoffs
- **creative_options**: 1-2 experimental/novel approaches
- **recommendations**: Practical next steps
- **_meta**: Includes clarifications_incorporated: true

**Cost:** $0.10-0.40 per planning call (uses gpt-4o with 16K max tokens)
**Total Two-Phase Cost:** ~$0.20-0.60 (clarify + finalize)
**Why Worth It:** Better alignment prevents $1-5+ wasted executions from wrong scope

**Examples:**

**Example 1: Documentation with Clarifications**
```python
# Step 1: Clarify
clarify_result = plan_strategy_clarify("Document the large repository")
# Returns: Questions about file count, format, coverage

# Step 2: User answers (via Claude Code AskUserQuestion)
user_answers = """
File count: 500 files
Format: API reference documentation
Coverage: All files (comprehensive)
"""

# Step 3: Finalize with clarifications
plan = plan_strategy_finalize(
    task_description="Document the large repository",
    clarifications=user_answers,
    scale="comprehensive",
    min_outputs=500,
    coverage_target="all files",
    priority="balanced"
)
# Returns: Strategy that processes all 500 files, produces 500 API docs
```

**Example 2: Analysis with Constraints**
```python
clarifications = """
Dataset size: 200 research papers (1MB total)
Analysis depth: Extract key findings + synthesize themes
Format: Structured JSON with categories
Budget: <$2
"""

plan = plan_strategy_finalize(
    task_description="Analyze research papers for antimicrobial resistance",
    clarifications=clarifications,
    scale="large",
    min_outputs=200,
    priority="cost"
)
```

**Clarifications Format:**
Can be:
- Plain text with user's answers
- Key-value pairs
- JSON string
- Conversational responses

**The planner will extract relevant information from any format.**

**Tips:**
- Include all relevant user answers in clarifications parameter
- Set scale/min_outputs/coverage_target based on user's answers
- The strategy will explicitly address user's clarifications
- Output includes "clarification_alignment" field showing how strategy addresses requirements

**Integration:**
- Second stage of two-tool workflow with `plan_strategy_clarify()`
- For clear tasks, use `plan_strategy()` directly instead
- Output format identical to `plan_strategy()` for consistency
