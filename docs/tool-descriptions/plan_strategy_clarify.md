Analyze a task and identify clarifying questions before planning (Phase 2: Multi-turn planning).

**When to use:**
- Task description is ambiguous about scale/scope/requirements
- "Large repo" or "comprehensive" without specifics
- Unclear how many items to process or outputs to generate
- Want to ensure strategy matches user intent before planning

**When NOT to use:**
- Task is already clear and well-specified
- Scale, outputs, and coverage are explicit
- Simple, straightforward tasks
- User has provided detailed requirements

**Workflow:**
1. Call `plan_strategy_clarify()` with task description
2. Review returned `clarity_assessment` and `ambiguities_found`
3. If `is_clear: true`, skip to `plan_strategy()` directly
4. If `is_clear: false`, ask user the `recommended_clarifications`
5. Collect user answers
6. Call `plan_strategy_finalize()` with answers as `clarifications` parameter

**Input:**
- task_description: What you want to accomplish (required)
- data_characteristics: Optional data details
- constraints: Optional constraints
- priority: "speed", "cost", "quality", or "balanced" (default)

**Output:**
JSON with:
- **clarity_assessment**: Is task clear? (is_clear: bool, confidence: 0-1, reasoning: string)
- **ambiguities_found**: List of specific ambiguities requiring clarification
  - category: "scale", "output", "scope", or "constraint"
  - question: Specific clarifying question
  - why_matters: How this affects strategy design
  - suggested_options: Possible answers
- **assumptions**: What we'll assume if user doesn't clarify (with risks)
- **recommended_clarifications**: Clear questions to ask the user

**Cost:** ~$0.10-0.20 per analysis (uses gpt-4o with 16K max tokens)
**Why max tokens:** Thorough ambiguity analysis prevents expensive mistakes - worth every token

**Examples:**

**Example 1: Ambiguous Task**
```python
result = plan_strategy_clarify(
    "Document the large repository",
    priority="balanced"
)
# Returns: is_clear=false, questions about file count, format, coverage level
```

**Example 2: Clear Task**
```python
result = plan_strategy_clarify(
    "Process 500 documents (5KB each), extract key findings, produce JSON summary. Budget <$5",
    priority="balanced"
)
# Returns: is_clear=true, no clarifications needed → proceed directly to plan_strategy()
```

**Tips:**
- Check `is_clear` before asking user questions
- If confidence > 0.8 and is_clear=true, skip clarification phase
- Present `suggested_options` as multiple choice when possible
- `assumptions` field shows what happens if user doesn't clarify (useful fallback)

**Why Use This vs. Single-Shot Planning:**
- ✅ Prevents expensive mistakes (wrong scale/scope costs $1-5+ to fix)
- ✅ Explicit about assumptions before committing to strategy
- ✅ Higher quality plans through user clarification
- ✅ Worth the extra $0.10 and 5 seconds for ambiguous tasks
- ⚠️ Only adds value when task is genuinely ambiguous

**Integration:**
- Works with `plan_strategy_finalize()` for two-stage planning
- Can bypass and use `plan_strategy()` directly if task is clear
- Claude Code can use AskUserQuestion tool to collect clarifications

**See Also:** "Planning Workflows" section in usage guide for when to use each approach
