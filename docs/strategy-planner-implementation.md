# Strategy Planner Implementation Plan

## Overview

A new MCP tool that helps users design optimal orchestration strategies by analyzing their task requirements and recommending pattern compositions from the documented 16 patterns.

**Core Philosophy**: The planner should encourage creative risk-taking while providing sound guidance. It presents multiple approaches (conservative, balanced, experimental) and shows tradeoffs explicitly.

## 1. Architecture

### Option A: LLM-Powered Planner (RECOMMENDED)

**How it works:**
1. Extract structured knowledge about all 16 patterns at server startup
2. When user calls `plan_strategy()`, construct a prompt with:
   - Pattern knowledge (condensed summaries)
   - User's task description and constraints
   - Model pricing information
   - Examples of creative compositions
3. Send to gpt-4 or curie with structured output format
4. Return JSON with strategy recommendations

**Pros:**
- Handles novel task types flexibly
- Can reason about pattern compositions creatively
- Adapts to user constraints naturally
- Learns from pattern documentation improvements

**Cons:**
- Costs $0.01-0.10 per planning call
- Adds latency (2-5 seconds)

**Cost Analysis:**
- Planning call: $0.01-0.10 (curie or gpt-4)
- Potential savings: $1-20 (by choosing optimal strategy)
- **ROI: 10-200× return on planning investment**

### Option B: Rule-Based Planner

**How it works:**
- Hard-coded decision tree based on task characteristics
- Pattern matching on keywords (e.g., "analyze 100 documents" → Pattern 1)
- Simpler but less flexible

**Pros:**
- Zero cost per call
- Fast (<100ms)

**Cons:**
- Brittle for novel tasks
- Requires manual updates for new patterns
- Can't reason about creative compositions
- Less likely to encourage risk-taking

### Option C: Hybrid Approach

**How it works:**
- Rule-based for common patterns (90% of cases)
- Falls back to LLM for complex/novel tasks

**Recommendation:** Start with **Option A** (LLM-powered). The cost is negligible compared to potential savings, and it aligns with the project's philosophy of creative orchestration.

## 2. Pattern Knowledge Extraction

### Startup: Build Pattern Knowledge Base

At server initialization, extract structured information from all pattern docs:

```python
import json
import re
from pathlib import Path

def _extract_pattern_knowledge() -> dict[int, dict]:
    """Extract structured knowledge from pattern docs."""
    pattern_dir = Path(_DOCS_DIR) / "patterns"
    patterns = {}

    for pattern_file in sorted(pattern_dir.glob("pattern-*.md")):
        # Extract pattern number from filename
        match = re.match(r"pattern-(\d+)-(.+)\.md", pattern_file.name)
        if not match:
            continue

        pattern_num = int(match.group(1))
        pattern_slug = match.group(2)

        content = pattern_file.read_text(encoding="utf-8")

        # Extract sections using regex
        title = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        when_to_use = re.search(r"## When to Use\n\n(.+?)(?=\n##|\Z)", content, re.DOTALL)
        improvements = re.search(r"## Improvements Over Baseline\n\n(.+?)(?=\n##|\Z)", content, re.DOTALL)
        cost = re.search(r"## Cost Analysis\n\n(.+?)(?=\n##|\Z)", content, re.DOTALL)

        patterns[pattern_num] = {
            "id": pattern_num,
            "slug": pattern_slug,
            "title": title.group(1) if title else f"Pattern {pattern_num}",
            "when_to_use": when_to_use.group(1).strip() if when_to_use else "",
            "improvements": improvements.group(1).strip() if improvements else "",
            "cost": cost.group(1).strip() if cost else "",
        }

    return patterns

# Load at startup
_PATTERN_KNOWLEDGE = _extract_pattern_knowledge()
```

### Condensed Pattern Summary for Prompts

Create a condensed summary (500-1000 tokens) suitable for including in planner prompts:

```python
def _create_pattern_summary() -> str:
    """Create condensed pattern summary for planner prompts."""
    lines = ["# Available Orchestration Patterns\n"]

    for pid, info in sorted(_PATTERN_KNOWLEDGE.items()):
        lines.append(f"\n## Pattern {pid}: {info['title']}")
        lines.append(f"\n{info['when_to_use'][:200]}...")  # First 200 chars

        # Extract key metrics if available
        if "latency" in info['improvements'].lower():
            lines.append("- **Strength**: Latency optimization")
        if "cost" in info['improvements'].lower() or "cheap" in info['improvements'].lower():
            lines.append("- **Strength**: Cost optimization")
        if "quality" in info['improvements'].lower():
            lines.append("- **Strength**: Quality optimization")

    return "\n".join(lines)

_PATTERN_SUMMARY = _create_pattern_summary()
```

## 3. Planner Tool Interface

### Input Parameters

```python
@mcp.tool()
def plan_strategy(
    task_description: str,
    data_characteristics: Optional[str] = None,
    constraints: Optional[str] = None,
    priority: str = "balanced",
) -> str:
    """
    Design an optimal orchestration strategy for your task.

    The planner analyzes your requirements and recommends pattern compositions
    from the 16 documented patterns, showing multiple approaches with tradeoffs.

    Args:
        task_description: What you want to accomplish (e.g., "Analyze 200 research papers
                          and extract key findings", "Generate code for 10 API endpoints")
        data_characteristics: Optional details about your data (e.g., "100KB total,
                              structured JSON", "Large text documents, ~5KB each")
        constraints: Optional constraints (e.g., "Must complete in <30s", "Budget <$1",
                     "Highest quality needed")
        priority: "speed", "cost", "quality", or "balanced" (default)

    Returns:
        JSON with recommended strategies:
        {
            "recommended": {
                "strategy_name": "Parallel Fan-Out with Critique",
                "patterns": [1, 4],
                "description": "...",
                "estimated_cost": "$0.50-1.00",
                "estimated_latency": "5-10s",
                "quality_score": "high",
                "implementation": ["step 1", "step 2", ...]
            },
            "alternatives": [
                {
                    "strategy_name": "Meta-Orchestration",
                    "patterns": [6],
                    "tradeoffs": "Slightly higher cost but more adaptive",
                    ...
                },
                ...
            ],
            "creative_options": [
                {
                    "strategy_name": "Speculative Ensemble with Backtracking",
                    "patterns": [7, 8, 12],
                    "risk_level": "experimental",
                    "potential_upside": "2-3× quality improvement if successful",
                    ...
                }
            ]
        }

    Cost: ~$0.01-0.10 per call (planning investment typically saves 10-200× in execution)

    Examples:
        plan_strategy("Analyze 100 customer reviews for sentiment and themes")
        plan_strategy("Generate test cases for complex API", constraints="Highest quality")
        plan_strategy("Process 500KB CSV of genomic data", priority="speed")
    """
    # Implementation below
```

### Output Format

```json
{
  "recommended": {
    "strategy_name": "Parallel Fan-Out with Critique Synthesis",
    "patterns": [1, 4, 10],
    "description": "Use Pattern 1 to extract key points from each paper in parallel with gpt-3.5-turbo ($0.002/1K), then Pattern 10 for hierarchical reduction to synthesize findings, and Pattern 4 for critique-refine on final output with gpt-4.",
    "estimated_cost": "$0.80-1.50",
    "estimated_latency": "8-15s",
    "estimated_quality": "high",
    "model_assignments": {
      "fan_out": "gpt-3.5-turbo",
      "reduction": "curie",
      "synthesis": "gpt-4"
    },
    "implementation": [
      "1. Load data with load_context()",
      "2. Chunk papers via py-exec",
      "3. Use map-async with gpt-3.5-turbo for parallel extraction",
      "4. Pattern 10: Tree aggregation to synthesize findings",
      "5. Pattern 4: Critique-refine final synthesis with gpt-4"
    ],
    "code_template": "(map-async (lambda (paper) ...) papers #:model \"gpt-3.5-turbo\")"
  },
  "alternatives": [
    {
      "strategy_name": "Meta-Orchestration (Let LLM Design Strategy)",
      "patterns": [6],
      "description": "Use Pattern 6 to let a model analyze the papers and design the optimal extraction strategy dynamically.",
      "tradeoffs": "Higher upfront cost ($0.10-0.30) but may find optimal approach for this specific corpus",
      "estimated_cost": "$1.00-2.00",
      "estimated_latency": "15-25s",
      "when_to_choose": "Unknown paper structure or highly variable content"
    },
    {
      "strategy_name": "Simple Sequential Processing",
      "patterns": [],
      "description": "Process papers sequentially with single gpt-4 call",
      "tradeoffs": "Much slower (5-10× latency) and more expensive (3-5× cost), but simpler code",
      "estimated_cost": "$3.00-6.00",
      "estimated_latency": "60-120s",
      "when_to_choose": "Only for very small datasets (<10 items)"
    }
  ],
  "creative_options": [
    {
      "strategy_name": "Speculative Execution with Ensemble Voting",
      "patterns": [7, 8],
      "description": "Launch 2-3 different extraction strategies in parallel (Pattern 7), then use ensemble voting (Pattern 8) to synthesize results",
      "risk_level": "experimental",
      "potential_upside": "2× quality improvement by combining diverse approaches",
      "estimated_cost": "$2.00-4.00",
      "estimated_latency": "10-15s",
      "when_to_try": "When quality is critical and budget allows experimentation"
    },
    {
      "strategy_name": "Active Learning with Backtracking",
      "patterns": [9, 12],
      "description": "Use cheap model on all papers, expensive model on uncertain cases (Pattern 9), with backtracking (Pattern 12) to explore alternative strategies if initial approach struggles",
      "risk_level": "moderate",
      "potential_upside": "Cost savings (50-70%) with quality preservation",
      "estimated_cost": "$0.30-0.80",
      "when_to_try": "Large datasets with variable difficulty"
    }
  ],
  "recommendations": [
    "Start with recommended strategy (Parallel Fan-Out + Critique)",
    "Test on 10-20 papers first to validate before scaling",
    "Consider creative_options if initial results need improvement",
    "Use Pattern 14 (Memoization) if you'll re-run similar analyses"
  ]
}
```

## 4. Implementation Code

### Complete Implementation

```python
import json
from typing import Optional

@mcp.tool()
def plan_strategy(
    task_description: str,
    data_characteristics: Optional[str] = None,
    constraints: Optional[str] = None,
    priority: str = "balanced",
) -> str:
    """Design an optimal orchestration strategy for your task."""

    # Construct planning prompt
    prompt = f"""You are an expert LLM orchestration architect. Given a task, recommend optimal orchestration strategies using the 16 available patterns.

# Task
{task_description}

# Data Characteristics
{data_characteristics or "Not specified"}

# Constraints
{constraints or "None specified"}

# Priority
{priority} (speed/cost/quality/balanced)

{_PATTERN_SUMMARY}

# Model Pricing (OpenAI)
- ada: $0.0004/1K (simple filtering, classification)
- gpt-3.5-turbo: $0.002/1K (general tasks, fan-out)
- curie: $0.002-0.03/1K (critique, summarization)
- gpt-4: $0.03-0.06/1K (complex reasoning, synthesis)
- code-davinci-002: $0.02-0.05/1K (code generation)

# Your Task
1. Analyze the task and recommend a PRIMARY strategy:
   - Which patterns to compose (2-4 patterns usually optimal)
   - Model assignments (use cheapest model that works)
   - Implementation steps
   - Estimated cost/latency/quality

2. Provide 2 ALTERNATIVES with different tradeoffs

3. Suggest 1-2 CREATIVE/EXPERIMENTAL options that:
   - Combine patterns in novel ways
   - Take calculated risks for potential upside
   - Clearly state risk level and when to try

4. IMPORTANT: Encourage risk-taking:
   - Testing 3 approaches costs $0.01-0.05
   - Wrong choice costs $1-5
   - Don't over-optimize for theoretical efficiency
   - Real-world experiments beat analysis

Output ONLY valid JSON following this schema:
{{
  "recommended": {{
    "strategy_name": str,
    "patterns": [int],
    "description": str,
    "estimated_cost": str,
    "estimated_latency": str,
    "estimated_quality": str,
    "model_assignments": dict,
    "implementation": [str],
    "code_template": str
  }},
  "alternatives": [{{
    "strategy_name": str,
    "patterns": [int],
    "description": str,
    "tradeoffs": str,
    "estimated_cost": str,
    "when_to_choose": str
  }}],
  "creative_options": [{{
    "strategy_name": str,
    "patterns": [int],
    "description": str,
    "risk_level": str,
    "potential_upside": str,
    "estimated_cost": str,
    "when_to_try": str
  }}],
  "recommendations": [str]
}}
"""

    # Call planner model (use curie for cost-effectiveness, gpt-4 for complex tasks)
    planner_model = "curie" if priority == "cost" else "gpt-4"

    try:
        result = _llm_call(
            prompt,
            model=planner_model,
            max_tokens=2000,
            temperature=0.7,  # Encourage creative strategies
        )

        # Validate JSON
        parsed = json.loads(result)

        # Add metadata
        parsed["_meta"] = {
            "planner_model": planner_model,
            "planning_cost": "$0.01-0.10",
            "task_analyzed": task_description[:100] + "..." if len(task_description) > 100 else task_description,
        }

        return json.dumps(parsed, indent=2)

    except json.JSONDecodeError as e:
        # Fallback: return structured error with basic recommendation
        return json.dumps({
            "error": "Failed to generate structured plan",
            "fallback_recommendation": "Start with Pattern 1 (Parallel Fan-Out) for most tasks. Call get_usage_guide() for detailed patterns.",
            "raw_output": result[:500] if 'result' in locals() else "No output generated"
        }, indent=2)
```

### Integration with Existing Tools

```python
# Add to tool description
plan_strategy.__doc__ = _TOOL_DESCRIPTIONS["plan_strategy"]

# Create docs/tool-descriptions/plan_strategy.md
```

## 5. Tool Description (plan_strategy.md)

Create `docs/tool-descriptions/plan_strategy.md`:

```markdown
Design an optimal orchestration strategy for your task by analyzing requirements and recommending pattern compositions.

**When to use:**
- Starting a new orchestration task
- Unsure which patterns to use
- Want to explore creative alternatives
- Need cost/latency/quality tradeoffs

**Philosophy:**
The planner encourages risk-taking and creative composition:
- Shows 3 approaches: recommended, alternatives, experimental
- Explicitly states tradeoffs and when to try each
- Testing costs $0.01-0.05; wrong choices cost $1-5
- Real experiments beat theoretical analysis

**Input:**
- task_description: What you want to accomplish
- data_characteristics: Optional data details (size, structure)
- constraints: Optional constraints (latency/cost/quality requirements)
- priority: "speed", "cost", "quality", or "balanced"

**Output:**
JSON with:
- Recommended strategy (pattern composition, models, cost estimates, implementation steps)
- 2 alternative approaches with tradeoffs
- 1-2 creative/experimental options
- Code templates and implementation guidance

**Cost:** $0.01-0.10 per planning call
**ROI:** Typically saves 10-200× planning cost by choosing optimal strategy

**Examples:**
```scheme
;; Example 1: Analyze documents
(plan_strategy
  "Analyze 200 research papers and extract key findings"
  #:data "~5KB per paper, ~1MB total"
  #:priority "balanced")

;; Example 2: Generate code
(plan_strategy
  "Generate test cases for REST API with 20 endpoints"
  #:constraints "Highest quality needed"
  #:priority "quality")

;; Example 3: Process large dataset
(plan_strategy
  "Process 500KB CSV of genomic data for AMR analysis"
  #:data "CSV, 10K rows × 50 columns"
  #:priority "speed")
```

**Integration:**
1. Call plan_strategy() to get recommendations
2. Review recommended strategy and alternatives
3. Use get_pattern_details([1, 4, 10]) to see full implementation examples
4. Implement with execute_scheme()
5. If results unsatisfactory, try creative_options or alternatives
```

## 6. Testing Strategy

### Test Cases

1. **Simple task** (should recommend Pattern 1 or direct approach)
   ```
   Task: "Summarize 10 news articles"
   Expected: Simple fan-out with gpt-3.5-turbo
   ```

2. **Complex reasoning** (should recommend Pattern 4 or 8)
   ```
   Task: "Compare 3 policy proposals and write analysis"
   Expected: Critique-refine or ensemble voting
   ```

3. **Large dataset** (should recommend Pattern 1 + 10)
   ```
   Task: "Analyze 1000 customer reviews"
   Expected: Parallel fan-out + tree aggregation
   ```

4. **Unknown structure** (should recommend Pattern 2 or 6)
   ```
   Task: "Analyze complex genomic dataset I'm uploading"
   Expected: Code generation or meta-orchestration
   ```

5. **Cost-constrained** (should recommend ada/gpt-3.5-turbo + Pattern 9)
   ```
   Task: "Process 10K documents"
   Constraints: "Budget <$5"
   Expected: Active learning with cheap models
   ```

6. **Latency-constrained** (should recommend Pattern 7 or speculative)
   ```
   Task: "Real-time analysis of streaming data"
   Constraints: "Must complete <5s"
   Expected: Speculative execution or stream processing
   ```

### Validation

```python
def test_planner():
    """Test planner with sample tasks."""
    test_cases = [
        ("Summarize 10 news articles", "balanced", [1]),
        ("Analyze 1000 reviews", "balanced", [1, 10]),
        ("Compare 3 proposals", "quality", [4, 8]),
    ]

    for task, priority, expected_patterns in test_cases:
        result = plan_strategy(task, priority=priority)
        parsed = json.loads(result)
        recommended = parsed["recommended"]["patterns"]

        print(f"Task: {task}")
        print(f"Expected: {expected_patterns}")
        print(f"Got: {recommended}")
        print(f"Match: {set(recommended) & set(expected_patterns)}")
        print()
```

## 7. Improvements and Future Work

### Phase 1 (MVP)
- [x] Extract pattern knowledge at startup
- [ ] Implement plan_strategy() tool
- [ ] Create tool description markdown
- [ ] Test with 6 representative tasks
- [ ] Add to MCP server

### Phase 2 (Enhancements)
- [ ] Add learning: Track which recommended strategies users actually implement
- [ ] Add feedback: Allow users to rate strategy effectiveness
- [ ] Pattern usage analytics: Which patterns are most recommended
- [ ] Refine prompts based on feedback

### Phase 3 (Advanced)
- [ ] Multi-turn planning: Allow user to refine plan iteratively
- [ ] Cost estimation model: More accurate cost predictions
- [ ] Benchmark database: Track actual costs/latency for different strategies
- [ ] Auto-tuning: Adjust recommendations based on historical performance

## 8. Integration Checklist

- [ ] Add `_extract_pattern_knowledge()` to mcp_server.py startup
- [ ] Add `_create_pattern_summary()` to mcp_server.py
- [ ] Implement `plan_strategy()` function
- [ ] Create `docs/tool-descriptions/plan_strategy.md`
- [ ] Add `_TOOL_DESCRIPTIONS["plan_strategy"]` mapping
- [ ] Test with sample tasks
- [ ] Update usage guide to mention plan_strategy tool
- [ ] Update README.md with planner example

## 9. Example Usage Flow

```scheme
;; User workflow with planner

;; Step 1: Load data
(load-context research-papers "papers")

;; Step 2: Get strategy recommendation
;; (User calls plan_strategy via MCP tool)
;; Returns JSON with recommended strategy: Parallel Fan-Out + Tree Aggregation

;; Step 3: Review pattern details
;; (User calls get_pattern_details([1, 10]) to see full examples)

;; Step 4: Implement recommended strategy
(define papers (py-eval "context.split('\\n\\n')"))

;; Pattern 1: Parallel fan-out
(define extractions
  (map-async
    (lambda (paper)
      (llm-query
        (string-append "Extract key findings from:\n" paper)
        #:model "gpt-3.5-turbo"))
    papers))

;; Pattern 10: Tree aggregation
(define final-synthesis
  (tree-reduce
    (lambda (left right)
      (llm-query
        (string-append "Synthesize findings:\n\nGroup A:\n" left "\n\nGroup B:\n" right)
        #:model "curie"))
    extractions
    #:leaf-fn identity
    #:branch-factor 3))

;; Optional: Pattern 4 critique if quality critical
(define final-output
  (llm-query
    (string-append "Review and refine this analysis:\n" final-synthesis)
    #:model "gpt-4"))

final-output
```

## 10. Cost-Benefit Analysis

### Planning Cost
- Per call: $0.01-0.10
- Latency: 2-5 seconds

### Potential Savings
- Wrong model choice: $1-5 wasted
- Wrong pattern: $5-20 wasted
- Missing optimization: 2-10× unnecessary cost

### Expected ROI
- Conservative: 10× (save $0.10 on $0.01 planning)
- Typical: 50× (save $1-5 on $0.02-0.10 planning)
- Best case: 200× (save $20 on $0.10 planning)

**Conclusion:** Planning investment is highly cost-effective for non-trivial tasks.

## 11. Risk Mitigation

### Risk: Planner recommendations are poor
**Mitigation:**
- Always provide alternatives
- Show clear tradeoffs
- Include simple baseline option
- Encourage testing on small sample first

### Risk: Planner adds latency
**Mitigation:**
- Use curie ($0.01-0.03) instead of gpt-4 for speed
- Cache common task patterns
- Make planning optional (expert users can skip)

### Risk: Planner is too conservative
**Mitigation:**
- Explicitly request "creative_options" in output
- Set temperature=0.7 for LLM call
- Prompt emphasizes risk-taking
- Show experimental approaches with "potential_upside"

### Risk: Users don't trust LLM-generated plans
**Mitigation:**
- Include rationale for each recommendation
- Show estimated metrics (cost/latency/quality)
- Provide code templates for validation
- Reference specific pattern docs for details

---

## Summary

This implementation plan provides:

1. **LLM-powered planner** that knows all 16 patterns
2. **Structured output** with recommended strategy, alternatives, and creative options
3. **Cost-effective** planning ($0.01-0.10) with 10-200× ROI
4. **Risk-taking encouragement** via creative_options and explicit tradeoffs
5. **Integration** with existing tools (get_pattern_details, execute_scheme)
6. **Testing strategy** with 6 representative task types
7. **Future improvements** including learning from user feedback

**Next steps:**
1. Implement `_extract_pattern_knowledge()` in mcp_server.py
2. Implement `plan_strategy()` function
3. Create tool description markdown
4. Test with sample tasks
5. Deploy and gather feedback
