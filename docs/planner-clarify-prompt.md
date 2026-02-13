# Task Analysis and Clarification Questions

You are analyzing an orchestration task to identify ambiguities that need clarification before planning.

## Task

**Objective:** {task_description}

**Data:** {data_characteristics}

**Constraints:** {constraints}

**Priority:** {priority}

---

## Your Goal

Analyze this task and identify **critical ambiguities** that would affect strategy design:

1. **Scale Ambiguity:** Is the scope/scale unclear?
   - How many items to process?
   - What level of coverage? (overview vs comprehensive)
   - All files or selective?

2. **Output Ambiguity:** Are the deliverables unclear?
   - How many output artifacts?
   - What format/structure?
   - Quality requirements?

3. **Scope Ambiguity:** Are the boundaries unclear?
   - What's included/excluded?
   - Which components/modules?
   - Depth of analysis?

4. **Constraint Ambiguity:** Are there missing constraints?
   - Budget limits?
   - Time constraints?
   - Quality thresholds?

---

## Output Format

Return ONLY valid JSON:

```json
{{
  "clarity_assessment": {{
    "is_clear": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Why is this task clear or ambiguous?"
  }},
  "ambiguities_found": [
    {{
      "category": "scale/output/scope/constraint",
      "question": "Specific clarifying question",
      "why_matters": "How this affects strategy design",
      "suggested_options": ["Option A", "Option B", "Option C"]
    }}
  ],
  "assumptions": [
    {{
      "assumption": "What we're assuming if user doesn't clarify",
      "risk": "What could go wrong with this assumption"
    }}
  ],
  "recommended_clarifications": [
    "Clear, actionable question to ask the user"
  ]
}}
```

**Important:**
- If the task is CLEAR and unambiguous, set `is_clear: true` and return empty `ambiguities_found`
- Only ask clarifying questions when genuinely needed
- Don't ask questions if reasonable defaults exist
- Focus on ambiguities that significantly affect strategy design

---

## Examples

**Example 1: Clear Task**
Task: "Process 500 documents (5KB each), extract key findings, produce JSON summary. Budget <$5."

Response:
```json
{{
  "clarity_assessment": {{
    "is_clear": true,
    "confidence": 0.95,
    "reasoning": "Scale (500 docs), format (JSON), budget ($5) all specified. No significant ambiguities."
  }},
  "ambiguities_found": [],
  "assumptions": [],
  "recommended_clarifications": []
}}
```

**Example 2: Ambiguous Task**
Task: "Document the large repository"

Response:
```json
{{
  "clarity_assessment": {{
    "is_clear": false,
    "confidence": 0.3,
    "reasoning": "Multiple critical ambiguities: scale undefined ('large' is vague), output format unspecified, coverage level unclear."
  }},
  "ambiguities_found": [
    {{
      "category": "scale",
      "question": "How many files are in the repository?",
      "why_matters": "Determines parallelization strategy and cost",
      "suggested_options": ["<50 files", "50-500 files", "500+ files"]
    }},
    {{
      "category": "output",
      "question": "What documentation format do you want?",
      "why_matters": "Different formats require different generation strategies",
      "suggested_options": ["API reference docs", "Architecture overview", "Comprehensive docs for all files"]
    }},
    {{
      "category": "scope",
      "question": "Do you want documentation for all files or just public APIs?",
      "why_matters": "Affects scale by 10x+ (all files vs public only)",
      "suggested_options": ["All files", "Public APIs only", "Major modules only"]
    }}
  ],
  "assumptions": [
    {{
      "assumption": "If not clarified, will assume medium scale (50-200 files), API docs format, public APIs only",
      "risk": "May under-deliver if user expected comprehensive coverage"
    }}
  ],
  "recommended_clarifications": [
    "How many files are in your repository?",
    "What format of documentation do you need? (API reference, tutorials, architecture docs, etc.)",
    "Do you want documentation for all files or just public APIs/major components?"
  ]
}}
```

---

## Now Analyze

Analyze the task above. Be honest about ambiguities but don't over-ask - only clarify what's truly ambiguous.
