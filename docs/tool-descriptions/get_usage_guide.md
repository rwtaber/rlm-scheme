Quick start guide for RLM-Scheme - composable combinators for LLM orchestration.

**Philosophy: Playbook, Not Cookbook**
This guide provides building blocks, not recipes. The ~17 combinators can be composed in infinite ways.

**What you'll get:**
- **Philosophy section**: "Playbook not cookbook" - why experimentation is cheaper than optimization
- **Core concepts**: Overview of ~17 combinators and their capabilities (not prescriptive usage)
- **Example strategies**: Standard, creative, AND experimental approaches (not just safe patterns)
- **What's Possible**: Exploration of composition capabilities and creative control flow
- **Execution model**: Persistent sandbox, async operations, state management, **Python bridge** (Scheme ↔ Python data transfer)
- **Python bridge patterns**: py-set!, py-eval, py-exec with common workflows (file I/O, data processing)
- **Model selection**: Cost/characteristics table (no "use X for Y" rules)
- **Workflows**: Flexible approaches for getting started
- **Tips for success**: Emphasis on experimentation and trying creative approaches

**Guide size:** ~540 lines (focused on possibilities, not prescriptions)

**When to use this:**
- First time using RLM-Scheme
- Need a conceptual overview before diving in
- Want to understand what's possible (not what's "correct")
- Looking for inspiration on creative compositions

**What makes this different:**
- **Not prescriptive**: Shows what's possible, not what you "should" do
- **Encourages experimentation**: Emphasizes low cost of testing vs high cost of wrong choice
- **Creative examples**: Includes experimental strategies alongside standard patterns
- **No "best practices"**: Only tradeoffs and possibilities

**For deeper documentation:**
- **Full combinator reference:** `get_combinator_reference()` - Complete docs for all 17 combinators
- **Strategy recommendations:** `plan_strategy(task, data, priority)` - Get creative custom strategies for your task
- **API reference:** `get_codegen_reference()` - Syntax reference

**Workflow:**
1. Read this guide to understand the building blocks
2. Call `plan_strategy()` to get creative recommendations
3. Try multiple strategies (testing is cheap!)
4. For deeper understanding, explore `get_combinator_reference()`
