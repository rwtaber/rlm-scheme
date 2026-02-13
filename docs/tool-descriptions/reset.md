Reset the sandbox to initial state, clearing all context and state.

**What gets cleared:**
- Scheme variables and bindings
- Python globals and imports
- Loaded context data (from load_context)
- Scope log and call tracking
- Token usage statistics

**What persists:**
- Files written to disk
- Checkpoints saved with checkpoint()

**When to use:**
- Between unrelated tasks
- After errors or unexpected state
- When starting fresh experiments
- To free memory from large data

**Example:** Call reset() before processing a new dataset to ensure no state leakage from previous work.
