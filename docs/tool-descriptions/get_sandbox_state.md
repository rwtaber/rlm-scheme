Inspect current sandbox state for debugging. Returns information about what state is currently active in the sandbox.

**Returns:**
- **scheme_variables**: List of Scheme variable names defined in the sandbox (not values)
- **python_available**: Whether Python bridge is accessible
- **checkpoints**: List of saved checkpoint keys on disk
- **scope_log_entries**: Number of entries in the scope tracking log

**Use cases:**
- Debug "unbound identifier" errors (check if variable exists)
- Verify state persistence between execute_scheme calls
- Check if checkpoints were saved successfully
- Understand what's currently in the sandbox

**What this DOES NOT show:**
- Variable values (could be very large)
- Python globals (would require subprocess query)
- File contents

**Example:**
```python
# After running some Scheme code
state = get_sandbox_state()
print(state)
# {
#   "scheme_variables": ["context", "result", "summaries"],
#   "python_available": true,
#   "checkpoints": ["phase1_data", "intermediate_results"],
#   "scope_log_entries": 12
# }
```

**When to use:**
- Before calling execute_scheme: Check if expected state exists
- After timeouts: Verify what survived
- Debugging pipelines: Understand state at each step
