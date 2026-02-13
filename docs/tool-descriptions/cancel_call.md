Cancel a running LLM call by its call ID.

**How it works:**
- Cancels async futures for pending API calls
- Terminates nested Racket subprocess for recursive calls
- Returns immediately (non-blocking)
- Does not affect token accounting for already-completed work

**When to use:**
- Stop long-running incorrect calls
- Cancel expensive operations after seeing initial bad results
- Recover from hung or stalled calls
- Manual intervention during debugging

**Workflow:**
1. Call get_status() to see active calls and their IDs
2. Identify the problematic call ID
3. Call cancel_call(call_id) to terminate it
4. Check get_status() again to confirm cancellation

**Returns:** Status message indicating success or if call ID not found.

**Note:** Cancellation is best-effort. Some calls may complete before cancellation takes effect.
