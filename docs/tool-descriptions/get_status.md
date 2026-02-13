Get current sandbox status and health information.

**Returns JSON with:**
- `active_calls`: List of in-progress LLM calls with IDs, instructions, and elapsed time
- `token_usage`: Total tokens used (prompt + completion)
- `rate_limits`: Current API rate limit status from OpenAI

**When to use:**
- Quick status check during long-running orchestrations
- Monitor token costs in real-time
- Find call IDs for cancellation
- Check rate limit status

**vs get_sandbox_state():**
- `get_status()`: Lightweight, real-time status (active calls + tokens)
- `get_sandbox_state()`: Deep debugging inspection (variables, checkpoints, stderr logs)

**Non-blocking:** Safe to call anytime without affecting execution.
