Get execution trace showing LLM call hierarchy and data flow.

**Returns JSON array with entries containing:**
- `op`: Operation type ('llm-query', 'syntax-e', 'datum->syntax', 'py-exec', 'py-eval', 'unsafe-*')
- `datum_preview`: First 80 chars of data passed to operation
- `scope`: Execution context ('host', 'sandbox', 'sub-N' for recursive calls)

**Use cases:**
- Trace data flow through orchestration pipeline
- Debug scope isolation issues
- Audit LLM interactions for security/compliance
- Understand execution order in complex compositions
- Identify which calls processed which data

**Example scenario:** After executing a nested strategy, call get_execution_trace() to see the tree of LLM calls, what data each received, and how results flowed back up.

**Tip:** Combine with get_sandbox_state() for comprehensive debugging.
