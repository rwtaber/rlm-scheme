#!/usr/bin/env python3
"""Python subprocess bridge for RLM-Scheme.

Reads JSON commands from stdin, writes JSON responses to stdout.
Maintains a persistent globals dict across calls.

Protocol:
  {"op": "exec", "code": "..."}       -> {"status": "ok", "stdout": "...", "stderr": "..."}
  {"op": "eval", "expr": "..."}       -> {"status": "ok", "value": ...}
  {"op": "call", "ref": "...", "method": "...", "args": [...]} -> {"status": "ok", "value": ...}
  {"op": "set-context", "data": "..."} -> {"status": "ok"}
"""

import sys
import json
import io
import traceback


def main():
    user_globals = {"__builtins__": __builtins__}
    refs = {}
    ref_counter = 0

    def make_ref(obj):
        nonlocal ref_counter
        key = f"obj_{ref_counter}"
        ref_counter += 1
        refs[key] = obj
        return {"__ref__": key}

    def to_json_value(val):
        """Convert a Python value to a JSON-serializable form."""
        if val is None or isinstance(val, (bool, int, float, str)):
            return val
        if isinstance(val, (list, tuple)):
            return [to_json_value(v) for v in val]
        if isinstance(val, dict):
            return {str(k): to_json_value(v) for k, v in val.items()}
        # Complex object — store as ref
        return make_ref(val)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError as e:
            respond({"status": "error", "message": f"Invalid JSON: {e}"})
            continue

        op = cmd.get("op")
        try:
            if op == "exec":
                code = cmd.get("code", "")
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()
                old_stdout, old_stderr = sys.stdout, sys.stderr
                sys.stdout = stdout_capture
                sys.stderr = stderr_capture
                try:
                    exec(code, user_globals)
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                respond({
                    "status": "ok",
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                })

            elif op == "eval":
                expr = cmd.get("expr", "")
                result = eval(expr, user_globals)
                respond({"status": "ok", "value": to_json_value(result)})

            elif op == "call":
                ref_key = cmd.get("ref")
                if isinstance(ref_key, dict):
                    ref_key = ref_key.get("__ref__", ref_key)
                method_name = cmd.get("method", "")
                args = cmd.get("args", [])
                obj = refs.get(ref_key)
                if obj is None:
                    respond({"status": "error", "message": f"Unknown ref: {ref_key}"})
                    continue
                method = getattr(obj, method_name)
                result = method(*args)
                respond({"status": "ok", "value": to_json_value(result)})

            elif op == "set-context":
                data = cmd.get("data", "")
                user_globals["context"] = data
                respond({"status": "ok"})

            elif op == "set-var":
                name = cmd.get("name", "")
                value = cmd.get("value", "")
                if not name.isidentifier():
                    respond({"status": "error", "message": f"Invalid variable name: {name}"})
                    continue
                user_globals[name] = value
                respond({"status": "ok"})

            else:
                respond({"status": "error", "message": f"Unknown op: {op}"})

        except Exception as e:
            respond({
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            })


def respond(obj):
    """Write a JSON response to stdout (the communication channel)."""
    # We need to write to the real stdout, not any captured one
    sys.__stdout__.write(json.dumps(obj) + "\n")
    sys.__stdout__.flush()


if __name__ == "__main__":
    main()
