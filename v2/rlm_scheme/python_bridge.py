"""Controlled Python subprocess for py-exec/py-eval/py-call/py-set!."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from typing import Any

# Imports allowed in bridge subprocess
ALLOWED_IMPORTS = frozenset({
    "json", "csv", "statistics", "collections", "re",
    "math", "itertools", "functools", "operator",
    "datetime", "decimal", "fractions",
    "io", "string", "textwrap",
    "copy", "enum", "dataclasses",
    "typing", "abc",
    "hashlib", "base64", "binascii",
})

# Imports explicitly forbidden
FORBIDDEN_IMPORTS = frozenset({
    "os", "sys", "subprocess", "shutil", "pathlib",
    "socket", "http", "urllib", "requests",
    "importlib", "ctypes", "signal",
    "multiprocessing", "threading",
})


class PythonBridgeError(Exception):
    """Raised when bridge execution fails."""
    pass


class PythonBridge:
    """Runs Python code in an isolated subprocess with restricted imports."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self._variables: dict[str, Any] = {}

    def set_variable(self, name: str, value: Any) -> None:
        """Set a variable available to bridge code."""
        self._variables[name] = value

    def execute(self, code: str) -> str:
        """Execute Python code and return its stdout output."""
        _validate_code(code)
        return self._run(code, mode="exec")

    def evaluate(self, expr: str) -> Any:
        """Evaluate a Python expression and return the result."""
        _validate_code(expr)
        return self._run(expr, mode="eval")

    def call(self, ref: str, method: str, *args: Any) -> Any:
        """Call a method on a stored variable."""
        code = f"__result = {ref}.{method}(*__args)"
        self._variables["__args"] = list(args)
        return self._run(code, mode="call")

    def _run(self, code: str, mode: str) -> Any:
        """Execute code in a subprocess."""
        # Build the wrapper script
        vars_json = json.dumps(self._variables, default=str)
        wrapper = textwrap.dedent(f"""\
            import json, sys
            __vars = json.loads({vars_json!r})
            globals().update(__vars)
            try:
                if {mode!r} == "eval":
                    __result = eval({code!r})
                    print(json.dumps({{"ok": True, "result": __result}}, default=str))
                elif {mode!r} == "call":
                    exec({code!r})
                    print(json.dumps({{"ok": True, "result": __result}}, default=str))
                else:
                    import io
                    __buf = io.StringIO()
                    sys.stdout = __buf
                    exec({code!r})
                    sys.stdout = sys.__stdout__
                    print(json.dumps({{"ok": True, "result": __buf.getvalue()}}, default=str))
            except Exception as e:
                sys.stdout = sys.__stdout__
                print(json.dumps({{"ok": False, "error": str(e)}}, default=str))
        """)

        try:
            proc = subprocess.run(
                [sys.executable, "-c", wrapper],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise PythonBridgeError(f"Bridge execution timed out after {self.timeout}s")

        if proc.returncode != 0 and not proc.stdout.strip():
            raise PythonBridgeError(f"Bridge process failed: {proc.stderr[:500]}")

        try:
            result = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            raise PythonBridgeError(f"Bridge returned invalid JSON: {proc.stdout[:500]}")

        if not result.get("ok"):
            raise PythonBridgeError(f"Bridge error: {result.get('error', 'unknown')}")

        return result["result"]

    def reset(self) -> None:
        """Clear all stored variables."""
        self._variables.clear()

    @property
    def alive(self) -> bool:
        """Check if subprocess execution is possible."""
        try:
            result = self._run("1 + 1", mode="eval")
            return result == 2
        except Exception:
            return False


def _validate_code(code: str) -> None:
    """Basic validation: check for forbidden imports."""
    # Simple pattern matching — not a security sandbox, just a guardrail
    import_pattern = r'\b(?:import|from)\s+(\w+)'
    import re as re_mod
    for match in re_mod.finditer(import_pattern, code):
        module = match.group(1)
        if module in FORBIDDEN_IMPORTS:
            raise PythonBridgeError(f"Forbidden import: {module}")
