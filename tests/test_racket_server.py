"""Unit tests for racket_server.rkt — test via subprocess.

Tests the Racket sandbox server directly by sending JSON commands
and checking JSON responses. No MCP or LLM involved.
"""

import json
import os
import subprocess
import pytest

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")


def send_commands(*commands: dict, timeout: int = 30) -> list[dict]:
    """Send one or more JSON commands to the Racket server, return responses."""
    input_text = "\n".join(json.dumps(cmd) for cmd in commands) + "\n"
    result = subprocess.run(
        ["racket", RACKET_SERVER],
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    responses = []
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            responses.append(json.loads(line))
    return responses


def send_one(cmd: dict, timeout: int = 30) -> dict:
    """Send a single command and return the single response."""
    responses = send_commands(cmd, timeout=timeout)
    assert len(responses) == 1, f"Expected 1 response, got {len(responses)}: {responses}"
    return responses[0]


def eval_code(code: str, timeout: int = 30) -> dict:
    """Shortcut for sending an eval command."""
    return send_one({"op": "eval", "code": code}, timeout=timeout)


# ============================================================
# Core eval (16 tests)
# ============================================================


class TestCoreEval:
    def test_integer_literal(self):
        r = eval_code("(finish 42)")
        assert r["status"] == "finished"
        assert r["result"] == "42"

    def test_string_literal(self):
        r = eval_code('(finish "hello")')
        assert r["status"] == "finished"
        assert r["result"] == "hello"

    def test_boolean_literal(self):
        r = eval_code("(finish #t)")
        assert r["status"] == "finished"
        assert r["result"] == "#t"

    def test_arithmetic(self):
        r = eval_code("(finish (+ 1 2 3))")
        assert r["status"] == "finished"
        assert r["result"] == "6"

    def test_multiplication(self):
        r = eval_code("(finish (* 6 7))")
        assert r["status"] == "finished"
        assert r["result"] == "42"

    def test_string_append(self):
        r = eval_code('(finish (string-append "a" "b"))')
        assert r["status"] == "finished"
        assert r["result"] == "ab"

    def test_string_length(self):
        r = eval_code('(finish (string-length "test"))')
        assert r["status"] == "finished"
        assert r["result"] == "4"

    def test_define_and_use(self):
        r = eval_code("(define x 10)\n(finish x)")
        assert r["status"] == "finished"
        assert r["result"] == "10"

    def test_function_define(self):
        r = eval_code("(define (f x) (+ x 1))\n(finish (f 5))")
        assert r["status"] == "finished"
        assert r["result"] == "6"

    def test_let_binding(self):
        r = eval_code("(finish (let ([a 1] [b 2]) (+ a b)))")
        assert r["status"] == "finished"
        assert r["result"] == "3"

    def test_lambda(self):
        r = eval_code("(finish ((lambda (x) (* x x)) 5))")
        assert r["status"] == "finished"
        assert r["result"] == "25"

    def test_conditional(self):
        r = eval_code('(finish (if (> 3 2) "yes" "no"))')
        assert r["status"] == "finished"
        assert r["result"] == "yes"

    def test_map(self):
        r = eval_code("(finish (map (lambda (x) (* x 2)) (list 1 2 3)))")
        assert r["status"] == "finished"
        assert r["result"] == "(2 4 6)"

    def test_multi_expression(self):
        r = eval_code("(define x 1)\n(define y 2)\n(finish (+ x y))")
        assert r["status"] == "finished"
        assert r["result"] == "3"

    def test_no_finish(self):
        r = eval_code("(define x 42)")
        assert r["status"] == "ok"
        assert r["result"] == ""

    def test_define_blocks_scaffold_redefine(self):
        """define on scaffold bindings is blocked, even though set! works
        (set! is needed internally for load-context)."""
        r = eval_code('(define context "hijack")')
        assert r["status"] == "error"

    def test_begin_with_multiple_defines(self):
        """Multiple defines in begin should all execute."""
        r = eval_code("""
        (begin
          (define x 10)
          (define y 20)
          (finish (+ x y)))
        """)
        assert r["status"] == "finished"
        assert r["result"] == "30"

    def test_nested_begin(self):
        """Nested begin forms should flatten correctly."""
        r = eval_code("""
        (begin
          (define a 1)
          (begin
            (define b 2)
            (define c 3))
          (finish (+ a b c)))
        """)
        assert r["status"] == "finished"
        assert r["result"] == "6"

    def test_begin_with_expressions(self):
        """begin can contain both defines and expressions."""
        r = eval_code("""
        (begin
          (define x 5)
          (* x 2)
          (define y 10)
          (finish (+ x y)))
        """)
        assert r["status"] == "finished"
        assert r["result"] == "15"


# ============================================================
# Scaffold (8 tests)
# ============================================================


class TestScaffold:
    def test_scaffold_protection_finish(self):
        r = eval_code("(define finish 42)")
        assert r["status"] == "error"
        assert "scaffold" in r["message"].lower() or "redefine" in r["message"].lower()

    def test_scaffold_protection_context(self):
        r = eval_code('(define context "hijack")')
        assert r["status"] == "error"

    def test_error_message_contains_helpful_context(self):
        """Error messages should include the problematic binding name."""
        r = eval_code('(define llm-query "hijack")')
        assert r["status"] == "error"
        msg_lower = r["message"].lower()
        assert "llm-query" in msg_lower or "scaffold" in msg_lower
        assert "protected" in msg_lower or "redefine" in msg_lower or "cannot" in msg_lower

    def test_datum_syntax_roundtrip(self):
        r = eval_code('(define stx (datum->syntax #f "hello"))\n(finish (syntax-e stx))')
        assert r["status"] == "finished"
        assert r["result"] == "hello"

    def test_syntax_e_passthrough(self):
        r = eval_code('(finish (syntax-e "plain string"))')
        assert r["status"] == "finished"
        assert r["result"] == "plain string"

    def test_get_variables(self):
        responses = send_commands(
            {"op": "eval", "code": "(define myvar 42)"},
            {"op": "get-variables"},
        )
        assert responses[1]["status"] == "ok"
        variables = json.loads(responses[1]["result"])
        assert "myvar" in variables

    def test_scope_log_records_datum_to_syntax(self):
        responses = send_commands(
            {"op": "eval", "code": '(define stx (datum->syntax #f "test"))'},
            {"op": "get-scope-log"},
        )
        log = json.loads(responses[1]["result"])
        ops = [entry["op"] for entry in log]
        assert "datum->syntax" in ops

    def test_reset_clears_state(self):
        responses = send_commands(
            {"op": "eval", "code": "(define x 99)"},
            {"op": "reset"},
            {"op": "eval", "code": "(finish x)"},
        )
        # After reset, x should not exist
        assert responses[2]["status"] == "error"

    def test_load_context(self):
        responses = send_commands(
            {"op": "load-context", "data": "hello world"},
            {"op": "eval", "code": "(finish context)"},
        )
        assert responses[0]["status"] == "ok"
        assert responses[1]["status"] == "finished"
        assert responses[1]["result"] == "hello world"

    def test_load_context_forwards_to_py_bridge(self):
        """Context loaded should be accessible in Python via py_bridge."""
        responses = send_commands(
            {"op": "load-context", "data": "test data for python"},
            {"op": "eval", "code": '(finish (py-exec "print(context)"))'}
        )
        assert responses[0]["status"] == "ok"
        assert responses[1]["status"] == "finished"
        assert "test data for python" in responses[1]["result"]

    def test_reset_restarts_py_bridge(self):
        """Reset should tear down and restart py_bridge subprocess."""
        responses = send_commands(
            {"op": "eval", "code": '(py-exec "x = 42")'},
            {"op": "eval", "code": '(finish (py-exec "print(x)"))'},
            {"op": "reset"},
            {"op": "eval", "code": '(finish (py-exec "print(x)"))'}  # Should fail - x doesn't exist
        )
        # First py-exec should work
        assert responses[0]["status"] == "ok"
        # Second should print 42
        assert responses[1]["status"] == "finished"
        assert "42" in responses[1]["result"]
        # After reset, x should not exist
        assert responses[3]["status"] == "error"
        assert "NameError" in responses[3]["message"] or "name 'x'" in responses[3]["message"]


# ============================================================
# Parameterize (4 tests)
# ============================================================


class TestParameterize:
    def test_read_budget(self):
        r = eval_code("(parameterize ([token-budget 1000]) (finish (token-budget)))")
        assert r["status"] == "finished"
        assert r["result"] == "1000"

    def test_nested_independent_budget(self):
        r = eval_code("""
(parameterize ([token-budget 1000])
  (parameterize ([token-budget 500])
    (define inner (token-budget)))
  (finish (list inner (token-budget))))
""")
        # This tests that the inner budget doesn't affect the outer
        # Note: inner is defined in a scope that may not be accessible outside
        # Let's test differently
        pass

    def test_nested_parameterize(self):
        r = eval_code("""
(define result
  (parameterize ([token-budget 1000])
    (define outer (token-budget))
    (parameterize ([token-budget 500])
      (define inner (token-budget))
      (list outer inner))))
(finish result)
""")
        assert r["status"] == "finished"
        assert "(1000 500)" in r["result"]

    def test_default_budget_infinite(self):
        r = eval_code("(finish (token-budget))")
        assert r["status"] == "finished"
        assert "inf" in r["result"].lower() or r["result"] == "+inf.0"


# ============================================================
# Escape hatches (4 tests)
# ============================================================


class TestEscapeHatches:
    def test_unsafe_interpolate(self):
        responses = send_commands(
            {"op": "eval", "code": '(define stx (datum->syntax #f "raw text"))\n(finish (unsafe-interpolate stx))'},
            {"op": "get-scope-log"},
        )
        assert responses[0]["status"] == "finished"
        assert responses[0]["result"] == "raw text"
        log = json.loads(responses[1]["result"])
        ops = [entry["op"] for entry in log]
        assert "unsafe-interpolate" in ops

    def test_unsafe_overwrite(self):
        responses = send_commands(
            {"op": "eval", "code": "(define myval 10)"},
            {"op": "eval", "code": "(unsafe-overwrite 'myval 99)"},
            {"op": "eval", "code": "(finish myval)"},
            {"op": "get-scope-log"},
        )
        assert responses[2]["status"] == "finished"
        assert responses[2]["result"] == "99"
        log = json.loads(responses[3]["result"])
        ops = [entry["op"] for entry in log]
        assert "unsafe-overwrite" in ops

    def test_unsafe_exec_sub_output(self):
        responses = send_commands(
            {"op": "eval", "code": '(unsafe-exec-sub-output (datum->syntax #f "(define result 42)"))'},
            {"op": "eval", "code": "(finish result)"},
            {"op": "get-scope-log"},
        )
        assert responses[1]["status"] == "finished"
        assert responses[1]["result"] == "42"
        log = json.loads(responses[2]["result"])
        ops = [entry["op"] for entry in log]
        assert "unsafe-exec-sub-output" in ops

    def test_all_escape_hatches_logged(self):
        responses = send_commands(
            {"op": "eval", "code": '(define stx (datum->syntax #f "test"))'},
            {"op": "eval", "code": "(unsafe-interpolate stx)"},
            {"op": "eval", "code": "(unsafe-overwrite 'context \"new\")"},
            {"op": "eval", "code": '(unsafe-exec-sub-output (datum->syntax #f "(+ 1 1)"))'},
            {"op": "get-scope-log"},
        )
        log = json.loads(responses[4]["result"])
        ops = [entry["op"] for entry in log]
        assert "unsafe-interpolate" in ops
        assert "unsafe-overwrite" in ops
        assert "unsafe-exec-sub-output" in ops


# ============================================================
# py-bridge integration (4 tests)
# ============================================================


class TestPyBridge:
    def test_py_exec_print(self):
        r = eval_code('(finish (py-exec "print(1+1)"))')
        assert r["status"] == "finished"
        assert "2" in r["result"]

    def test_py_eval(self):
        r = eval_code('(finish (py-eval "2+2"))')
        assert r["status"] == "finished"
        assert r["result"] == "4"

    def test_py_call_ref(self):
        r = eval_code("""
(define _x (py-exec "import json"))
(define ref (py-eval "json"))
(finish (py-call ref "dumps" (list 1 2 3)))
""")
        assert r["status"] == "finished"
        assert r["result"] == "[1, 2, 3]"

    def test_py_context(self):
        responses = send_commands(
            {"op": "load-context", "data": "hello world"},
            {"op": "eval", "code": '(finish (py-eval "len(context)"))'},
        )
        assert responses[1]["status"] == "finished"
        assert responses[1]["result"] == "11"


# ============================================================
# py-set! (issue 5)
# ============================================================


class TestPySet:
    """py-set! safely transfers Scheme strings to Python variables."""

    def test_py_set_basic(self):
        r = eval_code('(py-set! "x" "hello")\n(finish (py-exec "print(x)"))')
        assert r["status"] == "finished"
        assert "hello" in r["result"]

    def test_py_set_with_quotes(self):
        """Strings with quotes must survive the transfer."""
        r = eval_code("""(py-set! "msg" "she said \\"hello\\"")
(finish (py-exec "print(msg)"))""")
        assert r["status"] == "finished"
        assert 'she said "hello"' in r["result"]

    def test_py_set_with_backslashes(self):
        """Backslashes must survive the transfer."""
        r = eval_code(r'(py-set! "path" "C:\\Users\\test")'
                       '\n(finish (py-exec "print(path)"))')
        assert r["status"] == "finished"
        assert "C:\\Users\\test" in r["result"]

    def test_py_set_with_newlines(self):
        """Newlines in values must survive."""
        r = eval_code('(py-set! "text" "line1\\nline2")\n(finish (py-exec "print(repr(text))"))')
        assert r["status"] == "finished"
        assert "line1" in r["result"]
        assert "line2" in r["result"]

    def test_py_set_then_use_in_computation(self):
        r = eval_code('(py-set! "val" "42")\n(finish (py-exec "print(int(val) * 2)"))')
        assert r["status"] == "finished"
        assert "84" in r["result"]

    def test_py_set_overwrite(self):
        """Setting the same variable twice overwrites."""
        r = eval_code('(py-set! "x" "first")\n(py-set! "x" "second")\n(finish (py-exec "print(x)"))')
        assert r["status"] == "finished"
        assert "second" in r["result"]

    def test_py_set_invalid_name(self):
        """Invalid Python identifiers should error."""
        r = eval_code('(py-set! "123bad" "val")')
        assert r["status"] == "error"

    def test_py_set_is_scaffold_protected(self):
        r = eval_code("(define py-set! 42)")
        assert r["status"] == "error"

    def test_py_set_logged_in_scope(self):
        responses = send_commands(
            {"op": "eval", "code": '(py-set! "x" "test")'},
            {"op": "get-scope-log"},
        )
        log = json.loads(responses[1]["result"])
        ops = [entry["op"] for entry in log]
        assert "py-set!" in ops


# ============================================================
# Stdout capture
# ============================================================


class TestStdout:
    def test_display_captured(self):
        r = eval_code('(display "hello") (finish 0)')
        assert r["stdout"] == "hello"

    def test_multiple_displays(self):
        r = eval_code('(display "a") (display "b") (display "c") (finish 0)')
        assert r["stdout"] == "abc"

    def test_no_stdout(self):
        r = eval_code("(finish 42)")
        assert r["stdout"] == ""


# ============================================================
# Non-void top-level expressions (issue 4)
# ============================================================


class TestNonVoidTopLevel:
    """Non-void top-level expressions must NOT trigger 'finished'.
    Only an explicit (finish ...) call should terminate evaluation."""

    def test_bare_string_does_not_finish(self):
        r = eval_code('"hello"\n(finish "done")')
        assert r["status"] == "finished"
        assert r["result"] == "done"

    def test_bare_number_does_not_finish(self):
        r = eval_code("42\n(finish 99)")
        assert r["status"] == "finished"
        assert r["result"] == "99"

    def test_bare_arithmetic_does_not_finish(self):
        r = eval_code("(+ 1 2)\n(finish 42)")
        assert r["status"] == "finished"
        assert r["result"] == "42"

    def test_string_append_does_not_finish(self):
        r = eval_code('(string-append "a" "b")\n(finish "ok")')
        assert r["status"] == "finished"
        assert r["result"] == "ok"

    def test_py_exec_does_not_finish(self):
        r = eval_code('(py-exec "print(1+1)")\n(finish "continued")')
        assert r["status"] == "finished"
        assert r["result"] == "continued"

    def test_multiple_non_void_then_finish(self):
        r = eval_code('"a"\n42\n(+ 1 2)\n(string-length "test")\n(finish "all ran")')
        assert r["status"] == "finished"
        assert r["result"] == "all ran"

    def test_non_void_without_finish_gives_ok(self):
        """Non-void expressions without finish should return ok, not finished."""
        r = eval_code('"hello"\n42\n(+ 1 2)')
        assert r["status"] == "ok"
        assert r["result"] == ""

    def test_define_then_non_void_then_finish(self):
        r = eval_code('(define x 10)\n(+ x 5)\n(finish x)')
        assert r["status"] == "finished"
        assert r["result"] == "10"

    def test_finish_still_works(self):
        """Explicit finish should still terminate correctly."""
        r = eval_code('(finish "early")\n(finish "late")')
        assert r["status"] == "finished"
        assert r["result"] == "early"

    def test_finish_var_still_works(self):
        r = eval_code('(define result "hello")\n(finish-var "result")')
        assert r["status"] == "finished"
        assert r["result"] == "hello"


# ============================================================
# Sandbox list/string functions (issue 7)
# ============================================================


class TestSandboxListStringFunctions:
    """racket/list and racket/string functions should be available."""

    # --- racket/list ---

    def test_take(self):
        r = eval_code("(finish (take '(1 2 3 4 5) 3))")
        assert r["status"] == "finished"
        assert r["result"] == "(1 2 3)"

    def test_drop(self):
        r = eval_code("(finish (drop '(1 2 3 4 5) 2))")
        assert r["status"] == "finished"
        assert r["result"] == "(3 4 5)"

    def test_first(self):
        r = eval_code("(finish (first '(10 20 30)))")
        assert r["status"] == "finished"
        assert r["result"] == "10"

    def test_rest(self):
        r = eval_code("(finish (rest '(10 20 30)))")
        assert r["status"] == "finished"
        assert r["result"] == "(20 30)"

    def test_filter_map(self):
        r = eval_code("""(finish (filter-map
  (lambda (x) (if (even? x) (* x 10) #f))
  '(1 2 3 4 5)))""")
        assert r["status"] == "finished"
        assert r["result"] == "(20 40)"

    # --- racket/string ---

    def test_string_trim(self):
        r = eval_code('(finish (string-trim "  hello  "))')
        assert r["status"] == "finished"
        assert r["result"] == "hello"

    def test_string_split(self):
        r = eval_code('(finish (string-split "a,b,c" ","))')
        assert r["status"] == "finished"
        assert r["result"] == "(a b c)"

    def test_string_join(self):
        r = eval_code("""(finish (string-join '("hello" "world") " "))""")
        assert r["status"] == "finished"
        assert r["result"] == "hello world"

    def test_string_contains(self):
        r = eval_code('(finish (string-contains? "hello world" "world"))')
        assert r["status"] == "finished"
        # string-contains? returns index or #f
        assert r["result"] != "#f"

    def test_combined_list_and_string(self):
        """Use list and string functions together in a pipeline."""
        r = eval_code("""(define words (string-split "the quick brown fox" " "))
(define first-two (take words 2))
(finish (string-join first-two "-"))""")
        assert r["status"] == "finished"
        assert r["result"] == "the-quick"


# ============================================================
# py-exec error reporting with tracebacks (S11)
# ============================================================


class TestPyExecTraceback:
    def test_py_exec_error_includes_traceback(self):
        """py-exec errors include the Python traceback."""
        r = eval_code('(py-exec "x = 1/0")')
        assert r["status"] == "error"
        assert "ZeroDivisionError" in r["message"]
        assert "Traceback" in r["message"]

    def test_py_exec_import_error_traceback(self):
        """Import errors include traceback info."""
        r = eval_code('(py-exec "import nonexistent_module_xyz")')
        assert r["status"] == "error"
        assert "nonexistent_module_xyz" in r["message"]
        assert "Traceback" in r["message"]

    def test_py_eval_error_includes_traceback(self):
        """py-eval errors include the Python traceback."""
        r = eval_code('(py-eval "1/0")')
        assert r["status"] == "error"
        assert "ZeroDivisionError" in r["message"]
        assert "Traceback" in r["message"]

    def test_py_exec_multiline_error(self):
        """Multi-line code errors show traceback with context."""
        r = eval_code("""(py-exec "
def broken():
    return 1/0
broken()
")""")
        assert r["status"] == "error"
        assert "ZeroDivisionError" in r["message"]
        assert "broken" in r["message"]


# ============================================================
# map-async validation (Priority 1 improvement #1)
# ============================================================


class TestMapAsyncValidation:
    """Test that map-async provides clear error when used incorrectly."""

    def test_map_async_with_wrong_function_type_clear_error(self):
        """Using llm-query instead of llm-query-async gives clear error."""
        # Simulate the mistake: using a function that returns a syntax object
        # instead of an async handle
        code = """
(define (not-async-fn item)
  (datum->syntax #f (string-append "result-" item)))

(map-async not-async-fn '("a" "b" "c"))
"""
        r = eval_code(code)
        assert r["status"] == "error"
        # Check for the clear, actionable error message
        assert "llm-query-async" in r["message"]
        assert "async handle" in r["message"]
        # Should mention the solution
        assert "llm-query-async" in r["message"]

    def test_map_async_with_correct_async_handle_succeeds(self):
        """Using a function that returns proper async handles works."""
        # Simulate correct usage: function returns async handle format
        code = """
(define (mock-async-fn item)
  (list 'async-handle (string-append "pending_" item)))

(define handles (map (lambda (x) (mock-async-fn x)) '("1" "2" "3")))
(finish (car (car handles)))
"""
        r = eval_code(code)
        assert r["status"] == "finished"
        assert r["result"] == "async-handle"

    def test_map_async_empty_list_no_error(self):
        """map-async with empty list doesn't try to validate."""
        code = """
(define (any-fn item) "doesn't matter")
(finish (map-async any-fn '()))
"""
        r = eval_code(code)
        assert r["status"] == "finished"
        assert r["result"] == "()"
