"""Regression tests for critical bugs.

These tests prevent previously-fixed bugs from reoccurring.
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mcp_server import plan_strategy, RacketREPL


def run_racket(cmd):
    """Helper to run Racket commands (mimics other test files)."""
    repl = RacketREPL()
    try:
        return repl.send(cmd)
    finally:
        repl.close()


class TestPlanStrategyJSONExtraction:
    """Prevent regression of JSON extraction bug.

    Bug: LLMs often wrap JSON in markdown code fences (```json ... ```),
    which caused plan_strategy to fail with JSONDecodeError.

    Fix: Extract JSON from markdown fences before parsing.
    """

    def _mock_llm_with_fenced_json(self):
        """Mock LLM response with JSON wrapped in markdown fences."""
        json_content = {
            "recommended": {
                "strategy_name": "Test Strategy",
                "combinators": [{"combinator": "fan-out-aggregate"}],
                "code_template": "(define result ...)",
                "description": "Test description",
                "estimated_cost": "$0.10",
                "estimated_latency": "5s",
                "estimated_quality": "high",
                "why_this_composition": "Test reasoning"
            },
            "alternatives": [],
            "creative_options": [],
            "recommendations": []
        }
        # Return JSON wrapped in markdown fences (common LLM behavior)
        return {
            "text": f"```json\n{json.dumps(json_content, indent=2)}\n```",
            "prompt_tokens": 1000,
            "completion_tokens": 500
        }

    def _mock_llm_with_plain_json(self):
        """Mock LLM response with plain JSON (no fences)."""
        json_content = {
            "recommended": {
                "strategy_name": "Test Strategy",
                "combinators": [{"combinator": "fan-out-aggregate"}],
                "code_template": "(define result ...)",
                "description": "Test description",
                "estimated_cost": "$0.10",
                "estimated_latency": "5s",
                "estimated_quality": "high",
                "why_this_composition": "Test reasoning"
            },
            "alternatives": [],
            "creative_options": [],
            "recommendations": []
        }
        return {
            "text": json.dumps(json_content, indent=2),
            "prompt_tokens": 1000,
            "completion_tokens": 500
        }

    def _mock_llm_with_generic_fences(self):
        """Mock LLM response with JSON in generic code fences (no language marker)."""
        json_content = {
            "recommended": {
                "strategy_name": "Test Strategy",
                "combinators": [{"combinator": "fan-out-aggregate"}],
                "code_template": "(define result ...)",
                "description": "Test description",
                "estimated_cost": "$0.10",
                "estimated_latency": "5s",
                "estimated_quality": "high",
                "why_this_composition": "Test reasoning"
            },
            "alternatives": [],
            "creative_options": [],
            "recommendations": []
        }
        # Generic code fences without language marker
        return {
            "text": f"```\n{json.dumps(json_content, indent=2)}\n```",
            "prompt_tokens": 1000,
            "completion_tokens": 500
        }

    @patch("mcp_server.get_backend")
    def test_handles_json_in_markdown_fences(self, mock_backend):
        """plan_strategy extracts JSON from ```json fences."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_with_fenced_json())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Test task")
        data = json.loads(result)

        # Should successfully parse despite markdown fences
        assert "recommended" in data
        assert "error" not in data
        assert data["recommended"]["strategy_name"] == "Test Strategy"

    @patch("mcp_server.get_backend")
    def test_handles_plain_json(self, mock_backend):
        """plan_strategy handles plain JSON (no fences)."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_with_plain_json())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Test task")
        data = json.loads(result)

        # Should successfully parse plain JSON
        assert "recommended" in data
        assert "error" not in data

    @patch("mcp_server.get_backend")
    def test_handles_generic_fenced_json(self, mock_backend):
        """plan_strategy extracts JSON from generic ``` fences."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_with_generic_fences())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Test task")
        data = json.loads(result)

        # Should successfully parse JSON from generic fences
        assert "recommended" in data
        assert "error" not in data

    @patch("mcp_server.get_backend")
    def test_llm_wrapping_json_is_common(self, mock_backend):
        """Document that LLM JSON wrapping is common behavior.

        This test verifies our fix handles the most common LLM output format.
        """
        mock_repl = MagicMock()

        # Simulate what GPT-4o actually returns when asked for JSON
        actual_gpt4_style_response = {
            "text": '```json\n{\n  "recommended": {\n    "strategy_name": "Parallel Analysis"\n  }\n}\n```',
            "prompt_tokens": 1000,
            "completion_tokens": 100
        }
        mock_repl._call_llm = MagicMock(return_value=actual_gpt4_style_response)
        mock_backend.return_value = mock_repl

        # Should not crash with JSONDecodeError
        result = plan_strategy("Test task")
        # Should either succeed or return error dict, but not raise exception
        data = json.loads(result)
        assert isinstance(data, dict)


class TestPyBridgeAccessibility:
    """Prevent regression of Python bridge issues.

    Bug: Python bridge wasn't starting automatically, causing errors.

    Fix: Python bridge starts automatically when sandbox is created.
    User code can use py-exec/py-eval without manual initialization.
    """

    def test_py_exec_works_automatically(self):
        """py-exec works without manual initialization (auto-start)."""
        code = """
(define result (py-exec "print(2 + 2)"))
(finish result)
"""
        result = run_racket({"op": "eval", "code": code})
        assert result["status"] == "finished"
        assert "4" in result["result"]

    def test_py_eval_works_automatically(self):
        """py-eval works without manual initialization."""
        code = """
(define result (py-eval "3 * 4"))
(finish result)
"""
        result = run_racket({"op": "eval", "code": code})
        assert result["status"] == "finished"
        # py-eval returns the value directly
        assert "12" in str(result["result"])

    def test_py_set_and_exec_work_together(self):
        """py-set! and py-exec work together automatically."""
        code = """
(py-set! "x" "hello")
(define result (py-exec "print(x)"))
(finish result)
"""
        result = run_racket({"op": "eval", "code": code})
        assert result["status"] == "finished"
        assert "hello" in result["result"]


class TestDocumentationUpdates:
    """Verify documentation was updated for the fixes."""

    def test_api_reference_documents_python_bridge(self):
        """API reference documents Python bridge functionality."""
        from mcp_server import _CODE_GEN_API_REF
        assert "py-exec" in _CODE_GEN_API_REF
        assert "py-eval" in _CODE_GEN_API_REF
        # Should explain automatic startup
        assert "automatic" in _CODE_GEN_API_REF.lower() or "bridge" in _CODE_GEN_API_REF

    def test_api_reference_accessible_via_tool(self):
        """get_code_generation_api_reference returns updated docs."""
        from mcp_server import get_code_generation_api_reference
        api_ref = get_code_generation_api_reference()
        assert "py-exec" in api_ref
        assert "py-eval" in api_ref


class TestRegressionSuite:
    """High-level tests that would have caught these bugs."""

    @patch("mcp_server.get_backend")
    def test_plan_strategy_never_raises_json_decode_error(self, mock_backend):
        """plan_strategy should never raise JSONDecodeError.

        Even with malformed LLM output, should return error dict, not crash.
        """
        mock_repl = MagicMock()

        # Outputs that should trigger error handling (invalid JSON)
        invalid_json_outputs = [
            {"text": "```json\n{invalid json}\n```", "prompt_tokens": 100, "completion_tokens": 50},
            {"text": "Not JSON at all", "prompt_tokens": 100, "completion_tokens": 50},
        ]

        for output in invalid_json_outputs:
            mock_repl._call_llm = MagicMock(return_value=output)
            mock_backend.return_value = mock_repl

            # Should not raise exception
            result = plan_strategy("Test task")
            data = json.loads(result)  # Result itself should be valid JSON

            # Invalid JSON should trigger error response
            assert "error" in data or "fallback_recommendation" in data

        # Valid JSON (even if incomplete) should not crash
        valid_but_incomplete = {"text": "```\n{\"partial\": \"data\"}\n```", "prompt_tokens": 100, "completion_tokens": 50}
        mock_repl._call_llm = MagicMock(return_value=valid_but_incomplete)
        mock_backend.return_value = mock_repl
        result = plan_strategy("Test task")
        data = json.loads(result)  # Should not crash
        assert isinstance(data, dict)  # Should return some dict

    def test_all_python_functions_are_callable(self):
        """All documented Python bridge functions are callable."""
        functions_to_test = [
            "(py-exec \"print('test')\")",
            "(py-eval \"1 + 1\")",
            "(py-set! \"x\" \"value\")",
        ]

        for func_call in functions_to_test:
            code = f"""
{func_call}
(finish "ok")
"""
            result = run_racket({"op": "eval", "code": code})
            assert result["status"] == "finished", f"Failed to call: {func_call}"
