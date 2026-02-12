"""Tests for plan_strategy tool and combinator guidance.

Verifies that:
- plan_strategy returns valid JSON with required sections
- Combinator guidance is generated correctly
- Planner prompt template loads and formats correctly
- Model selection works (gpt-4o-mini for cost, gpt-4o for quality)
- Error handling works for invalid LLM responses
- Parameters are handled correctly (optional vs required)
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mcp_server
from mcp_server import (
    plan_strategy,
    _PLANNER_PROMPT_TEMPLATE,
)


# ============================================================
# Combinator guidance
# ============================================================


class TestCombinatorGuidance:
    """Combinator guidance is included in planner prompt."""

    def test_planner_prompt_has_guidance(self):
        """Planner prompt includes combinator guidance."""
        assert isinstance(_PLANNER_PROMPT_TEMPLATE, str)
        assert len(_PLANNER_PROMPT_TEMPLATE) > 100  # Should be substantial

    def test_planner_prompt_mentions_key_combinators(self):
        """Planner prompt mentions key combinators."""
        assert "fan-out-aggregate" in _PLANNER_PROMPT_TEMPLATE
        assert "critique-refine" in _PLANNER_PROMPT_TEMPLATE
        assert "tree-reduce" in _PLANNER_PROMPT_TEMPLATE
        assert "race" in _PLANNER_PROMPT_TEMPLATE
        assert "tiered" in _PLANNER_PROMPT_TEMPLATE
        assert "active-learning" in _PLANNER_PROMPT_TEMPLATE

    def test_planner_prompt_describes_usage(self):
        """Planner prompt describes when to use combinators."""
        # Should describe combinator purposes and characteristics
        # (Updated to match creative-focused prompt style)
        assert "parallel" in _PLANNER_PROMPT_TEMPLATE.lower()
        assert "sequential" in _PLANNER_PROMPT_TEMPLATE.lower() or "sequence" in _PLANNER_PROMPT_TEMPLATE.lower()
        # Should mention key concepts like cost, quality, or creative approaches
        assert "cost" in _PLANNER_PROMPT_TEMPLATE.lower() or "quality" in _PLANNER_PROMPT_TEMPLATE.lower()

    def test_planner_prompt_no_pattern_references(self):
        """Planner prompt does not reference pattern numbers."""
        # Should not contain "Pattern 1", "Pattern 4", etc.
        import re
        assert not re.search(r'Pattern \d+', _PLANNER_PROMPT_TEMPLATE), \
            "Planner prompt should not reference pattern numbers"


# ============================================================
# Planner prompt template
# ============================================================


class TestPlannerPromptTemplate:
    """Planner prompt template loads and formats correctly."""

    def test_template_is_loaded(self):
        """Template is loaded from docs/planner-prompt.md."""
        assert isinstance(_PLANNER_PROMPT_TEMPLATE, str)
        assert len(_PLANNER_PROMPT_TEMPLATE) > 100

    def test_template_has_placeholders(self):
        """Template contains format placeholders."""
        assert "{task_description}" in _PLANNER_PROMPT_TEMPLATE
        assert "{data_characteristics}" in _PLANNER_PROMPT_TEMPLATE
        assert "{constraints}" in _PLANNER_PROMPT_TEMPLATE
        assert "{priority}" in _PLANNER_PROMPT_TEMPLATE
        # Combinator guidance is now baked in, not a placeholder
        assert "{combinator_guidance}" not in _PLANNER_PROMPT_TEMPLATE

    def test_template_has_instructions(self):
        """Template contains key instructions."""
        assert "JSON" in _PLANNER_PROMPT_TEMPLATE
        assert "recommended" in _PLANNER_PROMPT_TEMPLATE
        assert "alternatives" in _PLANNER_PROMPT_TEMPLATE
        assert "creative" in _PLANNER_PROMPT_TEMPLATE.lower()

    def test_template_includes_model_pricing(self):
        """Template includes OpenAI model pricing."""
        # Updated to check for newer models (2026 pricing)
        assert "gpt-4" in _PLANNER_PROMPT_TEMPLATE  # Should include gpt-4o, gpt-4.1-nano, etc.
        assert "$0.0" in _PLANNER_PROMPT_TEMPLATE  # Should have some pricing information
        # Check for at least one of the newer models
        has_new_models = ("gpt-4.1-nano" in _PLANNER_PROMPT_TEMPLATE or
                         "gpt-4o-mini" in _PLANNER_PROMPT_TEMPLATE or
                         "gpt-4o" in _PLANNER_PROMPT_TEMPLATE)
        assert has_new_models, "Template should include newer GPT-4 models"


# ============================================================
# plan_strategy function tests
# ============================================================


class TestPlanStrategy:
    """plan_strategy returns valid structured recommendations."""

    def _mock_llm_response(self, valid=True):
        """Create a mock LLM response."""
        if valid:
            return {
                "text": json.dumps({
                    "recommended": {
                        "strategy_name": "Parallel Fan-Out with Tree Reduction",
                        "combinators": [
                            {
                                "combinator": "fan-out-aggregate",
                                "params": {
                                    "map-fn": "(lambda (item) (llm-query-async ...))",
                                    "reduce-fn": "(lambda (results) (tree-reduce ...))"
                                }
                            }
                        ],
                        "description": "Process items in parallel, then hierarchical synthesis",
                        "estimated_cost": "$0.50-1.00",
                        "estimated_latency": "5-10s",
                        "estimated_quality": "high",
                        "why_this_composition": "Parallel for speed, tree-reduce for context limits",
                        "code_template": "(define result (fan-out-aggregate ...))"
                    },
                    "alternatives": [
                        {
                            "strategy_name": "Sequential with Critique-Refine",
                            "combinators": [
                                {"combinator": "sequence"},
                                {"combinator": "critique-refine"}
                            ],
                            "description": "Sequential processing with quality refinement",
                            "tradeoffs": "Slower but higher quality",
                            "estimated_cost": "$1.00-2.00",
                            "when_to_choose": "Quality > speed"
                        }
                    ],
                    "creative_options": [
                        {
                            "strategy_name": "Ensemble Voting",
                            "combinators": [
                                {"combinator": "parallel"},
                                {"combinator": "vote"}
                            ],
                            "description": "Multiple strategies vote on result",
                            "risk_level": "experimental",
                            "potential_upside": "2x quality through consensus",
                            "estimated_cost": "$2.00-3.00",
                            "when_to_try": "Quality critical, high-stakes decisions"
                        }
                    ],
                    "recommendations": ["Test on sample first", "Use memoization for deduplication"]
                }),
                "prompt_tokens": 1000,
                "completion_tokens": 500
            }
        else:
            # Invalid JSON
            return {
                "text": "This is not valid JSON {invalid}",
                "prompt_tokens": 1000,
                "completion_tokens": 100
            }

    @patch("mcp_server.get_backend")
    def test_returns_valid_json(self, mock_backend):
        """plan_strategy returns valid JSON."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze 100 documents")
        data = json.loads(result)
        assert isinstance(data, dict)

    @patch("mcp_server.get_backend")
    def test_has_required_sections(self, mock_backend):
        """Response includes recommended, alternatives, creative_options."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze 100 documents")
        data = json.loads(result)

        assert "recommended" in data
        assert "alternatives" in data
        assert "creative_options" in data
        assert "recommendations" in data

    @patch("mcp_server.get_backend")
    def test_recommended_has_required_fields(self, mock_backend):
        """Recommended strategy has all required fields."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze 100 documents")
        data = json.loads(result)
        rec = data["recommended"]

        assert "strategy_name" in rec
        assert "combinators" in rec
        assert isinstance(rec["combinators"], list)
        assert "description" in rec
        assert "estimated_cost" in rec
        assert "estimated_latency" in rec
        assert "code_template" in rec

    @patch("mcp_server.get_backend")
    def test_includes_metadata(self, mock_backend):
        """Response includes _meta section with planner info."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze 100 documents")
        data = json.loads(result)

        assert "_meta" in data
        assert "planner_model" in data["_meta"]
        assert "planning_cost" in data["_meta"]
        assert "task_analyzed" in data["_meta"]

    @patch("mcp_server.get_backend")
    def test_uses_curie_for_cost_priority(self, mock_backend):
        """Uses curie model when priority=cost."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze documents", priority="cost")
        data = json.loads(result)

        # Check that gpt-4o-mini was used (updated from curie)
        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "gpt-4o-mini"

    @patch("mcp_server.get_backend")
    def test_uses_gpt4_for_other_priorities(self, mock_backend):
        """Uses gpt-4 model for non-cost priorities."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze documents", priority="quality")
        data = json.loads(result)

        # Check that gpt-4o was used (updated from gpt-4)
        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "gpt-4o"

    @patch("mcp_server.get_backend")
    def test_handles_optional_parameters(self, mock_backend):
        """Handles optional data_characteristics and constraints."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        # Should work with just task_description
        result1 = plan_strategy("Analyze documents")
        assert json.loads(result1)

        # Should work with all parameters
        result2 = plan_strategy(
            "Analyze documents",
            data_characteristics="100KB total",
            constraints="Budget <$1",
            priority="balanced"
        )
        assert json.loads(result2)

    @patch("mcp_server.get_backend")
    def test_prompt_includes_task_info(self, mock_backend):
        """Formatted prompt includes task description and parameters."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        task = "Analyze 100 research papers"
        data_chars = "5KB per paper"
        constraints = "Budget <$2"

        plan_strategy(task, data_characteristics=data_chars, constraints=constraints)

        # Check the prompt that was passed
        call_args = mock_repl._call_llm.call_args
        prompt = call_args[1]["instruction"]

        assert task in prompt
        assert data_chars in prompt
        assert constraints in prompt

    @patch("mcp_server.get_backend")
    def test_prompt_includes_combinator_guidance(self, mock_backend):
        """Formatted prompt includes combinator guidance."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Analyze documents")

        call_args = mock_repl._call_llm.call_args
        prompt = call_args[1]["instruction"]

        # Should include combinator information
        assert "fan-out-aggregate" in prompt or "combinator" in prompt.lower()
        assert "critique-refine" in prompt or "tree-reduce" in prompt

    @patch("mcp_server.get_backend")
    def test_sets_temperature_for_creativity(self, mock_backend):
        """Uses temperature=0.7 for creative strategy recommendations."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Analyze documents")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["temperature"] == 0.7

    @patch("mcp_server.get_backend")
    def test_handles_invalid_json_response(self, mock_backend):
        """Returns error JSON when LLM response is not valid JSON."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response(valid=False))
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze documents")
        data = json.loads(result)

        # Should have error and fallback recommendation
        assert "error" in data
        assert "fallback_recommendation" in data

    @patch("mcp_server.get_backend")
    def test_handles_llm_exception(self, mock_backend):
        """Returns error JSON when LLM call raises exception."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(side_effect=Exception("API error"))
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze documents")
        data = json.loads(result)

        # Should have error message
        assert "error" in data
        assert "API error" in data["error"]


# ============================================================
# Integration with other tools
# ============================================================


class TestPlannerIntegration:
    """plan_strategy integrates with other tools correctly."""

    def test_docstring_mentions_key_features(self):
        """plan_strategy has informative docstring."""
        doc = plan_strategy.__doc__
        assert "strategy" in doc.lower()
        # Should mention combinators, not patterns
        assert "combinator" in doc.lower() or "orchestration" in doc.lower()

    def test_is_registered_with_mcp(self):
        """plan_strategy is registered with FastMCP server."""
        # Verify it's in the tool descriptions mapping
        from mcp_server import _TOOL_DESCRIPTIONS
        assert "plan_strategy" in _TOOL_DESCRIPTIONS

        # Verify the function exists in mcp_server module
        assert hasattr(mcp_server, "plan_strategy")
        assert callable(mcp_server.plan_strategy)

    def test_tool_description_loaded(self):
        """Tool description is loaded from markdown file."""
        from mcp_server import _TOOL_DESCRIPTIONS
        assert "plan_strategy" in _TOOL_DESCRIPTIONS
        desc = _TOOL_DESCRIPTIONS["plan_strategy"]
        assert len(desc) > 50
        assert "strategy" in desc.lower()
