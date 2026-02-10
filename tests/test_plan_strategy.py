"""Tests for plan_strategy tool and pattern knowledge extraction.

Verifies that:
- plan_strategy returns valid JSON with required sections
- Pattern knowledge is extracted from docs at startup
- Planner prompt template loads and formats correctly
- Model selection works (curie for cost, gpt-4 otherwise)
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
    _extract_pattern_knowledge,
    _PATTERN_SUMMARY,
    _PLANNER_PROMPT_TEMPLATE,
)


# ============================================================
# Pattern knowledge extraction
# ============================================================


class TestPatternKnowledgeExtraction:
    """Pattern knowledge is extracted from markdown files at startup."""

    def test_pattern_knowledge_has_all_16_patterns(self):
        """All 16 patterns are extracted."""
        knowledge = _extract_pattern_knowledge()
        assert len(knowledge) == 16
        for i in range(1, 17):
            assert i in knowledge

    def test_pattern_has_required_fields(self):
        """Each pattern has id, title, when_to_use, improvements, strengths."""
        knowledge = _extract_pattern_knowledge()
        pattern = knowledge[1]  # Check pattern 1
        assert "id" in pattern
        assert "title" in pattern
        assert "when_to_use" in pattern
        assert "improvements" in pattern
        assert "strengths" in pattern

    def test_pattern_title_format(self):
        """Pattern titles are extracted correctly."""
        knowledge = _extract_pattern_knowledge()
        # Pattern 1 should be "Parallel Fan-Out (MapReduce)"
        assert "Fan-Out" in knowledge[1]["title"] or "MapReduce" in knowledge[1]["title"]
        # Pattern 4 should be "Critique-Refine"
        assert "Critique" in knowledge[4]["title"] or "Refine" in knowledge[4]["title"]

    def test_pattern_strengths_detected(self):
        """Pattern strengths (speed/cost/quality) are detected from improvements."""
        knowledge = _extract_pattern_knowledge()
        # Pattern 1 (Fan-Out) should have speed and/or cost strengths
        assert len(knowledge[1]["strengths"]) > 0
        # Pattern 4 (Critique-Refine) should have quality strength
        assert len(knowledge[4]["strengths"]) > 0

    def test_pattern_summary_is_string(self):
        """Pattern summary is a string."""
        assert isinstance(_PATTERN_SUMMARY, str)
        assert len(_PATTERN_SUMMARY) > 100  # Should be substantial

    def test_pattern_summary_includes_all_patterns(self):
        """Pattern summary mentions all 16 patterns."""
        for i in range(1, 17):
            assert f"Pattern {i}" in _PATTERN_SUMMARY


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
        assert "{pattern_summary}" in _PLANNER_PROMPT_TEMPLATE

    def test_template_has_instructions(self):
        """Template contains key instructions."""
        assert "JSON" in _PLANNER_PROMPT_TEMPLATE
        assert "recommended" in _PLANNER_PROMPT_TEMPLATE
        assert "alternatives" in _PLANNER_PROMPT_TEMPLATE
        assert "creative" in _PLANNER_PROMPT_TEMPLATE.lower()

    def test_template_includes_model_pricing(self):
        """Template includes OpenAI model pricing."""
        assert "gpt-3.5-turbo" in _PLANNER_PROMPT_TEMPLATE
        assert "ada" in _PLANNER_PROMPT_TEMPLATE
        assert "gpt-4" in _PLANNER_PROMPT_TEMPLATE
        assert "$0.002" in _PLANNER_PROMPT_TEMPLATE or "$0.0004" in _PLANNER_PROMPT_TEMPLATE


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
                        "strategy_name": "Parallel Fan-Out with Synthesis",
                        "patterns": [1, 4],
                        "description": "Process items in parallel, then synthesize",
                        "estimated_cost": "$0.50-1.00",
                        "estimated_latency": "5-10s",
                        "estimated_quality": "high",
                        "model_assignments": {"fan_out": "gpt-3.5-turbo", "synthesis": "gpt-4"},
                        "implementation": ["Load data", "Fan out", "Synthesize"],
                        "code_template": "(map-async ...)"
                    },
                    "alternatives": [
                        {
                            "strategy_name": "Meta-Orchestration",
                            "patterns": [6],
                            "description": "Let LLM design strategy",
                            "tradeoffs": "Higher cost but adaptive",
                            "estimated_cost": "$1.00-2.00",
                            "when_to_choose": "Unknown structure"
                        }
                    ],
                    "creative_options": [
                        {
                            "strategy_name": "Ensemble Voting",
                            "patterns": [8],
                            "description": "Multiple models vote",
                            "risk_level": "experimental",
                            "potential_upside": "2x quality",
                            "estimated_cost": "$2.00-3.00",
                            "when_to_try": "Quality critical"
                        }
                    ],
                    "recommendations": ["Test on sample first", "Use memoization"]
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
        assert "patterns" in rec
        assert isinstance(rec["patterns"], list)
        assert "description" in rec
        assert "estimated_cost" in rec
        assert "estimated_latency" in rec
        assert "model_assignments" in rec
        assert "implementation" in rec

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

        # Check that curie was used
        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "curie"

    @patch("mcp_server.get_backend")
    def test_uses_gpt4_for_other_priorities(self, mock_backend):
        """Uses gpt-4 model for non-cost priorities."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze documents", priority="quality")
        data = json.loads(result)

        # Check that gpt-4 was used
        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "gpt-4"

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
    def test_prompt_includes_pattern_summary(self, mock_backend):
        """Formatted prompt includes pattern summary."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Analyze documents")

        call_args = mock_repl._call_llm.call_args
        prompt = call_args[1]["instruction"]

        # Should include pattern information
        assert "Pattern 1" in prompt or "pattern 1" in prompt.lower()
        assert "Pattern 4" in prompt or "pattern 4" in prompt.lower()

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
        assert "pattern" in doc.lower()

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
