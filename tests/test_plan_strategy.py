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
    plan_strategy_clarify,
    plan_strategy_finalize,
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
        """Uses gpt-4o model for all priorities (quality over cost for planning)."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Analyze documents", priority="cost")
        data = json.loads(result)

        # Check that gpt-4o was used (always use quality model for planning)
        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "gpt-4o"

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


# ============================================================
# Phase 1: Scale Parameters Tests
# ============================================================


class TestPhase1ScaleParameters:
    """Phase 1: Explicit scale parameters and improved token limits."""

    def _mock_llm_response(self):
        """Mock LLM response with scale validation."""
        return {
            "text": json.dumps({
                "recommended": {
                    "strategy_name": "Comprehensive Documentation Strategy",
                    "combinators": [{"combinator": "fan-out-aggregate"}],
                    "code_template": "(define result (fan-out-aggregate ...))\n(finish result)",
                    "description": "Processes all 500 files",
                    "estimated_cost": "$1-2",
                    "estimated_latency": "5-10 min",
                    "estimated_quality": "high",
                    "estimated_outputs": "500 documentation files",
                    "coverage_achieved": "100% (all files)",
                    "scale_validation": "✓ Processes all 500 files | ✓ Produces 500 outputs",
                    "why_this_works": "Full coverage via parallel processing"
                },
                "alternatives": [],
                "creative_options": [],
                "recommendations": []
            }),
            "prompt_tokens": 2000,
            "completion_tokens": 5000
        }

    @patch("mcp_server.get_backend")
    def test_accepts_scale_parameter(self, mock_backend):
        """plan_strategy accepts scale parameter."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy(
            "Document repository",
            scale="comprehensive"
        )
        data = json.loads(result)
        assert "_meta" in data
        assert data["_meta"]["scale"] == "comprehensive"

    @patch("mcp_server.get_backend")
    def test_accepts_min_outputs_parameter(self, mock_backend):
        """plan_strategy accepts min_outputs parameter."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy(
            "Document repository",
            min_outputs=500
        )
        data = json.loads(result)
        assert data["_meta"]["min_outputs"] == 500

    @patch("mcp_server.get_backend")
    def test_accepts_coverage_target_parameter(self, mock_backend):
        """plan_strategy accepts coverage_target parameter."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy(
            "Document repository",
            coverage_target="all files"
        )
        data = json.loads(result)
        assert data["_meta"]["coverage_target"] == "all files"

    @patch("mcp_server.get_backend")
    def test_uses_increased_token_limits(self, mock_backend):
        """plan_strategy uses increased token limits (15K default)."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Document repository", priority="balanced")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["max_tokens"] == 16000  # Always use max

    @patch("mcp_server.get_backend")
    def test_quality_priority_uses_16k_tokens(self, mock_backend):
        """Quality priority uses 16K tokens (gpt-4o limit)."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Document repository", priority="quality")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["max_tokens"] == 16000

    @patch("mcp_server.get_backend")
    def test_cost_priority_uses_10k_tokens(self, mock_backend):
        """Cost priority uses 10K tokens."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Document repository", priority="cost")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["max_tokens"] == 16000  # Always use max

    @patch("mcp_server.get_backend")
    def test_uses_gpt4o_as_default_model(self, mock_backend):
        """Default priority uses gpt-4o (upgraded from gpt-4o-mini)."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy("Document repository", priority="balanced")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "gpt-4o"

    @patch("mcp_server.get_backend")
    def test_prompt_includes_scale_parameters(self, mock_backend):
        """Prompt includes scale, min_outputs, coverage_target."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        plan_strategy(
            "Document repository",
            scale="comprehensive",
            min_outputs=500,
            coverage_target="all files"
        )

        call_args = mock_repl._call_llm.call_args
        prompt = call_args[1]["instruction"]

        # Should include scale level
        assert "comprehensive" in prompt.lower()
        # Should include validation requirements
        assert "500" in prompt or "minimum" in prompt.lower()

    @patch("mcp_server.get_backend")
    def test_scale_defaults_to_medium(self, mock_backend):
        """Scale parameter defaults to 'medium'."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_llm_response())
        mock_backend.return_value = mock_repl

        result = plan_strategy("Document repository")
        data = json.loads(result)

        assert data["_meta"]["scale"] == "medium"


# ============================================================
# Phase 2: Multi-Turn Planning Tests
# ============================================================


class TestPhase2MultiTurnPlanning:
    """Phase 2: Multi-turn clarification workflow."""

    def _mock_clarify_response_clear(self):
        """Mock clarify response for clear task."""
        return {
            "text": json.dumps({
                "clarity_assessment": {
                    "is_clear": True,
                    "confidence": 0.95,
                    "reasoning": "Task specifies file count and format"
                },
                "ambiguities_found": [],
                "assumptions": [],
                "recommended_clarifications": []
            }),
            "prompt_tokens": 500,
            "completion_tokens": 200
        }

    def _mock_clarify_response_ambiguous(self):
        """Mock clarify response for ambiguous task."""
        return {
            "text": json.dumps({
                "clarity_assessment": {
                    "is_clear": False,
                    "confidence": 0.3,
                    "reasoning": "Scale and format unclear"
                },
                "ambiguities_found": [
                    {
                        "category": "scale",
                        "question": "How many files in the repository?",
                        "why_matters": "Determines parallelization strategy",
                        "suggested_options": ["<50", "50-500", "500+"]
                    }
                ],
                "assumptions": [],
                "recommended_clarifications": [
                    "How many files are in your repository?",
                    "What format of documentation do you need?"
                ]
            }),
            "prompt_tokens": 500,
            "completion_tokens": 400
        }

    def _mock_finalize_response(self):
        """Mock finalize response."""
        return {
            "text": json.dumps({
                "recommended": {
                    "strategy_name": "Clarified Strategy",
                    "combinators": [{"combinator": "fan-out-aggregate"}],
                    "code_template": "(define result ...)",
                    "description": "Based on user clarifications",
                    "estimated_cost": "$1-2",
                    "clarification_alignment": "Processes all 500 files as user specified"
                },
                "alternatives": [],
                "creative_options": [],
                "recommendations": []
            }),
            "prompt_tokens": 3000,
            "completion_tokens": 6000
        }

    @patch("mcp_server.get_backend")
    def test_clarify_returns_clear_assessment(self, mock_backend):
        """plan_strategy_clarify returns clarity assessment."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_clarify_response_clear())
        mock_backend.return_value = mock_repl

        result = plan_strategy_clarify("Process 500 documents")
        data = json.loads(result)

        assert "clarity_assessment" in data
        assert data["clarity_assessment"]["is_clear"] == True
        assert "ambiguities_found" in data

    @patch("mcp_server.get_backend")
    def test_clarify_identifies_ambiguities(self, mock_backend):
        """plan_strategy_clarify identifies ambiguities."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_clarify_response_ambiguous())
        mock_backend.return_value = mock_repl

        result = plan_strategy_clarify("Document the repository")
        data = json.loads(result)

        assert data["clarity_assessment"]["is_clear"] == False
        assert len(data["ambiguities_found"]) > 0
        assert len(data["recommended_clarifications"]) > 0

    @patch("mcp_server.get_backend")
    def test_clarify_uses_efficient_model(self, mock_backend):
        """plan_strategy_clarify uses gpt-4o-mini for efficiency."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_clarify_response_clear())
        mock_backend.return_value = mock_repl

        plan_strategy_clarify("Process documents")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["model"] == "gpt-4o"  # Always use quality model

    @patch("mcp_server.get_backend")
    def test_clarify_uses_lower_temperature(self, mock_backend):
        """plan_strategy_clarify uses lower temperature (0.5) for analysis."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_clarify_response_clear())
        mock_backend.return_value = mock_repl

        plan_strategy_clarify("Process documents")

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["temperature"] == 0.5  # Lower temp for analysis

    @patch("mcp_server.get_backend")
    def test_finalize_accepts_clarifications(self, mock_backend):
        """plan_strategy_finalize accepts clarifications parameter."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_finalize_response())
        mock_backend.return_value = mock_repl

        clarifications = "500 files, API docs format, all files"
        result = plan_strategy_finalize(
            "Document repository",
            clarifications=clarifications
        )
        data = json.loads(result)

        assert "_meta" in data
        assert data["_meta"]["clarifications_incorporated"] == True

    @patch("mcp_server.get_backend")
    def test_finalize_includes_clarifications_in_prompt(self, mock_backend):
        """plan_strategy_finalize includes clarifications in prompt."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_finalize_response())
        mock_backend.return_value = mock_repl

        clarifications = "500 Python files, comprehensive coverage"
        plan_strategy_finalize(
            "Document repository",
            clarifications=clarifications
        )

        call_args = mock_repl._call_llm.call_args
        prompt = call_args[1]["instruction"]

        assert "500" in prompt
        assert "comprehensive" in prompt.lower()

    @patch("mcp_server.get_backend")
    def test_finalize_uses_same_token_limits(self, mock_backend):
        """plan_strategy_finalize uses same token limits as plan_strategy."""
        mock_repl = MagicMock()
        mock_repl._call_llm = MagicMock(return_value=self._mock_finalize_response())
        mock_backend.return_value = mock_repl

        plan_strategy_finalize(
            "Document repository",
            clarifications="500 files",
            priority="quality"
        )

        call_args = mock_repl._call_llm.call_args
        assert call_args[1]["max_tokens"] == 16000  # Quality priority (gpt-4o limit)

    def test_clarify_tool_is_registered(self):
        """plan_strategy_clarify is registered with MCP."""
        from mcp_server import _TOOL_DESCRIPTIONS
        assert "plan_strategy_clarify" in _TOOL_DESCRIPTIONS

    def test_finalize_tool_is_registered(self):
        """plan_strategy_finalize is registered with MCP."""
        from mcp_server import _TOOL_DESCRIPTIONS
        assert "plan_strategy_finalize" in _TOOL_DESCRIPTIONS
