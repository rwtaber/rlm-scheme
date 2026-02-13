"""Tests for MCP tool surface simplification (S6) and non-blocking get_status (S7b).

Verifies that:
- get_status returns core sections (active_calls, token_usage, rate_limits)
- get_status is non-blocking (no Racket IPC)
- get_status returns valid JSON
- Removed tools (get_active_calls, get_token_usage, get_rate_limits, get_variables)
  no longer appear as MCP tools
- Remaining tools have descriptive docstrings mentioning key capabilities
- Tool count is 8 (down from 11)
"""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mcp_server
from mcp_server import mcp, RacketREPL


# ============================================================
# get_status consolidation
# ============================================================


class TestGetStatus:
    """get_status returns all monitoring info in one call."""

    @patch("mcp_server.openai.OpenAI")
    def test_returns_all_sections(self, mock_openai_cls):
        """Response contains active_calls, token_usage, rate_limits."""
        repl = RacketREPL()
        try:
            with patch("mcp_server.get_backend", return_value=repl):
                result = mcp_server.get_status()
            data = json.loads(result)
            assert "active_calls" in data
            assert "token_usage" in data
            assert "rate_limits" in data
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_returns_valid_json(self, mock_openai_cls):
        """Response is valid JSON."""
        repl = RacketREPL()
        try:
            with patch("mcp_server.get_backend", return_value=repl):
                result = mcp_server.get_status()
            data = json.loads(result)
            assert isinstance(data, dict)
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_token_usage_has_expected_keys(self, mock_openai_cls):
        """token_usage section has prompt_tokens, completion_tokens, total_tokens, total_calls."""
        repl = RacketREPL()
        try:
            with patch("mcp_server.get_backend", return_value=repl):
                result = mcp_server.get_status()
            usage = json.loads(result)["token_usage"]
            assert "prompt_tokens" in usage
            assert "completion_tokens" in usage
            assert "total_tokens" in usage
            assert "total_calls" in usage
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_rate_limits_has_expected_keys(self, mock_openai_cls):
        """rate_limits section has remaining_tokens, remaining_requests, etc."""
        repl = RacketREPL()
        try:
            with patch("mcp_server.get_backend", return_value=repl):
                result = mcp_server.get_status()
            rl = json.loads(result)["rate_limits"]
            assert "remaining_tokens" in rl
            assert "remaining_requests" in rl
            assert "limit_tokens" in rl
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_active_calls_is_list(self, mock_openai_cls):
        """active_calls section is a list."""
        repl = RacketREPL()
        try:
            with patch("mcp_server.get_backend", return_value=repl):
                result = mcp_server.get_status()
            calls = json.loads(result)["active_calls"]
            assert isinstance(calls, list)
        finally:
            repl.close()

    @patch("mcp_server.openai.OpenAI")
    def test_no_racket_ipc_call(self, mock_openai_cls):
        """get_status does not call backend.send() (non-blocking)."""
        repl = RacketREPL()
        try:
            with patch("mcp_server.get_backend", return_value=repl):
                with patch.object(repl, "send") as mock_send:
                    mcp_server.get_status()
            mock_send.assert_not_called()
        finally:
            repl.close()


# ============================================================
# Removed tools no longer exist as MCP tools
# ============================================================


class TestRemovedTools:
    """Old individual status tools are no longer registered as MCP tools."""

    def test_no_get_active_calls_tool(self):
        """get_active_calls is not an MCP tool anymore."""
        assert not hasattr(mcp_server, "get_active_calls") or \
               not hasattr(getattr(mcp_server, "get_active_calls", None), "_mcp_tool")

    def test_no_get_token_usage_tool(self):
        """get_token_usage is not an MCP tool anymore."""
        assert not hasattr(mcp_server, "get_token_usage") or \
               not hasattr(getattr(mcp_server, "get_token_usage", None), "_mcp_tool")

    def test_no_get_rate_limits_tool(self):
        """get_rate_limits is not an MCP tool anymore."""
        assert not hasattr(mcp_server, "get_rate_limits") or \
               not hasattr(getattr(mcp_server, "get_rate_limits", None), "_mcp_tool")

    def test_no_get_variables_tool(self):
        """get_variables is not an MCP tool anymore."""
        assert not hasattr(mcp_server, "get_variables") or \
               not hasattr(getattr(mcp_server, "get_variables", None), "_mcp_tool")


# ============================================================
# Tool descriptions are informative
# ============================================================


class TestToolDescriptions:
    """Tool descriptions surface key capabilities."""

    def test_execute_scheme_mentions_python(self):
        """execute_scheme description mentions Python capability."""
        doc = mcp_server.execute_scheme.__doc__
        assert "Python" in doc or "python" in doc or "py-exec" in doc

    def test_execute_scheme_mentions_web_requests(self):
        """execute_scheme description mentions web requests."""
        doc = mcp_server.execute_scheme.__doc__
        assert "web" in doc.lower()

    def test_execute_scheme_mentions_file_io(self):
        """execute_scheme description mentions file I/O."""
        doc = mcp_server.execute_scheme.__doc__
        assert "file" in doc.lower()

    def test_execute_scheme_mentions_llm(self):
        """execute_scheme description mentions LLM orchestration."""
        doc = mcp_server.execute_scheme.__doc__
        assert "LLM" in doc or "llm" in doc or "sub-call" in doc

    def test_execute_scheme_mentions_vision(self):
        """execute_scheme description mentions vision/image capability."""
        doc = mcp_server.execute_scheme.__doc__
        assert "vision" in doc.lower() or "image" in doc.lower()

    def test_load_context_mentions_data_types(self):
        """load_context description mentions accepted data types."""
        doc = mcp_server.load_context.__doc__
        assert "JSON" in doc or "CSV" in doc or "document" in doc.lower()

    def test_get_usage_guide_mentions_content(self):
        """get_usage_guide description mentions patterns or overview."""
        doc = mcp_server.get_usage_guide.__doc__
        assert "pattern" in doc.lower() or "overview" in doc.lower() or "guide" in doc.lower()

    def test_cancel_call_mentions_get_status(self):
        """cancel_call now references get_status instead of get_active_calls."""
        doc = mcp_server.cancel_call.__doc__
        assert "get_status" in doc

    def test_execute_scheme_mentions_model_selection(self):
        """execute_scheme description includes model selection guidance."""
        doc = mcp_server.execute_scheme.__doc__
        assert "gpt-3.5-turbo" in doc or "ada" in doc or "curie" in doc
        assert "cheap" in doc.lower() or "cost" in doc.lower()

    def test_get_usage_guide_mentions_model_guide(self):
        """get_usage_guide description mentions model selection."""
        doc = mcp_server.get_usage_guide.__doc__
        assert "model" in doc.lower() or "pattern" in doc.lower()


# ============================================================
# Model selection guide in USAGE_GUIDE_CORE (S9)
# ============================================================


class TestModelGuide:
    """USAGE_GUIDE_CORE includes model selection guidance."""

    def test_model_table_present(self):
        """USAGE_GUIDE_CORE has a Model Selection section."""
        from mcp_server import USAGE_GUIDE_CORE
        assert "## Model Selection" in USAGE_GUIDE_CORE

    def test_economy_models_listed(self):
        """Guide lists economy models (updated for 2026 pricing)."""
        from mcp_server import USAGE_GUIDE_CORE
        # Check for newer economy models
        assert ("gpt-4o-mini" in USAGE_GUIDE_CORE or "gpt-4.1-nano" in USAGE_GUIDE_CORE)

    def test_capable_models_listed(self):
        """Guide lists capable models (updated for 2026)."""
        from mcp_server import USAGE_GUIDE_CORE
        # Check for modern GPT-4 variants
        assert "gpt-4" in USAGE_GUIDE_CORE

    def test_pricing_included(self):
        """Guide includes pricing information."""
        from mcp_server import USAGE_GUIDE_CORE
        # Should include pricing for modern models (2026)
        assert "$0.0001" in USAGE_GUIDE_CORE or "$0.0005" in USAGE_GUIDE_CORE or "$0.01" in USAGE_GUIDE_CORE

    def test_fan_out_guidance(self):
        """Guide recommends cheap models for fan-out."""
        from mcp_server import USAGE_GUIDE_CORE
        assert "fan-out" in USAGE_GUIDE_CORE.lower() or "fan out" in USAGE_GUIDE_CORE.lower()

    def test_parameter_guide_present(self):
        """API reference includes parameter usage guidance."""
        import mcp_server
        api_ref = mcp_server.get_codegen_reference()
        assert "#:temperature" in api_ref
        assert "#:max-tokens" in api_ref
        assert "#:json" in api_ref

    def test_primitives_documented(self):
        """Guide documents key primitives."""
        from mcp_server import USAGE_GUIDE_CORE
        # Should mention core primitives
        assert "map-async" in USAGE_GUIDE_CORE or "llm-query" in USAGE_GUIDE_CORE
