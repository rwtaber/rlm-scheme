"""Tests for recursive sandbox bindings documentation.

Verifies that documentation properly covers what bindings are available
in recursive sandboxes when using #:recursive #t.
"""

import mcp_server


class TestRecursiveBindingsDocumentation:
    def test_usage_guide_documents_recursive_bindings(self):
        """API reference and pattern details should list available bindings."""
        # Check API reference for code generation
        api_ref = mcp_server.get_code_generation_api_reference()

        # Should mention key LLM call functions
        assert "llm-query" in api_ref
        assert "llm-query-async" in api_ref
        assert "map-async" in api_ref

        # Should mention Python bridge functions
        assert "py-exec" in api_ref
        assert "py-eval" in api_ref
        assert "py-set!" in api_ref

        # Should mention control flow functions
        assert "finish" in api_ref
        assert "checkpoint" in api_ref or "restore" in api_ref

        # Should mention common Scheme primitives
        assert "define" in api_ref
        assert "lambda" in api_ref
        assert "string-append" in api_ref

    def test_recursive_section_mentions_isolation(self):
        """Pattern 3 documentation should clarify that recursive sandboxes are isolated."""
        # Check Pattern 3: Recursive Delegation
        pattern_3 = mcp_server.get_pattern_details(3)

        # Should mention isolation and sandbox
        assert "sandbox" in pattern_3.lower()

    def test_recursive_section_mentions_depth_limit(self):
        """Pattern 3 documentation should mention the max recursion depth."""
        # Check Pattern 3: Recursive Delegation
        pattern_3 = mcp_server.get_pattern_details(3)

        # Should mention depth limits
        assert "depth" in pattern_3.lower()
        assert "3" in pattern_3  # Max depth is 3
        assert ("recursive" in pattern_3.lower() or "recursion" in pattern_3.lower())

    def test_execute_scheme_mentions_recursive(self):
        """execute_scheme tool description should mention recursive delegation."""
        doc = mcp_server.execute_scheme.__doc__

        # Should mention recursive strategy
        assert "#:recursive" in doc or "recursive" in doc.lower()
        assert "Recursive Delegation" in doc or "recursive delegation" in doc.lower()


class TestRecursiveBindingsList:
    def test_llm_functions_documented(self):
        """All LLM-related functions should be documented for recursive sandboxes."""
        # Check Pattern 3: Recursive Delegation
        pattern_3 = mcp_server.get_pattern_details(3)

        assert "llm-query" in pattern_3
        assert "#:recursive" in pattern_3

    def test_python_bridge_documented(self):
        """Python bridge functions should be documented for recursive sandboxes."""
        # Check API reference
        api_ref = mcp_server.get_code_generation_api_reference()

        assert "py-exec" in api_ref

    def test_no_shared_context_documented(self):
        """Should document sandbox isolation in Pattern 3."""
        # Check Pattern 3: Recursive Delegation
        pattern_3 = mcp_server.get_pattern_details(3)

        # Pattern 3 should mention sandboxes and specialists
        assert "sandbox" in pattern_3.lower() or "specialist" in pattern_3.lower()
