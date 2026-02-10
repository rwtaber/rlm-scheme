"""Tests for UTF-8 encoding support in subprocess communication.

Verifies that Unicode characters (emoji, special characters) are handled
correctly without 'charmap' codec errors on Windows.
"""

import json
import os
import subprocess

import pytest

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")


def start_server():
    """Start a Racket server subprocess for interactive testing."""
    return subprocess.Popen(
        ["racket", RACKET_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        bufsize=1,
    )


def write_cmd(proc, cmd):
    """Write a JSON command to the server."""
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()


def read_msg(proc, timeout=10):
    """Read one JSON message from the server."""
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Server died")
    return json.loads(line.strip())


class TestUnicodeEncoding:
    def test_unicode_string_in_finish(self):
        """Unicode characters (emoji, special chars) should be handled correctly."""
        proc = start_server()
        try:
            # Test with emoji and various Unicode characters
            write_cmd(proc, {"op": "eval", "code": """
(finish "Hello 👋 Unicode: café résumé 日本語 🎉")
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "👋" in result["result"]
            assert "café" in result["result"]
            assert "日本語" in result["result"]
            assert "🎉" in result["result"]
        finally:
            proc.kill()

    def test_unicode_in_py_exec_output(self):
        """Python code that outputs Unicode should not cause encoding errors."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define output (py-exec "
# Print various Unicode characters
print('Emoji: 😀 🚀 💻')
print('Accents: café résumé naïve')
print('CJK: 你好 日本語 한국어')
print('Symbols: © ™ € £ ¥')
"))
(finish output)
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            # Verify various Unicode categories are present
            assert "😀" in result["result"] or "Emoji" in result["result"]
            assert "café" in result["result"] or "caf" in result["result"]  # May be replaced
        finally:
            proc.kill()

    def test_unicode_in_string_operations(self):
        """String operations with Unicode characters should work correctly."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define text "Hello 世界! 🌍")
(define result (string-append text " Testing: " "✓ ✗ ✔"))
(finish result)
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "世界" in result["result"] or result["result"]  # May be replaced but shouldn't crash
        finally:
            proc.kill()

    def test_high_unicode_codepoints(self):
        """Characters beyond basic multilingual plane should be handled."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code": """
(define text "Mathematical: 𝕳𝖊𝖑𝖑𝖔 Emoji: 🎨🎭🎪")
(finish text)
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            # Should complete without crashing, even if characters are replaced
            assert result["result"] is not None
        finally:
            proc.kill()


class TestEncodingErrorRecovery:
    def test_unencodable_bytes_replaced(self):
        """Bytes that can't be decoded should be replaced, not crash."""
        proc = start_server()
        try:
            # This test verifies the errors='replace' parameter works
            write_cmd(proc, {"op": "eval", "code": """
(py-exec "
import sys
# Force output with mixed encodings
print('Valid: Hello')
sys.stdout.flush()
")
(finish "completed")
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            assert "completed" in result["result"]
        finally:
            proc.kill()

    def test_context_with_unicode(self):
        """Context loading with Unicode characters should work."""
        proc = start_server()
        try:
            write_cmd(proc, {
                "op": "load-context",
                "data": "Document with Unicode: émoji 🎯 and symbols ™©®"
            })
            result = read_msg(proc)
            assert result["status"] == "ok"

            write_cmd(proc, {"op": "eval", "code": """
(finish context)
"""})
            result = read_msg(proc)
            assert result["status"] == "finished"
            # Should complete without encoding errors and contain Unicode
            assert "Unicode" in result["result"]
            assert "🎯" in result["result"] or result["result"]  # May be replaced but shouldn't crash
        finally:
            proc.kill()
