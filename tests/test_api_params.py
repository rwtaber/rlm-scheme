"""Tests for sub-model API parameters: #:temperature, #:max-tokens, #:json.

Uses subprocess.Popen to talk to racket_server.rkt directly,
intercepting llm-query callbacks and verifying the forwarded parameters.
"""

import json
import os
import subprocess

import pytest

RACKET_SERVER = os.path.join(os.path.dirname(__file__), "..", "racket_server.rkt")


def start_server():
    return subprocess.Popen(
        ["racket", RACKET_SERVER],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )


def write_cmd(proc, cmd):
    proc.stdin.write(json.dumps(cmd) + "\n")
    proc.stdin.flush()


def read_msg(proc):
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("Server died")
    return json.loads(line.strip())


def write_response(proc, resp):
    proc.stdin.write(json.dumps(resp) + "\n")
    proc.stdin.flush()


# ============================================================
# Temperature
# ============================================================


class TestTemperature:
    def test_temperature_forwarded(self):
        """#:temperature value appears in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test" #:temperature 0.3)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["temperature"] == 0.3

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_temperature_default_absent(self):
        """Without #:temperature, the field is absent from the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert "temperature" not in msg

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_temperature_zero(self):
        """#:temperature 0 is forwarded (not treated as absent)."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test" #:temperature 0)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            # In Racket, 0 is falsy for our (if temperature ...) check,
            # but 0.0 should be handled. Let's see what actually comes through.
            # If temperature is 0 (an integer), Racket's (if 0 ...) evaluates the then-branch
            # because 0 is truthy in Racket (only #f is falsy).
            assert msg["temperature"] == 0

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()


# ============================================================
# Max tokens
# ============================================================


class TestMaxTokens:
    def test_max_tokens_forwarded(self):
        """#:max-tokens value appears in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test" #:max-tokens 500)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["max_tokens"] == 500

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_max_tokens_default_absent(self):
        """Without #:max-tokens, the field is absent from the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert "max_tokens" not in msg

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()


# ============================================================
# JSON mode
# ============================================================


class TestJsonMode:
    def test_json_mode_forwarded(self):
        """#:json #t sets json_mode to true in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "Return json" #:json #t)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["json_mode"] is True

            write_response(proc, {"result": "{\"key\": \"value\"}", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_json_mode_default_false(self):
        """Without #:json, json_mode is false."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["json_mode"] is False

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()


# ============================================================
# Combined parameters
# ============================================================


class TestCombinedParams:
    def test_all_params_together(self):
        """All API params can be set simultaneously."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "Return json data" #:model "gpt-4o-mini" #:temperature 0.5 #:max-tokens 200 #:json #t)))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["model"] == "gpt-4o-mini"
            assert msg["temperature"] == 0.5
            assert msg["max_tokens"] == 200
            assert msg["json_mode"] is True

            write_response(proc, {"result": "{}", "prompt_tokens": 5, "completion_tokens": 2})
            read_msg(proc)
        finally:
            proc.kill()

    def test_unsafe_raw_query_params(self):
        """API params work on unsafe-raw-query too."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (unsafe-raw-query #:instruction "Return json output" #:temperature 0.8 #:max-tokens 100 #:json #t))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["temperature"] == 0.8
            assert msg["max_tokens"] == 100
            assert msg["json_mode"] is True

            write_response(proc, {"result": "{}", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_async_params_forwarded(self):
        """API params work on llm-query-async too."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define h (llm-query-async #:instruction "Return json response" #:temperature 1.0 #:max-tokens 300 #:json #t))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            assert msg["temperature"] == 1.0
            assert msg["max_tokens"] == 300
            assert msg["json_mode"] is True

            # Read eval result (no finish, just define)
            result = read_msg(proc)
            assert result["status"] == "ok"
        finally:
            proc.kill()


# ============================================================
# Image parameters
# ============================================================


class TestImageParams:
    def test_single_image_forwarded(self):
        """#:image value appears in the callback JSON as images list."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "describe" #:image "data:image/png;base64,abc123")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["images"] == ["data:image/png;base64,abc123"]

            write_response(proc, {"result": "a cat", "prompt_tokens": 10, "completion_tokens": 5})
            read_msg(proc)
        finally:
            proc.kill()

    def test_multiple_images_forwarded(self):
        """#:images list appears in the callback JSON."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "compare" #:images (list "img1" "img2" "img3"))))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["images"] == ["img1", "img2", "img3"]

            write_response(proc, {"result": "different", "prompt_tokens": 10, "completion_tokens": 5})
            read_msg(proc)
        finally:
            proc.kill()

    def test_image_and_images_merged(self):
        """#:image and #:images are merged (image prepended to list)."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "compare" #:image "first" #:images (list "second" "third"))))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["images"] == ["first", "second", "third"]

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_no_image_no_images_field(self):
        """Without #:image or #:images, the images field is absent."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "test")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert "images" not in msg

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_image_with_async(self):
        """#:image works with llm-query-async."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define h (llm-query-async #:instruction "describe" #:image "b64data"))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query-async"
            assert msg["images"] == ["b64data"]

            result = read_msg(proc)
            assert result["status"] == "ok"
        finally:
            proc.kill()

    def test_image_with_unsafe_raw_query(self):
        """#:image works with unsafe-raw-query."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (unsafe-raw-query #:instruction "describe" #:image "b64data"))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["images"] == ["b64data"]

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()

    def test_image_with_data(self):
        """#:image and #:data can be used together."""
        proc = start_server()
        try:
            write_cmd(proc, {"op": "eval", "code":
                '(define _r (syntax-e (llm-query #:instruction "analyze" #:data "context text" #:image "img_b64")))'
            })
            msg = read_msg(proc)
            assert msg["op"] == "llm-query"
            assert msg["data"] == "context text"
            assert msg["images"] == ["img_b64"]

            write_response(proc, {"result": "ok", "prompt_tokens": 0, "completion_tokens": 0})
            read_msg(proc)
        finally:
            proc.kill()
