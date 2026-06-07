"""LLM provider: OpenAI calls with retry, backoff, and token accounting."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

from .models import RetryConfig, RetrySettings, TokenUsage

logger = logging.getLogger(__name__)


class LLMCallResult:
    """Result of a single LLM call."""

    def __init__(
        self,
        text: str,
        tokens: TokenUsage,
        model: str,
        elapsed: float,
        cached: bool = False,
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.model = model
        self.elapsed = elapsed
        self.cached = cached


class LLMProvider:
    """Makes OpenAI API calls with retry and token tracking."""

    def __init__(
        self,
        api_key: str | None = None,
        retry_settings: RetrySettings | None = None,
    ) -> None:
        self._api_key = api_key
        self._retry = retry_settings or RetrySettings()
        self._total_tokens = TokenUsage()
        self._total_calls = 0
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            import openai
            self._client = openai.AsyncOpenAI(api_key=self._api_key)
        return self._client

    def _get_retry_config(self, model: str) -> RetryConfig:
        overrides = self._retry.per_model_overrides.get(model)
        if overrides:
            return overrides
        return self._retry.defaults

    async def call(
        self,
        instruction: str,
        data: Any,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        images: list[dict[str, Any]] | None = None,
    ) -> LLMCallResult:
        """Make an LLM call with retry logic."""
        client = self._get_client()
        retry_cfg = self._get_retry_config(model)

        messages = self._build_messages(instruction, data, images)
        kwargs: dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Exception | None = None
        for attempt in range(retry_cfg.max_retries + 1):
            start = time.monotonic()
            try:
                response = await client.chat.completions.create(**kwargs)
                elapsed = time.monotonic() - start

                text = response.choices[0].message.content or ""
                usage = response.usage
                tokens = TokenUsage(
                    prompt=usage.prompt_tokens if usage else 0,
                    completion=usage.completion_tokens if usage else 0,
                    total=usage.total_tokens if usage else 0,
                )
                self._total_tokens.prompt += tokens.prompt
                self._total_tokens.completion += tokens.completion
                self._total_tokens.total += tokens.total
                self._total_calls += 1

                return LLMCallResult(
                    text=text, tokens=tokens, model=model, elapsed=elapsed
                )
            except Exception as e:
                last_error = e
                error_type = _classify_error(e)
                if error_type not in retry_cfg.retryable_error_types:
                    raise
                if attempt < retry_cfg.max_retries:
                    backoff = min(
                        retry_cfg.initial_backoff_seconds
                        * (retry_cfg.backoff_multiplier ** attempt),
                        retry_cfg.max_backoff_seconds,
                    )
                    # Check for Retry-After header
                    retry_after = getattr(e, "retry_after", None)
                    if retry_after is not None:
                        backoff = max(backoff, float(retry_after))
                    logger.warning(
                        "LLM call attempt %d failed (%s), retrying in %.1fs",
                        attempt + 1, error_type, backoff,
                    )
                    await asyncio.sleep(backoff)

        raise last_error  # type: ignore[misc]

    def _build_messages(
        self,
        instruction: str,
        data: Any,
        images: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Build chat messages from instruction + data."""
        content: list[dict[str, Any]] | str
        if images:
            parts: list[dict[str, Any]] = [{"type": "text", "text": instruction}]
            if data is not None:
                data_str = json.dumps(data, default=str) if not isinstance(data, str) else data
                parts.append({"type": "text", "text": f"\n\nData:\n{data_str}"})
            for img in images:
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": img.get("url", img.get("data", ""))},
                })
            content = parts
        else:
            data_str = ""
            if data is not None:
                data_str = json.dumps(data, default=str) if not isinstance(data, str) else data
            content = f"{instruction}\n\nData:\n{data_str}" if data_str else instruction

        return [{"role": "user", "content": content}]

    @property
    def total_tokens(self) -> TokenUsage:
        return self._total_tokens

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def reset_accounting(self) -> None:
        self._total_tokens = TokenUsage()
        self._total_calls = 0


class DryRunProvider(LLMProvider):
    """Stub provider that simulates LLM calls without hitting the API."""

    async def call(
        self,
        instruction: str,
        data: Any,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        images: list[dict[str, Any]] | None = None,
    ) -> LLMCallResult:
        # Estimate tokens from instruction + data length
        text_len = len(instruction) + len(json.dumps(data, default=str) if data else "")
        est_prompt = text_len // 4
        est_completion = max(est_prompt // 2, 50)
        tokens = TokenUsage(
            prompt=est_prompt,
            completion=est_completion,
            total=est_prompt + est_completion,
        )
        self._total_tokens.prompt += tokens.prompt
        self._total_tokens.completion += tokens.completion
        self._total_tokens.total += tokens.total
        self._total_calls += 1

        result_text = '{"simulated": true}' if json_mode else "[dry-run result]"
        return LLMCallResult(
            text=result_text, tokens=tokens, model=model, elapsed=0.0
        )


def content_hash(instruction: str, data: Any, model: str, temperature: float) -> str:
    """Compute a canonical content hash for cache lookup."""
    canonical = json.dumps(
        {"instruction": instruction, "data": data, "model": model, "temperature": temperature},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _classify_error(exc: Exception) -> str:
    """Classify an exception into an error type for retry decisions."""
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name:
        return "rate_limit"
    if "Timeout" in cls_name or "ETIMEDOUT" in str(exc):
        return "timeout"
    if "APIConnection" in cls_name or "Server" in cls_name:
        return "server_error"
    # Check for HTTP status codes
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status:
        if status == 429:
            return "rate_limit"
        if status in (408,):
            return "timeout"
        if status in (500, 502, 503, 504):
            return "server_error"
    return "client_error"
