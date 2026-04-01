# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM Chat Completions provider — OpenAI-compatible, manifest-aware.

Handles streaming, tool calling, thinking/reasoning traces, and
Nemotron 3 specific parameters (enable_thinking, thinking_budget, etc.).
Includes retry logic with exponential backoff for transient errors.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any, AsyncIterator

import httpx

from nemocode.config.schema import EndpointTier, Manifest
from nemocode.core.circuit_breaker import CircuitBreaker
from nemocode.core.logging_config import StructuredLogger
from nemocode.core.streaming import (
    CompletionResult,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from nemocode.providers import NIMProviderBase

logger = StructuredLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_LOCAL_BACKEND_LABELS = {
    EndpointTier.LOCAL_LLAMACPP: "llama.cpp",
    EndpointTier.LOCAL_NIM: "Local NIM",
    EndpointTier.LOCAL_OLLAMA: "Ollama",
    EndpointTier.LOCAL_SGLANG: "SGLang",
    EndpointTier.LOCAL_TRT_LLM: "TensorRT-LLM",
    EndpointTier.LOCAL_VLLM: "vLLM",
}
_LOCAL_SETUP_COMMANDS = {
    EndpointTier.LOCAL_LLAMACPP: "nemo setup llama-cpp",
    EndpointTier.LOCAL_NIM: "nemo setup nim",
    EndpointTier.LOCAL_OLLAMA: "nemo setup ollama",
    EndpointTier.LOCAL_SGLANG: "nemo setup sglang",
    EndpointTier.LOCAL_TRT_LLM: "nemo setup trt-llm",
    EndpointTier.LOCAL_VLLM: "nemo setup vllm",
}


def _is_verbose() -> bool:
    """Check if verbose debugging mode is enabled."""
    return os.environ.get("NEMOCODE_VERBOSE", "0") == "1"


async def _retry_delay(
    attempt: int, status_code: int | None = None, retry_after: str | None = None
) -> float:
    """Calculate and sleep for the appropriate retry delay.

    Uses Retry-After header if present (for 429), otherwise exponential backoff.
    """
    if retry_after and status_code == 429:
        try:
            delay = float(retry_after)
        except ValueError:
            delay = _RETRY_BASE_DELAY * (2**attempt)
    else:
        delay = _RETRY_BASE_DELAY * (2**attempt)
    # Cap at 30 seconds
    delay = min(delay, 30.0)
    logger.info(
        "api_retry",
        extra={
            "delay_s": delay,
            "attempt": attempt + 1,
            "max_retries": _MAX_RETRIES,
        },
    )
    await asyncio.sleep(delay)
    return delay


def _message_to_dict(msg: Message) -> dict[str, Any]:
    """Convert a Message to OpenAI-format dict."""
    d: dict[str, Any] = {"role": msg.role.value}

    if msg.role == Role.TOOL:
        d["content"] = msg.content
        d["tool_call_id"] = msg.tool_call_id or ""
        return d

    if msg.content:
        d["content"] = msg.content
    elif msg.role == Role.ASSISTANT:
        d["content"] = ""

    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in msg.tool_calls
        ]

    return d


class NIMChatProvider(NIMProviderBase):
    """OpenAI-compatible chat completions provider for NVIDIA NIM endpoints."""

    def __init__(
        self,
        endpoint,
        manifest: Manifest | None = None,
        api_key: str | None = None,
        endpoint_name: str | None = None,
    ) -> None:
        super().__init__(endpoint, api_key)
        self.manifest = manifest
        self.endpoint_name = endpoint_name
        self._circuit_breaker = CircuitBreaker()

    def _endpoint_label(self) -> str:
        return self.endpoint_name or self.endpoint.name or self.endpoint.model_id

    def _is_local_backend(self) -> bool:
        return self.endpoint.tier in _LOCAL_BACKEND_LABELS

    def _format_retry_message(
        self,
        detail: str,
        *,
        delay: float,
        status_code: int | None = None,
    ) -> str:
        label = self._endpoint_label()
        if self._is_local_backend():
            backend = _LOCAL_BACKEND_LABELS[self.endpoint.tier]
            readiness_note = (
                "The backend may still be loading the model."
                if status_code in {502, 503, 504}
                else "The server may not be running yet, or the model may still be loading."
            )
            return (
                f"\n[{backend} endpoint {label} at {self._base_url} is not reachable yet. "
                f"{readiness_note} Retrying in {delay:.0f}s...]\n"
            )
        return (
            f"\n[Connection error on {label} ({self._base_url}): {detail}. "
            f"Retrying in {delay:.0f}s...]\n"
        )

    def _classify_error(
        self,
        detail: str,
        *,
        status_code: int | None = None,
    ) -> dict[str, Any]:
        """Classify an error into a category with actionable guidance.

        Returns a dict with keys: category, message, suggestion.
        """
        lower_detail = detail.lower()

        # Auth errors (401, 403)
        if (
            status_code in {401, 403}
            or "unauthorized" in lower_detail
            or "forbidden" in lower_detail
        ):
            return {
                "category": "auth",
                "message": (
                    f"Authentication failed (HTTP {status_code or 'error'}): {detail}"
                    if detail
                    else f"Authentication failed (HTTP {status_code or 'error'}). "
                ),
                "suggestion": "Check your API key with: nemo auth",
            }

        # Quota/rate-limit errors (429)
        if status_code == 429 or "rate limit" in lower_detail or "quota" in lower_detail:
            return {
                "category": "quota",
                "message": (
                    f"Rate limited or quota exceeded (HTTP {status_code or '429'}): {detail}"
                    if detail
                    else f"Rate limited or quota exceeded (HTTP {status_code or '429'})."
                ),
                "suggestion": "Check endpoint status with: nemo endpoint status",
            }

        # Server errors (5xx)
        if status_code and 500 <= status_code < 600:
            return {
                "category": "server",
                "message": (
                    f"Server error (HTTP {status_code}): {detail}"
                    if detail
                    else (
                        f"Server error (HTTP {status_code}). "
                        "The service may be temporarily unavailable."
                    )
                ),
                "suggestion": "Check service health with: nemo doctor",
            }

        # Network errors (connection refused, timeout, etc.)
        network_indicators = [
            "connection refused",
            "connect timeout",
            "read timeout",
            "timed out",
            "network is unreachable",
            "name resolution",
            "ssl",
            "certificate",
        ]
        if any(indicator in lower_detail for indicator in network_indicators):
            return {
                "category": "network",
                "message": f"Network error: {detail}",
                "suggestion": "Check your connectivity and endpoint health with: nemo doctor",
            }

        # Fallback — include status code and detail
        parts = []
        if status_code:
            parts.append(f"HTTP {status_code}")
        if detail:
            parts.append(detail)
        return {
            "category": "unknown",
            "message": ": ".join(parts) if parts else "Unknown error",
            "suggestion": "Run: nemo doctor",
        }

    def _format_final_connection_error(
        self,
        detail: str,
        *,
        status_code: int | None = None,
        request_url: str | None = None,
        response_body: str | None = None,
    ) -> str:
        label = self._endpoint_label()
        if self._is_local_backend():
            backend = _LOCAL_BACKEND_LABELS[self.endpoint.tier]
            setup_cmd = _LOCAL_SETUP_COMMANDS.get(self.endpoint.tier)
            lines = [
                f"{backend} endpoint {label} at {self._base_url} is not reachable.",
                (
                    "The backend is up but not ready yet, or the model is still loading."
                    if status_code in {502, 503, 504}
                    else (
                        "The server may not be running yet, the model may still be loading, "
                        "or the port may be wrong."
                    )
                ),
            ]
            if self.endpoint_name:
                lines.append(f"Check it with: nemo endpoint test {self.endpoint_name}")
            if setup_cmd:
                lines.append(f"Setup/help: {setup_cmd}")
            if detail:
                lines.append(f"Last error: {detail}")
            return "\n".join(lines)

        # Hosted endpoint — classify and provide actionable guidance
        classification = self._classify_error(detail, status_code=status_code)
        lines = [
            f"{classification['message']}",
            f"Endpoint: {label} ({self._base_url})",
            classification["suggestion"],
        ]

        # Verbose mode: include full debugging details
        if _is_verbose():
            verbose_lines = ["", "--- Verbose Debug Info ---"]
            if request_url:
                verbose_lines.append(f"Request URL: {request_url}")
            if status_code:
                verbose_lines.append(f"HTTP Status: {status_code}")
            if response_body:
                verbose_lines.append(f"Response Body: {response_body}")
            if detail:
                verbose_lines.append(f"Full Error: {detail}")
            verbose_lines.append("--- End Verbose Debug Info ---")
            lines.extend(verbose_lines)

        return "\n".join(lines)

    def _build_body(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.endpoint.model_id,
            "messages": [_message_to_dict(m) for m in messages],
            "max_tokens": self.endpoint.max_tokens,
            "stream": stream,
            "temperature": 0.2,
            "top_p": 0.95,
        }

        # Apply manifest reasoning defaults via chat_template_kwargs
        # NVIDIA NIM expects thinking params inside extra_body.chat_template_kwargs
        if self.manifest and self.manifest.reasoning.supports_thinking:
            r = self.manifest.reasoning
            if r.thinking_param:
                chat_kwargs = body.setdefault("chat_template_kwargs", {})
                chat_kwargs[r.thinking_param] = True

        # Apply structured output response_format
        if response_format:
            if not isinstance(response_format, dict):
                raise ValueError("response_format must be a dict")
            if "type" not in response_format:
                raise ValueError("response_format must include 'type' key")
            body["response_format"] = response_format

        if tools:
            body["tools"] = tools
            # Some models need this to generate tool calls alongside content
            if self.manifest and self.manifest.force_nonempty_content:
                body["force_nonempty_content"] = True

        # Apply manifest extra body defaults
        if self.manifest and self.manifest.default_extra_body:
            body.update(self.manifest.default_extra_body)

        # Apply caller overrides
        if extra_body:
            body.update(extra_body)

        return body

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat completions, yielding chunks with text, thinking, tool calls, and usage.

        Retries on transient errors (429, 5xx, connection errors) with exponential backoff.
        Uses a circuit breaker for fast-fail when the endpoint is consistently failing.
        """
        # Fast-fail check via circuit breaker
        if not await self._circuit_breaker.can_execute():
            cooldown = self._circuit_breaker.cooldown_remaining
            logger.warning(
                "api_circuit_open",
                extra={
                    "endpoint_name": self.endpoint_name,
                    "cooldown_remaining_s": cooldown,
                },
            )
            yield StreamChunk(
                text=f"\n[Endpoint temporarily unavailable. Retrying in {cooldown:.0f}s...]\n",
                finish_reason="error",
            )
            return

        body = self._build_body(
            messages,
            tools=tools,
            stream=True,
            extra_body=extra_body,
            response_format=response_format,
        )
        url = f"{self._base_url}/chat/completions"

        last_error: str = ""
        last_request_url: str = url
        last_response_body: str = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                    async with client.stream(
                        "POST", url, json=body, headers=self._headers()
                    ) as resp:
                        if resp.status_code != 200:
                            error_body = await resp.aread()
                            error_text = error_body.decode(errors="replace")
                            last_response_body = error_text

                            # Retry on transient status codes
                            if (
                                resp.status_code in _RETRYABLE_STATUS_CODES
                                and attempt < _MAX_RETRIES
                            ):
                                retry_after = resp.headers.get("retry-after")
                                delay = await _retry_delay(attempt, resp.status_code, retry_after)
                                if resp.status_code == 429:
                                    yield StreamChunk(
                                        text=f"\n[Rate limited. Retrying in {delay:.0f}s...]\n"
                                    )
                                else:
                                    yield StreamChunk(
                                        text=self._format_retry_message(
                                            error_text,
                                            delay=delay,
                                            status_code=resp.status_code,
                                        )
                                    )
                                continue

                            yield StreamChunk(
                                text=self._format_final_connection_error(
                                    error_text,
                                    status_code=resp.status_code,
                                    request_url=last_request_url,
                                    response_body=error_text,
                                ),
                                finish_reason="error",
                            )
                            if resp.status_code == 429:
                                retry_after_hdr = resp.headers.get("retry-after")
                                if retry_after_hdr:
                                    try:
                                        await self._circuit_breaker.record_failure_with_retry_after(
                                            float(retry_after_hdr)
                                        )
                                    except ValueError:
                                        await self._circuit_breaker.record_failure()
                                else:
                                    await self._circuit_breaker.record_failure()
                            else:
                                await self._circuit_breaker.record_failure()
                            return

                        # Successful connection — stream the response
                        await self._circuit_breaker.record_success()
                        logger.info(
                            "api_request_start",
                            extra={
                                "endpoint_name": self.endpoint_name,
                                "method": "stream",
                            },
                        )
                        try:
                            async for chunk in self._process_stream(resp):
                                yield chunk
                        except httpx.RemoteProtocolError as exc:
                            # Server dropped mid-stream (e.g. OOM, crash).
                            # Cannot retry safely — partial data already yielded.
                            logger.warning("Server disconnected mid-stream: %s", exc)
                            yield StreamChunk(
                                text=f"\n[Server disconnected mid-stream: {exc}]\n",
                                finish_reason="error",
                            )
                        return  # Completed (possibly with mid-stream error)

            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.PoolTimeout,
                httpx.ConnectTimeout,
                httpx.RemoteProtocolError,
            ) as exc:
                last_error = str(exc)
                if attempt < _MAX_RETRIES:
                    delay = await _retry_delay(attempt)
                    yield StreamChunk(text=self._format_retry_message(last_error, delay=delay))
                    continue
            except httpx.ReadTimeout:
                last_error = "Request timed out"
                if attempt < _MAX_RETRIES:
                    delay = await _retry_delay(attempt)
                    yield StreamChunk(text=self._format_retry_message(last_error, delay=delay))
                    continue

        # All retries exhausted
        await self._circuit_breaker.record_failure()
        yield StreamChunk(
            text=self._format_final_connection_error(
                last_error,
                request_url=last_request_url,
                response_body=last_response_body,
            ),
            finish_reason="error",
        )

    async def _process_stream(self, resp: httpx.Response) -> AsyncIterator[StreamChunk]:
        """Process SSE lines from a successful streaming response."""
        tool_call_buffers: dict[int, dict[str, Any]] = {}

        async for line in resp.aiter_lines():
            # SSE spec: "data:" with or without trailing space
            if line.startswith("data: "):
                data = line[6:]
            elif line.startswith("data:"):
                data = line[5:].lstrip()
            else:
                continue
            if data.strip() == "[DONE]":
                break
            try:
                chunk_data = json.loads(data)
            except json.JSONDecodeError:
                continue

            choices = chunk_data.get("choices", [])
            if not choices:
                usage = chunk_data.get("usage")
                if usage:
                    logger.info(
                        "api_usage",
                        extra={
                            "endpoint_name": self.endpoint_name,
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                        },
                    )
                    yield StreamChunk(usage=usage)
                continue

            delta = choices[0].get("delta") or {}
            finish = choices[0].get("finish_reason")

            text = delta.get("content", "")
            thinking = delta.get("reasoning_content", "") or delta.get("thinking", "")

            # Handle streamed tool calls
            tc_deltas = delta.get("tool_calls") or []
            for tcd in tc_deltas:
                idx = tcd.get("index", 0)
                if idx not in tool_call_buffers:
                    tool_call_buffers[idx] = {
                        "id": tcd.get("id", f"tc_{uuid.uuid4().hex[:8]}"),
                        "name": "",
                        "arguments": "",
                    }
                buf = tool_call_buffers[idx]
                fn = tcd.get("function", {})
                if fn.get("name"):
                    buf["name"] = fn["name"]
                if fn.get("arguments"):
                    buf["arguments"] += fn["arguments"]

            if text or thinking:
                yield StreamChunk(text=text, thinking=thinking)

            if finish:
                completed_calls = []
                for buf in tool_call_buffers.values():
                    try:
                        args = json.loads(buf["arguments"]) if buf["arguments"] else {}
                    except json.JSONDecodeError:
                        args = {"_raw": buf["arguments"]}
                    completed_calls.append(
                        ToolCall(
                            id=buf["id"],
                            name=buf["name"],
                            arguments=args,
                        )
                    )
                usage = chunk_data.get("usage")
                yield StreamChunk(
                    tool_calls=completed_calls if completed_calls else [],
                    usage=usage,
                    finish_reason=finish,
                )

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> CompletionResult:
        """Non-streaming completion with retry logic.

        Uses a circuit breaker for fast-fail when the endpoint is consistently failing.
        """
        # Fast-fail check via circuit breaker
        if not await self._circuit_breaker.can_execute():
            cooldown = self._circuit_breaker.cooldown_remaining
            logger.warning(
                "api_circuit_open",
                extra={
                    "endpoint_name": self.endpoint_name,
                    "cooldown_remaining_s": cooldown,
                },
            )
            return CompletionResult(
                content=f"Endpoint temporarily unavailable. Retrying in {cooldown:.0f}s.",
                finish_reason="error",
            )

        body = self._build_body(
            messages,
            tools=tools,
            stream=False,
            extra_body=extra_body,
            response_format=response_format,
        )
        url = f"{self._base_url}/chat/completions"

        last_error = ""
        last_request_url = url
        last_response_body = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
                    resp = await client.post(url, json=body, headers=self._headers())

                    if resp.status_code != 200:
                        last_response_body = resp.text
                        if resp.status_code in _RETRYABLE_STATUS_CODES and attempt < _MAX_RETRIES:
                            retry_after = resp.headers.get("retry-after")
                            await _retry_delay(attempt, resp.status_code, retry_after)
                            continue
                        # Non-retryable or exhausted retries — record failure
                        if resp.status_code == 429:
                            retry_after_hdr = resp.headers.get("retry-after")
                            if retry_after_hdr:
                                try:
                                    await self._circuit_breaker.record_failure_with_retry_after(
                                        float(retry_after_hdr)
                                    )
                                except ValueError:
                                    await self._circuit_breaker.record_failure()
                            else:
                                await self._circuit_breaker.record_failure()
                        else:
                            await self._circuit_breaker.record_failure()
                        return CompletionResult(
                            content=self._format_final_connection_error(
                                resp.text,
                                status_code=resp.status_code,
                                request_url=last_request_url,
                                response_body=resp.text,
                            ),
                            finish_reason="error",
                        )
                    data = resp.json()
                    await self._circuit_breaker.record_success()
                    break  # success

            except (
                httpx.ConnectError,
                httpx.ReadError,
                httpx.WriteError,
                httpx.PoolTimeout,
                httpx.ConnectTimeout,
                httpx.ReadTimeout,
            ) as exc:
                last_error = str(exc)
                if attempt < _MAX_RETRIES:
                    await _retry_delay(attempt)
                    continue
                await self._circuit_breaker.record_failure()
                return CompletionResult(
                    content=self._format_final_connection_error(
                        last_error,
                        request_url=last_request_url,
                        response_body=last_response_body,
                    ),
                    finish_reason="error",
                )
        else:
            return CompletionResult(
                content=self._format_final_connection_error(
                    last_error,
                    request_url=last_request_url,
                    response_body=last_response_body,
                ),
                finish_reason="error",
            )

        choices = data.get("choices", [])
        choice = choices[0] if choices else {}
        message = choice.get("message") or {}

        usage = data.get("usage", {})
        if usage:
            logger.info(
                "api_complete",
                extra={
                    "endpoint_name": self.endpoint_name,
                    "method": "complete",
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "finish_reason": choice.get("finish_reason", ""),
                },
            )

        tool_calls = []
        for tc_data in message.get("tool_calls") or []:
            fn = tc_data.get("function", {})
            try:
                args = json.loads(fn.get("arguments", "{}"))
            except json.JSONDecodeError:
                args = {}
            tool_calls.append(
                ToolCall(
                    id=tc_data.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=args,
                )
            )

        return CompletionResult(
            content=message.get("content") or "",
            thinking=message.get("reasoning_content") or message.get("thinking") or "",
            tool_calls=tool_calls,
            usage=usage,
            finish_reason=choice.get("finish_reason", ""),
        )
