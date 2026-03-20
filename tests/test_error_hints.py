# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Parametrized tests covering all error patterns in _error_hint()."""

from __future__ import annotations

import pytest

from nemocode.cli.render import _error_hint


class TestErrorHintPatterns:
    """Exhaustive parametrized coverage of every error pattern in _error_hint."""

    @pytest.mark.parametrize(
        "error_text,expected_substring",
        [
            # Rate limit (429)
            ("Rate limit exceeded (429)", "endpoint"),
            ("429 Too Many Requests", "endpoint"),
            ("rate limit reached", "endpoint"),
            # Auth (401)
            ("401 Unauthorized", "API key"),
            ("Authentication failed", "API key"),
            ("Invalid api key provided", "API key"),
            # Forbidden (403)
            ("403 Forbidden", "permission"),
            ("Access forbidden for user", "permission"),
            # Not found (404)
            ("404 Not Found", "model"),
            ("Resource not found on server", "model"),
            # Timeout
            ("Request timed out after 30s", "timed out"),
            ("Connection timeout", "timed out"),
            # Context window
            ("context length exceeded", "compact"),
            ("context window limit reached", "compact"),
            ("This model's maximum context length is exceeded", "compact"),
            # Network / connection
            ("Connection refused by remote host", "connection"),
            ("Network error: unreachable", "connection"),
            # SSL
            ("SSL certificate error: unable to verify", "ssl"),
            ("Certificate verification failed", "ssl"),
            # DNS
            ("DNS resolution failed for api.nvidia.com", "dns"),
            ("Could not resolve hostname", "dns"),
            # Connection refused (econnrefused)
            ("ECONNREFUSED 127.0.0.1:8000", "refused"),
            ("Connection refused at port 443", "connection"),
            # Invalid request (400)
            ("400 Bad Request: Invalid request body", "invalid"),
            ("Invalid request parameters", "invalid"),
            # Internal server error (500)
            ("500 Internal Server Error", "server"),
            ("Internal server error occurred", "server"),
            # Bad gateway (502)
            ("502 Bad Gateway", "bad gateway"),
            ("Bad gateway error from upstream", "bad gateway"),
            # Service unavailable (503)
            ("503 Service Unavailable", "service unavailable"),
            ("Service unavailable due to maintenance", "service unavailable"),
            # Gateway timeout (504) — "timeout" matches before "gateway timeout"
            ("504 Gateway Timeout", "timed out"),
            ("Gateway timeout from proxy", "timed out"),
        ],
        ids=[
            "rate_limit_429",
            "rate_limit_too_many",
            "rate_limit_text",
            "auth_401",
            "auth_failed",
            "auth_api_key",
            "forbidden_403",
            "forbidden_text",
            "not_found_404",
            "not_found_text",
            "timeout_request",
            "timeout_connection",
            "context_length",
            "context_window",
            "context_max",
            "network_refused",
            "network_error",
            "ssl_cert",
            "ssl_verify",
            "dns_resolution",
            "dns_resolve",
            "econnrefused",
            "conn_refused",
            "invalid_400",
            "invalid_params",
            "server_500",
            "server_internal",
            "bad_gateway_502",
            "bad_gateway_text",
            "unavailable_503",
            "unavailable_text",
            "gateway_timeout_504",
            "gateway_timeout_text",
        ],
    )
    def test_error_pattern_returns_hint(self, error_text: str, expected_substring: str):
        hint = _error_hint(error_text)
        assert hint, f"Expected a hint for: {error_text!r}"
        assert expected_substring.lower() in hint.lower(), (
            f"Expected {expected_substring!r} in hint for {error_text!r}, got: {hint!r}"
        )


class TestErrorHintNoMatch:
    """Verify that generic errors return an empty string."""

    @pytest.mark.parametrize(
        "error_text",
        [
            "Something unexpected happened",
            "An unknown error occurred",
            "Oops",
            "",
            "TypeError: cannot add int and str",
            "ValueError: invalid literal",
            (
                "SGLang endpoint spark-sglang-super at http://localhost:8000/v1 is not reachable.\n"
                "Check it with: nemo endpoint test spark-sglang-super\n"
                "Setup/help: nemo setup sglang"
            ),
        ],
        ids=[
            "generic_unexpected",
            "generic_unknown",
            "generic_short",
            "empty_string",
            "type_error",
            "value_error",
            "actionable_endpoint_error",
        ],
    )
    def test_no_hint_for_unrecognized_error(self, error_text: str):
        assert _error_hint(error_text) == ""


class TestErrorHintCaseInsensitivity:
    """Verify that pattern matching is case-insensitive."""

    def test_uppercase_rate_limit(self):
        assert _error_hint("RATE LIMIT EXCEEDED") != ""

    def test_mixed_case_auth(self):
        assert _error_hint("Authentication Required") != ""

    def test_lowercase_ssl(self):
        assert _error_hint("ssl certificate problem") != ""

    def test_uppercase_dns(self):
        assert _error_hint("DNS RESOLUTION FAILED") != ""

    def test_mixed_case_context(self):
        assert _error_hint("Context Length Exceeded") != ""
