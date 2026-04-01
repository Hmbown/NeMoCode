# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Secret scanning and redaction for tool output."""

from __future__ import annotations

import re
from typing import ClassVar


class SecretScanner:
    """Scan text for secret-like patterns and redact them.

    This scanner uses a curated list of regex patterns to detect common
    credential formats in tool output (bash stdout/stderr, file contents,
    etc.) and replaces matches with ``[REDACTED]``.

    The scanner is conservative by design — it prioritises avoiding
    false positives over catching every possible secret.  Normal code
    patterns (variable names, file paths, short hex strings) are
    intentionally left untouched.

    Attributes:
        enabled: Whether scanning is active.  Set to ``False`` in
            trusted environments to skip redaction entirely.
    """

    # Patterns are compiled once at class definition time.
    # Order matters — more specific patterns should come first so they
    # consume the match before a broader pattern can fire.
    _PATTERNS: ClassVar[list[tuple[str, re.Pattern[str]]]] = [
        # ── Private key blocks ──────────────────────────────────────────
        (
            "private_key_block",
            re.compile(
                r"-----BEGIN\s+(?:RSA\s+)?(?:EC\s+)?(?:DSA\s+)?(?:OPENSSH\s+)?(?:ENCRYPTED\s+)?PRIVATE\s+KEY-----"
                r".*?"
                r"-----END\s+(?:RSA\s+)?(?:EC\s+)?(?:DSA\s+)?(?:OPENSSH\s+)?(?:ENCRYPTED\s+)?PRIVATE\s+KEY-----",
                re.DOTALL,
            ),
        ),
        # ── JWT tokens (three base64url segments starting with eyJ) ──────
        (
            "jwt_token",
            re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
        ),
        # ── AWS access key IDs ──────────────────────────────────────────
        (
            "aws_access_key",
            re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        ),
        # ── AWS secret access keys (40-char base64 after aws_secret) ────
        (
            "aws_secret_key",
            re.compile(r"(?i)(?:aws_secret_access_key|aws_secret_key)\s*[=:]\s*[A-Za-z0-9/+=]{40}"),
        ),
        # ── GitHub personal access / OAuth tokens ───────────────────────
        (
            "github_token",
            re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{20,}\b"),
        ),
        # ── Slack tokens ────────────────────────────────────────────────
        (
            "slack_token",
            re.compile(r"\bxox[bpsao]-[A-Za-z0-9-]+"),
        ),
        # ── Bearer tokens ───────────────────────────────────────────────
        (
            "bearer_token",
            re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.=]{20,}"),
        ),
        # ── Generic key=value assignments ───────────────────────────────
        #    Matches: api_key=..., apikey=..., API_KEY=..., password=...,
        #    secret=..., token=..., auth=..., credential=..., private_key=...
        #    Value must be 16+ non-whitespace chars to avoid false positives
        #    on normal config values.
        (
            "generic_key_value",
            re.compile(
                r"(?i)(?:api_?key|secret(?:_key)?|password|passwd|pwd|token|auth(?:_token)?|credential|private_key|access_key)\s*[=:]\s*\S{16,}"
            ),
        ),
        # ── Long hex strings that look like tokens ──────────────────────
        #    32+ consecutive hex chars, not preceded by 0x (code literal)
        #    and not part of a typical path or word.
        (
            "hex_token",
            re.compile(r"(?<![0-9a-fA-FxX_])[0-9a-fA-F]{32,}(?![0-9a-fA-F_])"),
        ),
    ]

    def __init__(self, *, enabled: bool = True) -> None:
        """Initialise the scanner.

        Args:
            enabled: If ``False``, ``scan_and_redact`` returns input
                unchanged.  Useful for trusted environments or testing.
        """
        self.enabled = enabled

    def scan_and_redact(self, text: str) -> str:
        """Find and replace secret-like patterns in *text*.

        Each pattern is applied in order; once a region of the string has
        been consumed by an earlier pattern it will not be matched again.

        Args:
            text: Arbitrary text that may contain secrets.

        Returns:
            The input text with every detected secret replaced by
            ``[REDACTED]``.  If the scanner is disabled, the original
            text is returned unchanged.
        """
        if not self.enabled:
            return text

        result = text
        for _name, pattern in self._PATTERNS:
            result = pattern.sub("[REDACTED]", result)
        return result


# Module-level singleton — import and call ``scanner.scan_and_redact(text)``
# or toggle ``scanner.enabled = False`` to disable globally.
scanner = SecretScanner()
