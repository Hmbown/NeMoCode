# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test credential management — keyring storage and env fallback."""

from __future__ import annotations

import os
from unittest.mock import patch

from nemocode.core.credentials import (
    _mask,
    get_credential,
    list_credentials,
    set_credential,
    storage_backend,
)


class TestCredentialMasking:
    def test_mask_short(self):
        assert _mask("abc") == "***"

    def test_mask_medium(self):
        result = _mask("nvapi-1234567890")
        assert result.startswith("nvap")
        assert result.endswith("7890")
        assert "*" in result

    def test_mask_preserves_length(self):
        value = "nvapi-abcdefghijklmnop"
        result = _mask(value)
        assert len(result) == len(value)


class TestEnvFallback:
    def test_get_from_env(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key-123"}):
            with patch("nemocode.core.credentials._keyring_available", return_value=False):
                result = get_credential("NVIDIA_API_KEY")
                assert result == "test-key-123"

    def test_get_missing(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("nemocode.core.credentials._keyring_available", return_value=False):
                result = get_credential("NONEXISTENT_KEY")
                assert result is None

    def test_set_without_keyring(self):
        with patch("nemocode.core.credentials._keyring_available", return_value=False):
            success, method = set_credential("TEST_KEY", "value")
            assert success is False
            assert method == "none"


class TestListCredentials:
    def test_list_all_known_keys(self):
        with patch("nemocode.core.credentials._keyring_available", return_value=False):
            with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test"}, clear=False):
                creds = list_credentials()
                assert "NVIDIA_API_KEY" in creds
                assert creds["NVIDIA_API_KEY"]["configured"] is True
                assert creds["NVIDIA_API_KEY"]["source"] == "env"

    def test_unconfigured_key(self):
        with patch("nemocode.core.credentials._keyring_available", return_value=False):
            with patch.dict(os.environ, {}, clear=False):
                # Ensure FIREWORKS_API_KEY is not in env
                os.environ.pop("FIREWORKS_API_KEY", None)
                creds = list_credentials()
                if not os.environ.get("FIREWORKS_API_KEY"):
                    assert creds["FIREWORKS_API_KEY"]["configured"] is False


class TestStorageBackend:
    def test_backend_without_keyring(self):
        with patch("nemocode.core.credentials._keyring_available", return_value=False):
            assert storage_backend() == "environment-only"
