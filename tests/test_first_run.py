# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for first-run experience."""

from __future__ import annotations

from unittest.mock import patch

from nemocode.core.first_run import (
    has_any_api_key,
    is_first_run,
    mark_first_run_done,
)


class TestFirstRun:
    def test_is_first_run_when_no_marker(self, tmp_path):
        with patch("nemocode.core.first_run._FIRST_RUN_MARKER", tmp_path / "marker"):
            assert is_first_run()

    def test_is_not_first_run_when_marker_exists(self, tmp_path):
        marker = tmp_path / "marker"
        marker.touch()
        with patch("nemocode.core.first_run._FIRST_RUN_MARKER", marker):
            assert not is_first_run()

    def test_mark_first_run_creates_marker(self, tmp_path):
        marker = tmp_path / "marker"
        with patch("nemocode.core.first_run._FIRST_RUN_MARKER", marker):
            with patch("nemocode.core.first_run._USER_CONFIG_DIR", tmp_path):
                mark_first_run_done()
        assert marker.exists()


class TestAPIKeyDetection:
    def test_no_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("nemocode.core.credentials.get_credential", return_value=None):
                assert not has_any_api_key()

    def test_has_nvidia_key_in_env(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "nvapi-test"}):
            assert has_any_api_key()

    def test_has_openrouter_key_in_env(self):
        with patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-or-test"}, clear=True):
            assert has_any_api_key()
