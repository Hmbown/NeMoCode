# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for smart formation auto-routing."""

from __future__ import annotations

from nemocode.core.router import TaskComplexity, classify_task


class TestClassifyTask:
    def test_simple_question(self):
        assert classify_task("what is this function?") == TaskComplexity.SIMPLE

    def test_simple_short(self):
        assert classify_task("explain foo") == TaskComplexity.SIMPLE

    def test_review_request(self):
        assert classify_task("review my changes") == TaskComplexity.REVIEW

    def test_complex_refactor(self):
        assert classify_task("refactor the authentication module") == TaskComplexity.COMPLEX

    def test_complex_multi_file(self):
        inp = "implement auth and add tests across all modules"
        assert classify_task(inp) == TaskComplexity.COMPLEX

    def test_moderate_default(self):
        assert classify_task("add error handling to the parser") == TaskComplexity.MODERATE

    def test_very_short_is_simple(self):
        assert classify_task("hi") == TaskComplexity.SIMPLE

    def test_long_input_is_complex(self):
        long_input = " ".join(["word"] * 25)
        assert classify_task(long_input) == TaskComplexity.COMPLEX

    def test_find_bugs(self):
        assert classify_task("find bugs in the scheduler") == TaskComplexity.REVIEW

    def test_where_is(self):
        assert classify_task("where is the config loaded?") == TaskComplexity.SIMPLE
