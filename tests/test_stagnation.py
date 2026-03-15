# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for stagnation detection and tool output truncation."""

from __future__ import annotations

from nemocode.core.scheduler import _StagnationTracker, _truncate_output


class TestStagnationTracker:
    def test_no_stagnation_initially(self):
        tracker = _StagnationTracker()
        assert tracker.is_stagnant() is None

    def test_repeated_tool_calls(self):
        tracker = _StagnationTracker()
        for _ in range(5):
            tracker.record_tool_call("read_file", {"path": "foo.py"})
        assert tracker.is_stagnant() is not None
        assert "repeated" in tracker.is_stagnant().lower()

    def test_different_tool_calls_no_stagnation(self):
        tracker = _StagnationTracker()
        for _ in range(20):
            tracker.record_tool_call("read_file", {"path": f"file_{_}.py"})
        assert tracker.is_stagnant() is None

    def test_repeated_errors(self):
        tracker = _StagnationTracker()
        for _ in range(5):
            tracker.record_error("File not found: foo.py")
        assert tracker.is_stagnant() is not None
        assert "error" in tracker.is_stagnant().lower()

    def test_idle_detection(self):
        """Only truly idle rounds (no text, no tools) trigger stagnation."""
        tracker = _StagnationTracker()
        for _ in range(15):
            tracker.record_turn("", had_tool_calls=False)
        assert tracker.is_stagnant() is not None
        assert "activity" in tracker.is_stagnant().lower()

    def test_tool_calls_count_as_progress(self):
        """Rounds with tool calls should NOT count as idle."""
        tracker = _StagnationTracker()
        for _ in range(20):
            tracker.record_turn("", had_tool_calls=True)
        assert tracker.is_stagnant() is None

    def test_text_output_counts_as_progress(self):
        tracker = _StagnationTracker()
        for _ in range(20):
            tracker.record_turn("some text")
        assert tracker.is_stagnant() is None

    def test_mixed_activity_no_stagnation(self):
        """Alternating text and tool rounds should not trigger stagnation."""
        tracker = _StagnationTracker()
        for i in range(20):
            if i % 3 == 0:
                tracker.record_turn("text output")
            else:
                tracker.record_turn("", had_tool_calls=True)
        assert tracker.is_stagnant() is None

    def test_reset(self):
        tracker = _StagnationTracker()
        for _ in range(5):
            tracker.record_tool_call("read_file", {"path": "foo.py"})
        assert tracker.is_stagnant() is not None
        tracker.reset()
        assert tracker.is_stagnant() is None


class TestTruncation:
    def test_short_output_unchanged(self):
        result = _truncate_output("short output", "test_tool")
        assert result == "short output"

    def test_long_output_truncated(self, tmp_path, monkeypatch):
        monkeypatch.setattr("nemocode.core.scheduler._SCRATCH_DIR", tmp_path)
        long_text = "x" * 30_000
        result = _truncate_output(long_text, "test_tool")
        assert "truncated" in result
        assert "30,000 chars" in result
        # Check scratch file was created
        scratch_files = list(tmp_path.glob("test_tool_*.txt"))
        assert len(scratch_files) == 1
        assert scratch_files[0].read_text() == long_text
