# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for diff stats and diff bar formatting in render.py."""

from __future__ import annotations

import json
from io import StringIO

from rich.console import Console

from nemocode.cli.render import (
    _diff_stats,
    _format_diff_bar,
    render_tool_result,
)


class TestDiffStats:
    def test_simple_diff(self):
        diff = "+++ a.py\n--- b.py\n@@ -1 +1 @@\n-old\n+new\n"
        added, removed = _diff_stats(diff)
        assert added == 1
        assert removed == 1

    def test_additions_only(self):
        diff = "+++ a.py\n--- b.py\n@@ -1 +1,3 @@\n line1\n+added1\n+added2\n"
        added, removed = _diff_stats(diff)
        assert added == 2
        assert removed == 0

    def test_removals_only(self):
        diff = "+++ a.py\n--- b.py\n@@ -1,3 +1 @@\n line1\n-del1\n-del2\n"
        added, removed = _diff_stats(diff)
        assert added == 0
        assert removed == 2

    def test_mixed_diff(self):
        diff = (
            "+++ a.py\n--- b.py\n@@ -1,5 +1,6 @@\n"
            " context\n-removed1\n-removed2\n+added1\n+added2\n+added3\n context\n"
        )
        added, removed = _diff_stats(diff)
        assert added == 3
        assert removed == 2

    def test_ignores_header_lines(self):
        """Lines starting with +++ or --- should not be counted."""
        diff = "+++ a.py\n--- b.py\n@@ -1 +1 @@\n+new line\n"
        added, removed = _diff_stats(diff)
        assert added == 1
        assert removed == 0

    def test_empty_diff(self):
        added, removed = _diff_stats("")
        assert added == 0
        assert removed == 0

    def test_context_only_diff(self):
        diff = "+++ a.py\n--- b.py\n@@ -1,2 +1,2 @@\n line1\n line2\n"
        added, removed = _diff_stats(diff)
        assert added == 0
        assert removed == 0


class TestFormatDiffBar:
    def test_empty(self):
        assert _format_diff_bar(0, 0) == ""

    def test_additions_only(self):
        result = _format_diff_bar(3, 0)
        assert result == "+3 -0 +++"

    def test_removals_only(self):
        result = _format_diff_bar(0, 2)
        assert result == "+0 -2 --"

    def test_mixed(self):
        result = _format_diff_bar(5, 2)
        assert result == "+5 -2 +++++--"  # 5 + 2 = 7 <= 10, shown directly

    def test_large_numbers_scale(self):
        """When total exceeds max_width, bars should scale proportionally."""
        result = _format_diff_bar(50, 50)
        assert "+50" in result
        assert "-50" in result
        # Bar portion should be at most ~10 chars
        bar = result.split()[-1]
        assert len(bar) <= 12  # some tolerance

    def test_single_addition(self):
        result = _format_diff_bar(1, 0)
        assert result == "+1 -0 +"

    def test_single_removal(self):
        result = _format_diff_bar(0, 1)
        assert result == "+0 -1 -"


class TestDiffStatsInline:
    """Verify diff stats appear inline on tool result success lines."""

    def _capture_console(self):
        buf = StringIO()
        con = Console(file=buf, no_color=True, width=120)
        return con, buf

    def test_edit_shows_diff_stats(self):
        con, buf = self._capture_console()
        diff = "--- a.py\n+++ a.py\n@@ -1,3 +1,5 @@\n line\n+new1\n+new2\n-old\n line\n"
        result = json.dumps({"status": "ok", "path": "a.py", "diff": diff})
        render_tool_result(con, "edit_file", result, False, elapsed=1.5)
        output = buf.getvalue()
        assert "+2/-1" in output

    def test_write_no_diff_no_stats(self):
        con, buf = self._capture_console()
        result = json.dumps({"status": "ok", "bytes": 500})
        render_tool_result(con, "write_file", result, False)
        output = buf.getvalue()
        assert "500" in output
        # No +/- stats without a diff
        assert "+0" not in output

    def test_empty_diff_no_stats(self):
        con, buf = self._capture_console()
        result = json.dumps({"status": "ok", "path": "a.py", "diff": ""})
        render_tool_result(con, "edit_file", result, False)
        output = buf.getvalue()
        # Should not show +/- for empty diff
        assert "+0" not in output
