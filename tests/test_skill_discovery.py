# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for skill auto-discovery from SKILL.md files."""

from __future__ import annotations

from nemocode.skills import MarkdownSkill, SkillRegistry, _parse_frontmatter


class TestFrontmatterParsing:
    def test_basic_frontmatter(self):
        content = """---
name: test
description: A test skill
usage: /test [args]
---

Do the thing.
"""
        fm, body = _parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "A test skill"
        assert body.strip() == "Do the thing."

    def test_no_frontmatter(self):
        content = "Just a plain markdown file."
        fm, body = _parse_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_quoted_values(self):
        content = '---\nname: "my-skill"\n---\nbody'
        fm, _ = _parse_frontmatter(content)
        assert fm["name"] == "my-skill"


class TestSkillDiscovery:
    def test_discover_skill_md(self, tmp_path):
        (tmp_path / "SKILL.md").write_text(
            "---\nname: deploy\ndescription: Deploy the app\n---\nRun deploy steps."
        )
        registry = SkillRegistry()
        count = registry.discover_from_directory(tmp_path)
        assert count == 1
        skill = registry.get("deploy")
        assert skill is not None
        assert skill.description == "Deploy the app"

    def test_discover_named_skill_md(self, tmp_path):
        (tmp_path / "lint.skill.md").write_text(
            "---\nname: lint\ndescription: Run linter\n---\nRun ruff check ."
        )
        registry = SkillRegistry()
        count = registry.discover_from_directory(tmp_path)
        assert count == 1
        assert registry.get("lint") is not None

    def test_discover_skills_dir(self, tmp_path):
        skills_dir = tmp_path / ".nemocode" / "skills"
        skills_dir.mkdir(parents=True)
        (skills_dir / "format.md").write_text(
            "---\nname: format\n---\nRun formatter."
        )
        registry = SkillRegistry()
        count = registry.discover_from_directory(tmp_path)
        assert count == 1
        assert registry.get("format") is not None

    def test_no_override_builtin(self, tmp_path):
        """Built-in skills should not be overridden."""
        (tmp_path / "SKILL.md").write_text(
            "---\nname: commit\n---\nCustom commit."
        )
        registry = SkillRegistry()
        # Register a "builtin" commit skill
        class FakeSkill(MarkdownSkill):
            pass
        registry.register(FakeSkill("commit", "built-in", "built-in prompt"))
        count = registry.discover_from_directory(tmp_path)
        assert count == 0  # Should not override

    def test_derive_name_from_filename(self, tmp_path):
        """When no name in frontmatter, derive from filename."""
        (tmp_path / "migrate.skill.md").write_text("Run migrations.")
        registry = SkillRegistry()
        count = registry.discover_from_directory(tmp_path)
        assert count == 1
        assert registry.get("migrate") is not None

    def test_empty_directory(self, tmp_path):
        registry = SkillRegistry()
        count = registry.discover_from_directory(tmp_path)
        assert count == 0
