# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for skills framework."""

from __future__ import annotations

from nemocode.skills import SkillRegistry, create_default_registry


class TestSkillRegistry:
    def test_default_registry_has_skills(self):
        registry = create_default_registry()
        skills = registry.list_skills()
        names = {s.name for s in skills}
        assert "commit" in names
        assert "review" in names

    def test_get_skill(self):
        registry = create_default_registry()
        commit = registry.get("commit")
        assert commit is not None
        assert commit.name == "commit"
        assert commit.description

    def test_get_unknown_skill(self):
        registry = create_default_registry()
        assert registry.get("nonexistent") is None

    def test_register_custom_skill(self):
        from nemocode.skills import Skill

        class CustomSkill(Skill):
            name = "custom"
            description = "A custom skill"

            async def run(self, args, context):
                return "custom result"

        registry = SkillRegistry()
        registry.register(CustomSkill())
        assert registry.get("custom") is not None
