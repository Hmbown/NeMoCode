# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Skills framework — LLM-powered slash commands.

Skills differ from plain slash commands: they gather context, prompt the model,
and execute the result. Think of them as mini-workflows triggered by /<name>.

Supports auto-discovery from SKILL.md files in the project directory.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Skill(ABC):
    """Base class for skills."""

    name: str = ""
    description: str = ""
    usage: str = ""

    @abstractmethod
    async def run(self, args: str, context: dict[str, Any]) -> str | None:
        """Execute the skill.

        args: The argument string after the slash command
        context: Dict with 'config', 'agent', 'console' keys
        Returns: Output string to display, or None if handled internally
        """
        ...


class MarkdownSkill(Skill):
    """A skill defined by a SKILL.md file.

    The markdown content is injected as a system prompt for the agent,
    which then executes the skill's intent.
    """

    def __init__(self, name: str, description: str, prompt: str, usage: str = "") -> None:
        self.name = name
        self.description = description
        self.usage = usage
        self._prompt = prompt

    async def run(self, args: str, context: dict[str, Any]) -> str | None:
        """Execute by sending the skill prompt to the agent."""
        agent = context.get("agent")
        if agent is None:
            return f"[Error: No agent available for skill /{self.name}]"

        # Build the skill prompt with args
        full_prompt = self._prompt
        if args:
            full_prompt = f"{self._prompt}\n\nUser arguments: {args}"

        # Run through the agent and collect output
        output_parts: list[str] = []
        async for event in agent.run(full_prompt):
            if event.kind == "text":
                output_parts.append(event.text)
        return "".join(output_parts) if output_parts else None


class SkillRegistry:
    """Registry of available skills."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def discover_from_directory(self, directory: Path) -> int:
        """Auto-discover skills from SKILL.md files.

        Looks for files matching:
          - SKILL.md (single skill)
          - *.skill.md (named skills)
          - .nemocode/skills/*.md

        SKILL.md format:
        ```markdown
        ---
        name: skill-name
        description: What the skill does
        usage: /skill-name [args]
        ---

        Prompt content for the agent...
        ```

        Returns the number of skills discovered.
        """
        count = 0

        # Pattern 1: SKILL.md in project root
        skill_md = directory / "SKILL.md"
        if skill_md.exists():
            count += self._load_skill_file(skill_md)

        # Pattern 2: *.skill.md files
        for f in directory.glob("*.skill.md"):
            count += self._load_skill_file(f)

        # Pattern 3: .nemocode/skills/*.md
        skills_dir = directory / ".nemocode" / "skills"
        if skills_dir.is_dir():
            for f in skills_dir.glob("*.md"):
                count += self._load_skill_file(f)

        if count > 0:
            logger.info("Discovered %d skill(s) from %s", count, directory)

        return count

    def _load_skill_file(self, path: Path) -> int:
        """Load a skill from a markdown file. Returns 1 on success, 0 on failure."""
        try:
            content = path.read_text()
        except Exception as e:
            logger.debug("Failed to read skill file %s: %s", path, e)
            return 0

        # Parse frontmatter
        frontmatter, body = _parse_frontmatter(content)

        name = frontmatter.get("name", "")
        if not name:
            # Derive name from filename
            stem = path.stem
            if stem.endswith(".skill"):
                name = stem[: -len(".skill")]
            elif stem == "SKILL":
                name = "custom"
            else:
                name = stem

        # Don't override built-in skills
        if name in self._skills:
            logger.debug("Skipping discovered skill '%s' — already registered", name)
            return 0

        description = frontmatter.get("description", f"Custom skill from {path.name}")
        usage = frontmatter.get("usage", f"/{name}")

        skill = MarkdownSkill(
            name=name,
            description=description,
            prompt=body.strip(),
            usage=usage,
        )
        self.register(skill)
        return 1


def _parse_frontmatter(content: str) -> tuple[dict[str, str], str]:
    """Parse YAML-like frontmatter from markdown content."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not match:
        return {}, content

    frontmatter_text = match.group(1)
    body = match.group(2)

    # Simple YAML parsing (key: value)
    fm: dict[str, str] = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip().strip('"').strip("'")

    return fm, body


def create_default_registry(project_dir: Path | None = None) -> SkillRegistry:
    """Create a registry with built-in skills + auto-discovered skills."""
    from nemocode.skills.commit import CommitSkill
    from nemocode.skills.review import ReviewSkill

    registry = SkillRegistry()
    registry.register(CommitSkill())
    registry.register(ReviewSkill())

    # Auto-discover from project directory
    if project_dir is None:
        project_dir = Path.cwd()
    registry.discover_from_directory(project_dir)

    return registry
