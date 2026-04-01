# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Configurable sandbox for tool execution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SandboxLevel(str, Enum):
    """Security levels for the execution sandbox."""

    STRICT = "strict"
    STANDARD = "standard"
    PERMISSIVE = "permissive"


@dataclass
class SandboxConfig:
    """Configuration for a sandbox instance.

    Attributes:
        level: The security level to enforce.
        project_root: Root directory that paths must stay within.
        max_cwd_depth: Maximum directory depth for cwd in bash execution.
        allowed_commands: Commands allowed in STRICT mode (override).
        blocked_commands: Commands blocked even in STANDARD mode.
    """

    level: SandboxLevel = SandboxLevel.STANDARD
    project_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    max_cwd_depth: int = 10
    allowed_commands: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)


class Sandbox:
    """Enforces sandbox policies for bash and filesystem operations.

    The sandbox controls what commands can be executed and which paths
    can be read or written based on the configured ``SandboxLevel``.

    Levels:
        STRICT: No bash execution, read-only filesystem, paths locked to project root.
        STANDARD: Bash with confirmation, read/write within project root (default).
        PERMISSIVE: No restrictions.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        """Initialize the sandbox.

        Args:
            config: Optional sandbox configuration. Defaults to STANDARD level
                with the current working directory as project root.
        """
        self.config = config or SandboxConfig()
        self._project_root = self.config.project_root.resolve()

    @classmethod
    def from_env(cls) -> Sandbox:
        """Create a sandbox configured from the environment.

        Reads ``NEMOCODE_SANDBOX_LEVEL`` (strict|standard|permissive).
        Defaults to STANDARD if unset or invalid.

        Returns:
            A Sandbox instance configured from the environment.
        """
        raw = os.environ.get("NEMOCODE_SANDBOX_LEVEL", "standard").strip().lower()
        try:
            level = SandboxLevel(raw)
        except ValueError:
            level = SandboxLevel.STANDARD
        return cls(SandboxConfig(level=level))

    # ------------------------------------------------------------------
    # Path validation
    # ------------------------------------------------------------------

    def validate_path(self, path: str) -> Path:
        """Resolve and validate a path within the allowed bounds.

        Args:
            path: The path string to validate.

        Returns:
            The resolved Path object.

        Raises:
            PermissionError: If the resolved path is outside the project root
                (except in PERMISSIVE mode).
        """
        resolved = Path(path).resolve()

        if self.config.level == SandboxLevel.PERMISSIVE:
            return resolved

        try:
            resolved.relative_to(self._project_root)
        except ValueError:
            raise PermissionError(f"Path outside project directory: {resolved}")
        return resolved

    # ------------------------------------------------------------------
    # Bash execution checks
    # ------------------------------------------------------------------

    def can_execute_command(self, cmd: str) -> bool:
        """Check whether a command is allowed to execute.

        Args:
            cmd: The shell command string.

        Returns:
            True if execution is allowed, False otherwise.
        """
        if self.config.level == SandboxLevel.PERMISSIVE:
            return True

        if self.config.level == SandboxLevel.STRICT:
            if self.config.allowed_commands:
                return any(cmd.startswith(allowed) for allowed in self.config.allowed_commands)
            return False

        # STANDARD: allow unless explicitly blocked
        if self.config.blocked_commands:
            return not any(cmd.startswith(blocked) for blocked in self.config.blocked_commands)
        return True

    # ------------------------------------------------------------------
    # Filesystem checks
    # ------------------------------------------------------------------

    def can_write(self) -> bool:
        """Check whether writing is allowed at the current sandbox level.

        This checks only the sandbox level, not path bounds.
        Path validation is handled separately by ``validate_path``.

        Returns:
            True if writing is allowed at this level, False otherwise.
        """
        if self.config.level == SandboxLevel.PERMISSIVE:
            return True

        if self.config.level == SandboxLevel.STRICT:
            return False

        # STANDARD: writes allowed (path bounds checked by validate_path)
        return True

    def can_read(self, path: str) -> bool:
        """Check whether reading from a path is allowed.

        Args:
            path: The source path.

        Returns:
            True if reading is allowed, False otherwise.
        """
        if self.config.level == SandboxLevel.PERMISSIVE:
            return True

        # STRICT and STANDARD: readable within project root
        try:
            self.validate_path(path)
            return True
        except PermissionError:
            return False

    # ------------------------------------------------------------------
    # Bash cwd validation
    # ------------------------------------------------------------------

    def validate_cwd(self, cwd: str) -> Path:
        """Validate a working directory for bash execution.

        Args:
            cwd: The working directory path.

        Returns:
            The resolved Path object.

        Raises:
            PermissionError: If the cwd is outside the project root or too deep.
        """
        if self.config.level == SandboxLevel.PERMISSIVE:
            return Path(cwd).resolve() if cwd else Path.cwd()

        if not cwd:
            return Path.cwd()

        work_dir = Path(cwd).resolve()
        try:
            work_dir.relative_to(self._project_root)
        except ValueError:
            raise PermissionError(f"cwd must be within project directory: {work_dir}")

        rel_parts = len(work_dir.relative_to(self._project_root).parts)
        if rel_parts > self.config.max_cwd_depth:
            raise PermissionError(
                f"cwd too deep (max {self.config.max_cwd_depth} levels): {work_dir}"
            )

        return work_dir
