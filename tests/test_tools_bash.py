# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json

import pytest

from nemocode.tools.bash import bash_exec


class TestBashExec:
    @pytest.mark.asyncio
    async def test_bash_exec_timeout_kills_process(self):
        result_str = await bash_exec("sleep 60", timeout=1)
        result = json.loads(result_str)

        assert "error" in result
        assert "timed out" in result["error"].lower()
