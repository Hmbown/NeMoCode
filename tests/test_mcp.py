# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Test MCP client."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from nemocode.config.schema import MCPServerConfig
from nemocode.tools.mcp import MCPClient


class TestMCPClient:
    def test_make_request(self):
        config = MCPServerConfig(name="test", command="echo", args=["hello"])
        client = MCPClient(config)
        req = client._make_request("tools/list", {})
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "tools/list"
        assert req["id"] == 1

    def test_request_id_increments(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        r1 = client._make_request("test", {})
        r2 = client._make_request("test", {})
        assert r2["id"] == r1["id"] + 1

    @pytest.mark.asyncio
    async def test_call_tool_without_connection(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        # Mock connect to simulate reconnect failure (no real server)
        with patch.object(client, "connect", new_callable=AsyncMock, return_value=[]):
            result = await client.call_tool("test_tool", {"arg": "value"})
        data = json.loads(result)
        assert "error" in data
