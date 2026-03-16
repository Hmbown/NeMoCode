# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for MCP tool discovery, JSON-RPC handshake, and tool registration."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemocode.config.schema import MCPServerConfig
from nemocode.tools import ToolRegistry
from nemocode.tools.mcp import MCPClient, register_mcp_tools

_EMPTY_SCHEMA = {"type": "object", "properties": {}}


class TestMakeRequest:
    def test_generates_correct_jsonrpc_format(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        req = client._make_request("tools/list", {"key": "val"})
        assert req["jsonrpc"] == "2.0"
        assert req["method"] == "tools/list"
        assert req["params"] == {"key": "val"}
        assert "id" in req

    def test_request_ids_increment(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        r1 = client._make_request("a", {})
        r2 = client._make_request("b", {})
        r3 = client._make_request("c", {})
        assert r2["id"] == r1["id"] + 1
        assert r3["id"] == r2["id"] + 1

    def test_first_request_id_is_one(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        req = client._make_request("test", {})
        assert req["id"] == 1

    def test_method_preserved_exactly(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        req = client._make_request("tools/call", {"name": "do_thing"})
        assert req["method"] == "tools/call"


class TestCallToolWithoutConnection:
    @pytest.mark.asyncio
    async def test_returns_error_json(self):
        config = MCPServerConfig(name="myserver", command="echo", args=[])
        client = MCPClient(config)
        result = await client.call_tool("test_tool", {"arg": "value"})
        data = json.loads(result)
        assert "error" in data

    @pytest.mark.asyncio
    async def test_error_mentions_server_name(self):
        config = MCPServerConfig(name="myserver", command="echo", args=[])
        client = MCPClient(config)
        result = await client.call_tool("test_tool", {})
        data = json.loads(result)
        assert "myserver" in data["error"]


class TestDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_on_dead_process_is_noop(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        # No process set — should not raise
        await client.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_on_already_terminated_process_is_noop(self):
        config = MCPServerConfig(name="test", command="echo", args=[])
        client = MCPClient(config)
        # Create a mock process that has already exited (returncode is not None)
        mock_proc = MagicMock()
        mock_proc.returncode = 0  # already exited
        client._process = mock_proc
        await client.disconnect()
        # terminate() should NOT be called because returncode is not None
        mock_proc.terminate.assert_not_called()


class TestRegisterMCPTools:
    @pytest.mark.asyncio
    async def test_creates_correct_tool_names(self):
        config = MCPServerConfig(name="fs_server", command="echo", args=[])
        registry = ToolRegistry()

        mock_tools = [
            {"name": "read", "description": "Read a file", "inputSchema": _EMPTY_SCHEMA},
            {"name": "write", "description": "Write a file", "inputSchema": _EMPTY_SCHEMA},
        ]

        with patch.object(MCPClient, "connect", new_callable=AsyncMock, return_value=mock_tools):
            await register_mcp_tools([config], registry)

        names = [td.name for td in registry.list_tools()]
        assert "mcp_fs_server_read" in names
        assert "mcp_fs_server_write" in names

    @pytest.mark.asyncio
    async def test_creates_correct_descriptions_with_prefix(self):
        config = MCPServerConfig(name="mytools", command="echo", args=[])
        registry = ToolRegistry()

        mock_tools = [
            {"name": "search", "description": "Search stuff", "inputSchema": _EMPTY_SCHEMA},
        ]

        with patch.object(MCPClient, "connect", new_callable=AsyncMock, return_value=mock_tools):
            await register_mcp_tools([config], registry)

        td = registry.get("mcp_mytools_search")
        assert td is not None
        assert td.description.startswith("[MCP:mytools]")
        assert "Search stuff" in td.description

    @pytest.mark.asyncio
    async def test_tool_category_is_mcp(self):
        config = MCPServerConfig(name="srv", command="echo", args=[])
        registry = ToolRegistry()

        mock_tools = [
            {"name": "do_thing", "description": "Does a thing", "inputSchema": _EMPTY_SCHEMA},
        ]

        with patch.object(MCPClient, "connect", new_callable=AsyncMock, return_value=mock_tools):
            await register_mcp_tools([config], registry)

        td = registry.get("mcp_srv_do_thing")
        assert td is not None
        assert td.category == "mcp"

    @pytest.mark.asyncio
    async def test_tool_function_closure_captures_correct_client_and_name(self):
        config = MCPServerConfig(name="srv", command="echo", args=[])
        registry = ToolRegistry()

        mock_tools = [
            {"name": "alpha", "description": "A", "inputSchema": _EMPTY_SCHEMA},
            {"name": "beta", "description": "B", "inputSchema": _EMPTY_SCHEMA},
        ]

        mock_connect = patch.object(
            MCPClient, "connect", new_callable=AsyncMock, return_value=mock_tools
        )
        mock_call_tool = patch.object(
            MCPClient, "call_tool", new_callable=AsyncMock, return_value='{"ok": true}'
        )
        with mock_connect, mock_call_tool as mock_call:
            await register_mcp_tools([config], registry)

            # Call the first registered tool
            td_alpha = registry.get("mcp_srv_alpha")
            assert td_alpha is not None
            await td_alpha.fn(foo="bar")
            mock_call.assert_called_with("alpha", {"foo": "bar"})

            # Call the second registered tool
            td_beta = registry.get("mcp_srv_beta")
            assert td_beta is not None
            await td_beta.fn(baz="qux")
            mock_call.assert_called_with("beta", {"baz": "qux"})

    @pytest.mark.asyncio
    async def test_returns_clients_list(self):
        config1 = MCPServerConfig(name="a", command="echo", args=[])
        config2 = MCPServerConfig(name="b", command="echo", args=[])
        registry = ToolRegistry()

        with patch.object(MCPClient, "connect", new_callable=AsyncMock, return_value=[]):
            clients = await register_mcp_tools([config1, config2], registry)

        assert len(clients) == 2

    @pytest.mark.asyncio
    async def test_empty_servers_list(self):
        registry = ToolRegistry()
        clients = await register_mcp_tools([], registry)
        assert clients == []
        assert registry.list_tools() == []

    @pytest.mark.asyncio
    async def test_default_schema_when_no_input_schema(self):
        config = MCPServerConfig(name="srv", command="echo", args=[])
        registry = ToolRegistry()

        mock_tools = [
            {"name": "simple", "description": "No schema provided"},
        ]

        with patch.object(MCPClient, "connect", new_callable=AsyncMock, return_value=mock_tools):
            await register_mcp_tools([config], registry)

        td = registry.get("mcp_srv_simple")
        assert td is not None
        assert td.parameters == {"type": "object", "properties": {}}
