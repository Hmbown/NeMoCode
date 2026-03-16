# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""MCP (Model Context Protocol) client — connect to MCP servers as tool sources.

Discovers tools from MCP servers configured in .nemocode.yaml and registers
them dynamically in the ToolRegistry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from nemocode.config.schema import MCPServerConfig
from nemocode.tools import ToolDef, ToolRegistry

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for connecting to MCP tool servers."""

    def __init__(self, server_config: MCPServerConfig) -> None:
        self.config = server_config
        self._process: asyncio.subprocess.Process | None = None
        self._tools: list[dict[str, Any]] = []
        self._request_id = 0

    async def connect(self) -> list[dict[str, Any]]:
        """Start the MCP server process and discover tools."""
        env = {**os.environ, **self.config.env}

        try:
            self._process = await asyncio.create_subprocess_exec(
                self.config.command,
                *self.config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Initialize the MCP connection
            init_msg = self._make_request(
                "initialize",
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "nemocode", "version": "0.2.0"},
                },
            )
            await self._send_request(init_msg)

            # Send initialized notification
            await self._send_notification("notifications/initialized", {})

            # Discover tools
            tools_msg = self._make_request("tools/list", {})
            tools_response = await self._send_request(tools_msg)

            if tools_response and "result" in tools_response:
                self._tools = tools_response["result"].get("tools", [])

            return self._tools
        except Exception as e:
            logger.error("MCP connect failed for %s: %s", self.config.name, e)
            return []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Automatically attempts one reconnect if the server process has died.
        """
        # Detect dead server and attempt reconnect
        if not self._process or self._process.returncode is not None:
            logger.warning("MCP server %s died, attempting reconnect...", self.config.name)
            try:
                await self.connect()
            except Exception as e:
                logger.error("MCP reconnect failed for %s: %s", self.config.name, e)
                return json.dumps(
                    {"error": f"MCP server {self.config.name} crashed and reconnect failed: {e}"}
                )
            if not self._process or self._process.returncode is not None:
                return json.dumps(
                    {"error": f"MCP server {self.config.name} is not running"}
                )

        msg = self._make_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        response = await self._send_request(msg)

        if response and "result" in response:
            content = response["result"].get("content", [])
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            return "\n".join(texts) if texts else json.dumps(response["result"])
        elif response and "error" in response:
            return json.dumps({"error": response["error"].get("message", "Unknown error")})
        return json.dumps({"error": "No response from MCP server"})

    async def disconnect(self) -> None:
        """Stop the MCP server process."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()

    def _make_request(self, method: str, params: dict) -> dict:
        self._request_id += 1
        return {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

    async def _send_notification(self, method: str, params: dict) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params}
        if self._process and self._process.stdin:
            data = json.dumps(msg) + "\n"
            self._process.stdin.write(data.encode())
            await self._process.stdin.drain()

    async def _send_request(self, msg: dict) -> dict | None:
        """Send a JSON-RPC request and wait for the matching response.

        Loops reading lines until we find a response whose 'id' matches
        the request's 'id', skipping server-initiated notifications.
        """
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        request_id = msg.get("id")
        data = json.dumps(msg) + "\n"
        self._process.stdin.write(data.encode())
        await self._process.stdin.drain()

        # Read lines until we get a response matching our request ID.
        # MCP servers may emit notifications between our request and the
        # response, so we must skip lines that are not our response.
        try:
            deadline = asyncio.get_event_loop().time() + 30
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    logger.debug("MCP response timeout for request %s", request_id)
                    return None
                line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=remaining,
                )
                if not line:
                    return None
                try:
                    response = json.loads(line.decode())
                except json.JSONDecodeError:
                    continue
                # Match on request ID; skip notifications (which have no 'id')
                if response.get("id") == request_id:
                    return response
                # If it is a notification or a response for a different
                # request, skip it and keep reading.
        except asyncio.TimeoutError:
            logger.debug("MCP response timeout for request %s", request_id)
        except Exception as e:
            logger.debug("MCP response error: %s", e)

        return None


async def register_mcp_tools(
    servers: list[MCPServerConfig],
    registry: ToolRegistry,
) -> list[MCPClient]:
    """Connect to MCP servers and register their tools in the registry."""
    clients = []

    for server_config in servers:
        client = MCPClient(server_config)
        tools = await client.connect()

        for tool_def in tools:
            name = f"mcp_{server_config.name}_{tool_def['name']}"
            description = tool_def.get("description", "")
            schema = tool_def.get("inputSchema", {"type": "object", "properties": {}})

            # Create a closure for the tool function
            _client = client
            _tool_name = tool_def["name"]

            async def mcp_tool_fn(_c=_client, _n=_tool_name, **kwargs) -> str:
                return await _c.call_tool(_n, kwargs)

            td = ToolDef(
                name=name,
                description=f"[MCP:{server_config.name}] {description}",
                parameters=schema,
                fn=mcp_tool_fn,
                category="mcp",
            )
            registry.register(td)

        clients.append(client)
        if tools:
            logger.info("MCP %s: registered %d tools", server_config.name, len(tools))

    return clients
