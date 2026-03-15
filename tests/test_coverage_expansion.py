# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Extended test coverage for critical gaps identified by code reviewers.

Covers: NIM chat body construction, SSE tool call assembly, agent loop
with tools, formation pipeline, session persistence roundtrip, metrics
accuracy, edit_file edge cases, and bash timeout cleanup.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from nemocode.config.schema import (
    Endpoint,
    EndpointTier,
    Formation,
    FormationRole,
    FormationSlot,
    Manifest,
    MoEConfig,
    NeMoCodeConfig,
    ReasoningConfig,
)
from nemocode.core.metrics import MetricsCollector, RequestMetrics
from nemocode.core.persistence import load_session, save_session
from nemocode.core.registry import Registry
from nemocode.core.scheduler import Scheduler
from nemocode.core.sessions import Session
from nemocode.core.streaming import (
    CompletionResult,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from nemocode.providers.nim_chat import NIMChatProvider
from nemocode.tools.loader import load_tools

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def nim_endpoint() -> Endpoint:
    return Endpoint(
        name="test-ep",
        tier=EndpointTier.DEV_HOSTED,
        base_url="https://integrate.api.nvidia.com/v1",
        api_key_env="NVIDIA_API_KEY",
        model_id="nvidia/nemotron-3-super-120b-a12b",
    )


@pytest.fixture
def nim_manifest() -> Manifest:
    return Manifest(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        display_name="Nemotron 3 Super",
        context_window=1_048_576,
        reasoning=ReasoningConfig(
            supports_thinking=True,
            thinking_param="enable_thinking",
        ),
        moe=MoEConfig(
            total_params_b=120,
            active_params_b=12,
        ),
    )


@pytest.fixture
def provider(nim_endpoint: Endpoint, nim_manifest: Manifest) -> NIMChatProvider:
    return NIMChatProvider(
        endpoint=nim_endpoint,
        manifest=nim_manifest,
        api_key="test-key",
    )


@pytest.fixture
def formation_config(nim_endpoint: Endpoint) -> NeMoCodeConfig:
    return NeMoCodeConfig(
        default_endpoint="test-ep",
        endpoints={"test-ep": nim_endpoint},
        formations={
            "planning": Formation(
                name="Planning",
                slots=[
                    FormationSlot(endpoint="test-ep", role=FormationRole.PLANNER),
                    FormationSlot(endpoint="test-ep", role=FormationRole.EXECUTOR),
                    FormationSlot(endpoint="test-ep", role=FormationRole.REVIEWER),
                ],
                verification_rounds=1,
            ),
        },
    )


# ===================================================================
# Test: NIM chat request body construction
# ===================================================================


class TestNIMChatBuildBody:
    def test_nim_chat_build_body(self, provider: NIMChatProvider):
        """Verify request body construction respects manifest reasoning config."""
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful."),
            Message(role=Role.USER, content="Hello"),
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        body = provider._build_body(messages, tools=tools, stream=True)

        # Core fields
        assert body["model"] == "nvidia/nemotron-3-super-120b-a12b"
        assert body["stream"] is True
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"

        # Manifest-driven: reasoning config should inject enable_thinking
        # inside chat_template_kwargs (NVIDIA NIM API format)
        assert body["chat_template_kwargs"]["enable_thinking"] is True

        # Tools included
        assert body["tools"] == tools

    def test_nim_chat_build_body_no_manifest(self, nim_endpoint: Endpoint):
        """Body without manifest should not crash or inject reasoning params."""
        provider = NIMChatProvider(endpoint=nim_endpoint, manifest=None, api_key="key")
        messages = [Message(role=Role.USER, content="Hi")]
        body = provider._build_body(messages, stream=False)
        assert body["model"] == nim_endpoint.model_id
        assert "chat_template_kwargs" not in body


# ===================================================================
# Test: SSE stream tool call assembly
# ===================================================================


class TestNIMChatStreamToolAssembly:
    @pytest.mark.asyncio
    async def test_nim_chat_stream_tool_assembly(self, provider: NIMChatProvider):
        """Mock SSE response with streamed tool calls and verify assembly."""

        # Simulate SSE lines for a streamed tool call response.
        # Build the JSON payloads programmatically to avoid long lines.
        chunk1 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "tc_abc",
                                    "function": {"name": "read_file", "arguments": '{"pa'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        chunk2 = json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": 'th": "test.py"}'}}
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            }
        )
        chunk3 = json.dumps(
            {
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 10},
            }
        )
        sse_lines = [
            f"data: {chunk1}",
            f"data: {chunk2}",
            f"data: {chunk3}",
            "data: [DONE]",
        ]

        async def mock_aiter_lines():
            for line in sse_lines:
                yield line

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = mock_aiter_lines

        chunks = []
        async for chunk in provider._process_stream(mock_response):
            chunks.append(chunk)

        # We should get a final chunk with the assembled tool call
        tool_chunks = [c for c in chunks if c.tool_calls]
        assert len(tool_chunks) == 1

        tc = tool_chunks[0].tool_calls[0]
        assert tc.id == "tc_abc"
        assert tc.name == "read_file"
        assert tc.arguments == {"path": "test.py"}


# ===================================================================
# Test: Agent loop with tool execution
# ===================================================================


class TestAgentLoopWithTools:
    @pytest.mark.asyncio
    async def test_agent_loop_with_tools(self, formation_config: NeMoCodeConfig):
        """Mock provider returning a tool call, verify tool execution."""

        # First response: tool call. Second response: text.
        call_count = 0

        async def mock_stream(messages, tools=None, extra_body=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call: model wants to use read_file
                yield StreamChunk(text="Let me read the file.")
                yield StreamChunk(
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="read_file",
                            arguments={"path": "/dev/null"},
                        )
                    ],
                    finish_reason="tool_calls",
                    usage={"prompt_tokens": 50, "completion_tokens": 20},
                )
            else:
                # Second call: model produces final answer
                yield StreamChunk(text="The file is empty.")
                yield StreamChunk(
                    finish_reason="stop",
                    usage={"prompt_tokens": 80, "completion_tokens": 15},
                )

        mock_provider = AsyncMock()
        mock_provider.stream = mock_stream

        registry = Registry(formation_config)
        tool_reg = load_tools(["fs"])

        with patch.object(registry, "get_chat_provider", return_value=mock_provider):
            scheduler = Scheduler(registry, tool_reg)
            events = []
            async for ev in scheduler.run_single("test-ep", "Read /dev/null"):
                events.append(ev)

        # Verify we got tool_call and tool_result events
        tool_call_events = [e for e in events if e.kind == "tool_call"]
        tool_result_events = [e for e in events if e.kind == "tool_result"]
        text_events = [e for e in events if e.kind == "text"]

        assert len(tool_call_events) >= 1
        assert tool_call_events[0].tool_name == "read_file"
        assert len(tool_result_events) >= 1
        assert len(text_events) >= 2  # "Let me read" + "The file is empty"


# ===================================================================
# Test: Formation pipeline (plan -> execute -> review)
# ===================================================================


class TestRunFormation:
    @pytest.mark.asyncio
    async def test_run_formation(self, formation_config: NeMoCodeConfig):
        """Mock multi-model pipeline: plan -> execute -> review."""
        mock_provider = AsyncMock()

        # complete() is used for planning (text-only mode)
        mock_provider.complete = AsyncMock(
            return_value=CompletionResult(
                content="1. Read the file\n2. Add error handling\n3. Run tests",
                finish_reason="stop",
            )
        )

        call_count = 0

        async def mock_stream(messages, tools=None, extra_body=None):
            nonlocal call_count
            call_count += 1
            # Both executor and reviewer just produce text
            if call_count == 1:
                yield StreamChunk(text="I implemented the changes.")
            else:
                yield StreamChunk(text="APPROVE. Changes look correct.")
            yield StreamChunk(
                finish_reason="stop",
                usage={"prompt_tokens": 100, "completion_tokens": 30},
            )

        mock_provider.stream = mock_stream

        registry = Registry(formation_config)
        tool_reg = load_tools(["fs"])

        with patch.object(registry, "get_chat_provider", return_value=mock_provider):
            scheduler = Scheduler(registry, tool_reg)
            events = []
            async for ev in scheduler.run_formation("planning", "Add error handling"):
                events.append(ev)

        # Verify we went through all three phases
        phase_events = [e for e in events if e.kind == "phase"]
        phases = [e.phase for e in phase_events]
        assert "planning" in phases
        assert "executing" in phases
        assert "reviewing" in phases

        # Verify the plan was produced
        text_events = [e for e in events if e.kind == "text"]
        all_text = "".join(e.text for e in text_events)
        assert "Read the file" in all_text  # from planner
        assert "APPROVE" in all_text  # from reviewer


# ===================================================================
# Test: Session persistence roundtrip
# ===================================================================


class TestSessionPersistenceRoundtrip:
    def test_session_persistence_roundtrip(self, tmp_path: Path):
        """Save a session, load it back, verify all messages match."""
        # Override the session directory for this test; disable SQLite to test JSON path
        with (
            patch("nemocode.core.persistence._SESSION_DIR", tmp_path),
            patch("nemocode.core.persistence._USE_SQLITE", False),
        ):
            # Create a session with diverse message types
            session = Session(id="roundtrip-test", endpoint_name="nim-super")
            session.add_system("You are a helpful assistant.")
            session.add_user("Hello, how are you?")
            session.add_assistant(Message(role=Role.ASSISTANT, content="I am doing well!"))
            session.add_user("Read test.py")
            session.add_assistant(
                Message(
                    role=Role.ASSISTANT,
                    content="Let me read that.",
                    tool_calls=[
                        ToolCall(
                            id="tc_1",
                            name="read_file",
                            arguments={"path": "test.py"},
                        )
                    ],
                )
            )
            session.add_tool_result("tc_1", "print('hello')")
            session.add_assistant(
                Message(role=Role.ASSISTANT, content="The file contains a print statement.")
            )
            session.usage.add(200, 100)

            # Save with metadata
            metadata = {"cwd": "/tmp/test", "git_branch": "main"}
            save_session(session, metadata=metadata)

            # Load and verify
            loaded = load_session("roundtrip-test")
            assert loaded is not None
            assert loaded.id == "roundtrip-test"
            assert loaded.endpoint_name == "nim-super"
            assert loaded.message_count() == session.message_count()
            assert loaded.usage.total_tokens == 300

            # Verify message content integrity
            for orig, loaded_msg in zip(session.messages, loaded.messages):
                assert orig.role == loaded_msg.role
                assert orig.content == loaded_msg.content
                assert len(orig.tool_calls) == len(loaded_msg.tool_calls)
                if orig.tool_calls:
                    assert orig.tool_calls[0].name == loaded_msg.tool_calls[0].name
                    assert orig.tool_calls[0].arguments == loaded_msg.tool_calls[0].arguments

            # Verify the file on disk contains metadata
            saved_data = json.loads((tmp_path / "roundtrip-test.json").read_text())
            assert saved_data["metadata"]["git_branch"] == "main"
            assert saved_data["metadata"]["cwd"] == "/tmp/test"


# ===================================================================
# Test: Metrics collector accuracy
# ===================================================================


class TestMetricsCollector:
    def test_metrics_collector(self):
        """Token and cost tracking accuracy across multiple requests."""
        collector = MetricsCollector()

        # Record two requests with known token counts
        m1 = RequestMetrics(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            endpoint_name="nim-super",
            prompt_tokens=1000,
            completion_tokens=500,
            total_time_ms=200.0,
            tool_calls=2,
            tool_errors=0,
        )
        m2 = RequestMetrics(
            model_id="nvidia/nemotron-3-super-120b-a12b",
            endpoint_name="nim-super",
            prompt_tokens=2000,
            completion_tokens=1000,
            total_time_ms=400.0,
            tool_calls=1,
            tool_errors=1,
        )

        collector.record(m1)
        collector.record(m2)

        assert collector.total_prompt_tokens == 3000
        assert collector.total_completion_tokens == 1500
        assert collector.total_tokens == 4500
        assert collector.request_count == 2
        assert collector.avg_latency_ms == 300.0

        # Verify cost calculation (Super: $0.35/M input, $0.40/M output)
        expected_cost = (3000 / 1_000_000) * 0.35 + (1500 / 1_000_000) * 0.40
        assert abs(collector.total_cost - expected_cost) < 1e-9

        # Summary should have all keys
        summary = collector.summary()
        assert summary["requests"] == 2
        assert summary["total_tokens"] == 4500
        assert summary["prompt_tokens"] == 3000
        assert summary["completion_tokens"] == 1500

        # Reset should clear everything
        collector.reset()
        assert collector.request_count == 0
        assert collector.total_tokens == 0


# ===================================================================
# Test: edit_file with multiple matches
# ===================================================================


class TestEditFileMultipleMatches:
    @pytest.mark.asyncio
    async def test_edit_file_multiple_matches(self, tmp_path: Path):
        """Verify 'found N times' error when old_string appears multiple times."""
        from nemocode.tools.fs import edit_file, set_project_root

        set_project_root(tmp_path)
        test_file = tmp_path / "multi.py"
        test_file.write_text("foo = 1\nbar = foo\nbaz = foo\n")

        result = await edit_file(str(test_file), "foo", "qux")
        data = json.loads(result)
        assert "error" in data
        assert "3 times" in data["error"]
        # File should be unchanged
        assert test_file.read_text() == "foo = 1\nbar = foo\nbaz = foo\n"


# ===================================================================
# Test: edit_file with empty old_string
# ===================================================================


class TestEditFileEmptyOldString:
    @pytest.mark.asyncio
    async def test_edit_file_empty_old_string(self, tmp_path: Path):
        """Verify clear error message when old_string is empty."""
        from nemocode.tools.fs import edit_file, set_project_root

        set_project_root(tmp_path)
        test_file = tmp_path / "empty_match.py"
        test_file.write_text("some content here\n")

        result = await edit_file(str(test_file), "", "new stuff")
        data = json.loads(result)
        assert "error" in data
        assert "must not be empty" in data["error"]
        # File should be unchanged
        assert test_file.read_text() == "some content here\n"


# ===================================================================
# Test: bash_exec timeout kills process
# ===================================================================


class TestBashExecTimeoutKillsProcess:
    @pytest.mark.asyncio
    async def test_bash_exec_timeout_kills_process(self):
        """Verify subprocess is killed on timeout and error is returned."""
        from nemocode.tools.bash import bash_exec

        # Use a 1-second timeout with a command that would run forever
        result = await bash_exec("sleep 60", timeout=1)
        data = json.loads(result)
        assert "error" in data
        assert "timed out" in data["error"].lower()
