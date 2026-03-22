# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
import uuid

import pytest

from nemocode.core.persistence import delete_session, load_session, save_session
from nemocode.core.sessions import Session
from nemocode.core.streaming import Message, Role, ToolCall


@pytest.mark.asyncio
async def test_session_persistence_roundtrip():
    session = Session(id=f"test-persist-{uuid.uuid4().hex[:8]}", endpoint_name="test-ep")
    session.add_system("You are a coding assistant.")
    session.add_user("Write a hello world function.")
    session.add_assistant(
        Message(
            role=Role.ASSISTANT,
            content="Here is a hello world function.",
            tool_calls=[
                ToolCall(
                    id="tc1",
                    name="write_file",
                    arguments={"path": "/tmp/hello.py", "content": "print('hello')"},
                )
            ],
        )
    )
    session.add_tool_result("tc1", json.dumps({"status": "ok"}))

    try:
        save_session(session)

        loaded = load_session(session.id)
        assert loaded is not None
        assert len(loaded.messages) == len(session.messages)

        assert loaded.messages[0].role == Role.SYSTEM
        assert loaded.messages[0].content == "You are a coding assistant."

        assert loaded.messages[1].role == Role.USER
        assert loaded.messages[1].content == "Write a hello world function."

        assert loaded.messages[2].role == Role.ASSISTANT
        assert loaded.messages[2].content == "Here is a hello world function."
        assert len(loaded.messages[2].tool_calls) == 1
        assert loaded.messages[2].tool_calls[0].name == "write_file"

        assert loaded.messages[3].role == Role.TOOL
        assert loaded.messages[3].tool_call_id == "tc1"

    finally:
        delete_session(session.id)
