# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for ContextManager.smart_compact() invariants."""

from __future__ import annotations

from nemocode.core.context import ContextManager
from nemocode.core.streaming import Message, Role, ToolCall


class TestPreservesSystemMessage:
    def test_system_message_is_first_after_compaction(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="You are a helpful assistant.")]
        for i in range(40):
            messages.append(Message(role=Role.USER, content=f"User msg {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst msg {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)
        assert result[0].role == Role.SYSTEM

    def test_system_message_content_preserved(self):
        mgr = ContextManager()
        system_content = "You are a coding agent. Follow instructions precisely."
        messages = [Message(role=Role.SYSTEM, content=system_content)]
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"Msg {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Reply {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)
        assert system_content in result[0].content


class TestKeepsLastNMessages:
    def test_last_n_messages_preserved(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Assistant {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)
        # Last message should be the most recent assistant message
        assert result[-1].content == "Assistant 29"
        # There should be at least 10 non-system messages in the result
        non_system = [m for m in result if m.role != Role.SYSTEM]
        assert len(non_system) >= 10

    def test_most_recent_user_message_kept(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        result = mgr.smart_compact(messages, keep_recent=6)
        contents = [m.content for m in result]
        assert "User 29" in contents
        assert "Asst 29" in contents


class TestToolCallPairIntegrity:
    def test_does_not_split_tool_call_tool_result_pairs(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]

        # Add several user/assistant exchanges followed by a tool call pair
        for i in range(20):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            if i % 5 == 3:
                # Insert a tool call + tool result pair
                tc = ToolCall(id=f"tc_{i}", name="read_file", arguments={"path": f"file{i}.py"})
                messages.append(
                    Message(role=Role.ASSISTANT, content="Let me read", tool_calls=[tc])
                )
                messages.append(
                    Message(role=Role.TOOL, content=f"content of file{i}", tool_call_id=f"tc_{i}")
                )
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)

        # Verify no TOOL message appears without its preceding ASSISTANT+tool_calls
        for idx, msg in enumerate(result):
            if msg.role == Role.TOOL:
                # The message before a TOOL should be ASSISTANT with tool_calls
                assert idx > 0, "TOOL message at position 0"
                prev = result[idx - 1]
                assert prev.role == Role.ASSISTANT and prev.tool_calls, (
                    f"TOOL at index {idx} not preceded by ASSISTANT with tool_calls"
                )

    def test_assistant_with_tool_calls_kept_with_results(self):
        """If an assistant message with tool_calls is in the kept tail,
        its corresponding TOOL result should also be present."""
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]
        for i in range(15):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        # Add a tool call pair near the end
        tc = ToolCall(id="tc_final", name="bash_exec", arguments={"command": "ls"})
        messages.append(Message(role=Role.USER, content="Run ls"))
        messages.append(Message(role=Role.ASSISTANT, content="Running", tool_calls=[tc]))
        messages.append(
            Message(role=Role.TOOL, content='{"exit_code": 0}', tool_call_id="tc_final")
        )
        messages.append(Message(role=Role.ASSISTANT, content="Done."))

        result = mgr.smart_compact(messages, keep_recent=6)

        # The tool pair should be intact in the result
        tool_msgs = [m for m in result if m.role == Role.TOOL]
        for tm in tool_msgs:
            # Find corresponding assistant msg
            idx = result.index(tm)
            prev = result[idx - 1]
            assert prev.role == Role.ASSISTANT and prev.tool_calls


class TestNoOpWhenMessagesFew:
    def test_noop_when_messages_fewer_than_keep_plus_one(self):
        mgr = ContextManager()
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="Hello"),
            Message(role=Role.ASSISTANT, content="Hi"),
        ]
        result = mgr.smart_compact(messages, keep_recent=10)
        assert result == messages

    def test_noop_when_exact_threshold(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]
        # Add exactly keep_recent messages
        for i in range(5):
            messages.append(Message(role=Role.USER, content=f"U{i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"A{i}"))
        # 11 total messages, keep_recent=10 -> 11 <= 10+1, so no-op
        result = mgr.smart_compact(messages, keep_recent=10)
        assert result == messages

    def test_noop_with_empty_messages(self):
        mgr = ContextManager()
        result = mgr.smart_compact([], keep_recent=10)
        assert result == []

    def test_noop_single_message(self):
        mgr = ContextManager()
        messages = [Message(role=Role.USER, content="Solo")]
        result = mgr.smart_compact(messages, keep_recent=10)
        assert result == messages


class TestSummaryAppendedToSystemMessage:
    def test_summary_appended_to_system_not_as_second_system(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="Base system prompt")]
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)

        # Count system messages — should be exactly 1
        system_msgs = [m for m in result if m.role == Role.SYSTEM]
        assert len(system_msgs) == 1

        # The system message should contain the summary
        assert "Conversation summary" in result[0].content
        # Original system content should be preserved
        assert "Base system prompt" in result[0].content

    def test_summary_mentions_compacted_count(self):
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)
        assert "messages compacted" in result[0].content

    def test_no_system_message_uses_user_context_note(self):
        """When there is no system message, summary is injected as a user message."""
        mgr = ContextManager()
        messages = []
        for i in range(30):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        result = mgr.smart_compact(messages, keep_recent=10)
        # First message should be a USER message with the summary
        assert result[0].role == Role.USER
        assert "Conversation summary" in result[0].content


class TestToolCallPairsAtCutBoundary:
    def test_cut_boundary_does_not_orphan_tool_result(self):
        """When the cut point lands between a tool_call assistant and its TOOL result,
        the cut should be adjusted to keep or exclude both."""
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]

        # Build messages so that a tool pair straddles the cut boundary
        for i in range(10):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        # Insert a tool call pair that will be near the cut point
        tc = ToolCall(id="tc_boundary", name="read_file", arguments={"path": "x.py"})
        messages.append(Message(role=Role.USER, content="Read x.py"))
        messages.append(Message(role=Role.ASSISTANT, content="Reading", tool_calls=[tc]))
        messages.append(
            Message(role=Role.TOOL, content="file contents", tool_call_id="tc_boundary")
        )
        messages.append(Message(role=Role.ASSISTANT, content="Here is the file"))

        # Add more messages after to push the tool pair toward the cut
        for i in range(10):
            messages.append(Message(role=Role.USER, content=f"More user {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"More asst {i}"))

        result = mgr.smart_compact(messages, keep_recent=15)

        # Verify invariant: no orphaned TOOL messages
        for idx, msg in enumerate(result):
            if msg.role == Role.TOOL:
                assert idx > 0
                prev = result[idx - 1]
                assert prev.role == Role.ASSISTANT and prev.tool_calls, (
                    f"Orphaned TOOL message at index {idx} in compacted result"
                )

    def test_multiple_tool_pairs_at_boundary(self):
        """Multiple consecutive tool pairs near the cut should all remain intact."""
        mgr = ContextManager()
        messages = [Message(role=Role.SYSTEM, content="System")]

        for i in range(8):
            messages.append(Message(role=Role.USER, content=f"User {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Asst {i}"))

        # Add 3 consecutive tool pairs
        for j in range(3):
            tc = ToolCall(id=f"tc_{j}", name="read_file", arguments={"path": f"f{j}.py"})
            messages.append(Message(role=Role.ASSISTANT, content=f"Reading f{j}", tool_calls=[tc]))
            messages.append(Message(role=Role.TOOL, content=f"content {j}", tool_call_id=f"tc_{j}"))

        messages.append(Message(role=Role.ASSISTANT, content="All files read."))

        for i in range(12):
            messages.append(Message(role=Role.USER, content=f"Later {i}"))
            messages.append(Message(role=Role.ASSISTANT, content=f"Reply {i}"))

        result = mgr.smart_compact(messages, keep_recent=15)

        # Verify no orphaned TOOL messages
        for idx, msg in enumerate(result):
            if msg.role == Role.TOOL:
                assert idx > 0
                prev = result[idx - 1]
                assert prev.role == Role.ASSISTANT and prev.tool_calls
