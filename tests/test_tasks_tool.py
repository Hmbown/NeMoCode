# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Tests for in-session task tracking tools."""

from __future__ import annotations

import json

import pytest

from nemocode.tools.tasks import _reset_tasks, create_task, list_tasks, update_task


@pytest.fixture(autouse=True)
def reset():
    _reset_tasks()
    yield
    _reset_tasks()


class TestCreateTask:
    @pytest.mark.asyncio
    async def test_create_basic(self):
        result = json.loads(await create_task("Fix bug"))
        assert result["status"] == "ok"
        assert result["task"]["title"] == "Fix bug"
        assert result["task"]["status"] == "pending"
        assert result["task"]["id"] == 1

    @pytest.mark.asyncio
    async def test_create_with_priority(self):
        result = json.loads(await create_task("Urgent fix", priority="high"))
        assert result["task"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_ids_increment(self):
        r1 = json.loads(await create_task("A"))
        r2 = json.loads(await create_task("B"))
        assert r1["task"]["id"] == 1
        assert r2["task"]["id"] == 2


class TestUpdateTask:
    @pytest.mark.asyncio
    async def test_update_status(self):
        await create_task("Task 1")
        result = json.loads(await update_task(1, status="done"))
        assert result["task"]["status"] == "done"

    @pytest.mark.asyncio
    async def test_update_not_found(self):
        result = json.loads(await update_task(999, status="done"))
        assert "error" in result


class TestListTasks:
    @pytest.mark.asyncio
    async def test_list_empty(self):
        result = json.loads(await list_tasks())
        assert result["count"] == 0

    @pytest.mark.asyncio
    async def test_list_all(self):
        await create_task("A")
        await create_task("B")
        result = json.loads(await list_tasks())
        assert result["count"] == 2

    @pytest.mark.asyncio
    async def test_list_filtered(self):
        await create_task("A")
        await create_task("B")
        await update_task(1, status="done")
        result = json.loads(await list_tasks(status_filter="done"))
        assert result["count"] == 1
        assert result["tasks"][0]["title"] == "A"
