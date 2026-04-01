from __future__ import annotations

import asyncio
import json

import pytest

from agent_swarm_port.in_process_teammate import inject_user_message_to_teammate
from agent_swarm_port.inprocess_runner import run_in_process_teammate, wait_for_next_prompt_or_shutdown
from agent_swarm_port.mailbox import TeammateMessage, clear_mailbox, read_mailbox, write_to_mailbox
from agent_swarm_port.runtime_state import AppStateStore
from agent_swarm_port.spawn_inprocess import spawn_in_process_teammate
from agent_swarm_port.task_list import create_task, get_task


@pytest.mark.asyncio
async def test_wait_for_next_prompt_priority_order(temp_home):
    store = AppStateStore()
    state, spawned = spawn_in_process_teammate(
        store.get_state(),
        name="worker1",
        team_name="alpha",
        prompt="initial",
        parent_session_id="alpha",
    )
    store.set_state(lambda _prev: state)

    injected = inject_user_message_to_teammate(spawned["task_id"], "from user", store.get_state())
    store.set_state(lambda _prev: injected)
    wait_result = await wait_for_next_prompt_or_shutdown(
        task_id=spawned["task_id"],
        store=store,
        task_list_id="alpha",
    )
    assert wait_result.type == "new_message"
    assert wait_result.from_ == "user"
    assert wait_result.message == "from user"

    write_to_mailbox(
        "worker1",
        TeammateMessage(from_="peer", text="peer message", timestamp="2026-03-31T00:00:00Z"),
        "alpha",
    )
    write_to_mailbox(
        "worker1",
        TeammateMessage(from_="team-lead", text="leader message", timestamp="2026-03-31T00:00:01Z"),
        "alpha",
    )
    wait_result = await wait_for_next_prompt_or_shutdown(
        task_id=spawned["task_id"],
        store=store,
        task_list_id="alpha",
    )
    assert wait_result.type == "new_message"
    assert wait_result.from_ == "team-lead"
    assert wait_result.message == "leader message"

    write_to_mailbox(
        "worker1",
        TeammateMessage(from_="peer", text="leftover", timestamp="2026-03-31T00:00:02Z"),
        "alpha",
    )
    shutdown_payload = json.dumps(
        {
            "type": "shutdown_request",
            "requestId": "shutdown-123@worker1",
            "from": "team-lead",
            "reason": "stop",
            "timestamp": "2026-03-31T00:00:03Z",
        }
    )
    write_to_mailbox(
        "worker1",
        TeammateMessage(from_="team-lead", text=shutdown_payload, timestamp="2026-03-31T00:00:03Z"),
        "alpha",
    )
    wait_result = await wait_for_next_prompt_or_shutdown(
        task_id=spawned["task_id"],
        store=store,
        task_list_id="alpha",
    )
    assert wait_result.type == "shutdown_request"
    assert wait_result.request is not None
    assert wait_result.request.reason == "stop"

    clear_mailbox("worker1", "alpha")
    create_task(
        "alpha",
        {
            "subject": "Claim me",
            "description": "Open task",
            "active_form": None,
            "status": "pending",
            "owner": None,
            "blocks": [],
            "blocked_by": [],
            "metadata": None,
        },
    )
    wait_result = await wait_for_next_prompt_or_shutdown(
        task_id=spawned["task_id"],
        store=store,
        task_list_id="alpha",
    )
    assert wait_result.type == "new_message"
    assert wait_result.from_ == "task-list"
    claimed = get_task("alpha", "1")
    assert claimed is not None
    assert claimed.owner == "worker1"
    assert claimed.status == "in_progress"


@pytest.mark.asyncio
async def test_run_in_process_teammate_sends_idle_notification_and_completes(temp_home):
    store = AppStateStore()
    state, spawned = spawn_in_process_teammate(
        store.get_state(),
        name="worker1",
        team_name="alpha",
        prompt="initial prompt",
        parent_session_id="alpha",
    )
    store.set_state(lambda _prev: state)

    calls: list[str] = []

    async def executor(prompt, task, store, abort_controller):
        calls.append(prompt)
        task.abort_controller.abort("done")
        return {"assistant_message": f"handled: {prompt}", "summary": "completed turn"}

    result = await run_in_process_teammate(
        task_id=spawned["task_id"],
        store=store,
        executor=executor,
        task_list_id="alpha",
    )
    assert result.success is True
    assert result.final_status == "completed"
    assert calls == ["initial prompt"]

    leader_mail = read_mailbox("team-lead", "alpha")
    assert leader_mail, "expected idle notification in leader mailbox"
    payload = json.loads(leader_mail[-1].text)
    assert payload["type"] == "idle_notification"
    assert payload["from"] == "worker1"

    task = store.get_state().tasks[spawned["task_id"]]
    assert task.status == "completed"
