"""Backend orchestration wrappers.

This module wraps the in-process spawn + runner flow into a backend-style API
so the Python port can start/stop teammates similarly to the source's backend
abstraction layer.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import time
from typing import Any

from .inprocess_runner import InProcessExecutor, run_in_process_teammate
from .models import AppState, TeamMember
from .runtime_state import AppStateStore
from .spawn_inprocess import spawn_in_process_teammate
from .team_store import read_team_file, set_member_active, write_team_file


@dataclass
class InProcessBackendHandle:
    task_id: str
    agent_id: str
    asyncio_task: asyncio.Task[Any]


async def _run_and_finalize(
    *,
    store: AppStateStore,
    task_id: str,
    executor: InProcessExecutor,
    task_list_id: str | None,
) -> Any:
    try:
        return await run_in_process_teammate(
            task_id=task_id,
            store=store,
            executor=executor,
            task_list_id=task_list_id,
        )
    finally:
        state = store.get_state()
        task = state.tasks.get(task_id)
        if getattr(task, "identity", None) is not None:
            set_member_active(task.identity.team_name, task.identity.agent_id, False)


def start_inprocess_backend(
    *,
    store: AppStateStore,
    name: str,
    team_name: str,
    prompt: str,
    parent_session_id: str,
    executor: InProcessExecutor,
    color: str | None = None,
    model: str | None = None,
    plan_mode_required: bool = False,
    task_list_id: str | None = None,
) -> InProcessBackendHandle:
    next_state, spawn_result = spawn_in_process_teammate(
        store.get_state(),
        name=name,
        team_name=team_name,
        prompt=prompt,
        parent_session_id=parent_session_id,
        color=color,
        plan_mode_required=plan_mode_required,
        model=model,
    )
    store.set_state(lambda _prev: next_state)

    team_file = read_team_file(team_name)
    if team_file is not None and not any(
        member.agent_id == spawn_result["agent_id"] for member in team_file.members
    ):
        team_file.members.append(
            TeamMember(
                agent_id=spawn_result["agent_id"],
                name=name,
                agent_type="in-process",
                model=model,
                prompt=prompt,
                color=color,
                plan_mode_required=plan_mode_required,
                joined_at=int(time() * 1000),
                cwd=store.get_state().current_cwd,
                session_id=parent_session_id,
                backend_type="in-process",
                is_active=True,
            )
        )
        write_team_file(team_name, team_file)

    runner_task = asyncio.create_task(
        _run_and_finalize(
            store=store,
            task_id=spawn_result["task_id"],
            executor=executor,
            task_list_id=task_list_id,
        )
    )
    return InProcessBackendHandle(
        task_id=spawn_result["task_id"],
        agent_id=spawn_result["agent_id"],
        asyncio_task=runner_task,
    )


def stop_inprocess_backend(handle: InProcessBackendHandle, *, store: AppStateStore) -> None:
    state = store.get_state()
    task = state.tasks.get(handle.task_id)
    if getattr(task, "abort_controller", None) is not None:
        task.abort_controller.abort("backend-stop")
