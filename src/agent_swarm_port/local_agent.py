"""Local background agent task helpers.

Ports the task-state and message-queue behavior from
``src/tasks/LocalAgentTask/LocalAgentTask.tsx``.
"""

from __future__ import annotations

import time
from dataclasses import replace

from .abort import AbortController, create_abort_controller, create_child_abort_controller
from .models import AgentProgress, AppState, LocalAgentTaskState
from .task_framework import create_task_state_base, generate_task_id, update_task_state


def is_local_agent_task(task: object) -> bool:
    return isinstance(task, LocalAgentTaskState)


def queue_pending_message(task_id: str, message: str, state: AppState) -> AppState:
    return update_task_state(task_id, state, lambda task: replace(task, pending_messages=[*task.pending_messages, message]))


def drain_pending_messages(task_id: str, state: AppState) -> tuple[AppState, list[str]]:
    task = state.tasks.get(task_id)
    if not isinstance(task, LocalAgentTaskState) or not task.pending_messages:
        return state, []
    drained = list(task.pending_messages)
    return update_task_state(task_id, state, lambda current: replace(current, pending_messages=[])), drained


def append_message_to_local_agent(task_id: str, message: dict, state: AppState) -> AppState:
    return update_task_state(task_id, state, lambda task: replace(task, messages=[*(task.messages or []), message]))


def register_async_agent(
    *,
    agent_id: str,
    description: str,
    prompt: str,
    selected_agent: dict | None,
    state: AppState,
    parent_abort_controller: AbortController | None = None,
    tool_use_id: str | None = None,
) -> tuple[AppState, LocalAgentTaskState]:
    abort_controller = create_child_abort_controller(parent_abort_controller) if parent_abort_controller else create_abort_controller()
    base = create_task_state_base(
        task_id=agent_id,
        task_type="local_agent",
        description=description,
        output_file=f"{agent_id}.output",
        tool_use_id=tool_use_id,
    )
    task = LocalAgentTaskState(
        **{**base.__dict__, "status": "running"},
        agent_id=agent_id,
        prompt=prompt,
        selected_agent=selected_agent,
        agent_type=(selected_agent or {}).get("agentType", "general-purpose"),
        abort_controller=abort_controller,
        retrieved=False,
        last_reported_tool_count=0,
        last_reported_token_count=0,
        is_backgrounded=True,
        pending_messages=[],
        retain=False,
        disk_loaded=False,
    )
    next_tasks = dict(state.tasks)
    next_tasks[agent_id] = task
    return replace(state, tasks=next_tasks), task


def register_agent_foreground(
    *,
    agent_id: str,
    description: str,
    prompt: str,
    selected_agent: dict | None,
    state: AppState,
    tool_use_id: str | None = None,
) -> tuple[AppState, LocalAgentTaskState]:
    base = create_task_state_base(
        task_id=agent_id,
        task_type="local_agent",
        description=description,
        output_file=f"{agent_id}.output",
        tool_use_id=tool_use_id,
    )
    task = LocalAgentTaskState(
        **{**base.__dict__, "status": "running"},
        agent_id=agent_id,
        prompt=prompt,
        selected_agent=selected_agent,
        agent_type=(selected_agent or {}).get("agentType", "general-purpose"),
        abort_controller=create_abort_controller(),
        retrieved=False,
        last_reported_tool_count=0,
        last_reported_token_count=0,
        is_backgrounded=False,
        pending_messages=[],
        retain=False,
        disk_loaded=False,
    )
    next_tasks = dict(state.tasks)
    next_tasks[agent_id] = task
    return replace(state, tasks=next_tasks), task


def update_agent_progress(task_id: str, progress: AgentProgress, state: AppState) -> AppState:
    return update_task_state(task_id, state, lambda task: replace(task, progress=progress, last_reported_tool_count=progress.tool_count, last_reported_token_count=progress.token_count))


def complete_agent_task(task_id: str, result: dict, state: AppState) -> AppState:
    return update_task_state(
        task_id,
        state,
        lambda task: replace(
            task,
            status="completed",
            result=result,
            end_time=int(time.time() * 1000),
            abort_controller=None,
        ),
    )


def fail_agent_task(task_id: str, error: str, state: AppState) -> AppState:
    return update_task_state(
        task_id,
        state,
        lambda task: replace(
            task,
            status="failed",
            error=error,
            end_time=int(time.time() * 1000),
            abort_controller=None,
        ),
    )


def kill_async_agent(task_id: str, state: AppState) -> AppState:
    task = state.tasks.get(task_id)
    if isinstance(task, LocalAgentTaskState) and task.abort_controller is not None:
        task.abort_controller.abort("killed")
    return update_task_state(task_id, state, lambda current: replace(current, status="killed", abort_controller=None, end_time=int(time.time() * 1000), notified=True))
