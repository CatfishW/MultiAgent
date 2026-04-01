"""In-process teammate task helpers.

Ports ``InProcessTeammateTask.tsx`` and related helper behavior.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .constants import TEAMMATE_MESSAGES_UI_CAP
from .models import AppState, InProcessTeammateTaskState
from .task_framework import is_terminal_task_status, update_task_state


def append_capped_message(messages: list[dict[str, Any]] | None, item: dict[str, Any]) -> list[dict[str, Any]]:
    if not messages:
        return [item]
    if len(messages) >= TEAMMATE_MESSAGES_UI_CAP:
        next_messages = list(messages[-(TEAMMATE_MESSAGES_UI_CAP - 1) :])
        next_messages.append(item)
        return next_messages
    return [*messages, item]


def is_in_process_teammate_task(task: object) -> bool:
    return isinstance(task, InProcessTeammateTaskState)


def request_teammate_shutdown(task_id: str, state: AppState) -> AppState:
    return update_task_state(
        task_id,
        state,
        lambda task: task if task.status != "running" or task.shutdown_requested else replace(task, shutdown_requested=True),
    )


def append_teammate_message(task_id: str, message: dict[str, Any], state: AppState) -> AppState:
    return update_task_state(
        task_id,
        state,
        lambda task: task if task.status != "running" else replace(task, messages=append_capped_message(task.messages, message)),
    )


def inject_user_message_to_teammate(task_id: str, message: str, state: AppState) -> AppState:
    def apply(task: InProcessTeammateTaskState) -> InProcessTeammateTaskState:
        if is_terminal_task_status(task.status):
            return task
        rendered = {"type": "user", "content": message}
        return replace(
            task,
            pending_user_messages=[*task.pending_user_messages, message],
            messages=append_capped_message(task.messages, rendered),
        )

    return update_task_state(task_id, state, apply)


def find_teammate_task_by_agent_id(agent_id: str, tasks: dict[str, object]) -> InProcessTeammateTaskState | None:
    fallback: InProcessTeammateTaskState | None = None
    for task in tasks.values():
        if isinstance(task, InProcessTeammateTaskState) and task.identity.agent_id == agent_id:
            if task.status == "running":
                return task
            if fallback is None:
                fallback = task
    return fallback


def get_all_in_process_teammate_tasks(tasks: dict[str, object]) -> list[InProcessTeammateTaskState]:
    return [task for task in tasks.values() if isinstance(task, InProcessTeammateTaskState)]


def get_running_teammates_sorted(tasks: dict[str, object]) -> list[InProcessTeammateTaskState]:
    return sorted(
        [task for task in get_all_in_process_teammate_tasks(tasks) if task.status == "running"],
        key=lambda item: item.identity.agent_name,
    )
