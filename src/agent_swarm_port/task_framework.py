"""In-memory task framework mirroring the source's task registration/update helpers."""

from __future__ import annotations

import secrets
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from .constants import PANEL_GRACE_MS, TASK_ID_ALPHABET, TASK_ID_PREFIXES, TERMINAL_BACKGROUND_TASK_STATUSES
from .models import AppState, TaskStateBase

T = TypeVar("T", bound=TaskStateBase)


def is_terminal_task_status(status: str) -> bool:
    return status in TERMINAL_BACKGROUND_TASK_STATUSES


def generate_task_id(task_type: str) -> str:
    prefix = TASK_ID_PREFIXES.get(task_type, "x")
    return prefix + "".join(secrets.choice(TASK_ID_ALPHABET) for _ in range(8))


def create_task_state_base(task_id: str, task_type: str, description: str, output_file: str, tool_use_id: str | None = None) -> TaskStateBase:
    return TaskStateBase(
        id=task_id,
        type=task_type,
        status="pending",
        description=description,
        start_time=int(time.time() * 1000),
        output_file=output_file,
        output_offset=0,
        notified=False,
        tool_use_id=tool_use_id,
    )


def register_task(task: TaskStateBase, state: AppState) -> AppState:
    existing = state.tasks.get(task.id)
    merged = task
    if existing is not None and hasattr(existing, "retain"):
        merged = replace(task)
        if hasattr(merged, "retain"):
            setattr(merged, "retain", getattr(existing, "retain"))
        if hasattr(merged, "messages"):
            setattr(merged, "messages", getattr(existing, "messages", None))
        if hasattr(merged, "disk_loaded"):
            setattr(merged, "disk_loaded", getattr(existing, "disk_loaded", False))
        if hasattr(merged, "pending_messages"):
            setattr(merged, "pending_messages", list(getattr(existing, "pending_messages", [])))
        merged.start_time = existing.start_time
    next_tasks = dict(state.tasks)
    next_tasks[task.id] = merged
    return replace(state, tasks=next_tasks)


def update_task_state(task_id: str, state: AppState, updater: Callable[[T], T]) -> AppState:
    task = state.tasks.get(task_id)
    if task is None:
        return state
    updated = updater(task)  # type: ignore[arg-type]
    if updated is task:
        return state
    next_tasks = dict(state.tasks)
    next_tasks[task_id] = updated
    return replace(state, tasks=next_tasks)


def evict_terminal_task(task_id: str, state: AppState) -> AppState:
    task = state.tasks.get(task_id)
    if task is None or not is_terminal_task_status(task.status) or not task.notified:
        return state
    if hasattr(task, "evict_after"):
        evict_after = getattr(task, "evict_after")
        if evict_after is not None and evict_after > int(time.time() * 1000):
            return state
    next_tasks = dict(state.tasks)
    next_tasks.pop(task_id, None)
    return replace(state, tasks=next_tasks)


def mark_task_terminal(task_id: str, state: AppState, status: str, error: str | None = None) -> AppState:
    def apply(task: T) -> T:
        if task.status != "running":
            return task
        updated = replace(task, status=status, notified=True, end_time=int(time.time() * 1000))
        if hasattr(updated, "error"):
            setattr(updated, "error", error)
        if hasattr(updated, "evict_after"):
            setattr(updated, "evict_after", int(time.time() * 1000) + PANEL_GRACE_MS)
        return updated

    return update_task_state(task_id, state, apply)
