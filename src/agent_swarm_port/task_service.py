"""Task-list service layer.

This module ports the tool-facing semantics of TaskCreate/TaskUpdate/TaskList/
TaskGet onto the file-backed task subsystem in :mod:`task_list`.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contexts import get_agent_id, get_agent_name, get_team_name, get_teammate_color
from .mailbox import TeammateMessage, create_task_assignment_message, protocol_to_json, write_to_mailbox
from .models import TaskListTask
from .task_list import (
    block_task,
    create_task,
    delete_task,
    get_task,
    get_task_list_id,
    list_tasks,
    update_task,
)


def resolve_task_list_id(
    session_id: str,
    *,
    explicit_task_list_id: str | None = None,
    teammate_team_name: str | None = None,
    leader_team_name: str | None = None,
) -> str:
    return get_task_list_id(
        session_id,
        explicit_task_list_id=explicit_task_list_id,
        teammate_team_name=teammate_team_name,
        leader_team_name=leader_team_name,
    )


def create_task_entry(
    *,
    task_list_id: str,
    subject: str,
    description: str,
    active_form: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> TaskListTask:
    task_id = create_task(
        task_list_id,
        {
            "subject": subject,
            "description": description,
            "active_form": active_form,
            "status": "pending",
            "owner": None,
            "blocks": [],
            "blocked_by": [],
            "metadata": metadata,
        },
    )
    task = get_task(task_list_id, task_id)
    if task is None:
        raise RuntimeError(f"Created task {task_id} but could not read it back")
    return task


def get_task_entry(*, task_list_id: str, task_id: str) -> TaskListTask | None:
    return get_task(task_list_id, task_id)


def list_task_entries(*, task_list_id: str, include_internal: bool = False) -> list[dict[str, Any]]:
    tasks = list_tasks(task_list_id)
    if not include_internal:
        tasks = [task for task in tasks if not ((task.metadata or {}).get("_internal"))]
    resolved_ids = {task.id for task in tasks if task.status == "completed"}
    return [
        {
            "id": task.id,
            "subject": task.subject,
            "description": task.description,
            "status": task.status,
            "owner": task.owner,
            "blocked_by": [item for item in task.blocked_by if item not in resolved_ids],
            "blocks": list(task.blocks),
            "active_form": task.active_form,
            "metadata": task.metadata,
        }
        for task in tasks
    ]


def update_task_entry(
    *,
    task_list_id: str,
    task_id: str,
    subject: str | None = None,
    description: str | None = None,
    active_form: str | None = None,
    status: str | None = None,
    owner: str | None = None,
    add_blocks: list[str] | None = None,
    add_blocked_by: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    notify_assignment: bool = True,
    team_name: str | None = None,
    sender_name: str | None = None,
    sender_color: str | None = None,
) -> dict[str, Any]:
    existing_task = get_task(task_list_id, task_id)
    if existing_task is None:
        return {
            "success": False,
            "task_id": task_id,
            "updated_fields": [],
            "error": "Task not found",
        }

    updated_fields: list[str] = []
    updates: dict[str, Any] = {}

    if subject is not None and subject != existing_task.subject:
        updates["subject"] = subject
        updated_fields.append("subject")
    if description is not None and description != existing_task.description:
        updates["description"] = description
        updated_fields.append("description")
    if active_form is not None and active_form != existing_task.active_form:
        updates["activeForm"] = active_form
        updated_fields.append("activeForm")
    if owner is not None and owner != existing_task.owner:
        updates["owner"] = owner
        updated_fields.append("owner")

    if status is not None:
        if status == "deleted":
            deleted = delete_task(task_list_id, task_id)
            return {
                "success": deleted,
                "task_id": task_id,
                "updated_fields": ["deleted"] if deleted else [],
                "status_change": {"from": existing_task.status, "to": "deleted"}
                if deleted
                else None,
                "error": None if deleted else "Failed to delete task",
            }
        if status != existing_task.status:
            updates["status"] = status
            updated_fields.append("status")

    if metadata is not None:
        merged = {**(existing_task.metadata or {})}
        for key, value in metadata.items():
            if value is None:
                merged.pop(key, None)
            else:
                merged[key] = value
        updates["metadata"] = merged
        updated_fields.append("metadata")

    if updates:
        update_task(task_list_id, task_id, updates)

    if notify_assignment and updates.get("owner"):
        effective_team_name = team_name or get_team_name()
        effective_sender_name = sender_name or get_agent_name() or "team-lead"
        effective_sender_color = sender_color or get_teammate_color()
        if effective_team_name:
            assignment = create_task_assignment_message(
                task_id,
                existing_task.subject,
                existing_task.description,
                effective_sender_name,
            )
            write_to_mailbox(
                updates["owner"],
                TeammateMessage(
                    from_=effective_sender_name,
                    text=protocol_to_json(assignment),
                    timestamp=assignment.timestamp,
                    color=effective_sender_color,
                ),
                effective_team_name,
            )

    if add_blocks:
        new_blocks = [item for item in add_blocks if item not in existing_task.blocks]
        for block_id in new_blocks:
            block_task(task_list_id, task_id, block_id)
        if new_blocks:
            updated_fields.append("blocks")

    if add_blocked_by:
        new_blocked_by = [
            item for item in add_blocked_by if item not in existing_task.blocked_by
        ]
        for blocker_id in new_blocked_by:
            block_task(task_list_id, blocker_id, task_id)
        if new_blocked_by:
            updated_fields.append("blockedBy")

    refreshed = get_task(task_list_id, task_id)
    return {
        "success": True,
        "task_id": task_id,
        "updated_fields": updated_fields,
        "status_change": {"from": existing_task.status, "to": updates["status"]}
        if "status" in updates
        else None,
        "verification_nudge_needed": bool(
            get_agent_id() is None
            and updates.get("status") == "completed"
            and _needs_verification_nudge(task_list_id)
        ),
        "task": asdict(refreshed) if refreshed else None,
    }


def _needs_verification_nudge(task_list_id: str) -> bool:
    tasks = list_tasks(task_list_id)
    if not tasks or len(tasks) < 3:
        return False
    if any(task.status != "completed" for task in tasks):
        return False
    return not any("verif" in task.subject.lower() for task in tasks)
