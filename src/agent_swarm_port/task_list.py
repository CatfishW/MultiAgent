"""File-backed task list subsystem mirroring ``src/utils/tasks.ts``."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .locks import file_lock
from .models import AgentStatus, TaskListTask, TeamMember
from .paths import get_task_path, get_tasks_dir, get_team_file_path, sanitize_name, sanitize_path_component


HIGH_WATER_MARK_FILE = ".highwatermark"
LOCK_FILE = ".lock"
_leader_team_name: str | None = None


class ClaimTaskResult(dict):
    pass


def set_leader_team_name(team_name: str) -> None:
    global _leader_team_name
    _leader_team_name = team_name


def clear_leader_team_name() -> None:
    global _leader_team_name
    _leader_team_name = None


def get_task_list_id(session_id: str, explicit_task_list_id: str | None = None, teammate_team_name: str | None = None, leader_team_name: str | None = None) -> str:
    return explicit_task_list_id or teammate_team_name or leader_team_name or _leader_team_name or session_id


def ensure_tasks_dir(task_list_id: str) -> Path:
    path = get_tasks_dir(task_list_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _lock_path(task_list_id: str) -> Path:
    path = ensure_tasks_dir(task_list_id) / LOCK_FILE
    if not path.exists():
        path.write_text("", encoding="utf-8")
    return path.with_suffix(".lck")


def _high_water_mark_path(task_list_id: str) -> Path:
    return ensure_tasks_dir(task_list_id) / HIGH_WATER_MARK_FILE


def _read_high_water_mark(task_list_id: str) -> int:
    try:
        return int(_high_water_mark_path(task_list_id).read_text(encoding="utf-8").strip() or "0")
    except (FileNotFoundError, ValueError):
        return 0


def _write_high_water_mark(task_list_id: str, value: int) -> None:
    _high_water_mark_path(task_list_id).write_text(str(value), encoding="utf-8")


def _find_highest_task_id_from_files(task_list_id: str) -> int:
    highest = 0
    for path in ensure_tasks_dir(task_list_id).glob("*.json"):
        try:
            highest = max(highest, int(path.stem))
        except ValueError:
            continue
    return highest


def _find_highest_task_id(task_list_id: str) -> int:
    return max(_find_highest_task_id_from_files(task_list_id), _read_high_water_mark(task_list_id))


def reset_task_list(task_list_id: str) -> None:
    lock = _lock_path(task_list_id)
    with file_lock(lock):
        highest = _find_highest_task_id_from_files(task_list_id)
        if highest > _read_high_water_mark(task_list_id):
            _write_high_water_mark(task_list_id, highest)
        for path in ensure_tasks_dir(task_list_id).glob("*.json"):
            path.unlink(missing_ok=True)


def create_task(task_list_id: str, task_data: dict[str, Any]) -> str:
    lock = _lock_path(task_list_id)
    with file_lock(lock):
        next_id = _find_highest_task_id(task_list_id) + 1
        task = TaskListTask(id=str(next_id), **task_data)
        get_task_path(task_list_id, task.id).write_text(json.dumps(asdict(task), indent=2), encoding="utf-8")
        return task.id


def get_task(task_list_id: str, task_id: str) -> TaskListTask | None:
    path = get_task_path(task_list_id, task_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    status = data.get("status")
    if status == "open":
        data["status"] = "pending"
    elif status == "resolved":
        data["status"] = "completed"
    elif status in {"planning", "implementing", "reviewing", "verifying"}:
        data["status"] = "in_progress"
    return TaskListTask(
        id=str(data["id"]),
        subject=data["subject"],
        description=data["description"],
        active_form=data.get("activeForm") or data.get("active_form"),
        owner=data.get("owner"),
        status=data["status"],
        blocks=list(data.get("blocks", [])),
        blocked_by=list(data.get("blockedBy") or data.get("blocked_by", [])),
        metadata=data.get("metadata"),
    )


def update_task(task_list_id: str, task_id: str, updates: dict[str, Any]) -> TaskListTask | None:
    task = get_task(task_list_id, task_id)
    if task is None:
        return None
    lock = get_task_path(task_list_id, task_id).with_suffix(".json.lock")
    with file_lock(lock):
        current = get_task(task_list_id, task_id)
        if current is None:
            return None
        merged = asdict(current)
        for key, value in updates.items():
            if key == "blockedBy":
                merged["blocked_by"] = value
            elif key == "activeForm":
                merged["active_form"] = value
            else:
                merged[key] = value
        updated = TaskListTask(
            id=task_id,
            subject=merged["subject"],
            description=merged["description"],
            active_form=merged.get("active_form") or merged.get("activeForm"),
            owner=merged.get("owner"),
            status=merged["status"],
            blocks=list(merged.get("blocks", [])),
            blocked_by=list(merged.get("blocked_by") or merged.get("blockedBy", [])),
            metadata=merged.get("metadata"),
        )
        get_task_path(task_list_id, task_id).write_text(json.dumps(asdict(updated), indent=2), encoding="utf-8")
        return updated


def delete_task(task_list_id: str, task_id: str) -> bool:
    path = get_task_path(task_list_id, task_id)
    try:
        numeric_id = int(task_id)
    except ValueError:
        numeric_id = -1
    if numeric_id > _read_high_water_mark(task_list_id):
        _write_high_water_mark(task_list_id, numeric_id)
    if not path.exists():
        return False
    path.unlink()
    for task in list_tasks(task_list_id):
        new_blocks = [value for value in task.blocks if value != task_id]
        new_blocked_by = [value for value in task.blocked_by if value != task_id]
        if new_blocks != task.blocks or new_blocked_by != task.blocked_by:
            update_task(task_list_id, task.id, {"blocks": new_blocks, "blockedBy": new_blocked_by})
    return True


def list_tasks(task_list_id: str) -> list[TaskListTask]:
    tasks: list[TaskListTask] = []
    for path in ensure_tasks_dir(task_list_id).glob("*.json"):
        task = get_task(task_list_id, path.stem)
        if task is not None:
            tasks.append(task)
    return sorted(tasks, key=lambda item: int(item.id) if item.id.isdigit() else item.id)


def block_task(task_list_id: str, from_task_id: str, to_task_id: str) -> bool:
    from_task = get_task(task_list_id, from_task_id)
    to_task = get_task(task_list_id, to_task_id)
    if from_task is None or to_task is None:
        return False
    if to_task_id not in from_task.blocks:
        update_task(task_list_id, from_task_id, {"blocks": [*from_task.blocks, to_task_id]})
    if from_task_id not in to_task.blocked_by:
        update_task(task_list_id, to_task_id, {"blockedBy": [*to_task.blocked_by, from_task_id]})
    return True


def claim_task(task_list_id: str, task_id: str, claimant_agent_id: str, check_agent_busy: bool = False) -> ClaimTaskResult:
    if get_task(task_list_id, task_id) is None:
        return ClaimTaskResult(success=False, reason="task_not_found")
    if check_agent_busy:
        return _claim_task_with_busy_check(task_list_id, task_id, claimant_agent_id)
    lock = get_task_path(task_list_id, task_id).with_suffix(".json.claim.lock")
    with file_lock(lock):
        task = get_task(task_list_id, task_id)
        if task is None:
            return ClaimTaskResult(success=False, reason="task_not_found")
        if task.owner and task.owner != claimant_agent_id:
            return ClaimTaskResult(success=False, reason="already_claimed", task=task)
        if task.status == "completed":
            return ClaimTaskResult(success=False, reason="already_resolved", task=task)
        unresolved = {item.id for item in list_tasks(task_list_id) if item.status != "completed"}
        blocked_by = [item_id for item_id in task.blocked_by if item_id in unresolved]
        if blocked_by:
            return ClaimTaskResult(success=False, reason="blocked", task=task, blockedByTasks=blocked_by)
        updated = update_task(task_list_id, task_id, {"owner": claimant_agent_id})
        return ClaimTaskResult(success=True, task=updated)


def _claim_task_with_busy_check(task_list_id: str, task_id: str, claimant_agent_id: str) -> ClaimTaskResult:
    with file_lock(_lock_path(task_list_id)):
        tasks = list_tasks(task_list_id)
        task = next((item for item in tasks if item.id == task_id), None)
        if task is None:
            return ClaimTaskResult(success=False, reason="task_not_found")
        if task.owner and task.owner != claimant_agent_id:
            return ClaimTaskResult(success=False, reason="already_claimed", task=task)
        if task.status == "completed":
            return ClaimTaskResult(success=False, reason="already_resolved", task=task)
        unresolved = {item.id for item in tasks if item.status != "completed"}
        blocked_by = [item_id for item_id in task.blocked_by if item_id in unresolved]
        if blocked_by:
            return ClaimTaskResult(success=False, reason="blocked", task=task, blockedByTasks=blocked_by)
        busy_with = [item.id for item in tasks if item.status != "completed" and item.owner == claimant_agent_id and item.id != task_id]
        if busy_with:
            return ClaimTaskResult(success=False, reason="agent_busy", task=task, busyWithTasks=busy_with)
        updated = update_task(task_list_id, task_id, {"owner": claimant_agent_id})
        return ClaimTaskResult(success=True, task=updated)


def get_agent_statuses(team_name: str) -> list[AgentStatus] | None:
    team_file_path = get_team_file_path(team_name)
    if not team_file_path.exists():
        return None
    data = json.loads(team_file_path.read_text(encoding="utf-8"))
    members = [
        TeamMember(
            agent_id=item["agentId"],
            name=item["name"],
            agent_type=item.get("agentType"),
        )
        for item in data.get("members", [])
    ]
    all_tasks = list_tasks(sanitize_name(team_name))
    unresolved_by_owner: dict[str, list[str]] = {}
    for task in all_tasks:
        if task.status != "completed" and task.owner:
            unresolved_by_owner.setdefault(task.owner, []).append(task.id)
    statuses: list[AgentStatus] = []
    for member in members:
        current_tasks = sorted(set(unresolved_by_owner.get(member.name, []) + unresolved_by_owner.get(member.agent_id, [])))
        statuses.append(
            AgentStatus(
                agent_id=member.agent_id,
                name=member.name,
                agent_type=member.agent_type,
                status="idle" if not current_tasks else "busy",
                current_tasks=current_tasks,
            )
        )
    return statuses


def unassign_teammate_tasks(team_name: str, teammate_id: str, teammate_name: str, reason: str) -> dict[str, Any]:
    unresolved = [
        task
        for task in list_tasks(sanitize_name(team_name))
        if task.status != "completed" and task.owner in {teammate_id, teammate_name}
    ]
    for task in unresolved:
        update_task(team_name, task.id, {"owner": None, "status": "pending"})
    action_verb = "was terminated" if reason == "terminated" else "has shut down"
    notification = f"{teammate_name} {action_verb}."
    if unresolved:
        task_list = ", ".join(f'#{task.id} "{task.subject}"' for task in unresolved)
        notification += (
            f" {len(unresolved)} task(s) were unassigned: {task_list}. "
            "Use TaskList to check availability and TaskUpdate with owner to reassign them to idle teammates."
        )
    return {
        "unassignedTasks": [{"id": task.id, "subject": task.subject} for task in unresolved],
        "notificationMessage": notification,
    }
