"""Team file persistence and cleanup helpers.

This mirrors the data model and basic mutation helpers from
``src/utils/swarm/teamHelpers.ts``.
"""

from __future__ import annotations

import json
import random
import shutil
import string
from dataclasses import asdict
from pathlib import Path
from time import time

from .constants import TEAM_LEAD_NAME
from .models import TeamContext, TeamContextMember, TeamFile, TeamMember
from .paths import get_task_path, get_tasks_dir, get_team_dir, get_team_file_path, sanitize_agent_name, sanitize_name
from .task_list import clear_leader_team_name, reset_task_list, set_leader_team_name

_session_cleanup_registry: set[str] = set()


def read_team_file(team_name: str) -> TeamFile | None:
    path = get_team_file_path(team_name)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return _team_file_from_dict(data)


def write_team_file(team_name: str, team_file: TeamFile) -> None:
    path = get_team_file_path(team_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_team_file_to_dict(team_file), indent=2), encoding="utf-8")


def remove_teammate_from_team_file(team_name: str, *, agent_id: str | None = None, name: str | None = None) -> bool:
    team_file = read_team_file(team_name)
    if team_file is None:
        return False
    original_len = len(team_file.members)
    team_file.members = [member for member in team_file.members if not ((agent_id and member.agent_id == agent_id) or (name and member.name == name))]
    if len(team_file.members) == original_len:
        return False
    write_team_file(team_name, team_file)
    return True


def remove_member_by_agent_id(team_name: str, agent_id: str) -> bool:
    return remove_teammate_from_team_file(team_name, agent_id=agent_id)


def set_member_mode(team_name: str, agent_id: str, mode: str) -> bool:
    team_file = read_team_file(team_name)
    if team_file is None:
        return False
    changed = False
    for member in team_file.members:
        if member.agent_id == agent_id:
            member.mode = mode
            changed = True
            break
    if changed:
        write_team_file(team_name, team_file)
    return changed


def set_member_active(team_name: str, agent_id: str, is_active: bool) -> bool:
    team_file = read_team_file(team_name)
    if team_file is None:
        return False
    changed = False
    for member in team_file.members:
        if member.agent_id == agent_id:
            member.is_active = is_active
            changed = True
            break
    if changed:
        write_team_file(team_name, team_file)
    return changed


def add_hidden_pane_id(team_name: str, pane_id: str) -> bool:
    team_file = read_team_file(team_name)
    if team_file is None:
        return False
    if pane_id not in team_file.hidden_pane_ids:
        team_file.hidden_pane_ids.append(pane_id)
        write_team_file(team_name, team_file)
    return True


def remove_hidden_pane_id(team_name: str, pane_id: str) -> bool:
    team_file = read_team_file(team_name)
    if team_file is None:
        return False
    if pane_id in team_file.hidden_pane_ids:
        team_file.hidden_pane_ids.remove(pane_id)
        write_team_file(team_name, team_file)
    return True


def cleanup_team_directories(team_name: str) -> None:
    shutil.rmtree(get_team_dir(team_name), ignore_errors=True)
    shutil.rmtree(get_tasks_dir(sanitize_name(team_name)), ignore_errors=True)


def register_team_for_session_cleanup(team_name: str) -> None:
    _session_cleanup_registry.add(team_name)


def unregister_team_for_session_cleanup(team_name: str) -> None:
    _session_cleanup_registry.discard(team_name)


def cleanup_session_teams() -> None:
    for team_name in list(_session_cleanup_registry):
        cleanup_team_directories(team_name)
        _session_cleanup_registry.discard(team_name)


def generate_word_slug() -> str:
    alphabet = string.ascii_lowercase
    return "".join(random.choice(alphabet) for _ in range(8))


def generate_unique_team_name(provided_name: str) -> str:
    if read_team_file(provided_name) is None:
        return provided_name
    return generate_word_slug()


def create_team_context(team_name: str, lead_agent_id: str, lead_agent_type: str, cwd: str, team_file_path: str) -> TeamContext:
    now_ms = int(time() * 1000)
    return TeamContext(
        team_name=team_name,
        team_file_path=team_file_path,
        lead_agent_id=lead_agent_id,
        teammates={
            lead_agent_id: TeamContextMember(
                name=TEAM_LEAD_NAME,
                agent_type=lead_agent_type,
                color=None,
                tmux_session_name="",
                tmux_pane_id="",
                cwd=cwd,
                spawned_at=now_ms,
            )
        },
    )


def create_team_file(*, team_name: str, description: str | None, lead_agent_id: str, lead_session_id: str, lead_agent_type: str, lead_model: str | None, cwd: str) -> TeamFile:
    now_ms = int(time() * 1000)
    return TeamFile(
        name=team_name,
        description=description,
        created_at=now_ms,
        lead_agent_id=lead_agent_id,
        lead_session_id=lead_session_id,
        members=[
            TeamMember(
                agent_id=lead_agent_id,
                name=TEAM_LEAD_NAME,
                agent_type=lead_agent_type,
                model=lead_model,
                joined_at=now_ms,
                tmux_pane_id="",
                cwd=cwd,
                subscriptions=[],
            )
        ],
    )


def initialize_team_storage(team_name: str) -> None:
    task_list_id = sanitize_name(team_name)
    reset_task_list(task_list_id)
    get_tasks_dir(task_list_id).mkdir(parents=True, exist_ok=True)
    set_leader_team_name(task_list_id)


def clear_team_storage_name_binding() -> None:
    clear_leader_team_name()


def _team_file_to_dict(team_file: TeamFile) -> dict:
    payload = asdict(team_file)
    payload["createdAt"] = payload.pop("created_at")
    payload["leadAgentId"] = payload.pop("lead_agent_id")
    payload["leadSessionId"] = payload.pop("lead_session_id")
    payload["hiddenPaneIds"] = payload.pop("hidden_pane_ids")
    payload["teamAllowedPaths"] = payload.pop("team_allowed_paths")
    for member in payload["members"]:
        member["agentId"] = member.pop("agent_id")
        member["agentType"] = member.pop("agent_type")
        member["joinedAt"] = member.pop("joined_at")
        member["tmuxPaneId"] = member.pop("tmux_pane_id")
        member["worktreePath"] = member.pop("worktree_path")
        member["sessionId"] = member.pop("session_id")
        member["backendType"] = member.pop("backend_type")
        member["isActive"] = member.pop("is_active")
        member["planModeRequired"] = member.pop("plan_mode_required")
    return payload


def _team_file_from_dict(data: dict) -> TeamFile:
    return TeamFile(
        name=data["name"],
        description=data.get("description"),
        created_at=data.get("createdAt") or data.get("created_at") or 0,
        lead_agent_id=data.get("leadAgentId") or data.get("lead_agent_id"),
        lead_session_id=data.get("leadSessionId") or data.get("lead_session_id"),
        hidden_pane_ids=list(data.get("hiddenPaneIds") or data.get("hidden_pane_ids") or []),
        team_allowed_paths=[],
        members=[
            TeamMember(
                agent_id=item.get("agentId") or item.get("agent_id"),
                name=item["name"],
                agent_type=item.get("agentType") or item.get("agent_type"),
                model=item.get("model"),
                prompt=item.get("prompt"),
                color=item.get("color"),
                plan_mode_required=(
                    item["planModeRequired"]
                    if "planModeRequired" in item
                    else item.get("plan_mode_required")
                ),
                joined_at=item.get("joinedAt") or item.get("joined_at") or 0,
                tmux_pane_id=item.get("tmuxPaneId") or item.get("tmux_pane_id", ""),
                cwd=item.get("cwd", ""),
                worktree_path=item.get("worktreePath") or item.get("worktree_path"),
                session_id=item.get("sessionId") or item.get("session_id"),
                subscriptions=list(item.get("subscriptions", [])),
                backend_type=item.get("backendType") or item.get("backend_type"),
                is_active=item.get("isActive") if "isActive" in item else item.get("is_active"),
                mode=item.get("mode"),
            )
            for item in data.get("members", [])
        ],
    )
