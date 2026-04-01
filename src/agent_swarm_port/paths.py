"""Filesystem paths and sanitizers for teams, inboxes, and task lists."""

from __future__ import annotations

import os
from pathlib import Path


def get_home_root() -> Path:
    return Path(os.environ.get("AGENT_SWARM_PORT_HOME", Path.home() / ".agent_swarm_port"))


def sanitize_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in name)


def sanitize_agent_name(name: str) -> str:
    return name.replace("@", "-")


def sanitize_path_component(component: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in component)


def get_teams_dir() -> Path:
    return get_home_root() / "teams"


def get_team_dir(team_name: str) -> Path:
    return get_teams_dir() / sanitize_name(team_name)


def get_team_file_path(team_name: str) -> Path:
    return get_team_dir(team_name) / "config.json"


def get_inbox_dir(team_name: str) -> Path:
    return get_team_dir(team_name) / "inboxes"


def get_inbox_path(agent_name: str, team_name: str) -> Path:
    return get_inbox_dir(team_name) / f"{sanitize_path_component(agent_name)}.json"


def get_tasks_dir(task_list_id: str) -> Path:
    return get_home_root() / "tasks" / sanitize_path_component(task_list_id)


def get_task_path(task_list_id: str, task_id: str) -> Path:
    return get_tasks_dir(task_list_id) / f"{sanitize_path_component(task_id)}.json"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
