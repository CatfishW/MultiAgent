"""Dataclass models for the focused Python swarm port."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

BackgroundTaskType = Literal[
    "local_bash",
    "local_agent",
    "remote_agent",
    "in_process_teammate",
    "local_workflow",
    "monitor_mcp",
    "dream",
]
BackgroundTaskStatus = Literal["pending", "running", "completed", "failed", "killed"]
TaskListStatus = Literal["pending", "in_progress", "completed"]
PermissionMode = Literal["default", "plan", "acceptEdits", "bypassPermissions"]
BackendType = Literal["tmux", "iterm2", "in-process", "remote", "unknown"]


@dataclass(kw_only=True)
class TaskStateBase:
    id: str
    type: BackgroundTaskType
    status: BackgroundTaskStatus
    description: str
    start_time: int
    output_file: str
    output_offset: int = 0
    notified: bool = False
    tool_use_id: str | None = None
    end_time: int | None = None
    total_paused_ms: int | None = None


@dataclass(kw_only=True)
class AgentProgress:
    tool_count: int = 0
    token_count: int = 0
    summary: str | None = None
    recent_activities: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class TeammateIdentity:
    agent_id: str
    agent_name: str
    team_name: str
    color: str | None
    plan_mode_required: bool
    parent_session_id: str


@dataclass
class InProcessTeammateTaskState(TaskStateBase):
    type: Literal["in_process_teammate"] = "in_process_teammate"
    identity: TeammateIdentity = field(default_factory=lambda: TeammateIdentity("", "", "", None, False, ""))
    prompt: str = ""
    model: str | None = None
    selected_agent: dict[str, Any] | None = None
    abort_controller: Any | None = None
    current_work_abort_controller: Any | None = None
    unregister_cleanup: Any | None = None
    awaiting_plan_approval: bool = False
    permission_mode: PermissionMode = "default"
    error: str | None = None
    result: dict[str, Any] | None = None
    progress: AgentProgress | None = None
    messages: list[dict[str, Any]] | None = field(default_factory=list)
    in_progress_tool_use_ids: set[str] | None = None
    pending_user_messages: list[str] = field(default_factory=list)
    spinner_verb: str | None = None
    past_tense_verb: str | None = None
    is_idle: bool = False
    shutdown_requested: bool = False
    on_idle_callbacks: list[Any] = field(default_factory=list)
    last_reported_tool_count: int = 0
    last_reported_token_count: int = 0


@dataclass
class LocalAgentTaskState(TaskStateBase):
    type: Literal["local_agent"] = "local_agent"
    agent_id: str = ""
    prompt: str = ""
    selected_agent: dict[str, Any] | None = None
    agent_type: str = "general-purpose"
    model: str | None = None
    abort_controller: Any | None = None
    unregister_cleanup: Any | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    progress: AgentProgress | None = None
    retrieved: bool = False
    messages: list[dict[str, Any]] | None = field(default_factory=list)
    last_reported_tool_count: int = 0
    last_reported_token_count: int = 0
    is_backgrounded: bool = True
    pending_messages: list[str] = field(default_factory=list)
    retain: bool = False
    disk_loaded: bool = False
    evict_after: int | None = None


@dataclass
class RemoteAgentTaskState(TaskStateBase):
    type: Literal["remote_agent"] = "remote_agent"
    agent_id: str = ""
    prompt: str = ""
    session_url: str | None = None
    error: str | None = None


@dataclass(kw_only=True)
class TeamAllowedPath:
    path: str
    tool_name: str
    added_by: str
    added_at: int


@dataclass(kw_only=True)
class TeamMember:
    agent_id: str
    name: str
    agent_type: str | None = None
    model: str | None = None
    prompt: str | None = None
    color: str | None = None
    plan_mode_required: bool | None = None
    joined_at: int = 0
    tmux_pane_id: str = ""
    cwd: str = ""
    worktree_path: str | None = None
    session_id: str | None = None
    subscriptions: list[str] = field(default_factory=list)
    backend_type: BackendType | None = None
    is_active: bool | None = None
    mode: PermissionMode | None = None


@dataclass(kw_only=True)
class TeamFile:
    name: str
    created_at: int
    lead_agent_id: str
    description: str | None = None
    lead_session_id: str | None = None
    hidden_pane_ids: list[str] = field(default_factory=list)
    team_allowed_paths: list[TeamAllowedPath] = field(default_factory=list)
    members: list[TeamMember] = field(default_factory=list)


@dataclass(kw_only=True)
class TeamContextMember:
    name: str
    agent_type: str | None = None
    color: str | None = None
    tmux_session_name: str = ""
    tmux_pane_id: str = ""
    cwd: str = ""
    spawned_at: int = 0


@dataclass(kw_only=True)
class TeamContext:
    team_name: str
    team_file_path: str
    lead_agent_id: str
    teammates: dict[str, TeamContextMember] = field(default_factory=dict)


@dataclass(kw_only=True)
class ToolPermissionContext:
    mode: PermissionMode = "default"
    additional_working_directories: dict[str, bool] = field(default_factory=dict)


@dataclass(kw_only=True)
class AppState:
    tasks: dict[str, TaskStateBase] = field(default_factory=dict)
    team_context: TeamContext | None = None
    inbox_messages: list[dict[str, Any]] = field(default_factory=list)
    agent_name_registry: dict[str, str] = field(default_factory=dict)
    tool_permission_context: ToolPermissionContext = field(default_factory=ToolPermissionContext)
    expanded_view: str = "none"
    main_loop_model: str | None = None
    main_loop_model_for_session: str | None = None
    current_cwd: str = "."
    viewing_agent_task_id: str | None = None
    foregrounded_task_id: str | None = None


@dataclass(kw_only=True)
class TaskListTask:
    id: str
    subject: str
    description: str
    status: TaskListStatus
    active_form: str | None = None
    owner: str | None = None
    blocks: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    metadata: dict[str, Any] | None = None


@dataclass(kw_only=True)
class AgentStatus:
    agent_id: str
    name: str
    agent_type: str | None
    status: Literal["idle", "busy"]
    current_tasks: list[str]
