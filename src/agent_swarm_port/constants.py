"""Shared constants mirrored from the TypeScript swarm/task subsystem."""

from __future__ import annotations

TEAM_LEAD_NAME = "team-lead"
TEAMMATE_MESSAGE_TAG = "teammate-message"
TEAMMATE_MESSAGES_UI_CAP = 50

TASK_ID_PREFIXES: dict[str, str] = {
    "local_bash": "b",
    "local_agent": "a",
    "remote_agent": "r",
    "in_process_teammate": "t",
    "local_workflow": "w",
    "monitor_mcp": "m",
    "dream": "d",
}
TASK_ID_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"

TASK_LIST_STATUSES = ("pending", "in_progress", "completed")
BACKGROUND_TASK_STATUSES = ("pending", "running", "completed", "failed", "killed")
TERMINAL_BACKGROUND_TASK_STATUSES = {"completed", "failed", "killed"}

DEFAULT_TASKS_MODE_TASK_LIST_ID = "tasklist"
POLL_INTERVAL_MS = 1000
WAIT_FOR_PROMPT_POLL_MS = 500
STOPPED_DISPLAY_MS = 3000
PANEL_GRACE_MS = 30000

IDLE_REASON_AVAILABLE = "available"
IDLE_REASON_INTERRUPTED = "interrupted"
IDLE_REASON_FAILED = "failed"
