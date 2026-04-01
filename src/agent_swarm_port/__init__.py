"""Focused Python port of the uploaded swarm/team coordination subsystem.

This package ports the source code's agent-team, mailbox, task-list,
in-process teammate orchestration, and coordinator-related mechanics into a
Python-first structure while preserving the original logic as closely as
possible.
"""

from .constants import TEAM_LEAD_NAME
from .ids import format_agent_id, generate_request_id, parse_agent_id, parse_request_id
from .models import AppState
from .runtime_state import AppStateStore
from .team_service import create_team, delete_team

__all__ = [
    "TEAM_LEAD_NAME",
    "format_agent_id",
    "generate_request_id",
    "parse_agent_id",
    "parse_request_id",
    "AppState",
    "AppStateStore",
    "create_team",
    "delete_team",
]
