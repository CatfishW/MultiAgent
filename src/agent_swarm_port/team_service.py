"""High-level team create/delete services.

This ports the core behavior of ``TeamCreateTool`` and ``TeamDeleteTool`` into
plain Python service functions.
"""

from __future__ import annotations

from dataclasses import replace

from .constants import TEAM_LEAD_NAME
from .ids import format_agent_id
from .models import AppState
from .paths import get_team_file_path
from .team_store import (
    cleanup_team_directories,
    clear_team_storage_name_binding,
    create_team_context,
    create_team_file,
    generate_unique_team_name,
    initialize_team_storage,
    read_team_file,
    register_team_for_session_cleanup,
    unregister_team_for_session_cleanup,
    write_team_file,
)


def create_team(
    state: AppState,
    *,
    team_name: str,
    session_id: str,
    cwd: str,
    description: str | None = None,
    agent_type: str | None = None,
    lead_model: str | None = None,
) -> tuple[AppState, dict]:
    """Create a new team and bind the leader's task-list context to it."""
    if state.team_context is not None:
        raise ValueError(
            f'Already leading team "{state.team_context.team_name}". Delete it before creating a new one.'
        )

    final_team_name = generate_unique_team_name(team_name)
    lead_agent_id = format_agent_id(TEAM_LEAD_NAME, final_team_name)
    lead_agent_type = agent_type or TEAM_LEAD_NAME

    team_file = create_team_file(
        team_name=final_team_name,
        description=description,
        lead_agent_id=lead_agent_id,
        lead_session_id=session_id,
        lead_agent_type=lead_agent_type,
        lead_model=lead_model,
        cwd=cwd,
    )
    write_team_file(final_team_name, team_file)
    register_team_for_session_cleanup(final_team_name)
    initialize_team_storage(final_team_name)

    team_file_path = str(get_team_file_path(final_team_name))
    next_state = replace(
        state,
        team_context=create_team_context(
            final_team_name,
            lead_agent_id,
            lead_agent_type,
            cwd,
            team_file_path,
        ),
    )
    return next_state, {
        "team_name": final_team_name,
        "team_file_path": team_file_path,
        "lead_agent_id": lead_agent_id,
    }


def delete_team(state: AppState) -> tuple[AppState, dict]:
    """Delete the leader's current team if no active non-lead members remain."""
    team_name = state.team_context.team_name if state.team_context else None
    if team_name:
        team_file = read_team_file(team_name)
        if team_file is not None:
            non_lead_members = [
                member for member in team_file.members if member.name != TEAM_LEAD_NAME
            ]
            active_members = [
                member for member in non_lead_members if member.is_active is not False
            ]
            if active_members:
                member_names = ", ".join(member.name for member in active_members)
                return state, {
                    "success": False,
                    "message": (
                        f"Cannot cleanup team with {len(active_members)} active member(s): "
                        f"{member_names}. Use requestShutdown first."
                    ),
                    "team_name": team_name,
                }

        cleanup_team_directories(team_name)
        unregister_team_for_session_cleanup(team_name)
        clear_team_storage_name_binding()

    next_state = replace(state, team_context=None, inbox_messages=[])
    return next_state, {
        "success": True,
        "message": (
            f'Cleaned up directories and worktrees for team "{team_name}"'
            if team_name
            else "No team name found, nothing to clean up"
        ),
        "team_name": team_name,
    }
