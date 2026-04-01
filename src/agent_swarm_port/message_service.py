"""Message routing service.

Ports the routing decisions from ``SendMessageTool``:

- local running background agents receive queued messages first
- otherwise messages go to teammate mailboxes (or broadcast)
- structured shutdown / plan-approval responses are encoded as protocol mail
"""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

from .constants import TEAM_LEAD_NAME
from .ids import generate_request_id
from .in_process_teammate import find_teammate_task_by_agent_id
from .local_agent import queue_pending_message
from .mailbox import (
    TeammateMessage,
    create_shutdown_approved_message,
    create_shutdown_rejected_message,
    create_shutdown_request_message,
    protocol_to_json,
    write_to_mailbox,
)
from .models import AppState, LocalAgentTaskState
from .runtime_state import AppStateStore
from .team_store import read_team_file

ResumeHandler = Callable[[str, str, AppStateStore], dict[str, Any] | None]


def _find_teammate_color(state: AppState, name: str) -> str | None:
    teammates = state.team_context.teammates if state.team_context else {}
    for teammate in teammates.values():
        if teammate.name == name:
            return teammate.color
    return None


def _resolve_sender_name(sender_name: str | None) -> str:
    return sender_name or TEAM_LEAD_NAME


def _resolve_team_name(state: AppState, team_name: str | None) -> str:
    resolved = team_name or (state.team_context.team_name if state.team_context else None)
    if not resolved:
        raise ValueError("Not in a team context and no team_name was provided")
    return resolved


def _find_running_local_agent(state: AppState, recipient: str) -> LocalAgentTaskState | None:
    alias_target = state.agent_name_registry.get(recipient)
    candidates = {recipient}
    if alias_target:
        candidates.add(alias_target)
    for task in state.tasks.values():
        if (
            isinstance(task, LocalAgentTaskState)
            and task.status == "running"
            and task.agent_id in candidates
        ):
            return task
    return None


def route_plain_message(
    *,
    store: AppStateStore,
    to: str,
    message: str,
    summary: str | None = None,
    team_name: str | None = None,
    sender_name: str | None = None,
    sender_color: str | None = None,
    resume_handler: ResumeHandler | None = None,
) -> dict[str, Any]:
    state = store.get_state()
    actual_sender_name = _resolve_sender_name(sender_name)
    actual_sender_color = sender_color

    if to != "*":
        running_local_agent = _find_running_local_agent(state, to)
        if running_local_agent is not None:
            store.set_state(lambda current: queue_pending_message(running_local_agent.id, message, current))
            return {
                "success": True,
                "message": f"Queued message for running local agent {running_local_agent.agent_id}",
                "routing": {
                    "sender": actual_sender_name,
                    "sender_color": actual_sender_color,
                    "target": running_local_agent.agent_id,
                    "summary": summary,
                    "content": message,
                },
            }

        if resume_handler is not None:
            resumed = resume_handler(to, message, store)
            if resumed:
                return {
                    "success": True,
                    "message": f"Resumed background local agent {to}",
                    "resume": resumed,
                }

    actual_team_name = _resolve_team_name(state, team_name)

    if to == "*":
        team_file = read_team_file(actual_team_name)
        if team_file is None:
            raise ValueError(f'Team "{actual_team_name}" does not exist')
        recipients = [
            member.name
            for member in team_file.members
            if member.name.lower() != actual_sender_name.lower()
        ]
        for recipient_name in recipients:
            write_to_mailbox(
                recipient_name,
                TeammateMessage(
                    from_=actual_sender_name,
                    text=message,
                    summary=summary,
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    color=actual_sender_color,
                ),
                actual_team_name,
            )
        return {
            "success": True,
            "message": (
                "No teammates to broadcast to (you are the only team member)"
                if not recipients
                else f"Message broadcast to {len(recipients)} teammate(s): {', '.join(recipients)}"
            ),
            "recipients": recipients,
            "routing": {
                "sender": actual_sender_name,
                "sender_color": actual_sender_color,
                "target": "@team",
                "summary": summary,
                "content": message,
            },
        }

    write_to_mailbox(
        to,
        TeammateMessage(
            from_=actual_sender_name,
            text=message,
            summary=summary,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            color=actual_sender_color,
        ),
        actual_team_name,
    )
    return {
        "success": True,
        "message": f"Message sent to {to}'s inbox",
        "routing": {
            "sender": actual_sender_name,
            "sender_color": actual_sender_color,
            "target": f"@{to}",
            "target_color": _find_teammate_color(state, to),
            "summary": summary,
            "content": message,
        },
    }


def send_shutdown_request(
    *,
    store: AppStateStore,
    target_name: str,
    reason: str | None = None,
    team_name: str | None = None,
    sender_name: str | None = None,
    sender_color: str | None = None,
) -> dict[str, Any]:
    state = store.get_state()
    actual_team_name = _resolve_team_name(state, team_name)
    actual_sender_name = _resolve_sender_name(sender_name)
    request_id = generate_request_id("shutdown", target_name)
    shutdown_message = create_shutdown_request_message(request_id, actual_sender_name, reason)
    write_to_mailbox(
        target_name,
        TeammateMessage(
            from_=actual_sender_name,
            text=protocol_to_json(shutdown_message),
            timestamp=shutdown_message.timestamp,
            color=sender_color,
        ),
        actual_team_name,
    )
    return {
        "success": True,
        "message": f"Shutdown request sent to {target_name}. Request ID: {request_id}",
        "request_id": request_id,
        "target": target_name,
    }


def send_shutdown_response(
    *,
    store: AppStateStore,
    request_id: str,
    approve: bool,
    team_name: str | None = None,
    sender_name: str | None = None,
    sender_color: str | None = None,
    pane_id: str | None = None,
    backend_type: str | None = None,
    reason: str | None = None,
    target_name: str = TEAM_LEAD_NAME,
) -> dict[str, Any]:
    state = store.get_state()
    actual_team_name = _resolve_team_name(state, team_name)
    actual_sender_name = _resolve_sender_name(sender_name)
    protocol = (
        create_shutdown_approved_message(request_id, actual_sender_name, pane_id, backend_type)
        if approve
        else create_shutdown_rejected_message(request_id, actual_sender_name, reason or "Rejected")
    )
    write_to_mailbox(
        target_name,
        TeammateMessage(
            from_=actual_sender_name,
            text=protocol_to_json(protocol),
            timestamp=protocol.timestamp,
            color=sender_color,
        ),
        actual_team_name,
    )

    if approve:
        task = find_teammate_task_by_agent_id(actual_sender_name, state.tasks)
        if task is None:
            for candidate in state.tasks.values():
                if getattr(getattr(candidate, "identity", None), "agent_name", None) == actual_sender_name:
                    task = candidate
                    break
        if task and task.abort_controller is not None:
            task.abort_controller.abort("shutdown-approved")

    return {
        "success": True,
        "message": (
            f"Shutdown approved for request {request_id}"
            if approve
            else f"Shutdown rejected for request {request_id}"
        ),
        "request_id": request_id,
    }
