"""Mailbox primitives and file-backed teammate message protocols.

This module combines the behavior from the source's ``mailbox.ts`` and
``teammateMailbox.ts``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from .constants import TEAMMATE_MESSAGE_TAG
from .ids import generate_request_id
from .locks import file_lock
from .paths import get_inbox_path


MessageSource = Literal["user", "teammate", "system", "tick", "task"]


@dataclass
class RuntimeMessage:
    id: str
    source: MessageSource
    content: str
    from_: str | None = None
    color: str | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class Mailbox:
    def __init__(self) -> None:
        self._queue: list[RuntimeMessage] = []
        self._waiters: list[tuple[Callable[[RuntimeMessage], bool], asyncio.Future[RuntimeMessage]]] = []
        self._revision = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def length(self) -> int:
        return len(self._queue)

    def send(self, message: RuntimeMessage) -> None:
        self._revision += 1
        for index, (predicate, future) in enumerate(list(self._waiters)):
            if predicate(message):
                self._waiters.pop(index)
                if not future.done():
                    future.set_result(message)
                return
        self._queue.append(message)

    def poll(self, predicate: Callable[[RuntimeMessage], bool] | None = None) -> RuntimeMessage | None:
        predicate = predicate or (lambda _msg: True)
        for index, message in enumerate(self._queue):
            if predicate(message):
                return self._queue.pop(index)
        return None

    async def receive(self, predicate: Callable[[RuntimeMessage], bool] | None = None) -> RuntimeMessage:
        predicate = predicate or (lambda _msg: True)
        existing = self.poll(predicate)
        if existing is not None:
            return existing
        fut: asyncio.Future[RuntimeMessage] = asyncio.get_running_loop().create_future()
        self._waiters.append((predicate, fut))
        return await fut


@dataclass
class TeammateMessage:
    from_: str
    text: str
    timestamp: str
    read: bool = False
    color: str | None = None
    summary: str | None = None


@dataclass
class IdleNotificationMessage:
    type: Literal["idle_notification"] = "idle_notification"
    from_: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    idle_reason: Literal["available", "interrupted", "failed"] | None = None
    summary: str | None = None
    completed_task_id: str | None = None
    completed_status: Literal["resolved", "blocked", "failed"] | None = None
    failure_reason: str | None = None


@dataclass
class ShutdownRequestMessage:
    type: Literal["shutdown_request"] = "shutdown_request"
    request_id: str = ""
    from_: str = ""
    reason: str | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class ShutdownResponseMessage:
    type: Literal["shutdown_response"] = "shutdown_response"
    request_id: str = ""
    approved: bool = False
    from_: str = ""
    reason: str | None = None
    pane_id: str | None = None
    backend_type: str | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class PlanApprovalResponseMessage:
    type: Literal["plan_approval_response"] = "plan_approval_response"
    request_id: str = ""
    approved: bool = False
    feedback: str | None = None
    permission_mode: str | None = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class PermissionRequestMessage:
    type: Literal["permission_request"] = "permission_request"
    request_id: str = ""
    agent_id: str = ""
    tool_name: str = ""
    tool_use_id: str = ""
    description: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    permission_suggestions: list[Any] = field(default_factory=list)


@dataclass
class PermissionResponseMessage:
    type: Literal["permission_response"] = "permission_response"
    request_id: str = ""
    subtype: Literal["success", "error"] = "success"
    error: str | None = None
    response: dict[str, Any] | None = None


@dataclass
class TaskAssignmentMessage:
    type: Literal["task_assignment"] = "task_assignment"
    task_id: str = ""
    subject: str = ""
    description: str = ""
    assigned_by: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def _ensure_inbox(team_name: str, agent_name: str) -> Path:
    path = get_inbox_path(agent_name, team_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
    return path


def _read_inbox(path: Path) -> list[TeammateMessage]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    return [TeammateMessage(from_=item.get("from") or item.get("from_"), text=item["text"], timestamp=item["timestamp"], read=item.get("read", False), color=item.get("color"), summary=item.get("summary")) for item in raw]


def read_mailbox(agent_name: str, team_name: str) -> list[TeammateMessage]:
    path = get_inbox_path(agent_name, team_name)
    return _read_inbox(path)


def read_unread_messages(agent_name: str, team_name: str) -> list[TeammateMessage]:
    return [message for message in read_mailbox(agent_name, team_name) if not message.read]


def write_to_mailbox(recipient_name: str, message: TeammateMessage, team_name: str) -> None:
    path = _ensure_inbox(team_name, recipient_name)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path, retries=10):
        messages = _read_inbox(path)
        messages.append(message)
        path.write_text(json.dumps([_serialize_teammate_message(item) for item in messages], indent=2), encoding="utf-8")


def mark_message_as_read_by_index(agent_name: str, team_name: str, message_index: int) -> None:
    path = _ensure_inbox(team_name, agent_name)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path, retries=10):
        messages = _read_inbox(path)
        if 0 <= message_index < len(messages):
            messages[message_index].read = True
            path.write_text(json.dumps([_serialize_teammate_message(item) for item in messages], indent=2), encoding="utf-8")


def mark_messages_as_read(agent_name: str, team_name: str) -> None:
    path = _ensure_inbox(team_name, agent_name)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with file_lock(lock_path, retries=10):
        messages = _read_inbox(path)
        for message in messages:
            message.read = True
        path.write_text(json.dumps([_serialize_teammate_message(item) for item in messages], indent=2), encoding="utf-8")


def clear_mailbox(agent_name: str, team_name: str) -> None:
    path = _ensure_inbox(team_name, agent_name)
    path.write_text("[]", encoding="utf-8")


def format_teammate_messages(messages: list[TeammateMessage]) -> str:
    rendered: list[str] = []
    for message in messages:
        color_attr = f' color="{message.color}"' if message.color else ""
        summary_attr = f' summary="{message.summary}"' if message.summary else ""
        rendered.append(
            f"<{TEAMMATE_MESSAGE_TAG} teammate_id=\"{message.from_}\"{color_attr}{summary_attr}>\n{message.text}\n</{TEAMMATE_MESSAGE_TAG}>"
        )
    return "\n\n".join(rendered)


def create_idle_notification(agent_id: str, **kwargs: Any) -> IdleNotificationMessage:
    return IdleNotificationMessage(from_=agent_id, **kwargs)


def create_shutdown_request_message(request_id: str, from_: str, reason: str | None = None) -> ShutdownRequestMessage:
    return ShutdownRequestMessage(request_id=request_id, from_=from_, reason=reason)


def create_shutdown_approved_message(request_id: str, from_: str, pane_id: str | None = None, backend_type: str | None = None) -> ShutdownResponseMessage:
    return ShutdownResponseMessage(request_id=request_id, approved=True, from_=from_, pane_id=pane_id, backend_type=backend_type)


def create_shutdown_rejected_message(request_id: str, from_: str, reason: str) -> ShutdownResponseMessage:
    return ShutdownResponseMessage(request_id=request_id, approved=False, from_=from_, reason=reason)


def create_permission_request_message(*, request_id: str, agent_id: str, tool_name: str, tool_use_id: str, description: str, input: dict[str, Any], permission_suggestions: list[Any] | None = None) -> PermissionRequestMessage:
    return PermissionRequestMessage(
        request_id=request_id,
        agent_id=agent_id,
        tool_name=tool_name,
        tool_use_id=tool_use_id,
        description=description,
        input=input,
        permission_suggestions=permission_suggestions or [],
    )


def create_permission_response_message(*, request_id: str, subtype: Literal["success", "error"], error: str | None = None, updated_input: dict[str, Any] | None = None, permission_updates: list[Any] | None = None) -> PermissionResponseMessage:
    if subtype == "error":
        return PermissionResponseMessage(request_id=request_id, subtype="error", error=error or "Permission denied")
    return PermissionResponseMessage(
        request_id=request_id,
        subtype="success",
        response={
            "updated_input": updated_input,
            "permission_updates": permission_updates or [],
        },
    )


def create_task_assignment_message(task_id: str, subject: str, description: str, assigned_by: str) -> TaskAssignmentMessage:
    return TaskAssignmentMessage(task_id=task_id, subject=subject, description=description, assigned_by=assigned_by)


def maybe_parse_protocol_message(message_text: str) -> dict[str, Any] | None:
    try:
        parsed = json.loads(message_text)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict) and isinstance(parsed.get("type"), str):
        return parsed
    return None


def is_idle_notification(message_text: str) -> IdleNotificationMessage | None:
    parsed = maybe_parse_protocol_message(message_text)
    if not parsed or parsed.get("type") != "idle_notification":
        return None
    return IdleNotificationMessage(
        from_=parsed.get("from") or parsed.get("from_", ""),
        timestamp=parsed.get("timestamp", ""),
        idle_reason=parsed.get("idleReason") or parsed.get("idle_reason"),
        summary=parsed.get("summary"),
        completed_task_id=parsed.get("completedTaskId") or parsed.get("completed_task_id"),
        completed_status=parsed.get("completedStatus") or parsed.get("completed_status"),
        failure_reason=parsed.get("failureReason") or parsed.get("failure_reason"),
    )


def is_shutdown_request(message_text: str) -> ShutdownRequestMessage | None:
    parsed = maybe_parse_protocol_message(message_text)
    if not parsed or parsed.get("type") != "shutdown_request":
        return None
    return ShutdownRequestMessage(
        request_id=parsed.get("requestId") or parsed.get("request_id", ""),
        from_=parsed.get("from", ""),
        reason=parsed.get("reason"),
        timestamp=parsed.get("timestamp", ""),
    )


def is_shutdown_response(message_text: str) -> ShutdownResponseMessage | None:
    parsed = maybe_parse_protocol_message(message_text)
    if not parsed or parsed.get("type") != "shutdown_response":
        return None
    return ShutdownResponseMessage(
        request_id=parsed.get("requestId") or parsed.get("request_id", ""),
        approved=bool(parsed.get("approved") if "approved" in parsed else parsed.get("approve")),
        from_=parsed.get("from", ""),
        reason=parsed.get("reason"),
        pane_id=parsed.get("paneId") or parsed.get("pane_id"),
        backend_type=parsed.get("backendType") or parsed.get("backend_type"),
        timestamp=parsed.get("timestamp", ""),
    )


def is_plan_approval_response(message_text: str) -> PlanApprovalResponseMessage | None:
    parsed = maybe_parse_protocol_message(message_text)
    if not parsed or parsed.get("type") != "plan_approval_response":
        return None
    return PlanApprovalResponseMessage(
        request_id=parsed.get("requestId") or parsed.get("request_id", ""),
        approved=bool(parsed.get("approved") if "approved" in parsed else parsed.get("approve")),
        feedback=parsed.get("feedback"),
        permission_mode=parsed.get("permissionMode") or parsed.get("permission_mode"),
        timestamp=parsed.get("timestamp", ""),
    )


def is_permission_request(message_text: str) -> PermissionRequestMessage | None:
    parsed = maybe_parse_protocol_message(message_text)
    if not parsed or parsed.get("type") != "permission_request":
        return None
    return PermissionRequestMessage(
        request_id=parsed.get("request_id", ""),
        agent_id=parsed.get("agent_id", ""),
        tool_name=parsed.get("tool_name", ""),
        tool_use_id=parsed.get("tool_use_id", ""),
        description=parsed.get("description", ""),
        input=parsed.get("input", {}),
        permission_suggestions=parsed.get("permission_suggestions", []),
    )


def is_permission_response(message_text: str) -> PermissionResponseMessage | None:
    parsed = maybe_parse_protocol_message(message_text)
    if not parsed or parsed.get("type") != "permission_response":
        return None
    return PermissionResponseMessage(
        request_id=parsed.get("request_id", ""),
        subtype=parsed.get("subtype", "success"),
        error=parsed.get("error"),
        response=parsed.get("response"),
    )


def generate_shutdown_request_id(target_name: str) -> str:
    return generate_request_id("shutdown", target_name)


def _serialize_teammate_message(message: TeammateMessage) -> dict[str, Any]:
    return {
        "from": message.from_,
        "text": message.text,
        "timestamp": message.timestamp,
        "read": message.read,
        "color": message.color,
        "summary": message.summary,
    }


def protocol_to_json(message: Any) -> str:
    payload = asdict(message)
    if "from_" in payload:
        payload["from"] = payload.pop("from_")
    return json.dumps(payload)
