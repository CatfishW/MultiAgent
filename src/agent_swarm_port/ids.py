"""Deterministic agent/request ID helpers.

Mirrors `src/utils/agentId.ts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import time_ns


@dataclass(frozen=True)
class ParsedAgentId:
    agent_name: str
    team_name: str


@dataclass(frozen=True)
class ParsedRequestId:
    request_type: str
    timestamp: int
    agent_id: str


def format_agent_id(agent_name: str, team_name: str) -> str:
    return f"{agent_name}@{team_name}"


def parse_agent_id(agent_id: str) -> ParsedAgentId | None:
    at_index = agent_id.find("@")
    if at_index == -1:
        return None
    return ParsedAgentId(agent_name=agent_id[:at_index], team_name=agent_id[at_index + 1 :])


def generate_request_id(request_type: str, agent_id: str) -> str:
    timestamp_ms = time_ns() // 1_000_000
    return f"{request_type}-{timestamp_ms}@{agent_id}"


def parse_request_id(request_id: str) -> ParsedRequestId | None:
    at_index = request_id.find("@")
    if at_index == -1:
        return None
    prefix = request_id[:at_index]
    agent_id = request_id[at_index + 1 :]
    dash_index = prefix.rfind("-")
    if dash_index == -1:
        return None
    request_type = prefix[:dash_index]
    timestamp_str = prefix[dash_index + 1 :]
    if not timestamp_str.isdigit():
        return None
    return ParsedRequestId(
        request_type=request_type,
        timestamp=int(timestamp_str),
        agent_id=agent_id,
    )
