"""Teammate runtime context based on ``contextvars``.

This is the Python analogue to the source's AsyncLocalStorage-based
``teammateContext.ts`` plus the higher-level identity helpers in
``teammate.ts``.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from .abort import AbortController


@dataclass(frozen=True)
class TeammateContext:
    agent_id: str
    agent_name: str
    team_name: str
    color: str | None
    plan_mode_required: bool
    parent_session_id: str
    is_in_process: bool
    abort_controller: AbortController


_teammate_context_var: ContextVar[TeammateContext | None] = ContextVar(
    "agent_swarm_teammate_context", default=None
)
_dynamic_team_context: TeammateContext | None = None


def create_teammate_context(
    *,
    agent_id: str,
    agent_name: str,
    team_name: str,
    color: str | None,
    plan_mode_required: bool,
    parent_session_id: str,
    abort_controller: AbortController,
) -> TeammateContext:
    return TeammateContext(
        agent_id=agent_id,
        agent_name=agent_name,
        team_name=team_name,
        color=color,
        plan_mode_required=plan_mode_required,
        parent_session_id=parent_session_id,
        is_in_process=True,
        abort_controller=abort_controller,
    )


def get_teammate_context() -> TeammateContext | None:
    return _teammate_context_var.get()


@contextmanager
def run_with_teammate_context(context: TeammateContext) -> Iterator[None]:
    token = _teammate_context_var.set(context)
    try:
        yield
    finally:
        _teammate_context_var.reset(token)


def is_in_process_teammate() -> bool:
    return _teammate_context_var.get() is not None


def set_dynamic_team_context(context: TeammateContext | None) -> None:
    global _dynamic_team_context
    _dynamic_team_context = context


def clear_dynamic_team_context() -> None:
    global _dynamic_team_context
    _dynamic_team_context = None


def get_dynamic_team_context() -> TeammateContext | None:
    return _dynamic_team_context


def _current_context() -> TeammateContext | None:
    return get_teammate_context() or _dynamic_team_context


def get_agent_id() -> str | None:
    ctx = _current_context()
    return ctx.agent_id if ctx else None


def get_agent_name() -> str | None:
    ctx = _current_context()
    return ctx.agent_name if ctx else None


def get_team_name(team_context: object | None = None) -> str | None:
    ctx = _current_context()
    if ctx:
        return ctx.team_name
    if team_context is None:
        return None
    return getattr(team_context, "team_name", None)


def get_teammate_color() -> str | None:
    ctx = _current_context()
    return ctx.color if ctx else None


def get_parent_session_id() -> str | None:
    ctx = _current_context()
    return ctx.parent_session_id if ctx else None


def is_teammate() -> bool:
    return _current_context() is not None


def is_plan_mode_required() -> bool:
    ctx = _current_context()
    return bool(ctx.plan_mode_required) if ctx else False
