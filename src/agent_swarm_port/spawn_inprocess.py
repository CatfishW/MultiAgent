"""Spawn/kill helpers for in-process teammates."""

from __future__ import annotations

import random
import time
from dataclasses import replace

from .abort import create_abort_controller
from .contexts import create_teammate_context
from .ids import format_agent_id
from .models import AppState, InProcessTeammateTaskState, TeammateIdentity
from .task_framework import create_task_state_base, generate_task_id

_SPINNER_VERBS = ["thinking", "planning", "checking", "coordinating", "reviewing"]
_PAST_TENSE_VERBS = ["finished", "checked", "reviewed", "planned", "completed"]


def spawn_in_process_teammate(
    state: AppState,
    *,
    name: str,
    team_name: str,
    prompt: str,
    parent_session_id: str,
    color: str | None = None,
    plan_mode_required: bool = False,
    model: str | None = None,
    tool_use_id: str | None = None,
) -> tuple[AppState, dict]:
    agent_id = format_agent_id(name, team_name)
    task_id = generate_task_id("in_process_teammate")
    abort_controller = create_abort_controller()
    identity = TeammateIdentity(
        agent_id=agent_id,
        agent_name=name,
        team_name=team_name,
        color=color,
        plan_mode_required=plan_mode_required,
        parent_session_id=parent_session_id,
    )
    teammate_context = create_teammate_context(
        agent_id=agent_id,
        agent_name=name,
        team_name=team_name,
        color=color,
        plan_mode_required=plan_mode_required,
        parent_session_id=parent_session_id,
        abort_controller=abort_controller,
    )
    description = f"{name}: {prompt[:50]}{'...' if len(prompt) > 50 else ''}"
    base = create_task_state_base(
        task_id=task_id,
        task_type="in_process_teammate",
        description=description,
        output_file=f"{task_id}.output",
        tool_use_id=tool_use_id,
    )
    task = InProcessTeammateTaskState(
        **{**base.__dict__, "status": "running"},
        identity=identity,
        prompt=prompt,
        model=model,
        abort_controller=abort_controller,
        awaiting_plan_approval=False,
        spinner_verb=random.choice(_SPINNER_VERBS),
        past_tense_verb=random.choice(_PAST_TENSE_VERBS),
        permission_mode="plan" if plan_mode_required else "default",
        is_idle=False,
        shutdown_requested=False,
        pending_user_messages=[],
        messages=[],
    )
    next_tasks = dict(state.tasks)
    next_tasks[task_id] = task
    return replace(state, tasks=next_tasks), {
        "success": True,
        "agent_id": agent_id,
        "task_id": task_id,
        "abort_controller": abort_controller,
        "teammate_context": teammate_context,
    }


def kill_in_process_teammate(task_id: str, state: AppState) -> AppState:
    task = state.tasks.get(task_id)
    if not isinstance(task, InProcessTeammateTaskState) or task.status != "running":
        return state
    if task.abort_controller is not None:
        task.abort_controller.abort("killed")
    next_tasks = dict(state.tasks)
    next_tasks[task_id] = replace(task, status="killed", notified=True, end_time=int(time.time() * 1000), abort_controller=None, is_idle=True, current_work_abort_controller=None)
    return replace(state, tasks=next_tasks)
