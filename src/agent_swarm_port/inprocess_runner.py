"""Continuous in-process teammate runner.

This module ports the coordination heart of
``src/utils/swarm/inProcessRunner.ts``:

- prompt loop for in-process teammates
- mailbox polling with the same priority order
- task-list claiming for idle workers
- idle notifications back to the team lead
- graceful completion on shutdown approval / abort

The actual LLM execution is intentionally injected as an async callable so the
coordination logic stays source-faithful while remaining framework-agnostic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import Any, Literal
import time

from .abort import AbortController, AbortError, create_child_abort_controller, sleep_with_abort
from .constants import TEAM_LEAD_NAME, WAIT_FOR_PROMPT_POLL_MS
from .contexts import create_teammate_context, run_with_teammate_context
from .in_process_teammate import append_capped_message
from .mailbox import (
    ShutdownRequestMessage,
    TeammateMessage,
    create_idle_notification,
    format_teammate_messages,
    is_shutdown_request,
    mark_message_as_read_by_index,
    protocol_to_json,
    read_mailbox,
    write_to_mailbox,
)
from .models import AgentProgress, InProcessTeammateTaskState, TaskListTask
from .runtime_state import AppStateStore
from .task_framework import update_task_state
from .task_list import claim_task, list_tasks, update_task


@dataclass
class WaitResult:
    type: Literal["shutdown_request", "new_message", "aborted"]
    message: str | None = None
    from_: str | None = None
    color: str | None = None
    summary: str | None = None
    request: ShutdownRequestMessage | None = None
    original_message: str | None = None


@dataclass
class ExecutorOutcome:
    assistant_message: str | None = None
    summary: str | None = None
    result: dict[str, Any] | None = None
    progress: AgentProgress | None = None
    completed_task_id: str | None = None
    completed_status: Literal["resolved", "blocked", "failed"] | None = None


InProcessExecutor = Callable[
    [str, InProcessTeammateTaskState, AppStateStore, AbortController],
    Awaitable[str | dict[str, Any] | ExecutorOutcome | None],
]


@dataclass
class InProcessRunnerResult:
    success: bool
    prompts_seen: list[str]
    final_status: Literal["completed", "failed"]
    error: str | None = None


def _get_task(store: AppStateStore, task_id: str) -> InProcessTeammateTaskState:
    task = store.get_state().tasks.get(task_id)
    if not isinstance(task, InProcessTeammateTaskState):
        raise KeyError(f"Task {task_id} is not an in-process teammate task")
    return task


def _task_list_id_for(task: InProcessTeammateTaskState, explicit: str | None) -> str:
    if explicit:
        return explicit
    return task.identity.parent_session_id


def find_available_task(tasks: list[TaskListTask]) -> TaskListTask | None:
    """Return the first pending, unowned, unblocked task."""
    unresolved_task_ids = {task.id for task in tasks if task.status != "completed"}
    for task in tasks:
        if task.status != "pending":
            continue
        if task.owner:
            continue
        if all(blocker not in unresolved_task_ids for blocker in task.blocked_by):
            return task
    return None


def format_task_as_prompt(task: TaskListTask) -> str:
    prompt = f"Complete all open tasks. Start with task #{task.id}: \n\n {task.subject}"
    if task.description:
        prompt += f"\n\n{task.description}"
    return prompt


def _format_as_teammate_message(
    from_: str,
    message: str,
    color: str | None = None,
    summary: str | None = None,
) -> str:
    return format_teammate_messages(
        [
            TeammateMessage(
                from_=from_,
                text=message,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                color=color,
                summary=summary,
            )
        ]
    )


async def send_idle_notification(
    agent_name: str,
    agent_color: str | None,
    team_name: str,
    *,
    idle_reason: Literal["available", "interrupted", "failed"] | None = None,
    summary: str | None = None,
    completed_task_id: str | None = None,
    completed_status: Literal["resolved", "blocked", "failed"] | None = None,
    failure_reason: str | None = None,
) -> None:
    notification = create_idle_notification(
        agent_name,
        idle_reason=idle_reason,
        summary=summary,
        completed_task_id=completed_task_id,
        completed_status=completed_status,
        failure_reason=failure_reason,
    )
    write_to_mailbox(
        TEAM_LEAD_NAME,
        TeammateMessage(
            from_=agent_name,
            text=protocol_to_json(notification),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            color=agent_color,
            summary=summary,
        ),
        team_name,
    )


async def try_claim_next_task(task_list_id: str, agent_name: str) -> str | None:
    try:
        tasks = list_tasks(task_list_id)
        available_task = find_available_task(tasks)
        if available_task is None:
            return None
        result = claim_task(task_list_id, available_task.id, agent_name)
        if not result.get("success"):
            return None
        update_task(task_list_id, available_task.id, {"status": "in_progress"})
        return format_task_as_prompt(available_task)
    except Exception:
        return None


async def wait_for_next_prompt_or_shutdown(
    *,
    task_id: str,
    store: AppStateStore,
    task_list_id: str,
) -> WaitResult:
    task = _get_task(store, task_id)
    identity = task.identity
    abort_controller = task.abort_controller
    if abort_controller is None:
        return WaitResult(type="aborted")

    poll_count = 0
    while not abort_controller.signal.aborted:
        current_task = _get_task(store, task_id)
        if current_task.pending_user_messages:
            message = current_task.pending_user_messages[0]

            def pop_pending(state):
                current = state.tasks.get(task_id)
                if not isinstance(current, InProcessTeammateTaskState):
                    return state
                return update_task_state(
                    task_id,
                    state,
                    lambda item: replace(
                        item,
                        pending_user_messages=item.pending_user_messages[1:],
                    ),
                )

            store.set_state(pop_pending)
            return WaitResult(type="new_message", message=message, from_="user")

        if poll_count > 0:
            try:
                await sleep_with_abort(WAIT_FOR_PROMPT_POLL_MS, abort_controller.signal)
            except AbortError:
                return WaitResult(type="aborted")
        poll_count += 1

        if abort_controller.signal.aborted:
            return WaitResult(type="aborted")

        try:
            all_messages = read_mailbox(identity.agent_name, identity.team_name)

            shutdown_index = -1
            shutdown_parsed: ShutdownRequestMessage | None = None
            for index, message in enumerate(all_messages):
                if message.read:
                    continue
                parsed = is_shutdown_request(message.text)
                if parsed is not None:
                    shutdown_index = index
                    shutdown_parsed = parsed
                    break
            if shutdown_index != -1 and shutdown_parsed is not None:
                original = all_messages[shutdown_index].text
                mark_message_as_read_by_index(
                    identity.agent_name,
                    identity.team_name,
                    shutdown_index,
                )
                return WaitResult(
                    type="shutdown_request",
                    request=shutdown_parsed,
                    original_message=original,
                )

            selected_index = -1
            for index, message in enumerate(all_messages):
                if not message.read and message.from_ == TEAM_LEAD_NAME:
                    selected_index = index
                    break
            if selected_index == -1:
                for index, message in enumerate(all_messages):
                    if not message.read:
                        selected_index = index
                        break

            if selected_index != -1:
                message = all_messages[selected_index]
                mark_message_as_read_by_index(
                    identity.agent_name,
                    identity.team_name,
                    selected_index,
                )
                return WaitResult(
                    type="new_message",
                    message=message.text,
                    from_=message.from_,
                    color=message.color,
                    summary=message.summary,
                )
        except Exception:
            pass

        task_prompt = await try_claim_next_task(task_list_id, identity.agent_name)
        if task_prompt is not None:
            return WaitResult(type="new_message", message=task_prompt, from_="task-list")

    return WaitResult(type="aborted")


def _normalize_outcome(
    value: str | dict[str, Any] | ExecutorOutcome | None,
) -> ExecutorOutcome:
    if value is None:
        return ExecutorOutcome()
    if isinstance(value, ExecutorOutcome):
        return value
    if isinstance(value, str):
        return ExecutorOutcome(assistant_message=value)
    if isinstance(value, dict):
        progress_value = value.get("progress")
        progress: AgentProgress | None
        if isinstance(progress_value, AgentProgress):
            progress = progress_value
        elif isinstance(progress_value, dict):
            progress = AgentProgress(**progress_value)
        else:
            progress = None
        return ExecutorOutcome(
            assistant_message=value.get("assistant_message"),
            summary=value.get("summary"),
            result=value.get("result"),
            progress=progress,
            completed_task_id=value.get("completed_task_id"),
            completed_status=value.get("completed_status"),
        )
    raise TypeError(f"Unsupported executor outcome: {type(value)!r}")


def _append_transcript_item(
    store: AppStateStore,
    task_id: str,
    item: dict[str, Any],
) -> None:
    store.set_state(
        lambda state: update_task_state(
            task_id,
            state,
            lambda task: replace(
                task,
                messages=append_capped_message(task.messages, item),
            ),
        )
    )


def _mark_idle(
    store: AppStateStore,
    task_id: str,
    *,
    progress: AgentProgress | None = None,
    assistant_message: str | None = None,
    result: dict[str, Any] | None = None,
) -> bool:
    previous = _get_task(store, task_id)
    was_already_idle = previous.is_idle

    def apply(task: InProcessTeammateTaskState) -> InProcessTeammateTaskState:
        callbacks = list(task.on_idle_callbacks)
        for callback in callbacks:
            with suppress(Exception):
                callback()
        new_progress = progress or task.progress
        if progress is None and assistant_message:
            new_progress = AgentProgress(summary=assistant_message[:120])
        return replace(
            task,
            progress=new_progress,
            result=result or task.result,
            is_idle=True,
            current_work_abort_controller=None,
            on_idle_callbacks=[],
        )

    store.set_state(lambda state: update_task_state(task_id, state, apply))
    return was_already_idle


def _mark_running(
    store: AppStateStore,
    task_id: str,
    work_abort_controller: AbortController,
) -> None:
    store.set_state(
        lambda state: update_task_state(
            task_id,
            state,
            lambda task: replace(
                task,
                status="running",
                is_idle=False,
                current_work_abort_controller=work_abort_controller,
            ),
        )
    )


def _mark_completed(store: AppStateStore, task_id: str) -> None:
    now_ms = int(time.time() * 1000)

    def apply(task: InProcessTeammateTaskState) -> InProcessTeammateTaskState:
        if task.status != "running":
            return task
        callbacks = list(task.on_idle_callbacks)
        for callback in callbacks:
            with suppress(Exception):
                callback()
        last_messages = [task.messages[-1]] if task.messages else []
        return replace(
            task,
            status="completed",
            notified=True,
            end_time=now_ms,
            messages=last_messages,
            pending_user_messages=[],
            in_progress_tool_use_ids=None,
            abort_controller=None,
            current_work_abort_controller=None,
            unregister_cleanup=None,
            on_idle_callbacks=[],
            is_idle=True,
        )

    store.set_state(lambda state: update_task_state(task_id, state, apply))


def _mark_failed(store: AppStateStore, task_id: str, error: str) -> None:
    now_ms = int(time.time() * 1000)

    def apply(task: InProcessTeammateTaskState) -> InProcessTeammateTaskState:
        if task.status != "running":
            return task
        callbacks = list(task.on_idle_callbacks)
        for callback in callbacks:
            with suppress(Exception):
                callback()
        last_messages = [task.messages[-1]] if task.messages else []
        return replace(
            task,
            status="failed",
            notified=True,
            error=error,
            is_idle=True,
            end_time=now_ms,
            messages=last_messages,
            pending_user_messages=[],
            in_progress_tool_use_ids=None,
            abort_controller=None,
            unregister_cleanup=None,
            current_work_abort_controller=None,
            on_idle_callbacks=[],
        )

    store.set_state(lambda state: update_task_state(task_id, state, apply))


async def run_in_process_teammate(
    *,
    task_id: str,
    store: AppStateStore,
    executor: InProcessExecutor,
    task_list_id: str | None = None,
) -> InProcessRunnerResult:
    task = _get_task(store, task_id)
    identity = task.identity
    abort_controller = task.abort_controller
    if abort_controller is None:
        raise RuntimeError(f"Task {task_id} has no abort controller")

    teammate_context = create_teammate_context(
        agent_id=identity.agent_id,
        agent_name=identity.agent_name,
        team_name=identity.team_name,
        color=identity.color,
        plan_mode_required=identity.plan_mode_required,
        parent_session_id=identity.parent_session_id,
        abort_controller=abort_controller,
    )
    prompts_seen: list[str] = []
    current_prompt = task.prompt
    actual_task_list_id = _task_list_id_for(task, task_list_id)
    should_exit = False

    try:
        while not should_exit:
            work_abort = create_child_abort_controller(abort_controller)
            _mark_running(store, task_id, work_abort)
            prompts_seen.append(current_prompt)

            work_was_aborted = False
            outcome = ExecutorOutcome()
            with run_with_teammate_context(teammate_context):
                snapshot = _get_task(store, task_id)
                try:
                    raw_outcome = await executor(current_prompt, snapshot, store, work_abort)
                    outcome = _normalize_outcome(raw_outcome)
                except AbortError:
                    work_was_aborted = True

            if outcome.assistant_message:
                _append_transcript_item(
                    store,
                    task_id,
                    {"type": "assistant", "content": outcome.assistant_message},
                )

            was_already_idle = _mark_idle(
                store,
                task_id,
                progress=outcome.progress,
                assistant_message=outcome.assistant_message,
                result=outcome.result,
            )

            if not was_already_idle:
                await send_idle_notification(
                    identity.agent_name,
                    identity.color,
                    identity.team_name,
                    idle_reason="interrupted" if work_was_aborted else "available",
                    summary=outcome.summary,
                    completed_task_id=outcome.completed_task_id,
                    completed_status=outcome.completed_status,
                )

            wait_result = await wait_for_next_prompt_or_shutdown(
                task_id=task_id,
                store=store,
                task_list_id=actual_task_list_id,
            )

            if wait_result.type == "shutdown_request":
                current_prompt = _format_as_teammate_message(
                    wait_result.request.from_ if wait_result.request else TEAM_LEAD_NAME,
                    wait_result.original_message or "",
                )
                _append_transcript_item(
                    store,
                    task_id,
                    {"type": "user", "content": current_prompt},
                )
            elif wait_result.type == "new_message":
                if wait_result.from_ == "user":
                    current_prompt = wait_result.message or ""
                else:
                    current_prompt = _format_as_teammate_message(
                        wait_result.from_ or TEAM_LEAD_NAME,
                        wait_result.message or "",
                        wait_result.color,
                        wait_result.summary,
                    )
                    _append_transcript_item(
                        store,
                        task_id,
                        {"type": "user", "content": current_prompt},
                    )
            else:
                should_exit = True

        _mark_completed(store, task_id)
        return InProcessRunnerResult(
            success=True,
            prompts_seen=prompts_seen,
            final_status="completed",
        )
    except Exception as exc:
        error_message = str(exc) or exc.__class__.__name__
        _mark_failed(store, task_id, error_message)
        await send_idle_notification(
            identity.agent_name,
            identity.color,
            identity.team_name,
            idle_reason="failed",
            completed_status="failed",
            failure_reason=error_message,
        )
        return InProcessRunnerResult(
            success=False,
            prompts_seen=prompts_seen,
            final_status="failed",
            error=error_message,
        )
