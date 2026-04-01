from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from agent_swarm_port.backends import start_inprocess_backend
from agent_swarm_port.models import AppState
from agent_swarm_port.runtime_state import AppStateStore
from agent_swarm_port.team_service import create_team, delete_team

from ..core.contracts import AgentResult


class SwarmRuntimeAdapter:
    """Optional bridge to the uploaded swarm port.

    The default execution path in this project is the lighter in-memory
    `FastGraphRuntime`. This adapter exists so experiments can still execute
    specialist roles on top of the uploaded in-process swarm runtime when a
    reviewer or ablation requires it.
    """

    def __init__(self, *, team_name: str = "edu-swarm", session_id: str = "edu-session", cwd: str = ".") -> None:
        self.store = AppStateStore(AppState(current_cwd=cwd))
        state, _ = create_team(self.store.get_state(), team_name=team_name, session_id=session_id, cwd=cwd, description="Educational benchmark swarm")
        self.store.set_state(lambda _prev: state)

    async def run_parallel_roles(
        self,
        roles: dict[str, Callable[[], Awaitable[AgentResult]]],
    ) -> dict[str, AgentResult]:
        handles = []
        for role_name, factory in roles.items():
            async def executor(_prompt, _task, _store, abort_controller, *, role_factory=factory, role=role_name):
                result = await role_factory()
                abort_controller.abort("done")
                return {
                    "assistant_message": result.text,
                    "summary": role,
                    "result": {
                        "role": result.role,
                        "text": result.text,
                        "confidence": result.confidence,
                        "artifacts": result.artifacts,
                        "citations": result.citations,
                    },
                }

            handles.append(
                start_inprocess_backend(
                    store=self.store,
                    name=role_name,
                    team_name=self.store.get_state().team_context.team_name if self.store.get_state().team_context else "edu-swarm",
                    prompt=f"run-role:{role_name}",
                    parent_session_id="edu-session",
                    executor=executor,
                    task_list_id="edu-session",
                )
            )
        await asyncio.gather(*(handle.asyncio_task for handle in handles))
        outputs: dict[str, AgentResult] = {}
        for handle in handles:
            task_state = self.store.get_state().tasks[handle.task_id]
            payload = task_state.result or {}
            outputs[handle.agent_id.split("@")[0]] = AgentResult(
                role=payload.get("role", handle.agent_id),
                text=payload.get("text", ""),
                confidence=float(payload.get("confidence", 0.5)),
                artifacts=payload.get("artifacts", {}),
                citations=list(payload.get("citations", [])),
            )
        return outputs

    def close(self) -> None:
        state, _ = delete_team(self.store.get_state())
        self.store.set_state(lambda _prev: state)
