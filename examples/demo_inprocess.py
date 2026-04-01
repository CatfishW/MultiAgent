from __future__ import annotations

import asyncio
import os
import tempfile

from agent_swarm_port.backends import start_inprocess_backend
from agent_swarm_port.mailbox import read_mailbox
from agent_swarm_port.runtime_state import AppStateStore
from agent_swarm_port.team_service import create_team


async def demo() -> None:
    temp_root = tempfile.mkdtemp(prefix="agent-swarm-port-")
    os.environ["AGENT_SWARM_PORT_HOME"] = temp_root

    store = AppStateStore()
    state, created = create_team(
        store.get_state(),
        team_name="demo-team",
        session_id="demo-team",
        cwd=os.getcwd(),
        description="Demo team",
    )
    store.set_state(lambda _prev: state)

    async def executor(prompt, task, store, abort_controller):
        if "shutdown_request" in prompt:
            task.abort_controller.abort("approved")
            return {"assistant_message": "Shutting down.", "summary": "approved shutdown"}
        return {"assistant_message": f"Handled: {prompt}", "summary": "available"}

    handle = start_inprocess_backend(
        store=store,
        name="worker-1",
        team_name=created["team_name"],
        prompt="Inspect the codebase",
        parent_session_id=created["team_name"],
        executor=executor,
    )

    await asyncio.sleep(0.8)
    leader_mail = read_mailbox("team-lead", created["team_name"])
    print(f"Leader inbox contains {len(leader_mail)} message(s)")

    handle.asyncio_task.cancel()
    try:
        await handle.asyncio_task
    except BaseException:
        pass


if __name__ == "__main__":
    asyncio.run(demo())
