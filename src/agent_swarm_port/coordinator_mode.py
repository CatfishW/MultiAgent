"""Coordinator-mode helpers.

This is a focused Python analogue to ``src/coordinator/coordinatorMode.ts``.
It keeps the session-mode matching and worker-context generation behavior
without copying the full product prompt verbatim.
"""

from __future__ import annotations

import os
from typing import Literal

_SESSION_MODE_ENV = "AGENT_SWARM_PORT_COORDINATOR_MODE"


def is_coordinator_mode() -> bool:
    return os.environ.get(_SESSION_MODE_ENV, "").lower() in {"1", "true", "yes", "on"}


def match_session_mode(
    session_mode: Literal["coordinator", "normal"] | None,
) -> str | None:
    if session_mode is None:
        return None
    current = is_coordinator_mode()
    wanted = session_mode == "coordinator"
    if current == wanted:
        return None
    if wanted:
        os.environ[_SESSION_MODE_ENV] = "1"
        return "Entered coordinator mode to match resumed session."
    os.environ.pop(_SESSION_MODE_ENV, None)
    return "Exited coordinator mode to match resumed session."


def get_coordinator_user_context(
    *,
    worker_tools: list[str],
    mcp_servers: list[str] | None = None,
    scratchpad_dir: str | None = None,
) -> dict[str, str]:
    if not is_coordinator_mode():
        return {}
    content = (
        "Workers spawned by the coordinator have access to these tools: "
        + ", ".join(sorted(worker_tools))
    )
    if mcp_servers:
        content += "\n\nWorkers also have access to MCP tools from: " + ", ".join(mcp_servers)
    if scratchpad_dir:
        content += (
            f"\n\nScratchpad directory: {scratchpad_dir}. "
            "Workers can use it for durable cross-worker knowledge."
        )
    return {"worker_tools_context": content}


def get_coordinator_system_prompt(worker_capabilities: str) -> str:
    return (
        "You are a coordinator agent. Launch workers for research, implementation, "
        "and verification; synthesize their results; and only delegate when that "
        "improves parallelism or preserves context. "
        f"Workers capabilities: {worker_capabilities}."
    )
