from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from ..core.contracts import AgentResult


class FastGraphRuntime:
    async def run_parallel(self, tasks: dict[str, Callable[[], Awaitable[AgentResult]]]) -> dict[str, AgentResult]:
        if not tasks:
            return {}
        names = list(tasks.keys())
        results = await asyncio.gather(*[tasks[name]() for name in names])
        return {name: result for name, result in zip(names, results)}
