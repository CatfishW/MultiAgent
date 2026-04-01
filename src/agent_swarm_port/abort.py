"""Abort-controller style cancellation helpers for Python async code.

The TypeScript source relies on AbortController extensively. This module keeps
that shape while mapping it onto ``asyncio.Event`` and task cancellation.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any


class AbortError(asyncio.CancelledError):
    """Raised when an ``AbortController`` has been aborted."""


@dataclass
class AbortSignal:
    _event: asyncio.Event = field(default_factory=asyncio.Event)
    _reason: str | None = None
    _callbacks: list[Callable[[], Any]] = field(default_factory=list)

    @property
    def aborted(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def add_callback(self, callback: Callable[[], Any]) -> None:
        if self.aborted:
            callback()
            return
        self._callbacks.append(callback)

    async def wait(self) -> None:
        await self._event.wait()

    def throw_if_aborted(self) -> None:
        if self.aborted:
            raise AbortError(self.reason or "Operation aborted")


@dataclass
class AbortController:
    signal: AbortSignal = field(default_factory=AbortSignal)

    def abort(self, reason: str | None = None) -> None:
        if self.signal.aborted:
            return
        self.signal._reason = reason
        self.signal._event.set()
        callbacks = list(self.signal._callbacks)
        self.signal._callbacks.clear()
        for callback in callbacks:
            with suppress(Exception):
                callback()


def create_abort_controller() -> AbortController:
    return AbortController()


def create_child_abort_controller(parent: AbortController | AbortSignal) -> AbortController:
    parent_signal = parent.signal if isinstance(parent, AbortController) else parent
    child = AbortController()
    parent_signal.add_callback(lambda: child.abort(parent_signal.reason))
    return child


async def sleep_with_abort(delay_ms: int, signal: AbortSignal) -> None:
    if signal.aborted:
        signal.throw_if_aborted()
    sleep_task = asyncio.create_task(asyncio.sleep(delay_ms / 1000))
    abort_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait({sleep_task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
        with suppress(BaseException):
            await task
    if abort_task in done:
        signal.throw_if_aborted()
    await sleep_task


async def run_cancellable(awaitable: Awaitable[Any], signal: AbortSignal) -> Any:
    signal.throw_if_aborted()
    task = asyncio.create_task(awaitable)
    abort_task = asyncio.create_task(signal.wait())
    done, pending = await asyncio.wait({task, abort_task}, return_when=asyncio.FIRST_COMPLETED)
    if abort_task in done:
        task.cancel()
        with suppress(BaseException):
            await task
        signal.throw_if_aborted()
    abort_task.cancel()
    with suppress(BaseException):
        await abort_task
    return await task
