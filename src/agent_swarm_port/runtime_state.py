"""Helpers for mutating the in-memory application state."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TypeVar

from .models import AppState, TaskStateBase

T = TypeVar("T", bound=TaskStateBase)


class AppStateStore:
    def __init__(self, initial: AppState | None = None) -> None:
        self._state = initial or AppState()

    def get_state(self) -> AppState:
        return self._state

    def set_state(self, updater: Callable[[AppState], AppState]) -> AppState:
        self._state = updater(self._state)
        return self._state

    def patch(self, **kwargs: object) -> AppState:
        self._state = replace(self._state, **kwargs)
        return self._state
