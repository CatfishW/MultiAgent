from __future__ import annotations

import os

import pytest

from agent_swarm_port.runtime_state import AppStateStore


@pytest.fixture()
def temp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SWARM_PORT_HOME", str(tmp_path / "agent_swarm_home"))
    return tmp_path / "agent_swarm_home"


@pytest.fixture()
def store(temp_home):
    return AppStateStore()
