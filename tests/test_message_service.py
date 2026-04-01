import json

from agent_swarm_port.local_agent import register_async_agent
from agent_swarm_port.mailbox import is_shutdown_request, read_mailbox
from agent_swarm_port.message_service import route_plain_message, send_shutdown_request
from agent_swarm_port.runtime_state import AppStateStore
from agent_swarm_port.team_service import create_team


def test_message_service_prefers_running_local_agent(temp_home):
    store = AppStateStore()
    state, _ = create_team(
        store.get_state(),
        team_name="alpha",
        session_id="alpha",
        cwd=".",
    )
    store.set_state(lambda _prev: state)
    next_state, task = register_async_agent(
        agent_id="builder@alpha",
        description="Builder",
        prompt="Wait",
        selected_agent={"agentType": "builder"},
        state=store.get_state(),
    )
    store.set_state(lambda _prev: next_state)

    result = route_plain_message(
        store=store,
        to="builder@alpha",
        message="continue",
        summary="resume build",
        team_name="alpha",
    )
    assert result["success"] is True
    assert store.get_state().tasks[task.id].pending_messages == ["continue"]


def test_message_service_mailbox_and_shutdown_request(temp_home):
    store = AppStateStore()
    state, _ = create_team(
        store.get_state(),
        team_name="alpha",
        session_id="alpha",
        cwd=".",
    )
    store.set_state(lambda _prev: state)

    route_plain_message(
        store=store,
        to="worker1",
        message="please inspect task #2",
        summary="inspection",
        team_name="alpha",
        sender_name="team-lead",
    )
    inbox = read_mailbox("worker1", "alpha")
    assert inbox[-1].text == "please inspect task #2"

    result = send_shutdown_request(
        store=store,
        target_name="worker1",
        reason="done for now",
        team_name="alpha",
        sender_name="team-lead",
    )
    assert result["success"] is True
    inbox = read_mailbox("worker1", "alpha")
    parsed = is_shutdown_request(inbox[-1].text)
    assert parsed is not None
    assert parsed.from_ == "team-lead"
    assert parsed.reason == "done for now"
