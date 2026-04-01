import json

from agent_swarm_port.models import TeamMember
from agent_swarm_port.mailbox import read_mailbox
from agent_swarm_port.runtime_state import AppStateStore
from agent_swarm_port.task_service import create_task_entry, list_task_entries, update_task_entry
from agent_swarm_port.team_service import create_team
from agent_swarm_port.team_store import read_team_file, write_team_file


def test_task_service_assignment_and_block_filtering(temp_home):
    store = AppStateStore()
    state, created = create_team(
        store.get_state(),
        team_name="alpha",
        session_id="alpha",
        cwd=".",
        description="alpha team",
    )
    store.set_state(lambda _prev: state)

    team_file = read_team_file("alpha")
    assert team_file is not None
    team_file.members.append(
        TeamMember(
            agent_id="worker1@alpha",
            name="worker1",
            agent_type="in-process",
            joined_at=1,
            cwd=".",
            backend_type="in-process",
            is_active=True,
        )
    )
    write_team_file("alpha", team_file)

    task1 = create_task_entry(task_list_id="alpha", subject="Research bug", description="Inspect logs")
    task2 = create_task_entry(task_list_id="alpha", subject="Ship fix", description="Patch code")

    update_task_entry(
        task_list_id="alpha",
        task_id=task1.id,
        status="completed",
    )
    update_task_entry(
        task_list_id="alpha",
        task_id=task2.id,
        owner="worker1",
        add_blocked_by=[task1.id],
        team_name="alpha",
        sender_name="team-lead",
    )

    mailbox = read_mailbox("worker1", "alpha")
    assert len(mailbox) == 1
    payload = json.loads(mailbox[0].text)
    assert payload["type"] == "task_assignment"
    assert payload["task_id"] == task2.id

    listed = list_task_entries(task_list_id="alpha")
    listed_task2 = next(item for item in listed if item["id"] == task2.id)
    assert listed_task2["owner"] == "worker1"
    assert listed_task2["blocked_by"] == []
