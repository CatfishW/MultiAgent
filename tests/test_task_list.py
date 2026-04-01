from agent_swarm_port.task_list import (
    block_task,
    claim_task,
    create_task,
    get_task,
    list_tasks,
    unassign_teammate_tasks,
    update_task,
)


def test_task_list_block_claim_and_unassign(temp_home):
    task_list_id = "alpha"
    task1 = create_task(
        task_list_id,
        {
            "subject": "Investigate",
            "description": "Inspect logs",
            "active_form": None,
            "status": "pending",
            "owner": None,
            "blocks": [],
            "blocked_by": [],
            "metadata": None,
        },
    )
    task2 = create_task(
        task_list_id,
        {
            "subject": "Implement",
            "description": "Write fix",
            "active_form": None,
            "status": "pending",
            "owner": None,
            "blocks": [],
            "blocked_by": [],
            "metadata": None,
        },
    )

    assert block_task(task_list_id, task1, task2) is True

    blocked = claim_task(task_list_id, task2, "worker-a")
    assert blocked["success"] is False
    assert blocked["reason"] == "blocked"

    claimed = claim_task(task_list_id, task1, "worker-a")
    assert claimed["success"] is True
    update_task(task_list_id, task1, {"status": "completed"})

    claimed2 = claim_task(task_list_id, task2, "worker-a")
    assert claimed2["success"] is True
    update_task(task_list_id, task2, {"status": "in_progress"})

    result = unassign_teammate_tasks(task_list_id, "worker-a", "worker-a", "shutdown")
    assert result["unassignedTasks"] == [{"id": task2, "subject": "Implement"}]
    assert get_task(task_list_id, task2).owner is None
    assert get_task(task_list_id, task2).status == "pending"

    tasks = list_tasks(task_list_id)
    assert [task.id for task in tasks] == [task1, task2]
