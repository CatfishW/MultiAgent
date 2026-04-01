# Source mapping

Focused mapping between the copied TypeScript swarm subsystem and the Python
port.

| TypeScript source | Python port |
|---|---|
| `src/utils/agentId.ts` | `src/agent_swarm_port/ids.py` |
| `src/utils/teammate.ts` + `src/utils/teammateContext.ts` | `src/agent_swarm_port/contexts.py` |
| `src/utils/mailbox.ts` + `src/utils/teammateMailbox.ts` | `src/agent_swarm_port/mailbox.py` |
| `src/utils/tasks.ts` | `src/agent_swarm_port/task_list.py` |
| `src/utils/task/framework.ts` + `src/Task.ts` | `src/agent_swarm_port/task_framework.py` |
| `src/tasks/LocalAgentTask/LocalAgentTask.tsx` | `src/agent_swarm_port/local_agent.py` |
| `src/tasks/InProcessTeammateTask/InProcessTeammateTask.tsx` | `src/agent_swarm_port/in_process_teammate.py` |
| `src/utils/swarm/teamHelpers.ts` | `src/agent_swarm_port/team_store.py` |
| `src/tools/TeamCreateTool/TeamCreateTool.ts` | `src/agent_swarm_port/team_service.py#create_team` |
| `src/tools/TeamDeleteTool/TeamDeleteTool.ts` | `src/agent_swarm_port/team_service.py#delete_team` |
| `src/utils/swarm/spawnInProcess.ts` | `src/agent_swarm_port/spawn_inprocess.py` |
| `src/utils/swarm/inProcessRunner.ts` | `src/agent_swarm_port/inprocess_runner.py` |
| `src/tools/TaskCreateTool/TaskCreateTool.ts` | `src/agent_swarm_port/task_service.py#create_task_entry` |
| `src/tools/TaskUpdateTool/TaskUpdateTool.ts` | `src/agent_swarm_port/task_service.py#update_task_entry` |
| `src/tools/TaskListTool/TaskListTool.ts` | `src/agent_swarm_port/task_service.py#list_task_entries` |
| `src/tools/TaskGetTool/TaskGetTool.ts` | `src/agent_swarm_port/task_service.py#get_task_entry` |
| `src/tools/SendMessageTool/SendMessageTool.ts` | `src/agent_swarm_port/message_service.py` |
| `src/coordinator/coordinatorMode.ts` | `src/agent_swarm_port/coordinator_mode.py` |
| backend-layer coordination spread across swarm files | `src/agent_swarm_port/backends.py` |

## Intentional adaptation points

The following are deliberate substitutions rather than one-for-one runtime
copies:

- the original `runAgent()` call path is replaced by an injected async
  executor callback
- tmux / iTerm2 / UI-specific control paths are not reimplemented here
- resume of background local agents is exposed as a pluggable callback in
  `message_service.py` instead of a hard dependency on the original runtime
