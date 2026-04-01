# Deep analysis of the source swarm subsystem

## 1. Identity and lineage

The codebase uses deterministic IDs to keep team membership and requests
traceable across file-backed state and in-memory runtime state.

- agent ID: `agentName@teamName`
- request ID: `requestType-timestamp@agentId`

This makes teammate identity stable across:

- task ownership
- mailbox routing
- shutdown approval flow
- permission bridge / plan approval flow
- analytics attribution

## 2. Team model

A team has exactly one lead and zero or more teammates.

The team file stores:

- `leadAgentId`, `leadSessionId`, `createdAt`, `description`
- hidden panes / backend state
- team-allowed paths
- `members[]` with fields such as:
  - `agentId`, `name`, `agentType`, `model`, `prompt`, `color`
  - `planModeRequired`, `joinedAt`
  - `tmuxPaneId`, `cwd`, `worktreePath`
  - `sessionId`, `subscriptions`
  - `backendType`, `isActive`, `mode`

Important lifecycle semantics extracted from the source:

- `TeamCreateTool` initializes the team file, registers cleanup, resets the
  team's task list, and binds the leader's task-list context to the team
- `TeamDeleteTool` refuses deletion while active non-lead members remain
- leader context and teammate runtime context are related but separate

## 3. Coordination planes

The swarm logic is split across several coordination planes.

### A. In-memory `AppState`
Used for live task/task-panel state, pending transcript-injected user messages,
and foreground/background runtime state.

### B. File-backed team config
Used to persist team membership and backend metadata.

### C. File-backed teammate mailbox
Used for inter-agent coordination, including:

- plain teammate messages
- shutdown requests / responses
- plan approval responses
- permission sync traffic
- task assignment notifications
- idle notifications back to the leader

### D. File-backed task list
Used as the durable shared work queue.

Core semantics preserved in the port:

- sequential numeric task IDs
- locking around creates/updates/claims
- explicit `blocks` and `blockedBy`
- claim refusal reasons:
  - `task_not_found`
  - `already_claimed`
  - `already_resolved`
  - `blocked`
  - `agent_busy`

## 4. Background task model

The app-level background task model is separate from the file-backed task list.

Background task types seen in the source:

- `local_bash`
- `local_agent`
- `remote_agent`
- `in_process_teammate`
- `local_workflow`
- `monitor_mcp`
- `dream`

Background task statuses:

- `pending`
- `running`
- `completed`
- `failed`
- `killed`

Task-list statuses are different and much smaller:

- `pending`
- `in_progress`
- `completed`

## 5. In-process teammate orchestration

The most important extracted behavior is the continuous loop used by
in-process teammates.

### Spawn phase

The source creates:

- deterministic teammate identity
- a dedicated abort controller
- an in-process teammate task entry in `AppState`
- per-teammate runtime context

### Run loop behavior

The teammate does **not** terminate after a single prompt. It repeatedly:

1. works on the current prompt
2. transitions to idle
3. notifies the leader of idleness
4. waits for the next prompt, shutdown request, or available task
5. resumes work

### Poll priority order

The exact source ordering matters and is preserved in the Python port.

While idle, the teammate checks in this order:

1. pending user messages injected from transcript/UI state
2. abort status
3. unread mailbox shutdown requests
4. unread team-lead mailbox messages
5. first unread peer mailbox message
6. first claimable unowned unblocked task from the shared task list

That ordering prevents leader messages and shutdown requests from being starved
behind peer chatter.

### Idle notifications

Idle notifications are sent back to the leader over the file-backed mailbox.
They can indicate:

- available
- interrupted
- failed

## 6. Task-list claiming behavior

The source's task-list claim flow is conservative.

A teammate can claim a task only when:

- the task exists
- it is not completed
- it is not already owned by another agent
- all blockers are resolved
- optional busy-check passes when enabled

After a successful claim, the source immediately marks the task as
`in_progress` so UI state and teammate status stay aligned.

## 7. SendMessage routing behavior

Plain text messages follow this routing order:

1. try a running local background agent first
2. otherwise try background-agent resume flow
3. otherwise route to teammate mailbox or broadcast

Structured messages implement protocol-level workflows:

- shutdown request
- shutdown response
- plan approval response

For in-process teammates, shutdown approval ultimately aborts the teammate's
controller, which causes the idle wait loop to exit and the task to finalize
as completed.

## 8. Why the Python port is organized this way

The port is grouped around the same boundaries as the source:

- state model
- file persistence
- runtime loop
- backend wrapper
- service/tool layer

That keeps the coordination semantics readable and avoids flattening everything
into one giant translation file.
