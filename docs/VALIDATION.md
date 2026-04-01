# Validation

Validation run performed in the build container:

```bash
python -m pytest -q
```

Observed result:

```text
........                                                                 [100%]
```

Covered areas:

- deterministic ID round-trips
- task list block / claim / unassign semantics
- task service assignment mail + blocker filtering
- message routing priority for running local agents and shutdown requests
- in-process waiter priority ordering
- in-process runner idle notification + graceful completion path
