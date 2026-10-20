# TorchFrequentDirections — SAE V3.3

Framework: JohnnyZPiggy/SAE-Framework at
`a60ed92d80a8866b413b928d609cc3afb82ec408`, complete immutable [.sae](.sae/AGENTS.md).
Project rules take precedence over the snapshot's examples and historical diagrams.

- Scientist supplies objectives and may stop work. GPT Mentor/Architect owns math,
  numerical acceptance and merge decisions. Codex implements/tests and submits PRs;
  never self-approves or merges. No routine Scientist approval queue.
- Follow [Engineer](.sae/protocols/ENGINEER.md), [Role Recovery](.sae/workflows/ROLE_RECOVERY.md),
  [Lean Lab](.sae/workflows/LEAN_RESEARCH_LAB.md), [Task Management](.sae/workflows/TASK_MANAGEMENT.md)
  and [Standing Authorization](.sae/governance/STANDING_AUTHORIZATION.md).
- Issue #1 is one CPU-first EXPLORE batch, maximum 3600 seconds from claim,
  maximum three repairs per failure class. Record actual failures, source identity,
  tests, dtype, seeds and limitations. GPU optional, never implied by CPU success.
- Existing-account compute only, at most one session with finite timeout and
  authenticated cleanup if used; no billing, background services, Actions or TPU.
- Preserve MIT LICENSE and the exact .sae snapshot; no external paper/code copies
  without verified rights. Do not modify sibling projects.
- Final response includes the [Next Action](.sae/workflows/NEXT_ACTION.md) footer.
