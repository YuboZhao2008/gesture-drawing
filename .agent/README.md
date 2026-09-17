# Multi-agent engineering harness

These files are reusable prompts and repository policy for one accountable lead
and bounded specialists. They require no custom runtime, service, or new dependency.
The lead owns the outcome from objective through validated local integration.

## Starting a task

1. Go to the repository root.
2. Ensure main is clean: inspect `git status --short --branch` and
   `git worktree list --porcelain` before switching branches. Preserve any existing
   work; do not clean it away. Use the existing main checkout if already registered.
3. Start one lead Codex session with access to this repository and Git.
4. Paste [START_LEAD.md](START_LEAD.md), replacing `<OBJECTIVE>` with the final
   engineering outcome and any constraints or acceptance criteria.
5. The lead reads [AGENTS.md](../AGENTS.md) and
   [lead-orchestrator.md](lead-orchestrator.md).
6. The lead plans, delegates where useful, supervises, reviews, integrates, validates,
   and reports the final clean local main SHA. The human need not relay messages.

An explicit instruction to finish on a named task branch overrides the usual main
destination. Starting a lead does not authorize remote pushes or history rewriting.

## Files and capabilities

| File | Purpose |
| --- | --- |
| `lead-orchestrator.md` | Execution, delegation, checkpoints, and final ownership |
| `merge-policy.md` | Review, freeze, sequential integration, conflict/rejection rules |
| `validation.md` | Project commands, isolation, hardware and performance evidence |
| `state.example.json` | Blank checkpoint structure; never a live historical task |
| `roles/architect.md` | Contracts and integration hazards |
| `roles/implementer.md` | Bounded implementation and committed handoff |
| `roles/reviewer.md` | Independent adversarial review |
| `roles/performance.md` | Comparable measurements and tradeoffs |
| `roles/release.md` | Final tree and release evidence review |

Prefer native subagents when the running client exposes them. Do not assume that
spawning an agent creates a worktree: the lead must create and assign isolation,
and each writer must verify its actual working directory and branch.

Separate Codex sessions and separate worktrees can use the same policies if the
lead has a supported way to start/resume them, send instructions, and retrieve
their results. A shared checkpoint records progress; it is not a messaging service.
If those capabilities are unavailable, the lead performs the roles sequentially
itself and reports the lack of independent review. Never ask the human to ferry
specialist messages or imply that these Markdown files enable missing tools.
Permissions, session limits, and access to devices still come from the host.

Codex's [official subagent guidance](https://learn.chatgpt.com/docs/agent-configuration/subagents)
describes native delegation; its
[AGENTS.md guidance](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
describes instruction discovery. Other agents should read these files explicitly.
The `.agent/roles/` files are prompts to include in assignments, not installed agents.

## Checkpoints and interrupted sessions

Keep the live checkpoint outside the tracked tree, under the shared Git directory:
resolve `git rev-parse --path-format=absolute --git-common-dir`, then use
`orchestration/<task-id>/state.json` beneath that directory. This shares recovery
state across worktrees without leaving untracked files in main. The lead may create
that task-specific directory if permitted; do not change Git's internal files.
If permissions disallow it, use an allowed durable location outside the checkout
and report the absolute path. Record that path in progress/handoff updates.

Copy [state.example.json](state.example.json) for each new task; replace its example
agent/task entries. Only the lead updates the checkpoint. Record dependencies,
ownership, session IDs, current step, full SHAs, review decisions, validation output
locations, and the next action after meaningful changes. Never edit the template to
store a live run or reuse an old SHA as evidence for a new objective.

To recover:

1. Read repository policy and the checkpoint. Verify the former lead is stopped
   before taking ownership; identify still-running specialists and device users.
2. Inspect `git worktree list --porcelain`, `git branch -vv`, and each registered
   task checkout's `git status --short --branch` and `git log -5 --oneline`.
   Inspect only this repository's worktrees, and do not disturb unrelated tasks.
3. Reconcile recorded SHAs with actual branches/commits and merged ancestry. Git is
   the source of truth; a `completed` flag or a stale approval is not evidence.
4. Inspect any in-progress merge and unknown changes before continuing. Resume a
   known safe operation; do not reset, overwrite, or force-remove a checkout.
5. Contact existing specialists for targeted continuation rather than duplicating
   their work. Invalidate approvals if their tips moved. Reconfirm device ownership.
6. Resume at the first unverified dependency/review/validation step. Missing evidence
   means revalidation, not assumed success. Update the checkpoint and continue to
   final integration, or report the precise external blocker if unable to proceed.

Keep recovery records locally; commit durable outcome/validation documentation when
appropriate, without machine-specific paths, private transcripts, or temporary logs.
