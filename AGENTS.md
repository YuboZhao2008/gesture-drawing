# Engineering agent policy

This is the authoritative repository engineering-agent policy. Follow applicable
platform instructions and explicit human task constraints. Read this file and
[the lead workflow](.agent/lead-orchestrator.md) before orchestrating work; read
[ARCHITECTURE.md](ARCHITECTURE.md) before changing implementation. More specific
directory instructions supplement this policy within their scope.

## Lead ownership

- Exactly one lead/integration owner is active at a time. Record its session ID
  and ownership in the task checkpoint; verify a previous lead has stopped before
  taking over. A checkpoint is a record, not an automatic lock.
- The lead owns the final repository state, task graph, supervision, review,
  revisions, conflict resolution, validation, and final reporting.
- Specialists never merge themselves into main. The lead independently reviews
  every specialist's actual diff and evidence before integration.
- Prefer native subagents when available. The lead communicates with specialists
  directly; the human must not serve as a message relay. Delegate only useful,
  bounded work, and retain integration authority even when using separate sessions.

## Branches and ownership

- During multi-agent work, main is integration/release only. Specialists never
  develop directly on main. Use task branches and isolated worktrees for parallel
  writers; the lead integrates approved commits one at a time.
- No force pushes, `reset --hard`, or history rewriting unless the human explicitly
  requests the relevant operation. No remote push unless explicitly requested.
- Assign bounded subsystem/file ownership without overlapping writers. Shared
  contracts, configuration, dependency declarations, and root documentation belong
  to the lead unless explicitly delegated to one owner.
- Specialists propose necessary shared-file changes to the lead; they must not
  silently edit unrelated modules or expand their scope.

## Worktrees and safety

- Record each assigned worktree's absolute path, branch, owner, base SHA, and status.
  Inspect `git worktree list --porcelain` before creating, moving, or removing one.
- Existing registered worktrees, including nested directories, are real checkouts,
  never disposable folders. Do not repurpose another session's worktree.
- Before removing a worktree, verify registration, resolved path, stopped owner,
  committed handoff, and clean status including untracked files. Use Git's worktree
  removal, without force; leaving a clean registered worktree is acceptable.
- Preserve unrelated user work. Inspect untracked files before any removal. Never
  erase, stash, overwrite, or commit unknown changes merely to obtain a clean tree.
- Do not access unrelated projects or unnecessarily change system/global settings.
  Use explicit repository/worktree paths for commands; respect sandbox permissions.

## Specialist completion and review

- Implementation specialists validate and commit their completed work, provide
  the full final SHA and test results, leave their worktree clean, and stop editing
  after handoff. Report-only specialists provide the reviewed SHA and findings;
  they need no empty commit.
- The lead inspects actual commits, architecture compatibility, tests and failure
  cases, relevant performance impact, new dependencies, and shared configuration.
- Passing specialist tests alone never grants merge approval. Approval applies
  only to the reviewed SHA; later changes invalidate it and require review again.
- The lead can request targeted revisions, fix issues during integration, or
  reject changes that regress behavior/performance, break architecture, introduce
  unnecessary complexity, or conflict with a better implementation.

## Validation and device access

- Follow [.agent/validation.md](.agent/validation.md): targeted checks first, full
  combined tests, lint, formatting, typing, and dependency health after integration.
  Do not weaken tests to obtain green output.
- Measure comparable before/after workloads for performance claims. Reject or
  tune regressions; distinguish throughput, software timing, and physical latency.
- The lead schedules exclusive webcam/device test windows. Never run multiple
  agents against the same device. Record owner, device, and release of the window.
  Distinguish automated synthetic checks from hardware and human physical tests.

## Finalization

- Follow [.agent/merge-policy.md](.agent/merge-policy.md). Before claiming completion:
  no unresolved conflicts, required validation passed on the final integrated tree,
  local main clean, final commit SHA known, limitations and unperformed checks
  reported honestly. Preserve unknown work even if it prevents completion.
- Honor an explicit task instruction to finish on another branch instead of main;
  report that branch and do not merge to main against that instruction.
- Do not substitute specialist reports for an integrated result. Record reviewed,
  merged, and rejected SHAs and validation evidence; document the final result.
