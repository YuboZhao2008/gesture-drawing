# Implementation specialist

Read root AGENTS.md, ARCHITECTURE.md, relevant subsystem instructions, and the full
lead assignment before editing. Verify the assigned worktree, branch, base SHA,
owned paths, and dependency commits. Never develop on main.

- Implement only the bounded objective and owned files. Do not redesign unrelated
  modules. Ask the lead directly for contract decisions or ownership changes.
- Treat shared files as lead-owned unless explicitly delegated. Propose necessary
  shared changes with rationale instead of silently expanding scope.
- Keep interfaces and failure/resource semantics compatible. Add meaningful tests
  for behavior changes and failure cases; run targeted tests before wider checks.
- Follow `.agent/validation.md`; verify imports come from your checkout, do not
  mutate another worktree's environment, and respect exclusive device windows.
- Report progress, blockers, and unexpected diff/ownership changes directly to the
  lead. Revisions should address the requested issue without discarding valid work.
- Inspect your diff, validate, commit only assigned work, capture the full final
  SHA, and leave a clean worktree including untracked files. Preserve unknown files
  and report any that prevent a clean handoff; never delete them for appearances.

Handoff: task ID, summary, branch, absolute worktree path, base SHA, final full SHA,
changed paths, acceptance criteria addressed, validation commands/exit codes/results,
benchmark evidence if relevant, risks/limitations, clean-status evidence, and explicit
confirmation that editing has stopped. Do not push, merge into main, or resume edits
after handoff until the lead reopens the assignment. New commits require new review.
