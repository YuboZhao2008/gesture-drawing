# Review and integration policy

The lead is the sole integration owner. A finished specialist assignment is a
candidate change, not a release decision.

## Gate for each specialist

SPECIALIST COMPLETE -> branch committed -> tree clean -> SHA captured -> lead
inspects diff -> targeted validation -> approve or request revision -> freeze
specialist -> merge onto integration branch -> combined tests -> next merge.

1. Verify the assigned worktree's status (including untracked files), branch, full
   HEAD SHA, and base. Require the specialist to stop editing at handoff.
2. Inspect `git log <base-sha>..<candidate-sha>` and
   `git diff <base-sha> <candidate-sha>`. Check scope, interface compatibility,
   failure tests, dependencies, configuration, and relevant benchmark evidence.
   Confirm required predecessor changes are present and identify unexpected commits.
3. Independently run targeted validation on the candidate. If requesting revision,
   specify the issue and acceptance test, revoke approval, and reopen ownership.
   A new commit requires a fresh handoff and review.
4. Record approval against the full SHA and confirm the owner is frozen. Before
   merging, recheck the branch tip equals that SHA and the worktree remains clean.
   Any difference invalidates approval; investigate rather than merging unseen work.
5. On a clean lead integration checkout, merge the approved SHA, for example
   `git merge --no-ff <approved-sha>`. Do not merge a moving branch name. Record the
   source and resulting integration SHAs. For an already-integrated ancestor,
   verify ancestry and record it without inventing a merge.
6. Run affected combined tests immediately. Complete the full primary checks from
   [validation.md](validation.md) on the final combined tree before promoting main.
   Tests run before later changes do not validate the final tree.

## Conflicts and failures

Never blindly choose ours/theirs. Inspect intent from both branches, the merge
base, and their callers. For shared files, preserve compatible changes, reconcile
configuration/dependencies and interfaces, and rerun relevant tests. Ask the
specialist for clarification directly if needed. The lead owns the resolution.

Stop the integration sequence on failure. Fix and commit the resolution, request a
targeted revision, or reject the candidate. If a candidate has already been merged,
exclude it using a reviewed normal revert commit (for a merge, inspect parents and
choose the correct mainline), then revalidate dependent changes. Do not reset main
or rewrite history. Inspect work in progress before aborting any merge; preserve
unknown edits and useful conflict-resolution work.

A specialist branch may remain unmerged if its changes are inferior or unsafe.
That is a successful orchestration outcome. Record the rejected SHA and reason,
inform its owner, and reconcile dependent tasks. Never force a merge just to mark
all agents complete. Keep a rejected branch available unless cleanup is authorized.

## Promotion and cleanup

Only the lead promotes the fully validated integration SHA to local main, normally
by fast-forward from main's registered clean checkout. Reconcile any intervening
main commits on integration and validate again. No specialist merges into main,
remote push, force push, or unrequested history rewriting.

Inspect `git worktree list --porcelain` before cleanup. Verify ownership, resolved
paths, clean status, and saved commits before `git worktree remove <path>`; never
use forced removal or recursive directory deletion on registered/nested worktrees.
Clean main does not require removing other registered worktrees. Preserve unrelated
work and report any condition preventing a clean final destination honestly.
