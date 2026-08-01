---
name: commit
description: Run the complete elspice-mna quality gate, review and stage the diff deliberately, update the dated journal, and create a Conventional Commit. Use whenever committing changes in this repository.
---

# Commit elspice-mna changes

1. Read `AGENTS.md`, the newest journal entry, and `docs/gotchas/INDEX.md`.
2. Inspect `git status`, `git diff`, and any already-staged diff. Check for
   secrets, caches, native libraries, transient LaTeX files, debug output, and
   unrelated user changes.
3. Run the standard gate:

   ```bash
   make test
   ```

4. If equations, converter models, Python bindings, notebooks, PDFs, or their
   generators changed, also run:

   ```bash
   make artifacts
   git diff --exit-code -- notebooks artifacts
   ```

   A drift failure means generated outputs must be reviewed and committed with
   their source change; never hide it.
5. Run `pre-commit run --all-files`. Fix the cause of every failure.
6. Obtain the real time with `date '+%Y-%m-%d %H:%M'` and add a newest-first
   entry to `docs/journal/YYYY-MM.md` describing the request, findings, changes,
   verification, and remaining work. Include it in the same changeset so a
   journal-only follow-up does not create an infinite bookkeeping loop.
7. Review the final diff, then stage explicit paths.
8. Commit with `type(scope): imperative summary`. Allowed types are `feat`,
   `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`,
   and `revert`.
9. Confirm `git status --short` is empty. Never use `--no-verify` or `SKIP=`.
