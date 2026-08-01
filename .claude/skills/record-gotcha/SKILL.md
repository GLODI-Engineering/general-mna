---
name: record-gotcha
description: Record a reproducible, non-obvious elspice-mna parser, matrix, binding, toolchain, or artifact-generation trap as institutional memory. Use when a discovery would cost a future contributor meaningful time to rediscover.
---

# Record an elspice-mna gotcha

Do not use this for a typo, an issue already explained by the current task, or
an unverified suspicion. Use it for a reproducible failure whose symptom does
not reveal its cause.

1. Read `docs/gotchas/INDEX.md` and avoid duplicates.
2. Reproduce the problem with the smallest circuit, netlist, command, or
   environment. Record expected and actual behavior.
3. Allocate the next three-digit `GOTCHA-NNN` identifier.
4. Copy `docs/gotchas/_TEMPLATE.md` to
   `docs/gotchas/GOTCHA-NNN-short-slug.md` and fill every field. Use the date
   from `date '+%Y-%m-%d'`, not conversational context.
5. State state-variable ordering, current direction, polarity, parameter units,
   and relevant tool versions when they affect reproduction.
6. Run `scripts/gotchas-index.sh` and commit the regenerated index with the
   gotcha.
7. Return to the original task. Recording a gotcha does not authorize a broader
   fix.

Keep one problem per document and keep it under roughly 300 lines. Append a
history item when status changes; do not erase the original discovery.
