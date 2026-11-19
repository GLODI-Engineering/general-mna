# Agent guidelines

`general-mna` is a published Rust crate that turns SPICE netlists into modified
nodal systems and state-space models. It also exposes Python bindings and
verified converter state-space-averaging reports.

## Authority order

1. Mechanical gates: hooks, formatters, Clippy, tests, and artifact validators.
2. Skills in `.claude/skills/<name>/SKILL.md` for recurring workflows.
3. This file.
4. Architecture and verification documents under `docs/`.

Never bypass a failing hook. Fix the cause.

## Project boundaries

- `spice-core` is a read-only path dependency from the sibling `spice-lsp`
  repository. Do not edit that repository unless the user explicitly expands
  the task to it.
- Keep parsing in `spice-core`; keep MNA stamping, reduction, averaging, and
  bindings here.
- Preserve the equation convention `A X + K dot(X) = B u`, with sources and
  their values represented explicitly by the assembled system.
- Build symbolic systems before numeric evaluation. Do not replace exact
  expressions with floating-point shortcuts in the educational API.
- State ordering, current direction, and output polarity are public contracts.
  Tests and generated reports must state them explicitly.

## Layout

```text
src/          symbolic MNA, numeric reduction, averaging, Python adapter
tests/        Rust integration tests
python/       PyO3 package and Python tests
scripts/      notebook, PDF, and validation tooling
notebooks/    generated educational converter notebooks
artifacts/    generated TeX/PDF reports
docs/         architecture, journal, gotchas, and verification records
```

## Development cycle

The gates below say how code is written. This says how the work is recorded, which matters for
the same reason: someone reads it later, and what they can reconstruct is what was written down.

1. **A large instruction becomes a milestone**, named for the outcome rather than the task
   ("Rules and diagnostics", not "implement the rules spec").
2. **One issue per deliverable phase or decision**, each stating the problem rather than the
   patch. The instruction as a whole — its reasoning, what it leaves open — is an issue in that
   milestone too, and the rest reference it with `Part of #N`.
3. **A branch and a pull request per issue**, the description ending `Closes #N`, assigned to the
   milestone. Nothing lands on the default branch directly.
4. **The merge closes the issue.** Closing it by hand loses the link between the issue and the
   commit that answered it, which is the whole point of the exercise.
5. **Anything found but not done becomes its own issue**, not a sentence in a pull request body.
   A merged pull request's prose is invisible; an open issue is not. A deliberate omission with
   a reason is worth more than a half-finished corner of a change.
6. **Gates run before pushing**, never after. A red branch spends someone's CI and tells the
   reader the author did not look.

Finish a repository's open pull request before starting new work in it, so review has one thing
to hold at a time.

## Required gates

Before committing, use the `commit` skill and run:

```bash
make test
```

When converter equations, bindings, notebook generation, or report generation
change, also run:

```bash
make artifacts
```

Use `verify-circuit-model` for any new circuit topology, stamp, state-space
reduction, transfer function, or averaging change. Numerical agreement alone is
not proof: derive the expected equations independently and test signs, units,
ordering, DC operating point, transfer coefficients, poles, and zeros.

## Skills

| Task | Skill |
|---|---|
| Gate, review, and commit changes | `.claude/skills/commit/SKILL.md` |
| Verify an MNA or converter model | `.claude/skills/verify-circuit-model/SKILL.md` |
| Preserve a costly reproducible discovery | `.claude/skills/record-gotcha/SKILL.md` |

Tools without project-skill discovery must open the matching `SKILL.md` and
follow it directly.

## Journal and gotchas

Read the newest top entry in `docs/journal/` when continuing prior work. Include
one dated journal entry with every commit, using a time verified with
`date '+%Y-%m-%d %H:%M'`. Keep entries newest-first and append history rather
than rewriting it.

Use a decision document for durable design rationale. Use `record-gotcha` for a
reproducible toolchain, parser, matrix, binding, or artifact-generation trap
that another contributor would otherwise have to rediscover.

## Commit rules

- Conventional Commits: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`,
  `test`, `build`, `ci`, `chore`, or `revert`.
- Review the complete diff and stage deliberately.
- Never use `--no-verify`, `SKIP=`, or similar bypasses.
- Never commit generated native libraries, caches, or transient LaTeX files.
