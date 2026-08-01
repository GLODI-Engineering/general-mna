# Agent guidelines

`elspice-mna` is an educational Rust library that turns SPICE netlists into
modified nodal systems and state-space models. It also exposes Python bindings
and verified converter state-space-averaging reports.

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
