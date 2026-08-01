---
name: verify-circuit-model
description: Verify an elspice-mna element stamp, circuit reduction, converter average, transfer function, or generated educational report against independently derived equations. Use for new devices, topologies, state-space outputs, averaging changes, Python parity, notebooks, or PDFs.
---

# Verify a circuit model

## Establish the contract

1. Read `AGENTS.md`, `docs/architecture.md`, relevant tests, and overlapping
   entries in `docs/gotchas/INDEX.md`.
2. Write down independently, before trusting program output:
   - node-voltage and branch-current directions;
   - state and input ordering;
   - physical output polarity;
   - parameter units and admissible operating region;
   - expected `A X + K dot(X) = B u` stamps.
3. For a switched converter, derive each phase equation, then compute
   `Abar = D Aon + (1-D) Aoff`, the DC point
   `X = -Abar^-1 Bgbar Vg`, and
   `Bd = (Aon-Aoff)X + (Bg_on-Bg_off)Vg`.

## Build executable evidence

1. Add a minimal Rust test for the symbolic stamp or system.
2. Add a numeric test using simple, dimensionally clear values.
3. Verify matrix dimensions, ordering, signs, DC gain, transfer coefficients,
   poles, and zeros. For boost-derived topologies, explicitly test any
   right-half-plane zero.
4. If the Python API exposes the result, add a parity assertion against Rust.
5. Prefer exact expressions for the reference derivation. Use tolerances only
   at the numeric boundary and explain their scale.

## Run the relevant gates

```bash
make test
```

If notebooks or reports show the model:

```bash
make artifacts
git diff -- notebooks artifacts
```

Review generated formulas and polarity visually; a successful generator is not
proof that the displayed mathematics is correct. Record assumptions and known
model limits such as ideal components and CCM in the report.
