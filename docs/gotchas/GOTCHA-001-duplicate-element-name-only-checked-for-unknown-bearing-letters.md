---
id: GOTCHA-001
discovered: 2026-11-18
discovered_by: GLODI Engineering
scope:
  - src/builder.rs
severity: high
status: fixed
reproducibility: full
tags: [netlist, naming, silent-divergence]
---

# GOTCHA-001 — A duplicate element name was rejected only when the name became an unknown

## Symptom

Two devices of one name in the same netlist. For some device letters this was a build error; for
the rest the deck built, solved, and returned **numerically correct voltages**, with no error and
no warning:

```text
V1 a 0 10
Rload a b 10
Rload b 0 20
```

$V(b) = 10 \cdot \frac{20}{10 + 20} = 6.667\ \mathrm{V}$ — the right answer for the circuit as
drawn, from a netlist that names two different resistors `Rload`.

Which letters rejected it looked arbitrary from the outside:

| duplicate | before this fix |
|---|---|
| two resistors, two capacitors | accepted silently |
| two inductors, two voltage sources, two current sources, two diodes | rejected |

## Reproduction

`MnaBuilder::new(Dialect::Ngspice).build_fragment(...)` on the deck above, in any version up to
and including 0.5.0. It returns `Ok`. Replacing both `Rload` cards with `L1` returns
`Err(DuplicateElement("L1"))`.

## Root cause

There was never a uniqueness check. The rejection was a **side effect** of building two maps
keyed by the normalised name:

- `UnknownIndex::add_branch` — entered by `V`, `L`, `E`, `H`, the letters that carry a branch
  current unknown;
- `input_by_element` — entered by `V`, `I` and `D`, the letters that own an input column.

A letter in neither map (`R`, `C`, `G`, `F`, `K`) had nothing to collide with, so nothing
objected.

Nothing then went wrong *numerically*, which is why it survived: stamping is positional. Each
card contributes its own conductance at its own node pair and never looks its own name up. What
breaks is every later reference **by name** — a controlling branch, a `.IC` target, a caller's
`switch_states` entry, a downstream rule or measurement card, a GUI highlight. Each of those
resolves one name to one of the two devices, chosen by iteration order, and the author is never
told the other one exists.

## Fix or workaround

Fixed in 0.5.1: one pass over every parsed element, before any of the maps are built, rejecting a
repeated case-insensitive name whatever the device letter. The error is the same
`BuildError::DuplicateElement` with the same message, so the letters that already failed fail
identically. The two now-redundant guards inside the input loop were removed; `add_branch` keeps
its own as a lower-layer invariant.

The check runs on the **flattened** statement list, so two instances of one `.subckt` are still
fine — their body elements are `X1.R1` and `X2.R1`.

No workaround was needed before the fix beyond not writing a name twice, which is precisely what
could not be detected.

## Prevention

`tests/mna.rs`, the block of tests under *Element names are unique, whatever the device letter* —
one per uncaught letter, one asserting the already-caught letters keep their exact message, one
asserting distinct names still build, and one pinning the check to the post-flattening list.

The general lesson, worth applying to any future namespace in this crate: **a uniqueness rule
enforced as a side effect of building an index only covers the keys that index happens to hold.**
Check the rule where it is stated, once, over everything it applies to.

## References

- Issue #30 — A duplicate element name is rejected only when the name becomes an unknown.
- Found while designing the simulator-to-GUI output contract, which needs a name to identify
  exactly one device.

## History

- 2026-11-18: Discovered while checking whether a netlist element name can identify a device for
  a downstream consumer. Fixed the same day.
