# elspice-mna

`elspice-mna` is an educational Rust library that converts a SPICE netlist
into the modified nodal analysis (MNA) descriptor equation

```text
A x(t) + K dx(t)/dt = B u(t)
```

It deliberately separates three jobs:

1. [`spice-core`](../spice-lsp/servers/core) parses ngspice or Xyce text.
2. This crate constructs symbolic MNA matrices and converter phase models.
3. A thin Python/PyO3 or JavaScript/WASM package can render PDFs, hand the
   expressions to SymPy, or drive an interactive state-space application.

The old Python ElSpice project was used as a behavioral reference. It is not
modified or imported by this repository.

## Implemented

- Symbolic stamps for `R`, `C`, `L`, `V`, `I`, `G` (VCCS), `E` (VCVS), `F`
  (CCCS), and `H` (CCVS).
- Mutual inductance with the SPICE relation `M = k*sqrt(L1*L2)`.
- Case-insensitive node, element, and controlling-source lookup.
- SPICE numeric suffixes (`k`, `meg`, `m`, `u`, `n`, `p`, and others).
- Explicit errors for unsupported nonlinear devices; they are never silently
  removed from a circuit.
- Name-based `Ron`/`Roff` switch overrides suitable for converter phases.
- Weighted descriptor averaging for any number of switching phases.
- The two-phase small-signal duty perturbation vector.
- Numeric Schur-complement reduction to `dot(x) = A_s*x + B_s*u` when the
  chosen MNA storage coordinates are independent.
- A string DTO that maps directly to Python dictionaries, JSON objects, or
  TypeScript interfaces without committing the core crate to one binding
  framework.

Nonlinear semiconductor linearization, subcircuit flattening, behavioral
sources, and symbolic Schur-complement reduction are intentionally future
work. The builder reports such devices instead of producing an incomplete
matrix.

## Basic use

```rust
use std::collections::BTreeMap;
use elspice_mna::MnaBuilder;
use spice_core::Dialect;

let netlist = "V1 in 0 1\nR1 in out R\nC1 out 0 C";
let mna = MnaBuilder::new(Dialect::Ngspice).build_fragment(netlist)?;

assert_eq!(mna.unknowns, ["V(in)", "V(out)", "I(V1)"]);

let values = BTreeMap::from([
    ("R".to_string(), 1_000.0),
    ("C".to_string(), 1e-6),
]);
let state_space = mna.evaluate(&values)?.to_state_space(1e-12)?;
// state_space.states == ["V(out)"]
# Ok::<(), Box<dyn std::error::Error>>(())
```

Run the complete example with:

```bash
cargo run --example rc_state_space
```

Use `build_document` instead of `build_fragment` for a normal SPICE file with
a mandatory title line.

## Converter averaging

Build each topology from the same parsed netlist and override switch names:

```rust
use elspice_mna::{
    average, BuildOptions, Expression, MnaBuilder, SwitchState, WeightedPhase,
};
use spice_core::Dialect;

let netlist = "V1 in 0 Vin\nS1 in sw ctrl 0 ideal\nL1 sw out L\nC1 out 0 C\nR1 out 0 R";

let mut on_options = BuildOptions::default();
on_options.set_switch("S1", SwitchState::On);
let on = MnaBuilder::with_options(Dialect::Ngspice, on_options)
    .build_fragment(netlist)?;

let mut off_options = BuildOptions::default();
off_options.set_switch("S1", SwitchState::Off);
let off = MnaBuilder::with_options(Dialect::Ngspice, off_options)
    .build_fragment(netlist)?;

let d = Expression::symbol("D");
let one_minus_d = Expression::one() - d.clone();
let averaged = average(&[
    WeightedPhase { system: &on, weight: &d },
    WeightedPhase { system: &off, weight: &one_minus_d },
])?;
# Ok::<(), Box<dyn std::error::Error>>(())
```

The override applies a resistor between the first two terminals of a named
device. This preserves one unknown vector across all phases, which is required
for averaging. It also allows a MOSFET instance to be treated as an idealized
educational switch without implementing a MOS compact model.

## Python, SymPy, and JavaScript

`MnaSystem::to_string_system()` returns row-major `A`, `K`, `B`, and expanded
`U` arrays plus unknown/input names. The expression syntax (`+`, `*`, `/`, and
`sqrt`) is accepted by SymPy and common JavaScript math-expression libraries.

A future PyO3 wrapper only needs to expose:

```text
build_fragment(source, dialect, switch_states) -> StringMnaSystem
```

The same DTO can be exported by `wasm-bindgen` for a browser application. Keep
symbolic matrix reduction in SymPy; use this crate's numeric reduction for an
interactive browser after parameter values are supplied.

## Development

The parser dependency is currently a sibling path so local parser work is used
immediately without editing either repository:

```toml
spice-core = { path = "../spice-lsp/servers/core" }
```

Once `spice-core` has a published version or stable Git tag, replace the path
with that immutable dependency for portable builds.

Quality gates:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

