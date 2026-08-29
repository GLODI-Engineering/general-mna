# general-mna

`general-mna` is an educational Rust library that converts a SPICE netlist
into the modified nodal analysis (MNA) descriptor equation

```text
A x(t) + K dx(t)/dt = B u(t)
```

It deliberately separates three jobs:

1. [`general-spice-core`](../spice-lsp/servers/core) parses ngspice or Xyce text.
2. This crate constructs symbolic MNA matrices and converter phase models.
3. A thin Python/PyO3 or JavaScript/WASM package can render PDFs, hand the
   expressions to SymPy, or drive an interactive state-space application.

The old Python ElSpice project was used as a behavioral reference. It is not
modified or imported by this repository.

## Implemented

- Symbolic stamps for `R`, `C`, `L`, `V`, `I`, `G` (VCCS), `E` (VCVS), `F`
  (CCCS), and `H` (CCVS).
- Time-varying `V`/`I` sources — `SIN`/`PULSE`/`EXP`/`PWL`/`SFFM`, the same five forms common to
  both ngspice and Xyce (see `spice-lsp/docs/GRAMMAR.md`). This crate has no notion of "now": a
  source using one of these is stamped as a per-instance symbol (`Expression::symbol(name)`,
  the same pattern diode Norton currents already use below) rather than a baked literal, and
  `MnaSystem::transient_sources` exposes the parsed `TransientFunction` for the caller to
  evaluate at its own current `t` and supply via `evaluate()`'s `values` map every step.
- `D` (diode) as a companion-model conductance plus a Norton current source,
  both left as per-instance symbolic parameters (`{name}_G`, `{name}_Ioff`)
  rather than parsed from the netlist — this crate has no diode physics of
  its own; an external caller (e.g. a piecewise-linear or nonlinear circuit
  solver) supplies numeric values for those symbols via `evaluate()`,
  potentially different ones every call. See
  `NumericMnaSystem::input_values`/`::u` below.
- Mutual inductance with the SPICE relation `M = k*sqrt(L1*L2)`.
- Case-insensitive node, element, and controlling-source lookup.
- SPICE numeric suffixes (`k`, `meg`, `m`, `u`, `n`, `p`, and others).
- Explicit errors for devices with no linear-or-externally-parameterized
  stamp (e.g. a BJT); they are never silently removed from a circuit.
- Name-based `Ron`/`Roff` switch overrides suitable for converter phases.
- Weighted descriptor averaging for any number of switching phases.
- The two-phase small-signal duty perturbation vector.
- Numeric Schur-complement reduction to `dot(x) = A_s*x + B_s*u` when the
  chosen MNA storage coordinates are independent.
- A string DTO that maps directly to Python dictionaries, JSON objects, or
  TypeScript interfaces without committing the core crate to one binding
  framework.

Full nonlinear semiconductor device physics, subcircuit flattening,
behavioral sources, and symbolic Schur-complement reduction are intentionally
future work. The builder reports devices with no stamp at all instead of
producing an incomplete matrix.

### Numeric evaluation of source values (`NumericMnaSystem`)

`MnaSystem::evaluate()` numerically evaluates `a`, `k`, `b`, and now also each
input's own value (`input_values`) and the expanded right-hand side (`u`) —
this matters for a source whose numeric value genuinely changes between
`evaluate()` calls (a diode's `{name}_Ioff`, or `{name}_G`, resolved
differently every simulation timestep by an external solver) as opposed to a
netlist-declared `V`/`I` source whose literal value never changes after
parsing. Evaluating an input's value is best-effort: a symbol with no entry
in the `values` map (e.g. a duty ratio a caller only cares about symbolically
while evaluating unrelated matrices) yields `f64::NAN` for that input rather
than failing the whole `evaluate()` call, and contributes `0.0` (not `NAN`)
to every row of `u`.

## Basic use

```rust
use std::collections::BTreeMap;
use general_mna::MnaBuilder;
use general_spice_core::Dialect;

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

## Full system: electrical + block/signal-domain graph (`build_system`)

`MnaBuilder::build_fragment` above only sees electrical elements. A netlist may also declare
`kind=...` block/signal-domain statements (PID compensators, PWM modulators, logic, coordinate
transforms — see `spice-lsp/docs/GRAMMAR.md` §12) and `.subckt`/`X`-instance hierarchy. For
those, use `build_system` instead — it flattens hierarchy, builds the electrical `MnaSystem`,
and dispatches every block statement, all in one call:

```rust
use general_mna::{build_system};
use general_spice_core::Dialect;

let netlist = "V1 in 0 1\nR1 in out R\nC1 out 0 C\n\
     ERR kind=const value=0\nPID1 kind=pid in=ERR kp=1 ki=0 kd=0 n=1 clamp_lo=-1 clamp_hi=1";
let system = build_system(netlist, Dialect::Ngspice)?;
// system.mna            -> the same MnaSystem build_fragment produces
// system.ideal_diodes   -> named `D` instances (kind=ideal_diode, or no kind= at all)
// system.ideal_switches -> named ideal-switch instances (kind=ideal_switch)
// system.gates          -> each ideal switch's resolved gate-drive signal
// system.blocks         -> every `kind=...` instance, in declaration order
# Ok::<(), Box<dyn std::error::Error>>(())
```

This crate does not evaluate the block graph over time — [`block_graph`] only defines the type
vocabulary (`BlockKind`, `Signal`, `GateBinding`, ...); a sibling `dae-runtime` crate imports
these types to actually step them through a transient simulation alongside the electrical
solve. Every type and field there is documented — `cargo doc --open -p general-mna` and start
at the `block_graph` module.

## Converter averaging

Build each topology from the same parsed netlist and override switch names:

```rust
use general_mna::{
    average, BuildOptions, Expression, MnaBuilder, SwitchState, WeightedPhase,
};
use general_spice_core::Dialect;

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
for averaging. It also allows an ideal-switch instance to be treated as an
idealized educational switch without implementing a MOS compact model.

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

### Native Python binding

The optional PyO3 module exposes both symbolic MNA construction and native
numeric state-space reduction:

```bash
./python/build_binding.sh
PYTHONPATH=python python3 -c \
  "from elspice_mna import build_mna; print(build_mna('R1 1 0 1k'))"
```

For a packaged installation, `python/pyproject.toml` is configured for
Maturin and builds the same `elspice_mna._native` module.

## Verified converter examples

The Python layer uses the Rust binding to construct the exact ON and OFF MNA
topologies, reduces each descriptor system symbolically with SymPy, and then
performs state-space averaging. Separate textbook Kirchhoff equations provide
the verification oracle—generated artifacts are rejected if any symbolic
matrix or transfer-function residual is nonzero.

Generated notebooks:

- [`notebooks/buck_state_space_averaging.ipynb`](notebooks/buck_state_space_averaging.ipynb)
- [`notebooks/boost_state_space_averaging.ipynb`](notebooks/boost_state_space_averaging.ipynb)
- [`notebooks/buck_boost_state_space_averaging.ipynb`](notebooks/buck_boost_state_space_averaging.ipynb)

Generated reports:

- [`artifacts/buck_converter_state_space.pdf`](artifacts/buck_converter_state_space.pdf)
- [`artifacts/boost_converter_state_space.pdf`](artifacts/boost_converter_state_space.pdf)
- [`artifacts/buck_boost_converter_state_space.pdf`](artifacts/buck_boost_converter_state_space.pdf)

Regenerate and verify everything with:

```bash
make artifacts
make test
```

## Development

The parser dependency is currently a sibling path so local parser work is used
immediately without editing either repository:

```toml
general-spice-core = { path = "../spice-lsp/servers/core" }
```

Once `general-spice-core` has a published version or stable Git tag, replace the path
with that immutable dependency for portable builds.

Quality gates:

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test --all-targets
```

### Contributor workflow

Repository guidance is in [`AGENTS.md`](AGENTS.md). Install the file-hygiene,
Rust-formatting, Python-compilation, gotcha-index, and Conventional Commit hooks
once per clone:

```bash
pre-commit install --install-hooks
```

Recurring workflows live in [`.claude/skills`](.claude/skills), and the dated
continuity log lives in [`docs/journal`](docs/journal).
