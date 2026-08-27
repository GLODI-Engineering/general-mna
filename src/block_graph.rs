//! The block/signal-domain type vocabulary: `BlockKind`/`BlockInstance` (one named block and
//! what it computes), `Signal` (where a block's input comes from), `GateBinding` (how a
//! MOSFET's gate state resolves from a block's output), `ProbeTarget`/`PidClamp` (the two
//! `BlockKind` variants complex enough to need their own payload type). Moved here from
//! `dae-runtime` (Phase 2 of the format-unification plan — see `general-simulator`'s own
//! `docs/journal/`): `general-mna` is now the single place a netlist's *meaning* is built, both
//! electrical (`MnaSystem`, unchanged) and signal-domain (this module) — `dae-runtime` only
//! evaluates a graph of these types over time, it doesn't define what they are or parse them
//! out of source text anymore.
//!
//! Every block does one job and is meant to be wired to the others by the caller — an error
//! signal is a `Sum` block's output, a filtered-derivative PID compensator is a
//! `TransferFunction` given its own `N(s)/D(s)` coefficients, and a frequency-modulated PWM
//! carrier is `Pid -> Gain -> PhaseShiftPwm`, not a single function that bakes a specific
//! topology together.
//!
//! **There is no separate "closed-loop mode," and no non-block-driven gate at all** — every
//! [`GateBinding`] is `Block(name)`, reading a named block's current output (`>= 0.5` means
//! on), whether that block's own input chain traces back to a [`BlockKind::Probe`] of the
//! circuit's own state or is just a `Const` — both are resolved by exactly the same code, every
//! step, because from the solver's point of view they're the same kind of question: "what's
//! this device's terminal condition right now."
//!
//! This module defines *what these types are*; evaluating them over time (`BlockState`,
//! `evaluate_blocks`, `topological_order`, `simulate_transient_with_blocks`) stays in
//! `dae-runtime`, which imports everything here.

use std::collections::BTreeMap;

use continuous_blocks::{
    CoordinateTransform, Hysteresis, MathFn1, MathFn2, MathFn3, Pid, Pmsm, StateSpace,
    TransferFunction, Vco,
};

use crate::{SwitchState, TransientFunction};

/// Where a block's input value comes from: another block's output this same step, or that (or
/// any) block's own output from the *previous* step. There is deliberately **no** variant that
/// reads a circuit quantity (`V(node)`/`I(branch)`) directly — that crossing from the physical
/// domain into the signal domain must go through an explicit, named [`BlockKind::Probe`] block
/// instead (referenced afterward like any other block, via `Signal::Block`). See
/// [`BlockKind::Probe`]'s own doc comment for why this boundary is enforced rather than
/// implicit, the same way a reference tool/Simscape requires an explicit PS-a reference tool Converter block
/// between a physical port and a signal port instead of wiring them together directly.
#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    /// Another block's output *this* step — may name any block in the same slice, declared
    /// before or after this one: [`topological_order`] derives each step's actual evaluation
    /// order from the full `Signal::Block` dependency graph, not declaration position, so
    /// "before/after" in the source text no longer has to match causal order (see that
    /// function). A genuine same-step cycle among these edges (`A` depends on `B` depends on
    /// `A`, however indirectly, including a block naming itself) is a model error reported as
    /// `DaeError::AlgebraicLoop` before any step is solved — see [`BlockInstance`]'s own doc
    /// comment.
    Block(String),
    /// A named block's own output from the *previous* step (`0.0` before the first step,
    /// matching every dynamic block's own "starts at rest" convention). Unlike `Signal::Block`,
    /// this is not a same-step dependency at all — it reads state fixed before this step even
    /// starts — so it never contributes an edge to the dependency graph [`topological_order`]
    /// builds, and is consequently the *sanctioned* way to close what would otherwise be a
    /// same-step cycle (e.g. a current controller regulating a [`BlockKind::Pmsm`]'s own
    /// `id`/`iq` outputs, or a PLL's angle estimate feeding the very
    /// [`BlockKind::CoordinateTransform`] `Park` block that produced its own error signal): the
    /// one-sample delay every real digital controller reading its own last output already has.
    BlockPrev(String),
}

/// The value one named block produces (or one `Signal` resolves to) at a given step: a bare
/// scalar — everything this graph supported before this variant existed — or a fixed-length
/// vector of scalars. Arity is fixed once a block declares it (known at graph-build time from
/// its own parameters, e.g. a matrix `Gain`'s own row count, a `Const` vector literal's own
/// length, `StateSpace`'s own `C` row count) — there is no dynamic/runtime-determined length.
///
/// **No type tagging beyond `Scalar`/`Vector` exists, or is needed.** This graph has never had
/// a `bool` type, even for scalars — a "boolean" signal (a `Hysteresis` output, a PWM
/// `main`/`complement` output) has always just been an `f64` interpreted via a `>= 0.5`
/// threshold by whatever reads it. A `Vector`'s own elements are exactly as untyped as a scalar
/// signal already was; nothing tracked "this element is really a voltage" vs. "this element is
/// really a gate command" before, and nothing needs to start now.
///
/// Which `BlockKind`s accept/produce a `Vector`, and the exact broadcast/reduction/rejection
/// rule for each, is a per-block-kind decision worked out in
/// `general-simulator`'s own `book/dev-guide/src/vector-signals.md`.
#[derive(Debug, Clone, PartialEq)]
pub enum SignalValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

impl SignalValue {
    /// `1` for a `Scalar`, the element count for a `Vector`.
    pub fn len(&self) -> usize {
        match self {
            SignalValue::Scalar(_) => 1,
            SignalValue::Vector(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `Some(x)` for a `Scalar`, `None` for a `Vector` — the check every scalar-only `BlockKind`
    /// makes on each of its own inputs before doing anything else.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            SignalValue::Scalar(x) => Some(*x),
            SignalValue::Vector(_) => None,
        }
    }

    /// This value's own elements, in order, as a plain slice — a `Scalar` is a length-1 slice.
    /// The flattening primitive every input-bundling block (`cscript`, `statespace`,
    /// `CoordinateTransform`, `Pmsm`) builds its own input vector from: several `SignalValue`s,
    /// scalar or vector, concatenated in declared order.
    pub fn as_slice(&self) -> &[f64] {
        match self {
            SignalValue::Scalar(x) => std::slice::from_ref(x),
            SignalValue::Vector(v) => v,
        }
    }
}

/// What a [`BlockKind::Probe`] reads from the circuit's own previous-step operating point —
/// `V(node)` or `I(branch)`, anything [`OperatingPoint::value`] accepts, keyed by exactly the
/// same `V(...)`/`I(...)` naming convention `general-mna` itself uses for MNA unknowns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeTarget {
    Voltage(String),
    Current(String),
}

/// A [`BlockKind::Pid`]'s anti-windup bound: fixed at model-build time, or read fresh from the
/// graph every step. `Fixed` is every existing use of `Pid` before this variant existed —
/// unchanged behavior, one input (the error signal) as always. `Dynamic` is for a controller
/// whose *achievable* output range genuinely depends on other, still-evolving state (e.g. a
/// current-loop PID commanding a pole voltage that can't physically exceed roughly half the
/// DC bus voltage, itself still rising during a soft-start ramp) — a `Fixed` bound sized for
/// the final steady-state range is badly oversized early on, so the PID's own anti-windup never
/// engages even though the real plant is already saturated far below that fixed bound,
/// producing sustained, hard-to-diagnose windup-driven oscillation (worked example: an
/// `elspice-pwl-pfc-three-phase-vsc` experiment's own three-phase active-front-end current loop,
/// in the sibling `internal-archive` repo). `Dynamic` reads two *extra* inputs beyond
/// the error signal, in order `(clamp_lo, clamp_hi)`, evaluated fresh every step exactly like
/// any other block input — see [`BlockInstance`]'s own doc comment for the resulting input
/// count.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PidClamp {
    Fixed(f64, f64),
    Dynamic,
}

/// A [`BlockKind::Const`]'s own fixed value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    Scalar(f64),
    Vector(Vec<f64>),
}

/// A [`BlockKind::Gain`]'s own scale factor — a plain scalar (the only form before vector
/// signals existed, unchanged), or a fixed `M x N` matrix (row-major, `matrix[row][col]`) for a
/// genuine matrix-vector product.
#[derive(Debug, Clone, PartialEq)]
pub enum GainValue {
    Scalar(f64),
    Matrix(Vec<Vec<f64>>),
}

/// One block's behavior. `Const`/`Pwc`/`Pwl`/`Sin`/`Pulse`/`Exp`/`Sffm` are sources (zero
/// inputs); `Sum`/`Gain` are stateless (recomputed fresh from their inputs every step);
/// `Pid`/`StateSpace`/`TransferFunction`/`Vco` carry their own state forward across steps.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockKind {
    /// A fixed value, ignoring time — e.g. a nominal frequency or a fixed setpoint (`Scalar`),
    /// or a fixed constant vector (`Vector`, e.g. a per-element offset feeding a vector `Sum`).
    Const(ConstValue),
    /// The current step's own simulated time (seconds), zero inputs — the standard
    /// block-diagram "clock" source, needed to build a genuine `sin(2*pi*f*t)`-style time
    /// varying signal out of `MathFn1`/`Gain` blocks (there's otherwise no way for a block to
    /// see `t` directly; `Pwc`/`Pwl`'s own use of it is internal to those blocks alone).
    Time,
    /// A piecewise-**constant** function of time: the value from the last point at or before
    /// `t` (the first point's value for `t` before it). Used for reference schedules, including
    /// step tests (two points is a step at the second point's time). `repeat`, if `true`, wraps
    /// `t` into `[points[0].0, points.last().0)` (period = last time − first time) once `t`
    /// passes the last point, instead of holding flat forever — a periodic step/square-like
    /// waveform. **Not the same interpolation as [`BlockKind::Pwl`]** (this block is named
    /// `pwc` at the CLI level specifically to avoid the ambiguity a shared `pwl` name would
    /// create with the real piecewise-*linear* SPICE-matching source below — see that variant's
    /// own doc comment).
    Pwc {
        points: Vec<(f64, f64)>,
        repeat: bool,
    },
    /// A piecewise-**linear** function of time — real SPICE `PWL(t1 v1 t2 v2 ...)` semantics,
    /// linearly interpolated between breakpoints, held at the first/last point's value before/
    /// after the breakpoint range (matching `general_mna::TransientFunction::Pwl`'s own
    /// electrical-domain behavior exactly, so the same breakpoint list means the same waveform
    /// whether it drives a `V`/`I` source directly or a signal-domain reference through this
    /// block). `repeat`, if `true`, wraps `t` into `[points[0].0, points.last().0)` once past
    /// the last point instead of holding flat — the electrical-domain `PWL` source has no such
    /// option (SPICE's own repeat semantics aren't implemented there), so a genuinely periodic
    /// piecewise-linear waveform (a triangle/sawtooth reference, a repeating ramp) is only
    /// available here in the signal domain.
    Pwl {
        points: Vec<(f64, f64)>,
        repeat: bool,
    },
    /// One of the electrical domain's four other time-varying source forms
    /// (`general_mna::TransientFunction::Sin`/`Pulse`/`Exp`/`Sffm` — never `::Pwl`, which this
    /// module models as its own [`BlockKind::Pwl`] above instead, specifically to add the
    /// `repeat` option `TransientFunction` doesn't have), reused directly rather than
    /// reimplemented, so a `V`/`I` source and a signal-domain reference built from the same
    /// parameters produce bit-for-bit the same waveform. Zero inputs; evaluated via
    /// [`TransientFunction::value_at`] at the current step's own `t`.
    Waveform(TransientFunction),
    /// Weighted sum of its inputs, one sign per input (`+1.0`/`-1.0` for an error junction).
    Sum(Vec<f64>),
    /// Scales its single input. `Scalar(k)`: `Scalar -> Scalar` (`k*x`) or `Vector(N) ->
    /// Vector(N)` (every element scaled by `k`, broadcast). `Matrix(K)` (an `M x N` matrix):
    /// requires a `Vector` input of length exactly `N` (rejects a `Scalar`, rejects a `Vector`
    /// of the wrong length), output is `Vector(M)`, the matrix-vector product `K*x`.
    Gain(GainValue),
    /// A compiled PID with two-sided conditional-integration anti-windup against `clamp` — see
    /// [`crate::simulate_closed_loop`]'s doc comment for why two-sided anti-windup matters; the
    /// mechanism here is identical, just attached to this block instead of baked into a whole
    /// controller function. `clamp` is this PID's own notion of "my output is saturated,"
    /// independent of whatever downstream `Gain`/`Vco` blocks do to it after — same as a real
    /// PID block's own configured output limits. See [`PidClamp`] for the fixed-vs-dynamic
    /// choice and what it changes about this block's own input count.
    Pid { pid: Pid, clamp: PidClamp },
    /// An arbitrary continuous-time block given directly as its own `(A, B, C, D)` matrices — a
    /// compensator/filter that doesn't already have a named convenience constructor, e.g. a
    /// low-pass filter placed ahead of a `Pid` to damp a resonant plant. Genuinely MIMO: `B`'s
    /// own column count (`StateSpace::inputs()`) and `C`'s own row count (`StateSpace::
    /// outputs()`) are not constrained to `1` — a single-input single-output declaration (the
    /// only shape this block supported before vector signals existed) is simply the `1x1` case,
    /// unchanged. `outputs() == 1` produces a `SignalValue::Scalar`; `outputs() > 1` produces a
    /// `SignalValue::Vector` of that length. Inputs are the flattened concatenation of this
    /// block's own declared `Signal`s (scalar or vector, in order) — see
    /// `evaluate_blocks`'/`dae-runtime`'s own input-flattening convention, shared with
    /// `CoordinateTransform`/`Pmsm`/`CScript`. No anti-windup (that's specifically a `Pid`
    /// output's own concern, not every dynamic block's); stepped forward unconditionally every
    /// timestep via `StateSpace::rk4_step`.
    StateSpace(StateSpace),
    /// A single-input single-output block given as a rational `N(s)/D(s)` (numerator/
    /// denominator coefficients, highest-degree first) rather than a `Pid`'s `Kp`/`Ki`/`Kd`
    /// convenience parameterization — e.g. a hand-derived PID-with-filtered-derivative
    /// compensator (`C(s) = Kp + Ki/s + Kd*N*s/(s+N)`, put over one denominator first: a pure
    /// derivative term alone is non-causal/unrealizable, so every real PID, textbook or
    /// otherwise, filters it — see [`Pid::to_transfer_function`]'s own doc comment for the
    /// derivation). Compiled once via [`TransferFunction::to_state_space`]; no anti-windup, for
    /// the same reason `StateSpace` above has none.
    TransferFunction(TransferFunction),
    /// A voltage-controlled oscillator (see [`Vco`]) — a bare, standalone oscillator producing
    /// a `[0, 1)` ramp, still useful on its own (a raw frequency-to-ramp conversion for
    /// something other than gate control). Not how a gate-driving PWM modulator gets its own
    /// switching frequency, though — see [`BlockKind::Pwm`]/[`BlockKind::PhaseShiftPwm`] below,
    /// PWM Modulator 1/2, neither of which reads this block at all.
    Vco(Vco),
    /// **PWM Modulator 1**: fixed-frequency, duty-driven, **active-high complementary** PWM.
    /// One input, `duty` (`[0,1]`, clamped, read fresh every step from anywhere in the graph —
    /// a `Pid`, a filtered `TransferFunction`, a plain `Const`...), fixed carrier frequency
    /// `freq_hz`. Two outputs, following the [`BlockKind::CScript`] `output_names` convention:
    /// `output_names[0]` (aliasing this block's own `.name`) is the main signal, `output_names[1]`
    /// its active-high complement — the fusion of what used to be two separate `GateBinding`
    /// variants (`Pwm`/`PwmComplement`) into one component, per explicit request. `red`/`fed`
    /// (seconds) are independent per-edge dead-time delays — see
    /// [`math_ops::complementary_pwm_with_deadtime`] for the exact rising-edge-only-delay
    /// semantics and why `red=fed=0.0` recovers the ideal, gap-free, overlap-free pair exactly.
    /// Stateless: a pure function of `(t, duty)` every step, no internal oscillator.
    Pwm {
        freq_hz: f64,
        red: f64,
        fed: f64,
        output_names: Vec<String>,
    },
    /// **PWM Modulator 2**: frequency+phase+duty-driven, **active-high complementary** PWM —
    /// the fusion of what used to be two separate `GateBinding` variants (`Vco`/`VcoPhase`)
    /// plus a block-driven `duty` neither had, again with the same dead-time/complementary-
    /// output treatment as [`BlockKind::Pwm`]. This is *not* a variant of [`BlockKind::Vco`] —
    /// it owns its own frequency-integration state directly (`osc` reuses [`Vco`]'s own
    /// clamp-and-integrate math purely as an implementation detail, the same formula, not a
    /// shared block reference), so two instances fed the *same* `freq` input stay bit-for-bit
    /// phase-synchronized (deterministic integration, same `dt`, same starting phase `0.0`),
    /// the way e.g. a dual-active-bridge's two legs need to be, without a separately-declared
    /// shared oscillator block in between. Three inputs, in order: `freq` (Hz, clamped
    /// internally to `[osc.f_min, osc.f_max]`), `phase` (`[0,1)`, a phase-shift command as a
    /// fraction of one carrier period — *not* this block's own internal integration state, a
    /// different thing), `duty` (`[0,1]`, clamped). `red`/`fed` are in seconds, exactly like
    /// [`BlockKind::Pwm`]'s own, but converted to a phase fraction using *this step's own*
    /// resolved frequency (not a fixed constant) — this matters for a variable-frequency
    /// converter, since the same absolute dead time eats a larger fraction of the period at
    /// higher switching frequency, a real effect on e.g. a resonant converter's own ZVS margin,
    /// not just bookkeeping.
    PhaseShiftPwm {
        osc: Vco,
        red: f64,
        fed: f64,
        output_names: Vec<String>,
    },
    /// Multiplies all its inputs together (see [`math_ops::product`]).
    Product,
    /// Clamps its single input to `[-limit, limit]` (see [`math_ops::saturation`]).
    Saturation(f64),
    /// Linear interpolation through a fixed `(x, y)` table (see
    /// [`continuous_blocks::waveform_arithmetic::table`]) — a `table(x, a, b, c, d, ...)`-
    /// style lookup.
    Table(Vec<(f64, f64)>),
    /// One of the single-argument real waveform-arithmetic functions (`cos`, `sin`, `exp`,
    /// `sqrt`, ... — see [`continuous_blocks::waveform_arithmetic`] for the full list and what
    /// was deliberately left out).
    MathFn1(MathFn1),
    /// One of the two-argument real waveform-arithmetic functions (`atan2`, `hypot`, `pow`,
    /// `min`, `max`, ...).
    MathFn2(MathFn2),
    /// One of the three-argument real waveform-arithmetic functions (`if`, `limit`).
    MathFn3(MathFn3),
    /// A Schmitt-trigger comparator (see [`Hysteresis`]) — bang-bang/hysteresis-band control,
    /// used when there's no fixed switching frequency to modulate a duty command onto (unlike
    /// `Pid` feeding a [`BlockKind::Pwm`]). Its output is `1.0`/`0.0`, read directly by a
    /// [`GateBinding::Block`] rather than compared against a carrier.
    Hysteresis(Hysteresis),
    /// A dynamically-loaded, user-supplied block (see [`cscript_ffi`]): `lib` is a precompiled
    /// shared library exporting `cscript_start`/`cscript_output`/(optionally)`cscript_free`/
    /// `cscript_clone`, filling `output_names.len()` outputs. Unlike every other `BlockKind`,
    /// this one can carry state no Rust type here knows anything about — see [`cscript_ffi`]'s
    /// own module doc comment for the full C-side contract and why [`TimeStep::Adaptive`]
    /// requires `cscript_clone` to be exported.
    ///
    /// `sample_time`, if given, makes this block run on its *own* fixed-period sample grid
    /// (like a discrete controller block with a configured `Ts` in any block-diagram tool),
    /// independent of the circuit's own resolved step size: `cscript_output` is only actually
    /// called once accumulated time since the last call reaches `sample_time`, and this block's
    /// output holds its last value (zero-order hold) on every step in between — the right model
    /// for something like a fixed-frequency digital controller, which genuinely does not run at
    /// the power stage's own (much finer, and possibly adaptive/irregular) step rate. `None`
    /// (the default) calls `cscript_output` every resolved circuit step instead, passing that
    /// step's own `dt` — the right choice for a block meant to behave continuously.
    ///
    /// `xc_count`, if nonzero, declares this block as owning that many continuous states the
    /// *solver itself* numerically integrates (one independent RK4 per block, the same
    /// convention every other dynamic `BlockKind` here uses), as opposed to a block hand-
    /// integrating its own state inside `cscript_output` (still the right choice for a fixed-
    /// rate discrete recursion, e.g. a bilinear-transform-derived filter — see `cscript_ffi`'s
    /// own module doc comment, "The optional continuous-state (`xc`) contract," for exactly
    /// which case this is for). `0` (the default) is the plain, single-function contract
    /// unchanged from before this field existed: `lib` must export `cscript_start`/
    /// `cscript_output`/(optionally) `cscript_free`/`cscript_clone`. Nonzero requires `lib` to
    /// export `cscript_start`/`cscript_derivative`/`cscript_output_xc`/(optionally)
    /// `cscript_free`/`cscript_clone` *instead of* `cscript_output`.
    CScript {
        lib: std::path::PathBuf,
        output_names: Vec<String>,
        sample_time: Option<f64>,
        xc_count: usize,
    },
    /// A dynamically-loaded, user-supplied Python block (see `pyblock_ffi`) — the Python-hosted
    /// counterpart to [`BlockKind::CScript`]: an escape hatch to a full scripting language
    /// (including `numpy`/`scipy`) when the existing block library doesn't cover something.
    /// `path` is a `.py` file
    /// exporting `start`/`output` (or, if `xc_count > 0`, `start`/`derivative`/`output_xc`
    /// *instead of* `output` — the same continuous-state split `CScript`'s own `xc_count` has,
    /// for the same reason). `output_names`/`sample_time`/`xc_count` all have exactly the same
    /// meaning as `CScript`'s own fields — deliberately scalar-only outputs, the same zero-
    /// order-hold `sample_time` convention, the same solver-integrated-vs-hand-managed state
    /// split. See `general-simulator`'s own `book/dev-guide/src/python-blocks.md` for the full
    /// design and the measurements behind it.
    PyBlock {
        path: std::path::PathBuf,
        output_names: Vec<String>,
        sample_time: Option<f64>,
        xc_count: usize,
    },
    /// A plain, stateless, positionally-called Python function — a genuinely separate contract
    /// from [`BlockKind::PyBlock`] above (not a mode of it): no `start`, no persistent `state`,
    /// no `t`/`dt` boilerplate, just `function` (a name inside `path`'s own `.py` file) called
    /// with each declared `inputs=` entry as its own positional argument (`f(*inputs)`, never
    /// bundled into one list), returning a single value or a tuple/list matching `output_names`.
    /// The closest equivalent to a plain named-input/named-output function block in other
    /// block-diagram tools. `output_names`/`sample_time` mean the same as `PyBlock`'s own (still
    /// scalar-only outputs, still the same zero-order-hold convention) — there is no `xc_count`
    /// here at all, since a purely stateless function has nothing for a continuous state to mean.
    /// See `pyblock_ffi::PyFunctionInstance`'s own module doc comment for the full contract.
    PyFunction {
        path: std::path::PathBuf,
        function: String,
        output_names: Vec<String>,
        sample_time: Option<f64>,
    },
    /// One of the six Clarke/Park coordinate transforms (see
    /// [`continuous_blocks::CoordinateTransform`]) — the standard `abc`/`alpha-beta-0`/`d-q-0`
    /// change of basis used to regulate a three-phase quantity (grid-tied PFC, motor drive) with
    /// a `Pid` on a DC-like `d`/`q` value instead of chasing a sine wave directly. Stateless and
    /// multi-output, following exactly the same `output_names` convention [`BlockKind::CScript`]
    /// established: `inputs` supplies `kind.input_count()` values in the order
    /// [`CoordinateTransform::call`] expects, `output_names.len()` must equal 3 (this family's
    /// output count — see [`CoordinateTransform::output_names`] for the conventional per-
    /// transform names, e.g. `["alpha", "beta", "zero"]` for `Clarke`), the block's own `.name`
    /// binds to the first (primary) output, and the remaining two are inserted under their own
    /// `output_names` entries so a downstream block can reference them directly via
    /// `Signal::Block(name)`.
    CoordinateTransform {
        kind: CoordinateTransform,
        output_names: Vec<String>,
    },
    /// A permanent-magnet synchronous motor (see [`continuous_blocks::Pmsm`]) — genuinely
    /// nonlinear (bilinear speed/current coupling), so like [`BlockKind::Vco`] it carries its
    /// own state and is integrated via its own `step()` (RK4) rather than compiled to a
    /// [`StateSpace`]. Three inputs, in order: `vd`, `vq` (rotor-frame stator voltage commands,
    /// V — typically a `ClarkeParkInv`'s output, or a `CoordinateTransform` intermediate wired
    /// through a `Pid`), and `t_load` (N*m, the mechanical load torque). Four outputs, same
    /// `output_names` convention as [`BlockKind::CoordinateTransform`]/[`BlockKind::CScript`]
    /// (`output_names.len()` must be 4, the block's own name aliases the first/primary output):
    /// `id`, `iq` (A), `omega_m` (mechanical speed, rad/s), and `theta_e` (electrical angle,
    /// already wrapped to `[0, 2*pi)` via [`continuous_blocks::Pmsm::theta_e_wrapped`] — ready
    /// to feed a [`BlockKind::CoordinateTransform`] `Park`/`ClarkePark` block directly). Starts
    /// at rest (`id = iq = omega_m = theta_e = 0`) — no initial-condition override, matching
    /// every other dynamic block in this graph.
    Pmsm {
        pmsm: Pmsm,
        output_names: Vec<String>,
    },
    /// The **PS-to-Signal** converter: the *only* way a circuit quantity (`V(node)`/
    /// `I(branch)`) enters the signal domain. Zero block-graph inputs (it reads the circuit's
    /// own previous-step operating point directly, the same `point_prev` lookup a bare
    /// `meas:`-style reference used to do before this was enforced) — its value is then an
    /// ordinary block output, read by any downstream block via `Signal::Block(this_block's_name)`
    /// exactly like any other source block (`Const`/`Pwl`/`Time`). Modeled directly on
    /// a reference tool/Simscape's own PS-a reference tool Converter: a physical port and a signal port are
    /// type-distinct there and cannot be wired together without one of these in between: this
    /// is the same rule, enforced the same way, at the netlist level instead of a GUI's wiring
    /// canvas (a future UI enforcing the same rule visually is the intended companion, not a
    /// replacement for this).
    Probe(ProbeTarget),
    /// The **Signal-to-PS** converter for a discrete physical actuation: the *only* legal
    /// target for a [`GateBinding::Block`]'s own named block — `dae-runtime` rejects a
    /// `GateBinding` naming anything else with `DaeError::GateTargetNotSig2Gate`. Purely an
    /// identity pass-through numerically (`value =
    /// input`); its entire purpose is marking, at the netlist level, exactly where a signal
    /// stops being "just a number a controller computed" and starts being "a command that
    /// actuates a physical switch" — the discrete-actuation counterpart to
    /// [`BlockKind::Sig2Voltage`]/[`BlockKind::Sig2Current`]'s continuous case below. One input.
    Sig2Gate,
    /// The **Signal-to-PS** converter for a continuous quantity, closing the write-direction
    /// gap [`BlockKind::Probe`] doesn't (a probe only ever reads): the *only* legal way a
    /// signal-domain block's output drives an independent voltage source's own magnitude. A `V`
    /// element's own literal value field in the netlist names this block directly (e.g. `V1 a 0
    /// VDRV`, where `VDRV` is a declared `Sig2Voltage` block) — `general-mna`'s own
    /// `Expression::parse_scalar` already accepts a bare symbol there with no change needed on
    /// that side; `dae-runtime` requires, at validation time, that any such symbol naming a
    /// declared block resolve to exactly this kind (see
    /// `DaeError::SourceNotSig2PhysicalConverter`), and every step, substitutes this block's own
    /// just-computed output value into the circuit solve in that symbol's place — a real,
    /// bidirectional physical/control coupling `Signal::Measure`'s read-only predecessor could
    /// never express (see `elspice-pwl-buck-dc-motor-cascade`'s own README for the concrete gap
    /// this closes: a block could observe a circuit's voltage but never load it). Purely an
    /// identity pass-through numerically, same as [`BlockKind::Sig2Gate`]; the type-distinct
    /// name is what the enforcement (and, later, a UI) keys on. One input.
    Sig2Voltage,
    /// The [`BlockKind::Sig2Voltage`] counterpart for an `I` (independent current source)
    /// element's own literal value field. One input.
    Sig2Current,
}

/// One named block instance and where its inputs (if any) come from.
/// `Const`/`Pwc`/`Pwl`/`Waveform`/`Probe` blocks must have zero inputs; `Sum`/`Product` need one
/// input per sign/factor; `Gain`/
/// `StateSpace`/`TransferFunction`/`Vco`/`Saturation`/`Table`/`MathFn1`/`Sig2Gate`/
/// `Sig2Voltage`/`Sig2Current` each need exactly one; `Pid` needs exactly one (the error signal)
/// when its `clamp` is `PidClamp::Fixed`, or exactly three (`error, clamp_lo, clamp_hi`, in that
/// order) when `PidClamp::Dynamic`; `MathFn2` needs two; `MathFn3` needs three;
/// `CoordinateTransform` needs `kind.input_count()` (3 for `Clarke`/`ClarkeInv`, 4 for the
/// others — see [`continuous_blocks::CoordinateTransform::input_count`]); `Pmsm` needs exactly
/// three (`vd`, `vq`, `t_load`, in that order). Evaluated once per step in the causal order
/// [`topological_order`] derives from the slice's own `Signal::Block` dependency graph — *not*
/// the order the slice happens to be given in; a `Signal::Block` input may name any block in
/// the same slice regardless of declared position (source blocks, naturally, need none, and a
/// genuine cycle among these edges is rejected as `DaeError::AlgebraicLoop` before any step
/// runs — see `topological_order`'s own doc comment for how).
#[derive(Debug, Clone, PartialEq)]
pub struct BlockInstance {
    pub name: String,
    pub kind: BlockKind,
    pub inputs: Vec<Signal>,
}

/// How one MOSFET's gate state is resolved, every step: always from a named block's current
/// output, on while it's `>= 0.5`. No non-block-driven variant exists — even a permanently-off
/// gate is an explicit `Const(0.0)` wired through a [`BlockKind::Sig2Gate`], the same as every
/// other gate — and no bare carrier-comparator variant exists either: that comparison now lives
/// entirely inside gate-driving `BlockKind`s themselves ([`BlockKind::Pwm`]/
/// [`BlockKind::PhaseShiftPwm`], or a hand-built chain of ordinary blocks), so `GateBinding` has
/// exactly one job — reading a number and thresholding it — regardless of what produced that
/// number: a modulator's own main/complement output, a [`BlockKind::Hysteresis`] block (no
/// carrier at all, event-driven bang-bang switching), or any other block a caller composes.
#[derive(Debug, Clone, PartialEq)]
pub enum GateBinding {
    /// On while the named block's current output is `>= 0.5`.
    Block(String),
}

impl GateBinding {
    /// The block this binding reads from.
    pub fn source_blocks(&self) -> [Option<&str>; 2] {
        match self {
            GateBinding::Block(name) => [Some(name), None],
        }
    }

    /// # Panics
    /// If the named block's own current value is a `SignalValue::Vector` — unreachable in
    /// practice, since the *only* legal `GateBinding` target is a `BlockKind::Sig2Gate`
    /// converter (enforced before any step runs), and `Sig2Gate` itself rejects a `Vector`
    /// input at evaluation time — there is no way for a validly-targeted gate to ever see one
    /// here. Also panics if `name` isn't in `outputs` at all, exactly as before this variant
    /// existed (equally unreachable, for the same "checked before any step runs" reason).
    pub fn resolve(&self, outputs: &BTreeMap<String, SignalValue>) -> SwitchState {
        let GateBinding::Block(name) = self;
        let value = outputs[name.as_str()].as_scalar().unwrap_or_else(|| {
            panic!(
                "gate '{name}' resolved to a vector signal -- unreachable, since Sig2Gate \
                 itself must already reject a vector input"
            )
        });
        if value >= 0.5 {
            SwitchState::On
        } else {
            SwitchState::Off
        }
    }
}
/// A short, human-readable name for a `BlockKind`, for error messages that need to say what a
/// mistargeted block actually is (e.g. `DaeError::GateTargetNotSig2Gate`/
/// `SourceNotSig2PhysicalConverter`) without dumping its full parameter set.
pub fn block_kind_name(kind: &BlockKind) -> &'static str {
    match kind {
        BlockKind::Const(_) => "const",
        BlockKind::Time => "time",
        BlockKind::Pwc { .. } => "pwc",
        BlockKind::Pwl { .. } => "pwl",
        BlockKind::Waveform(TransientFunction::Sin { .. }) => "sinwave",
        BlockKind::Waveform(TransientFunction::Pulse { .. }) => "pulsewave",
        BlockKind::Waveform(TransientFunction::Exp { .. }) => "expwave",
        BlockKind::Waveform(TransientFunction::Sffm { .. }) => "sffmwave",
        // Never actually constructed (general-simulator-cli only builds `Waveform` from
        // Sin/Pulse/Exp/Sffm — a Pwl-shaped waveform always goes through `BlockKind::Pwl`
        // above instead, since only that variant supports `repeat`), but `TransientFunction`
        // is a 5-variant enum so this match must still be exhaustive.
        BlockKind::Waveform(TransientFunction::Pwl(_)) => "waveform(pwl)",
        BlockKind::Sum(_) => "sum",
        BlockKind::Gain(_) => "gain",
        BlockKind::Pid { .. } => "pid",
        BlockKind::StateSpace(_) => "statespace",
        BlockKind::TransferFunction(_) => "tf",
        BlockKind::Vco(_) => "vco",
        BlockKind::Pwm { .. } => "pwm",
        BlockKind::PhaseShiftPwm { .. } => "pspwm",
        BlockKind::Product => "product",
        BlockKind::Saturation(_) => "saturation",
        BlockKind::Table(_) => "table",
        BlockKind::MathFn1(f) => f.name(),
        BlockKind::MathFn2(f) => f.name(),
        BlockKind::MathFn3(f) => f.name(),
        BlockKind::Hysteresis(_) => "hysteresis",
        BlockKind::CScript { .. } => "cscript",
        BlockKind::PyBlock { .. } => "pyblock",
        BlockKind::PyFunction { .. } => "pyfunc",
        BlockKind::CoordinateTransform { kind, .. } => kind.name(),
        BlockKind::Pmsm { .. } => "pmsm",
        BlockKind::Probe(_) => "probe",
        BlockKind::Sig2Gate => "sig2gate",
        BlockKind::Sig2Voltage => "sig2voltage",
        BlockKind::Sig2Current => "sig2current",
    }
}
