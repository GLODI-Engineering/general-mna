//! The block/signal-domain type vocabulary: `BlockKind`/`BlockInstance` (one named block and
//! what it computes), `Signal` (where a block's input comes from), `GateBinding` (how a
//! an ideal switch's gate state resolves from a block's output), `Phys2SigTarget`/`PidClamp` (the two
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
//! on), whether that block's own input chain traces back to a [`BlockKind::Phys2Sig`] of the
//! circuit's own state or is just a `Const` — both are resolved by exactly the same code, every
//! step, because from the solver's point of view they're the same kind of question: "what's
//! this device's terminal condition right now."
//!
//! This module defines *what these types are*; evaluating them over time (`BlockState`,
//! `evaluate_blocks`, `topological_order`, `simulate_transient_with_blocks`) stays in
//! `dae-runtime`, which imports everything here.

use std::collections::BTreeMap;

use continuous_blocks::{
    CoordinateTransform, DiscretePid, FlipFlopKind, Hysteresis, LatchPriority, LogicOp, MathFn1,
    MathFn2, MathFn3, Pid, Pmsm, StateSpace, TransferFunction, Vco,
};

use crate::{SwitchState, TransientFunction};

/// Where a block's input value comes from: another block's output this same step, or that (or
/// any) block's own output from the *previous* step. There is deliberately **no** variant that
/// reads a circuit quantity (`V(node)`/`I(branch)`) directly — that crossing from the physical
/// domain into the signal domain must go through an explicit, named [`BlockKind::Phys2Sig`] block
/// instead (referenced afterward like any other block, via `Signal::Block`). See
/// [`BlockKind::Phys2Sig`]'s own doc comment for why this boundary is enforced rather than
/// implicit, the same way a real block-diagram tool requires an explicit physical/signal
/// converter block between a physical port and a signal port instead of wiring them together
/// directly.
#[derive(Debug, Clone, PartialEq)]
pub enum Signal {
    /// Another block's output *this* step — may name any block in the same slice, declared
    /// before or after this one: `topological_order` derives each step's actual evaluation
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
    /// starts — so it never contributes an edge to the dependency graph `topological_order`
    /// (in `dae-runtime`) builds, and is consequently the *sanctioned* way to close what would otherwise be a
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
    /// A single number.
    Scalar(f64),
    /// A fixed-length bundle of numbers.
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

    /// `true` only for an empty `Vector` (a `Scalar` is always length 1).
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

/// What a [`BlockKind::Phys2Sig`] reads from the circuit's own previous-step operating point —
/// `V(node)` or `I(branch)`, anything `OperatingPoint::value` (in `dae-runtime`) accepts, keyed by exactly the
/// same `V(...)`/`I(...)` naming convention `general-mna` itself uses for MNA unknowns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Phys2SigTarget {
    /// `V(node)` — the named node's voltage.
    Voltage(String),
    /// `I(branch)` — the named branch's current.
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
    /// A fixed `(lo, hi)` anti-windup bound, set once at model-build time.
    Fixed(f64, f64),
    /// The anti-windup bound is read fresh from two extra block inputs every step.
    Dynamic,
}

/// A [`BlockKind::Const`]'s own fixed value.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstValue {
    /// A fixed number.
    Scalar(f64),
    /// A fixed vector.
    Vector(Vec<f64>),
}

/// A [`BlockKind::Gain`]'s own scale factor — a plain scalar (the only form before vector
/// signals existed, unchanged), or a fixed `M x N` matrix (row-major, `matrix[row][col]`) for a
/// genuine matrix-vector product.
#[derive(Debug, Clone, PartialEq)]
pub enum GainValue {
    /// A plain scale factor.
    Scalar(f64),
    /// A fixed `M x N` matrix, row-major (`matrix[row][col]`).
    Matrix(Vec<Vec<f64>>),
}

/// How a `CScript`/`PyBlock`/`PyFunction`/`OctFunc` block's own execution is scheduled, independent of
/// the circuit's own resolved step (see each `BlockKind` variant's own `sample_time` field).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SampleTimeSpec {
    /// A fixed sample period `period` (from `ts=`/`freq=`), with an optional phase `offset`
    /// (from `to=`, required to satisfy `0.0 <= offset < period`) delaying when the *first* hit
    /// lands. Every hit after the first is spaced by exactly `period`.
    ///
    /// **Default, when `to=` is omitted: `offset == 0.0`** — this preserves the exact
    /// pre-existing behavior every `ts=`/`freq=`-using netlist already had before this field
    /// could carry an offset at all: `dae-runtime` seeds its own per-instance accumulator so the
    /// *first* `evaluate_blocks` call a block participates in is always due, unconditionally,
    /// regardless of `period`'s own value (verified directly against `dae-runtime`'s own
    /// construction code, not assumed) — i.e. the pre-existing behavior already *was* "no
    /// offset," not "one period's delay" as it might look at first glance. An explicit `to=X`
    /// (`0 < X < period`) genuinely delays the first hit to `t=X` instead.
    Periodic {
        /// The fixed sample period.
        period: f64,
        /// Phase offset delaying the first hit; `0.0 <= offset < period`.
        offset: f64,
    },
    /// The block computes its own next execution time itself, every time it runs (`ts=variable`
    /// in the netlist) — for a block whose own next event is known analytically at runtime (a
    /// modulator's own scheduled switching instant) rather than fixed at parse time. Requires
    /// the backing `.so`/`.py` to export the corresponding optional next-sample-hit
    /// function/symbol; see `cscript_ffi`'s/`pyblock_ffi`'s own module doc comments. Not
    /// available for [`BlockKind::PyFunction`]/[`BlockKind::OctFunc`] (rejected at parse time) —
    /// a stateless function call has no instance to remember a requested next-hit time against.
    Variable,
}

/// One block's behavior. `Const`/`Pwc`/`Pwl`/`Sin`/`Pulse`/`Exp`/`Sffm` are sources (zero
/// inputs); `Sum`/`Gain` are stateless (recomputed fresh from their inputs every step);
/// `Pid`/`StateSpace`/`TransferFunction`/`Vco` carry their own state forward across steps.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockKind {
    /// <!-- component -->
    /// # Constant
    /// **Purpose:** a fixed reference value — a setpoint, bias, or offset that doesn't vary with
    /// time.
    /// **Library:** Sources
    ///
    /// ## Description
    /// A fixed value, independent of time: a scalar (`value=5`) or a fixed vector
    /// (`value=[1,2,3]`, a Python-list literal). No inputs, no state, nothing to evaluate per
    /// step. The vector form exists specifically to feed the vector-aware math blocks — a
    /// per-element offset into a vector `Sum`, or a matrix `Gain`'s input.
    ///
    /// ## Parameters
    /// - `value=<f64>` — a plain number (the scalar form).
    /// - `value=[x1,x2,...]` — a Python-style list literal (the vector form). Entries must be
    ///   finite: `nan`/`inf` are rejected at parse time, since a NaN would otherwise silently
    ///   propagate into downstream sorts/comparisons and panic the process.
    ///
    /// ## Errors
    /// - `value=abc` (scalar not a number) — rejected at parse time:
    ///   `field 'value' is not a number`.
    /// - `value=[1,x,3]` (vector entry not a number) — rejected at parse time:
    ///   `field 'value' entry 'x' is not a number`.
    /// - `value=[1,nan,3]` (non-finite vector entry) — rejected at parse time:
    ///   `field 'value' entry 'nan' must be a finite number (got NaN)`.
    /// - A missing `value=` is the generic `missing field 'value'` error every `kind=` block
    ///   shares — see [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=const value=<f64> | value=[x1,x2,...]
    /// ```
    ///
    /// ## Example
    /// A scalar setpoint and a constant vector offset, unchanged at every step:
    /// ```text
    /// SETPOINT kind=const value=5
    /// OFFSETS kind=const value=[1,2,3]
    /// ```
    Const(ConstValue),
    /// <!-- component -->
    /// # Time
    /// **Purpose:** the current simulated time as a signal — the block-diagram "clock" source.
    /// **Library:** Sources
    ///
    /// ## Description
    /// Outputs the current step's own simulated time in seconds. Takes no fields and no inputs.
    /// Its job is building genuine time-varying signals out of the stateless math blocks: there
    /// is otherwise no way for a block to see `t` directly (`pwc`/`pwl`/`waveform`'s own use of
    /// time is internal to those blocks alone). The standard `sin(2*pi*f*t)` recipe is
    /// `time -> gain(k=2*pi*f) -> sin` — see the Example.
    ///
    /// ## Parameters
    /// None — `kind=time` takes no fields at all.
    ///
    /// ## Errors
    /// None — there is nothing component-specific to get wrong beyond the generic
    /// `unknown device kind` error for a misspelled kind.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=time
    /// ```
    ///
    /// ## Example
    /// A 1 kHz sine wave built entirely from ordinary blocks (`6283.185307179586` is
    /// `2*pi*1000`):
    /// ```text
    /// T kind=time
    /// G kind=gain in=T k=6283.185307179586
    /// S kind=sin in=G
    /// ```
    Time,
    /// <!-- component -->
    /// # Piecewise-Constant Source (pwc)
    /// **Purpose:** a time schedule that holds a constant value between its breakpoints —
    /// reference schedules and step tests.
    /// **Library:** Sources
    ///
    /// ## Description
    /// A piecewise-**constant** function of time: the value is the last breakpoint's value at or
    /// before `t` (the first point's value before it, the last point's value forever after). Two
    /// points make a step test; more points make an arbitrary staircase schedule. With
    /// `repeat=true`, `t` wraps into `[points[0].0, points[last].0)` (period = last time − first
    /// time) once past the last point, making a periodic square/staircase waveform — a repeating
    /// list needs a final point back at the value the cycle starts from, since the wrapped `t`
    /// lands in `[first, last)` and never reaches the last point's own value.
    ///
    /// Not the same interpolation as [`BlockKind::Pwl`] (real SPICE piecewise-*linear*
    /// semantics): this block is named `pwc` at the CLI level specifically to avoid the
    /// ambiguity a shared `pwl` name would create — `pwc` holds between breakpoints, `pwl`
    /// interpolates.
    ///
    /// ## Parameters
    /// - `points=[[t1,v1],[t2,v2],...]` — `(time, value)` breakpoints as a Python list of
    ///   2-element lists. Sorted ascending by `t` at parse time, so declaration order doesn't
    ///   matter.
    /// - `repeat=true` — optional (default: hold flat past the last point); wraps `t` into the
    ///   breakpoint span as above. Matched exactly: any other spelling (including `True`) means
    ///   no repeat.
    ///
    /// ## Errors
    /// - A `points=` entry that is not a 2-element `[x,y]` list — rejected at parse time:
    ///   `field 'points': each entry must be a 2-element '[x,y]' list (got '[1.0]')`.
    /// - `points=abc` (not a Python list at all) — rejected at parse time:
    ///   `field 'points' must be a Python-style list, e.g. '[1,2,3]' or '[[1,2],[3,4]]' (got
    ///   'abc')`.
    /// - A missing `points=` is the generic `missing field 'points'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pwc points=[[t1,v1],[t2,v2],...] [repeat=true]
    /// ```
    ///
    /// ## Example
    /// A step test at `t = 0.001`, and a periodic square wave (period 0.001):
    /// ```text
    /// STEP kind=pwc points=[[0,0],[0.001,1]]
    /// SQUARE kind=pwc points=[[0,0],[0.0005,1],[0.001,0]] repeat=true
    /// ```
    Pwc {
        /// `(time, value)` breakpoints, in order.
        points: Vec<(f64, f64)>,
        /// If `true`, wraps `t` into the breakpoint range instead of holding flat past the last point.
        repeat: bool,
    },
    /// <!-- component -->
    /// # Piecewise-Linear Source (pwl)
    /// **Purpose:** a time waveform linearly interpolated between breakpoints — real SPICE PWL
    /// semantics in the signal domain.
    /// **Library:** Sources
    ///
    /// ## Description
    /// A piecewise-**linear** function of time: linear interpolation between breakpoints, held
    /// at the first/last point's value before/after the breakpoint range — exactly
    /// [`TransientFunction::Pwl`]'s own electrical-domain behavior, so the same breakpoint list
    /// means the same waveform whether it drives a `V`/`I` source directly or a signal-domain
    /// reference through this block. With `repeat=true`, `t` wraps into `[points[0].0,
    /// points[last].0)` (period = last time − first time) once past the last point — the
    /// electrical-domain `PWL` source has no such option, so a genuinely periodic
    /// piecewise-linear waveform (a triangle/sawtooth reference, a repeating ramp) is only
    /// available here in the signal domain. Not the same interpolation as [`BlockKind::Pwc`]
    /// (piecewise-*constant*): `pwl` interpolates, `pwc` holds.
    ///
    /// ## Parameters
    /// - `points=[[t1,v1],[t2,v2],...]` — `(time, value)` breakpoints as a Python list of
    ///   2-element lists. Sorted ascending by `t` at parse time, so declaration order doesn't
    ///   matter.
    /// - `repeat=true` — optional (default: hold flat past the last point); wraps `t` into the
    ///   breakpoint span as above. Matched exactly: any other spelling (including `True`) means
    ///   no repeat.
    ///
    /// ## Errors
    /// - A `points=` entry that is not a 2-element `[x,y]` list — rejected at parse time:
    ///   `field 'points': each entry must be a 2-element '[x,y]' list (got '[1.0]')`.
    /// - `points=` not a Python list at all — rejected at parse time with the same
    ///   `field 'points' must be a Python-style list, ...` error as `pwc` (shared parser — see
    ///   the pwc entry).
    /// - A missing `points=` is the generic `missing field 'points'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pwl points=[[t1,v1],[t2,v2],...] [repeat=true]
    /// ```
    ///
    /// ## Example
    /// A ramp to 1, and a periodic triangle (period 0.001):
    /// ```text
    /// RAMP kind=pwl points=[[0,0],[0.001,1]]
    /// TRI kind=pwl points=[[0,0],[0.0005,1],[0.001,0]] repeat=true
    /// ```
    Pwl {
        /// `(time, value)` breakpoints, in order.
        points: Vec<(f64, f64)>,
        /// If `true`, wraps `t` into the breakpoint range instead of holding flat past the last point.
        repeat: bool,
    },
    /// <!-- component -->
    /// # Waveform Sources (sinwave / pulsewave / expwave / sffmwave)
    /// **Purpose:** the electrical domain's four non-PWL time-varying source forms, reusable as
    /// zero-input signal references.
    /// **Library:** Sources
    ///
    /// ## Description
    /// One of [`TransientFunction`]'s four non-`Pwl` variants, reused directly rather than
    /// reimplemented: `kind=sinwave`/`pulsewave`/`expwave`/`sffmwave` take the same field
    /// names, order and defaults as SPICE's `SIN()`/`PULSE()`/`EXP()`/`SFFM()` sources, so a
    /// `V`/`I` source and a signal-domain reference built from the same parameters produce
    /// bit-for-bit the same waveform (evaluated via [`TransientFunction::value_at`] at the
    /// current step's own `t`). See [`TransientFunction`]'s own doc comment for the exact
    /// formula each form implements (offset + damped sinusoid, periodic trapezoid, two-stage
    /// exponential, single-frequency FM). The fifth form, `PWL`, is its own block kind here
    /// ([`BlockKind::Pwl`]) — the only member of the family with the `repeat` option
    /// [`TransientFunction`] doesn't have.
    ///
    /// ## Parameters
    /// - `sinwave`: `va=<f64>` `freq=<f64>` required (amplitude; frequency in Hz). Optional:
    ///   `v0=<f64>` (0), `td=<f64>` (0), `theta=<f64>` (0, exponential damping), `phase=<f64>`
    ///   (0, degrees).
    /// - `pulsewave`: `v1=<f64>` `v2=<f64>` required (low/high levels). Optional: `td` (0),
    ///   `tr` (0), `tf` (0), `pw` (default: effectively forever), `per` (effectively forever).
    ///   `tr`/`tf` of exactly 0 are instantaneous edges.
    /// - `expwave`: `v1=<f64>` `v2=<f64>` required. Optional: `td1` (0), `tau1` (1), `td2`
    ///   (effectively never — not `td1`), `tau2` (1).
    /// - `sffmwave`: `va=<f64>` `fc=<f64>` `fs=<f64>` required (carrier amplitude, carrier
    ///   frequency, signal frequency, all Hz/amplitude as in SPICE). Optional: `v0` (0),
    ///   `mdi` (0, modulation index).
    /// - An *optional* field that is present but doesn't parse as a number **silently falls back
    ///   to its default** — `td=abc` behaves exactly like an omitted `td`, with no error.
    ///
    /// ## Errors
    /// - A required field missing (`sinwave` without `va`) — the generic `missing field 'va'`
    ///   error (see [`BlockInstance`]'s own doc comment).
    /// - A required field not a number (`va=abc`) — rejected at parse time:
    ///   `field 'va' is not a number`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=sinwave    va=<f64> freq=<f64> [v0=<f64>] [td=<f64>] [theta=<f64>] [phase=<f64>]
    /// NAME kind=pulsewave  v1=<f64> v2=<f64> [td=<f64>] [tr=<f64>] [tf=<f64>] [pw=<f64>] [per=<f64>]
    /// NAME kind=expwave    v1=<f64> v2=<f64> [td1=<f64>] [tau1=<f64>] [td2=<f64>] [tau2=<f64>]
    /// NAME kind=sffmwave   va=<f64> fc=<f64> fs=<f64> [v0=<f64>] [mdi=<f64>]
    /// ```
    ///
    /// ## Example
    /// A 1 kHz sinusoid, a delayed 1 kHz square wave, an exponential rise/fall, and an
    /// unmodulated FM carrier, all as signal-domain references:
    /// ```text
    /// SINREF kind=sinwave va=10 freq=1000
    /// PULSEREF kind=pulsewave v1=0 v2=10 td=0.001 tr=0.0001 tf=0.0001 pw=0.0002 per=0.001
    /// EXPREF kind=expwave v1=0 v2=1 td1=0 tau1=0.0001 td2=0.001 tau2=0.0001
    /// FMREF kind=sffmwave v0=0 va=1 fc=1000 mdi=0 fs=100
    /// ```
    Waveform(TransientFunction),
    /// <!-- component -->
    /// # Sum
    /// **Purpose:** weighted sum of its inputs, one sign per input — e.g. an error junction.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// Computes $s_1 x_1 + s_2 x_2 + \dots$ with one sign per input (`+1.0`/`-1.0` for an
    /// error junction; any number is allowed, so a weighted average is the same block with
    /// fractional signs). Input-shape rule (shared with `Product`): all-`Scalar` inputs give a
    /// `Scalar`; otherwise every input must be a `Vector` of one common length `N`, reduced
    /// elementwise to a `Vector(N)` — a mix of `Scalar` and `Vector` inputs is rejected rather
    /// than guessed at (an N-input reduction has no unambiguous placement for a lone scalar
    /// once more than one vector is already present).
    ///
    /// ## Parameters
    /// - `inputs=<sig1,sig2,...>` — comma-separated input signals, one per term.
    /// - `signs=<s1,s2,...>` — one sign (any number) per input, comma-separated, exactly as many
    ///   entries as `inputs` has.
    ///
    /// ## Errors
    /// - `inputs` and `signs` counts differ — rejected at parse time:
    ///   `'inputs' has 2 entries but 'signs' has 1 (need one sign per input)`.
    /// - A `signs` entry that is not a number — rejected at parse time:
    ///   `field 'signs' entry 'x' is not a number`.
    /// - Mixed `Scalar`/`Vector` inputs — rejected at evaluation time:
    ///   `VectorSignalNotSupported { block: "<name>" }`.
    /// - `Vector` inputs of differing lengths — rejected at evaluation time:
    ///   `VectorSignalSizeMismatch { block: "<name>", expected: N, got: M }`.
    /// - A missing `inputs=`/`signs=` is the generic `missing field '<key>'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=sum inputs=<sig1,sig2,...> signs=<s1,s2,...>
    /// ```
    ///
    /// ## Example
    /// An error junction (`A - B`) and an elementwise vector sum:
    /// ```text
    /// A kind=const value=3
    /// B kind=const value=1
    /// ERR kind=sum inputs=A,B signs=1,-1
    /// OA kind=const value=[1,2]
    /// OB kind=const value=[10,20]
    /// VSUM kind=sum inputs=OA,OB signs=1,1
    /// ```
    Sum(Vec<f64>),
    /// <!-- component -->
    /// # Gain
    /// **Purpose:** scales its single input — a plain factor, or a full matrix-vector product.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// `k` as a plain number (`Scalar`): `Scalar -> Scalar` (`k*x`), or `Vector(N) ->
    /// Vector(N)` (every element scaled by `k`, broadcast). `k` as a matrix (`[[...],[...]]`,
    /// an `M x N` matrix, row-major `matrix[row][col]`): requires a `Vector` input of length
    /// exactly `N`, and outputs `Vector(M)`, the matrix-vector product $K\,x$ — the one math
    /// block that changes a signal's length. A flat list (`k=[1,2,3]`) is not a valid `Gain`
    /// shape at all (unlike `Const`, `Gain`'s own value is never itself a vector) and is
    /// rejected at parse time.
    ///
    /// ## Parameters
    /// - `in=<signal>` — the single input.
    /// - `k=<f64>` — a plain scale factor.
    /// - `k=[[r1...],[r2...],...]` — an `M x N` matrix (Python-style nested list, row-major);
    ///   the input must be a `Vector` of length `N`, the output is `Vector(M)`.
    ///
    /// ## Errors
    /// - `k=[1,2,3]` (flat list) — rejected at parse time: `field 'k' must be a scalar (e.g.
    ///   '2.0') or a matrix (e.g. '[[1,0],[0,1]]') -- got a flat list '[1,2,3]', which is not a
    ///   valid Gain shape`.
    /// - `k=abc` — rejected at parse time: `field 'k' is not a number`.
    /// - A matrix `k` with a `Scalar` input — rejected at evaluation time:
    ///   `VectorSignalNotSupported { block: "<name>" }`.
    /// - A matrix `k` with a `Vector` input whose length isn't `N` — rejected at evaluation
    ///   time: `VectorSignalSizeMismatch { block: "<name>", expected: N, got: M }`.
    /// - A missing `in=`/`k=` is the generic `missing field '<key>'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=gain in=<signal> k=<f64> | k=[[r1...],[r2...],...]
    /// ```
    ///
    /// ## Example
    /// A scalar gain and a 2x3 matrix gain fed by a length-3 vector:
    /// ```text
    /// IN kind=const value=2
    /// G kind=gain in=IN k=3
    /// V kind=const value=[1,2,3]
    /// M kind=gain in=V k=[[1,0,0],[0,1,0]]
    /// ```
    Gain(GainValue),
    /// <!-- component -->
    /// # PID Controller
    /// **Purpose:** closed-loop error compensation with a filtered derivative and anti-windup.
    /// **Library:** Control / Continuous
    ///
    /// ## Description
    /// A parallel PID ($C(s) = K_p + \dfrac{K_i}{s} + \dfrac{K_d N s}{s + N}$, filtered-derivative
    /// form — see `continuous_blocks::Pid`'s own doc comment for the transfer-function
    /// derivation) with
    /// two-sided conditional-integration anti-windup against `clamp`: while the compensator's
    /// raw (unclamped) output would exceed `clamp`'s bound, the integral term stops
    /// accumulating in the direction that would make the saturation worse, but keeps
    /// integrating normally in the direction that would relieve it — so the controller recovers
    /// immediately once the error's sign allows it to, instead of first having to "unwind" a
    /// runaway integral term. `clamp` is this PID's own notion of "my output is saturated,"
    /// independent of whatever downstream `Gain`/`Vco` block does to the value afterward — the
    /// same role a real PID block's own configured output limits play.
    ///
    /// See [`PidClamp`] for the fixed-vs-dynamic choice, which also changes this block's own
    /// input count (one, just the error signal, for `Fixed`; three — error, `clamp_lo`,
    /// `clamp_hi` — for `Dynamic`). Reach for `Dynamic` specifically when the achievable output
    /// range is itself state-dependent (e.g. a current-loop PID commanding a pole voltage that
    /// can't physically exceed roughly half a DC bus voltage that is itself still rising during
    /// a soft-start ramp) — a `Fixed` bound sized for the final steady-state range is badly
    /// oversized early on, so anti-windup never engages even though the real plant is already
    /// saturated far below that fixed bound, producing sustained, hard-to-diagnose
    /// windup-driven oscillation.
    ///
    /// ## Parameters
    /// - `in=<signal>` — the error signal (one input).
    /// - `kp=<f64>`, `ki=<f64>`, `kd=<f64>` — proportional/integral/derivative gains. No sign or
    ///   magnitude restriction (a negative gain is a legitimate, if unusual, choice depending on
    ///   the plant's own sign convention).
    /// - `n=<f64>` — the derivative filter coefficient (the filter pole sits at `-n`); must be
    ///   `> 0`. A larger `n` tracks the ideal (unrealizable) pure derivative more closely, at the
    ///   cost of stiffening the resulting ODE.
    /// - Anti-windup bound, exactly one of:
    ///   - `clamp_lo=<f64> clamp_hi=<f64>` — fixed bound ([`PidClamp::Fixed`]).
    ///   - `clamp_lo_in=<signal> clamp_hi_in=<signal>` — read fresh from the graph every step
    ///     ([`PidClamp::Dynamic`]), added as two extra inputs after the error signal, in that
    ///     order.
    /// - `ic=<f64>` — optional (default: start from rest); pre-loads the integrator so the
    ///   controller output starts at this value with zero error — a bumpless start at a known
    ///   operating point. Resolved through `continuous_blocks::TransferFunction::settled_state`
    ///   on this PID's own transfer function.
    ///
    /// ## Errors
    /// - `n <= 0.0` — rejected at parse time (`invalid PID (NonPositiveFilterCoefficient)`): a
    ///   non-positive filter coefficient puts the derivative filter's pole at the origin or in
    ///   the right half-plane, a non-causal or outright unstable filter.
    /// - Giving only one of `clamp_lo_in`/`clamp_hi_in` (not both, and not neither) — rejected at
    ///   parse time: `'clamp_lo_in'/'clamp_hi_in' must both be given together (dynamic clamp) or
    ///   both omitted (fixed clamp= clamp_lo/clamp_hi)`.
    /// - A missing or non-numeric field (`kp`/`ki`/`kd`/`n`/`clamp_lo`/`clamp_hi`) is the same
    ///   generic `missing field '<key>'`/`field '<key>' is not a number` error every `kind=`
    ///   block produces — see [`BlockInstance`]'s own doc comment.
    /// - `ic=` with `ki=0` — rejected at parse time: `field 'ic' pre-loads the integrator, but
    ///   this PID has ki=0 -- there is no integrator to hold it`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pid in=<signal> kp=<f64> ki=<f64> kd=<f64> n=<f64> \
    ///      (clamp_lo=<f64> clamp_hi=<f64> | clamp_lo_in=<signal> clamp_hi_in=<signal>) [ic=<f64>]
    /// ```
    ///
    /// ## Example
    /// A fixed-clamp PI (no derivative action) regulating an error signal `ERR` to `[-1, 1]`:
    /// ```text
    /// PID1 kind=pid in=ERR kp=1 ki=0 kd=0 n=1 clamp_lo=-1 clamp_hi=1
    /// ```
    Pid {
        /// The compiled PID gains/state.
        pid: Pid,
        /// The fixed-vs-dynamic anti-windup bound.
        clamp: PidClamp,
    },
    /// <!-- component -->
    /// # State-Space
    /// **Purpose:** an arbitrary continuous-time linear block given directly as `(A, B, C, D)`
    /// matrices.
    /// **Library:** Control / Continuous
    ///
    /// ## Description
    /// A compensator/filter that doesn't already have a named convenience constructor, e.g. a
    /// low-pass filter placed ahead of a `Pid` to damp a resonant plant:
    /// $\dot{x} = Ax + Bu, \quad y = Cx + Du$. Genuinely MIMO: `B`'s own column count
    /// (`StateSpace::inputs()`) and `C`'s own row count (`StateSpace::outputs()`) are not
    /// constrained to `1` — a single-input single-output declaration is simply the `1x1` case.
    /// `outputs() == 1` produces a `SignalValue::Scalar`; `outputs() > 1` produces a
    /// `SignalValue::Vector` of that length. Inputs are the flattened concatenation of this
    /// block's own declared `Signal`s (scalar or vector, in order) — see `evaluate_blocks`'/
    /// `dae-runtime`'s own input-flattening convention, shared with
    /// `CoordinateTransform`/`Pmsm`/`CScript`. No anti-windup (that's specifically a `Pid`
    /// output's own concern, not every dynamic block's); stepped forward unconditionally every
    /// timestep via `StateSpace::rk4_step` (RK4, not backward Euler — this block's own state is
    /// never part of the circuit's own descriptor-DAE solve).
    ///
    /// ## Parameters
    /// - `a=<matrix>` — the `n x n` dynamics matrix, e.g. `a=[[0,1],[-1,-1]]`.
    /// - `b=<matrix|vector>` — `n x p` (`p` inputs); a flat vector (`b=[1,0]`) is SISO shorthand
    ///   for `p=1`, matching the pre-vector-signals convention unchanged.
    /// - `c=<matrix|vector>` — `q x n` (`q` outputs); same flat-vector SISO shorthand for `q=1`.
    /// - `d=<matrix|scalar>` — optional, defaults to an all-zero `q x n` matrix; a bare scalar
    ///   is only accepted when `q=1` and `p=1` (SISO) — a MIMO system must give the full matrix,
    ///   since a lone scalar has no unambiguous placement in a larger `D`.
    /// - `in=<signal>` (SISO, `p` must be `1`) or `inputs=<sig1,...>` (exactly `p` entries,
    ///   each independently scalar or vector; the *flattened* total length must equal `p`,
    ///   checked at evaluation time, not parse time, since a vector signal's own arity isn't
    ///   knowable from netlist text alone).
    /// - `ic=[x1,...,xn]` — optional (default: every state zero); the initial state vector
    ///   $x(0)$, one entry per state. A bare number (`ic=2`) is accepted for a one-state system.
    ///
    /// ## Errors
    /// - `q != 1 || p != 1` with a bare scalar `d=` — rejected at parse time: `field 'd' is a
    ///   bare scalar, but this system has <q> output(s) and <p> input(s) -- declare
    ///   'd=[[...],...]' (<q>x<p>) for a MIMO system, a bare scalar is only valid for a 1x1
    ///   (SISO) one`.
    /// - There is no `e=` (descriptor-matrix) field exposed at the netlist level yet, so
    ///   `StateSpaceError::NonInvertibleDescriptorMatrix` (a singular descriptor matrix) is
    ///   currently unreachable through this parser — `e` is always `None` (ordinary,
    ///   non-descriptor state-space) for every netlist-declared instance.
    /// - `ic=` with a number of entries other than the block's own state count — rejected at
    ///   parse time: `field 'ic' has <m> value(s), but this block has <n> state(s)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=statespace a=<matrix> b=<matrix|vector> c=<matrix|vector> [d=<matrix|scalar>] \
    ///      (in=<signal> | inputs=<sig1,sig2,...>) [ic=<vector>]
    /// ```
    ///
    /// ## Example
    /// A first-order low-pass ($\dot{x} = -x + u$, $y = x$, pole at $s=-1$) — verified end to
    /// end, `SRC=1` settling toward `SS1=1`:
    /// ```text
    /// SRC kind=const value=1
    /// SS1 kind=statespace a=[[-1]] b=[1] c=[1] in=SRC
    /// ```
    StateSpace(StateSpace),
    /// <!-- component -->
    /// # Transfer Function
    /// **Purpose:** a single-input single-output block given as a rational $N(s)/D(s)$.
    /// **Library:** Control / Continuous
    ///
    /// ## Description
    /// Numerator/denominator coefficients, highest-degree first, rather than a `Pid`'s
    /// `Kp`/`Ki`/`Kd` convenience parameterization — e.g. a hand-derived PID-with-filtered-
    /// derivative compensator ($C(s) = K_p + K_i/s + K_d N s/(s+N)$, put over one denominator
    /// first: a pure derivative term alone is non-causal/unrealizable, so every real PID,
    /// textbook or otherwise, filters it — see `continuous_blocks::Pid::to_transfer_function`'s
    /// own doc comment for the derivation). Compiled once via
    /// `TransferFunction::to_state_space` (controllable canonical form); no anti-windup, for
    /// the same reason [`BlockKind::StateSpace`] above has none.
    ///
    /// ## Parameters
    /// - `num=<vector>` — numerator coefficients, highest-degree first, e.g. `num=[1,2]` for
    ///   $s + 2$.
    /// - `den=<vector>` — denominator coefficients, highest-degree first, e.g. `den=[1,3,2]`
    ///   for $s^2 + 3s + 2$. Must be non-empty and nonzero-leading; `deg(num) <= deg(den)`
    ///   (a proper transfer function — every realizable continuous-time physical system is).
    /// - `in=<signal>` — the single input (one input, always — this block is SISO by
    ///   definition).
    /// - `ic=[x1,...,xn]` — optional (default: every state zero); the initial state of the
    ///   controllable canonical realization (see
    ///   `continuous_blocks::TransferFunction::to_state_space`), $n = \deg(\text{den})$ entries.
    ///   With the denominator made monic, $x_1$ is the state whose $k$-th derivative is
    ///   $x_{k+1}$, and the output is $y = \sum_i b_i x_{i+1} + d\,u$.
    /// - `y0=<f64>` — optional, mutually exclusive with `ic=`; starts the filter already settled
    ///   at output $y_0$: $x = (y_0 / N(0), 0, \dots, 0)$ with $N(0)$ the normalized numerator's
    ///   constant term (see `continuous_blocks::TransferFunction::settled_state`). Fed the
    ///   matching constant input $u = y_0 / G(0)$ the output stays at $y_0$ exactly; a pole at
    ///   the origin is fine (the matching input is then $0$).
    ///
    /// ## Errors
    /// - `den=[]` (empty) — rejected at parse time: `invalid transfer function
    ///   (EmptyDenominator)`.
    /// - `den`'s first (highest-degree) coefficient is `0.0` — rejected: `invalid transfer
    ///   function (ZeroLeadingDenominatorCoefficient)` (a zero leading coefficient means the
    ///   stated degree is a lie — the true denominator degree is lower).
    /// - `deg(num) > deg(den)` (more numerator coefficients than denominator ones) — rejected:
    ///   `invalid transfer function (ImproperTransferFunction)` — an improper transfer function
    ///   has no causal realization.
    /// - `ic=` with a number of entries other than the block's own state count — rejected at
    ///   parse time: `field 'ic' has <m> value(s), but this block has <n> state(s)`.
    /// - `ic=` and `y0=` both given — rejected at parse time: `'ic' (the state vector) and 'y0'
    ///   (the settled output) both set an initial condition -- give one or the other`.
    /// - `y0=` on a transfer function whose numerator is zero at DC — rejected at parse time:
    ///   `field 'y0': this transfer function's numerator is zero at DC, so its settled output is
    ///   always 0 and no state holds the requested one`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=tf num=<vector> den=<vector> in=<signal> [ic=<vector> | y0=<f64>]
    /// ```
    ///
    /// ## Example
    /// The same first-order low-pass as `StateSpace`'s own example ($1/(s+1)$), verified to
    /// settle toward the same steady-state value:
    /// ```text
    /// SRC kind=const value=1
    /// TF1 kind=tf num=[1] den=[1,1] in=SRC
    /// ```
    TransferFunction(TransferFunction),
    /// <!-- component -->
    /// # Discrete State-Space
    /// **Purpose:** the discrete-domain counterpart to `StateSpace` — a plain linear recursion.
    /// **Library:** Control / Discrete
    ///
    /// ## Description
    /// $x_{i+1} = Ax_i + Bu_i$, $y_i = Cx_i + Du_i$ — no RK4, no `dt` at all: `A`/`B`/`C`/`D`
    /// are already discrete-domain matrices, given directly by the netlist author, not derived
    /// from continuous ones. `sample_time` is **required** here (unlike `cscript`'s own
    /// optional field) and must be periodic — a discrete system's own dynamics *are* its
    /// sample period, there is no "continuous" fallback the way omitting it means for
    /// `cscript`, and a solver-chosen variable schedule doesn't compose with a fixed-period
    /// recursion at all. See `general-simulator`'s own `book/dev-guide/src/discrete-time-blocks.md`.
    ///
    /// ## Parameters
    /// - `a=`/`b=`/`c=`/`d=`/`in=`/`inputs=` — identical shape and MIMO rules to
    ///   [`BlockKind::StateSpace`]'s own Parameters (see there); the only difference is these
    ///   matrices are already discrete-domain.
    /// - `ts=<f64>` / `freq=<f64>` (with optional `to=<f64>` offset) — required; unlike
    ///   `cscript`, `ts=variable` is not accepted here (see Errors).
    /// - `ic=[x1,...,xn]` — optional (default: every state zero); the initial state vector
    ///   $x_0$, one entry per state (a bare number for a one-state system). With a nonzero
    ///   sample `offset`, the output held before the first sample hit is $C\,x_0$.
    ///
    /// ## Errors
    /// - Same MIMO scalar-`d` rejection as `StateSpace` (see there).
    /// - `ts=`/`freq=` missing entirely — rejected at parse time: `missing 'ts=' or 'freq='
    ///   -- a discrete-time block's own sample period is not optional (its dynamics *are* that
    ///   period), unlike the optional 'ts='/'freq=' on kind=cscript/pyblock/pyfunc`.
    /// - `ts=variable` — rejected at parse time: `ts=variable is not available here -- a
    ///   discrete block's own recursion has its coefficients baked in at one fixed sample
    ///   period, which 'the block decides its own next execution time' doesn't compose with`.
    /// - `ic=` with a number of entries other than the block's own state count — rejected at
    ///   parse time: `field 'ic' has <m> value(s), but this block has <n> state(s)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=discretestatespace a=<matrix> b=<matrix|vector> c=<matrix|vector> \
    ///      [d=<matrix|scalar>] (ts=<f64> | freq=<f64> [to=<f64>]) \
    ///      (in=<signal> | inputs=<sig1,sig2,...>) [ic=<vector>]
    /// ```
    ///
    /// ## Example
    /// A discrete integrator ($x_{i+1} = x_i + 0.1 \cdot u_i$, $y=x$, `ts=0.1`) fed a constant
    /// `1` — verified end to end, `DS1` reaching `0.5` after 5 sample hits ($5 \times 0.1$):
    /// ```text
    /// SRC kind=const value=1
    /// DS1 kind=discretestatespace a=[[1]] b=[0.1] c=[1] ts=0.1 in=SRC
    /// ```
    DiscreteStateSpace {
        /// The discrete-domain `A`/`B`/`C`/`D` matrices.
        ss: StateSpace,
        /// Must be [`SampleTimeSpec::Periodic`] — a discrete system's dynamics are its sample period.
        sample_time: SampleTimeSpec,
    },
    /// <!-- component -->
    /// # Discrete Transfer Function
    /// **Purpose:** the discrete-domain counterpart to `Transfer Function` — $Y(z)/U(z) = N(z)/D(z)$.
    /// **Library:** Control / Discrete
    ///
    /// ## Description
    /// Coefficients already in the $z$-domain. Realized via the exact same
    /// `TransferFunction::to_state_space` the continuous version uses (coefficient-to-
    /// companion-form conversion is domain-agnostic algebra — it doesn't know or care whether
    /// the variable is called $s$ or $z$), evaluated via `StateSpace::discrete_step` instead of
    /// `rk4_step`. Same mandatory-periodic-`sample_time` rule as [`BlockKind::DiscreteStateSpace`].
    ///
    /// ## Parameters
    /// - `num=`/`den=`/`in=` — identical shape and validity rules to
    ///   [`BlockKind::TransferFunction`]'s own Parameters (see there); coefficients are already
    ///   in the $z$-domain rather than $s$.
    /// - `ts=<f64>` / `freq=<f64>` (with optional `to=<f64>`) — required, same rule as
    ///   `DiscreteStateSpace`.
    /// - `ic=[x1,...,xn]` — optional (default: every state zero); the initial state of the same
    ///   controllable canonical realization as [`BlockKind::TransferFunction`]'s own `ic=`,
    ///   stepped as a difference equation.
    /// - `y0=<f64>` — optional, mutually exclusive with `ic=`; starts the filter already settled
    ///   at output $y_0$ — every canonical state equal to $y_0 / N(1)$, with $N(1)$ the
    ///   normalized numerator evaluated at $z = 1$ (see
    ///   `continuous_blocks::TransferFunction::settled_state_discrete`). Fed the matching
    ///   constant input $u = y_0 / H(1)$ the output stays at $y_0$ exactly.
    ///
    /// ## Errors
    /// Identical to [`BlockKind::TransferFunction`]'s own (`EmptyDenominator`,
    /// `ZeroLeadingDenominatorCoefficient`, `ImproperTransferFunction`), plus the same
    /// missing-`ts=`/`ts=variable`-rejected errors as `DiscreteStateSpace`.
    /// - `ic=` and `y0=` both given — rejected at parse time: `'ic' (the state vector) and 'y0'
    ///   (the settled output) both set an initial condition -- give one or the other`.
    /// - `y0=` on a transfer function whose numerator is zero at DC — rejected at parse time:
    ///   `field 'y0': this transfer function's numerator is zero at DC, so its settled output is
    ///   always 0 and no state holds the requested one`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=discretetf num=<vector> den=<vector> (ts=<f64> | freq=<f64> [to=<f64>]) in=<signal> \
    ///      [ic=<vector> | y0=<f64>]
    /// ```
    ///
    /// ## Example
    /// The exact $z$-domain realization of `DiscreteStateSpace`'s own example above ($Y(z)/U(z)
    /// = 0.1/(z-1)$, a discrete integrator) — verified to reach the same `DTF1=0.5` after 5
    /// sample hits:
    /// ```text
    /// SRC kind=const value=1
    /// DTF1 kind=discretetf num=[0.1] den=[1,-1] ts=0.1 in=SRC
    /// ```
    DiscreteTransferFunction {
        /// The `z`-domain numerator/denominator coefficients.
        tf: TransferFunction,
        /// Must be [`SampleTimeSpec::Periodic`].
        sample_time: SampleTimeSpec,
    },
    /// <!-- component -->
    /// # Discrete PID Controller
    /// **Purpose:** the discrete-domain counterpart to `PID Controller`.
    /// **Library:** Control / Discrete
    ///
    /// ## Description
    /// Reuses the exact same [`PidClamp`] anti-windup mechanism as [`BlockKind::Pid`], but the
    /// integral/derivative actions advance once per declared sample period via
    /// `continuous_blocks::DiscretePid::step` instead of RK4-integrating continuously. Not
    /// built by converting to one combined $z$-domain transfer function and back — realized
    /// directly from the block's own diagram (three parallel branches, summed). Same
    /// mandatory-periodic-`sample_time` rule as `DiscreteStateSpace`/`DiscreteTransferFunction`
    /// above — this block's own integration `period` is set *from* `sample_time`, not a
    /// separately-entered value.
    ///
    /// ## Parameters
    /// - `in=<signal>`, `kp=`/`ki=`/`kd=`/`n=`, and the fixed-vs-dynamic clamp fields
    ///   (`clamp_lo=`/`clamp_hi=` or `clamp_lo_in=`/`clamp_hi_in=`) — identical meaning to
    ///   [`BlockKind::Pid`]'s own Parameters (see there).
    /// - `ts=<f64>` / `freq=<f64>` (with optional `to=<f64>`) — required.
    /// - `integration_method=<forward|backward|trapezoidal>` — optional, defaults to
    ///   `forward` (forward Euler); selects the discrete integration rule the integral term
    ///   uses each sample period.
    /// - `ic=<f64>` — optional (default: start from rest); pre-loads the integrator so the
    ///   controller output starts at this value with zero error (the integrator's accumulated
    ///   value is set to $\text{ic}/k_i$).
    ///
    /// ## Errors
    /// - Same `n <= 0.0` and clamp-pairing errors as [`BlockKind::Pid`] (see there).
    /// - Same missing-`ts=`/`ts=variable`-rejected errors as `DiscreteStateSpace`.
    /// - `integration_method` present but not one of the three known names — rejected at parse
    ///   time: `unknown integration_method '<value>' (expected 'forward', 'backward', or
    ///   'trapezoidal')`.
    /// - `ic=` with `ki=0` — rejected at parse time: `field 'ic' pre-loads the integrator, but
    ///   this PID has ki=0 -- there is no integrator to hold it`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=discretepid in=<signal> kp=<f64> ki=<f64> kd=<f64> n=<f64> \
    ///      (ts=<f64> | freq=<f64> [to=<f64>]) \
    ///      (clamp_lo=<f64> clamp_hi=<f64> | clamp_lo_in=<signal> clamp_hi_in=<signal>) \
    ///      [integration_method=<forward|backward|trapezoidal>] [ic=<f64>]
    /// ```
    ///
    /// ## Example
    /// A discrete PI (no derivative action) sampled every 0.1 s, integrating a fixed error
    /// signal once per sample hit — verified end to end, `DPID1` stepping `0.5 -> 0.525 -> 0.55
    /// -> 0.575 -> 0.6` at each `ts=0.1` boundary (unchanged between hits, matching the
    /// discrete-recursion contract exactly):
    /// ```text
    /// ERR kind=const value=0.5
    /// DPID1 kind=discretepid in=ERR kp=1 ki=0.5 kd=0 n=1 ts=0.1 clamp_lo=-1 clamp_hi=1
    /// ```
    DiscretePid {
        /// The compiled discrete PID gains/state.
        pid: DiscretePid,
        /// The fixed-vs-dynamic anti-windup bound.
        clamp: PidClamp,
        /// Must be [`SampleTimeSpec::Periodic`]; also sets `pid`'s own `period`.
        sample_time: SampleTimeSpec,
    },
    /// <!-- component -->
    /// # VCO (Voltage-Controlled Oscillator)
    /// **Purpose:** a bare, standalone oscillator producing a `[0, 1)` ramp from a frequency input.
    /// **Library:** Control / Continuous
    ///
    /// ## Description
    /// A raw frequency-to-ramp conversion, still useful on its own outside gate-driving PWM —
    /// e.g. feeding a `MathFn1::Sin`/`Gain` chain to build a variable-frequency sinusoidal
    /// reference. Not how a gate-driving PWM modulator gets its own switching frequency, though
    /// — see [`BlockKind::Pwm`]/[`BlockKind::PhaseShiftPwm`] below, neither of which reads this
    /// block at all (each owns its own internal oscillator).
    ///
    /// ## Parameters
    /// - `f_min=<f64>`, `f_max=<f64>` — the frequency clamp range, Hz; the one input (`in=`) is
    ///   internally clamped to `[f_min, f_max]` before integration.
    /// - `in=<signal>` — the frequency command, Hz (one input).
    /// - `ic=<f64>` — optional (default `0`); the initial phase, in cycles,
    ///   $0 \le \text{ic} < 1$.
    ///
    /// ## Errors
    /// - `f_min > f_max` — rejected at parse time: `invalid vco (FMinExceedsFMax)`.
    /// - `ic=` outside $0 \le \text{ic} < 1$ — rejected at parse time: `field 'ic' is the initial
    ///   phase in cycles and must satisfy 0 <= ic < 1 (got <value>)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=vco f_min=<f64> f_max=<f64> in=<signal> [ic=<f64>]
    /// ```
    ///
    /// ## Example
    /// A fixed 10 Hz oscillator, verified to produce a `[0,1)` ramp (not stuck at a constant):
    /// ```text
    /// FREQ kind=const value=10
    /// VCO1 kind=vco f_min=1 f_max=1000 in=FREQ
    /// ```
    Vco(Vco),
    /// <!-- component -->
    /// # PWM Modulator 1
    /// **Purpose:** fixed-frequency, duty-driven, active-high complementary PWM.
    /// **Library:** Power Electronics
    ///
    /// ## Description
    /// One input, `duty` (`[0,1]`, clamped, read fresh every step from anywhere in the graph —
    /// a `Pid`, a filtered `TransferFunction`, a plain `Const`...), fixed carrier frequency
    /// `freq_hz`. Two outputs, following the [`BlockKind::CScript`] `output_names` convention:
    /// `output_names[0]` (aliasing this block's own `.name`) is the main signal, `output_names[1]`
    /// its active-high complement. `red`/`fed` (seconds) are independent per-edge dead-time
    /// delays — see `math_ops::complementary_pwm_with_deadtime` (in `dae-runtime`) for the
    /// exact rising-edge-only-delay semantics and why `red=fed=0.0` recovers the ideal,
    /// gap-free, overlap-free pair exactly. Stateless: a pure function of `(t, duty)` every
    /// step, no internal oscillator.
    ///
    /// ## Parameters
    /// - `freq=<f64>` — fixed carrier frequency, Hz.
    /// - `in=<signal>` — the duty command, `[0,1]` (clamped internally; no error for a value
    ///   outside that range).
    /// - `red=<f64>`, `fed=<f64>` — optional, default `0.0`; rising/falling-edge dead time,
    ///   seconds.
    /// - `outputs=<main,complement>` — optional; defaults to `<name>,<name>_comp`. Exactly 2
    ///   entries if given.
    ///
    /// ## Errors
    /// - `outputs=` with a count other than 2 — rejected at parse time: `'outputs' needs
    ///   exactly 2 entries (main, complement; got <n>)`.
    /// - `red=`/`fed=` present but not a number — the generic `field '<key>' is not a number`
    ///   error.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pwm freq=<f64> in=<signal> [red=<f64>] [fed=<f64>] [outputs=<main,complement>]
    /// ```
    ///
    /// ## Example
    /// A 10 kHz, 30% duty modulator with no dead time — verified end to end: `PWM1` high
    /// (`>= 0.99`) for the first ~30% of each 100 µs period and its complement exactly
    /// inverted, `PWM1 + PWM1_comp == 1` at every sampled instant:
    /// ```text
    /// DUTY kind=const value=0.3
    /// PWM1 kind=pwm freq=10000 in=DUTY
    /// ```
    Pwm {
        /// Fixed carrier frequency, Hz.
        freq_hz: f64,
        /// Rising-edge dead time, seconds.
        red: f64,
        /// Falling-edge dead time, seconds.
        fed: f64,
        /// `[main, complement]` output names.
        output_names: Vec<String>,
    },
    /// <!-- component -->
    /// # PWM Modulator 2 (Phase-Shift PWM)
    /// **Purpose:** frequency+phase+duty-driven, active-high complementary PWM with its own
    /// internal oscillator.
    /// **Library:** Power Electronics
    ///
    /// ## Description
    /// This is *not* a variant of [`BlockKind::Vco`] — it owns its own frequency-integration
    /// state directly (`osc` reuses [`Vco`]'s own clamp-and-integrate math purely as an
    /// implementation detail, the same formula, not a shared block reference), so two instances
    /// fed the *same* `freq` input stay bit-for-bit phase-synchronized (deterministic
    /// integration, same `dt`, same starting phase `0.0`), the way e.g. a dual-active-bridge's
    /// two legs need to be, without a separately-declared shared oscillator block in between.
    /// `red`/`fed` are in seconds, exactly like [`BlockKind::Pwm`]'s own, but converted to a
    /// phase fraction using *this step's own* resolved frequency (not a fixed constant) — this
    /// matters for a variable-frequency converter, since the same absolute dead time eats a
    /// larger fraction of the period at higher switching frequency, a real effect on e.g. a
    /// resonant converter's own ZVS margin, not just bookkeeping.
    ///
    /// ## Parameters
    /// - `f_min=<f64>`, `f_max=<f64>` — the internal oscillator's frequency clamp range, Hz.
    /// - `inputs=<freq,phase,duty>` — exactly 3, in this order: `freq` (Hz, clamped internally
    ///   to `[f_min, f_max]`), `phase` (`[0,1)`, a phase-shift command as a fraction of one
    ///   carrier period — *not* this block's own internal integration state, a different
    ///   thing; applied as a **lead**: conduction begins at $(1 - \text{phase})$ of the
    ///   carrier period, so `0.25` starts the pulse three quarters of the way through the
    ///   period, not one quarter — see the user guide's gate-bindings chapter for the run that
    ///   shows it), `duty` (`[0,1]`, clamped).
    /// - `red=<f64>`, `fed=<f64>` — optional, default `0.0`; seconds, converted internally to a
    ///   phase fraction using the current resolved frequency.
    /// - `outputs=<main,complement>` — optional; defaults to `<name>,<name>_comp`.
    /// - `ic=<f64>` — optional (default `0`); the carrier's own initial phase, in cycles,
    ///   $0 \le \text{ic} < 1$ — applied before, and independent of, the `phase` input's own
    ///   offset.
    ///
    /// ## Errors
    /// - `inputs=` with a count other than 3 — rejected at parse time: `kind='pspwm' needs 3
    ///   inputs (freq,phase,duty; got <n>)`.
    /// - `f_min > f_max` — rejected at parse time: `invalid pspwm oscillator
    ///   (FMinExceedsFMax)`.
    /// - `outputs=` with a count other than 2 — same error as [`BlockKind::Pwm`]'s own.
    /// - `ic=` outside $0 \le \text{ic} < 1$ — rejected at parse time: `field 'ic' is the initial
    ///   phase in cycles and must satisfy 0 <= ic < 1 (got <value>)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pspwm f_min=<f64> f_max=<f64> inputs=<freq_signal,phase_signal,duty_signal> \
    ///      [red=<f64>] [fed=<f64>] [outputs=<main,complement>] [ic=<f64>]
    /// ```
    ///
    /// ## Example
    /// A 10 kHz, zero-phase-shift, 40% duty modulator — verified end to end: both outputs
    /// `[0,1]`-valued, `PSPWM1 + PSPWM1_comp == 1` at every sampled instant:
    /// ```text
    /// FREQ kind=const value=10000
    /// PHASE kind=const value=0
    /// DUTY kind=const value=0.4
    /// PSPWM1 kind=pspwm f_min=1000 f_max=50000 inputs=FREQ,PHASE,DUTY
    /// ```
    PhaseShiftPwm {
        /// The internal frequency-integrating oscillator.
        osc: Vco,
        /// Rising-edge dead time, seconds.
        red: f64,
        /// Falling-edge dead time, seconds.
        fed: f64,
        /// `[main, complement]` output names.
        output_names: Vec<String>,
    },
    /// <!-- component -->
    /// # Product
    /// **Purpose:** multiplies all its inputs together.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// The product $x_1 \cdot x_2 \cdot \dots$ of its inputs. Input-shape rule shared with
    /// `Sum`: all-`Scalar` inputs give a `Scalar`; otherwise every input must be a `Vector` of
    /// one common length `N`, multiplied elementwise to a `Vector(N)` — a mix of `Scalar` and
    /// `Vector` inputs is rejected rather than guessed at (see the Sum entry).
    ///
    /// ## Parameters
    /// - `inputs=<sig1,sig2,...>` — comma-separated input signals, one per factor.
    ///
    /// ## Errors
    /// - Mixed `Scalar`/`Vector` inputs — rejected at evaluation time:
    ///   `VectorSignalNotSupported { block: "<name>" }`.
    /// - `Vector` inputs of differing lengths — rejected at evaluation time:
    ///   `VectorSignalSizeMismatch { block: "<name>", expected: N, got: M }`.
    /// - A missing `inputs=` is the generic `missing field 'inputs'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=product inputs=<sig1,sig2,...>
    /// ```
    ///
    /// ## Example
    /// ```text
    /// A kind=const value=2
    /// B kind=const value=3
    /// C kind=const value=4
    /// P kind=product inputs=A,B,C
    /// ```
    Product,
    /// <!-- component -->
    /// # Saturation
    /// **Purpose:** clamps its single input to a symmetric ±limit.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// Clamps its input to `[-limit, limit]` (elementwise for a `Vector` input) — the
    /// block-diagram saturation/limiter. See `continuous_blocks::math_ops::saturation`'s own
    /// doc comment for why this is a plain direct evaluation rather than a PWL device: the
    /// active segment depends only on this block's own input, so there is no
    /// simultaneous-resolution problem the way there is for a diode's terminal voltage.
    ///
    /// ## Parameters
    /// - `in=<signal>` — the single input.
    /// - `limit=<f64>` — the symmetric bound. Must be non-negative — but this is **not** checked
    ///   at parse time; a negative `limit` panics the process at the first evaluation (see
    ///   Errors).
    ///
    /// ## Errors
    /// - `limit=-1` — not a clean error: parse time never validates `limit`, so the evaluation
    ///   code's own assert fires and the whole process panics with
    ///   `limit must be nonnegative`. Keep `limit >= 0`.
    /// - A missing `in=`/`limit=` or a non-numeric `limit=` is the generic
    ///   `missing field '<key>'`/`field '<key>' is not a number` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=saturation in=<signal> limit=<f64>
    /// ```
    ///
    /// ## Example
    /// The time ramp (0..2 s) clamped to ±1 — passes through while `|x| <= 1`, holds 1 beyond:
    /// ```text
    /// X kind=time
    /// LIM kind=saturation in=X limit=1
    /// ```
    Saturation(f64),
    /// <!-- component -->
    /// # Table Lookup
    /// **Purpose:** linear interpolation through a fixed `(x, y)` table.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// A `table(x, a, b, c, d, ...)`-style lookup: linear interpolation through the fixed
    /// `(x, y)` breakpoints (elementwise for a `Vector` input), clamped — not extrapolated —
    /// outside the range, holding the nearest endpoint's `y`. Implemented by
    /// `continuous_blocks::waveform_arithmetic::table`.
    ///
    /// ## Parameters
    /// - `in=<signal>` — the single input (`x`).
    /// - `points=[[x1,y1],[x2,y2],...]` — the lookup table as a Python list of 2-element lists.
    ///   Sorted ascending by `x` at parse time, so declaration order doesn't matter. Must have
    ///   at least one point — but that is **not** checked at parse time; an empty list panics
    ///   the process at the first evaluation (see Errors).
    ///
    /// ## Errors
    /// - A `points=` entry that is not a 2-element `[x,y]` list — rejected at parse time:
    ///   `field 'points': each entry must be a 2-element '[x,y]' list (got '[1.0]')`.
    /// - `points=[]` — not a clean error: parse time never rejects an empty list, so the
    ///   evaluation code's own assert fires and the whole process panics with
    ///   `table needs at least one point`.
    /// - A missing `in=`/`points=` is the generic `missing field '<key>'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=table in=<signal> points=[[x1,y1],[x2,y2],...]
    /// ```
    ///
    /// ## Example
    /// A trapezoid profile (ramp up, flat, ramp down, held at 0) driven by the time ramp:
    /// ```text
    /// X kind=time
    /// T kind=table in=X points=[[0,0],[1,10],[2,10],[3,0]]
    /// ```
    Table(Vec<(f64, f64)>),
    /// <!-- component -->
    /// # Waveform Arithmetic Functions (unary)
    /// **Purpose:** one of the 26 single-argument real functions — trig, exponential/log,
    /// rounding, sign, step — as a stateless block.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// One entry for the whole unary family: every member is a pure `Scalar -> Scalar` function
    /// (elementwise `Vector -> Vector` for a `Vector` input), selected by its own `kind=` name,
    /// so they all share one input/error contract. Each name maps onto the same-named `f64`
    /// method where one exists (`continuous_blocks::waveform_arithmetic::MathFn1::call`), with
    /// the same domain/range behavior — trig in radians, `sqrt`/`ln` NaN for negative
    /// arguments, and so on. What the library as a whole deliberately leaves out (a
    /// finite-difference derivative, noise/random generators, complex-data functions, boolean
    /// comparison operators) is argued in `continuous_blocks::waveform_arithmetic`'s own module
    /// doc comment.
    ///
    /// | name | formula | notes |
    /// |---|---|---|
    /// | `abs` | $|x|$ | |
    /// | `acos` | $\arccos x$ | |
    /// | `acosh` | $\operatorname{arcosh} x$ | |
    /// | `asin` | $\arcsin x$ | |
    /// | `asinh` | $\operatorname{arsinh} x$ | |
    /// | `atan` | $\arctan x$ | |
    /// | `atanh` | $\operatorname{artanh} x$ | |
    /// | `buf` | $1$ if $x > 0.5$ else $0$ | threshold/buffer, companion to `inv` |
    /// | `ceil` | $\lceil x \rceil$ | |
    /// | `cos` | $\cos x$ | radians |
    /// | `cosh` | $\cosh x$ | |
    /// | `exp` | $e^x$ | |
    /// | `floor` | $\lfloor x \rfloor$ | |
    /// | `int` | truncation toward zero | |
    /// | `inv` | $0$ if $x > 0.5$ else $1$ | complement of `buf` |
    /// | `ln` | $\ln x$ | natural log |
    /// | `log10` | $\log_{10} x$ | |
    /// | `round` | round half away from zero | |
    /// | `sgn` | $-1$/$0$/$1$ | sign; 0 at the origin (unlike `f64::signum`, which returns 1 there) |
    /// | `sin` | $\sin x$ | radians |
    /// | `sinh` | $\sinh x$ | |
    /// | `sqrt` | $\sqrt{x}$ | |
    /// | `tan` | $\tan x$ | radians |
    /// | `tanh` | $\tanh x$ | |
    /// | `u` | $1$ if $x > 0$ else $0$ | unit step |
    /// | `uramp` | $x$ if $x > 0$ else $0$ | ramp that starts at the origin (Relu) |
    ///
    /// ## Parameters
    /// - `in=<signal>` — the single argument.
    ///
    /// ## Errors
    /// - A misspelled/unknown name (`kind=cosx`) — rejected at parse time:
    ///   `unknown device kind 'cosx'`.
    /// - A missing `in=` is the generic `missing field 'in'` error — see [`BlockInstance`]'s own
    ///   doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=<fn> in=<signal>
    /// ```
    ///
    /// ## Example
    /// `sin(2*pi*1000*t)` built from ordinary blocks, plus `cos`/`exp` and the elementwise
    /// vector form:
    /// ```text
    /// T kind=time
    /// G kind=gain in=T k=6283.185307179586
    /// S kind=sin in=G
    /// C kind=cos in=G
    /// E kind=exp in=T
    /// SV kind=const value=[1,4,9]
    /// SQ kind=sqrt in=SV
    /// ```
    MathFn1(MathFn1),
    /// <!-- component -->
    /// # Waveform Arithmetic Functions (binary)
    /// **Purpose:** one of the eight two-argument real functions — `atan2`/`hypot`/`pow` family,
    /// `min`/`max`, `anglewrap` — as a stateless block.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// One entry for the whole binary family: every member is a pure two-argument function
    /// selected by its own `kind=` name, taking `in1=`/`in2=` in that order. Input-shape rule:
    /// two `Scalar`s give a `Scalar`; otherwise every `Vector` operand must share one length
    /// `N` (elementwise application, with a lone `Scalar` operand broadcast against it).
    ///
    /// | name | formula | notes |
    /// |---|---|---|
    /// | `atan2` | $\operatorname{atan2}(x, y)$ | Rust's `x.atan2(y)`: the four-quadrant angle whose tangent is `x/y` |
    /// | `anglewrap` | $\operatorname{atan2}(\beta, \alpha)$ wrapped to $[0, 2\pi)$ | takes `(alpha, beta)`; the angle-tracking half of a synchronous-reference-frame PLL — see below |
    /// | `hypot` | $\sqrt{x^2 + y^2}$ | |
    /// | `max` | $\max(x, y)$ | |
    /// | `min` | $\min(x, y)$ | |
    /// | `pow` | $x^y$ | plain `f64::powf` (negative base with non-integer exponent is NaN) |
    /// | `pwr` | $|x|^y$ | like `pow` but always real-valued (a negative base raised to a non-integer power would otherwise be complex) |
    /// | `pwrs` | $\operatorname{sgn}(x)\,|x|^y$ | `pwr` with `x`'s original sign reapplied — odd-symmetric |
    ///
    /// `anglewrap` is the special member: `(alpha, beta) -> theta` — the `atan2(beta, alpha)`
    /// angle shifted by a full turn when negative, so `theta` stays in `[0, 2*pi)` and is a
    /// monotonically-sensible input to `park`/`clarke_park` across a full revolution — the
    /// angle-tracking half of a synchronous-reference-frame PLL (see
    /// `continuous_blocks::coordinate_transforms::anglewrap`, whose output feeds `park`/
    /// `clarke_park`). It lives here rather than with the Clarke/Park transforms because it is
    /// single-output: an ordinary two-argument function like every other member.
    ///
    /// ## Parameters
    /// - `in1=<signal>` `in2=<signal>` — the two arguments, in formula order above.
    ///
    /// ## Errors
    /// - A misspelled/unknown name — the same `unknown device kind '<name>'` error as the
    ///   unary entry.
    /// - `Vector` operands of differing lengths — rejected at evaluation time:
    ///   `VectorSignalSizeMismatch { block: "<name>", expected: N, got: M }`.
    /// - A missing `in1=`/`in2=` is the generic `missing field '<key>'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=<fn> in1=<signal> in2=<signal>
    /// ```
    ///
    /// ## Example
    /// The textbook 3-4-5 triangle, and the same pair's `anglewrap` result (`atan2(4,3)` ~
    /// 0.9273 rad):
    /// ```text
    /// X kind=const value=3
    /// Y kind=const value=4
    /// H kind=hypot in1=X in2=Y
    /// AW kind=anglewrap in1=X in2=Y
    /// ```
    MathFn2(MathFn2),
    /// <!-- component -->
    /// # Waveform Arithmetic Functions (ternary: if, limit)
    /// **Purpose:** a threshold select (`if`) and a clamp (`limit`) — the two three-argument
    /// functions.
    /// **Library:** Sources / Math operations
    ///
    /// ## Description
    /// One entry for the two three-argument functions — genuinely different semantics, but one
    /// shared input/error contract (documented at the type's own granularity). Both take
    /// `in1=`/`in2=`/`in3=` in the order below, and both use the binary entry's `Scalar`/
    /// `Vector` rule unchanged.
    ///
    /// | name | formula | notes |
    /// |---|---|---|
    /// | `if` | $y$ if $x > 0.5$ else $z$ | select/comparator: `in2` while `in1 > 0.5`, else `in3` |
    /// | `limit` | clamp $x$ to $[\min(y, z), \max(y, z)]$ | the "intermediate value" of `x`, `y`, `z` — the bounds may be given in either order |
    ///
    /// `limit`'s bounds are not fixed at ±`limit` the way `kind=saturation`'s are — they are two
    /// ordinary inputs, so they can themselves come from elsewhere in the graph.
    ///
    /// ## Parameters
    /// - `in1=<signal>` `in2=<signal>` `in3=<signal>` — the three arguments, in formula order
    ///   above.
    ///
    /// ## Errors
    /// - A misspelled/unknown name — the same `unknown device kind '<name>'` error as the
    ///   unary entry.
    /// - A missing `in1=`/`in2=`/`in3=` is the generic `missing field '<key>'` error — see
    ///   [`BlockInstance`]'s own doc comment.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=if     in1=<signal> in2=<signal> in3=<signal>
    /// NAME kind=limit  in1=<signal> in2=<signal> in3=<signal>
    /// ```
    ///
    /// ## Example
    /// A select picking between two references, and a clamp of 5 to the span [0, 2]:
    /// ```text
    /// CND kind=const value=1
    /// THEN kind=const value=10
    /// ELSE kind=const value=20
    /// SEL kind=if in1=CND in2=THEN in3=ELSE
    /// V kind=const value=5
    /// LO kind=const value=0
    /// HI kind=const value=2
    /// L kind=limit in1=V in2=LO in3=HI
    /// ```
    MathFn3(MathFn3),
    /// <!-- component -->
    /// # Hysteresis (Schmitt Trigger)
    /// **Purpose:** a bang-bang/hysteresis-band comparator.
    /// **Library:** Logic
    ///
    /// ## Description
    /// Stays HIGH until the input drops below `low`, stays LOW until the input rises above
    /// `high` (see `continuous_blocks::Hysteresis`) — used for current-mode control when
    /// there's no fixed switching frequency to modulate a duty command onto (unlike `Pid`
    /// feeding a [`BlockKind::Pwm`]). Its output is `1.0`/`0.0`, read directly by a
    /// [`GateBinding::Block`] rather than compared against a carrier. Deliberately not a
    /// `StateSpace`: the on/off memory is a genuine discrete latch, not a linear dynamic.
    ///
    /// ## Parameters
    /// - `high=<f64>`, `low=<f64>` — the two switching thresholds; must satisfy `low <= high`.
    /// - `in=<signal>` — the input signal (one input).
    /// - `ic=0|1` — optional (default `0`); the output the block starts the run holding, until
    ///   its own inputs first change it.
    ///
    /// ## Errors
    /// - `low > high` — rejected at parse time: `invalid hysteresis (LowExceedsHigh)`.
    /// - `ic=` that is neither `0` nor `1` — rejected at parse time: `field 'ic' is the initial
    ///   logic output and must be 0 or 1 (got '<value>')`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=hysteresis high=<f64> low=<f64> in=<signal> [ic=0|1]
    /// ```
    ///
    /// ## Example
    /// A `[-1, 1]` band, verified end to end: a rising `Time`-driven ramp is LOW until crossing
    /// `1`, then HIGH:
    /// ```text
    /// T kind=time
    /// RAMP kind=gain k=1 in=T
    /// HYST1 kind=hysteresis high=1 low=-1 in=RAMP
    /// ```
    Hysteresis(Hysteresis),
    /// <!-- component -->
    /// # Logic Gate
    /// **Purpose:** a combinational digital logic gate.
    /// **Library:** Logic
    ///
    /// ## Description
    /// `and`/`or`/`xor`/`nand`/`nor`/`xnor` (N-input, `N >= 2`) or `not` (exactly 1 input) —
    /// see `continuous_blocks::LogicOp::call` for the exact reduction rule (`Xor`/`Xnor` use the
    /// standard N-input generalization: true iff an odd/even number of inputs are true, not
    /// "exactly one"). Every input/output is an ordinary `f64` signal, thresholded at `>= 0.5`
    /// — the same convention `GateBinding`/[`BlockKind::Hysteresis`] already use; there is no
    /// dedicated boolean `SignalValue` variant anywhere in this crate. Purely combinational —
    /// no persistent state, recomputed fresh every step, the same as
    /// [`BlockKind::Sum`]/[`BlockKind::Gain`] already are.
    ///
    /// ## Parameters
    /// - `kind=<and|or|xor|nand|nor|xnor|not>` — which gate.
    /// - `inputs=<sig1,sig2,...>` — for every kind except `not`; at least 2 entries.
    /// - `in=<signal>` — for `kind=not` only; exactly 1 input.
    ///
    /// ## Errors
    /// - `and`/`or`/`xor`/`nand`/`nor`/`xnor` with fewer than 2 `inputs=` entries — rejected at
    ///   parse time: `kind=<name> needs at least 2 comma-separated inputs= entries (got <n>)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=<and|or|xor|nand|nor|xnor> inputs=<sig1,sig2,...>
    /// NAME kind=not in=<signal>
    /// ```
    ///
    /// ## Example
    /// A 2-input `and`, verified end to end (`1 and 0 = 0`, `1 and 1 = 1`):
    /// ```text
    /// A kind=const value=1
    /// B kind=const value=0
    /// G1 kind=and inputs=A,B
    /// ```
    LogicGate(LogicOp),
    /// <!-- component -->
    /// # SR Latch
    /// **Purpose:** a level-triggered set/reset latch.
    /// **Library:** Logic
    ///
    /// ## Description
    /// No clock — `set`/`reset` act every step, immediately. The one genuinely new state shape
    /// in this family (a single persisted `bool`, updated every step from whichever of
    /// `set`/`reset` is currently asserted — see `continuous_blocks::srlatch_next`):
    /// `set,reset -> q_next`: `false,false` holds; `true,false` sets `true`; `false,true` sets
    /// `false`; `true,true` (both at once) resolves per `priority`. `Set` (the default) is the
    /// right default for this block's own motivating use case, a sticky fault latch — a fault
    /// occurring in the same step a reset is asserted should still latch, not silently clear.
    ///
    /// ## Parameters
    /// - `set=<signal>`, `reset=<signal>` — both required.
    /// - `priority=<set|reset>` — optional, defaults to `set`; which input wins when both are
    ///   asserted simultaneously.
    /// - `ic=0|1` — optional (default `0`); the output the block starts the run holding, until
    ///   its own inputs first change it.
    ///
    /// ## Errors
    /// - `priority=` present but not `set`/`reset` — rejected at parse time: `field 'priority'
    ///   must be 'set' or 'reset' (got '<value>')`.
    /// - `ic=` that is neither `0` nor `1` — rejected at parse time: `field 'ic' is the initial
    ///   logic output and must be 0 or 1 (got '<value>')`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=srlatch set=<signal> reset=<signal> [priority=<set|reset>] [ic=0|1]
    /// ```
    ///
    /// ## Example
    /// A fault latch, verified end to end: `FAULT` pulses to `1` briefly then returns to `0`
    /// at `t=0.1`, but `LATCH1` stays `1` afterward (no separate reset asserted):
    /// ```text
    /// FAULT kind=pwc points=[[0,1],[0.1,0]]
    /// CLEAR kind=const value=0
    /// LATCH1 kind=srlatch set=FAULT reset=CLEAR
    /// ```
    SrLatch {
        /// Which input wins when `set` and `reset` are asserted simultaneously.
        priority: LatchPriority,
    },
    /// <!-- component -->
    /// # Flip-Flop
    /// **Purpose:** an edge-triggered D/JK/T flip-flop.
    /// **Library:** Logic
    ///
    /// ## Description
    /// Unlike [`BlockKind::SrLatch`] above, the next-state rule only fires at a detected rising
    /// `clk` edge (`continuous_blocks::rising_edge`); the previous `clk` sample is held
    /// alongside the output `q` between edges, the bookkeeping a level-triggered latch doesn't
    /// need at all.
    ///
    /// ## Parameters
    /// - `kind=<dff|jkff|tff>` — which flip-flop.
    /// - `clk=<signal>` — the clock input, required for every kind.
    /// - `d=<signal>` — for `kind=dff` only: the data input.
    /// - `t=<signal>` — for `kind=tff` only: the toggle-enable input.
    /// - `j=<signal>`, `k=<signal>` — for `kind=jkff` only: both required.
    /// - `reset=<signal>` — optional; if given, declares a synchronous reset input.
    /// - `ic=0|1` — optional (default `0`); the output the block starts the run holding, until
    ///   its own inputs first change it.
    ///
    /// ## Errors
    /// - The kind-specific data input (`d=`/`t=`/`j=`+`k=`) missing for its own kind — the
    ///   generic `missing field '<key>'` error.
    /// - `ic=` that is neither `0` nor `1` — rejected at parse time: `field 'ic' is the initial
    ///   logic output and must be 0 or 1 (got '<value>')`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=dff clk=<signal> d=<signal> [reset=<signal>] [ic=0|1]
    /// NAME kind=tff clk=<signal> t=<signal> [reset=<signal>] [ic=0|1]
    /// NAME kind=jkff clk=<signal> j=<signal> k=<signal> [reset=<signal>] [ic=0|1]
    /// ```
    ///
    /// ## Example
    /// A D flip-flop, verified end to end: `Q1` only updates to `D`'s current value at a clock
    /// rising edge, not continuously:
    /// ```text
    /// CLK kind=pwc points=[[0,0],[5e-3,1]]
    /// D kind=const value=1
    /// Q1 kind=dff clk=CLK d=D
    /// ```
    FlipFlop {
        /// `dff`/`jkff`/`tff`.
        kind: FlipFlopKind,
        /// Whether this instance declares a synchronous reset input.
        reset: bool,
    },
    /// <!-- component -->
    /// # Counter
    /// **Purpose:** a clocked up/down counter.
    /// **Library:** Logic
    ///
    /// ## Description
    /// Shares the same rising-edge-detection skeleton [`BlockKind::FlipFlop`] uses, generalized
    /// from a single bit to an integer count. `up_down`/`modulus`/`reset` are all optional:
    /// omitting `up_down` always increments; omitting `modulus` gives a free-running signed
    /// `i64` that wraps only at `i64`'s own bounds; omitting `reset` means no synchronous reset
    /// input at all.
    ///
    /// ## Parameters
    /// - `clk=<signal>` — the clock input, required.
    /// - `up_down=<signal>` — optional; if given, declares a direction input (counts down when
    ///   its value thresholds true, up otherwise).
    /// - `modulus=<u32>` — optional; wraps the count into `[0, modulus)` if given.
    /// - `reset=<signal>` — optional; if given, declares a synchronous reset input.
    /// - `ic=<integer>` — optional (default `0`); the initial count. With `modulus=`, must
    ///   satisfy $0 \le \text{ic} < \text{modulus}$.
    ///
    /// ## Errors
    /// - `modulus=` present but not a non-negative integer — rejected at parse time: `field
    ///   'modulus' is not a non-negative integer`.
    /// - `up_down=`/`reset=` declared (field present) but the signal itself malformed — the
    ///   generic `missing field '<key>'`/parse-signal errors.
    /// - `ic=` not an integer — rejected at parse time: `field 'ic' is not an integer`.
    /// - `ic=` outside the modulus — rejected at parse time: `field 'ic' must satisfy 0 <= ic <
    ///   modulus (<m>) (got <value>)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=counter clk=<signal> [up_down=<signal>] [modulus=<u32>] [reset=<signal>] \
    ///      [ic=<integer>]
    /// ```
    ///
    /// ## Example
    /// A free-running up counter clocked by a repeating pulse train — verified end to end,
    /// `CNT1` increasing by exactly `1` at each detected clock rising edge:
    /// ```text
    /// CLK kind=pwc points=[[0,0],[1e-3,1],[2e-3,0]] repeat=true
    /// CNT1 kind=counter clk=CLK
    /// ```
    Counter {
        /// Whether this instance declares an up/down direction input (unset: always increments).
        up_down: bool,
        /// Wraps at this value if set; free-running `i64` otherwise.
        modulus: Option<u32>,
        /// Whether this instance declares a synchronous reset input.
        reset: bool,
    },
    /// <!-- component -->
    /// # CScript
    /// **Purpose:** call into a user-supplied, precompiled native shared library once per
    /// (block-local) step — an escape hatch for stateful behavior no existing block covers.
    /// **Library:** Extensibility
    ///
    /// ## Description
    /// A dynamically-loaded, user-supplied block: `lib` is a precompiled shared library
    /// (`.so`/`.dylib`/`.dll`) exporting `cscript_start`/`cscript_output`/(optionally)
    /// `cscript_free`/`cscript_clone`, filling `output_names.len()` outputs. Unlike every
    /// other `BlockKind`, this one can carry state no Rust type here knows anything about —
    /// see `cscript_ffi`'s own module doc comment (in the sibling `cscript-ffi` crate) for the
    /// full C-side contract. **This is the one place in the workspace where arbitrary native
    /// code loads into the process** — calling into it is unsafe by construction (in-process,
    /// full host privileges, can corrupt memory or crash the process); this is an accepted,
    /// bounded risk of an explicitly opt-in feature (the user names their own `.so` in their
    /// own netlist), not something sandboxed away.
    ///
    /// `sample_time`, if given, makes this block run on its *own* schedule, independent of the
    /// circuit's own resolved step size — either a fixed period (optionally phase-offset), or a
    /// schedule the block computes for itself every time it runs. `None` (the default) calls
    /// `cscript_output` every resolved circuit step instead, passing that step's own `dt` — the
    /// right choice for a block meant to behave continuously. See [`SampleTimeSpec`]'s own doc
    /// comment for the two `Some(..)` cases.
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
    ///
    /// ## Parameters
    /// - `lib=<path>` — the precompiled shared library. Like every field, this is whitespace-
    ///   tokenized; **a path containing whitespace must be `"double-quoted"`**
    ///   (`lib="my libs/gain.so"`) — verified directly against a real space-containing
    ///   directory. An unquoted path with no whitespace needs no quotes, unchanged.
    /// - `outputs=<name1,name2,...>` — this block's own output names; defaults to a single
    ///   output aliasing the block's own `.name` if omitted.
    /// - `in=<signal>` or `inputs=<sig1,sig2,...>` — exactly one of these; `inputs=` for more
    ///   than one input, in the order `cscript_output`'s own `in`/`in_len` receives them.
    /// - `ts=<f64>` / `freq=<f64>` / `to=<f64>` / `ts=variable` — optional; see
    ///   [`SampleTimeSpec`]. Omit entirely to run every resolved circuit step.
    /// - `xc_count=<usize>` — optional, default `0`; number of solver-integrated continuous
    ///   states (switches the required C export set — see Description above).
    ///
    /// ## Errors
    /// - `lib`/`in` missing, or both `in` and `inputs` given, or neither — the generic
    ///   `missing field '<key>'` error (see [`BlockInstance`]'s own doc comment); there is no
    ///   `inputs`-vs-`in` mutual-exclusion check specifically, `inputs=` simply takes priority
    ///   when both are present.
    /// - `xc_count` present but not a non-negative integer — `field 'xc_count' is not a
    ///   non-negative integer`.
    /// - `ts=variable` combined with `freq=`/`to=` — rejected at parse time: `ts=variable is
    ///   exclusive with 'freq'/'to' (the block computes its own schedule; there's no fixed
    ///   period/offset to combine it with)`.
    /// - **Adaptive step-size control requires `cscript_clone`** to be exported — the solver
    ///   retries a rejected trial step from an independent copy of every block's state. Running
    ///   a `CScript` block with no `cscript_clone` export under adaptive stepping (i.e. no
    ///   `--dt` given to the CLI) fails at the first step with
    ///   `CScriptRequiresCloneForAdaptiveStep { block_name: "<name>" }` (verified directly: this
    ///   is the CLI's raw `{e:?}` `Debug` formatting of `dae_runtime`'s own error type, not a
    ///   hand-written message). Use a fixed `--dt` instead if the library doesn't export it.
    /// - A `.so` missing a required export, or a signature mismatch, surfaces as a load-time or
    ///   call-time failure from `cscript_ffi` itself — see that crate's own error type.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=cscript lib=<path> (in=<signal> | inputs=<sig1,sig2,...>) \
    ///      [outputs=<name1,name2,...>] [ts=<f64>|freq=<f64> [to=<f64>] | ts=variable] \
    ///      [xc_count=<usize>]
    /// ```
    ///
    /// ## Example
    /// A fixed-gain block computed in C (`out = 2 * in`), run every circuit step — `lib` names
    /// a shared library compiled from a two-function `.c` file exporting exactly
    /// `cscript_start`/`cscript_output`. Verified end to end (compiled, run through the real
    /// CLI, `G1`'s output confirmed to track `2 * SRC` every step, e.g. `SRC=3` produces
    /// `G1=6`):
    /// ```text
    /// SRC kind=const value=3
    /// G1 kind=cscript lib=gain.so in=SRC
    /// ```
    CScript {
        /// The precompiled shared library exporting the `cscript_*` C ABI.
        lib: std::path::PathBuf,
        /// This block's own output names, in order.
        output_names: Vec<String>,
        /// `None` runs every resolved circuit step; `Some` runs on its own schedule.
        sample_time: Option<SampleTimeSpec>,
        /// Number of solver-integrated continuous states this block owns; `0` for the plain contract.
        xc_count: usize,
    },
    /// <!-- component -->
    /// # PyBlock
    /// **Purpose:** call into a user-supplied, stateful Python function once per step — the
    /// Python-hosted counterpart to [`BlockKind::CScript`].
    /// **Library:** Extensibility
    ///
    /// ## Description
    /// A dynamically-loaded, user-supplied Python block (see `pyblock_ffi`): an escape hatch
    /// to a full scripting language (including `numpy`/`scipy`) when the existing block
    /// library doesn't cover something. `path` is a `.py` file exporting `start`/`output` (or,
    /// if `xc_count > 0`, `start`/`derivative`/`output_xc` *instead of* `output` — the same
    /// continuous-state split `CScript`'s own `xc_count` has, for the same reason).
    /// `output_names`/`sample_time`/`xc_count` all have exactly the same meaning as `CScript`'s
    /// own fields — deliberately scalar-only outputs, the same zero-order-hold `sample_time`
    /// convention, the same solver-integrated-vs-hand-managed state split. See
    /// `general-simulator`'s own `book/dev-guide/src/python-blocks.md` for the full design and
    /// the measurements behind it.
    ///
    /// ## Parameters
    /// - `path=<path>` — the `.py` file exporting `start`/`output` (or the `xc` contract). Same
    ///   whitespace-tokenization rule as `CScript`'s `lib=` — `"double-quote"` a path containing
    ///   whitespace.
    /// - `outputs=<name1,name2,...>` — output names; defaults to a single output aliasing the
    ///   block's own `.name`.
    /// - `in=<signal>` or `inputs=<sig1,sig2,...>` — exactly one of these.
    /// - `ts=<f64>` / `freq=<f64>` / `to=<f64>` / `ts=variable` — optional; see
    ///   [`SampleTimeSpec`]. `ts=variable` here requires the `.py` file to also export
    ///   `next_sample_hit` (unlike [`BlockKind::PyFunction`], where `ts=variable` is rejected
    ///   outright).
    /// - `xc_count=<usize>` — optional, default `0`.
    ///
    /// ## Errors
    /// - **This block requires the CLI/library built with the `python` Cargo feature enabled**
    ///   (`cargo build --features python`) — it is *not* on by default, since it needs a
    ///   discoverable Python/`libpython` at build/link time and `kind=cscript` already covers
    ///   the same "dynamically-loaded custom block" need without that dependency. Without it,
    ///   any `kind=pyblock`/`kind=pyfunc` netlist fails at the first step with
    ///   `PythonSupportNotCompiledIn { block_name: "<name>" }` (verified directly — the CLI's
    ///   raw `{e:?}` formatting of `dae_runtime`'s own error, not a hand-written message).
    /// - `path`/`in` missing, or malformed `outputs=`/`xc_count=` — same generic/`xc_count`
    ///   errors as [`BlockKind::CScript`].
    /// - A `.py` file missing a required function, raising an exception, or returning the wrong
    ///   arity surfaces as a Python-traceback-bearing failure from `pyblock_ffi` itself at call
    ///   time, not at parse time.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pyblock path=<path> (in=<signal> | inputs=<sig1,sig2,...>) \
    ///      [outputs=<name1,name2,...>] [ts=<f64>|freq=<f64> [to=<f64>] | ts=variable] \
    ///      [xc_count=<usize>]
    /// ```
    ///
    /// ## Example
    /// A fixed-gain block computed in Python (`out = 2 * in`) — verified end to end against a
    /// `--features python` build of the CLI, `SRC=3` producing `G1=6`:
    /// ```text
    /// SRC kind=const value=3
    /// G1 kind=pyblock path=gain.py in=SRC
    /// ```
    /// where `gain.py` is:
    /// ```python
    /// def start():
    ///     return None
    ///
    /// def output(state, t, dt, inputs):
    ///     return inputs[0] * 2.0
    /// ```
    /// The `xc_count>0` contract (`derivative`/`output_xc`) is verified separately end to end
    /// against its own analytic solution ($\dot{x}_c = 1 - x_c$, starting at rest, so
    /// $x_c(t) = 1 - e^{-t}$) — see `doc-verify/pyblock/xc_example.cir`.
    PyBlock {
        /// The `.py` file exporting `start`/`output` (or `start`/`derivative`/`output_xc`).
        path: std::path::PathBuf,
        /// This block's own output names, in order.
        output_names: Vec<String>,
        /// `None` runs every resolved circuit step; `Some` runs on its own schedule.
        sample_time: Option<SampleTimeSpec>,
        /// Number of solver-integrated continuous states this block owns; `0` for the plain contract.
        xc_count: usize,
    },
    /// <!-- component -->
    /// # PyFunction
    /// **Purpose:** call a plain, stateless, named Python function once per step.
    /// **Library:** Extensibility
    ///
    /// ## Description
    /// A genuinely separate contract from [`BlockKind::PyBlock`] above (not a mode of it): no
    /// `start`, no persistent `state`, no `t`/`dt` boilerplate, just `function` (a name inside
    /// `path`'s own `.py` file) called with each declared `inputs=` entry as its own
    /// positional argument (`f(*inputs)`, never bundled into one list), returning a single
    /// value or a tuple/list matching `output_names`. The closest equivalent to a plain
    /// named-input/named-output function block in other block-diagram tools. `output_names`/
    /// `sample_time` mean the same as `PyBlock`'s own (still scalar-only outputs, still the
    /// same zero-order-hold convention) — there is no `xc_count` here at all, since a purely
    /// stateless function has nothing for a continuous state to mean. See
    /// `pyblock_ffi::PyFunctionInstance`'s own module doc comment for the full contract.
    ///
    /// ## Parameters
    /// - `path=<path>` — the `.py` file containing `function`. `"double-quote"` a path
    ///   containing whitespace, same as `CScript`'s `lib=`.
    /// - `function=<name>` — required, no default; the function called as `f(*inputs)`.
    /// - `outputs=<name1,name2,...>` — output names; defaults to a single output aliasing the
    ///   block's own `.name`.
    /// - `in=<signal>` or `inputs=<sig1,sig2,...>` — exactly one of these.
    /// - `ts=<f64>` / `freq=<f64>` / `to=<f64>` — optional, [`SampleTimeSpec::Periodic`] only;
    ///   `ts=variable` is rejected (see Errors).
    ///
    /// ## Errors
    /// - **Requires a `--features python` build** — see [`BlockKind::PyBlock`]'s own Errors
    ///   section; identical `PythonSupportNotCompiledIn` failure.
    /// - `ts=variable` — rejected at parse time, verified directly: `ts=variable is not
    ///   available for kind=pyfunc (a stateless function call has no instance to remember a
    ///   requested next-hit time against) -- use kind=pyblock instead`. A stateless function
    ///   call has no instance to remember "when am I next due" against.
    /// - `function` missing — the generic `missing field 'function'` error.
    /// - A `.py` file missing `function`, raising an exception, or returning the wrong arity
    ///   surfaces as a Python-traceback-bearing failure from `pyblock_ffi` at call time.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pyfunc path=<path> function=<name> (in=<signal> | inputs=<sig1,sig2,...>) \
    ///      [outputs=<name1,name2,...>] [ts=<f64> | freq=<f64> [to=<f64>]]
    /// ```
    ///
    /// ## Example
    /// Verified end to end against a `--features python` build, `SRC=3` producing `G1=4`:
    /// ```text
    /// SRC kind=const value=3
    /// G1 kind=pyfunc path=add.py function=add_one in=SRC
    /// ```
    /// where `add.py` is:
    /// ```python
    /// def add_one(x):
    ///     return x + 1.0
    /// ```
    /// `inputs=A,B` calling `f(A, B)` as two genuinely distinct positional arguments (never
    /// bundled into one list) is verified separately — see `doc-verify/pyfunc/example_two_inputs.cir`.
    PyFunction {
        /// The `.py` file containing `function`.
        path: std::path::PathBuf,
        /// The function name called with each declared input as a positional argument.
        function: String,
        /// This block's own output names, in order.
        output_names: Vec<String>,
        /// `None` runs every resolved circuit step; `Some` must be [`SampleTimeSpec::Periodic`].
        sample_time: Option<SampleTimeSpec>,
    },
    /// <!-- component -->
    /// # OctFunc
    /// **Purpose:** call a plain, stateless, named Octave-compatible function once per step.
    /// **Library:** Extensibility
    ///
    /// ## Description
    /// The `.m`-file counterpart to [`BlockKind::PyFunction`] above -- same genuinely separate,
    /// stateless contract (no `start`, no persistent state, no `t`/`dt` boilerplate), for
    /// people with legacy Octave-compatible `.m` scripts instead of Python. `function` (a
    /// name inside `path`'s own `.m` file, which must itself be named `<function>.m` -- Octave's
    /// own requirement that a function's file name match its function name) is called with each
    /// declared `inputs=` entry as its own positional argument (`f(arg0, arg1, ...)`, never
    /// bundled into one array), returning a single value or as many comma-separated return
    /// values as `output_names` declares (`[y1, y2] = f(...)`). `output_names`/`sample_time`
    /// mean the same as `PyFunction`'s own -- still scalar-only outputs, still the same
    /// zero-order-hold convention -- and there is no `xc_count` here either, for the same reason
    /// `PyFunction` has none: a purely stateless function has nothing for a continuous state to
    /// mean.
    ///
    /// Unlike `PyFunction`, this block never embeds an interpreter in-process: it calls into a
    /// separately installed `octave-cli` binary as a subprocess (one shared, persistent process
    /// for the whole simulation run, spawned lazily on first use), specifically so this
    /// permissively-licensed project never links against GPLv3-licensed Octave code -- see
    /// `octave_ffi`'s own module doc comment and `general-simulator`'s own
    /// `book/dev-guide/src/octave-blocks.md` for the full licensing rationale and the measured
    /// costs behind this design.
    ///
    /// ## Parameters
    /// - `path=<path>` — the `.m` file containing `function`; its own file name (minus `.m`)
    ///   must equal `function`. `"double-quote"` a path containing whitespace, same as
    ///   `CScript`'s `lib=`.
    /// - `function=<name>` — required, no default; the function called as `f(arg0, arg1, ...)`,
    ///   and the required base name of `path`'s own file.
    /// - `outputs=<name1,name2,...>` — output names; defaults to a single output aliasing the
    ///   block's own `.name`.
    /// - `in=<signal>` or `inputs=<sig1,sig2,...>` — exactly one of these.
    /// - `ts=<f64>` / `freq=<f64>` / `to=<f64>` — optional, [`SampleTimeSpec::Periodic`] only;
    ///   `ts=variable` is rejected (see Errors), for exactly the reason `PyFunction`'s own is.
    ///
    /// ## Errors
    /// - `ts=variable` — rejected at parse time, the same wording `PyFunction` uses (a stateless
    ///   function call has no instance to remember a requested next-hit time against): `ts=variable
    ///   is not available for kind=octfunc (a stateless function call has no instance to remember
    ///   a requested next-hit time against)`.
    /// - `function` missing — the generic `missing field 'function'` error.
    /// - `octave-cli` not found on `PATH` when this block is first evaluated, an Octave-side
    ///   exception raised by the called function, or a malformed/desynchronized reply from the
    ///   shared session — surfaces as a call-time failure from `octave_ffi` itself (see
    ///   `octave_ffi::OctaveError`'s own three cases: not-found, process-exited, and Octave-side
    ///   runtime error), not at parse time.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=octfunc path=<file.m> function=<name> (in=<signal> | inputs=<sig1,sig2,...>) \
    ///      [outputs=<name1,name2,...>] [ts=<f64> | freq=<f64> [to=<f64>]]
    /// ```
    ///
    /// ## Example
    /// Verified end to end against the real CLI (no special feature flag needed -- unlike
    /// `kind=pyblock`/`kind=pyfunc`, this block builds and runs unconditionally), `SRC=3`
    /// producing `G1=4`:
    /// ```text
    /// SRC kind=const value=3
    /// G1 kind=octfunc path=add_one.m function=add_one in=SRC
    /// ```
    /// where `add_one.m` is:
    /// ```text
    /// function y = add_one(x)
    ///   y = x + 1;
    /// end
    /// ```
    /// `inputs=A,B` calling `f(A, B)` as two genuinely distinct positional arguments is verified
    /// separately -- see `doc-verify/octfunc/example_two_inputs.cir`.
    OctFunc {
        /// The `.m` file containing `function` -- its own base file name must equal `function`.
        path: std::path::PathBuf,
        /// The function name called with each declared input as a positional argument.
        function: String,
        /// This block's own output names, in order.
        output_names: Vec<String>,
        /// `None` runs every resolved circuit step; `Some` must be [`SampleTimeSpec::Periodic`].
        sample_time: Option<SampleTimeSpec>,
    },
    /// <!-- component -->
    /// # OctBlock
    /// **Purpose:** call into a user-supplied, *stateful* Octave-compatible function once per
    /// step -- the Octave-hosted counterpart to [`BlockKind::PyBlock`], and the stateful sibling
    /// of [`BlockKind::OctFunc`] (not a mode of it -- see that variant's own doc comment for the
    /// stateless contract this one deliberately does not share).
    /// **Library:** Extensibility
    ///
    /// ## Description
    /// An escape hatch to Octave-compatible `.m`-file functions with real persistent state
    /// across steps -- for people with legacy Octave-compatible `.m` scripts who need
    /// `pyblock`'s own `start`/`output`/`update`/`xc_count` contract, not `octfunc`'s stateless
    /// one. `output_names`/`sample_time`/`xc_count` all mean exactly what [`BlockKind::PyBlock`]'s
    /// own do (scalar-only outputs, the same zero-order-hold `sample_time` convention, the same
    /// solver-integrated-vs-hand-managed `xc` split).
    ///
    /// **`path=` names a directory, not a single file** -- the one place this contract cannot
    /// simply reuse `OctFunc`'s own single-`.m`-file convention: Octave only auto-loads the
    /// *first* `function ... end` block in a `.m` file by filename, so a second function in the
    /// same file is an invisible private subfunction (confirmed empirically against real
    /// `octave-cli`). Every contract function therefore needs **its own file**, named
    /// `<function>_<role>.m`, all living together in `path`'s own directory:
    ///
    /// | File | Required | Signature |
    /// |---|---|---|
    /// | `<function>_start.m` | always | `state = <function>_start()` |
    /// | `<function>.m` | unless `xc_count>0` | `[new_state, y1, y2, ...] = <function>(state, t, dt, u1, u2, ...)` |
    /// | `<function>_derivative.m` | only if `xc_count>0` | `xc_dot = <function>_derivative(state, u1, ..., xc)` -- pure, must not mutate `state`; called up to 4x/step (RK4 stages), including from discarded trial steps under adaptive stepping |
    /// | `<function>_output_xc.m` | only if `xc_count>0` | `[new_state, y1, ...] = <function>_output_xc(state, t, dt, u1, ..., xc)` -- `xc` here is this step's own already-integrated continuous state |
    /// | `<function>_update.m` | optional | `new_state = <function>_update(state, t, dt, u1, ...)` -- called once per resolved step, after output; if absent, `output`/`output_xc`'s own returned `new_state` is the only state advance |
    /// | `<function>_next_sample_hit.m` | only if `ts=variable` | `seconds = <function>_next_sample_hit(state, t, dt, u1, ...)` |
    ///
    /// A user coming from Octave/legacy-language habits of "one script, many local functions"
    /// hits this immediately -- this is not a buried caveat, it is the load-bearing difference
    /// from every other block kind's own `path=`.
    ///
    /// Each instance's own opaque `state` (whatever `<function>_start` returns) lives entirely
    /// **inside the shared Octave session itself**, in one global struct keyed by this block's
    /// own instance name (`global __gs_state; __gs_state.('<name>') = ...`) -- it is never
    /// serialized back into Rust at all, unlike `PyBlock`'s own `state` (an opaque Python
    /// object `pyblock_ffi` holds a handle to) or `CScript`'s (an opaque `void*`). Two instances
    /// of the *same* Octave function never contaminate each other's state -- validated directly
    /// against real `octave-cli` (interleaved calls, zero cross-contamination; see
    /// `octave_ffi::OctaveSession`'s own `tests/stateful_session.rs`). The one piece of state
    /// that *does* cross the pipe as explicit numeric data (the same `%.17g`-per-line protocol
    /// `octfunc` already uses, extended to a literal Octave vector) is the solver-owned `xc`
    /// continuous-state vector for the `xc_count > 0` case -- that's the *solver's* state, not
    /// the block's own opaque one, so it has to be explicit, exactly mirroring `cscript`'s/
    /// `pyblock`'s own `xc` vs. opaque-`state` split. See `general-simulator`'s own
    /// `book/dev-guide/src/octave-blocks.md` for the full protocol, licensing rationale, and
    /// measured costs.
    ///
    /// Like `OctFunc`, this block shares **one** persistent `octave-cli` subprocess with every
    /// other Octave-hosted block kind in the same run (spawned lazily, on first use) -- never a
    /// second process.
    ///
    /// ## Parameters
    /// - `path=<dir>` — the directory containing `<function>_*.m` (see the file table above).
    ///   `"double-quote"` a path containing whitespace, same as `CScript`'s `lib=`.
    /// - `function=<name>` — required, no default; the base name every `<function>_<role>.m`
    ///   file shares.
    /// - `outputs=<name1,name2,...>` — output names; defaults to a single output aliasing the
    ///   block's own `.name`.
    /// - `in=<signal>` or `inputs=<sig1,sig2,...>` — exactly one of these.
    /// - `ts=<f64>` / `freq=<f64>` / `to=<f64>` / `ts=variable` — optional; see
    ///   [`SampleTimeSpec`]. `ts=variable` here requires `<function>_next_sample_hit.m` to also
    ///   exist in `path`'s own directory.
    /// - `xc_count=<usize>` — optional, default `0`.
    ///
    /// ## Errors
    /// - `path`/`function`/`in` missing, or malformed `outputs=`/`xc_count=` — same generic/
    ///   `xc_count` errors as [`BlockKind::CScript`].
    /// - A required `<function>_*.m` file missing from `path`'s own directory (per the file
    ///   table above, given this block's own `xc_count`/`ts=`) — checked once, up front, at
    ///   construction time, *before* `octave-cli` is ever asked about it: `dae_runtime::DaeError
    ///   ::OctBlockMissingRequiredFile { block_name, expected_path }`.
    /// - `ts=variable` without `<function>_next_sample_hit.m` present — checked the same way:
    ///   `dae_runtime::DaeError::OctBlockRequiresNextSampleHitForVariableSampleTime`.
    /// - `octave-cli` not found on `PATH`, an Octave-side exception raised by a called function,
    ///   or a malformed/desynchronized reply — surfaces as a call-time failure from `octave_ffi`
    ///   itself (`dae_runtime::DaeError::Octave`), not at parse time.
    /// - **`kind=octblock` does not support `TimeStep::Adaptive`** — this instance's own opaque
    ///   state lives inside the one shared `octave-cli` session, which is never cloned, so a
    ///   rejected adaptive trial step's own state-mutating calls could not be rolled back; a
    ///   netlist using `kind=octblock` under an adaptive run fails at construction with
    ///   `dae_runtime::DaeError::OctBlockDoesNotSupportAdaptiveStep { block_name }`. Use a fixed
    ///   `--dt`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=octblock path=<dir> function=<name> (in=<signal> | inputs=<sig1,sig2,...>) \
    ///      [outputs=<name1,name2,...>] [ts=<f64>|freq=<f64> [to=<f64>] | ts=variable] \
    ///      [xc_count=<usize>]
    /// ```
    ///
    /// ## Example
    /// A stateful accumulator (`out = state + in`, state advances by `in` every step) verified
    /// end to end against the real CLI, `SRC=3` producing `G1=3,6,9,...` across successive
    /// steps:
    /// ```text
    /// SRC kind=const value=3
    /// G1 kind=octblock path=accum function=accumulate in=SRC
    /// ```
    /// where `accum/accumulate_start.m` is:
    /// ```text
    /// function state = accumulate_start()
    ///   state = 0;
    /// end
    /// ```
    /// and `accum/accumulate.m` is:
    /// ```text
    /// function [new_state, y] = accumulate(state, t, dt, u)
    ///   new_state = state + u;
    ///   y = new_state;
    /// end
    /// ```
    /// The `xc_count>0` contract (`_derivative`/`_output_xc`) is verified separately end to end
    /// against its own analytic solution ($\dot{x}_c = 1 - x_c$, starting at rest, so
    /// $x_c(t) = 1 - e^{-t}$) — see `doc-verify/octblock/xc_example.cir`, mirroring
    /// `doc-verify/pyblock/xc_example.cir`'s own precedent.
    OctBlock {
        /// The directory containing `<function>_start.m`, `<function>.m` (or
        /// `<function>_derivative.m`/`<function>_output_xc.m`), and optionally
        /// `<function>_update.m`/`<function>_next_sample_hit.m`.
        path: std::path::PathBuf,
        /// The shared base name every `<function>_<role>.m` file in `path` uses.
        function: String,
        /// This block's own output names, in order.
        output_names: Vec<String>,
        /// `None` runs every resolved circuit step; `Some` runs on its own schedule.
        sample_time: Option<SampleTimeSpec>,
        /// Number of solver-integrated continuous states this block owns; `0` for the plain contract.
        xc_count: usize,
    },
    /// <!-- component -->
    /// # Coordinate Transform (Clarke/Park)
    /// **Purpose:** one of the six Clarke/Park three-phase coordinate transforms.
    /// **Library:** Coordinate Transforms
    ///
    /// ## Description
    /// The standard `abc`/`alpha-beta-0`/`d-q-0` change of basis (see
    /// `continuous_blocks::CoordinateTransform`) used to regulate a three-phase quantity
    /// (grid-tied PFC, motor drive) with a `Pid` on a DC-like `d`/`q` value instead of chasing
    /// a sine wave directly. Stateless and multi-output, following exactly the same
    /// `output_names` convention [`BlockKind::CScript`] established: `inputs` supplies
    /// `kind.input_count()` values in the order `CoordinateTransform::call` expects,
    /// `output_names.len()` must equal 3 (this family's output count — the conventional
    /// per-transform names, e.g. `["alpha", "beta", "zero"]` for `Clarke`), the block's own
    /// `.name` binds to the first (primary) output, and the remaining two are inserted under
    /// their own `output_names` entries so a downstream block can reference them directly via
    /// `Signal::Block(name)`.
    ///
    /// ## Parameters
    /// - `kind=<clarke|clarkeinv|park|parkinv|clarkepark|clarkeparkinv>` — which transform.
    /// - `inputs=<sig1,...>` — `3` entries for `clarke`/`clarkeinv`, `4` for the other four
    ///   (`park`/`parkinv`/`clarkepark`/`clarkeparkinv` also take an angle input).
    /// - `outputs=<name1,name2,name3>` — optional; defaults to `<name>,<name>_<suffix2>,
    ///   <name>_<suffix3>` using this transform's own conventional output names.
    ///
    /// ## Errors
    /// - Wrong `inputs=` count for this transform — rejected at parse time: `kind='<kind>'
    ///   needs <n> inputs (got <m>)`.
    /// - `outputs=` with a count other than 3 — rejected at parse time: `'outputs' needs
    ///   exactly 3 entries (got <n>)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=clarke inputs=<a,b,c> [outputs=<name1,name2,name3>]
    /// NAME kind=park inputs=<a,b,c,theta> [outputs=<name1,name2,name3>]
    /// ```
    ///
    /// ## Example
    /// A balanced three-phase set through `clarke` — verified end to end: for
    /// $a=1,b=-0.5,c=-0.5$ (a peak on phase A), $\alpha = 1$, $\beta = 0$ exactly:
    /// ```text
    /// A kind=const value=1
    /// B kind=const value=-0.5
    /// C kind=const value=-0.5
    /// CT1 kind=clarke inputs=A,B,C
    /// ```
    CoordinateTransform {
        /// Which of the six Clarke/Park transforms this is.
        kind: CoordinateTransform,
        /// Output names; length must equal 3 for this family.
        output_names: Vec<String>,
    },
    /// <!-- component -->
    /// # PMSM (Permanent-Magnet Synchronous Motor)
    /// **Purpose:** a nonlinear permanent-magnet synchronous motor model.
    /// **Library:** Machines
    ///
    /// ## Description
    /// Genuinely nonlinear (bilinear speed/current coupling — see `continuous_blocks::Pmsm`),
    /// so like [`BlockKind::Vco`] it carries its own state and is integrated via its own
    /// `step()` (RK4) rather than compiled to a `StateSpace`. Three inputs, in order: `vd`,
    /// `vq` (rotor-frame stator voltage commands, V — typically a `ClarkeParkInv`'s output, or
    /// a `CoordinateTransform` intermediate wired through a `Pid`), and `t_load` (N·m, the
    /// mechanical load torque). Four outputs, same `output_names` convention as
    /// [`BlockKind::CoordinateTransform`]/[`BlockKind::CScript`]: `id`, `iq` (A), `omega_m`
    /// (mechanical speed, rad/s), and `theta_e` (electrical angle, already wrapped to
    /// $[0, 2\pi)$ via `Pmsm::theta_e_wrapped` — ready to feed a `Park`/`ClarkePark` block
    /// directly). Starts at rest (`id = iq = omega_m = theta_e = 0`) — no initial-condition
    /// override, matching every other dynamic block in this graph. Surface-mount motors have
    /// `l_d == l_q` (no reluctance torque term); interior-PM motors have `l_d != l_q`.
    ///
    /// ## Parameters
    /// - `r_s=<f64>` — stator resistance, Ω; must be `>= 0`.
    /// - `l_d=<f64>`, `l_q=<f64>` — d/q-axis inductance, H; both must be `> 0`.
    /// - `lambda_pm=<f64>` — permanent-magnet flux linkage, Wb.
    /// - `pole_pairs=<f64>` — must be `> 0`.
    /// - `inertia=<f64>` — rotor inertia, kg·m²; must be `> 0`.
    /// - `friction=<f64>` — viscous friction coefficient; must be `>= 0`.
    /// - `inputs=<vd,vq,t_load>` — exactly 3, in this order.
    /// - `outputs=<id,iq,omega_m,theta_e>` — optional; defaults to `<name>,<name>_iq,
    ///   <name>_omega_m,<name>_theta_e`.
    /// - `ic=[id,iq,omega_m,theta_e]` — optional (default: all zero); the four machine states at
    ///   $t = 0$: $d$/$q$ currents (A), mechanical speed (rad/s), electrical angle (rad).
    ///
    /// ## Errors
    /// - `r_s < 0` — `invalid pmsm (NegativeResistance)`.
    /// - `l_d <= 0` or `l_q <= 0` — `invalid pmsm (NonPositiveInductance)`.
    /// - `pole_pairs <= 0` — `invalid pmsm (NonPositivePolePairs)`.
    /// - `inertia <= 0` — `invalid pmsm (NonPositiveInertia)`.
    /// - `friction < 0` — `invalid pmsm (NegativeFriction)`.
    /// - `ic=` with other than four entries — rejected at parse time: `field 'ic' has <m>
    ///   value(s), but this block has 4 state(s)`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=pmsm r_s=<f64> l_d=<f64> l_q=<f64> lambda_pm=<f64> pole_pairs=<f64> \
    ///      inertia=<f64> friction=<f64> inputs=<vd,vq,t_load> [outputs=<id,iq,omega_m,theta_e>] \
    ///      [ic=<vector>]
    /// ```
    ///
    /// ## Example
    /// A small surface-mount motor at rest with zero applied voltage/load — verified to run
    /// without error and stay at rest (`id = iq = omega_m = 0`, no torque applied):
    /// ```text
    /// VD kind=const value=0
    /// VQ kind=const value=0
    /// TLOAD kind=const value=0
    /// M1 kind=pmsm r_s=0.5 l_d=1e-3 l_q=1e-3 lambda_pm=0.05 pole_pairs=4 inertia=1e-5 \
    ///      friction=1e-6 inputs=VD,VQ,TLOAD
    /// ```
    Pmsm {
        /// The motor's own parameters/state.
        pmsm: Pmsm,
        /// Output names; length must equal 4 (`id`, `iq`, `omega_m`, `theta_e`).
        output_names: Vec<String>,
    },
    /// <!-- component -->
    /// # Phys2Sig (Physical-to-Signal Converter)
    /// **Purpose:** the only way a circuit quantity enters the signal domain.
    /// **Library:** Electrical Interface
    ///
    /// ## Description
    /// Zero block-graph inputs (it reads the circuit's own previous-step operating point
    /// directly). Its value is then an ordinary block output, read by any downstream block via
    /// `Signal::Block(this_block's_name)` exactly like any other source block
    /// (`Const`/`Pwl`/`Time`). Modeled directly on the physical/signal converter block a real
    /// block-diagram simulation tool requires: a physical port and a signal port are
    /// type-distinct there and cannot be wired together without one of these in between — the
    /// same rule, enforced the same way, at the netlist level instead of a GUI's wiring canvas.
    /// Named symmetrically with [`BlockKind::Sig2Phys`] (its write-direction counterpart):
    /// `phys2sig` reads a circuit quantity into the signal domain, `sig2phys` drives a
    /// signal-domain value onto a circuit quantity.
    ///
    /// ## Parameters
    /// - `node=<name>` — reads `V(node)`. Mutually exclusive with `branch=`.
    /// - `branch=<name>` — reads `I(branch)`. Mutually exclusive with `node=`.
    ///
    /// ## Errors
    /// - Both `node=` and `branch=` given — rejected at parse time: `'node' and 'branch' are
    ///   mutually exclusive (a phys2sig reads either a node voltage or a branch current, never
    ///   both)`.
    /// - Neither given — rejected at parse time: `kind='phys2sig' needs 'node=<name>' (reads
    ///   V(node)) or 'branch=<name>' (reads I(branch))`.
    /// - `branch=<name>` naming an element that has no MNA branch-current unknown at all —
    ///   rejected at build time (once the full circuit is known), not silently read back as
    ///   `0.0`: only `V`/`L`/`E` (VCVS)/`H` (CCVS) elements carry a branch-current unknown (see
    ///   `MnaBuilder::is_branch_device`); naming anything else (a resistor, a capacitor, a
    ///   diode, an ideal switch, ...) is `line N: device '<name>' kind='phys2sig'
    ///   branch='<branch>' has no current unknown available (only V/L/E/H sources carry a
    ///   branch-current unknown) -- insert a 0V voltage source in series in the branch you want
    ///   to measure and probe its current instead` — the standard SPICE "ammeter" idiom: a 0V
    ///   source in series perturbs nothing electrically, but (being a `V` element) gets its own
    ///   branch-current unknown for free, which `branch=` can then read.
    /// - `branch=<name>` naming an element that doesn't exist anywhere in the netlist at all
    ///   (not just the wrong type) — a distinct build-time error: `line N: device '<name>'
    ///   kind='phys2sig' branch='<branch>' names no such element`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=phys2sig node=<node_name>
    /// NAME kind=phys2sig branch=<branch_name>
    /// ```
    ///
    /// ## Example
    /// Reading a resistor-divider node's own voltage — verified end to end, `PROBE1` tracking
    /// `V(out)` exactly:
    /// ```text
    /// V1 in 0 5
    /// R1 in out 1k
    /// R2 out 0 1k
    /// PROBE1 kind=phys2sig node=out
    /// ```
    ///
    /// The ammeter idiom — measuring the current through `R1` (which has no branch-current
    /// unknown of its own) by inserting a 0V source `VAMM` in series and reading *that* instead;
    /// verified end to end, `IMEAS` tracking `5 / (1000 + 1000) = 0.0025` A exactly (the 0V
    /// source doesn't perturb the divider at all):
    /// ```text
    /// V1 in 0 5
    /// VAMM in mid 0
    /// R1 mid out 1000
    /// R2 out 0 1000
    /// IMEAS kind=phys2sig branch=VAMM
    /// ```
    Phys2Sig(Phys2SigTarget),
    /// <!-- component -->
    /// # Sig2Phys (Signal-to-Physical Converter)
    /// **Purpose:** the only legal way a signal-domain block's output drives a `V`/`I` source's
    /// magnitude or an ideal switch's gate.
    /// **Library:** Electrical Interface
    ///
    /// ## Description
    /// Closes the write-direction gap [`BlockKind::Phys2Sig`] doesn't (a phys2sig only ever
    /// reads). A
    /// `V`/`I` element's own literal value field in the netlist names this block directly (e.g.
    /// `V1 a 0 VDRV`, where `VDRV` is a declared `domain=voltage` block) — `general-mna`'s own
    /// `Expression::parse_scalar` already accepts a bare symbol there with no change needed on
    /// that side; `dae-runtime` requires, at validation time, that any such symbol naming a
    /// declared block resolve to exactly this kind with the matching `domain` for the device
    /// letter driving it (`V` requires `domain=voltage`, `I` requires `domain=current`) —
    /// `DaeError::SourceNotSig2PhysicalConverter` otherwise — and every step, substitutes this
    /// block's own just-computed output value into the circuit solve in that symbol's place.
    ///
    /// Also the *only* legal target for a [`GateBinding::Block`]'s own named block, and there
    /// *only* the `domain=voltage` form — `dae-runtime` rejects a `GateBinding` naming anything
    /// else, including a `domain=current` `Sig2Phys`, with `DaeError::GateTargetNotSig2Voltage`.
    /// An ideal switch's gate is itself a voltage (`V_GS` against `v_th`), not a distinct
    /// discrete-actuation signal domain, so there is no separate gate-only converter — one type,
    /// `Sig2Phys`, parameterized by [`PhysicalDomain`], is the whole Signal-to-PS boundary for
    /// "a signal-domain block's output drives a physical voltage or current," whether that
    /// voltage happens to source a node, gate a switch, or drive a current source. Purely an
    /// identity pass-through numerically (`value = input`) in every role; the `domain` field is
    /// what the enforcement keys on. One input.
    ///
    /// ## Parameters
    /// - `domain=<voltage|current>` — which physical quantity this converter drives. Required.
    /// - `in=<signal>` — the value to drive onto the physical side (one input).
    ///
    /// ## Errors
    /// - `domain=` missing entirely — rejected at parse time: `line N: device '<name>' missing
    ///   field 'domain'`.
    /// - `domain=` present but not `voltage`/`current` — rejected at parse time: `line N: device
    ///   '<name>' field 'domain' must be 'voltage' or 'current' (got '<value>')`.
    /// - Beyond that, none specific to this `kind=` — every real enforcement (a `V`/`I`/gate
    ///   target actually naming a `Sig2Phys` of the right domain) happens at `dae-runtime`'s
    ///   validation stage, not netlist parse time. `general-mna` itself only ever sees the
    ///   device-source half of a document (with `--devices <file>`, it doesn't see the netlist
    ///   at all), so it cannot detect a converter's name reused as an ordinary circuit *node* —
    ///   that mistake produces no parse error here. `dae-runtime` rejects it as
    ///   `DaeError::Sig2PhysUsedAsCircuitNode`: this block has no terminals and stamps nothing
    ///   into the MNA system, so a net sharing its name is merely undriven, and KCL solves it to
    ///   a silently wrong `0` rather than failing — a converter must always be *referenced by
    ///   name* (a source's value field, or a switch's `gate=`/`ctrl=`), never wired.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=sig2phys domain=voltage in=<signal>
    /// NAME kind=sig2phys domain=current in=<signal>
    /// ```
    ///
    /// ## Example
    /// A block-driven voltage source — verified end to end, `V1`'s magnitude tracking a
    /// `Const` through the converter exactly:
    /// ```text
    /// CMD kind=const value=5
    /// VDRV kind=sig2phys domain=voltage in=CMD
    /// V1 a 0 VDRV
    /// R1 a 0 1k
    /// ```
    ///
    /// A block-driven current source — verified end to end, `I1`'s magnitude tracking a
    /// `Const` through the converter exactly:
    /// ```text
    /// CMD kind=const value=0.1
    /// IDRV kind=sig2phys domain=current in=CMD
    /// I1 a 0 IDRV
    /// R1 a 0 1k
    /// ```
    Sig2Phys {
        /// Which physical quantity this converter drives — `V`-source targets and
        /// `GateBinding::Block` targets require `Voltage`; `I`-source targets require
        /// `Current`.
        domain: PhysicalDomain,
    },
}

/// Which physical quantity a [`BlockKind::Sig2Phys`] converter drives. Purely a type-level tag
/// for the enforcement sites in `dae-runtime` (`DaeError::GateTargetNotSig2Voltage`,
/// `DaeError::SourceNotSig2PhysicalConverter`) — evaluation itself (`value = input`) is
/// identical for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicalDomain {
    /// Drives a `V` source's magnitude, or a [`GateBinding::Block`] target.
    Voltage,
    /// Drives an `I` source's magnitude.
    Current,
}

/// One named block instance and where its inputs (if any) come from.
/// `Const`/`Pwc`/`Pwl`/`Waveform`/`Phys2Sig` blocks must have zero inputs; `Sum`/`Product` need one
/// input per sign/factor; `Gain`/
/// `StateSpace`/`TransferFunction`/`Vco`/`Saturation`/`Table`/`MathFn1`/`Sig2Phys`
/// each need exactly one; `Pid` needs exactly one (the error signal)
/// when its `clamp` is `PidClamp::Fixed`, or exactly three (`error, clamp_lo, clamp_hi`, in that
/// order) when `PidClamp::Dynamic`; `MathFn2` needs two; `MathFn3` needs three;
/// `CoordinateTransform` needs `kind.input_count()` (3 for `Clarke`/`ClarkeInv`, 4 for the
/// others — see [`continuous_blocks::CoordinateTransform::input_count`]); `Pmsm` needs exactly
/// three (`vd`, `vq`, `t_load`, in that order). Evaluated once per step in the causal order
/// `topological_order` (in `dae-runtime`) derives from the slice's own `Signal::Block` dependency graph — *not*
/// the order the slice happens to be given in; a `Signal::Block` input may name any block in
/// the same slice regardless of declared position (source blocks, naturally, need none, and a
/// genuine cycle among these edges is rejected as `DaeError::AlgebraicLoop` before any step
/// runs — see `topological_order`'s own doc comment for how).
///
/// **Generic field-parsing errors**, shared by every `kind=` block and not repeated per
/// variant below: a field required by that `kind` but absent from the line is `line N: device
/// '<name>' missing field '<key>'`; present but not parseable as the expected type (almost
/// always `f64`) is `line N: device '<name>' field '<key>' is not a number`; present but read
/// by nothing for that `kind` (a misspelling, or a field that belongs to another kind) is
/// `line N: device '<name>' unknown field '<key>' (kind=<kind> accepts only: ...)`, listing the
/// keys that `kind` consulted on that line — the same shape a device card's unknown field
/// reports. Nothing on a block line is silently ignored: the only keys a kind accepts are the
/// ones documented in its own `## Parameters` (plus `ic=`/`y0=` below), and the escape hatches
/// (`cscript`/`pyblock`/`pyfunc`/`octfunc`/`octblock`) take no free-form parameters from the
/// netlist line either.
///
/// **Initial conditions.** Every block that carries state across steps accepts an optional
/// `ic=` (each such variant's own `## Parameters` says what it means there); an `ic=` on any
/// other kind is rejected rather than dropped — `line N: device '<name>' field 'ic': this kind
/// has no state to initialize -- 'ic' is accepted on ...` — as is a non-finite entry
/// (`initial condition must be finite`). `cscript`/`pyblock`/`octblock` initialize their own
/// state and take no `ic=`.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockInstance {
    /// This block's own unique name.
    pub name: String,
    /// What this block computes.
    pub kind: BlockKind,
    /// Where each input comes from, in the order `kind` expects.
    pub inputs: Vec<Signal>,
    /// The state this block starts the run in, or `None` to start from rest (every state zero,
    /// every logic output low) exactly as before this field existed. Already resolved into the
    /// block's own state layout and already validated by the netlist parser, so `dae-runtime`
    /// only has to copy it in:
    ///
    /// - `StateSpace`/`DiscreteStateSpace` — the state vector $x(0)$, one entry per state.
    /// - `TransferFunction`/`DiscreteTransferFunction` — the state vector of the controllable
    ///   canonical realization (`ic=`), or the settled state a netlist's `y0=` resolves to (see
    ///   `continuous_blocks::TransferFunction::settled_state`).
    /// - `Pid` — the canonical state holding the controller output at the netlist's `ic=` with
    ///   zero error; `DiscretePid` — one entry, the integrator's accumulated value (`ic / ki`).
    /// - `Vco`/`PhaseShiftPwm` — one entry, the phase in cycles, $0 \le \phi_0 < 1$.
    /// - `Pmsm` — four entries, $(i_d, i_q, \omega_m, \theta_e)$.
    /// - `Hysteresis`/`SrLatch`/`FlipFlop` — one entry, `0.0` or `1.0`, the initial output.
    /// - `Counter` — one entry, the (integer-valued) initial count.
    ///
    /// `None` on every other kind: a stateless block has nothing to initialize, and
    /// `cscript`/`pyblock`/`octblock` own their initialization themselves.
    pub ic: Option<Vec<f64>>,
}

/// How one ideal switch's gate state is resolved, every step: always from a named block's
/// current output, on while it's `>= 0.5`. No non-block-driven variant exists — even a
/// permanently-off gate is an explicit `Const(0.0)` wired through a `domain=voltage`
/// [`BlockKind::Sig2Phys`],
/// the same as every other gate — and no bare carrier-comparator variant exists either: that
/// comparison now lives entirely inside gate-driving `BlockKind`s themselves
/// ([`BlockKind::Pwm`]/[`BlockKind::PhaseShiftPwm`], or a hand-built chain of ordinary blocks),
/// so `GateBinding` has exactly one job — reading a number and thresholding it — regardless of
/// what produced that number: a modulator's own main/complement output, a
/// [`BlockKind::Hysteresis`] block (no carrier at all, event-driven bang-bang switching), or any
/// other block a caller composes.
#[derive(Debug, Clone, PartialEq)]
pub enum GateBinding {
    /// <!-- component -->
    /// # Gate Binding
    /// **Purpose:** how an ideal switch's gate state is resolved from a named block, every step.
    /// **Library:** Electrical Interface
    ///
    /// ## Description
    /// This is not a `kind=` block itself — it's the `gate=`/`ctrl=` field pair on a
    /// `kind=ideal_switch` device line. `GateBinding` has exactly one variant and exactly one
    /// job: reading a named block's current output and thresholding it at `>= 0.5`. No
    /// non-block-driven variant exists — even a permanently-off gate is an explicit `Const(0.0)`
    /// wired through a `domain=voltage` [`BlockKind::Sig2Phys`], the same as every other gate.
    /// The named block (`ctrl=`) must itself be a `domain=voltage` [`BlockKind::Sig2Phys`] — an
    /// ideal switch's gate is itself a voltage ($V_{GS}$ against $v_{th}$), not a distinct
    /// discrete-actuation signal domain (and never `domain=current`, even though `Sig2Phys`
    /// itself allows that domain for `I`-source targets).
    ///
    /// ## Parameters
    /// - `gate=block` — the only accepted value; every gate is block-driven.
    /// - `ctrl=<name>` — the name of a declared `domain=voltage` [`BlockKind::Sig2Phys`] block.
    ///
    /// ## Errors
    /// - `gate=` missing entirely on a `kind=ideal_switch` line — rejected at parse time:
    ///   `missing field 'gate' (gate=block ctrl=<name> -- every gate is block-driven, see this
    ///   file's own module doc comment)`.
    /// - `gate=` present but not `block` — rejected at parse time: `unknown gate spec '<value>'
    ///   (only gate=block ctrl=<name> exists -- every gate is block-driven)`.
    /// - `ctrl=` naming a block that isn't a `domain=voltage` [`BlockKind::Sig2Phys`] (including
    ///   a `domain=current` one) — rejected at `dae-runtime`'s validation stage (not netlist
    ///   parse time): `DaeError::GateTargetNotSig2Voltage`.
    ///
    /// ## Netlist form
    /// ```text
    /// NAME kind=ideal_switch r_on=<f64> g_breakdown=<f64> v_breakdown=<f64> g_off=<f64> \
    ///      v_th=<f64> g_on=<f64> gate=block ctrl=<sig2phys_voltage_block_name>
    /// ```
    ///
    /// ## Example
    /// An ideal switch permanently held on via a `Const(1)` wired through `Sig2Phys` — the
    /// plain `D1 in out idealswitchmodel` line gives the electrical connectivity (drain, source;
    /// the model name is unused for a `kind=ideal_switch`-overridden device), the same-named
    /// `kind=ideal_switch` line supplies the physics and gate. Verified end to end: `V(out)`
    /// settles to `5 * 1000 / (1000 + 0.1) ≈ 4.9995` (a fully-on 0.1 Ω ideal switch in series
    /// with a 1 kΩ load):
    /// ```text
    /// ONVAL kind=const value=1
    /// ONGATE kind=sig2phys domain=voltage in=ONVAL
    /// V1 in 0 5
    /// D1 in out idealswitchmodel
    /// D1 kind=ideal_switch r_on=0.1 g_breakdown=0 v_breakdown=-100 g_off=0 v_th=0.5 g_on=5 \
    ///      gate=block ctrl=ONGATE
    /// R1 out 0 1k
    /// ```
    ///
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
    /// practice, since the *only* legal `GateBinding` target is a `domain=voltage`
    /// `BlockKind::Sig2Phys` converter (enforced before any step runs), and `Sig2Phys` itself
    /// rejects a `Vector` input at evaluation time — there is no way for a validly-targeted gate
    /// to ever see one here. Also panics if `name` isn't in `outputs` at all, exactly as before
    /// this variant existed (equally unreachable, for the same "checked before any step runs"
    /// reason).
    pub fn resolve(&self, outputs: &BTreeMap<String, SignalValue>) -> SwitchState {
        let GateBinding::Block(name) = self;
        let value = outputs[name.as_str()].as_scalar().unwrap_or_else(|| {
            panic!(
                "gate '{name}' resolved to a vector signal -- unreachable, since Sig2Phys \
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
/// mistargeted block actually is (e.g. `DaeError::GateTargetNotSig2Voltage`/
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
        BlockKind::DiscreteStateSpace { .. } => "discretestatespace",
        BlockKind::DiscreteTransferFunction { .. } => "discretetf",
        BlockKind::DiscretePid { .. } => "discretepid",
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
        BlockKind::LogicGate(op) => op.name(),
        BlockKind::SrLatch { .. } => "srlatch",
        BlockKind::FlipFlop { kind, .. } => kind.name(),
        BlockKind::Counter { .. } => "counter",
        BlockKind::CScript { .. } => "cscript",
        BlockKind::PyBlock { .. } => "pyblock",
        BlockKind::PyFunction { .. } => "pyfunc",
        BlockKind::OctFunc { .. } => "octfunc",
        BlockKind::OctBlock { .. } => "octblock",
        BlockKind::CoordinateTransform { kind, .. } => kind.name(),
        BlockKind::Pmsm { .. } => "pmsm",
        BlockKind::Phys2Sig(_) => "phys2sig",
        BlockKind::Sig2Phys {
            domain: PhysicalDomain::Voltage,
        } => "sig2phys(domain=voltage)",
        BlockKind::Sig2Phys {
            domain: PhysicalDomain::Current,
        } => "sig2phys(domain=current)",
    }
}
