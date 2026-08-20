//! Time-varying independent-source functions: `SIN`, `PULSE`, `EXP`, `PWL`, `SFFM` — the same
//! five transient-function forms `spice-core`'s own grammar reference documents as common to
//! both ngspice and Xyce (`docs/GRAMMAR.md` in that repo, "same names/arg order in both, minor
//! default-value wording differences"), same argument order.
//!
//! `spice-core` tokenizes a `V`/`I` element's parameters purely lexically (whitespace-split,
//! parentheses left attached to whichever token they landed on — confirmed directly this
//! session, not assumed): `SIN(0 10 1000)` arrives as `["SIN(0", "10", "1000)"]`, three plain
//! string tokens, with no notion that they form one function call. This module owns turning
//! that back into a real function: [`TransientFunction::parse`] detects and reassembles the
//! call, [`TransientFunction::value_at`] evaluates it at a given time.
//!
//! This module has **no notion of "now"** on its own — a [`TransientFunction`] is pure data
//! plus a pure `f64 -> f64` function. Feeding a source's *current* value into a solve at each
//! step is the caller's job, exactly the same delegation this crate already uses for a
//! PWL diode's per-step Norton current (`{name}_Ioff`, an [`crate::Expression::symbol`] the
//! caller supplies numerically via [`crate::MnaSystem::evaluate`]'s `values` map) — a
//! `TransientFunction`-bearing source is stamped as that same kind of named symbol (see
//! `builder.rs`'s `source_value`), not as a baked-in literal, so the caller supplies
//! `values.insert(name, transient_sources[name].value_at(t))` before every step, using
//! [`crate::MnaSystem::transient_sources`] to know which sources need it.

/// One parsed transient-function call, in the same field order and names ngspice/Xyce use.
#[derive(Debug, Clone, PartialEq)]
pub enum TransientFunction {
    /// `SIN(V0 VA FREQ TD THETA PHASE)` — a damped sinusoid superimposed on an offset,
    /// starting flat at `V0` until `TD`, then `V0 + VA*sin(2*pi*FREQ*(t-TD) + PHASE) *
    /// exp(-(t-TD)*THETA)` (`THETA=0` is a plain undamped sinusoid; `FREQ` in Hz, `PHASE` in
    /// degrees, matching every SPICE dialect's own convention for this source).
    Sin {
        v0: f64,
        va: f64,
        freq: f64,
        td: f64,
        theta: f64,
        phase: f64,
    },
    /// `PULSE(V1 V2 TD TR TF PW PER)` — a periodic trapezoid: `V1` until `TD`, linear ramp to
    /// `V2` over `TR`, hold at `V2` for `PW`, linear ramp back to `V1` over `TF`, hold at `V1`
    /// for the remainder of one period `PER`, repeating. `TR`/`TF` of exactly `0` are treated
    /// as instantaneous edges (division by a real, if tiny, ramp time is what every real SPICE
    /// implementation does instead; `0` is special-cased here to avoid dividing by zero).
    Pulse {
        v1: f64,
        v2: f64,
        td: f64,
        tr: f64,
        tf: f64,
        pw: f64,
        per: f64,
    },
    /// `EXP(V1 V2 TD1 TAU1 TD2 TAU2)` — flat at `V1` until `TD1`, exponential transition
    /// toward `V2` with time constant `TAU1` starting at `TD1`, exponential transition back
    /// toward `V1` with time constant `TAU2` starting at `TD2`.
    Exp {
        v1: f64,
        v2: f64,
        td1: f64,
        tau1: f64,
        td2: f64,
        tau2: f64,
    },
    /// `PWL(t1 v1 t2 v2 ...)` — linearly interpolated between explicit `(time, value)`
    /// breakpoints (unlike `dae_runtime::block_graph::BlockKind::Pwl`, which is
    /// piecewise-*constant* — this is real SPICE `PWL`, piecewise-*linear*, matching the
    /// source syntax's own name). Held at the first point's value before the first breakpoint,
    /// and at the last point's value after the last one.
    Pwl(PwlPoints),
    /// `SFFM(V0 VA FC MDI FS)` — single-frequency FM: `V0 + VA*sin(2*pi*FC*t + MDI*sin(2*pi*FS*t))`.
    Sffm {
        v0: f64,
        va: f64,
        fc: f64,
        mdi: f64,
        fs: f64,
    },
}

/// A breakpoint list for [`TransientFunction::Pwl`] — heap-allocated (unlike every other
/// variant's plain `f64` fields): a real `PWL` source can have an arbitrary number of
/// breakpoints, and a fixed-capacity inline array sized for a generous case would make every
/// `TransientFunction` value pay for that space even for a plain `SIN`/`PULSE`/`EXP`/`SFFM`
/// (`clippy::large_enum_variant` flags exactly this).
#[derive(Debug, Clone, PartialEq)]
pub struct PwlPoints(Vec<(f64, f64)>);

impl PwlPoints {
    pub fn new(points: &[(f64, f64)]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        Some(PwlPoints(points.to_vec()))
    }

    pub fn as_slice(&self) -> &[(f64, f64)] {
        &self.0
    }
}

impl TransientFunction {
    /// Evaluates this source at time `t` (seconds), per the standard SPICE definition for its
    /// own function form.
    pub fn value_at(&self, t: f64) -> f64 {
        match self {
            TransientFunction::Sin {
                v0,
                va,
                freq,
                td,
                theta,
                phase,
            } => {
                if t < *td {
                    *v0
                } else {
                    let dt = t - td;
                    let phase_rad = phase.to_radians();
                    let envelope = if *theta == 0.0 {
                        1.0
                    } else {
                        (-dt * theta).exp()
                    };
                    v0 + va * (2.0 * std::f64::consts::PI * freq * dt + phase_rad).sin() * envelope
                }
            }
            TransientFunction::Pulse {
                v1,
                v2,
                td,
                tr,
                tf,
                pw,
                per,
            } => {
                let t = if t < *td {
                    0.0
                } else {
                    (t - td) % per.max(f64::EPSILON)
                };
                if t < *tr {
                    if *tr <= 0.0 {
                        *v2
                    } else {
                        v1 + (v2 - v1) * (t / tr)
                    }
                } else if t < tr + pw {
                    *v2
                } else if t < tr + pw + tf {
                    if *tf <= 0.0 {
                        *v1
                    } else {
                        v2 + (v1 - v2) * ((t - tr - pw) / tf)
                    }
                } else {
                    *v1
                }
            }
            TransientFunction::Exp {
                v1,
                v2,
                td1,
                tau1,
                td2,
                tau2,
            } => {
                if t < *td1 {
                    *v1
                } else if t < *td2 {
                    v1 + (v2 - v1) * (1.0 - (-(t - td1) / tau1.max(f64::EPSILON)).exp())
                } else {
                    let rise_at_td2 =
                        v1 + (v2 - v1) * (1.0 - (-(td2 - td1) / tau1.max(f64::EPSILON)).exp());
                    rise_at_td2
                        + (v1 - rise_at_td2) * (1.0 - (-(t - td2) / tau2.max(f64::EPSILON)).exp())
                }
            }
            TransientFunction::Pwl(points) => {
                let points = points.as_slice();
                if t <= points[0].0 {
                    return points[0].1;
                }
                let last = points[points.len() - 1];
                if t >= last.0 {
                    return last.1;
                }
                for window in points.windows(2) {
                    let (t0, v0) = window[0];
                    let (t1, v1) = window[1];
                    if t >= t0 && t <= t1 {
                        if (t1 - t0).abs() < f64::EPSILON {
                            return v1;
                        }
                        return v0 + (v1 - v0) * ((t - t0) / (t1 - t0));
                    }
                }
                last.1
            }
            TransientFunction::Sffm {
                v0,
                va,
                fc,
                mdi,
                fs,
            } => {
                let two_pi = 2.0 * std::f64::consts::PI;
                v0 + va * (two_pi * fc * t + mdi * (two_pi * fs * t).sin()).sin()
            }
        }
    }

    /// Detects and parses one of the five transient-function forms from an already-tokenized
    /// parameter list (as `spice-core` hands back — see the module doc comment for exactly how
    /// mangled that tokenization is: parentheses stuck to whichever token they landed on).
    /// Returns `None` if `tokens` doesn't start with a recognized function name, so the caller
    /// can fall back to treating the source as an ordinary static value.
    pub fn parse(tokens: &[String]) -> Option<TransientFunction> {
        let first = tokens.first()?;
        let paren = first.find('(')?;
        let name = first[..paren].to_ascii_uppercase();

        // Reassemble the numeric argument list: the rest of `first` after '(', then every
        // later token, with a trailing ')' stripped off whichever token has it.
        let mut args: Vec<f64> = Vec::new();
        let first_rest = first[paren + 1..].trim_end_matches(')');
        if !first_rest.is_empty() {
            args.push(first_rest.parse().ok()?);
        }
        for token in &tokens[1..] {
            let cleaned = token.trim_end_matches(')');
            if cleaned.is_empty() {
                continue;
            }
            args.push(cleaned.parse().ok()?);
        }

        match name.as_str() {
            "SIN" => {
                let get = |i: usize, default: f64| args.get(i).copied().unwrap_or(default);
                Some(TransientFunction::Sin {
                    v0: get(0, 0.0),
                    va: get(1, 0.0),
                    freq: get(2, 0.0),
                    td: get(3, 0.0),
                    theta: get(4, 0.0),
                    phase: get(5, 0.0),
                })
            }
            "PULSE" => {
                if args.len() < 2 {
                    return None;
                }
                let get = |i: usize, default: f64| args.get(i).copied().unwrap_or(default);
                Some(TransientFunction::Pulse {
                    v1: args[0],
                    v2: args[1],
                    td: get(2, 0.0),
                    tr: get(3, 0.0),
                    tf: get(4, 0.0),
                    pw: get(5, f64::MAX / 4.0),
                    per: get(6, f64::MAX / 4.0),
                })
            }
            "EXP" => {
                if args.len() < 2 {
                    return None;
                }
                let get = |i: usize, default: f64| args.get(i).copied().unwrap_or(default);
                Some(TransientFunction::Exp {
                    v1: args[0],
                    v2: args[1],
                    td1: get(2, 0.0),
                    tau1: get(3, 1.0),
                    td2: get(4, get(2, 0.0)),
                    tau2: get(5, 1.0),
                })
            }
            "PWL" => {
                if args.len() < 2 || args.len() % 2 != 0 {
                    return None;
                }
                let points: Vec<(f64, f64)> = args.chunks_exact(2).map(|c| (c[0], c[1])).collect();
                PwlPoints::new(&points).map(TransientFunction::Pwl)
            }
            "SFFM" => {
                let get = |i: usize, default: f64| args.get(i).copied().unwrap_or(default);
                Some(TransientFunction::Sffm {
                    v0: get(0, 0.0),
                    va: get(1, 0.0),
                    fc: get(2, 0.0),
                    mdi: get(3, 0.0),
                    fs: get(4, 0.0),
                })
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(s: &str) -> Vec<String> {
        s.split_whitespace().map(String::from).collect()
    }

    #[test]
    fn sin_matches_hand_computed_values() {
        let tf = TransientFunction::parse(&toks("SIN(0 10 1000 0 0 0)")).unwrap();
        assert!((tf.value_at(0.0) - 0.0).abs() < 1e-9);
        // Quarter period: t = 1/(4*1000) = 250us -> sin(pi/2) = 1 -> value = 10
        assert!((tf.value_at(2.5e-4) - 10.0).abs() < 1e-6);
        // Half period -> back to 0
        assert!((tf.value_at(5e-4) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn sin_respects_delay_and_damping() {
        let tf = TransientFunction::parse(&toks("SIN(1 2 100 0.01 5 0)")).unwrap();
        assert_eq!(tf.value_at(0.0), 1.0); // before td, flat at v0
        assert_eq!(tf.value_at(0.005), 1.0);
        // Right at td, undamped sin(0) = 0, so value = v0 exactly.
        assert!((tf.value_at(0.01) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pulse_matches_hand_computed_trapezoid() {
        // V1=0 V2=10 TD=0 TR=1 TF=1 PW=2 PER=10 (arbitrary time units for a clean hand check)
        let tf = TransientFunction::parse(&toks("PULSE(0 10 0 1 1 2 10)")).unwrap();
        assert_eq!(tf.value_at(0.0), 0.0);
        assert_eq!(tf.value_at(0.5), 5.0); // mid-rise
        assert_eq!(tf.value_at(1.0), 10.0); // start of plateau
        assert_eq!(tf.value_at(2.5), 10.0); // mid-plateau
        assert_eq!(tf.value_at(3.5), 5.0); // mid-fall
        assert_eq!(tf.value_at(4.0), 0.0); // back to v1
        assert_eq!(tf.value_at(9.9), 0.0); // still low, before next period
        assert_eq!(tf.value_at(10.5), 5.0); // second period, mid-rise
    }

    #[test]
    fn exp_matches_hand_computed_values() {
        // V1=0 V2=1 TD1=0 TAU1=1 TD2=1000 TAU2=1 (TD2 far away so only the rise matters here)
        let tf = TransientFunction::parse(&toks("EXP(0 1 0 1 1000 1)")).unwrap();
        assert_eq!(tf.value_at(0.0), 0.0);
        // At t=tau1, rise = 1 - e^-1 ~= 0.6321
        assert!((tf.value_at(1.0) - (1.0 - (-1.0_f64).exp())).abs() < 1e-9);
    }

    #[test]
    fn pwl_interpolates_linearly_and_holds_ends() {
        let tf = TransientFunction::parse(&toks("PWL(0 0 1 10 2 0)")).unwrap();
        assert_eq!(tf.value_at(-1.0), 0.0); // before first point: held
        assert_eq!(tf.value_at(0.0), 0.0);
        assert_eq!(tf.value_at(0.5), 5.0); // linear interpolation
        assert_eq!(tf.value_at(1.0), 10.0);
        assert_eq!(tf.value_at(1.5), 5.0);
        assert_eq!(tf.value_at(2.0), 0.0);
        assert_eq!(tf.value_at(3.0), 0.0); // after last point: held
    }

    #[test]
    fn sffm_matches_hand_computed_value_at_t_zero() {
        let tf = TransientFunction::parse(&toks("SFFM(0 5 1000 10 100)")).unwrap();
        // At t=0, both sines are 0 -> value = v0 = 0.
        assert_eq!(tf.value_at(0.0), 0.0);
    }

    #[test]
    fn non_function_tokens_return_none() {
        assert!(TransientFunction::parse(&toks("10")).is_none());
        assert!(TransientFunction::parse(&toks("DC 5")).is_none());
    }
}
