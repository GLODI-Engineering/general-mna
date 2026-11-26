//! A netlist number is finite, or the deck is refused (#54).
//!
//! `parse_spice_number` falls back to Rust's own float parser, which accepts `nan`, `inf` and
//! `-inf`, and #43 made that parser the single grammar for every field a netlist carries. So
//! until this was closed, every one of these decks built without a word and then produced a
//! number nothing in the output described:
//!
//! | deck | what the build produced |
//! |---|---|
//! | `R1 in 0 nan` | $A = \begin{bmatrix}\text{NaN} & 1\\ 1 & 0\end{bmatrix}$, and the solve returns NaN for *every* unknown, including branches R1 is not in |
//! | `V1 in 0 nan` | $u = [0, 0]$ — the NaN is dropped and the source silently becomes $0\text{V}$ |
//! | `R1 in 0 inf` | a conductance of exactly $0$ — a silent open circuit, here a singular system |
//! | `R1 in 0 -inf` | a conductance of $-0$ |
//! | `C1 in 0 inf` | $K$ carries an infinity |
//! | `.param rl=nan` | the NaN reaches $A$ through the parameter |
//! | `r_on=nan` | `shared_r_on` is NaN, and every gated-on switch in the deck is stamped with it |
//!
//! Each case here asserts the **diagnosis**: that the deck is refused, and that the refusal
//! names the device whose field carried the value. A run that stops with the device's name is
//! the whole point — one bad character used to poison a solution with nothing pointing at it.

use general_mna::{build_system, parse_and_flatten, MnaBuilder};
use general_spice_core::dialect::Dialect;

/// Builds through the ordinary two-step path (`parse_and_flatten` then `MnaBuilder`), the one
/// a plain electrical deck takes, and returns the refusal.
fn electrical_error(source: &str) -> String {
    let statements = parse_and_flatten(source, Dialect::Ngspice)
        .unwrap_or_else(|e| panic!("parse refused {source:?} before the build could: {e}"));
    match MnaBuilder::new(Dialect::Ngspice).build_statements(&statements) {
        Ok(_) => panic!("built {source:?} without refusing a non-finite value"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_resistor_valued_nan_is_refused_by_name() {
    let error = electrical_error("V1 in 0 5\nR1 in 0 nan\n");
    assert!(error.contains("R1"), "does not name the device: {error}");
    assert!(error.contains("finite"), "does not say why: {error}");
}

#[test]
fn a_capacitor_valued_inf_is_refused_by_name() {
    let error = electrical_error("V1 in 0 5\nR1 in 0 1k\nC1 in 0 inf\n");
    assert!(error.contains("C1"), "does not name the device: {error}");
    assert!(error.contains("finite"), "does not say why: {error}");
}

/// The worst of the set, because it does not even produce a NaN: the source value was dropped
/// and the deck simulated as though `V1` were $0\text{V}$.
#[test]
fn a_source_valued_nan_is_refused_rather_than_read_as_zero() {
    let error = electrical_error("V1 in 0 nan\nR1 in 0 1k\n");
    assert!(error.contains("V1"), "does not name the device: {error}");
    assert!(error.contains("finite"), "does not say why: {error}");
}

/// `inf` is the one that looks deliberate — an "ideal" resistance, an "infinite" load. It is
/// not an idiom this project supports: an open branch is written by leaving the device out.
#[test]
fn an_infinite_resistance_is_refused_rather_than_read_as_an_open_branch() {
    for spelling in ["inf", "-inf", "INF", "Inf", "infinity", "NaN", "-nan"] {
        let error = electrical_error(&format!("V1 in 0 5\nR1 in 0 {spelling}\n"));
        assert!(
            error.contains("R1") && error.contains("finite"),
            "'{spelling}' gave an unhelpful refusal: {error}"
        );
    }
}

#[test]
fn a_non_finite_param_default_is_refused() {
    let error = electrical_error(".param rl=nan\nV1 in 0 5\nR1 in 0 {rl}\n");
    assert!(error.contains("finite"), "does not say why: {error}");
}

/// A waveform argument goes through the same grammar, and the parameter-list validator already
/// refuses a list it cannot parse by name.
#[test]
fn a_non_finite_waveform_argument_is_refused() {
    let error = electrical_error("V1 in 0 PULSE(0 nan 1m 100n 100n 2m 4m)\nR1 in 0 1k\n");
    assert!(error.contains("V1"), "does not name the device: {error}");
}

/// A `kind=` block field is the same grammar again (#43), so it is refused the same way, with
/// the line, the device and the field named.
#[test]
fn a_non_finite_block_field_is_refused_with_its_field_named() {
    let error = build_system("K1 kind=gain k=inf in=X\n", Dialect::Ngspice).unwrap_err();
    assert!(error.contains("K1"), "does not name the device: {error}");
    assert!(error.contains("'k'"), "does not name the field: {error}");
}

/// An ideal switch's `r_on` reaches `general-simulator` as `shared_r_on`, outside the matrix
/// entirely, so nothing downstream would have refused it either.
#[test]
fn a_non_finite_ideal_switch_r_on_is_refused() {
    let source = "D1 in 0 dm\n\
                  D1 kind=ideal_switch r_on=nan body_g_breakdown=0 body_v_breakdown=-100 \
                  body_g_off=0 body_v_f=1 body_g_on=1 gate=block ctrl=G\n";
    let error = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(error.contains("D1"), "does not name the device: {error}");
    assert!(error.contains("'r_on'"), "does not name the field: {error}");
}

/// The other direction: every ordinary spelling of a finite number still builds. A refusal
/// that also rejected `1n` or `300n` would be worse than the hole it closed.
#[test]
fn ordinary_finite_values_still_build() {
    for value in [
        "5", "1k", "1MEG", "300n", "1e-9", "47p", "-2.5", "0", "1.5u", "300.",
    ] {
        let source = format!("V1 in 0 5\nR1 in 0 {value}\n");
        let statements = parse_and_flatten(&source, Dialect::Ngspice).unwrap();
        assert!(
            MnaBuilder::new(Dialect::Ngspice)
                .build_statements(&statements)
                .is_ok(),
            "refused the finite value '{value}'"
        );
    }
}
