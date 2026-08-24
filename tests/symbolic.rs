use std::collections::BTreeMap;

use general_mna::{Matrix, MnaBuilder};
use general_spice_core::Dialect;

fn build(netlist: &str) -> general_mna::MnaSystem {
    MnaBuilder::new(Dialect::Ngspice)
        .build_fragment(netlist)
        .unwrap()
}

fn state_index(states: &[String], name: &str) -> usize {
    states.iter().position(|item| item == name).unwrap()
}

#[test]
fn symbolic_rc_lowpass_matches_hand_derived_transfer_function() {
    // dV2/dt = -(1/RC) V2 + (1/RC) V1  =>  G(s) = (1/RC) / (s + 1/RC)
    let system = build("V1 1 0 DC 1\nR1 1 2 R\nC1 2 0 C");
    let state_space = system.to_symbolic_state_space().unwrap();
    assert_eq!(state_space.states, ["V(2)"]);
    assert_eq!(state_space.inputs, ["V1"]);

    let tf = state_space
        .transfer_function(state_index(&state_space.states, "V(2)"), 0)
        .unwrap();

    let values = BTreeMap::from([("R".to_string(), 1000.0), ("C".to_string(), 1e-6)]);
    let denominator: Vec<f64> = tf
        .denominator
        .iter()
        .map(|c| c.evaluate(&values).unwrap())
        .collect();
    let numerator: Vec<f64> = tf
        .numerator
        .iter()
        .map(|c| c.evaluate(&values).unwrap())
        .collect();

    // 1 / (RC) = 1 / (1000 * 1e-6) = 1000
    assert_eq!(denominator.len(), 2);
    assert!((denominator[0] - 1.0).abs() < 1e-9);
    assert!((denominator[1] - 1000.0).abs() < 1e-6);
    assert_eq!(numerator.len(), 1);
    assert!((numerator[0] - 1000.0).abs() < 1e-6);
}

#[test]
fn symbolic_state_space_matches_numeric_reduction_state_order() {
    let system = build("V1 1 0 DC 1\nR1 1 2 R\nC1 2 0 C");
    let symbolic_ss = system.to_symbolic_state_space().unwrap();
    let numeric_ss = system
        .evaluate(&BTreeMap::from([
            ("R".to_string(), 1000.0),
            ("C".to_string(), 1e-6),
        ]))
        .unwrap()
        .to_state_space(1e-12)
        .unwrap();

    assert_eq!(symbolic_ss.states, numeric_ss.states);
    assert_eq!(symbolic_ss.mna_state_indices, numeric_ss.mna_state_indices);
}

#[test]
fn faddeev_leverrier_matches_independent_numeric_recursion_for_fourth_order_ladder() {
    // A 4th-order LC ladder low-pass filter: two inductors, two capacitors.
    let netlist = "V1 1 0 DC 1\n\
                   R1 1 2 Rs\n\
                   L1 2 3 L1v\n\
                   C1 3 0 C1v\n\
                   L2 3 4 L2v\n\
                   C2 4 0 C2v\n\
                   R2 4 0 Rl";
    let system = build(netlist);
    let values = BTreeMap::from([
        ("Rs".to_string(), 2.5),
        ("L1v".to_string(), 2e-3),
        ("C1v".to_string(), 4e-6),
        ("L2v".to_string(), 3e-3),
        ("C2v".to_string(), 5e-6),
        ("Rl".to_string(), 50.0),
    ]);

    let symbolic_ss = system.to_symbolic_state_space().unwrap();
    assert_eq!(symbolic_ss.states.len(), 4, "expected 4 reactive states");

    let numeric_ss = system
        .evaluate(&values)
        .unwrap()
        .to_state_space(1e-12)
        .unwrap();
    assert_eq!(symbolic_ss.states, numeric_ss.states);

    let output_index = state_index(&symbolic_ss.states, "V(4)");
    let tf = symbolic_ss.transfer_function(output_index, 0).unwrap();

    let denominator: Vec<f64> = tf
        .denominator
        .iter()
        .map(|c| c.evaluate(&values).unwrap())
        .collect();
    let numerator: Vec<f64> = tf
        .numerator
        .iter()
        .map(|c| c.evaluate(&values).unwrap())
        .collect();

    // Independently re-derive the same characteristic polynomial and
    // adjugate numerically (fresh implementation, not reusing the crate's
    // Faddeev-LeVerrier) to cross-check the symbolic recursion. Coefficients
    // span many orders of magnitude here, so compare with a relative
    // tolerance rather than a fixed absolute one.
    fn assert_close_relative(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() / scale < 1e-9,
            "{actual} vs {expected}"
        );
    }

    let expected = numeric_faddeev_leverrier(&numeric_ss.a);
    assert_eq!(denominator.len(), expected.denominator.len());
    for (actual, expected) in denominator.iter().zip(expected.denominator.iter()) {
        assert_close_relative(*actual, *expected);
    }

    let expected_numerator = expected.numerator_for(output_index, &numeric_ss.b, 0);
    assert_eq!(numerator.len(), expected_numerator.len());
    for (actual, expected) in numerator.iter().zip(expected_numerator.iter()) {
        assert_close_relative(*actual, *expected);
    }
}

#[test]
fn invalid_state_or_input_index_reports_bounds_error() {
    let system = build("V1 1 0 DC 1\nR1 1 2 R\nC1 2 0 C");
    let state_space = system.to_symbolic_state_space().unwrap();

    assert!(state_space.transfer_function(5, 0).is_err());
    assert!(state_space.transfer_function(0, 5).is_err());
}

#[test]
fn capacitor_only_loop_defers_singularity_to_evaluation() {
    // Structural (exact-zero) pivoting cannot see that Ca + Cc - Ca - Cc
    // cancels for *these specific* symbols, since Expression never combines
    // like symbolic terms. Reduction itself succeeds; the singularity only
    // appears once a coefficient is evaluated at concrete parameter values.
    // A source is required to observe this: with no source at all, B has no
    // columns and the (trivially zero) A matrix never actually exercises the
    // singular K_ss block.
    let system = build("C1 1 2 Ca\nC2 2 3 Cb\nC3 3 1 Cc\nI1 0 1 DC Iin");
    let state_space = system.to_symbolic_state_space().unwrap();
    assert_eq!(state_space.states.len(), 3);
    assert_eq!(state_space.inputs, ["I1"]);

    let values = BTreeMap::from([
        ("Ca".to_string(), 1e-6),
        ("Cb".to_string(), 1e-6),
        ("Cc".to_string(), 1e-6),
        ("Iin".to_string(), 1e-3),
    ]);
    let hits_division_by_zero = state_space
        .b
        .iter()
        .any(|entry| entry.evaluate(&values).is_err());
    assert!(
        hits_division_by_zero,
        "expected at least one B-matrix entry to divide by zero for this topology"
    );
}

struct NumericFaddeevLeverrier {
    denominator: Vec<f64>,
    adjugate_terms: Vec<Matrix<f64>>,
}

impl NumericFaddeevLeverrier {
    fn numerator_for(&self, output_index: usize, b: &Matrix<f64>, input_index: usize) -> Vec<f64> {
        self.adjugate_terms
            .iter()
            .map(|term| {
                let n = term.cols();
                (0..n)
                    .map(|column| term[(output_index, column)] * b[(column, input_index)])
                    .sum()
            })
            .collect()
    }
}

/// Independent, from-scratch numeric Faddeev-LeVerrier recursion (plain
/// `f64`, no shared code with `general_mna::faddeev_leverrier`) used only to
/// cross-check the symbolic recursion in tests.
fn numeric_faddeev_leverrier(a: &Matrix<f64>) -> NumericFaddeevLeverrier {
    let n = a.rows();
    let identity = |size: usize| -> Matrix<f64> {
        let mut result = Matrix::filled(size, size, 0.0);
        for index in 0..size {
            result[(index, index)] = 1.0;
        }
        result
    };
    let multiply = |lhs: &Matrix<f64>, rhs: &Matrix<f64>| -> Matrix<f64> {
        let mut result = Matrix::filled(lhs.rows(), rhs.cols(), 0.0);
        for row in 0..lhs.rows() {
            for col in 0..rhs.cols() {
                result[(row, col)] = (0..lhs.cols())
                    .map(|inner| lhs[(row, inner)] * rhs[(inner, col)])
                    .sum();
            }
        }
        result
    };
    let add = |lhs: &Matrix<f64>, rhs: &Matrix<f64>| -> Matrix<f64> {
        Matrix::from_vec(
            lhs.rows(),
            lhs.cols(),
            lhs.iter().zip(rhs.iter()).map(|(l, r)| l + r).collect(),
        )
        .unwrap()
    };
    let scale = |m: &Matrix<f64>, s: f64| -> Matrix<f64> { m.map(|v| v * s) };
    let trace = |m: &Matrix<f64>| -> f64 { (0..m.rows()).map(|i| m[(i, i)]).sum() };

    let mut m = Matrix::filled(n, n, 0.0);
    let mut c = 1.0;
    let mut denominator = vec![c];
    let mut adjugate_terms = Vec::with_capacity(n);
    for k in 1..=n {
        m = add(&multiply(a, &m), &scale(&identity(n), c));
        c = -trace(&multiply(a, &m)) / k as f64;
        denominator.push(c);
        adjugate_terms.push(m.clone());
    }

    NumericFaddeevLeverrier {
        denominator,
        adjugate_terms,
    }
}
