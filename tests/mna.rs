use std::collections::BTreeMap;

use elspice_mna::{
    average, BuildError, BuildOptions, Expression, MnaBuilder, SwitchState, WeightedPhase,
};
use spice_core::Dialect;

fn numeric(
    system: &elspice_mna::MnaSystem,
    values: &[(&str, f64)],
) -> elspice_mna::NumericMnaSystem {
    let values = values
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect::<BTreeMap<_, _>>();
    system.evaluate(&values).unwrap()
}

fn index(system: &elspice_mna::MnaSystem, name: &str) -> usize {
    system
        .unknowns
        .iter()
        .position(|item| item == name)
        .unwrap()
}

#[test]
fn voltage_divider_builds_expected_descriptor_system() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 1 0 DC 10\nR1 1 2 2k\nR2 2 0 3k")
        .unwrap();
    let evaluated = numeric(&system, &[]);

    assert_eq!(system.unknowns, ["V(1)", "V(2)", "I(V1)"]);
    assert_eq!(system.inputs, ["V1"]);
    assert!((evaluated.a[(0, 0)] - 1.0 / 2000.0).abs() < 1e-15);
    assert!((evaluated.a[(1, 1)] - (1.0 / 2000.0 + 1.0 / 3000.0)).abs() < 1e-15);
    assert_eq!(evaluated.a[(0, 2)], 1.0);
    assert_eq!(evaluated.a[(2, 0)], 1.0);
    assert_eq!(evaluated.b[(2, 0)], 1.0);
    assert_eq!(system.u[2], Expression::Constant(10.0));
}

#[test]
fn rc_descriptor_reduces_to_expected_state_space() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 1 0 1\nR1 1 2 R\nC1 2 0 C")
        .unwrap();
    let evaluated = numeric(&system, &[("R", 1000.0), ("C", 1e-6)]);
    let state_space = evaluated.to_state_space(1e-12).unwrap();

    assert_eq!(state_space.states, ["V(2)"]);
    assert!((state_space.a[(0, 0)] + 1000.0).abs() < 1e-9);
    assert!((state_space.b[(0, 0)] - 1000.0).abs() < 1e-9);
}

#[test]
fn controlled_source_can_precede_its_controlling_voltage_source() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("F1 2 0 Vsense 2\nR1 2 0 1k\nVsense 1 0 1")
        .unwrap();
    let evaluated = numeric(&system, &[]);
    let output = index(&system, "V(2)");
    let control = index(&system, "I(Vsense)");
    assert_eq!(evaluated.a[(output, control)], 2.0);
}

#[test]
fn mutual_inductance_uses_spice_coupling_coefficient_definition() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("L1 1 0 4\nL2 2 0 9\nK1 L1 L2 0.5")
        .unwrap();
    let evaluated = numeric(&system, &[]);
    let first = index(&system, "I(L1)");
    let second = index(&system, "I(L2)");
    assert!((evaluated.k[(first, second)] + 3.0).abs() < 1e-12);
    assert_eq!(evaluated.k[(first, second)], evaluated.k[(second, first)]);
}

#[test]
fn converter_switch_phases_average_with_duty_ratio() {
    let netlist = "V1 in 0 Vin\nS1 in sw ctrl 0 ideal\nL1 sw out L\nC1 out 0 C\nR1 out 0 R";
    let mut on_options = BuildOptions::default();
    on_options.set_switch("s1", SwitchState::On);
    let on = MnaBuilder::with_options(Dialect::Ngspice, on_options)
        .build_fragment(netlist)
        .unwrap();

    let mut off_options = BuildOptions::default();
    off_options.set_switch("S1", SwitchState::Off);
    let off = MnaBuilder::with_options(Dialect::Ngspice, off_options)
        .build_fragment(netlist)
        .unwrap();

    assert!(!on.unknowns.iter().any(|name| name == "V(ctrl)"));
    let duty = Expression::symbol("D");
    let complement = Expression::one() - duty.clone();
    let averaged = average(&[
        WeightedPhase {
            system: &on,
            weight: &duty,
        },
        WeightedPhase {
            system: &off,
            weight: &complement,
        },
    ])
    .unwrap();

    let input = index(&averaged, "V(in)");
    let switched = index(&averaged, "V(sw)");
    let expression = averaged.a[(input, switched)].to_string();
    assert!(expression.contains("D"));
    assert!(expression.contains("Ron"));
    assert!(expression.contains("Roff"));

    let evaluated = numeric(
        &averaged,
        &[
            ("D", 0.4),
            ("Ron", 1e-3),
            ("Roff", 1e9),
            ("L", 100e-6),
            ("C", 100e-6),
            ("R", 10.0),
        ],
    );
    let state_space = evaluated.to_state_space(1e-12).unwrap();
    assert_eq!(state_space.states.len(), 2);
    assert!(state_space.states.contains(&"I(L1)".to_string()));
    assert!(state_space.states.contains(&"V(out)".to_string()));
}

#[test]
fn nonlinear_devices_are_not_silently_discarded() {
    let error = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("D1 1 0 diode_model")
        .unwrap_err();
    assert!(matches!(error, BuildError::UnsupportedElement(_)));
}

#[test]
fn parameter_defaults_and_numeric_lookup_are_case_insensitive() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment(".param RLOAD=2k\nV1 1 0 1\nR1 1 0 rload")
        .unwrap();
    let evaluated = numeric(&system, &[]);
    assert!((evaluated.a[(0, 0)] - 0.0005).abs() < 1e-15);
}
