use std::collections::BTreeMap;

use general_mna::{
    average, BuildError, BuildOptions, Expression, MnaBuilder, StateSpaceError, SwitchState,
    WeightedPhase,
};
use general_spice_core::Dialect;

fn numeric(
    system: &general_mna::MnaSystem,
    values: &[(&str, f64)],
) -> general_mna::NumericMnaSystem {
    let values = values
        .iter()
        .map(|(name, value)| ((*name).to_string(), *value))
        .collect::<BTreeMap<_, _>>();
    system.evaluate(&values).unwrap()
}

fn index(system: &general_mna::MnaSystem, name: &str) -> usize {
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
fn capacitor_only_loop_reports_precise_singular_storage_error() {
    // A floating triangle of capacitors with no resistive path to ground:
    // the three node-voltage KCL rows sum to zero, so the reduced storage
    // block is rank-deficient (one voltage is not independent of the rest).
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("C1 1 2 C\nC2 2 3 C\nC3 3 1 C")
        .unwrap();
    let evaluated = numeric(&system, &[("C", 1e-6)]);
    let err = evaluated.to_state_space(1e-12).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("capacitor-only loop"), "{message}");
    match err {
        StateSpaceError::SingularStorageBlock { unknown, block } => {
            assert!(unknown.starts_with("V("), "{unknown}");
            assert_eq!(block.len(), 3);
        }
        other => panic!("expected SingularStorageBlock, got {other:?}"),
    }
}

#[test]
fn perfectly_coupled_inductors_report_precise_singular_storage_error() {
    // Two independently driven inductors coupled at k=1 (an ideal, lossless
    // transformer with no leakage inductance) make the 2x2 inductance
    // sub-matrix exactly singular: L1*L2*(1 - k^2) = 0.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment(
            "V1 1 0 DC 1\nR1 1 2 1\nL1 2 0 1m\nV2 3 0 DC 1\nR2 3 4 1\nL2 4 0 1m\nK1 L1 L2 1",
        )
        .unwrap();
    let evaluated = numeric(&system, &[]);
    let err = evaluated.to_state_space(1e-12).unwrap_err();

    let message = err.to_string();
    assert!(message.contains("inductor-only cut-set"), "{message}");
    match err {
        StateSpaceError::SingularStorageBlock { unknown, block } => {
            assert!(unknown.starts_with("I("), "{unknown}");
            assert_eq!(block.len(), 2);
        }
        other => panic!("expected SingularStorageBlock, got {other:?}"),
    }
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
    // 'D' (diode) is intentionally supported now, as a companion-model conductance plus a
    // Norton current source with per-instance symbolic parameters (see
    // `diode_is_stamped_as_symbolic_conductance_plus_norton_current_source` below) — the split
    // exists exactly so an external caller (e.g. a piecewise-linear circuit solver) can decide
    // per-timestep numeric values without this crate knowing anything about diode physics. A
    // device with no linear-or-externally-parameterized stamp at all, like a BJT, must still be
    // reported rather than silently dropped.
    let error = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("Q1 1 0 2 bjt_model")
        .unwrap_err();
    assert!(matches!(error, BuildError::UnsupportedElement(_)));
}

#[test]
fn diode_is_stamped_as_symbolic_conductance_plus_norton_current_source() {
    // D1 anode=1, cathode=0 (ground); companion model I = G*V + Ioff, with G and Ioff left as
    // per-instance symbols (`D1_G`, `D1_Ioff`) for the caller to resolve numerically however it
    // decides which segment of a piecewise-linear device curve is currently active.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("D1 1 0 diode_model")
        .unwrap();

    assert_eq!(system.unknowns, ["V(1)"]);
    assert_eq!(system.inputs, ["D1"]);
    assert_eq!(system.a[(0, 0)].to_string(), "D1_G");
    assert_eq!(system.input_values[0].to_string(), "D1_Ioff");

    // Node 1 is the only row; B's entry there must be -1 (matching a real `I` source's own
    // sign convention, since the diode's Norton current source is stamped by literally the
    // same `stamp_current_source` function) so that u[0] = B[0][0] * Ioff = -Ioff.
    assert_eq!(system.b[(0, 0)].to_string(), "-1");
    assert_eq!(system.u[0].to_string(), "-1 * D1_Ioff");

    let evaluated = system
        .evaluate(&BTreeMap::from([
            ("D1_G".to_string(), 2.0),
            ("D1_Ioff".to_string(), 3.0),
        ]))
        .unwrap();
    assert!((evaluated.a[(0, 0)] - 2.0).abs() < 1e-12);
    assert!((evaluated.u[0] - (-3.0)).abs() < 1e-12);
}

#[test]
fn parameter_defaults_and_numeric_lookup_are_case_insensitive() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment(".param RLOAD=2k\nV1 1 0 1\nR1 1 0 rload")
        .unwrap();
    let evaluated = numeric(&system, &[]);
    assert!((evaluated.a[(0, 0)] - 0.0005).abs() < 1e-15);
}

#[test]
fn sin_source_is_stamped_as_a_symbol_not_a_baked_literal() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 1 0 SIN(0 10 1000 0 0 0)\nR1 1 0 1k")
        .unwrap();

    assert_eq!(system.inputs, ["V1"]);
    // A transient-function source is a symbol referencing its own element name -- not a
    // baked-in literal -- exactly like a PWL diode's `{name}_Ioff` above, so the caller can
    // supply a different numeric value every step.
    assert_eq!(system.input_values[0].to_string(), "V1");
    assert_eq!(system.transient_sources.len(), 1);
    let sin = &system.transient_sources["V1"];
    // Quarter period of a 1000Hz sine starting at 0: t=250us -> value = 10.
    assert!((sin.value_at(2.5e-4) - 10.0).abs() < 1e-6);

    // V1's own KVL row is "I(V1)" (a voltage source's B column is stamped on its own branch
    // row, not directly on a node row -- see `stamp_voltage_source`), not row 0.
    let branch_row = index(&system, "I(V1)");

    // Without the caller supplying "V1" in `values`, u is 0 there (documented NAN-as-zero
    // fallback for a symbol the caller didn't provide).
    let evaluated_missing = system.evaluate(&BTreeMap::new()).unwrap();
    assert_eq!(evaluated_missing.u[branch_row], 0.0);

    // With the caller supplying the transient function's own value at some t, u reflects it
    // exactly -- this is the whole point: the same MnaSystem, evaluated with different
    // caller-supplied "V1" values, reproduces different circuit states as if V1 varies with
    // time, without elspice-mna itself knowing anything about "time."
    let evaluated_at_quarter_period = system
        .evaluate(&BTreeMap::from([("V1".to_string(), sin.value_at(2.5e-4))]))
        .unwrap();
    assert!((evaluated_at_quarter_period.u[branch_row] - 10.0).abs() < 1e-9);
}

#[test]
fn plain_dc_source_is_unaffected_by_transient_source_support() {
    // A source with no transient function must keep the exact existing behavior: a baked
    // literal, and an empty transient_sources map -- confirms adding SIN/PULSE/EXP/PWL/SFFM
    // support didn't change anything for the (overwhelmingly common) static-value case.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 1 0 DC 10\nR1 1 0 1k")
        .unwrap();
    assert_eq!(system.input_values[0].to_string(), "10");
    assert!(system.transient_sources.is_empty());
}

// ---------------------------------------------------------------------------------------------
// Unknown trailing fields on a device card (general-mna#2)
// ---------------------------------------------------------------------------------------------

fn build_error(source: &str) -> BuildError {
    MnaBuilder::new(Dialect::Ngspice)
        .build_fragment(source)
        .unwrap_err()
}

#[test]
fn unknown_device_field_is_rejected_with_a_line_numbered_message() {
    // The bug this test exists for: `wibble=5` used to be dropped without a word, so the deck
    // ran and produced a plausible waveform for a circuit nobody wrote.
    let error = build_error("V1 a 0 10\nR1 a b 1000\nC1 b 0 1e-6 wibble=5\n");
    assert_eq!(
        error.to_string(),
        "line 3: device 'C1' unknown field 'wibble' (device C accepts only: ic)"
    );
}

#[test]
fn unknown_device_field_names_a_letter_with_no_fields_at_all() {
    let error = build_error("V1 a 0 10\nR1 a 0 1000 tc1=0.001\n");
    assert_eq!(
        error.to_string(),
        "line 2: device 'R1' unknown field 'tc1' (device R accepts no key=value fields)"
    );
}

#[test]
fn extra_positional_parameter_on_a_two_terminal_value_device_is_rejected() {
    let error = build_error("V1 a 0 10\nR1 a 0 1000 2000\n");
    assert_eq!(
        error.to_string(),
        "line 2: device 'R1' unexpected extra parameter '2000' (device R takes exactly one value)"
    );
}

#[test]
fn device_field_check_is_case_insensitive_and_accepts_the_known_key() {
    // `IC=` and `ic=` are the same field, and neither is an error on a capacitor.
    for source in [
        "C1 b 0 1e-6 IC=5\nR1 a b 1\nV1 a 0 1",
        "C1 b 0 1e-6 ic=5\nR1 a b 1\nV1 a 0 1",
    ] {
        MnaBuilder::new(Dialect::Ngspice)
            .build_fragment(source)
            .unwrap();
    }
}

#[test]
fn unstamped_devices_keep_their_own_parameter_syntax() {
    // An `M` card's `L=1u W=10u` is valid netlist text this crate simply has no model for.
    // Field-checking it would turn `IgnoreWithWarning` into a hard error, which is not this
    // change's business -- only devices with a stamp are held to a known parameter grammar.
    let options = BuildOptions {
        unsupported_elements: general_mna::UnsupportedElementPolicy::IgnoreWithWarning,
        ..BuildOptions::default()
    };
    let system = MnaBuilder::with_options(Dialect::Ngspice, options)
        .build_fragment("V1 d 0 5\nR1 d 0 1k\nM1 d g s b NMOS L=1u W=10u\n")
        .unwrap();
    assert_eq!(system.warnings.len(), 1);
}

#[test]
fn existing_source_syntax_still_parses() {
    // Positional `DC`/`AC`/transient-function tokens on V/I, a model name on D, and a
    // controlling-source name on F are all open positional grammars that must keep working.
    for source in [
        "V1 1 0 DC 10\nR1 1 0 1k",
        "V1 1 0 SIN(0 10 1000 0 0 0)\nR1 1 0 1k",
        "F1 2 0 Vsense 2\nR1 2 0 1k\nVsense 1 0 1",
        "L1 1 0 4\nL2 2 0 9\nK1 L1 L2 0.5",
    ] {
        MnaBuilder::new(Dialect::Ngspice)
            .build_fragment(source)
            .unwrap_or_else(|e| panic!("{source:?} should still parse, got {e}"));
    }
}

// ---------------------------------------------------------------------------------------------
// ic= initial conditions (general-mna#3)
// ---------------------------------------------------------------------------------------------

#[test]
fn no_ic_means_no_initial_state_at_all() {
    // Ok(None), not an all-zero vector: a caller keeps its own "start from rest" default.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a b 1000\nC1 b 0 1e-6")
        .unwrap();
    assert!(system.initial_conditions.is_empty());
    assert_eq!(
        system
            .initial_state(
                &BTreeMap::new(),
                general_mna::DEFAULT_INITIAL_STATE_TOLERANCE
            )
            .unwrap(),
        None
    );
}

#[test]
fn capacitor_ic_holds_its_voltage_and_the_rest_of_the_circuit_follows() {
    // Hand-derived: C1 is a 5 V source to ground at t = 0, so V(b) = 5 and V(a) = 10 (V1 is
    // ideal). R1 then carries (10 - 5)/1000 = 5 mA from a to b, which V1 must supply, and a
    // voltage source's branch unknown is the current leaving its first node, so I(V1) = -5 mA.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a b 1000\nC1 b 0 1e-6 ic=5")
        .unwrap();
    assert_eq!(system.initial_conditions.len(), 1);
    let x = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    assert_eq!(x.len(), system.order());
    assert!((x[index(&system, "V(b)")] - 5.0).abs() < 1e-12);
    assert!((x[index(&system, "V(a)")] - 10.0).abs() < 1e-12);
    assert!((x[index(&system, "I(V1)")] + 5e-3).abs() < 1e-12);
}

#[test]
fn capacitor_ic_is_first_node_minus_second_node() {
    // Same capacitor, written the other way round: ic is V(first) - V(second), so V(b) = -5.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a b 1000\nC1 0 b 1e-6 ic=5")
        .unwrap();
    let x = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    assert!((x[index(&system, "V(b)")] + 5.0).abs() < 1e-12);
}

#[test]
fn capacitor_ic_works_between_two_floating_nodes() {
    // The case a bare assignment cannot express: neither terminal is ground, so only the
    // *difference* is declared and the constrained operating point has to settle the rest.
    // R1 (a->b) and R2 (c->0) are 1 k each and carry the same current i; V(b) - V(c) = 5 is
    // held by C1, so 10 - 1000i - 5 - 1000i = 0 => i = 2.5 mA, V(b) = 7.5, V(c) = 2.5.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a b 1000\nC1 b c 1e-6 ic=5\nR2 c 0 1000")
        .unwrap();
    let x = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    assert!((x[index(&system, "V(b)")] - 7.5).abs() < 1e-12);
    assert!((x[index(&system, "V(c)")] - 2.5).abs() < 1e-12);
}

#[test]
fn inductor_ic_is_the_current_from_the_first_node_to_the_second() {
    // The sign convention people get wrong, pinned down. `L1 a b ... ic=12` puts 12 A into a
    // and out of b, which is exactly the sign of the system's own I(L1) unknown. The 12 A
    // must come from V1, so I(V1) = -12; and R1 (b->0) carries it, so V(b) = 12 * 10 = 120.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nL1 a b 5e-6 ic=12\nR1 b 0 10")
        .unwrap();
    let x = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    assert!((x[index(&system, "I(L1)")] - 12.0).abs() < 1e-12);
    assert!((x[index(&system, "I(V1)")] + 12.0).abs() < 1e-12);
    assert!((x[index(&system, "V(b)")] - 120.0).abs() < 1e-12);
    assert!((x[index(&system, "V(a)")] - 10.0).abs() < 1e-12);
}

#[test]
fn reversing_an_inductor_reverses_the_declared_current() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nL1 b a 5e-6 ic=12\nR1 b 0 10")
        .unwrap();
    let x = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    // I(L1) is still +12 (it is the b -> a current now), so the physical current through R1
    // reverses: 12 A now flow out of node b into the inductor, so V(b) = -120.
    assert!((x[index(&system, "I(L1)")] - 12.0).abs() < 1e-12);
    assert!((x[index(&system, "V(b)")] + 120.0).abs() < 1e-12);
}

#[test]
fn ic_free_storage_is_open_or_short_at_the_ic_operating_point() {
    // C2 has no ic, so it is an open circuit at t = 0 and draws nothing: V(c) = V(b) = 5,
    // unchanged from the single-capacitor case above.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a b 1000\nC1 b 0 1e-6 ic=5\nC2 b c 1e-6\n R2 c 0 1000")
        .unwrap();
    let x = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    assert!((x[index(&system, "V(b)")] - 5.0).abs() < 1e-12);
    assert!(x[index(&system, "V(c)")].abs() < 1e-12);
}

#[test]
fn an_ic_can_be_a_symbol_resolved_at_evaluation_time() {
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a b 1000\nC1 b 0 1e-6 ic=Vc0")
        .unwrap();
    let x = system
        .initial_state(
            &BTreeMap::from([("Vc0".to_string(), 3.0)]),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap()
        .unwrap();
    assert!((x[index(&system, "V(b)")] - 3.0).abs() < 1e-12);
}

#[test]
fn contradictory_ic_has_no_unique_operating_point() {
    // C1 sits directly across an ideal voltage source, which already fixes its voltage, so the
    // constrained system is inconsistent rather than merely over-determined.
    let system = MnaBuilder::new(Dialect::Ngspice)
        .build_fragment("V1 a 0 10\nR1 a 0 1000\nC1 a 0 1e-6 ic=5")
        .unwrap();
    let error = system
        .initial_state(
            &BTreeMap::new(),
            general_mna::DEFAULT_INITIAL_STATE_TOLERANCE,
        )
        .unwrap_err();
    assert!(
        error
            .to_string()
            .starts_with("no unique ic= operating point"),
        "{error}"
    );
}
