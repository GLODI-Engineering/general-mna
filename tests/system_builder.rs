//! `general_mna::build_system` — the unified builder that turns one source text into both the
//! electrical `MnaSystem` and the signal-domain block graph in a single pass, via
//! `general-spice-core`'s new first-class `kind=...` grammar (no `*`-disguise needed). This is
//! Phase 2 of the format-unification plan: `general-simulator` no longer parses the block DSL
//! itself, it only evaluates what this builder hands it.

use general_mna::block_graph::{
    BlockKind, ConstValue, GateBinding, PidClamp, SampleTimeSpec, Signal,
};
use general_mna::{build_system, System};
use general_spice_core::dialect::Dialect;

#[test]
fn a_plain_electrical_only_deck_still_builds_an_mna_system() {
    let source = "V1 1 0 5\nR1 1 0 1000\n";
    let System {
        mna,
        ideal_diodes,
        ideal_switches,
        gates,
        blocks,
        shared_r_on,
    } = build_system(source, Dialect::Ngspice).unwrap();
    assert!(ideal_diodes.is_empty());
    assert!(ideal_switches.is_empty());
    assert!(gates.is_empty());
    assert!(blocks.is_empty());
    assert_eq!(shared_r_on, 0.0);
    // A real MnaSystem was actually built -- not just an empty placeholder.
    assert!(!mna.unknowns.is_empty());
}

#[test]
fn a_block_line_parses_without_any_star_disguise() {
    // The whole point of Phase 1/2: no more `* NAME kind=...` needed.
    let source = "V1 1 0 5\nR1 1 0 1000\nDUTY kind=const value=0.5\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].name, "DUTY");
    assert_eq!(blocks[0].kind, BlockKind::Const(ConstValue::Scalar(0.5)));
}

#[test]
fn a_python_list_field_builds_the_real_matrix() {
    let source =
        "V1 1 0 5\nR1 1 0 1000\nFILTER kind=statespace a=[[-1000]] b=[1000] c=[1] d=0 in=DUTY\n\
         DUTY kind=const value=1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let filter = blocks.iter().find(|b| b.name == "FILTER").unwrap();
    match &filter.kind {
        BlockKind::StateSpace(ss) => {
            assert_eq!(ss.a, vec![vec![-1000.0]]);
            assert_eq!(ss.b, vec![vec![1000.0]]);
            assert_eq!(ss.c, vec![vec![1.0]]);
        }
        other => panic!("expected StateSpace, got {other:?}"),
    }
    assert_eq!(filter.inputs, vec![Signal::Block("DUTY".to_string())]);
}

#[test]
fn an_ideal_switch_line_correlates_by_name_with_its_block_gate() {
    // The kind=ideal_switch line's own device NAME is what ties it to the real SPICE D-element
    // line sharing that name -- a real structural correlation-by-name, not a coincidence.
    let source = "V1 vin 0 400\nD1 vin vx idealswitchmodel\nR1 vx 0 1000\n\
                  OFFVAL kind=const value=0\n\
                  OFFGATE kind=sig2phys domain=voltage in=OFFVAL\n\
                  D1 kind=ideal_switch r_on=0.01 g_breakdown=0 v_breakdown=-1e6 g_off=1e-6 \
                  v_th=1e6 g_on=0 gate=block ctrl=OFFGATE\n";
    let System {
        ideal_switches,
        gates,
        shared_r_on,
        blocks,
        ..
    } = build_system(source, Dialect::Ngspice).unwrap();
    assert!(ideal_switches.contains_key("D1"));
    assert_eq!(shared_r_on, 0.01);
    assert_eq!(
        gates.get("D1"),
        Some(&GateBinding::Block("OFFGATE".to_string()))
    );
    assert_eq!(blocks.len(), 2); // OFFVAL, OFFGATE (D1 itself became an ideal switch+gate, not a block)
}

#[test]
fn mismatched_ideal_switch_r_on_is_a_clear_error() {
    let source = "D1 a b m1\nD2 c d m2\n\
                  D1 kind=ideal_switch r_on=0.01 g_breakdown=0 v_breakdown=-100 g_off=0 v_th=1 \
                  g_on=1 gate=block ctrl=G\n\
                  D2 kind=ideal_switch r_on=0.05 g_breakdown=0 v_breakdown=-100 g_off=0 v_th=1 \
                  g_on=1 gate=block ctrl=G\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(err.contains("same r_on"), "unexpected error: {err}");
}

#[test]
fn a_phys2sig_branch_naming_a_v_source_builds_fine() {
    // V is a branch device (has an MNA branch-current unknown) -- the ordinary, supported case.
    let source = "V1 a 0 5\nR1 a 0 1000\nIMEAS kind=phys2sig branch=V1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    assert_eq!(blocks.len(), 1);
}

#[test]
fn a_phys2sig_branch_naming_a_non_branch_device_is_a_clear_build_time_error() {
    // R1 is a resistor -- not V/L/E/H -- so it has no branch-current unknown at all. Must be a
    // hard build-time error (not a silent 0.0 read-back), and must suggest the ammeter fix.
    let source = "V1 a 0 5\nR1 a 0 1000\nIMEAS kind=phys2sig branch=R1\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(
        err.contains("has no current unknown available"),
        "unexpected error: {err}"
    );
    assert!(
        err.contains("insert a 0V voltage source in series"),
        "unexpected error: {err}"
    );
}

#[test]
fn a_phys2sig_branch_naming_a_nonexistent_element_is_a_distinct_clear_error() {
    let source = "V1 a 0 5\nR1 a 0 1000\nIMEAS kind=phys2sig branch=NOPE\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(
        err.contains("names no such element"),
        "unexpected error: {err}"
    );
}

#[test]
fn old_star_disguised_lines_are_ignored_as_ordinary_comments() {
    // Backward compatibility: an un-migrated deck using the old convention still parses as a
    // pure electrical circuit -- the disguised block lines are just comments, invisible to this
    // builder exactly like they're invisible to any other SPICE tool.
    let source = "V1 1 0 5\nR1 1 0 1000\n* DUTY kind=const value=0.5\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    assert!(blocks.is_empty());
}

#[test]
fn a_block_declared_inside_a_subckt_body_still_parses_as_a_block_instance() {
    // Confirms a block statement nested inside a .subckt body is recognized and, per Phase 3's
    // hierarchy::flatten, only reaches the built System once the subckt is actually instantiated
    // (an X-call) -- an uninstantiated subckt definition contributes nothing on its own, exactly
    // like an uninstantiated real SPICE subcircuit contributes no electrical elements either.
    let source = ".subckt reg vin vout\nMOD kind=const value=0.5\n.ends reg\nX1 a b reg\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].name, "X1.MOD");
}

#[test]
fn logic_gates_parse_with_the_right_arity_and_op() {
    use continuous_blocks::LogicOp;

    let source = "AND1 kind=and inputs=A,B,C\n\
                  NOT1 kind=not in=A\n\
                  A kind=const value=1\n\
                  B kind=const value=0\n\
                  C kind=const value=1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let and1 = blocks.iter().find(|b| b.name == "AND1").unwrap();
    assert_eq!(and1.kind, BlockKind::LogicGate(LogicOp::And));
    assert_eq!(and1.inputs.len(), 3);

    let not1 = blocks.iter().find(|b| b.name == "NOT1").unwrap();
    assert_eq!(not1.kind, BlockKind::LogicGate(LogicOp::Not));
    assert_eq!(not1.inputs.len(), 1);
}

#[test]
fn a_single_input_and_gate_is_a_clear_error_not_a_silent_pass_through() {
    let source = "AND1 kind=and inputs=A\nA kind=const value=1\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(err.contains("at least 2"), "error: {err}");
}

#[test]
fn srlatch_parses_set_reset_inputs_and_defaults_to_set_priority() {
    use continuous_blocks::LatchPriority;

    let source = "FAULT kind=srlatch set=TRIP reset=CLR\n\
                  TRIP kind=const value=0\n\
                  CLR kind=const value=0\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let fault = blocks.iter().find(|b| b.name == "FAULT").unwrap();
    assert_eq!(
        fault.kind,
        BlockKind::SrLatch {
            priority: LatchPriority::Set
        }
    );
    assert_eq!(fault.inputs.len(), 2);
}

#[test]
fn dff_parses_clk_and_d_in_that_order() {
    use continuous_blocks::FlipFlopKind;

    let source = "Q kind=dff clk=CLK d=D\n\
                  CLK kind=const value=0\n\
                  D kind=const value=1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let q = blocks.iter().find(|b| b.name == "Q").unwrap();
    assert_eq!(
        q.kind,
        BlockKind::FlipFlop {
            kind: FlipFlopKind::D,
            reset: false
        }
    );
    assert_eq!(
        q.inputs,
        vec![
            Signal::Block("CLK".to_string()),
            Signal::Block("D".to_string())
        ]
    );
}

#[test]
fn counter_with_no_optional_fields_has_only_a_clk_input() {
    let source = "CNT kind=counter clk=CLK\nCLK kind=const value=0\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let cnt = blocks.iter().find(|b| b.name == "CNT").unwrap();
    assert_eq!(
        cnt.kind,
        BlockKind::Counter {
            up_down: false,
            modulus: None,
            reset: false
        }
    );
    assert_eq!(cnt.inputs.len(), 1);
}

#[test]
fn counter_with_modulus_and_reset_declares_both_extra_inputs() {
    let source = "CNT kind=counter clk=CLK modulus=10 reset=R\n\
                  CLK kind=const value=0\n\
                  R kind=const value=0\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let cnt = blocks.iter().find(|b| b.name == "CNT").unwrap();
    assert_eq!(
        cnt.kind,
        BlockKind::Counter {
            up_down: false,
            modulus: Some(10),
            reset: true
        }
    );
    assert_eq!(cnt.inputs.len(), 2); // clk, reset (no up_down)
}

#[test]
fn discretestatespace_parses_matrices_and_a_mandatory_periodic_sample_time() {
    let source = "FILTER kind=discretestatespace a=[[0.5]] b=[2] c=[1] d=0 in=DUTY ts=0.1\n\
         DUTY kind=const value=1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let filter = blocks.iter().find(|b| b.name == "FILTER").unwrap();
    match &filter.kind {
        BlockKind::DiscreteStateSpace { ss, sample_time } => {
            assert_eq!(ss.a, vec![vec![0.5]]);
            assert_eq!(ss.b, vec![vec![2.0]]);
            assert_eq!(
                *sample_time,
                SampleTimeSpec::Periodic {
                    period: 0.1,
                    offset: 0.0
                }
            );
        }
        other => panic!("expected DiscreteStateSpace, got {other:?}"),
    }
}

#[test]
fn discretestatespace_without_ts_or_freq_is_a_clear_error_not_silently_continuous() {
    let source = "FILTER kind=discretestatespace a=[[0.5]] b=[2] c=[1] d=0 in=DUTY\n\
                  DUTY kind=const value=1\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(err.contains("missing 'ts=' or 'freq='"), "got: {err}");
}

#[test]
fn discretestatespace_rejects_ts_variable() {
    let source = "FILTER kind=discretestatespace a=[[0.5]] b=[2] c=[1] d=0 in=DUTY ts=variable\n\
                  DUTY kind=const value=1\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(err.contains("ts=variable is not available"), "got: {err}");
}

#[test]
fn discretetf_parses_num_den_and_sample_time() {
    let source = "FILTER kind=discretetf num=[1] den=[1,-0.5] in=DUTY freq=10\n\
                  DUTY kind=const value=1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let filter = blocks.iter().find(|b| b.name == "FILTER").unwrap();
    match &filter.kind {
        BlockKind::DiscreteTransferFunction { tf, sample_time } => {
            assert_eq!(tf.num, vec![1.0]);
            assert_eq!(tf.den, vec![1.0, -0.5]);
            assert_eq!(
                *sample_time,
                SampleTimeSpec::Periodic {
                    period: 0.1,
                    offset: 0.0
                }
            );
        }
        other => panic!("expected DiscreteTransferFunction, got {other:?}"),
    }
}

#[test]
fn discretepid_defaults_to_forward_euler_and_parses_fixed_clamp() {
    let source = "CTRL kind=discretepid kp=1 ki=2 kd=0 n=1 in=ERR ts=0.01 clamp_lo=-1 clamp_hi=1\n\
         ERR kind=const value=0\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let ctrl = blocks.iter().find(|b| b.name == "CTRL").unwrap();
    match &ctrl.kind {
        BlockKind::DiscretePid {
            pid,
            clamp,
            sample_time,
        } => {
            assert_eq!(pid.kp, 1.0);
            assert_eq!(pid.period, 0.01);
            assert_eq!(
                pid.method,
                continuous_blocks::DiscreteIntegrationMethod::ForwardEuler
            );
            assert_eq!(*clamp, PidClamp::Fixed(-1.0, 1.0));
            assert_eq!(
                *sample_time,
                SampleTimeSpec::Periodic {
                    period: 0.01,
                    offset: 0.0
                }
            );
        }
        other => panic!("expected DiscretePid, got {other:?}"),
    }
}

#[test]
fn discretepid_honors_an_explicit_integration_method() {
    let source = "CTRL kind=discretepid kp=0 ki=1 kd=0 n=1 in=ERR ts=0.01 clamp_lo=-1 clamp_hi=1 \
         integration_method=trapezoidal\n\
         ERR kind=const value=0\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let ctrl = blocks.iter().find(|b| b.name == "CTRL").unwrap();
    match &ctrl.kind {
        BlockKind::DiscretePid { pid, .. } => {
            assert_eq!(
                pid.method,
                continuous_blocks::DiscreteIntegrationMethod::Trapezoidal
            );
        }
        other => panic!("expected DiscretePid, got {other:?}"),
    }
}

#[test]
fn discretepid_unknown_integration_method_is_a_clear_error() {
    let source = "CTRL kind=discretepid kp=0 ki=1 kd=0 n=1 in=ERR ts=0.01 clamp_lo=-1 clamp_hi=1 \
         integration_method=nonsense\n\
         ERR kind=const value=0\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(err.contains("unknown integration_method"), "got: {err}");
}

// ---- ic= / y0= on stateful blocks (#8) ----

fn only_block_ic(source: &str) -> Option<Vec<f64>> {
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    blocks[0].ic.clone()
}

fn build_error(source: &str) -> String {
    build_system(source, Dialect::Ngspice).unwrap_err()
}

#[test]
fn a_block_without_ic_starts_from_rest() {
    assert_eq!(
        only_block_ic("G kind=tf in=U num=[1] den=[1,1]\nU kind=const value=1\n"),
        None
    );
}

#[test]
fn statespace_ic_is_the_state_vector_and_a_scalar_is_allowed_for_one_state() {
    assert_eq!(
        only_block_ic(
            "G kind=statespace in=U a=[[0,1],[-2,-3]] b=[0,1] c=[1,0] ic=[0.5,-1]\n\
             U kind=const value=1\n"
        ),
        Some(vec![0.5, -1.0])
    );
    assert_eq!(
        only_block_ic(
            "G kind=discretestatespace in=U a=[[0.5]] b=[1] c=[1] ts=0.1 ic=2\n\
             U kind=const value=1\n"
        ),
        Some(vec![2.0])
    );
}

#[test]
fn a_wrong_length_ic_is_rejected_with_the_state_count() {
    let err = build_error(
        "G kind=statespace in=U a=[[0,1],[-2,-3]] b=[0,1] c=[1,0] ic=[0.5]\n\
         U kind=const value=1\n",
    );
    assert_eq!(
        err,
        "line 1: device 'G' field 'ic' has 1 value(s), but this block has 2 state(s)"
    );
}

/// `(s+3)/(s^2+3s+2)` settled at `y0=6`: `x1 = 6/b0 = 6/3 = 2` -- the same hand-derived case
/// `continuous-blocks` tests as a real equilibrium.
#[test]
fn tf_y0_resolves_to_the_settled_canonical_state() {
    assert_eq!(
        only_block_ic("G kind=tf in=U num=[1,3] den=[1,3,2] y0=6\nU kind=const value=4\n"),
        Some(vec![2.0, 0.0])
    );
    assert_eq!(
        only_block_ic("G kind=tf in=U num=[1,3] den=[1,3,2] ic=[1,-1]\nU kind=const value=4\n"),
        Some(vec![1.0, -1.0])
    );
    // H(z) = (0.5z+0.25)/(z^2-0.5z), N(1) = 0.75: every state is 3/0.75 = 4.
    assert_eq!(
        only_block_ic(
            "G kind=discretetf in=U num=[0.5,0.25] den=[1,-0.5,0] ts=0.1 y0=3\n\
             U kind=const value=2\n"
        ),
        Some(vec![4.0, 4.0])
    );
}

#[test]
fn tf_ic_and_y0_together_or_y0_on_a_highpass_are_rejected() {
    let err = build_error("G kind=tf in=U num=[1] den=[1,1] ic=[1] y0=1\nU kind=const value=1\n");
    assert!(err.contains("give one or the other"), "{err}");
    let err = build_error("G kind=tf in=U num=[1,0] den=[1,1] y0=1\nU kind=const value=1\n");
    assert!(err.contains("numerator is zero at DC"), "{err}");
    let err =
        build_error("G kind=statespace in=U a=[[-1]] b=[1] c=[1] y0=1\nU kind=const value=1\n");
    assert!(
        err.contains("only valid on kind=tf/kind=discretetf"),
        "{err}"
    );
}

/// `kp=2 ki=3 kd=0 n=10`: `(2s^2+23s+30)/(s^2+10s)`, `d=2`, remainder `3s+30`, so `b0=30`,
/// `a0=0`, and an output of `ic=0.6` at zero error is `x1 = 0.6/30 = 0.02`.
#[test]
fn pid_ic_is_the_held_output_at_zero_error() {
    let ic = only_block_ic(
        "C kind=pid in=E kp=2 ki=3 kd=0 n=10 clamp_lo=-1 clamp_hi=1 ic=0.6\n\
         E kind=const value=0\n",
    )
    .unwrap();
    assert_eq!(ic.len(), 2);
    assert!((ic[0] - 0.02).abs() < 1e-15 && ic[1] == 0.0, "{ic:?}");

    // Discrete: the integrator's accumulated value, ic / ki.
    assert_eq!(
        only_block_ic(
            "C kind=discretepid in=E kp=1 ki=2 kd=0 n=1 ts=0.01 clamp_lo=-1 clamp_hi=1 ic=0.6\n\
             E kind=const value=0\n"
        ),
        Some(vec![0.3])
    );

    let err = build_error(
        "C kind=pid in=E kp=2 ki=0 kd=0 n=10 clamp_lo=-1 clamp_hi=1 ic=0.6\nE kind=const value=0\n",
    );
    assert!(err.contains("ki=0"), "{err}");
}

#[test]
fn a_phase_ic_must_lie_in_the_unit_interval() {
    assert_eq!(
        only_block_ic("O kind=vco in=F f_min=1 f_max=10 ic=0.25\nF kind=const value=5\n"),
        Some(vec![0.25])
    );
    let err = build_error("O kind=vco in=F f_min=1 f_max=10 ic=1\nF kind=const value=5\n");
    assert!(err.contains("0 <= ic < 1 (got 1)"), "{err}");
}

#[test]
fn logic_and_counter_ic_are_checked_against_their_own_domains() {
    assert_eq!(
        only_block_ic("Q kind=srlatch set=S reset=S ic=1\nS kind=const value=0\n"),
        Some(vec![1.0])
    );
    assert_eq!(
        only_block_ic("Q kind=dff clk=S d=S ic=1\nS kind=const value=0\n"),
        Some(vec![1.0])
    );
    assert_eq!(
        only_block_ic("H kind=hysteresis in=S high=1 low=0 ic=1\nS kind=const value=0\n"),
        Some(vec![1.0])
    );
    let err = build_error("Q kind=srlatch set=S reset=S ic=0.5\nS kind=const value=0\n");
    assert!(err.contains("must be 0 or 1 (got '0.5')"), "{err}");

    assert_eq!(
        only_block_ic("N kind=counter clk=S modulus=10 ic=7\nS kind=const value=0\n"),
        Some(vec![7.0])
    );
    assert_eq!(
        only_block_ic("N kind=counter clk=S ic=-3\nS kind=const value=0\n"),
        Some(vec![-3.0])
    );
    let err = build_error("N kind=counter clk=S modulus=10 ic=10\nS kind=const value=0\n");
    assert!(err.contains("0 <= ic < modulus (10) (got 10)"), "{err}");
    let err = build_error("N kind=counter clk=S ic=1.5\nS kind=const value=0\n");
    assert!(err.contains("is not an integer"), "{err}");
}

#[test]
fn pmsm_ic_is_the_four_machine_states() {
    assert_eq!(
        only_block_ic(
            "M kind=pmsm r_s=0.5 l_d=1e-3 l_q=1e-3 lambda_pm=0.05 pole_pairs=4 inertia=1e-5 \
             friction=0 inputs=Z,Z,Z ic=[1,2,300,0.5]\nZ kind=const value=0\n"
        ),
        Some(vec![1.0, 2.0, 300.0, 0.5])
    );
}

#[test]
fn ic_on_a_stateless_block_is_rejected_not_dropped() {
    let err = build_error("G kind=gain in=U k=2 ic=1\nU kind=const value=1\n");
    assert!(
        err.starts_with("line 1: device 'G' field 'ic': this kind has no state"),
        "{err}"
    );
    let err = build_error("G kind=tf in=U num=[1] den=[1,1] ic=[inf]\nU kind=const value=1\n");
    assert!(err.contains("finite"), "{err}");
}

/// A block declared inside a `.subckt` keeps its `ic=` through flattening.
#[test]
fn a_subcircuit_block_keeps_its_ic() {
    let source = ".subckt filt in out\nR1 in out 1\nF kind=tf in=U num=[1] den=[1,1] y0=2\n\
                  U kind=const value=1\n.ends filt\nV1 a 0 1\nX1 a b filt\nR2 b 0 1\n";
    let System { blocks, .. } = build_system(source, Dialect::Ngspice).unwrap();
    let f = blocks
        .iter()
        .find(|b| b.name.ends_with('F') || b.name.contains(".F"))
        .unwrap();
    assert_eq!(f.ic, Some(vec![2.0]));
}
