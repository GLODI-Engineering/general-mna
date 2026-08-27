//! `general_mna::build_system` — the unified builder that turns one source text into both the
//! electrical `MnaSystem` and the signal-domain block graph in a single pass, via
//! `general-spice-core`'s new first-class `kind=...` grammar (no `*`-disguise needed). This is
//! Phase 2 of the format-unification plan: `general-simulator` no longer parses the block DSL
//! itself, it only evaluates what this builder hands it.

use general_mna::block_graph::{BlockKind, ConstValue, GateBinding, Signal};
use general_mna::{build_system, System};
use general_spice_core::dialect::Dialect;

#[test]
fn a_plain_electrical_only_deck_still_builds_an_mna_system() {
    let source = "V1 1 0 5\nR1 1 0 1000\n";
    let System {
        mna,
        diodes,
        mosfets,
        gates,
        blocks,
        shared_r_on,
    } = build_system(source, Dialect::Ngspice).unwrap();
    assert!(diodes.is_empty());
    assert!(mosfets.is_empty());
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
fn a_mosfet_line_correlates_by_name_with_its_block_gate() {
    // The kind=mosfet line's own device NAME is what ties it to the real SPICE D-element line
    // sharing that name -- a real structural correlation-by-name, not a coincidence.
    let source = "V1 vin 0 400\nD1 vin vx mosfetmodel\nR1 vx 0 1000\n\
                  OFFVAL kind=const value=0\n\
                  OFFGATE kind=sig2gate in=OFFVAL\n\
                  D1 kind=mosfet r_on=0.01 g_breakdown=0 v_breakdown=-1e6 g_off=1e-6 v_th=1e6 \
                  g_on=0 gate=block ctrl=OFFGATE\n";
    let System {
        mosfets,
        gates,
        shared_r_on,
        blocks,
        ..
    } = build_system(source, Dialect::Ngspice).unwrap();
    assert!(mosfets.contains_key("D1"));
    assert_eq!(shared_r_on, 0.01);
    assert_eq!(
        gates.get("D1"),
        Some(&GateBinding::Block("OFFGATE".to_string()))
    );
    assert_eq!(blocks.len(), 2); // OFFVAL, OFFGATE (D1 itself became a mosfet+gate, not a block)
}

#[test]
fn mismatched_mosfet_r_on_is_a_clear_error() {
    let source = "D1 a b m1\nD2 c d m2\n\
                  D1 kind=mosfet r_on=0.01 g_breakdown=0 v_breakdown=-100 g_off=0 v_th=1 g_on=1 \
                  gate=block ctrl=G\n\
                  D2 kind=mosfet r_on=0.05 g_breakdown=0 v_breakdown=-100 g_off=0 v_th=1 g_on=1 \
                  gate=block ctrl=G\n";
    let err = build_system(source, Dialect::Ngspice).unwrap_err();
    assert!(err.contains("same r_on"), "unexpected error: {err}");
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
