//! A `.subckt`'s formal parameters are bound where it is instantiated (#63).
//!
//! Both halves of the grammar parsed long before this worked. `.subckt rpart p n Rv=1000`
//! landed in `Subckt::params` and `X1 a b rpart Rv=100` landed in the call site's
//! `raw_params`, and nothing read either one: `hierarchy::expand_body` threaded a `port_map`
//! for nodes and had no scope at all for parameters. So the body's `{Rv}` reached the builder
//! as the name of a parameter nothing declared, and the deck below — the one this file exists
//! for — failed with `missing value for symbol 'Rv'`.
//!
//! What makes it worth a file of its own is the shape of the near miss. A top-level
//! `.param Rv=330` has always worked, so the evaluator and the symbol table were never the
//! problem; only the **scope** was missing. That is why the repair is a binding pass here
//! rather than anything in `expression.rs`, and why these assertions are about *which* value
//! each instance sees rather than about arithmetic.
//!
//! Three rules are asserted rather than assumed:
//!
//! 1. **two instances of one definition disagree about a value, on purpose** — the whole point
//!    of a parameterised subcircuit, and the thing that used to compile byte-identically;
//! 2. **a body sees its own formals and not its caller's** — SPICE's own scoping, and the only
//!    rule under which a definition means the same thing wherever it is called from;
//! 3. **an override that cannot be honoured is refused, not dropped** — a mistyped parameter
//!    name would otherwise solve a different circuit than the deck describes.

use general_mna::{build_system, parse_and_flatten};
use general_spice_core::ast::Statement;
use general_spice_core::dialect::Dialect;

/// The acceptance deck from the issue: one definition, two instances, two resistances.
const TWO_INSTANCES: &str = "\
.subckt RPART p n Rv=1000
R p n {Rv}
.ends RPART

V1 A 0 5
X1 A MID RPART Rv=100
X2 MID 0 RPART Rv=220
";

/// Flattens, or panics with the refusal.
fn flat(source: &str) -> Vec<Statement> {
    parse_and_flatten(source, Dialect::Ngspice)
        .unwrap_or_else(|e| panic!("flatten refused this deck: {e}"))
}

/// The refusal, or a panic saying the deck was accepted.
fn refusal(source: &str) -> String {
    match parse_and_flatten(source, Dialect::Ngspice) {
        Ok(_) => panic!("this deck was accepted:\n{source}"),
        Err(e) => e,
    }
}

/// Every element card in the flattened deck, as `name` -> its value tokens joined by spaces.
fn values(statements: &[Statement]) -> Vec<(String, String)> {
    statements
        .iter()
        .filter_map(|s| match s {
            Statement::ElementInstance(ei) => Some((ei.name.clone(), ei.raw_params.join(" "))),
            _ => None,
        })
        .collect()
}

#[test]
fn two_instances_of_one_definition_carry_their_own_value() {
    let values = values(&flat(TWO_INSTANCES));
    assert_eq!(
        values,
        vec![
            ("V1".to_string(), "5".to_string()),
            ("X1.R".to_string(), "{100}".to_string()),
            ("X2.R".to_string(), "{220}".to_string()),
        ],
        "the two instances must differ by the value each was given"
    );
}

#[test]
fn and_the_assembled_system_is_the_series_pair_those_values_describe() {
    // The assertion that cannot be satisfied by rewriting text: 100 Ω and 220 Ω in series
    // across a 5 V source. `A[0][0]` is node A's own conductance (1/100) and `A[1][1]` is
    // node MID's (1/100 + 1/220) -- the numbers a solver reads, derived independently here
    // rather than copied from a run.
    let system = build_system(TWO_INSTANCES, Dialect::Ngspice).expect("the deck builds");
    assert_eq!(system.mna.unknowns, vec!["V(A)", "V(MID)", "I(V1)"]);
    let evaluate = |row: usize, column: usize| {
        system.mna.a[(row, column)]
            .evaluate(&Default::default())
            .unwrap_or_else(|e| panic!("A[{row}][{column}] did not resolve to a number: {e}"))
    };
    assert!(
        (evaluate(0, 0) - 1.0 / 100.0).abs() < 1e-15,
        "{}",
        evaluate(0, 0)
    );
    assert!(
        (evaluate(1, 1) - (1.0 / 100.0 + 1.0 / 220.0)).abs() < 1e-15,
        "{}",
        evaluate(1, 1)
    );
    // And the pair is actually coupled, so this is one system rather than two decks.
    assert!(
        (evaluate(0, 1) + 1.0 / 100.0).abs() < 1e-15,
        "{}",
        evaluate(0, 1)
    );
}

#[test]
fn a_formal_falls_back_to_the_default_its_definition_declares() {
    let source = TWO_INSTANCES.replace("X1 A MID RPART Rv=100", "X1 A MID RPART");
    let values = values(&flat(&source));
    assert_eq!(values[1], ("X1.R".to_string(), "{1000}".to_string()));
    assert_eq!(values[2], ("X2.R".to_string(), "{220}".to_string()));
}

#[test]
fn a_body_sees_its_own_formals_and_none_of_its_callers() {
    // `inner` declares `Ri`; `outer` declares `Ro`. A scope that *extended* its caller's would
    // leave `inner`'s `{Ro}` resolvable -- and a definition would then mean different things
    // depending on who called it, which is the one thing a reusable definition cannot do.
    let source = "\
.subckt inner p n Ri=5
R p n {Ri}
RSHADOW p n {Ro}
.ends inner
.subckt outer p n Ro=9
X1 p n inner Ri=7
.ends outer
V1 A 0 5
XO A 0 outer Ro=3
";
    let values = values(&flat(source));
    let shadow = values
        .iter()
        .find(|(name, _)| name.ends_with("RSHADOW"))
        .expect("the shadow resistor survived expansion");
    assert_eq!(
        shadow.1, "{Ro}",
        "`Ro` belongs to the caller, so `inner` must not resolve it"
    );
    let bound = values
        .iter()
        .find(|(name, _)| name.ends_with(".R"))
        .expect("inner's own resistor survived expansion");
    assert_eq!(bound.1, "{7}", "and `inner`'s own formal must resolve");
}

#[test]
fn a_call_sites_value_is_written_in_the_callers_scope() {
    // `X1 ... Ri={Ro}` inside `outer`: the override is an expression in the *caller's* terms,
    // so it is resolved there before it is bound here. Without that, `inner` would be handed
    // the name `Ro`, which it cannot see by the rule above -- the override would bind to
    // nothing and fail two levels from where it was written.
    let source = "\
.subckt inner p n Ri=5
R p n {Ri}
.ends inner
.subckt outer p n Ro=9
X1 p n inner Ri={Ro}
.ends outer
V1 A 0 5
XO A 0 outer Ro=330
";
    let values = values(&flat(source));
    assert_eq!(values[1], ("XO.X1.R".to_string(), "{330}".to_string()));
}

#[test]
fn a_nested_definition_composes_both_the_path_and_the_value() {
    let source = "\
.subckt leaf p n Rl=1
R p n {Rl}
.ends leaf
.subckt mid p n Rm=2
X1 p n leaf Rl={Rm}
X2 p n leaf
.ends mid
V1 A 0 5
XM A 0 mid Rm=47
";
    let values = values(&flat(source));
    assert_eq!(
        values[1..],
        [
            ("XM.X1.R".to_string(), "{47}".to_string()),
            ("XM.X2.R".to_string(), "{1}".to_string()),
        ],
        "one instance takes the parent's value, the other its own default"
    );
}

#[test]
fn a_block_inside_a_parameterised_body_resolves_its_formals_too() {
    // A `.subckt` body is a group of statements, not a group of *electrical* statements, so a
    // signal-domain block inside one is as much part of the circuit as a resistor. A `{Kp}` it
    // could not resolve would fail as an undeclared symbol with nothing pointing at the
    // instance that was supposed to supply it.
    let source = "\
.subckt reg vin vout Kp=1
GAIN1 kind=gain gain={Kp} in=vin
.ends reg
V1 A 0 5
X1 A B reg Kp=12
";
    let blocks: Vec<(String, String)> = flat(source)
        .iter()
        .filter_map(|s| match s {
            Statement::BlockInstance(bi) => Some((
                bi.name.clone(),
                bi.fields
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join(" "),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(blocks.len(), 1, "{blocks:?}");
    assert!(
        blocks[0].1.contains("gain={12}"),
        "the block's own value must be bound too: {:?}",
        blocks[0]
    );
    // And the field that names a signal is still *resolved*, not substituted -- the two kinds
    // of field must not have swapped rules.
    assert!(blocks[0].1.contains("in=A"), "{:?}", blocks[0]);
}

#[test]
fn an_override_naming_a_parameter_the_definition_does_not_declare_is_refused() {
    let source = TWO_INSTANCES.replace("X1 A MID RPART Rv=100", "X1 A MID RPART Rw=100");
    let said = refusal(&source);
    assert!(said.contains("X1"), "does not name the instance: {said}");
    assert!(said.contains("Rw"), "does not name the parameter: {said}");
    assert!(
        said.contains("RPART"),
        "does not name the definition: {said}"
    );
    assert!(
        said.contains("Rv"),
        "does not say what it could have meant: {said}"
    );
}

#[test]
fn a_definition_declaring_no_parameters_says_so_rather_than_listing_nothing() {
    let source = "\
.subckt plain p n
R p n 1k
.ends plain
V1 A 0 5
X1 A 0 plain Rv=100
";
    let said = refusal(source);
    assert!(said.contains("no parameters"), "{said}");
}

#[test]
fn a_formal_with_neither_a_default_nor_a_value_is_refused_where_it_is_instantiated() {
    // A formal declared with no `=value` -- the `Option<String>` in `Subckt::params`. The
    // value has to come from somewhere, and letting it through would reach the builder as an
    // undeclared symbol, reported much later and naming a parameter rather than the instance
    // that owes it a value.
    //
    // It has to be written *after* a parameter that does carry an `=`: the parser switches
    // from the port list to the parameter list at the first token containing `=`, so
    // `.subckt rpart p n Rv` declares a third **port** called `Rv`, not a formal. That is
    // real SPICE's own ambiguity rather than a parser shortcut -- nothing in the line
    // distinguishes a bare name in the port list from a bare name after it -- and it is why
    // this deck says `Rbase=1k Rv`.
    let source = "\
.subckt rpart p n Rbase=1k Rv
R p n {Rv}
.ends rpart
V1 A 0 5
X1 A 0 rpart
";
    let said = refusal(source);
    assert!(said.contains("X1"), "{said}");
    assert!(said.contains("Rv"), "{said}");
    assert!(said.contains("no default"), "{said}");
}

#[test]
fn the_same_parameter_set_twice_on_one_call_is_refused() {
    let source = TWO_INSTANCES.replace("X1 A MID RPART Rv=100", "X1 A MID RPART Rv=100 Rv=220");
    let said = refusal(&source);
    assert!(said.contains("more than once"), "{said}");
    assert!(said.contains("Rv"), "{said}");
}

#[test]
fn a_param_inside_a_body_that_depends_on_a_formal_is_refused() {
    // A `.param` has one global name once the hierarchy is flat, so two instances would both
    // declare it and the last one would win. That is only *wrong* when its value depends on a
    // formal -- which is exactly when the two instances disagree -- so that case is refused
    // instead of being resolved to whichever came last.
    let source = "\
.subckt rpart p n Rv=1000
.param half={Rv}
R p n {half}
.ends rpart
V1 A 0 5
X1 A MID rpart Rv=100
X2 MID 0 rpart Rv=220
";
    let said = refusal(source);
    assert!(said.contains("half"), "{said}");
    assert!(said.contains("flatten"), "{said}");
    // And one that depends on nothing instance-specific still passes through, so the refusal
    // is about the dependency rather than about `.param` inside a body.
    let fine = source.replace(".param half={Rv}", ".param half=500");
    assert_eq!(values(&flat(&fine))[1].1, "{half}");
}

#[test]
fn a_deck_with_no_parameters_anywhere_flattens_exactly_as_it_always_did() {
    // The regression guard for every deck that worked before this pass existed: with no
    // formals declared, nothing is substituted and a braced value reaches the builder as the
    // symbol it was, for the top-level `.param` table to resolve.
    let source = "\
.param Rload=330
.subckt pair p n
R1 p mid {Rload}
R2 mid n {Rload}
.ends pair
V1 A 0 5
X1 A 0 pair
";
    let values = values(&flat(source));
    assert_eq!(
        values[1..],
        [
            ("X1.R1".to_string(), "{Rload}".to_string()),
            ("X1.R2".to_string(), "{Rload}".to_string()),
        ]
    );
    // And it still builds, because the top-level `.param` is what resolves it.
    build_system(source, Dialect::Ngspice).expect("a top-level .param still resolves");
}
