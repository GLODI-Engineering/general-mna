//! A `.model` inside a `.subckt` body that depends on a formal parameter is refused, and one
//! that does not still passes through (#67).
//!
//! #63 bound a definition's formal parameters and substituted them into every expression
//! segment its body expands — device cards, block fields, `.IC` values — and refused a
//! body-level `.param` whose value depends on a formal, because a `.param`'s name is global
//! once the hierarchy is flat: two instances would both declare it and whichever expanded last
//! would win.
//!
//! A `.model` has exactly that property and had none of the handling. `.model DX D(IS={Isat})`
//! inside a parameterised body was passed through untouched, which fails *later* as an
//! undeclared symbol naming the parameter rather than the instance that was supposed to supply
//! it — and substituting it instead would have produced the silent case the `.param` rule
//! exists to refuse: two `.model DX` cards with different values, one of them discarded without
//! a word.
//!
//! So the rule is the `.param` one, and this file asserts **both halves of it** rather than
//! one. A refusal nothing can pass is as wrong as a drop nothing refuses: if these tests only
//! proved that a parameterised `.model` is refused, the same green suite would cover an
//! implementation that refused every `.model` in every body, which would break the one kind
//! anything in this project actually writes.

use general_mna::parse_and_flatten;
use general_spice_core::ast::Statement;
use general_spice_core::dialect::Dialect;

/// A diode-and-resistor definition instantiated twice, with the body's `.model` line left as a
/// `{}` placeholder for each test to fill in.
fn deck(model_line: &str) -> String {
    format!(
        "\
.subckt DPART p n Rv=1000 Isat=1e-14
{model_line}
R p n {{Rv}}
D p n DX
.ends DPART

V1 A 0 5
X1 A MID DPART Rv=100
X2 MID 0 DPART Rv=220
"
    )
}

/// Flattens, or panics with the refusal.
fn flat(source: &str) -> Vec<Statement> {
    parse_and_flatten(source, Dialect::Ngspice)
        .unwrap_or_else(|e| panic!("flatten refused this deck: {e}\n{source}"))
}

/// The refusal, or a panic saying the deck was accepted.
fn refusal(source: &str) -> String {
    match parse_and_flatten(source, Dialect::Ngspice) {
        Ok(_) => panic!("this deck was accepted:\n{source}"),
        Err(e) => e,
    }
}

/// Every `.model` in the flattened deck, as `name` -> its parameter text, in order.
fn models(statements: &[Statement]) -> Vec<(String, String)> {
    statements
        .iter()
        .filter_map(|s| match s {
            Statement::Model(m) => Some((m.name.clone(), m.raw_params.clone())),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Refused: the value is instance-specific, so the single global name cannot carry it
// ---------------------------------------------------------------------------------------------

#[test]
fn a_model_in_a_body_written_in_terms_of_a_formal_is_refused() {
    let said = refusal(&deck(".model DX D(IS={Isat})"));
    assert!(said.contains("DX"), "does not name the model: {said}");
    assert!(said.contains("Isat"), "does not name the parameter: {said}");
    assert!(
        said.contains("X1") || said.contains("X2"),
        "does not name the instance it was expanding: {said}"
    );
    assert!(said.contains("line 2"), "does not name the line: {said}");
}

#[test]
fn the_refusal_says_why_rather_than_only_that_it_refused() {
    let said = refusal(&deck(".model DX D(IS={Isat})"));
    assert!(
        said.contains("global"),
        "does not say the name is global once flattened: {said}"
    );
    assert!(
        said.contains("device card"),
        "does not say where the value belongs instead: {said}"
    );
}

#[test]
fn the_single_quoted_expression_form_is_refused_too() {
    // `'...'` is the other expression delimiter the grammar provides (docs/GRAMMAR.md §5), so a
    // rule that only looked inside braces would be a rule with a spelling that evades it.
    let said = refusal(&deck(".model DX D(IS='Isat*2')"));
    assert!(said.contains("Isat"), "does not name the parameter: {said}");
}

#[test]
fn every_formal_the_model_names_is_listed_not_just_the_first() {
    let said = refusal(&deck(".model DX D(IS={Isat} RS={Rv})"));
    assert!(said.contains("Isat"), "does not name Isat: {said}");
    assert!(said.contains("Rv"), "does not name Rv: {said}");
}

#[test]
fn a_model_in_a_nested_body_is_refused_with_the_whole_instantiation_path() {
    let source = "\
.subckt INNER p n Isat=1e-14
.model DX D(IS={Isat})
D p n DX
.ends INNER

.subckt OUTER p n
X9 p n INNER Isat=2e-14
.ends OUTER

V1 A 0 5
X1 A 0 OUTER
";
    let said = refusal(source);
    assert!(
        said.contains("X1.X9"),
        "does not name the instantiation path: {said}"
    );
}

// ---------------------------------------------------------------------------------------------
// Accepted: the value means the same thing in every instance, so one global name carries it
// ---------------------------------------------------------------------------------------------

#[test]
fn a_model_in_a_parameterised_body_that_depends_on_nothing_instance_specific_still_passes() {
    // The body *is* parameterised and the resistor *does* use the formal -- only the `.model`
    // does not. This is the case every existing deck in this project writes, and the one a
    // blanket refusal would break.
    let flat = flat(&deck(".model DX D(IS=1e-14 N=1)"));
    assert_eq!(
        models(&flat),
        vec![
            ("DX".to_string(), "(IS=1e-14 N=1)".to_string()),
            ("DX".to_string(), "(IS=1e-14 N=1)".to_string()),
        ],
        "the two instances' identical models should both pass through byte-identically"
    );
}

#[test]
fn a_model_in_a_body_written_in_terms_of_a_global_param_still_passes() {
    // A global `.param` is the same value for every instance, so the `.model` means one thing
    // however many times it expands. It is left as written for the solver's own symbol table to
    // resolve -- which is also why this pass must ask about formals by name rather than refuse
    // every brace it finds.
    let source = format!(
        ".param GLOBAL_IS=1e-14\n{}",
        deck(".model DX D(IS={GLOBAL_IS})")
    );
    let flat = flat(&source);
    assert_eq!(
        models(&flat),
        vec![
            ("DX".to_string(), "(IS={GLOBAL_IS})".to_string()),
            ("DX".to_string(), "(IS={GLOBAL_IS})".to_string()),
        ],
        "a global reference should reach the solver as written, not be refused or substituted"
    );
}

#[test]
fn a_formals_name_inside_a_longer_identifier_is_not_a_reference_to_it() {
    // `Isat` is a formal; `Isatur` is not. Matching on substrings rather than whole identifiers
    // would refuse this deck, and the refusal would be impossible to argue with from the text.
    let flat = flat(&deck(".model DX D(IS={Isatur})"));
    assert_eq!(
        models(&flat)[0].1,
        "(IS={Isatur})".to_string(),
        "a longer identifier that merely starts with a formal's name should pass through"
    );
}

#[test]
fn a_top_level_model_is_untouched_by_this_rule() {
    // Nothing declares a formal at the true top level, so no `.model` there can depend on one.
    // Asserted rather than assumed because the rule is implemented in the same pass that
    // walks the top level.
    let source = "\
.param GLOBAL_IS=1e-14
.model DX D(IS={GLOBAL_IS})
V1 A 0 5
D1 A 0 DX
";
    let flat = flat(source);
    assert_eq!(
        models(&flat),
        vec![("DX".to_string(), "(IS={GLOBAL_IS})".to_string())]
    );
}

#[test]
fn a_model_in_an_unparameterised_body_passes_however_it_is_written() {
    // No formals at all in scope: there is nothing for a reference to be instance-specific
    // *about*, so even a braced one is the solver's business.
    let source = "\
.subckt DPART p n
.model DX D(IS={GLOBAL_IS})
D p n DX
.ends DPART

.param GLOBAL_IS=1e-14
V1 A 0 5
X1 A 0 DPART
";
    assert_eq!(
        models(&flat(source)),
        vec![("DX".to_string(), "(IS={GLOBAL_IS})".to_string())]
    );
}

// ---------------------------------------------------------------------------------------------
// The rule did not change what #63 already settled
// ---------------------------------------------------------------------------------------------

#[test]
fn the_body_s_device_cards_are_still_substituted_around_the_model() {
    // The `.model` is refused *instead of* being substituted, and this asserts the choice did
    // not quietly stop the substitution that #63 established for everything else on the body.
    let flat = flat(&deck(".model DX D(IS=1e-14)"));
    let resistors: Vec<String> = flat
        .iter()
        .filter_map(|s| match s {
            Statement::ElementInstance(ei) if ei.device_letter == 'R' => {
                Some(format!("{} {}", ei.name, ei.raw_params.join(" ")))
            }
            _ => None,
        })
        .collect();
    assert_eq!(resistors, vec!["X1.R {100}", "X2.R {220}"]);
}
