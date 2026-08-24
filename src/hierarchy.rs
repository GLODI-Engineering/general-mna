//! Expands `.subckt`/`X`-instance hierarchy into one flat `Vec<Statement>` before the existing
//! electrical (`MnaBuilder`) and block-graph (`build_kind`) builders ever see it — the same
//! approach every real SPICE tool uses: hierarchy is a real, reusable, dotted-path-named
//! structure at the *model* level (this module), but the actual numeric solve still operates on
//! one coupled, flat system (`general-mna`'s existing builders, unchanged).
//!
//! `.subckt` bodies may contain block/signal-domain statements too (nothing about
//! `general-spice-core`'s grammar restricts them to electrical elements — a `.subckt` is just a
//! named, reusable *group* of statements) — an instance's internal block graph is expanded and
//! dotted-path-renamed exactly like its electrical elements are.
//!
//! **Signal-domain ports**: a `.subckt`'s declared port list (`Subckt::nodes`) is just names,
//! positionally bound at the `X`-call site — nothing in it is electrical-specific. The same
//! `port_map` substitution [`expand_body`] already applies to every electrical node also applies,
//! unchanged, to a [`BlockInstance`]'s own name and every signal-reference field it carries — so
//! declaring an extra "port" purely to carry a signal binding, with no electrical meaning at all,
//! already works with zero extra machinery:
//!
//! - **Signal in**: `.subckt reg vin vout ctrl` with an internal block field `in=ctrl` — calling
//!   `X1 vsrc vload EXT_DUTY reg` binds `ctrl` to `EXT_DUTY`, so the internal block reads
//!   whatever `EXT_DUTY` (a block declared outside the subckt, at whatever scope `X1` itself is
//!   called from) produces.
//! - **Signal out**: an internal block whose own `.name` *is* the declared port name (e.g. a
//!   `kind=probe` block literally named `reading` inside `.subckt sensor vin vout reading`)
//!   becomes addressable from outside under whatever name the caller bound that port to (`X1 a b
//!   MEASURED sensor` exposes it as `MEASURED`) — symmetric with how an internal node named the
//!   same as an electrical port is externally addressable through the caller's own binding.
//!
//! A block reference that resolves to a name nothing declares (not a port, not another block in
//! the same body) still surfaces as the same `UnknownBlockInput` error it always has, just after
//! expansion instead of before.

use std::collections::HashMap;

use general_spice_core::ast::{BlockInstance, ElementInstance, Statement, Subckt};

/// Recursively expands every `.subckt`/`X`-instance pair in `statements` into one flat list,
/// with every internal device/block name dotted-path-prefixed by its instantiation chain (e.g.
/// `X1.R1`, `X1.X2.C3`) so multiple instances of the same subcircuit never collide and the
/// hierarchy stays visible in every name downstream (gate bindings, probes, CSV columns).
/// `.subckt`/`.ends` statements themselves are consumed, not passed through — the returned list
/// contains only the expanded, top-level-equivalent statements a flat builder already knows how
/// to handle.
pub fn flatten(statements: &[Statement]) -> Result<Vec<Statement>, String> {
    let (defs, root) = split_definitions(statements)?;
    expand_body(&root, &defs, "", &HashMap::new(), &[])
}

/// One `.subckt` definition: its own header (name/port list/params) plus every statement
/// lexically inside it, up to (not including) the matching `.ends` — nested `.subckt`
/// definitions inside it are pulled out into `defs` too (SPICE's own subckt namespace is flat
/// and global, not nested, regardless of where a `.subckt` is textually declared — docs/
/// GRAMMAR.md §6.1).
type Definitions = HashMap<String, (Subckt, Vec<Statement>)>;

/// Walks `statements` once, pulling every `.subckt ... .ends` pair out into `defs` (keyed by
/// name, case-sensitive — SPICE's own case-insensitivity is a `general-spice-core` symbol-table
/// concern this module doesn't re-implement; every existing/expected use writes the `X` call
/// with the exact declared case) and returning everything else as the top-level root to expand.
/// Mirrors `general_spice_core::symbols::scope::build_scope_tree`'s own nesting-tracked walk,
/// but flat (one global `defs` map) rather than a real tree, since subckt names are global.
fn split_definitions(statements: &[Statement]) -> Result<(Definitions, Vec<Statement>), String> {
    let mut defs = Definitions::new();
    let mut root = Vec::new();
    let mut stack: Vec<(Subckt, Vec<Statement>)> = Vec::new();

    for stmt in statements {
        match stmt {
            Statement::Subckt(sc) => stack.push((sc.clone(), Vec::new())),
            Statement::Ends(name, span) => {
                let Some((sc, body)) = stack.pop() else {
                    return Err(format!(
                        "line {}: .ends without a matching .subckt",
                        span.start
                    ));
                };
                if let Some(n) = name {
                    if n != &sc.name {
                        return Err(format!(
                            "line {}: .ends name '{n}' does not match .subckt name '{}'",
                            span.start, sc.name
                        ));
                    }
                }
                if defs.insert(sc.name.clone(), (sc.clone(), body)).is_some() {
                    return Err(format!("duplicate .subckt definition '{}'", sc.name));
                }
            }
            other => {
                if let Some((_, body)) = stack.last_mut() {
                    body.push(other.clone());
                } else {
                    root.push(other.clone());
                }
            }
        }
    }
    if let Some((sc, _)) = stack.last() {
        return Err(format!(".subckt '{}' has no matching .ends", sc.name));
    }
    Ok((defs, root))
}

/// Expands one body of statements (either the true top-level root, or one subckt instance's own
/// body) into a flat list. `prefix` is this call's own dotted-path prefix (`""` at the true
/// root, `"X1."` inside `X1`'s expansion, `"X1.X2."` inside a nested instance, ...); `port_map`
/// resolves this body's own port *names* (as declared in its `.subckt` header) to whatever the
/// caller actually bound them to (already-fully-resolved names, from the caller's own
/// perspective — resolution composes correctly across nesting depth because each level only
/// ever needs to resolve its own immediate ports). `chain` is the list of subckt names currently
/// being expanded, top-down, purely for cycle detection (a subckt directly or transitively
/// instantiating itself would recurse forever otherwise).
fn expand_body(
    body: &[Statement],
    defs: &Definitions,
    prefix: &str,
    port_map: &HashMap<String, String>,
    chain: &[String],
) -> Result<Vec<Statement>, String> {
    let resolve = |name: &str| -> String {
        // Ground ("0"/"gnd", case-insensitive) is SPICE's one globally-scoped node name -- true
        // even inside a .subckt body, never subject to dotted-path mangling or port binding
        // (mangling it would silently float every internal element whose only other terminal
        // is a real, externally-connected node, e.g. a capacitor to "0" inside a subckt body:
        // its "ground" terminal would become a private, otherwise-untouched node instead of the
        // one true reference the rest of the circuit shares, open-circuiting it in effect).
        if name == "0" || name.eq_ignore_ascii_case("gnd") {
            return name.to_string();
        }
        match port_map.get(name) {
            Some(bound) => bound.clone(),
            None => format!("{prefix}{name}"),
        }
    };

    let mut out = Vec::new();
    for stmt in body {
        match stmt {
            Statement::ElementInstance(ei) => {
                let mangled_nodes: Vec<String> = ei.nodes.iter().map(|n| resolve(n)).collect();
                match &ei.subckt_name {
                    None => {
                        let mut mangled = ei.clone();
                        mangled.name = format!("{prefix}{}", ei.name);
                        mangled.nodes = mangled_nodes;
                        out.push(Statement::ElementInstance(mangled));
                    }
                    Some(sub_name) => {
                        let instance_name = format!("{prefix}{}", ei.name);
                        out.extend(instantiate(
                            sub_name,
                            &instance_name,
                            &mangled_nodes,
                            defs,
                            chain,
                            ei,
                        )?);
                    }
                }
            }
            Statement::BlockInstance(bi) => {
                out.push(Statement::BlockInstance(mangle_block(bi, prefix, &resolve)));
            }
            // .subckt/.ends can't appear inside a body here -- split_definitions already
            // consumed every matching pair, at every nesting depth, into `defs`.
            Statement::Subckt(_) | Statement::Ends(_, _) => unreachable!(
                "split_definitions consumes every .subckt/.ends pair before expand_body runs"
            ),
            other => out.push(other.clone()),
        }
    }
    Ok(out)
}

/// Expands one `X`-instance: looks up its definition, builds the port-binding map (definition's
/// own declared ports -> the caller's already-resolved node list, positional, SPICE's own
/// convention), checks for a recursive instantiation cycle, and recursively expands the
/// definition's own body one level deeper.
fn instantiate(
    sub_name: &str,
    instance_name: &str,
    caller_nodes: &[String],
    defs: &Definitions,
    chain: &[String],
    call_site: &ElementInstance,
) -> Result<Vec<Statement>, String> {
    let Some((def, body)) = defs.get(sub_name) else {
        return Err(format!(
            "line {}: device '{}' instantiates undefined subcircuit '{sub_name}'",
            call_site.span.start, call_site.name
        ));
    };
    if chain.iter().any(|n| n == sub_name) {
        let mut cycle: Vec<&str> = chain.iter().map(String::as_str).collect();
        cycle.push(sub_name);
        return Err(format!(
            "recursive subcircuit instantiation: {}",
            cycle.join(" -> ")
        ));
    }
    if def.nodes.len() != caller_nodes.len() {
        return Err(format!(
            "line {}: device '{}' instantiates '{sub_name}' with {} node(s), but '{sub_name}' \
             declares {}",
            call_site.span.start,
            call_site.name,
            caller_nodes.len(),
            def.nodes.len()
        ));
    }

    let port_map: HashMap<String, String> = def
        .nodes
        .iter()
        .cloned()
        .zip(caller_nodes.iter().cloned())
        .collect();
    let new_prefix = format!("{instance_name}.");
    let mut new_chain = chain.to_vec();
    new_chain.push(sub_name.to_string());

    expand_body(body, defs, &new_prefix, &port_map, &new_chain)
}

/// Renames a block statement's own name (mangled, unless it happens to be a bound port — same
/// resolution rule as an electrical node) and every signal-reference field's value. `outputs=`
/// entries are always mangled (never port-resolved): exposing an inner block's *extra* output
/// as a signal port is part of the deferred external-signal-port feature this pass doesn't
/// implement (see this module's own doc comment).
fn mangle_block(
    bi: &BlockInstance,
    prefix: &str,
    resolve: &impl Fn(&str) -> String,
) -> BlockInstance {
    const SIGNAL_FIELDS: [&str; 5] = ["in", "inputs", "ctrl", "clamp_lo_in", "clamp_hi_in"];

    let mut mangled = bi.clone();
    mangled.name = resolve(&bi.name);
    for (key, value) in &mut mangled.fields {
        if key == "outputs" {
            *value = value
                .split(',')
                .map(|s| format!("{prefix}{s}"))
                .collect::<Vec<_>>()
                .join(",");
        } else if SIGNAL_FIELDS.contains(&key.as_str()) {
            *value = value
                .split(',')
                .map(|s| match s.strip_prefix("prev:") {
                    Some(name) => format!("prev:{}", resolve(name)),
                    None => resolve(s),
                })
                .collect::<Vec<_>>()
                .join(",");
        }
    }
    mangled
}

#[cfg(test)]
mod tests {
    use super::*;
    use general_spice_core::ast::{ElementInstance, Statement};

    fn element(letter: char, name: &str, nodes: &[&str]) -> Statement {
        Statement::ElementInstance(ElementInstance {
            device_letter: letter,
            name: name.to_string(),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
            raw_params: vec!["100".to_string()],
            subckt_name: None,
            span: 1..2,
        })
    }

    fn x_call(name: &str, nodes: &[&str], subckt: &str) -> Statement {
        Statement::ElementInstance(ElementInstance {
            device_letter: 'X',
            name: name.to_string(),
            nodes: nodes.iter().map(|s| s.to_string()).collect(),
            raw_params: vec![],
            subckt_name: Some(subckt.to_string()),
            span: 1..2,
        })
    }

    fn subckt(name: &str, ports: &[&str]) -> Statement {
        Statement::Subckt(Subckt {
            name: name.to_string(),
            nodes: ports.iter().map(|s| s.to_string()).collect(),
            params: vec![],
            span: 1..2,
        })
    }

    fn ends(name: &str) -> Statement {
        Statement::Ends(Some(name.to_string()), 1..2)
    }

    #[test]
    fn a_single_instance_expands_with_dotted_path_names() {
        let statements = vec![
            subckt("div", &["a", "b"]),
            element('R', "R1", &["a", "mid"]),
            element('R', "R2", &["mid", "b"]),
            ends("div"),
            x_call("X1", &["vin", "0"], "div"),
        ];
        let flat = flatten(&statements).unwrap();
        assert_eq!(flat.len(), 2);
        let names: Vec<&str> = flat
            .iter()
            .map(|s| match s {
                Statement::ElementInstance(ei) => ei.name.as_str(),
                _ => panic!("expected ElementInstance"),
            })
            .collect();
        assert_eq!(names, vec!["X1.R1", "X1.R2"]);
        let Statement::ElementInstance(r1) = &flat[0] else {
            unreachable!()
        };
        assert_eq!(r1.nodes, vec!["vin", "X1.mid"]);
        let Statement::ElementInstance(r2) = &flat[1] else {
            unreachable!()
        };
        assert_eq!(r2.nodes, vec!["X1.mid", "0"]);
    }

    #[test]
    fn two_instances_of_the_same_subckt_get_distinct_internal_names() {
        let statements = vec![
            subckt("div", &["a", "b"]),
            element('R', "R1", &["a", "mid"]),
            element('R', "R2", &["mid", "b"]),
            ends("div"),
            x_call("X1", &["n1", "0"], "div"),
            x_call("X2", &["n2", "0"], "div"),
        ];
        let flat = flatten(&statements).unwrap();
        assert_eq!(flat.len(), 4);
        let Statement::ElementInstance(x1_r1) = &flat[0] else {
            unreachable!()
        };
        let Statement::ElementInstance(x2_r1) = &flat[2] else {
            unreachable!()
        };
        assert_eq!(x1_r1.nodes[1], "X1.mid");
        assert_eq!(x2_r1.nodes[1], "X2.mid");
        assert_ne!(x1_r1.name, x2_r1.name);
    }

    #[test]
    fn nested_subckt_instantiation_composes_the_dotted_path() {
        let statements = vec![
            subckt("inner", &["p", "q"]),
            element('R', "R1", &["p", "q"]),
            ends("inner"),
            subckt("outer", &["a", "b"]),
            x_call("XI", &["a", "b"], "inner"),
            ends("outer"),
            x_call("X1", &["vin", "0"], "outer"),
        ];
        let flat = flatten(&statements).unwrap();
        assert_eq!(flat.len(), 1);
        let Statement::ElementInstance(r1) = &flat[0] else {
            unreachable!()
        };
        assert_eq!(r1.name, "X1.XI.R1");
        assert_eq!(r1.nodes, vec!["vin", "0"]);
    }

    #[test]
    fn ground_inside_a_subckt_body_stays_the_global_reference_not_a_private_node() {
        // A capacitor whose far terminal is "0" inside the body must resolve to the real,
        // globally-shared ground -- not a per-instance private node -- or it silently floats
        // (open-circuits), decoupled from the rest of the circuit and from other instances'
        // own "0" references.
        let statements = vec![
            subckt("rc_leg", &["in", "out"]),
            element('R', "R1", &["in", "out"]),
            element('C', "C1", &["out", "0"]),
            ends("rc_leg"),
            x_call("X1", &["vin", "mid"], "rc_leg"),
            x_call("X2", &["mid", "vout"], "rc_leg"),
        ];
        let flat = flatten(&statements).unwrap();
        let grounded: Vec<&str> = flat
            .iter()
            .filter_map(|s| match s {
                Statement::ElementInstance(ei) if ei.name.ends_with(".C1") => {
                    Some(ei.nodes[1].as_str())
                }
                _ => None,
            })
            .collect();
        assert_eq!(grounded, vec!["0", "0"]);
    }

    #[test]
    fn a_recursive_subckt_is_rejected_not_infinite_looped() {
        let statements = vec![
            subckt("loop", &["a"]),
            x_call("X1", &["a"], "loop"),
            ends("loop"),
            x_call("XTOP", &["n"], "loop"),
        ];
        let err = flatten(&statements).unwrap_err();
        assert!(err.contains("recursive"), "unexpected error: {err}");
    }

    #[test]
    fn an_undefined_subckt_reference_is_a_clear_error() {
        let statements = vec![x_call("X1", &["a", "b"], "nope")];
        let err = flatten(&statements).unwrap_err();
        assert!(err.contains("undefined subcircuit"), "unexpected: {err}");
    }

    #[test]
    fn a_node_count_mismatch_is_a_clear_error() {
        let statements = vec![
            subckt("div", &["a", "b"]),
            element('R', "R1", &["a", "b"]),
            ends("div"),
            x_call("X1", &["only_one"], "div"),
        ];
        let err = flatten(&statements).unwrap_err();
        assert!(err.contains("node(s)"), "unexpected error: {err}");
    }

    #[test]
    fn a_block_statement_inside_a_subckt_body_is_mangled_and_its_signal_refs_rewritten() {
        use general_spice_core::ast::BlockInstance;

        let statements = vec![
            subckt("reg", &["vin", "vout"]),
            Statement::BlockInstance(BlockInstance {
                name: "DUTY".to_string(),
                fields: vec![("kind".to_string(), "const".to_string())],
                span: 1..2,
            }),
            Statement::BlockInstance(BlockInstance {
                name: "MOD".to_string(),
                fields: vec![
                    ("kind".to_string(), "pwm".to_string()),
                    ("in".to_string(), "DUTY".to_string()),
                ],
                span: 1..2,
            }),
            ends("reg"),
            x_call("X1", &["vin", "vout"], "reg"),
        ];
        let flat = flatten(&statements).unwrap();
        assert_eq!(flat.len(), 2);
        let Statement::BlockInstance(duty) = &flat[0] else {
            unreachable!()
        };
        assert_eq!(duty.name, "X1.DUTY");
        let Statement::BlockInstance(modu) = &flat[1] else {
            unreachable!()
        };
        assert_eq!(modu.name, "X1.MOD");
        let in_field = modu.fields.iter().find(|(k, _)| k == "in").unwrap();
        assert_eq!(in_field.1, "X1.DUTY");
    }

    #[test]
    fn a_signal_can_be_bound_into_a_subckt_instance_from_outside_via_a_declared_port() {
        // A .subckt's port list isn't electrical-only -- it's just names, positionally bound at
        // the call site. Declaring an extra "port" purely to carry a signal name in/out lets an
        // external block feed a subckt instance's internal block graph (or vice versa), the same
        // uniform dotted-path resolution used for electrical nodes.
        use general_spice_core::ast::BlockInstance;

        let statements = vec![
            Statement::BlockInstance(BlockInstance {
                name: "EXT_DUTY".to_string(),
                fields: vec![("kind".to_string(), "const".to_string())],
                span: 1..2,
            }),
            subckt("reg", &["vin", "vout", "ctrl"]),
            Statement::BlockInstance(BlockInstance {
                name: "MOD".to_string(),
                fields: vec![
                    ("kind".to_string(), "pwm".to_string()),
                    ("in".to_string(), "ctrl".to_string()),
                ],
                span: 1..2,
            }),
            ends("reg"),
            x_call("X1", &["vsrc", "vload", "EXT_DUTY"], "reg"),
        ];
        let flat = flatten(&statements).unwrap();
        assert_eq!(flat.len(), 2);
        let Statement::BlockInstance(ext_duty) = &flat[0] else {
            unreachable!()
        };
        assert_eq!(ext_duty.name, "EXT_DUTY");
        let Statement::BlockInstance(modu) = &flat[1] else {
            unreachable!()
        };
        assert_eq!(modu.name, "X1.MOD");
        // "ctrl" is a declared port bound to "EXT_DUTY" at the call site -- it must resolve to
        // the real external block's own (unprefixed, since it's declared at the true root) name,
        // not "X1.ctrl" or "X1.EXT_DUTY".
        let in_field = modu.fields.iter().find(|(k, _)| k == "in").unwrap();
        assert_eq!(in_field.1, "EXT_DUTY");
    }

    #[test]
    fn a_subckt_instance_can_expose_an_internal_block_as_a_signal_output_port() {
        // The reverse direction: an internal block's own output reaching outside the instance
        // through a declared port, so an external block can read it as an ordinary
        // Signal::Block(name) reference.
        use general_spice_core::ast::BlockInstance;

        let statements = vec![
            subckt("sensor", &["vin", "vout", "reading"]),
            element('R', "R1", &["vin", "vout"]),
            Statement::BlockInstance(BlockInstance {
                name: "reading".to_string(),
                fields: vec![
                    ("kind".to_string(), "probe".to_string()),
                    ("node".to_string(), "vout".to_string()),
                ],
                span: 1..2,
            }),
            ends("sensor"),
            x_call("X1", &["a", "b", "MEASURED"], "sensor"),
            Statement::BlockInstance(BlockInstance {
                name: "GAIN1".to_string(),
                fields: vec![
                    ("kind".to_string(), "gain".to_string()),
                    ("in".to_string(), "MEASURED".to_string()),
                ],
                span: 1..2,
            }),
        ];
        let flat = flatten(&statements).unwrap();
        let sensor_block = flat
            .iter()
            .find_map(|s| match s {
                Statement::BlockInstance(bi)
                    if bi.fields.iter().any(|(k, v)| k == "kind" && v == "probe") =>
                {
                    Some(bi)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(sensor_block.name, "MEASURED");
        let gain_block = flat
            .iter()
            .find_map(|s| match s {
                Statement::BlockInstance(bi) if bi.name == "GAIN1" => Some(bi),
                _ => None,
            })
            .unwrap();
        let in_field = gain_block.fields.iter().find(|(k, _)| k == "in").unwrap();
        assert_eq!(in_field.1, "MEASURED");
    }
}
