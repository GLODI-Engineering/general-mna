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
//!   `kind=phys2sig` block literally named `reading` inside `.subckt sensor vin vout reading`)
//!   becomes addressable from outside under whatever name the caller bound that port to (`X1 a b
//!   MEASURED sensor` exposes it as `MEASURED`) — symmetric with how an internal node named the
//!   same as an electrical port is externally addressable through the caller's own binding.
//!
//! A block reference that resolves to a name nothing declares (not a port, not another block in
//! the same body) still surfaces as the same `UnknownBlockInput` error it always has, just after
//! expansion instead of before.
//!
//! **Formal parameters**: a `.subckt`'s header may declare parameters with defaults
//! (`.subckt rpart p n Rv=1000`) and an `X` call may override them (`X1 a b rpart Rv=100`).
//! Both halves parse, and until general-mna#63 neither was *bound*: the body's `{Rv}` reached
//! the builder as the name of a parameter nothing declared, and the solve failed with a missing
//! symbol. [`expand_body`] now carries a parameter scope beside its `port_map`, built by
//! [`bind_params`] from the definition's declared defaults overridden by the call site's
//! `name=value` tail, and substitutes it into every expression segment it expands — so `{Rv}`
//! inside the body becomes `{100}` for one instance and `{220}` for the next.
//!
//! The scope **shadows rather than extends**: a body sees its own formals and whatever is
//! global, never the caller's formals, which is SPICE's own rule and the only one under which a
//! definition means the same thing wherever it is called from. An override naming a parameter
//! the definition does not declare is refused rather than ignored, because a dropped override
//! is a silently different circuit.
//!
//! **A `.param` or a `.model` inside a body** is *not* scoped by this pass, and cannot be: both
//! names are global once the hierarchy is flat, so two instances of one definition would declare
//! the same name twice and the later one would win. Where the value depends on a formal — exactly
//! the case where the two instances disagree — that is refused with the parameter named, rather
//! than quietly resolved to one instance's value (`.param`: general-mna#63; `.model`:
//! general-mna#67). Either one that depends on nothing instance-specific passes through as it
//! always has.
//!
//! **`.IC` inside a body**: a `.IC V(<node>)=` / `V(<n1>,<n2>)=` / `I(<element>)=` written inside
//! a `.subckt` names that body's own nodes and elements, so each target goes through the very
//! same resolution ([`mangle_ic_target`]) before the builder's unknown-name check sees it.

use std::collections::HashMap;

use general_spice_core::ast::{BlockInstance, ElementInstance, Statement, Subckt};

/// Recursively expands every `.subckt`/`X`-instance pair in `statements` into one flat list,
/// with every internal device/block name dotted-path-prefixed by its instantiation chain (e.g.
/// `X1.R1`, `X1.X2.C3`) so multiple instances of the same subcircuit never collide and the
/// hierarchy stays visible in every name downstream (gate bindings, phys2sig blocks, CSV columns).
/// `.subckt`/`.ends` statements themselves are consumed, not passed through — the returned list
/// contains only the expanded, top-level-equivalent statements a flat builder already knows how
/// to handle.
pub fn flatten(statements: &[Statement]) -> Result<Vec<Statement>, String> {
    let (defs, root) = split_definitions(statements)?;
    expand_body(&root, &defs, "", &HashMap::new(), &HashMap::new(), &[])
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
/// ever needs to resolve its own immediate ports). `params` is the same thing for this body's
/// *formal parameters* — each declared name mapped to the text the call site bound it to,
/// already resolved in the caller's own scope ([`bind_params`]) — and it replaces rather than
/// extends the caller's, so a body sees its own formals and nothing of its caller's. `chain` is
/// the list of subckt names currently being expanded, top-down, purely for cycle detection (a
/// subckt directly or transitively instantiating itself would recurse forever otherwise).
fn expand_body(
    body: &[Statement],
    defs: &Definitions,
    prefix: &str,
    port_map: &HashMap<String, String>,
    params: &HashMap<String, String>,
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

    let substitute = |text: &str| -> String { substitute_params(text, params) };

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
                        // Every value this card carries is written in the body's own scope, so
                        // a formal parameter named in one is resolved here, before any reader
                        // of the card sees it. A token with no expression segment -- a model
                        // name, a plain number, the `F`/`H` control name handled just below --
                        // is returned unchanged, so this cannot touch what it does not mean to.
                        mangled.raw_params = ei
                            .raw_params
                            .iter()
                            .map(|value| substitute(value))
                            .collect();
                        // An `F`/`H` card names the voltage source whose branch current
                        // controls it, and that name is scoped to the body it is written in.
                        // Flattening renames every element in that body, so the reference has
                        // to move with them. Without this, `F1 p1 p2 VS 0.25` inside a
                        // `.subckt` survives expansion still pointing at `VS` while the source
                        // itself became `X1.VS`, and the build fails with
                        // `unknown controlling branch 'VS'`.
                        //
                        // This is what stopped an ideal transformer from being a reusable
                        // subcircuit: the E/F pair one is built from needs exactly this
                        // reference, and it is the only device grammar that carries an element
                        // name in `raw_params` rather than a node in `nodes`.
                        if matches!(ei.device_letter, 'F' | 'H') {
                            if let Some(control) = mangled.raw_params.first_mut() {
                                *control = format!("{prefix}{control}");
                            }
                        }
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
                            params,
                        )?);
                    }
                }
            }
            Statement::BlockInstance(bi) => {
                out.push(Statement::BlockInstance(mangle_block(
                    bi,
                    prefix,
                    &resolve,
                    &substitute,
                )));
            }
            Statement::Ic(assignments, span) => {
                let mangled = assignments
                    .iter()
                    .map(|(target, value)| (mangle_ic_target(target, &resolve), substitute(value)))
                    .collect();
                out.push(Statement::Ic(mangled, span.clone()));
            }
            // A `.param` written inside a body is *not* scoped by this pass: once the hierarchy
            // is flat its name is global, so two instances of the same definition would both
            // declare it and whichever came last would win. That is only wrong when its value
            // depends on a formal -- precisely the case where the two instances disagree -- so
            // that case is refused rather than quietly resolved to one of them. A body-level
            // `.param` that depends on nothing instance-specific means the same thing however
            // many times it is expanded, and passes through as it always has.
            Statement::Param(p) | Statement::GlobalParam(p) => {
                for (name, raw) in &p.assignments {
                    let named = formals_named_in(raw, params);
                    if !named.is_empty() {
                        return Err(format!(
                            "line {}: '.param {name}'{} is written in terms of the formal \
                             parameter{} {}, and a `.param` is global once the hierarchy is \
                             flattened -- two instances would disagree about its value. Write \
                             the expression at the use site instead.",
                            p.span.start,
                            written_inside(prefix),
                            if named.len() == 1 { "" } else { "s" },
                            named.join(", "),
                        ));
                    }
                }
                out.push(stmt.clone());
            }
            // A `.model`'s name has the same property a `.param`'s does -- global once the
            // hierarchy is flat -- and none of the handling, which is general-mna#67. Two
            // instances of a definition carrying `.model DX D(IS={Isat})` would emit two
            // `.model DX` cards with different values, and whichever came last would win: the
            // silent wrong answer the `.param` rule exists to refuse. So it is refused on the
            // same grounds, rather than substituted into. A `.model` that depends on nothing
            // instance-specific -- the only kind anything in this project writes today -- means
            // the same thing however many times it is expanded and passes through untouched,
            // including one written in terms of a global `.param`, which is not a formal and so
            // is not named here.
            Statement::Model(m) => {
                let named = formals_named_in(&m.raw_params, params);
                if !named.is_empty() {
                    return Err(format!(
                        "line {}: '.model {}'{} is written in terms of the formal parameter{} \
                         {}, and a `.model`'s name is global once the hierarchy is flattened -- \
                         two instances would emit the same model with different values and the \
                         last one would win. Write the instance-specific value on the device \
                         card instead, where each instance has its own.",
                        m.span.start,
                        m.name,
                        written_inside(prefix),
                        if named.len() == 1 { "" } else { "s" },
                        named.join(", "),
                    ));
                }
                out.push(stmt.clone());
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

/// Where a refused body-level statement was written, for a message a reader can act on:
/// `" inside instance 'X1.X2'"` when there is an instantiation path, and nothing at all at the
/// true top level — where neither refusal can fire, since nothing declares a formal there, but
/// where a bare line number would otherwise read as though an instance were involved.
fn written_inside(prefix: &str) -> String {
    match prefix.trim_end_matches('.') {
        "" => String::new(),
        path => format!(" inside instance '{path}'"),
    }
}

/// Expands one `X`-instance: looks up its definition, builds the port-binding map (definition's
/// own declared ports -> the caller's already-resolved node list, positional, SPICE's own
/// convention) and the parameter scope ([`bind_params`]), checks for a recursive instantiation
/// cycle, and recursively expands the definition's own body one level deeper.
fn instantiate(
    sub_name: &str,
    instance_name: &str,
    caller_nodes: &[String],
    defs: &Definitions,
    chain: &[String],
    call_site: &ElementInstance,
    caller_params: &HashMap<String, String>,
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
    let params = bind_params(def, call_site, caller_params)?;
    let new_prefix = format!("{instance_name}.");
    let mut new_chain = chain.to_vec();
    new_chain.push(sub_name.to_string());

    expand_body(body, defs, &new_prefix, &port_map, &params, &new_chain)
}

/// Binds one instance's formal parameters: every name the definition declares, mapped to the
/// text that instance gives it, falling back to the definition's own declared default.
///
/// Three refusals, each because the alternative is a circuit that is quietly not the one that
/// was written:
///
/// - an override naming a parameter the definition does not declare — ignoring it would solve a
///   different circuit than the deck says, and a typo in a parameter name is exactly how that
///   happens;
/// - the same parameter set twice on one call, which has no defensible reading;
/// - a declared parameter with neither a default nor a value, which would otherwise reach the
///   builder as an undeclared symbol and be reported much later, somewhere else.
///
/// A call-site value is written in the *caller's* scope, so it is resolved there
/// ([`substitute_params`] against `caller_params`) before it is bound — that is what makes
/// `X1 a b inner Rv={Rout}` work inside a definition whose own formal is `Rout`. A *default* is
/// written in the definition's scope and is left as text, so a default naming a global
/// `.param` still resolves at the top level where that `.param` lives.
fn bind_params(
    def: &Subckt,
    call_site: &ElementInstance,
    caller_params: &HashMap<String, String>,
) -> Result<HashMap<String, String>, String> {
    let declared = |name: &str| def.params.iter().any(|(formal, _)| formal == name);

    let mut given: HashMap<String, String> = HashMap::new();
    for token in &call_site.raw_params {
        let Some((name, value)) = token.split_once('=') else {
            return Err(format!(
                "line {}: device '{}' instantiates '{}' with a trailing token '{token}' that is \
                 not <name>=<value>",
                call_site.span.start, call_site.name, def.name
            ));
        };
        let name = name.trim();
        if name.is_empty() {
            return Err(format!(
                "line {}: device '{}' instantiates '{}' with a parameter assignment '{token}' \
                 that has no parameter name",
                call_site.span.start, call_site.name, def.name
            ));
        }
        if !declared(name) {
            let names: Vec<&str> = def.params.iter().map(|(f, _)| f.as_str()).collect();
            return Err(format!(
                "line {}: device '{}' sets parameter '{name}', which '.subckt {}' does not \
                 declare ({})",
                call_site.span.start,
                call_site.name,
                def.name,
                if names.is_empty() {
                    "it declares no parameters".to_string()
                } else {
                    format!("it declares {}", names.join(", "))
                }
            ));
        }
        if given
            .insert(
                name.to_string(),
                substitute_params(value.trim(), caller_params),
            )
            .is_some()
        {
            return Err(format!(
                "line {}: device '{}' sets parameter '{name}' more than once",
                call_site.span.start, call_site.name
            ));
        }
    }

    let mut bound = HashMap::new();
    for (formal, default) in &def.params {
        let value = match given.remove(formal) {
            Some(value) => value,
            None => match default {
                Some(default) => default.clone(),
                None => {
                    return Err(format!(
                        "line {}: device '{}' gives no value for '.subckt {}' parameter \
                         '{formal}', which declares no default",
                        call_site.span.start, call_site.name, def.name
                    ))
                }
            },
        };
        bound.insert(formal.clone(), value);
    }
    Ok(bound)
}

/// Substitutes `params` into the *expression segments* of one value, leaving everything else as
/// written: `{Rv}` becomes `{100}` and `'Rv*2'` becomes `'100*2'`, while `100k`, a model name
/// and an `F`-card's control-source name are returned byte-identical.
///
/// Only `{...}` and `'...'` are rewritten, and this is a deliberate limit rather than an
/// oversight: a bare token in a value position can be a model name, a device keyword or a
/// behavioural-source fragment, so substituting a bare `Rv` would mean guessing which of those
/// a three-letter token is. A netlist that wants a parameter in a value says so with the braces
/// the grammar provides (`docs/GRAMMAR.md` §5).
fn substitute_params(text: &str, params: &HashMap<String, String>) -> String {
    if params.is_empty() || !text.contains(['{', '\'']) {
        return text.to_string();
    }
    rewrite_segments(text, |segment, out| {
        out.push_str(&map_identifiers(segment, |ident| {
            params.get(ident).map(|value| bound_text(value))
        }));
    })
}

/// Every parameter in `params` that one value's expression segments actually name, sorted and
/// without repeats — the question [`expand_body`] asks of a `.param` or a `.model` written
/// inside a body before refusing it.
///
/// It answers by **name** rather than by whether substituting changed the text, because the two
/// differ in one case that matters: a parameter bound to its own name (`X1 a b def Rv=Rv`, where
/// the caller declares no `Rv` either) substitutes to byte-identical text, so a
/// changed-or-not test would call that body instance-independent when it is not. Sharing
/// [`rewrite_segments`] and [`map_identifiers`] with [`substitute_params`] is what keeps the two
/// answers about one piece of text — *does it depend on a formal* and *what does it become* —
/// from drifting apart as the grammar they scan grows.
fn formals_named_in(text: &str, params: &HashMap<String, String>) -> Vec<String> {
    if params.is_empty() || !text.contains(['{', '\'']) {
        return Vec::new();
    }
    let mut named: Vec<String> = Vec::new();
    rewrite_segments(text, |segment, _| {
        let _ = map_identifiers(segment, |ident| {
            if params.contains_key(ident) && !named.iter().any(|seen| seen == ident) {
                named.push(ident.to_string());
            }
            None
        });
    });
    named.sort();
    named
}

/// Walks `text` and rewrites only its expression segments: the inner text of each `{...}` and
/// `'...'` is handed to `rewrite`, which appends whatever takes its place, and every byte
/// outside a segment — including the delimiters themselves — is copied as written.
///
/// An unterminated segment is a malformed value, and the reader of that value says so with the
/// line and the field named. Rewriting the fragment here would only move the complaint
/// somewhere it makes less sense, so the remainder is copied unchanged.
fn rewrite_segments(text: &str, mut rewrite: impl FnMut(&str, &mut String)) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(open) = rest.find(['{', '\'']) {
        let close_char = if rest.as_bytes()[open] == b'{' {
            '}'
        } else {
            '\''
        };
        out.push_str(&rest[..=open]);
        let after = &rest[open + 1..];
        match after.find(close_char) {
            Some(close) => {
                rewrite(&after[..close], &mut out);
                out.push(close_char);
                rest = &after[close + 1..];
            }
            None => {
                out.push_str(after);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// Rewrites every identifier in one expression segment for which `map` returns a replacement,
/// and copies the rest. An identifier is `[A-Za-z_][A-Za-z0-9_]*`, scanned rather than split
/// on, so `Rv` in `Rvalue` is not a match and `2*Rv` is.
fn map_identifiers(segment: &str, mut map: impl FnMut(&str) -> Option<String>) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut chars = segment.chars().peekable();
    while let Some(character) = chars.next() {
        if !character.is_ascii_alphabetic() && character != '_' {
            out.push(character);
            continue;
        }
        let mut ident = String::from(character);
        while let Some(&next) = chars.peek() {
            if next.is_ascii_alphanumeric() || next == '_' {
                ident.push(next);
                chars.next();
            } else {
                break;
            }
        }
        match map(&ident) {
            Some(replacement) => out.push_str(&replacement),
            None => out.push_str(&ident),
        }
    }
    out
}

/// The text one bound value contributes *inside* an expression segment.
///
/// A number or a bare name goes in as itself, so the overwhelmingly common `{Rv}` with `Rv=100`
/// becomes `{100}` and stays readable as a value. Anything else is parenthesised, because
/// `{Rv*2}` with `Rv=3+1` would otherwise become `{3+1*2}` — a different number, silently.
fn bound_text(value: &str) -> String {
    let inner = value
        .trim()
        .trim_matches(|c| matches!(c, '{' | '}' | '\'' | '"'))
        .trim();
    let atomic = crate::expression::parse_spice_number(inner).is_some()
        || (!inner.is_empty() && inner.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
    if atomic {
        inner.to_string()
    } else {
        format!("({inner})")
    }
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
    substitute: &impl Fn(&str) -> String,
) -> BlockInstance {
    const SIGNAL_FIELDS: [&str; 5] = ["in", "inputs", "ctrl", "clamp_lo_in", "clamp_hi_in"];
    // `kind=phys2sig`'s own reference fields (`node=<electrical node>`, `branch=<element name>` --
    // see `system_builder::build_kind`'s `"phys2sig"` arm) name an *electrical* thing, not a
    // signal, but still need the exact same dotted-path/port/ground resolution as an
    // ElementInstance's own nodes -- a phys2sig declared inside a .subckt body pointing at a
    // purely-internal node (not one of the declared ports) would otherwise keep its unmangled
    // name after flattening and silently fail to resolve (or, worse, accidentally match an
    // unrelated same-named node elsewhere in the flattened circuit). Each is a single value,
    // never comma-separated and never `prev:`-prefixed (that's a signal-domain convention only).
    const ELECTRICAL_REF_FIELDS: [&str; 2] = ["node", "branch"];

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
        } else if ELECTRICAL_REF_FIELDS.contains(&key.as_str()) {
            *value = resolve(value);
        } else {
            // Every other field is a *value*: `freq=100k`, `gain={Kp}`, `kind=pwm`. A formal
            // parameter named in one is resolved here for the same reason an element card's is
            // -- a block inside a parameterised subcircuit is as much part of the circuit as a
            // resistor, and a `{Kp}` it could not resolve would fail as an undeclared symbol.
            *value = substitute(value);
        }
    }
    mangled
}

/// Resolves one `.IC` assignment target the way an element card's nodes and a `phys2sig`'s
/// `node=`/`branch=` fields are resolved: `V(<node>)` and `V(<n1>,<n2>)` name electrical nodes
/// (a declared port becomes the caller's connecting node, an internal node gets the dotted
/// prefix, ground stays ground) and `I(<element>)` names an element, which gets the same
/// dotted prefix its own card got. Without this a `.IC` inside a `.subckt` body kept its
/// unmangled inner name after flattening and failed the builder's "no element connects to"
/// check instead of binding (general-mna#18). Anything that is not of the `<letter>(...)` form
/// is passed through untouched: `MnaBuilder`'s own `.IC` validation rejects it with the
/// line-numbered message it always has, and rewriting it here would only obscure that.
fn mangle_ic_target(target: &str, resolve: &impl Fn(&str) -> String) -> String {
    let trimmed = target.trim();
    let Some((kind, inner)) = trimmed
        .strip_suffix(')')
        .and_then(|without_close| without_close.split_once('('))
    else {
        return target.to_string();
    };
    let resolved = inner
        .split(',')
        .map(|name| resolve(name.trim()))
        .collect::<Vec<_>>()
        .join(",");
    format!("{kind}({resolved})")
}

#[cfg(test)]
mod tests {
    use super::*;
    use general_spice_core::ast::{ElementInstance, Statement};

    /// One bound scope, written the way a call site would leave it.
    fn scope(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_string()))
            .collect()
    }

    #[test]
    fn a_braced_formal_becomes_the_value_the_instance_bound() {
        let params = scope(&[("Rv", "100")]);
        assert_eq!(substitute_params("{Rv}", &params), "{100}");
        assert_eq!(substitute_params("'Rv*2'", &params), "'100*2'");
        assert_eq!(substitute_params("{2*Rv + 1}", &params), "{2*100 + 1}");
    }

    #[test]
    fn an_identifier_is_matched_whole_rather_than_by_prefix() {
        // The reason this is scanned rather than `replace`d: `Rvalue` contains `Rv`, and a
        // textual replace would turn `{Rvalue}` into `{100alue}` -- a value that still parses
        // as a number in some spellings, which is the worst available outcome.
        let params = scope(&[("Rv", "100")]);
        assert_eq!(substitute_params("{Rvalue}", &params), "{Rvalue}");
        assert_eq!(substitute_params("{my_Rv}", &params), "{my_Rv}");
        assert_eq!(substitute_params("{Rv_2}", &params), "{Rv_2}");
    }

    #[test]
    fn nothing_outside_an_expression_segment_is_touched() {
        // A value position holds model names, keywords and device fragments as well as
        // expressions. Substituting a bare token would mean guessing which of those a token
        // is, so only `{...}` and `'...'` are rewritten -- and this is the assertion that says
        // so, not a comment.
        let params = scope(&[("Rv", "100"), ("D", "7")]);
        for written in ["Rv", "100k", "D", "D_IDEAL", "PULSE(0 Rv 0 1n)"] {
            assert_eq!(
                substitute_params(written, &params),
                written,
                "{written} was rewritten outside an expression segment"
            );
        }
        // And the same token *inside* braces is rewritten, so the test above is a statement
        // about the delimiters rather than about the scope being empty.
        assert_eq!(
            substitute_params("PULSE(0 {Rv} 0 1n)", &params),
            "PULSE(0 {100} 0 1n)"
        );
    }

    #[test]
    fn an_empty_scope_returns_every_value_byte_identical() {
        // The true top level has no formals, so every deck that worked before this pass
        // existed must come through it unchanged.
        let empty = HashMap::new();
        for written in ["{Rload}", "'2*Rload'", "1k", "{}", "PWL(0 0 1u 5)"] {
            assert_eq!(substitute_params(written, &empty), written);
        }
    }

    #[test]
    fn a_bound_value_that_is_not_atomic_is_parenthesised() {
        // `{Rv*2}` with `Rv=3+1` is 8, not 5. Without the parentheses the substitution would
        // make it 5 and nothing would say a word.
        assert_eq!(bound_text("100"), "100");
        assert_eq!(bound_text("100k"), "100k");
        assert_eq!(bound_text("-1e-3"), "-1e-3");
        assert_eq!(bound_text("Rload"), "Rload");
        assert_eq!(bound_text("3+1"), "(3+1)");
        assert_eq!(bound_text("{3+1}"), "(3+1)");
        let params = scope(&[("Rv", "3+1")]);
        assert_eq!(substitute_params("{Rv*2}", &params), "{(3+1)*2}");
    }

    #[test]
    fn an_unterminated_segment_is_left_for_the_reader_of_the_value_to_refuse() {
        // Not substituted, and that is the point: an unterminated `{` is a malformed value,
        // and the field's own reader names the line and the field when it refuses it. Guessing
        // where the expression was meant to end would move the complaint somewhere it makes
        // less sense, and `{100` is no more valid than `{Rv`.
        let params = scope(&[("Rv", "100")]);
        assert_eq!(substitute_params("{Rv", &params), "{Rv");
    }

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

    /// An `F`/`H` card names the voltage source whose branch current controls
    /// it. That name is scoped to the body it is written in, so flattening has
    /// to move it along with the element it names.
    ///
    /// This is the shape an ideal transformer takes as a reusable `.subckt` —
    /// a VCVS, a 0 V sense source and a CCCS that references it. Before this,
    /// the reference survived expansion unprefixed and the build failed with
    /// `unknown controlling branch 'VS'`.
    #[test]
    fn a_current_controlled_source_follows_its_controlling_branch_across_a_subckt() {
        let f_card = Statement::ElementInstance(ElementInstance {
            device_letter: 'F',
            name: "F1".to_string(),
            nodes: vec!["p1".to_string(), "p2".to_string()],
            raw_params: vec!["VS".to_string(), "0.25".to_string()],
            subckt_name: None,
            span: 1..2,
        });
        let statements = vec![
            subckt("xfmr", &["p1", "p2"]),
            element('V', "VS", &["p1", "mid"]),
            f_card,
            ends("xfmr"),
            x_call("X1", &["a", "b"], "xfmr"),
        ];
        let flat = flatten(&statements).unwrap();
        let Statement::ElementInstance(source) = &flat[0] else {
            unreachable!()
        };
        let Statement::ElementInstance(cccs) = &flat[1] else {
            unreachable!()
        };
        assert_eq!(source.name, "X1.VS");
        assert_eq!(cccs.name, "X1.F1");
        // The reference must name the source as it is now called, not as it
        // was written inside the body.
        assert_eq!(cccs.raw_params[0], "X1.VS");
        assert_eq!(cccs.raw_params[1], "0.25", "the gain is left alone");
    }

    /// Only `F`/`H` carry an element name in `raw_params`; every other device
    /// puts a value there, and prefixing that would corrupt it.
    #[test]
    fn a_plain_element_keeps_its_value_untouched_across_a_subckt() {
        let statements = vec![
            subckt("div", &["a", "b"]),
            element('R', "R1", &["a", "b"]),
            ends("div"),
            x_call("X1", &["in", "out"], "div"),
        ];
        let flat = flatten(&statements).unwrap();
        let Statement::ElementInstance(r) = &flat[0] else {
            unreachable!()
        };
        assert_eq!(r.name, "X1.R1");
        assert_eq!(r.raw_params[0], "100");
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
                    ("kind".to_string(), "phys2sig".to_string()),
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
                    if bi
                        .fields
                        .iter()
                        .any(|(k, v)| k == "kind" && v == "phys2sig") =>
                {
                    Some(bi)
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(sensor_block.name, "MEASURED");
        // The phys2sig's own `node=vout` field must resolve through the same port binding as an
        // electrical node ("vout" is a declared port, bound to "b" at the call site) -- not be
        // left as the literal, unmangled "vout".
        let node_field = sensor_block
            .fields
            .iter()
            .find(|(k, _)| k == "node")
            .unwrap();
        assert_eq!(node_field.1, "b");
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

    #[test]
    fn a_dot_ic_inside_a_subckt_body_resolves_its_targets_like_element_nodes() {
        // general-mna#18: `.IC` targets get the same port/prefix/ground resolution as an
        // element card's nodes -- a port binds to the caller's node, an internal node and an
        // element name get the `X1.` prefix, ground stays ground, and a malformed target is
        // forwarded untouched for the builder to reject with its usual message.
        let statements = vec![
            subckt("filt", &["in", "out"]),
            element('L', "L1", &["in", "mid"]),
            element('C', "C1", &["mid", "0"]),
            element('R', "R1", &["mid", "out"]),
            Statement::Ic(
                vec![
                    ("V(mid)".to_string(), "5".to_string()),
                    ("V(out)".to_string(), "1".to_string()),
                    ("V(mid, 0)".to_string(), "5".to_string()),
                    ("I(L1)".to_string(), "2".to_string()),
                    ("mid".to_string(), "5".to_string()),
                ],
                4..5,
            ),
            ends("filt"),
            x_call("X1", &["a", "b"], "filt"),
        ];
        let flat = flatten(&statements).unwrap();
        let Some(Statement::Ic(assignments, span)) = flat.last() else {
            panic!("expected the .IC to be forwarded last");
        };
        assert_eq!(span, &(4..5));
        let targets: Vec<&str> = assignments.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            targets,
            vec!["V(X1.mid)", "V(b)", "V(X1.mid,0)", "I(X1.L1)", "mid"]
        );
        let values: Vec<&str> = assignments.iter().map(|(_, v)| v.as_str()).collect();
        assert_eq!(values, vec!["5", "1", "5", "2", "5"]);
    }

    #[test]
    fn a_phys2sig_inside_a_subckt_body_resolves_its_node_field_to_the_dotted_path_name() {
        // Regression: a phys2sig's `node=`/`branch=` fields reference an *electrical* thing, not a
        // signal -- they must get the same dotted-path mangling as an ElementInstance's own
        // nodes when the node is purely internal (not a declared port), or the phys2sig
        // silently keeps referencing the unmangled, undeclared name after flattening.
        use general_spice_core::ast::BlockInstance;

        let statements = vec![
            subckt("leg", &["vin", "vout"]),
            element('R', "R1", &["vin", "mid"]),
            element('R', "R2", &["mid", "vout"]),
            Statement::BlockInstance(BlockInstance {
                name: "VMID".to_string(),
                fields: vec![
                    ("kind".to_string(), "phys2sig".to_string()),
                    ("node".to_string(), "mid".to_string()),
                ],
                span: 1..2,
            }),
            Statement::BlockInstance(BlockInstance {
                name: "IR1".to_string(),
                fields: vec![
                    ("kind".to_string(), "phys2sig".to_string()),
                    ("branch".to_string(), "R1".to_string()),
                ],
                span: 1..2,
            }),
            ends("leg"),
            x_call("X1", &["a", "b"], "leg"),
        ];
        let flat = flatten(&statements).unwrap();
        let vmid = flat
            .iter()
            .find_map(|s| match s {
                Statement::BlockInstance(bi) if bi.name == "X1.VMID" => Some(bi),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            vmid.fields.iter().find(|(k, _)| k == "node").unwrap().1,
            "X1.mid"
        );
        let ir1 = flat
            .iter()
            .find_map(|s| match s {
                Statement::BlockInstance(bi) if bi.name == "X1.IR1" => Some(bi),
                _ => None,
            })
            .unwrap();
        assert_eq!(
            ir1.fields.iter().find(|(k, _)| k == "branch").unwrap().1,
            "X1.R1"
        );
    }
}
