//! Builds a unified [`System`] (electrical MNA matrices + block/signal-domain graph) from one
//! source text, via `general-spice-core`'s grammar — both the electrical `ElementInstance`
//! lines (fed to the existing [`crate::MnaBuilder`], unchanged) and the new first-class
//! `BlockInstance` lines (see `general-spice-core`'s own `docs/GRAMMAR.md` §12). Replaces the
//! hand-rolled ~700-line parser that used to live in `general-simulator-cli`'s `main.rs`, which
//! worked directly off `*`-disguised comment lines instead of real grammar.

use std::collections::BTreeMap;

use continuous_blocks::{
    CoordinateTransform, DiscreteIntegrationMethod, DiscretePid, FlipFlopKind, Hysteresis,
    LatchPriority, LogicOp, Pid, StateSpace, TransferFunction, Vco,
};
use general_spice_core::ast::Statement;
use general_spice_core::dialect::Dialect;
use general_spice_core::{lexer, parser};
use pwl_devices::{IdealDiode, IdealSwitch};

use crate::block_graph::{
    BlockInstance, BlockKind, ConstValue, GainValue, GateBinding, Phys2SigTarget, PhysicalDomain,
    PidClamp, SampleTimeSpec, Signal,
};
use crate::hierarchy;
use crate::{MnaBuilder, MnaSystem, TransientFunction};

/// The result of [`build_system`]: everything a simulator needs, already split into its own
/// electrical (`mna`/`ideal_diodes`/`ideal_switches`/`gates`) and signal-domain (`blocks`)
/// halves.
#[derive(Debug)]
pub struct System {
    /// The electrical MNA system, built by [`crate::MnaBuilder`] from the netlist's element lines.
    pub mna: MnaSystem,
    /// Every `D` instance, keyed by name, as a companion ideal-diode model.
    pub ideal_diodes: BTreeMap<String, IdealDiode>,
    /// Every ideal-switch instance, keyed by name.
    pub ideal_switches: BTreeMap<String, IdealSwitch>,
    /// Each ideal switch's resolved gate-drive signal, keyed by the same name as
    /// `ideal_switches`.
    pub gates: BTreeMap<String, GateBinding>,
    /// Every block/signal-domain (`kind=...`) instance, in source declaration order.
    pub blocks: Vec<BlockInstance>,
    /// Every ideal switch's shared on-resistance (`dae-runtime`'s switch model uses one shared
    /// value per call) — `0.0` if there are no ideal switches at all. `build_system` already
    /// enforces every declared ideal switch agrees on this value.
    pub shared_r_on: f64,
}

enum Kind {
    IdealDiode(IdealDiode),
    IdealSwitch {
        ideal_switch: IdealSwitch,
        r_on: f64,
        gate: GateBinding,
    },
    Block(BlockInstance),
}

/// Parses `source` under `dialect` and flattens every `.subckt`/`X`-instance into one flat,
/// dotted-path-named statement list (see `hierarchy::flatten`) — the shared first step behind
/// [`build_system`] and behind any downstream consumer (e.g. `dae-runtime`) that needs to build
/// its own `MnaSystem` from the same, already-hierarchy-resolved statements rather than
/// re-parsing raw text (and silently losing hierarchy) itself.
///
/// **`source` is parsed as a fragment, not a full document** (`general_spice_core::parser::parse`,
/// not `parse_document`) — unlike the `MnaBuilder::build_document`/`build_fragment` distinction
/// in this same crate's own README, there is no mandatory title line here: the *first* line of
/// `source` is a real element/block statement, not skipped. A `general-simulator` netlist that
/// leads with a title line (the normal SPICE-file convention) will have that line parsed as an
/// element/block declaration and almost always fail — verified directly: an otherwise-correct
/// netlist with a leading `"<title>"` text line failed with `UnsupportedElement`, naming a
/// device letter taken from the title's own first word, not from anything the netlist author
/// wrote as a device.
pub fn parse_and_flatten(source: &str, dialect: Dialect) -> Result<Vec<Statement>, String> {
    let processed = lexer::preprocess(source, dialect);
    let statements: Vec<Statement> = parser::parse(&processed, dialect)
        .into_iter()
        .collect::<Result<_, _>>()
        .map_err(|e: parser::ParseError| format!("line {}: {}", e.span.start, e.message))?;
    hierarchy::flatten(&statements)
}

/// Parses `source` under `dialect` and builds the complete [`System`] — the one entry point a
/// simulator needs; no parsing code of its own required downstream. Electrical statements go
/// through the existing [`MnaBuilder`] machinery unchanged; block/signal-domain statements
/// (`Statement::BlockInstance`) are dispatched by `build_kind` into an `ideal_diode`/
/// `ideal_switch`/block entry, keyed by the statement's own name.
///
/// **No mandatory title line** — see [`parse_and_flatten`]'s own doc comment, which this
/// function calls first.
pub fn build_system(source: &str, dialect: Dialect) -> Result<System, String> {
    let statements = parse_and_flatten(source, dialect)?;

    let mna = MnaBuilder::new(dialect)
        .build_statements(&statements)
        .map_err(|e| format!("{e:?}"))?;

    let mut ideal_diodes = BTreeMap::new();
    let mut ideal_switches = BTreeMap::new();
    let mut gates = BTreeMap::new();
    let mut blocks = Vec::new();
    let mut shared_r_on: Option<f64> = None;
    // (phys2sig name, branch=<name>, 1-based line number) for every `kind=phys2sig
    // branch=<name>` block -- checked below, once `mna.unknowns` (and every element's own name)
    // is fully known, against the classic MNA footgun: `branch=` naming an element with no
    // current unknown at all (only V/L/E/H sources get one -- see
    // `MnaBuilder::is_branch_device`), which used to silently read back 0.0 forever instead of
    // erroring.
    let mut phys2sig_branch_checks: Vec<(String, String, usize)> = Vec::new();

    for stmt in &statements {
        let Statement::BlockInstance(bi) = stmt else {
            continue;
        };
        match build_kind(bi)? {
            Kind::IdealDiode(d) => {
                ideal_diodes.insert(bi.name.clone(), d);
            }
            Kind::IdealSwitch {
                ideal_switch,
                r_on,
                gate,
            } => {
                match shared_r_on {
                    None => shared_r_on = Some(r_on),
                    Some(existing) if (existing - r_on).abs() > 1e-15 => {
                        return Err(format!(
                            "all ideal switches must share the same r_on (dae-runtime's switch \
                             model uses one shared on-resistance per call); got {existing} and \
                             {r_on}"
                        ));
                    }
                    Some(_) => {}
                }
                gates.insert(bi.name.clone(), gate);
                ideal_switches.insert(bi.name.clone(), ideal_switch);
            }
            Kind::Block(instance) => {
                if let BlockKind::Phys2Sig(Phys2SigTarget::Current(branch)) = &instance.kind {
                    let line_number = bi.span.start.saturating_sub(1);
                    phys2sig_branch_checks.push((
                        instance.name.clone(),
                        branch.clone(),
                        line_number,
                    ));
                }
                blocks.push(instance);
            }
        }
    }

    for (phys2sig_name, branch, line_number) in &phys2sig_branch_checks {
        let unknown = format!("I({branch})");
        if mna.unknowns.iter().any(|u| u == &unknown) {
            continue;
        }
        let element_exists = statements.iter().any(|s| {
            matches!(
                s,
                Statement::ElementInstance(e) if e.name.eq_ignore_ascii_case(branch)
            )
        });
        if element_exists {
            return Err(format!(
                "line {}: device '{phys2sig_name}' kind='phys2sig' branch='{branch}' has no \
                 current unknown available (only V/L/E/H sources carry a branch-current \
                 unknown) -- insert a 0V voltage source in series in the branch you want to \
                 measure and probe its current instead",
                line_number + 1
            ));
        }
        return Err(format!(
            "line {}: device '{phys2sig_name}' kind='phys2sig' branch='{branch}' names no such \
             element",
            line_number + 1
        ));
    }

    Ok(System {
        mna,
        ideal_diodes,
        ideal_switches,
        gates,
        blocks,
        shared_r_on: shared_r_on.unwrap_or(0.0),
    })
}

fn parse_signal(text: &str) -> Signal {
    match text.strip_prefix("prev:") {
        Some(name) => Signal::BlockPrev(name.to_string()),
        None => Signal::Block(text.to_string()),
    }
}

/// Splits `text` on top-level commas only — one inside a nested `[...]` (bracket depth > 0)
/// doesn't count — so `"[1,2],[3,4]"` splits into `["[1,2]", "[3,4]"]`, not four pieces. The
/// one primitive every Python-list-literal field below (`num=`/`den=`/`a=`/`b=`/`c=`/`points=`)
/// is built from, so a matrix's row separator and a vector's entry separator are the same
/// operation applied at a different nesting depth, not two different parsers.
fn split_top_level(text: &str) -> Vec<&str> {
    let mut depth = 0i32;
    let mut start = 0usize;
    let mut parts = Vec::new();
    for (i, c) in text.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(text[start..i].trim());
                start = i + c.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(text[start..].trim());
    parts
}

/// Strips a field's required outer `[...]` — every list-valued field here is a real Python list
/// literal (`num=[1,2,3]`, `a=[[1,2],[3,4]]`), never the bare comma/semicolon/colon delimiters
/// an earlier version of this grammar used, so a Python list built from the same numbers can be
/// pasted into a `.cir` file (spaces after commas included: [`split_top_level`]/the `f64` parse
/// below both trim) without reformatting.
fn strip_brackets<'a>(
    text: &'a str,
    name: &str,
    field: &str,
    line_number: usize,
) -> Result<&'a str, String> {
    let trimmed = text.trim();
    trimmed
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| {
            format!(
                "line {}: device '{name}' field '{field}' must be a Python-style list, e.g. \
                 '[1,2,3]' or '[[1,2],[3,4]]' (got '{text}')",
                line_number + 1
            )
        })
}

/// Parses a `[[<x>,<y>],[<x>,<y>],...]` point list (a `kind=pwc`/`kind=pwl` reference schedule,
/// or a `kind=table` lookup table) — a Python list of 2-element `[x, y]` lists, exactly the
/// shape `numpy.array(points)` would expect for an `Nx2` table. Sorted ascending by `x` on
/// return.
fn parse_xy_points(
    text: &str,
    name: &str,
    field: &str,
    line_number: usize,
) -> Result<Vec<(f64, f64)>, String> {
    let rows = parse_matrix_rows(text, name, field, line_number)?;
    let mut points = Vec::with_capacity(rows.len());
    for row in rows {
        let [x, y] = row.as_slice() else {
            return Err(format!(
                "line {}: device '{name}' field '{field}': each entry must be a 2-element \
                 '[x,y]' list (got '{row:?}')",
                line_number + 1
            ));
        };
        points.push((*x, *y));
    }
    points.sort_by(|a, b| a.0.total_cmp(&b.0));
    Ok(points)
}

/// Parses the `ts=`/`freq=`/`to=` fields shared by
/// `kind=cscript`/`kind=pyblock`/`kind=pyfunc`/`kind=octfunc` into a [`SampleTimeSpec`] —
/// `ts=variable` (only when `allow_variable`, i.e. never for `pyfunc`/`octfunc`, both
/// stateless) selects [`SampleTimeSpec::Variable`]; a numeric `ts=`/`freq=` selects
/// [`SampleTimeSpec::Periodic`], with `to=` giving its own explicit `offset` (defaulting to
/// `0.0` when omitted — see [`SampleTimeSpec`]'s own doc comment for why `0.0`, not `period`, is
/// what actually preserves this project's pre-existing behavior for every netlist that never
/// used `to=` at all). `to=`, when given, must satisfy `0.0 <= to < period`. `kind_str` (the
/// netlist's own `kind=` token, e.g. `"pyfunc"`/`"octfunc"`) names the specific block kind in
/// the `ts=variable`-rejected error message when `!allow_variable`; unused otherwise.
fn parse_sample_time(
    fields: &BTreeMap<String, String>,
    name: &str,
    line_number: usize,
    allow_variable: bool,
    kind_str: &str,
) -> Result<Option<SampleTimeSpec>, String> {
    let get = |key: &str| -> Result<f64, String> {
        fields
            .get(key)
            .ok_or_else(|| {
                format!(
                    "line {}: device '{name}' missing field '{key}'",
                    line_number + 1
                )
            })?
            .parse::<f64>()
            .map_err(|_| {
                format!(
                    "line {}: device '{name}' field '{key}' is not a number",
                    line_number + 1
                )
            })
    };

    if fields.get("ts").map(String::as_str) == Some("variable") {
        if !allow_variable {
            // pyfunc keeps its own pre-existing exact wording (including the "use kind=pyblock
            // instead" suggestion -- a real, stateful alternative exists for it); octfunc has no
            // stateful counterpart in this workspace to suggest switching to, so its own message
            // simply omits that clause rather than suggesting something that doesn't exist.
            let suggestion = if kind_str == "pyfunc" {
                " -- use kind=pyblock instead"
            } else {
                ""
            };
            return Err(format!(
                "line {}: device '{name}': ts=variable is not available for kind={kind_str} (a \
                 stateless function call has no instance to remember a requested next-hit \
                 time against){suggestion}",
                line_number + 1
            ));
        }
        if fields.contains_key("freq") || fields.contains_key("to") {
            return Err(format!(
                "line {}: device '{name}': ts=variable is exclusive with 'freq'/'to' (the \
                 block computes its own schedule; there's no fixed period/offset to combine \
                 it with)",
                line_number + 1
            ));
        }
        return Ok(Some(SampleTimeSpec::Variable));
    }

    let period = match (fields.get("ts"), fields.get("freq")) {
        (Some(_), Some(_)) => {
            return Err(format!(
                "line {}: device '{name}': 'ts' and 'freq' are mutually exclusive (both set \
                 this block's sample time)",
                line_number + 1
            ))
        }
        (Some(_), None) => Some(get("ts")?),
        (None, Some(_)) => Some(1.0 / get("freq")?),
        (None, None) => None,
    };

    match (period, fields.get("to")) {
        (None, Some(_)) => Err(format!(
            "line {}: device '{name}': 'to' requires 'ts' or 'freq' (a phase offset has \
             nothing to offset without a period)",
            line_number + 1
        )),
        (None, None) => Ok(None),
        (Some(period), None) => Ok(Some(SampleTimeSpec::Periodic {
            period,
            offset: 0.0,
        })),
        (Some(period), Some(_)) => {
            let offset = get("to")?;
            if !(0.0..period).contains(&offset) {
                return Err(format!(
                    "line {}: device '{name}': 'to' ({offset}) must satisfy 0 <= to < ts \
                     ({period})",
                    line_number + 1
                ));
            }
            Ok(Some(SampleTimeSpec::Periodic { period, offset }))
        }
    }
}

/// Parses the same `ts=`/`freq=`/`to=` fields as [`parse_sample_time`], but for the three
/// discrete-time blocks (`kind=discretestatespace`/`discretetf`/`discretepid`) where a sample
/// time isn't optional the way it is for `cscript`/`pyblock`/`pyfunc` -- a discrete system's own
/// dynamics *are* its sample period, so there's no "continuous" fallback for an omitted `ts=`/
/// `freq=` to mean, and `ts=variable` doesn't compose with a fixed-period recursion at all (its
/// own coefficients are baked in at that one period). See `discrete-time-blocks.md`.
fn parse_required_periodic_sample_time(
    fields: &BTreeMap<String, String>,
    name: &str,
    line_number: usize,
) -> Result<SampleTimeSpec, String> {
    match parse_sample_time(fields, name, line_number, true, "")? {
        Some(SampleTimeSpec::Periodic { period, offset }) => {
            Ok(SampleTimeSpec::Periodic { period, offset })
        }
        Some(SampleTimeSpec::Variable) => Err(format!(
            "line {}: device '{name}': ts=variable is not available here -- a discrete block's \
             own recursion has its coefficients baked in at one fixed sample period, which \
             'the block decides its own next execution time' doesn't compose with",
            line_number + 1
        )),
        None => Err(format!(
            "line {}: device '{name}': missing 'ts=' or 'freq=' -- a discrete-time block's own \
             sample period is not optional (its dynamics *are* that period), unlike the \
             optional 'ts='/'freq=' on kind=cscript/pyblock/pyfunc",
            line_number + 1
        )),
    }
}

/// Parses a `kind=const` block's `value=` field: a bare scalar (`value=5`, unchanged from
/// before vector signals existed) or a Python-style flat list (`value=[1,2,3]`), a fixed
/// constant vector.
fn parse_const_value(
    text: &str,
    name: &str,
    field: &str,
    line_number: usize,
) -> Result<ConstValue, String> {
    if text.trim().starts_with('[') {
        Ok(ConstValue::Vector(parse_vector(
            text,
            name,
            field,
            line_number,
        )?))
    } else {
        text.trim()
            .parse::<f64>()
            .map(ConstValue::Scalar)
            .map_err(|_| {
                format!(
                    "line {}: device '{name}' field '{field}' is not a number",
                    line_number + 1
                )
            })
    }
}

/// Parses a `kind=gain` block's `k=` field: a bare scalar (`k=2.0`, unchanged from before
/// vector signals existed) or a Python-style *nested* list (`k=[[1,0],[0,1]]`), a fixed `M x N`
/// matrix for a genuine matrix-vector product. A flat list (`k=[1,2,3]`) is not a valid `Gain`
/// shape at all -- unlike `Const`, `Gain`'s own value is never itself a vector -- so it's
/// rejected here with a clear error rather than silently guessed at.
fn parse_gain_value(
    text: &str,
    name: &str,
    field: &str,
    line_number: usize,
) -> Result<GainValue, String> {
    let trimmed = text.trim();
    if trimmed.starts_with('[') {
        let inner = strip_brackets(text, name, field, line_number)?;
        if inner.trim_start().starts_with('[') {
            return Ok(GainValue::Matrix(parse_matrix_rows(
                text,
                name,
                field,
                line_number,
            )?));
        }
        return Err(format!(
            "line {}: device '{name}' field '{field}' must be a scalar (e.g. '2.0') or a \
             matrix (e.g. '[[1,0],[0,1]]') -- got a flat list '{text}', which is not a valid \
             Gain shape",
            line_number + 1
        ));
    }
    trimmed.parse::<f64>().map(GainValue::Scalar).map_err(|_| {
        format!(
            "line {}: device '{name}' field '{field}' is not a number",
            line_number + 1
        )
    })
}

/// Parses a `kind=statespace` block's `b=`/`c=` field: a flat list (`b=[1,0]`, `c=[1,0]` — the
/// original SISO shorthand, unchanged) or a genuine matrix (`b=[[1,0],[0,1]]`,
/// `c=[[1,0],[0,1]]`, for real MIMO). `as_column`: `true` wraps a flat-list result as an `n x 1`
/// column matrix (`b`'s own shorthand meaning "exactly one input"); `false` wraps it as a `1 x
/// n` row matrix (`c`'s own shorthand meaning "exactly one output").
fn parse_matrix_or_vector_shorthand(
    text: &str,
    name: &str,
    field: &str,
    line_number: usize,
    as_column: bool,
) -> Result<Vec<Vec<f64>>, String> {
    let inner = strip_brackets(text, name, field, line_number)?;
    if inner.trim_start().starts_with('[') {
        return parse_matrix_rows(text, name, field, line_number);
    }
    let v = parse_vector(text, name, field, line_number)?;
    Ok(if as_column {
        v.into_iter().map(|x| vec![x]).collect()
    } else {
        vec![v]
    })
}

/// Parses a Python-style list of numbers, e.g. a `kind=statespace` block's `b=[1,2]`/`c=[1,0]`
/// vector or a `kind=tf` block's `num=[1,2]`/`den=[1,3,2]` coefficients.
fn parse_vector(
    text: &str,
    name: &str,
    field: &str,
    line_number: usize,
) -> Result<Vec<f64>, String> {
    let inner = strip_brackets(text, name, field, line_number)?;
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)
        .into_iter()
        .map(|v| {
            let parsed: f64 = v.parse().map_err(|_| {
                format!(
                    "line {}: device '{name}' field '{field}' entry '{v}' is not a number",
                    line_number + 1
                )
            })?;
            // `f64::from_str` accepts "nan"/"inf"/"-inf" as valid floats, but a NaN or
            // infinity here would otherwise silently reach a downstream `partial_cmp().unwrap()`
            // (this file's own `points` sort, or `lcp-solver`'s pivot selection) and panic the
            // whole process instead of failing this one netlist line cleanly.
            if !parsed.is_finite() {
                return Err(format!(
                    "line {}: device '{name}' field '{field}' entry '{v}' must be a finite \
                     number (got {parsed})",
                    line_number + 1
                ));
            }
            Ok(parsed)
        })
        .collect()
}

/// Parses a `kind=statespace` block's `a` matrix: a Python-style list of row lists, e.g.
/// `a=[[1,2],[3,4]]` — exactly `numpy.array([[1,2],[3,4]])`'s own literal shape.
fn parse_matrix_rows(
    text: &str,
    name: &str,
    field: &str,
    line_number: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let inner = strip_brackets(text, name, field, line_number)?;
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    split_top_level(inner)
        .into_iter()
        .map(|row| parse_vector(row, name, field, line_number))
        .collect()
}

fn build_kind(stmt: &general_spice_core::ast::BlockInstance) -> Result<Kind, String> {
    let name = &stmt.name;
    let line_number = stmt.span.start.saturating_sub(1);
    let fields: BTreeMap<String, String> = stmt.fields.iter().cloned().collect();

    let get = |key: &str| -> Result<f64, String> {
        fields
            .get(key)
            .ok_or_else(|| {
                format!(
                    "line {}: device '{name}' missing field '{key}'",
                    line_number + 1
                )
            })?
            .parse::<f64>()
            .map_err(|_| {
                format!(
                    "line {}: device '{name}' field '{key}' is not a number",
                    line_number + 1
                )
            })
    };

    let get_str = |key: &str| -> Result<String, String> {
        fields.get(key).cloned().ok_or_else(|| {
            format!(
                "line {}: device '{name}' missing field '{key}'",
                line_number + 1
            )
        })
    };

    let kind = fields
        .get("kind")
        .map(String::as_str)
        .unwrap_or("ideal_diode");
    let entry = match kind {
        "ideal_diode" => Kind::IdealDiode(IdealDiode::new(
            get("g_breakdown")?,
            get("v_breakdown")?,
            get("g_off")?,
            get("v_th")?,
            get("g_on")?,
        )),
        "ideal_switch" => {
            let body_diode = IdealDiode::new(
                get("g_breakdown")?,
                get("v_breakdown")?,
                get("g_off")?,
                get("v_th")?,
                get("g_on")?,
            );
            let r_on = get("r_on")?;
            // Every gate is block-driven -- gate=block ctrl=<name>, reading that block's
            // current output (>= 0.5 means on). No other gate= spelling exists: a
            // permanently-off gate is an explicit `Const(0.0)` wired through
            // `kind=sig2phys domain=voltage` (an ideal switch's gate is itself a voltage, the
            // same Signal-to-PS boundary a V-source's own magnitude uses -- no dedicated
            // gate-only converter).
            let gate = match fields.get("gate").map(String::as_str) {
                Some("block") => GateBinding::Block(get_str("ctrl")?),
                Some(other) => {
                    return Err(format!(
                        "line {}: unknown gate spec '{other}' (only gate=block ctrl=<name> \
                             exists -- every gate is block-driven)",
                        line_number + 1
                    ))
                }
                None => {
                    return Err(format!(
                        "line {}: device '{name}' missing field 'gate' (gate=block \
                             ctrl=<name> -- every gate is block-driven, see this file's own \
                             module doc comment)",
                        line_number + 1
                    ))
                }
            };
            Kind::IdealSwitch {
                ideal_switch: IdealSwitch::new(r_on, body_diode),
                r_on,
                gate,
            }
        }
        "const" => Kind::Block(BlockInstance {
            name: name.to_string(),
            kind: BlockKind::Const(parse_const_value(
                &get_str("value")?,
                name,
                "value",
                line_number,
            )?),
            inputs: Vec::new(),
        }),
        "time" => Kind::Block(BlockInstance {
            name: name.to_string(),
            kind: BlockKind::Time,
            inputs: Vec::new(),
        }),
        // "repeat=true" is optional on both "pwc" and "pwl" (default false, unchanged
        // hold-flat-past-the-end behavior) -- wraps time into the breakpoint list's own
        // [first, last) span once past the last point, making it periodic. See
        // `dae_runtime::block_graph`'s own doc comment on `BlockKind::Pwc`/`BlockKind::Pwl`
        // for why these are two distinct block kinds (interpolation style) rather than one
        // with a flag, and why "pwl" here means real piecewise-*linear* SPICE PWL semantics
        // while the older piecewise-*constant* block was renamed to "pwc" to free that name
        // up.
        "pwc" => {
            let points = parse_xy_points(&get_str("points")?, name, "points", line_number)?;
            let repeat = fields.get("repeat").map(|s| s == "true").unwrap_or(false);
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Pwc { points, repeat },
                inputs: Vec::new(),
            })
        }
        "pwl" => {
            let points = parse_xy_points(&get_str("points")?, name, "points", line_number)?;
            let repeat = fields.get("repeat").map(|s| s == "true").unwrap_or(false);
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Pwl { points, repeat },
                inputs: Vec::new(),
            })
        }
        // "sinwave"/"pulsewave"/"expwave"/"sffmwave": the electrical domain's other four
        // time-varying source forms (see `general_mna::TransientFunction`'s own doc comment
        // for the exact formula each implements), reused directly rather than
        // reimplemented, with the same field names/order/defaults as ngspice/Xyce's own
        // SIN()/PULSE()/EXP()/SFFM() -- so a `kind=sinwave ...` reference schedule and a
        // `V1 a 0 SIN(...)` source built from the same numbers are bit-for-bit the same
        // waveform. Named "...wave" rather than the bare SPICE keyword specifically to
        // avoid colliding with the pre-existing `kind=sin`/`kind=exp` waveform-arithmetic
        // *functions* (`sin(x)`/`exp(x)` of an input signal, see the `MathFn1` fallback
        // dispatch below) -- "pulse"/"sffm" have no such collision today, but are named the
        // same way for consistency across the family rather than only where forced to.
        "sinwave" => {
            let get_opt = |key: &str, default: f64| -> f64 {
                fields
                    .get(key)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(default)
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Waveform(TransientFunction::Sin {
                    v0: get_opt("v0", 0.0),
                    va: get("va")?,
                    freq: get("freq")?,
                    td: get_opt("td", 0.0),
                    theta: get_opt("theta", 0.0),
                    phase: get_opt("phase", 0.0),
                }),
                inputs: Vec::new(),
            })
        }
        "pulsewave" => {
            let get_opt = |key: &str, default: f64| -> f64 {
                fields
                    .get(key)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(default)
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Waveform(TransientFunction::Pulse {
                    v1: get("v1")?,
                    v2: get("v2")?,
                    td: get_opt("td", 0.0),
                    tr: get_opt("tr", 0.0),
                    tf: get_opt("tf", 0.0),
                    pw: get_opt("pw", f64::MAX / 4.0),
                    per: get_opt("per", f64::MAX / 4.0),
                }),
                inputs: Vec::new(),
            })
        }
        "expwave" => {
            let get_opt = |key: &str, default: f64| -> f64 {
                fields
                    .get(key)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(default)
            };
            let td1 = get_opt("td1", 0.0);
            // td2 defaults to "effectively never" (matching pulsewave's own pw/per
            // defaults just above), NOT td1 -- TransientFunction::Exp treats `t < td2` as
            // "still in the rise phase," so a naive td1 default would make every omitted-
            // td2 call fall straight into the *fall* phase at t=0 instead of never falling
            // at all (a real bug caught by this file's own dae-runtime-level test).
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Waveform(TransientFunction::Exp {
                    v1: get("v1")?,
                    v2: get("v2")?,
                    td1,
                    tau1: get_opt("tau1", 1.0),
                    td2: get_opt("td2", f64::MAX / 4.0),
                    tau2: get_opt("tau2", 1.0),
                }),
                inputs: Vec::new(),
            })
        }
        "sffmwave" => {
            let get_opt = |key: &str, default: f64| -> f64 {
                fields
                    .get(key)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(default)
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Waveform(TransientFunction::Sffm {
                    v0: get_opt("v0", 0.0),
                    va: get("va")?,
                    fc: get("fc")?,
                    mdi: get_opt("mdi", 0.0),
                    fs: get("fs")?,
                }),
                inputs: Vec::new(),
            })
        }
        "sum" => {
            let inputs: Vec<Signal> = get_str("inputs")?.split(',').map(parse_signal).collect();
            let signs: Vec<f64> = get_str("signs")?
                .split(',')
                .map(|s| {
                    s.parse::<f64>().map_err(|_| {
                        format!(
                            "line {}: device '{name}' field 'signs' entry '{s}' is not a \
                                 number",
                            line_number + 1
                        )
                    })
                })
                .collect::<Result<_, _>>()?;
            if inputs.len() != signs.len() {
                return Err(format!(
                    "line {}: device '{name}': 'inputs' has {} entries but 'signs' has {} \
                         (need one sign per input)",
                    line_number + 1,
                    inputs.len(),
                    signs.len()
                ));
            }
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Sum(signs),
                inputs,
            })
        }
        "gain" => Kind::Block(BlockInstance {
            name: name.to_string(),
            kind: BlockKind::Gain(parse_gain_value(&get_str("k")?, name, "k", line_number)?),
            inputs: vec![parse_signal(&get_str("in")?)],
        }),
        "pid" => {
            let pid = Pid::new(get("kp")?, get("ki")?, get("kd")?, get("n")?).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid PID ({e:?})",
                    line_number + 1
                )
            })?;
            let error_input = parse_signal(&get_str("in")?);
            let (clamp, inputs) = match (fields.get("clamp_lo_in"), fields.get("clamp_hi_in")) {
                (Some(lo), Some(hi)) => (
                    PidClamp::Dynamic,
                    vec![error_input, parse_signal(lo), parse_signal(hi)],
                ),
                (None, None) => (
                    PidClamp::Fixed(get("clamp_lo")?, get("clamp_hi")?),
                    vec![error_input],
                ),
                _ => {
                    return Err(format!(
                        "line {}: device '{name}': 'clamp_lo_in'/'clamp_hi_in' must both be \
                             given together (dynamic clamp) or both omitted (fixed clamp= \
                             clamp_lo/clamp_hi)",
                        line_number + 1
                    ))
                }
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Pid { pid, clamp },
                inputs,
            })
        }
        "vco" => {
            let vco = Vco::new(get("f_min")?, get("f_max")?).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid vco ({e:?})",
                    line_number + 1
                )
            })?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Vco(vco),
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        // "pwm"/"pspwm": fixed-frequency and frequency+phase+duty active-high-complementary
        // PWM modulators (see `dae_runtime::BlockKind::Pwm`/`PhaseShiftPwm`'s own doc
        // comments for the full design) -- both default `red`/`fed` (dead time, seconds) to
        // 0.0, the ideal gap-free/overlap-free complementary pair.
        "pwm" => {
            let output_names = match fields.get("outputs") {
                Some(names) => {
                    let names: Vec<String> = names.split(',').map(str::to_string).collect();
                    if names.len() != 2 {
                        return Err(format!(
                            "line {}: device '{name}': 'outputs' needs exactly 2 entries \
                                 (main, complement; got {})",
                            line_number + 1,
                            names.len()
                        ));
                    }
                    names
                }
                None => vec![name.to_string(), format!("{name}_comp")],
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Pwm {
                    freq_hz: get("freq")?,
                    red: fields
                        .get("red")
                        .map(|s| s.parse::<f64>())
                        .transpose()
                        .map_err(|_| {
                            format!(
                                "line {}: device '{name}' field 'red' is not a number",
                                line_number + 1
                            )
                        })?
                        .unwrap_or(0.0),
                    fed: fields
                        .get("fed")
                        .map(|s| s.parse::<f64>())
                        .transpose()
                        .map_err(|_| {
                            format!(
                                "line {}: device '{name}' field 'fed' is not a number",
                                line_number + 1
                            )
                        })?
                        .unwrap_or(0.0),
                    output_names,
                },
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        "pspwm" => {
            let osc = Vco::new(get("f_min")?, get("f_max")?).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid pspwm oscillator ({e:?})",
                    line_number + 1
                )
            })?;
            let inputs: Vec<Signal> = get_str("inputs")?.split(',').map(parse_signal).collect();
            if inputs.len() != 3 {
                return Err(format!(
                    "line {}: device '{name}' kind='pspwm' needs 3 inputs \
                         (freq,phase,duty; got {})",
                    line_number + 1,
                    inputs.len()
                ));
            }
            let output_names = match fields.get("outputs") {
                Some(names) => {
                    let names: Vec<String> = names.split(',').map(str::to_string).collect();
                    if names.len() != 2 {
                        return Err(format!(
                            "line {}: device '{name}': 'outputs' needs exactly 2 entries \
                                 (main, complement; got {})",
                            line_number + 1,
                            names.len()
                        ));
                    }
                    names
                }
                None => vec![name.to_string(), format!("{name}_comp")],
            };
            let get_opt = |key: &str| -> Result<f64, String> {
                fields
                    .get(key)
                    .map(|s| s.parse::<f64>())
                    .transpose()
                    .map_err(|_| {
                        format!(
                            "line {}: device '{name}' field '{key}' is not a number",
                            line_number + 1
                        )
                    })
                    .map(|v| v.unwrap_or(0.0))
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::PhaseShiftPwm {
                    osc,
                    red: get_opt("red")?,
                    fed: get_opt("fed")?,
                    output_names,
                },
                inputs,
            })
        }
        "hysteresis" => {
            let hysteresis = Hysteresis::new(get("high")?, get("low")?).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid hysteresis ({e:?})",
                    line_number + 1
                )
            })?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Hysteresis(hysteresis),
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        "srlatch" => {
            let priority = match fields.get("priority") {
                Some(s) => LatchPriority::from_name(s).ok_or_else(|| {
                    format!(
                        "line {}: device '{name}' field 'priority' must be 'set' or 'reset' \
                         (got '{s}')",
                        line_number + 1
                    )
                })?,
                None => LatchPriority::Set,
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::SrLatch { priority },
                inputs: vec![
                    parse_signal(&get_str("set")?),
                    parse_signal(&get_str("reset")?),
                ],
            })
        }
        "counter" => {
            let modulus = match fields.get("modulus") {
                Some(s) => Some(s.parse::<u32>().map_err(|_| {
                    format!(
                        "line {}: device '{name}' field 'modulus' is not a non-negative integer",
                        line_number + 1
                    )
                })?),
                None => None,
            };
            let mut inputs = vec![parse_signal(&get_str("clk")?)];
            let up_down = fields.contains_key("up_down");
            if up_down {
                inputs.push(parse_signal(&get_str("up_down")?));
            }
            let reset = fields.contains_key("reset");
            if reset {
                inputs.push(parse_signal(&get_str("reset")?));
            }
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Counter {
                    up_down,
                    modulus,
                    reset,
                },
                inputs,
            })
        }
        "cscript" => {
            let lib = std::path::PathBuf::from(get_str("lib")?);
            let output_names = match fields.get("outputs") {
                Some(names) => names.split(',').map(str::to_string).collect(),
                None => vec![name.to_string()],
            };
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            let sample_time = parse_sample_time(&fields, name, line_number, true, "cscript")?;
            let xc_count = match fields.get("xc_count") {
                Some(s) => s.parse::<usize>().map_err(|_| {
                    format!(
                        "line {}: device '{name}' field 'xc_count' is not a non-negative \
                         integer",
                        line_number + 1
                    )
                })?,
                None => 0,
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::CScript {
                    lib,
                    output_names,
                    sample_time,
                    xc_count,
                },
                inputs,
            })
        }
        "pyblock" => {
            // Same field set as "cscript" above (output_names/inputs/sample_time/xc_count
            // parsing all identical), except `path=` names a `.py` source file instead of a
            // precompiled `.so` -- see BlockKind::PyBlock's own doc comment.
            let path = std::path::PathBuf::from(get_str("path")?);
            let output_names = match fields.get("outputs") {
                Some(names) => names.split(',').map(str::to_string).collect(),
                None => vec![name.to_string()],
            };
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            let sample_time = parse_sample_time(&fields, name, line_number, true, "pyblock")?;
            let xc_count = match fields.get("xc_count") {
                Some(s) => s.parse::<usize>().map_err(|_| {
                    format!(
                        "line {}: device '{name}' field 'xc_count' is not a non-negative \
                         integer",
                        line_number + 1
                    )
                })?,
                None => 0,
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::PyBlock {
                    path,
                    output_names,
                    sample_time,
                    xc_count,
                },
                inputs,
            })
        }
        "pyfunc" => {
            // A genuinely separate contract from "pyblock" above (not a mode of it) -- see
            // BlockKind::PyFunction's own doc comment. `function=` is required (no default):
            // this contract exists specifically for a descriptively-named plain function, not
            // a fixed `output` entry point.
            let path = std::path::PathBuf::from(get_str("path")?);
            let function = get_str("function")?;
            let output_names = match fields.get("outputs") {
                Some(names) => names.split(',').map(str::to_string).collect(),
                None => vec![name.to_string()],
            };
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            let sample_time = parse_sample_time(&fields, name, line_number, false, "pyfunc")?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::PyFunction {
                    path,
                    function,
                    output_names,
                    sample_time,
                },
                inputs,
            })
        }
        "octfunc" => {
            // The .m-file counterpart to "pyfunc" above -- identical field set/parsing (see
            // BlockKind::OctFunc's own doc comment): `function=` required, no default,
            // ts=variable rejected the same way and with the same wording (a stateless function
            // call has no instance to remember a next-hit time against).
            let path = std::path::PathBuf::from(get_str("path")?);
            let function = get_str("function")?;
            let output_names = match fields.get("outputs") {
                Some(names) => names.split(',').map(str::to_string).collect(),
                None => vec![name.to_string()],
            };
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            let sample_time = parse_sample_time(&fields, name, line_number, false, "octfunc")?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::OctFunc {
                    path,
                    function,
                    output_names,
                    sample_time,
                },
                inputs,
            })
        }
        "octblock" => {
            // The stateful sibling of "octfunc" above (not a mode of it -- see
            // BlockKind::OctBlock's own doc comment): same output_names/inputs/sample_time/
            // xc_count field shape "pyblock" uses (ts=variable allowed), plus `function=`
            // required the way "octfunc"/"pyfunc" both require it -- `path=` here names a
            // *directory* (the file-per-function contract), not a single `.m` file.
            let path = std::path::PathBuf::from(get_str("path")?);
            let function = get_str("function")?;
            let output_names = match fields.get("outputs") {
                Some(names) => names.split(',').map(str::to_string).collect(),
                None => vec![name.to_string()],
            };
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            let sample_time = parse_sample_time(&fields, name, line_number, true, "octblock")?;
            let xc_count = match fields.get("xc_count") {
                Some(s) => s.parse::<usize>().map_err(|_| {
                    format!(
                        "line {}: device '{name}' field 'xc_count' is not a non-negative \
                         integer",
                        line_number + 1
                    )
                })?,
                None => 0,
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::OctBlock {
                    path,
                    function,
                    output_names,
                    sample_time,
                    xc_count,
                },
                inputs,
            })
        }
        "statespace" => {
            let a = parse_matrix_rows(&get_str("a")?, name, "a", line_number)?;
            // `b=`/`c=` accept either their original SISO shorthand (a flat list -- `b=[1,0]`
            // means "one input," `c=[1,0]` means "one output," exactly as before vector
            // signals existed) or a genuine matrix (`b=[[1,0],[0,1]]`, n x p for p inputs;
            // `c=[[1,0]]`, q x n for q outputs) for a real MIMO declaration. Detected the same
            // way `parse_gain_value` detects a matrix: does the field, after stripping its own
            // outer brackets, immediately start with another `[`.
            let b = parse_matrix_or_vector_shorthand(&get_str("b")?, name, "b", line_number, true)?;
            let c =
                parse_matrix_or_vector_shorthand(&get_str("c")?, name, "c", line_number, false)?;
            let p = b.first().map_or(0, |row| row.len());
            let q = c.len();
            // `d=` defaults to an all-zero q x p matrix when omitted (generalizing the old
            // default of 0.0 cleanly); given as a matrix (`d=[[..],..]`) for MIMO, or a bare
            // scalar only when the system is genuinely 1x1 (SISO) -- a lone scalar has no
            // unambiguous placement in a larger D matrix, so it's rejected rather than guessed
            // at once q>1 or p>1.
            let d = match fields.get("d") {
                None => vec![vec![0.0; p]; q],
                Some(text) if text.trim().starts_with('[') => {
                    parse_matrix_rows(text, name, "d", line_number)?
                }
                Some(text) => {
                    if q != 1 || p != 1 {
                        return Err(format!(
                            "line {}: device '{name}' field 'd' is a bare scalar, but this \
                             system has {q} output(s) and {p} input(s) -- declare \
                             'd=[[...],...]' ({q}x{p}) for a MIMO system, a bare scalar is only \
                             valid for a 1x1 (SISO) one",
                            line_number + 1
                        ));
                    }
                    let v: f64 = text.trim().parse().map_err(|_| {
                        format!(
                            "line {}: device '{name}' field 'd' is not a number",
                            line_number + 1
                        )
                    })?;
                    vec![vec![v]]
                }
            };
            // No `e=` field exposed at the CLI level yet, so this is always the trivial
            // (always-valid) e=None case -- StateSpace::new still runs the same check every
            // other block-kind constructor here does, so a future `e=` field only has to
            // add parsing, not a new validation path.
            let ss = StateSpace::new(a, b, c, d, None).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid statespace ({e:?})",
                    line_number + 1
                )
            })?;
            // `in=<signal>` (SISO shorthand, p must be 1) or `inputs=<signal>,<signal>,...`
            // (exactly p entries, each independently scalar or vector -- flattened by
            // `dae-runtime`'s own evaluate_blocks into the u vector `StateSpace::rk4_step`
            // expects; the *total* flattened length must equal p, checked there, not here,
            // since arity from a vector-valued upstream signal isn't knowable from netlist text
            // alone).
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::StateSpace(ss),
                inputs,
            })
        }
        "tf" => {
            let num = parse_vector(&get_str("num")?, name, "num", line_number)?;
            let den = parse_vector(&get_str("den")?, name, "den", line_number)?;
            let tf = TransferFunction::new(num, den).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid transfer function ({e:?})",
                    line_number + 1
                )
            })?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::TransferFunction(tf),
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        "discretestatespace" => {
            let a = parse_matrix_rows(&get_str("a")?, name, "a", line_number)?;
            let b = parse_matrix_or_vector_shorthand(&get_str("b")?, name, "b", line_number, true)?;
            let c =
                parse_matrix_or_vector_shorthand(&get_str("c")?, name, "c", line_number, false)?;
            let p = b.first().map_or(0, |row| row.len());
            let q = c.len();
            let d = match fields.get("d") {
                None => vec![vec![0.0; p]; q],
                Some(text) if text.trim().starts_with('[') => {
                    parse_matrix_rows(text, name, "d", line_number)?
                }
                Some(text) => {
                    if q != 1 || p != 1 {
                        return Err(format!(
                            "line {}: device '{name}' field 'd' is a bare scalar, but this \
                             system has {q} output(s) and {p} input(s) -- declare \
                             'd=[[...],...]' ({q}x{p}) for a MIMO system, a bare scalar is only \
                             valid for a 1x1 (SISO) one",
                            line_number + 1
                        ));
                    }
                    let v: f64 = text.trim().parse().map_err(|_| {
                        format!(
                            "line {}: device '{name}' field 'd' is not a number",
                            line_number + 1
                        )
                    })?;
                    vec![vec![v]]
                }
            };
            let ss = StateSpace::new(a, b, c, d, None).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid discrete statespace ({e:?})",
                    line_number + 1
                )
            })?;
            let sample_time = parse_required_periodic_sample_time(&fields, name, line_number)?;
            let inputs = match fields.get("inputs") {
                Some(list) => list.split(',').map(parse_signal).collect(),
                None => vec![parse_signal(&get_str("in")?)],
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::DiscreteStateSpace { ss, sample_time },
                inputs,
            })
        }
        "discretetf" => {
            let num = parse_vector(&get_str("num")?, name, "num", line_number)?;
            let den = parse_vector(&get_str("den")?, name, "den", line_number)?;
            let tf = TransferFunction::new(num, den).map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid discrete transfer function ({e:?})",
                    line_number + 1
                )
            })?;
            let sample_time = parse_required_periodic_sample_time(&fields, name, line_number)?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::DiscreteTransferFunction { tf, sample_time },
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        "discretepid" => {
            let sample_time = parse_required_periodic_sample_time(&fields, name, line_number)?;
            let period = match sample_time {
                SampleTimeSpec::Periodic { period, .. } => period,
                SampleTimeSpec::Variable => {
                    unreachable!("parse_required_periodic_sample_time already rejected ts=variable")
                }
            };
            let method_name = fields
                .get("integration_method")
                .map(String::as_str)
                .unwrap_or("forward");
            let method = DiscreteIntegrationMethod::from_name(method_name).ok_or_else(|| {
                format!(
                    "line {}: device '{name}': unknown integration_method '{method_name}' \
                     (expected 'forward', 'backward', or 'trapezoidal')",
                    line_number + 1
                )
            })?;
            let pid = DiscretePid {
                kp: get("kp")?,
                ki: get("ki")?,
                kd: get("kd")?,
                n: get("n")?,
                period,
                method,
            };
            let error_input = parse_signal(&get_str("in")?);
            let (clamp, inputs) = match (fields.get("clamp_lo_in"), fields.get("clamp_hi_in")) {
                (Some(lo), Some(hi)) => (
                    PidClamp::Dynamic,
                    vec![error_input, parse_signal(lo), parse_signal(hi)],
                ),
                (None, None) => (
                    PidClamp::Fixed(get("clamp_lo")?, get("clamp_hi")?),
                    vec![error_input],
                ),
                _ => {
                    return Err(format!(
                        "line {}: device '{name}': 'clamp_lo_in'/'clamp_hi_in' must both be \
                             given together (dynamic clamp) or both omitted (fixed clamp= \
                             clamp_lo/clamp_hi)",
                        line_number + 1
                    ))
                }
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::DiscretePid {
                    pid,
                    clamp,
                    sample_time,
                },
                inputs,
            })
        }
        "product" => Kind::Block(BlockInstance {
            name: name.to_string(),
            kind: BlockKind::Product,
            inputs: get_str("inputs")?.split(',').map(parse_signal).collect(),
        }),
        "saturation" => Kind::Block(BlockInstance {
            name: name.to_string(),
            kind: BlockKind::Saturation(get("limit")?),
            inputs: vec![parse_signal(&get_str("in")?)],
        }),
        "table" => {
            let points = parse_xy_points(&get_str("points")?, name, "points", line_number)?;
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Table(points),
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        "phys2sig" => {
            let target = match (fields.get("node"), fields.get("branch")) {
                (Some(node), None) => Phys2SigTarget::Voltage(node.clone()),
                (None, Some(branch)) => Phys2SigTarget::Current(branch.clone()),
                (Some(_), Some(_)) => {
                    return Err(format!(
                        "line {}: device '{name}': 'node' and 'branch' are mutually \
                             exclusive (a phys2sig reads either a node voltage or a branch \
                             current, never both)",
                        line_number + 1
                    ))
                }
                (None, None) => {
                    return Err(format!(
                        "line {}: device '{name}' kind='phys2sig' needs 'node=<name>' (reads \
                             V(node)) or 'branch=<name>' (reads I(branch))",
                        line_number + 1
                    ))
                }
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Phys2Sig(target),
                inputs: Vec::new(),
            })
        }
        "sig2phys" => {
            let domain = match get_str("domain")?.as_str() {
                "voltage" => PhysicalDomain::Voltage,
                "current" => PhysicalDomain::Current,
                other => {
                    return Err(format!(
                        "line {}: device '{name}' field 'domain' must be 'voltage' or \
                             'current' (got '{other}')",
                        line_number + 1
                    ))
                }
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Sig2Phys { domain },
                inputs: vec![parse_signal(&get_str("in")?)],
            })
        }
        "clarke" | "clarkeinv" | "park" | "parkinv" | "clarkepark" | "clarkeparkinv" => {
            let ct = match kind {
                "clarke" => CoordinateTransform::Clarke,
                "clarkeinv" => CoordinateTransform::ClarkeInv,
                "park" => CoordinateTransform::Park,
                "parkinv" => CoordinateTransform::ParkInv,
                "clarkepark" => CoordinateTransform::ClarkePark,
                "clarkeparkinv" => CoordinateTransform::ClarkeParkInv,
                _ => unreachable!("matched above"),
            };
            let inputs: Vec<Signal> = get_str("inputs")?.split(',').map(parse_signal).collect();
            if inputs.len() != ct.input_count() {
                return Err(format!(
                    "line {}: device '{name}' kind='{kind}' needs {} inputs (got {})",
                    line_number + 1,
                    ct.input_count(),
                    inputs.len()
                ));
            }
            let output_names = match fields.get("outputs") {
                Some(names) => {
                    let names: Vec<String> = names.split(',').map(str::to_string).collect();
                    if names.len() != 3 {
                        return Err(format!(
                            "line {}: device '{name}': 'outputs' needs exactly 3 entries \
                                 (got {})",
                            line_number + 1,
                            names.len()
                        ));
                    }
                    names
                }
                // Default: block's own name aliases the first (primary) output, same as
                // `kind=cscript`'s default; the other two get readable auto-generated names
                // from this transform's own conventional output names (e.g. `<name>_beta`).
                None => {
                    let suffixes = ct.output_names();
                    std::iter::once(name.to_string())
                        .chain(suffixes[1..].iter().map(|s| format!("{name}_{s}")))
                        .collect()
                }
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::CoordinateTransform {
                    kind: ct,
                    output_names,
                },
                inputs,
            })
        }
        "pmsm" => {
            let pmsm = continuous_blocks::Pmsm::new(
                get("r_s")?,
                get("l_d")?,
                get("l_q")?,
                get("lambda_pm")?,
                get("pole_pairs")?,
                get("inertia")?,
                get("friction")?,
            )
            .map_err(|e| {
                format!(
                    "line {}: device '{name}': invalid pmsm ({e:?})",
                    line_number + 1
                )
            })?;
            let inputs: Vec<Signal> = get_str("inputs")?.split(',').map(parse_signal).collect();
            if inputs.len() != 3 {
                return Err(format!(
                    "line {}: device '{name}' kind='pmsm' needs 3 inputs (vd,vq,t_load; \
                         got {})",
                    line_number + 1,
                    inputs.len()
                ));
            }
            let output_names = match fields.get("outputs") {
                Some(names) => {
                    let names: Vec<String> = names.split(',').map(str::to_string).collect();
                    if names.len() != 4 {
                        return Err(format!(
                            "line {}: device '{name}': 'outputs' needs exactly 4 entries \
                                 (got {})",
                            line_number + 1,
                            names.len()
                        ));
                    }
                    names
                }
                // Default: block's own name aliases the first (primary) output (`id`), same
                // convention as `kind=cscript`/`kind=clarke`; the other three get readable
                // auto-generated names.
                None => {
                    let suffixes = ["id", "iq", "omega_m", "theta_e"];
                    std::iter::once(name.to_string())
                        .chain(suffixes[1..].iter().map(|s| format!("{name}_{s}")))
                        .collect()
                }
            };
            Kind::Block(BlockInstance {
                name: name.to_string(),
                kind: BlockKind::Pmsm { pmsm, output_names },
                inputs,
            })
        }
        other => {
            // Not one of the block kinds above: try the real-valued waveform-arithmetic
            // function library (cos, sin, exp, sqrt, atan2, hypot, if, limit, ...) before
            // giving up — see `continuous_blocks::waveform_arithmetic` for the full list.
            if let Some(f) = continuous_blocks::MathFn1::from_name(other) {
                Kind::Block(BlockInstance {
                    name: name.to_string(),
                    kind: BlockKind::MathFn1(f),
                    inputs: vec![parse_signal(&get_str("in")?)],
                })
            } else if let Some(f) = continuous_blocks::MathFn2::from_name(other) {
                Kind::Block(BlockInstance {
                    name: name.to_string(),
                    kind: BlockKind::MathFn2(f),
                    inputs: vec![
                        parse_signal(&get_str("in1")?),
                        parse_signal(&get_str("in2")?),
                    ],
                })
            } else if let Some(f) = continuous_blocks::MathFn3::from_name(other) {
                Kind::Block(BlockInstance {
                    name: name.to_string(),
                    kind: BlockKind::MathFn3(f),
                    inputs: vec![
                        parse_signal(&get_str("in1")?),
                        parse_signal(&get_str("in2")?),
                        parse_signal(&get_str("in3")?),
                    ],
                })
            } else if let Some(op) = LogicOp::from_name(other) {
                let inputs = if op.is_unary() {
                    vec![parse_signal(&get_str("in")?)]
                } else {
                    let list = get_str("inputs")?;
                    let parsed: Vec<Signal> = list.split(',').map(parse_signal).collect();
                    if parsed.len() < 2 {
                        return Err(format!(
                            "line {}: device '{name}': kind={other} needs at least 2 \
                             comma-separated inputs= entries (got {})",
                            line_number + 1,
                            parsed.len()
                        ));
                    }
                    parsed
                };
                Kind::Block(BlockInstance {
                    name: name.to_string(),
                    kind: BlockKind::LogicGate(op),
                    inputs,
                })
            } else if let Some(kind) = FlipFlopKind::from_name(other) {
                let mut inputs = vec![parse_signal(&get_str("clk")?)];
                match kind {
                    FlipFlopKind::D => inputs.push(parse_signal(&get_str("d")?)),
                    FlipFlopKind::T => inputs.push(parse_signal(&get_str("t")?)),
                    FlipFlopKind::Jk => {
                        inputs.push(parse_signal(&get_str("j")?));
                        inputs.push(parse_signal(&get_str("k")?));
                    }
                }
                let reset = fields.contains_key("reset");
                if reset {
                    inputs.push(parse_signal(&get_str("reset")?));
                }
                Kind::Block(BlockInstance {
                    name: name.to_string(),
                    kind: BlockKind::FlipFlop { kind, reset },
                    inputs,
                })
            } else {
                return Err(format!(
                    "line {}: unknown device kind '{other}'",
                    line_number + 1
                ));
            }
        }
    };

    Ok(entry)
}
