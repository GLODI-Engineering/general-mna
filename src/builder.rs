use std::collections::{BTreeMap, HashMap};
use std::fmt;

use general_spice_core::ast::{ElementInstance, Statement};
use general_spice_core::{lexer, parser, Dialect};

use crate::{Expression, InitialCondition, Matrix, MnaSystem, TransientFunction};

/// Treatment of parsed devices for which this linear MNA crate has no stamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedElementPolicy {
    /// Stop and report the unsupported element.
    Error,
    /// Skip it and append a warning to the result.
    IgnoreWithWarning,
}

/// State assigned to an idealized converter switch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwitchState {
    /// Use [`BuildOptions::on_resistance`].
    On,
    /// Use [`BuildOptions::off_resistance`].
    Off,
}

/// Options controlling matrix construction.
#[derive(Debug, Clone, PartialEq)]
pub struct BuildOptions {
    /// Handling of nonlinear or otherwise unstamped devices.
    pub unsupported_elements: UnsupportedElementPolicy,
    /// Case-insensitive element-name to switch state mapping.
    pub switch_states: BTreeMap<String, SwitchState>,
    /// Resistance used for switches in the on state.
    pub on_resistance: Expression,
    /// Resistance used for switches in the off state.
    pub off_resistance: Expression,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            unsupported_elements: UnsupportedElementPolicy::Error,
            switch_states: BTreeMap::new(),
            on_resistance: Expression::symbol("Ron"),
            off_resistance: Expression::symbol("Roff"),
        }
    }
}

impl BuildOptions {
    /// Adds or replaces a case-insensitive switch-state override.
    pub fn set_switch(&mut self, element_name: impl Into<String>, state: SwitchState) {
        self.switch_states
            .insert(normalize(&element_name.into()), state);
    }
}

/// Builds MNA systems from `general-spice-core` statements or netlist text.
#[derive(Debug, Clone)]
pub struct MnaBuilder {
    dialect: Dialect,
    options: BuildOptions,
}

impl MnaBuilder {
    /// Creates a builder for a SPICE dialect.
    pub fn new(dialect: Dialect) -> Self {
        Self {
            dialect,
            options: BuildOptions::default(),
        }
    }

    /// Creates a builder with explicit options.
    pub fn with_options(dialect: Dialect, options: BuildOptions) -> Self {
        Self { dialect, options }
    }

    /// Returns the build options.
    pub fn options(&self) -> &BuildOptions {
        &self.options
    }

    /// Returns mutable build options.
    pub fn options_mut(&mut self) -> &mut BuildOptions {
        &mut self.options
    }

    /// Parses a title-less netlist fragment and builds its MNA system.
    pub fn build_fragment(&self, source: &str) -> Result<MnaSystem, BuildError> {
        let lines = lexer::preprocess(source, self.dialect);
        let parsed = parser::parse(&lines, self.dialect);
        self.finish_parse(parsed)
    }

    /// Parses a complete SPICE document, whose first line is the title.
    pub fn build_document(&self, source: &str) -> Result<MnaSystem, BuildError> {
        let lines = lexer::preprocess(source, self.dialect);
        let parsed = parser::parse_document(&lines, self.dialect);
        self.finish_parse(parsed)
    }

    fn finish_parse(&self, parsed: Vec<parser::ParseResult>) -> Result<MnaSystem, BuildError> {
        let mut statements = Vec::with_capacity(parsed.len());
        for result in parsed {
            match result {
                Ok(statement) => statements.push(statement),
                Err(error) => {
                    return Err(BuildError::Parse {
                        message: error.message,
                        line_start: error.span.start,
                        line_end: error.span.end,
                    })
                }
            }
        }
        self.build_statements(&statements)
    }

    /// Builds an MNA system from an already parsed AST.
    pub fn build_statements(&self, statements: &[Statement]) -> Result<MnaSystem, BuildError> {
        let elements: Vec<&ElementInstance> = statements
            .iter()
            .filter_map(|statement| match statement {
                Statement::ElementInstance(element) => Some(element),
                _ => None,
            })
            .collect();

        for element in &elements {
            // A switch-assigned element is stamped as a plain on/off admittance whatever its
            // device letter, so its own positional grammar is not this crate's to enforce --
            // only its `key=value` tokens, which would still be silently dropped.
            let grammar = match self.switch_state(element) {
                Some(_) => Some(PositionalGrammar::Unchecked),
                None if is_stamped_letter(element.device_letter) => positional_grammar(element),
                None => None,
            };
            if let Some(grammar) = grammar {
                validate_trailing_fields(element, grammar)?;
            }
        }

        let mut index = UnknownIndex::default();
        for element in &elements {
            if element.device_letter == 'K' {
                continue;
            }
            let node_count = if self.switch_state(element).is_some() {
                element.nodes.len().min(2)
            } else {
                element.nodes.len()
            };
            for node in &element.nodes[..node_count] {
                index.add_node(node);
            }
        }
        index.finish_nodes();

        for element in &elements {
            if self.is_branch_device(element) {
                index.add_branch(&element.name)?;
            }
        }

        let mut inputs = Vec::new();
        let mut input_values = Vec::new();
        let mut input_by_element = HashMap::new();
        let mut transient_sources = BTreeMap::new();
        for element in &elements {
            if matches!(element.device_letter, 'V' | 'I') {
                let column = inputs.len();
                let key = normalize(&element.name);
                if input_by_element.insert(key, column).is_some() {
                    return Err(BuildError::DuplicateElement(element.name.clone()));
                }
                inputs.push(element.name.clone());
                if let Some(transient_fn) = TransientFunction::parse_params(&element.raw_params) {
                    // A time-varying source is stamped as a named symbol, not a baked
                    // literal, exactly like a PWL diode's own `{name}_Ioff` just below --
                    // the caller supplies its numeric value fresh at every step via
                    // `evaluate`'s `values` map, using `transient_sources` to know it needs
                    // to (see this type's own doc comment for the full contract).
                    transient_sources.insert(element.name.clone(), transient_fn);
                    input_values.push(Expression::symbol(element.name.clone()));
                } else {
                    input_values.push(source_value(element)?);
                }
            } else if element.device_letter == 'D' {
                // A piecewise-linear (or otherwise externally companion-modeled) diode is
                // stamped as a conductance (see the 'D' arm below) plus a Norton current
                // source of value `{name}_Ioff`. Both are per-instance symbolic parameters —
                // not parsed from the netlist, since which segment (and therefore which
                // numeric G/Ioff) is active is decided per timestep by the caller's own
                // piecewise-linear mode-selection, not by this crate. See `elspice-pwl`'s
                // `docs/architecture.md` for why this is the right split of responsibility.
                let column = inputs.len();
                let key = normalize(&element.name);
                if input_by_element.insert(key, column).is_some() {
                    return Err(BuildError::DuplicateElement(element.name.clone()));
                }
                inputs.push(element.name.clone());
                input_values.push(Expression::symbol(format!("{}_Ioff", element.name)));
            }
        }

        // `ic=` initial conditions, resolved against the unknown ordering now that `index` is
        // final. Deliberately collected *after* stamping and kept out of `a`/`k`/`b`/`u`: an
        // initial condition constrains `x` at one instant, it does not change the circuit's
        // equations. `MnaSystem::initial_state` turns these into a consistent starting vector.
        let mut initial_conditions = Vec::new();
        for element in &elements {
            if self.switch_state(element).is_some() {
                continue;
            }
            let Some(raw) = initial_condition_field(element) else {
                continue;
            };
            let value = Expression::parse_scalar(raw).map_err(|message| {
                BuildError::InvalidElementField {
                    line: element.span.start,
                    name: element.name.clone(),
                    message: format!("field 'ic' is not a value: {message}"),
                }
            })?;
            match element.device_letter {
                'C' => {
                    let (positive, negative) = two_nodes(element)?;
                    initial_conditions.push(InitialCondition::CapacitorVoltage {
                        element: element.name.clone(),
                        positive: index.node(positive),
                        negative: index.node(negative),
                        value,
                    });
                }
                'L' => initial_conditions.push(InitialCondition::InductorCurrent {
                    element: element.name.clone(),
                    branch: index.branch(&element.name)?,
                    value,
                }),
                // Unreachable: `accepted_fields` only offers `ic` on `C`/`L`, so any other
                // letter carrying one was already rejected by `validate_trailing_fields`.
                _ => unreachable!("ic= accepted only on C and L"),
            }
        }

        let order = index.names.len();
        let mut a = Matrix::filled(order, order, Expression::zero());
        let mut k = Matrix::filled(order, order, Expression::zero());
        let mut b = Matrix::filled(order, inputs.len(), Expression::zero());
        let mut warnings = Vec::new();

        let inductances = elements
            .iter()
            .filter(|element| element.device_letter == 'L')
            .map(|element| Ok((normalize(&element.name), scalar_value(element)?)))
            .collect::<Result<HashMap<_, _>, BuildError>>()?;

        for element in elements {
            if let Some(state) = self.switch_state(element) {
                let resistance = match state {
                    SwitchState::On => self.options.on_resistance.clone(),
                    SwitchState::Off => self.options.off_resistance.clone(),
                };
                stamp_admittance(&mut a, &index, element, resistance.reciprocal())?;
                continue;
            }

            match element.device_letter {
                'R' => {
                    stamp_admittance(&mut a, &index, element, scalar_value(element)?.reciprocal())?
                }
                'C' => stamp_admittance(&mut k, &index, element, scalar_value(element)?)?,
                'L' => stamp_inductor(&mut a, &mut k, &index, element)?,
                'V' => stamp_voltage_source(
                    &mut a,
                    &mut b,
                    &index,
                    element,
                    input_by_element[&normalize(&element.name)],
                )?,
                'I' => stamp_current_source(
                    &mut b,
                    &index,
                    element,
                    input_by_element[&normalize(&element.name)],
                )?,
                'G' => stamp_vccs(&mut a, &index, element)?,
                'E' => stamp_vcvs(&mut a, &index, element)?,
                'F' => stamp_cccs(&mut a, &index, element)?,
                'H' => stamp_ccvs(&mut a, &index, element)?,
                'K' => stamp_mutual_inductance(&mut k, &index, element, &inductances)?,
                'D' => {
                    let conductance = Expression::symbol(format!("{}_G", element.name));
                    stamp_admittance(&mut a, &index, element, conductance)?;
                    stamp_current_source(
                        &mut b,
                        &index,
                        element,
                        input_by_element[&normalize(&element.name)],
                    )?;
                }
                _ => self.unsupported(element, &mut warnings)?,
            }
        }

        let mut u = vec![Expression::zero(); order];
        for row in 0..order {
            for column in 0..inputs.len() {
                u[row] = u[row].clone() + b[(row, column)].clone() * input_values[column].clone();
            }
        }

        let mut parameter_defaults = BTreeMap::new();
        for statement in statements {
            if let Statement::Param(param) | Statement::GlobalParam(param) = statement {
                for (name, raw_value) in &param.assignments {
                    if let Ok(value) = Expression::parse_scalar(raw_value) {
                        parameter_defaults.insert(name.clone(), value);
                    } else {
                        warnings.push(format!(
                            "parameter '{name}' has an expression that is preserved by general-spice-core but is not a scalar MNA value"
                        ));
                    }
                }
            }
        }

        Ok(MnaSystem {
            a,
            k,
            b,
            u,
            unknowns: index.names,
            inputs,
            input_values,
            transient_sources,
            initial_conditions,
            parameter_defaults,
            warnings,
        })
    }

    fn is_branch_device(&self, element: &ElementInstance) -> bool {
        self.switch_state(element).is_none()
            && matches!(element.device_letter, 'V' | 'L' | 'E' | 'H')
    }

    fn switch_state(&self, element: &ElementInstance) -> Option<SwitchState> {
        self.options
            .switch_states
            .get(&normalize(&element.name))
            .copied()
    }

    fn unsupported(
        &self,
        element: &ElementInstance,
        warnings: &mut Vec<String>,
    ) -> Result<(), BuildError> {
        let message = format!(
            "element '{}' (device {}) has no linear MNA stamp",
            element.name, element.device_letter
        );
        match self.options.unsupported_elements {
            UnsupportedElementPolicy::Error => Err(BuildError::UnsupportedElement(message)),
            UnsupportedElementPolicy::IgnoreWithWarning => {
                warnings.push(message);
                Ok(())
            }
        }
    }
}

#[derive(Debug, Default)]
struct UnknownIndex {
    names: Vec<String>,
    nodes: HashMap<String, usize>,
    branches: HashMap<String, usize>,
    nodes_finished: bool,
}

impl UnknownIndex {
    fn add_node(&mut self, name: &str) {
        if is_ground(name) {
            return;
        }
        let key = normalize(name);
        if !self.nodes.contains_key(&key) {
            let index = self.names.len();
            self.nodes.insert(key, index);
            self.names.push(format!("V({name})"));
        }
    }

    fn finish_nodes(&mut self) {
        self.nodes_finished = true;
    }

    fn add_branch(&mut self, name: &str) -> Result<(), BuildError> {
        debug_assert!(self.nodes_finished);
        let key = normalize(name);
        if self.branches.contains_key(&key) {
            return Err(BuildError::DuplicateElement(name.to_string()));
        }
        let index = self.names.len();
        self.branches.insert(key, index);
        self.names.push(format!("I({name})"));
        Ok(())
    }

    fn node(&self, name: &str) -> Option<usize> {
        if is_ground(name) {
            None
        } else {
            self.nodes.get(&normalize(name)).copied()
        }
    }

    fn branch(&self, name: &str) -> Result<usize, BuildError> {
        self.branches
            .get(&normalize(name))
            .copied()
            .ok_or_else(|| BuildError::UnknownControllingBranch(name.to_string()))
    }
}

fn stamp(
    matrix: &mut Matrix<Expression>,
    row: Option<usize>,
    col: Option<usize>,
    value: Expression,
) {
    if let (Some(row), Some(col)) = (row, col) {
        matrix[(row, col)] = matrix[(row, col)].clone() + value;
    }
}

fn stamp_input(matrix: &mut Matrix<Expression>, row: Option<usize>, col: usize, value: Expression) {
    if let Some(row) = row {
        matrix[(row, col)] = matrix[(row, col)].clone() + value;
    }
}

fn two_nodes(element: &ElementInstance) -> Result<(&str, &str), BuildError> {
    match element.nodes.as_slice() {
        [positive, negative, ..] => Ok((positive, negative)),
        _ => Err(BuildError::InvalidElement {
            name: element.name.clone(),
            message: "requires two terminals".into(),
        }),
    }
}

fn four_nodes(element: &ElementInstance) -> Result<(&str, &str, &str, &str), BuildError> {
    match element.nodes.as_slice() {
        [out_positive, out_negative, ctrl_positive, ctrl_negative] => {
            Ok((out_positive, out_negative, ctrl_positive, ctrl_negative))
        }
        _ => Err(BuildError::InvalidElement {
            name: element.name.clone(),
            message: "requires the classic four-node linear form".into(),
        }),
    }
}

fn stamp_admittance(
    matrix: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
    admittance: Expression,
) -> Result<(), BuildError> {
    let (positive, negative) = two_nodes(element)?;
    let p = index.node(positive);
    let n = index.node(negative);
    stamp(matrix, p, p, admittance.clone());
    stamp(matrix, n, n, admittance.clone());
    stamp(matrix, p, n, -admittance.clone());
    stamp(matrix, n, p, -admittance);
    Ok(())
}

fn stamp_branch_incidence(
    matrix: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
) -> Result<usize, BuildError> {
    let (positive, negative) = two_nodes(element)?;
    let branch = index.branch(&element.name)?;
    let p = index.node(positive);
    let n = index.node(negative);
    stamp(matrix, p, Some(branch), Expression::one());
    stamp(matrix, n, Some(branch), -Expression::one());
    stamp(matrix, Some(branch), p, Expression::one());
    stamp(matrix, Some(branch), n, -Expression::one());
    Ok(branch)
}

fn stamp_inductor(
    a: &mut Matrix<Expression>,
    k: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
) -> Result<(), BuildError> {
    let branch = stamp_branch_incidence(a, index, element)?;
    stamp(k, Some(branch), Some(branch), -scalar_value(element)?);
    Ok(())
}

fn stamp_voltage_source(
    a: &mut Matrix<Expression>,
    b: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
    input: usize,
) -> Result<(), BuildError> {
    let branch = stamp_branch_incidence(a, index, element)?;
    stamp_input(b, Some(branch), input, Expression::one());
    Ok(())
}

fn stamp_current_source(
    b: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
    input: usize,
) -> Result<(), BuildError> {
    let (positive, negative) = two_nodes(element)?;
    stamp_input(b, index.node(positive), input, -Expression::one());
    stamp_input(b, index.node(negative), input, Expression::one());
    Ok(())
}

fn stamp_vccs(
    a: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
) -> Result<(), BuildError> {
    let (op, on, cp, cn) = four_nodes(element)?;
    let gain = scalar_value(element)?;
    stamp(a, index.node(op), index.node(cp), gain.clone());
    stamp(a, index.node(op), index.node(cn), -gain.clone());
    stamp(a, index.node(on), index.node(cp), -gain.clone());
    stamp(a, index.node(on), index.node(cn), gain);
    Ok(())
}

fn stamp_vcvs(
    a: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
) -> Result<(), BuildError> {
    let (_, _, cp, cn) = four_nodes(element)?;
    let branch = stamp_branch_incidence(a, index, element)?;
    let gain = scalar_value(element)?;
    stamp(a, Some(branch), index.node(cp), -gain.clone());
    stamp(a, Some(branch), index.node(cn), gain);
    Ok(())
}

fn stamp_cccs(
    a: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
) -> Result<(), BuildError> {
    let (positive, negative) = two_nodes(element)?;
    let (control, gain) = control_and_gain(element)?;
    let control = index.branch(control)?;
    stamp(a, index.node(positive), Some(control), gain.clone());
    stamp(a, index.node(negative), Some(control), -gain);
    Ok(())
}

fn stamp_ccvs(
    a: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
) -> Result<(), BuildError> {
    let branch = stamp_branch_incidence(a, index, element)?;
    let (control, gain) = control_and_gain(element)?;
    let control = index.branch(control)?;
    stamp(a, Some(branch), Some(control), -gain);
    Ok(())
}

fn stamp_mutual_inductance(
    k_matrix: &mut Matrix<Expression>,
    index: &UnknownIndex,
    element: &ElementInstance,
    inductances: &HashMap<String, Expression>,
) -> Result<(), BuildError> {
    if element.nodes.len() < 2 {
        return Err(BuildError::InvalidElement {
            name: element.name.clone(),
            message: "requires at least two inductor names".into(),
        });
    }
    let coefficient = scalar_value(element)?;
    for first in 0..element.nodes.len() {
        for second in first + 1..element.nodes.len() {
            let first_name = &element.nodes[first];
            let second_name = &element.nodes[second];
            let first_l = inductances
                .get(&normalize(first_name))
                .ok_or_else(|| BuildError::UnknownInductor(first_name.clone()))?;
            let second_l = inductances
                .get(&normalize(second_name))
                .ok_or_else(|| BuildError::UnknownInductor(second_name.clone()))?;
            let mutual = coefficient.clone() * (first_l.clone() * second_l.clone()).sqrt();
            let first_branch = index.branch(first_name)?;
            let second_branch = index.branch(second_name)?;
            stamp(
                k_matrix,
                Some(first_branch),
                Some(second_branch),
                -mutual.clone(),
            );
            stamp(k_matrix, Some(second_branch), Some(first_branch), -mutual);
        }
    }
    Ok(())
}

/// Device letters this crate has a linear stamp for, and therefore holds to a known parameter
/// grammar. Anything else is `UnsupportedElementPolicy`'s business (error, or skip with a
/// warning) and is deliberately *not* field-checked here: a `M`/`Q`/`J` card's own `L=1u W=10u`
/// is perfectly valid netlist text this crate simply has no model for, and rejecting its fields
/// would turn `IgnoreWithWarning` into a hard error.
fn is_stamped_letter(device_letter: char) -> bool {
    matches!(
        device_letter,
        'R' | 'C' | 'L' | 'K' | 'V' | 'I' | 'E' | 'F' | 'G' | 'H' | 'D'
    )
}

/// The trailing `key=value` fields this crate understands on a device card it stamps, keyed by
/// device letter and compared case-insensitively.
///
/// `general-spice-core` deliberately hands every token after the node list over uninterpreted,
/// in [`ElementInstance::raw_params`]; deciding what they mean is this crate's job. Any key not
/// listed here is rejected rather than dropped, because dropping it is not a harmless no-op —
/// a silently ignored `tc1=`/`temp=`/`m=`, or a misspelled `ic=`, produces a plausible-looking
/// waveform for a *different* circuit than the one the author wrote, with nothing on screen to
/// say so.
fn accepted_fields(device_letter: char) -> &'static [&'static str] {
    match device_letter {
        // See `MnaSystem::initial_conditions` for what `ic=` means on each, sign included.
        'C' | 'L' => &["ic"],
        _ => &[],
    }
}

/// What a stamped device letter's *positional* (non-`key=value`) parameter list may contain.
///
/// `general-spice-core` hands every token after the node list over uninterpreted, so a
/// positional token this crate does not read is dropped exactly as silently as an unknown
/// field was before — `V1 1 0 10 wibble` and `V1 1 0 SNI(0 1 1k)` both used to simulate a
/// plain 10 V / 0 V source and say nothing. These are the shapes each letter actually permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PositionalGrammar {
    /// Exactly one token, the element's own value. `R`/`C`/`L` (resistance, capacitance,
    /// inductance), `K` (coupling coefficient — `general-spice-core` puts every earlier token
    /// in `nodes` as an inductor name), and `E`/`G` in the classic four-node form, whose single
    /// parameter is the gain.
    OneValue,
    /// Exactly two tokens: a controlling source's name, then a gain — `F`/`H`. The name itself
    /// is checked later, by `UnknownControllingBranch`.
    ControlAndGain,
    /// An independent source's clause list — `V`/`I`. See [`validate_source_positional`].
    SourceClauses,
    /// A diode's model name, optional area factor and optional `OFF` — `D`. See
    /// [`validate_diode_positional`], including what about it is deliberately not checked.
    DiodeModel,
    /// Positional tokens this crate deliberately does not check, while still checking the
    /// card's `key=value` fields. Used for a switch-assigned element, whose parameters are
    /// replaced wholesale by the on/off resistance.
    Unchecked,
}

/// The positional grammar to hold `element` to, or `None` to skip validating the card entirely.
///
/// `None` is only ever returned for the two-node behavioral form of `E`/`G`
/// (`E1 out 0 VALUE={...}` / `TABLE` / `POLY`), which this crate does not stamp at all: its
/// tokens are neither a gain nor `key=value` fields, and `four_nodes` already rejects the card
/// with a message that says exactly what is missing ("requires the classic four-node linear
/// form"). Field-checking it first would replace that with a misleading `unknown field 'value'`.
fn positional_grammar(element: &ElementInstance) -> Option<PositionalGrammar> {
    match element.device_letter {
        'R' | 'C' | 'L' | 'K' => Some(PositionalGrammar::OneValue),
        'E' | 'G' if element.nodes.len() == 4 => Some(PositionalGrammar::OneValue),
        'E' | 'G' => None,
        'F' | 'H' => Some(PositionalGrammar::ControlAndGain),
        'V' | 'I' => Some(PositionalGrammar::SourceClauses),
        'D' => Some(PositionalGrammar::DiodeModel),
        _ => Some(PositionalGrammar::Unchecked),
    }
}

/// Splits a raw trailing token into `(key, value)` if it has the `key=value` shape, with the key
/// lowercased for the case-insensitive comparison every SPICE dialect expects. A token with no
/// `=`, or with an empty key (`=5`), is positional, not a field.
fn as_field(token: &str) -> Option<(String, &str)> {
    let (key, value) = token.split_once('=')?;
    let key = key.trim();
    if key.is_empty() {
        return None;
    }
    Some((key.to_ascii_lowercase(), value))
}

/// Rejects a trailing token this crate would otherwise silently discard.
///
/// Mirrors the block (`kind=...`) parser's own diagnostics in
/// [`crate::system_builder`] — same `line N: device '<name>' ...` shape — so a device card and a
/// block line report a bad field the same way.
///
/// Two passes, because they fail for different reasons: every `key=value` token is checked
/// against [`accepted_fields`], and whatever is left over — the positional tokens, in order —
/// is checked against the letter's own [`PositionalGrammar`].
fn validate_trailing_fields(
    element: &ElementInstance,
    grammar: PositionalGrammar,
) -> Result<(), BuildError> {
    let letter = element.device_letter;
    let accepted = accepted_fields(letter);
    let mut positional: Vec<&str> = Vec::new();

    for token in &element.raw_params {
        let Some((key, _)) = as_field(token) else {
            positional.push(token.as_str());
            continue;
        };
        if accepted.iter().any(|allowed| *allowed == key) {
            continue;
        }
        return Err(BuildError::InvalidElementField {
            line: element.span.start,
            name: element.name.clone(),
            message: match accepted {
                [] => {
                    format!("unknown field '{key}' (device {letter} accepts no key=value fields)")
                }
                _ => format!(
                    "unknown field '{key}' (device {letter} accepts only: {})",
                    accepted.join(", ")
                ),
            },
        });
    }

    match grammar {
        PositionalGrammar::OneValue if positional.len() > 1 => Err(field_error(
            element,
            format!(
                "unexpected extra parameter '{}' (device {letter} takes exactly one value{})",
                positional[1],
                accepted_suffix(accepted)
            ),
        )),
        PositionalGrammar::ControlAndGain if positional.len() > 2 => Err(field_error(
            element,
            format!(
                "unexpected extra parameter '{}' (device {letter} takes exactly two values, a \
                 controlling source name and a gain)",
                positional[2]
            ),
        )),
        PositionalGrammar::SourceClauses => validate_source_positional(element, &positional),
        PositionalGrammar::DiodeModel => validate_diode_positional(element, &positional),
        _ => Ok(()),
    }
}

/// A positional-token diagnostic on `element`, in the same shape as the field diagnostics above.
fn field_error(element: &ElementInstance, message: String) -> BuildError {
    BuildError::InvalidElementField {
        line: element.span.start,
        name: element.name.clone(),
        message,
    }
}

/// Whether `token` is a value this crate can read — a SPICE number with or without a suffix, a
/// symbol resolved later from `.param`, or a braced expression. Not a judgement about physical
/// units, only about shape.
fn is_value_token(token: &str) -> bool {
    Expression::parse_scalar(token).is_ok()
}

/// Validates the positional parameters of an independent source card (`V`/`I`).
///
/// The grammar is a sequence of clauses, in any order, each of which this crate either reads or
/// knowingly ignores:
///
/// - a bare value as the very first token — the implicit DC value (`V1 1 0 10`);
/// - `DC <value>` — the explicit spelling of the same thing;
/// - `AC <magnitude> [<phase>]` — read by no analysis this crate implements yet, but valid, and
///   so accepted rather than rejected;
/// - one call to one of [`TransientFunction::FUNCTION_NAMES`], whose arguments must actually
///   parse as that function's parameter list.
///
/// Anything else is a typo. A second bare value (`V1 1 0 10 20`), a trailing word
/// (`V1 1 0 10 wibble`), a misspelled function (`SNI(0 1 1k)`), a `DC` with nothing after it,
/// and an unterminated argument list are all rejected here instead of being dropped.
///
/// Deliberately not checked: a symbol where an `AC` magnitude or phase belongs. `AC {gain}` is
/// a legitimate parameterized magnitude, and nothing in the token itself distinguishes it from
/// a misspelling, so at most two value-shaped tokens are consumed after `AC` and a third one
/// falls through to the "unexpected extra value" arm.
fn validate_source_positional(
    element: &ElementInstance,
    positional: &[&str],
) -> Result<(), BuildError> {
    let letter = element.device_letter;
    let mut index = 0usize;
    let mut seen_dc = false;
    let mut seen_ac = false;
    let mut seen_function = false;

    while index < positional.len() {
        let token = positional[index];
        if token.eq_ignore_ascii_case("dc") {
            if seen_dc {
                return Err(field_error(
                    element,
                    format!("device {letter} declares a DC value more than once"),
                ));
            }
            seen_dc = true;
            match positional.get(index + 1) {
                Some(value) if is_value_token(value) => index += 2,
                Some(value) => {
                    return Err(field_error(
                        element,
                        format!("'dc' is followed by '{value}', which is not a value"),
                    ))
                }
                None => {
                    return Err(field_error(
                        element,
                        "'dc' is not followed by a value".to_string(),
                    ))
                }
            }
        } else if token.eq_ignore_ascii_case("ac") {
            if seen_ac {
                return Err(field_error(
                    element,
                    format!("device {letter} declares an AC specification more than once"),
                ));
            }
            seen_ac = true;
            let mut consumed = 0usize;
            while consumed < 2 {
                match positional.get(index + 1 + consumed) {
                    Some(next)
                        if !next.eq_ignore_ascii_case("dc")
                            && !next.eq_ignore_ascii_case("ac")
                            && !next.contains('(')
                            && is_value_token(next) =>
                    {
                        consumed += 1
                    }
                    _ => break,
                }
            }
            if consumed == 0 {
                return Err(field_error(
                    element,
                    "'ac' is not followed by a magnitude".to_string(),
                ));
            }
            index += 1 + consumed;
        } else if let Some(paren) = token.find('(') {
            let name = &token[..paren];
            if !TransientFunction::is_function_name(name) {
                return Err(field_error(
                    element,
                    format!(
                        "unknown source function '{name}' (device {letter} accepts {})",
                        TransientFunction::FUNCTION_NAMES.join("/")
                    ),
                ));
            }
            if seen_function {
                return Err(field_error(
                    element,
                    format!("device {letter} declares more than one transient function"),
                ));
            }
            seen_function = true;
            let Some(end) = positional[index..]
                .iter()
                .position(|token| token.contains(')'))
                .map(|offset| offset + index)
            else {
                return Err(field_error(
                    element,
                    format!("'{name}(' is never closed by a ')'"),
                ));
            };
            let call: Vec<String> = positional[index..=end]
                .iter()
                .map(|token| (*token).to_string())
                .collect();
            if TransientFunction::parse(&call).is_none() {
                return Err(field_error(
                    element,
                    format!("'{}' is not a valid {name} parameter list", call.join(" ")),
                ));
            }
            index = end + 1;
        } else if index == 0 && is_value_token(token) {
            // The implicit DC value, which SPICE allows only in first position.
            seen_dc = true;
            index += 1;
        } else {
            return Err(field_error(
                element,
                format!(
                    "unexpected extra parameter '{token}' (device {letter} takes a DC value, an \
                     'ac' specification, and/or one of {}{})",
                    TransientFunction::FUNCTION_NAMES.join("/"),
                    accepted_suffix(accepted_fields(letter))
                ),
            ));
        }
    }
    Ok(())
}

/// Validates the positional parameters of a diode card (`D`).
///
/// SPICE's grammar here is a model name, an optional area factor and an optional `OFF` hint, so
/// at most one non-`OFF` token beyond the model name is permitted, it has to be value-shaped,
/// and `OFF` may not be repeated.
///
/// **The model name itself is deliberately not checked, and cannot be.** This crate does not
/// stamp a diode from a device model at all — a `D` card becomes a symbolic conductance
/// `{name}_G` plus a Norton current `{name}_Ioff` that the caller resolves per timestep (see
/// the `'D'` arm of `build_statements`), so there is no model parameter set here to check a
/// name against. Even a crate that did have one could not decide the question from the netlist
/// alone: the `.model` card may live behind an `.include` this crate never resolves, so an
/// unrecognized name means "not found here", not "misspelled". A misspelled diode model name
/// is therefore a known, documented limit of this validation, recorded in the README.
fn validate_diode_positional(
    element: &ElementInstance,
    positional: &[&str],
) -> Result<(), BuildError> {
    let mut off = 0usize;
    let mut values: Vec<&str> = Vec::new();
    for token in positional {
        if token.eq_ignore_ascii_case("off") {
            off += 1;
        } else {
            values.push(token);
        }
    }
    if off > 1 {
        return Err(field_error(
            element,
            "'off' is given more than once".to_string(),
        ));
    }
    if let Some(area) = values.get(1) {
        if !is_value_token(area) {
            return Err(field_error(
                element,
                format!("'{area}' is not a value (device D's second parameter is its area factor)"),
            ));
        }
    }
    if let Some(extra) = values.get(2) {
        return Err(field_error(
            element,
            format!(
                "unexpected extra parameter '{extra}' (device D takes a model name, an optional \
                 area factor and an optional 'off')"
            ),
        ));
    }
    Ok(())
}

/// The ", plus ..." tail of the extra-positional message, so a resistor (no fields at all) does
/// not advertise an empty list.
fn accepted_suffix(accepted: &'static [&'static str]) -> String {
    match accepted {
        [] => String::new(),
        _ => format!(", plus optional {}", accepted.join("/")),
    }
}

/// The raw text of a device card's `ic=` field, if it declares one. Already validated as an
/// accepted key by [`validate_trailing_fields`], so reaching this on anything but `C`/`L` is a
/// bug, not bad input.
fn initial_condition_field(element: &ElementInstance) -> Option<&str> {
    element
        .raw_params
        .iter()
        .find_map(|token| match as_field(token) {
            Some((key, value)) if key == "ic" => Some(value),
            _ => None,
        })
}

fn scalar_value(element: &ElementInstance) -> Result<Expression, BuildError> {
    let raw = element
        .raw_params
        .first()
        .ok_or_else(|| BuildError::InvalidElement {
            name: element.name.clone(),
            message: "has no scalar value".into(),
        })?;
    Expression::parse_scalar(raw).map_err(|message| BuildError::InvalidElement {
        name: element.name.clone(),
        message,
    })
}

/// The DC value an independent source card declares, for a card
/// [`TransientFunction::parse_params`] found no waveform in.
///
/// Only the DC clause is read: an explicit `DC <value>`, or a bare leading value, which SPICE
/// allows as the implicit spelling of the same thing. A card that declares neither -- one
/// carrying only an `AC` specification, say -- has a DC value of zero, SPICE's own default,
/// rather than whatever `AC` happens to tokenize into. (It used to be read as `params.first()`
/// unconditionally, which turned `V1 1 0 AC 1` into a source whose value was a symbol named
/// `AC`.) `key=value` tokens are skipped: they are fields, already validated, never the value.
fn source_value(element: &ElementInstance) -> Result<Expression, BuildError> {
    let params: Vec<&str> = element
        .raw_params
        .iter()
        .filter(|token| as_field(token).is_none())
        .map(|token| token.as_str())
        .collect();
    let raw = match params
        .iter()
        .position(|token| token.eq_ignore_ascii_case("dc"))
        .and_then(|index| params.get(index + 1))
    {
        Some(value) => *value,
        None => match params.first() {
            Some(first) if !first.eq_ignore_ascii_case("ac") && !first.contains('(') => *first,
            Some(_) => return Ok(Expression::zero()),
            None => {
                return Err(BuildError::InvalidElement {
                    name: element.name.clone(),
                    message: "has no source value".into(),
                })
            }
        },
    };
    Expression::parse_scalar(raw).map_err(|message| BuildError::InvalidElement {
        name: element.name.clone(),
        message,
    })
}

fn control_and_gain(element: &ElementInstance) -> Result<(&str, Expression), BuildError> {
    match element.raw_params.as_slice() {
        [control, gain, ..] => Expression::parse_scalar(gain)
            .map(|gain| (control.as_str(), gain))
            .map_err(|message| BuildError::InvalidElement {
                name: element.name.clone(),
                message,
            }),
        _ => Err(BuildError::InvalidElement {
            name: element.name.clone(),
            message: "requires a controlling source and gain".into(),
        }),
    }
}

fn normalize(value: &str) -> String {
    value.to_ascii_uppercase()
}

fn is_ground(value: &str) -> bool {
    value == "0" || value.eq_ignore_ascii_case("gnd")
}

/// Error returned while parsing or stamping a netlist.
#[derive(Debug, Clone, PartialEq)]
pub enum BuildError {
    /// `general-spice-core` rejected a statement.
    Parse {
        /// Parser message.
        message: String,
        /// First physical line.
        line_start: usize,
        /// Exclusive final physical line.
        line_end: usize,
    },
    /// A device does not have a linear stamp.
    UnsupportedElement(String),
    /// An element was syntactically parsed but cannot be stamped.
    InvalidElement {
        /// Element name.
        name: String,
        /// Explanation.
        message: String,
    },
    /// A device card carries a trailing parameter this crate does not recognize. Reported
    /// rather than silently discarded — see [`validate_trailing_fields`].
    InvalidElementField {
        /// 1-based physical line the device card was parsed from.
        line: usize,
        /// Element name.
        name: String,
        /// Explanation, already phrased to follow `device '<name>' `.
        message: String,
    },
    /// Two branch devices have the same case-insensitive name.
    DuplicateElement(String),
    /// A current-controlled source refers to a device without a branch current.
    UnknownControllingBranch(String),
    /// A mutual-inductance element refers to an unknown inductor.
    UnknownInductor(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Parse {
                message,
                line_start,
                line_end,
            } => write!(
                f,
                "parse error at lines {line_start}..{line_end}: {message}"
            ),
            Self::UnsupportedElement(message) => f.write_str(message),
            Self::InvalidElement { name, message } => {
                write!(f, "cannot stamp element '{name}': {message}")
            }
            Self::InvalidElementField {
                line,
                name,
                message,
            } => write!(f, "line {line}: device '{name}' {message}"),
            Self::DuplicateElement(name) => {
                write!(f, "duplicate case-insensitive element name '{name}'")
            }
            Self::UnknownControllingBranch(name) => {
                write!(f, "unknown controlling branch '{name}'")
            }
            Self::UnknownInductor(name) => write!(f, "unknown coupled inductor '{name}'"),
        }
    }
}

impl std::error::Error for BuildError {}
