use std::collections::{BTreeMap, HashMap};
use std::fmt;

use general_spice_core::ast::{ElementInstance, Statement};
use general_spice_core::{lexer, parser, Dialect};

use crate::{Expression, Matrix, MnaSystem, TransientFunction};

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

/// Builds MNA systems from `spice-core` statements or netlist text.
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
                if let Some(transient_fn) = TransientFunction::parse(&element.raw_params) {
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
                            "parameter '{name}' has an expression that is preserved by spice-core but is not a scalar MNA value"
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

fn source_value(element: &ElementInstance) -> Result<Expression, BuildError> {
    let params = &element.raw_params;
    let raw = params
        .iter()
        .position(|token| token.eq_ignore_ascii_case("dc"))
        .and_then(|index| params.get(index + 1))
        .or_else(|| params.first())
        .ok_or_else(|| BuildError::InvalidElement {
            name: element.name.clone(),
            message: "has no source value".into(),
        })?;
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
    /// `spice-core` rejected a statement.
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
