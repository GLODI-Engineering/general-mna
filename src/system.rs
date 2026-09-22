use std::collections::BTreeMap;

use crate::{Expression, Matrix, TransientFunction};

/// One `ic=` (or `.IC`) initial condition, already resolved against this system's own unknown
/// ordering.
///
/// **Sign conventions, stated once, explicitly** — these are the half that gets misremembered:
///
/// - A capacitor's `ic` is the voltage of its **first** node minus its **second**:
///   `C1 b 0 1e-6 ic=5` means `V(b) - V(0) = 5 V` at `t = 0`.
/// - An inductor's `ic` is the current flowing **from its first node to its second node,
///   through the inductor**: `L1 a b 5e-6 ic=12` means 12 A enter the inductor at `a` and
///   leave it at `b` at `t = 0`. This is exactly the sign of the system's own `I(L1)` unknown,
///   because `stamp_branch_incidence` puts `+1` in the first node's KCL row — so a positive
///   branch current leaves that node and enters the element. Writing `L1 b a 5e-6 ic=12`
///   instead declares the opposite physical current.
#[derive(Debug, Clone, PartialEq)]
pub enum InitialCondition {
    /// `ic=` on a `C` card: the voltage across the capacitor at `t = 0`.
    CapacitorVoltage {
        /// The capacitor's element name, for diagnostics.
        element: String,
        /// Index into `unknowns` of the first node's voltage, `None` when it is ground.
        positive: Option<usize>,
        /// Index into `unknowns` of the second node's voltage, `None` when it is ground.
        negative: Option<usize>,
        /// The declared value, in volts.
        value: Expression,
    },
    /// `ic=` on an `L` card: the current through the inductor at `t = 0`.
    InductorCurrent {
        /// The inductor's element name, for diagnostics.
        element: String,
        /// Index into `unknowns` of this inductor's own branch-current unknown.
        branch: usize,
        /// The declared value, in amperes, first node -> second node.
        value: Expression,
    },
}

impl InitialCondition {
    /// The element the condition was declared on.
    pub fn element(&self) -> &str {
        match self {
            Self::CapacitorVoltage { element, .. } | Self::InductorCurrent { element, .. } => {
                element
            }
        }
    }

    /// The declared value, still symbolic.
    pub fn value(&self) -> &Expression {
        match self {
            Self::CapacitorVoltage { value, .. } | Self::InductorCurrent { value, .. } => value,
        }
    }
}

/// A symbolic MNA descriptor system.
#[derive(Debug, Clone, PartialEq)]
pub struct MnaSystem {
    /// Memoryless matrix in `A*x + K*dot(x) = B*u`.
    pub a: Matrix<Expression>,
    /// Storage matrix in `A*x + K*dot(x) = B*u`.
    pub k: Matrix<Expression>,
    /// Independent-source incidence matrix.
    pub b: Matrix<Expression>,
    /// Expanded right-hand-side vector using the source values from the netlist.
    pub u: Vec<Expression>,
    /// MNA unknown names in matrix order.
    pub unknowns: Vec<String>,
    /// Independent source names in `B` column order.
    pub inputs: Vec<String>,
    /// Source values in `inputs` order.
    pub input_values: Vec<Expression>,
    /// Every `V`/`I` source declared with a `SIN`/`PULSE`/`EXP`/`PWL`/`SFFM` transient
    /// function, keyed by source name — such a source's own `input_values` entry is
    /// `Expression::symbol(name)` (not a baked literal), so the caller must supply
    /// `values.insert(name, transient_sources[name].value_at(t))` before every
    /// [`MnaSystem::evaluate`] call, exactly the way a PWL diode's `{name}_Ioff` symbol
    /// already has to be supplied per step — see `transient_source`'s own module doc comment.
    /// Empty for a netlist with no such sources (every existing caller is unaffected).
    pub transient_sources: BTreeMap<String, TransientFunction>,
    /// Every `ic=` initial condition declared on a storage element, in netlist order, followed
    /// by every `.IC V(node)=` / `.IC V(n1,n2)=` / `.IC I(inductor)=` assignment, in netlist
    /// order (a `.IC` node voltage is a [`InitialCondition::CapacitorVoltage`] whose second
    /// terminal is ground, labelled `.IC V(node)`). Empty for a netlist that declares none,
    /// which is every netlist that predates the feature.
    ///
    /// These are *not* folded into `a`/`k`/`b`/`u` — the descriptor system is the circuit's
    /// equations, and an initial condition is a statement about `x` at one instant, not about
    /// the equations. Turn them into a consistent starting `x` with
    /// [`MnaSystem::initial_state`], which is what a transient run actually needs.
    pub initial_conditions: Vec<InitialCondition>,
    /// Defaults collected from `.param` statements.
    pub parameter_defaults: BTreeMap<String, Expression>,
    /// Non-fatal build messages, primarily for deliberately ignored elements.
    pub warnings: Vec<String>,
}

impl MnaSystem {
    /// Number of MNA unknowns.
    pub fn order(&self) -> usize {
        self.unknowns.len()
    }

    /// Converts the system into strings suitable for a thin Python or WASM DTO.
    pub fn to_string_system(&self) -> StringMnaSystem {
        StringMnaSystem {
            rows: self.order(),
            inputs_count: self.inputs.len(),
            a: self.a.iter().map(ToString::to_string).collect(),
            k: self.k.iter().map(ToString::to_string).collect(),
            b: self.b.iter().map(ToString::to_string).collect(),
            u: self.u.iter().map(ToString::to_string).collect(),
            unknowns: self.unknowns.clone(),
            inputs: self.inputs.clone(),
            input_values: self.input_values.iter().map(ToString::to_string).collect(),
            parameter_defaults: self
                .parameter_defaults
                .iter()
                .map(|(name, value)| (name.clone(), value.to_string()))
                .collect(),
            warnings: self.warnings.clone(),
        }
    }
}

/// Serialization-neutral string representation for language bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StringMnaSystem {
    /// Number of rows and columns in `a` and `k`.
    pub rows: usize,
    /// Number of columns in `b`.
    pub inputs_count: usize,
    /// Row-major memoryless matrix.
    pub a: Vec<String>,
    /// Row-major storage matrix.
    pub k: Vec<String>,
    /// Row-major input-incidence matrix.
    pub b: Vec<String>,
    /// Expanded source vector.
    pub u: Vec<String>,
    /// Unknown names.
    pub unknowns: Vec<String>,
    /// Input names.
    pub inputs: Vec<String>,
    /// Netlist source values.
    pub input_values: Vec<String>,
    /// Netlist parameter defaults.
    pub parameter_defaults: BTreeMap<String, String>,
    /// Build warnings.
    pub warnings: Vec<String>,
}
